# Copyright (c) ASAPP, Inc.
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Union
import logging
import os
import sys

import joblib
import fire
import fairseq
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from tokenizers import Tokenizer
import re

from functools import partial
import torch.multiprocessing as mp

from feature_utils import dump_feature, dump_cluster
from einops import rearrange
import torchaudio


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("generate_pseudo_language")


def convert_int_to_chr(line: str, chr_shift: int = 33):
    if isinstance(line, str):
        line = line.strip().split()
    return "".join([chr(int(c) + chr_shift) for c in line])


def merge_duplicates(line: str):
    return re.sub(r"(.)\1+", r"\1", line, 0, re.MULTILINE)


def get_aligned_bpe(
    bpe_ids: Union[str, List[int]],
    cids: Union[str, List[int]],
    tokenizer: Tokenizer,
    chr_shift: int = 33,
):
    """
    args:
        bpe_ids: list of bpe indices
        cids: list of cluster indices
        tokenizer: tokenizer
        chr_shift: shift the cluster id when coverting with "chr" function
    """
    if isinstance(bpe_ids, str):
        bpe_ids = [int(i) for i in bpe_ids.strip().split()]
    if isinstance(cids, str):
        cids = [int(i) for i in cids.strip().split()]
    results = []
    bpe_pointer = -1
    prev_chrd = ""
    decoded = ""

    try:
        for i, idx in enumerate(cids):
            chrd = chr(int(idx) + chr_shift)
            if chrd != prev_chrd:
                if len(decoded) == 0:
                    bpe_pointer += 1
                    decoded = tokenizer.decode([int(bpe_ids[bpe_pointer])])

                assert chrd == decoded[0], f"{chrd} != {decoded[0]}"
                results.append(bpe_ids[bpe_pointer])
                decoded = decoded[1:]

            else:
                results.append(bpe_ids[bpe_pointer])
            prev_chrd = chrd

        assert decoded == ""
        assert bpe_pointer == len(bpe_ids) - 1
    except:
        true_bpe_ids = tokenizer.encode(merge_duplicates(convert_int_to_chr(cids))).ids
        print(bpe_ids)
        print(true_bpe_ids)
        return get_aligned_bpe(true_bpe_ids, cids, tokenizer, chr_shift)
    return results


class MfccFeatureReader(object):
    def __init__(self, sample_rate, pool_k=1, pool_s=1):
        self.sample_rate = sample_rate
        if pool_k == 1 and pool_s == 1:
            self.pooler = None
        else:
            self.pooler = nn.AvgPool1d(pool_k, stride=pool_s)

    def read_audio(self, path, ref_len=None):
        wav, sr = sf.read(path)
        assert sr == self.sample_rate, sr
        if wav.ndim == 2:
            wav = wav.mean(-1)
        assert wav.ndim == 1, wav.ndim
        if ref_len is not None and abs(ref_len - len(wav)) > 160:
            logging.warning(f"ref {ref_len} != read {len(wav)} ({path})")
        return wav

    def get_feats(self, path, ref_len=None):
        x = self.read_audio(path, ref_len)
        with torch.no_grad():
            x = torch.from_numpy(x).float()
            x = x.view(1, -1)

            mfccs = torchaudio.compliance.kaldi.mfcc(
                waveform=x, sample_frequency=self.sample_rate, use_energy=False,
            )  # (time, freq)
            mfccs = mfccs.transpose(0, 1)  # (freq, time)
            deltas = torchaudio.functional.compute_deltas(mfccs)
            ddeltas = torchaudio.functional.compute_deltas(deltas)
            concat = torch.cat([mfccs, deltas, ddeltas], dim=0)
            if self.pooler is not None:
                concat = self.pooler(concat.unsqueeze(0)).squeeze(0)
            concat = concat.transpose(0, 1).contiguous()  # (freq, time)
            return concat


class FBankFeatureReader(object):
    def __init__(self, sample_rate, pool_k=1, pool_s=1):
        self.sample_rate = sample_rate
        if pool_k == 1 and pool_s == 1:
            self.pooler = None
        else:
            self.pooler = nn.AvgPool1d(pool_k, stride=pool_s)

    def read_audio(self, path, ref_len=None):
        wav, sr = sf.read(path)
        assert sr == self.sample_rate, sr
        if wav.ndim == 2:
            wav = wav.mean(-1)
        assert wav.ndim == 1, wav.ndim
        if ref_len is not None and abs(ref_len - len(wav)) > 160:
            logging.warning(f"ref {ref_len} != read {len(wav)} ({path})")
        return wav

    def get_feats(self, path, ref_len=None):
        x = self.read_audio(path, ref_len)
        with torch.no_grad():
            x = torch.from_numpy(x).float()
            x = x.view(1, -1)

            fbank = torchaudio.compliance.kaldi.fbank(
                num_mel_bins=80,
                waveform=x,
                sample_frequency=self.sample_rate,
                use_energy=False,
            )  # (time, freq)
            fbank = fbank.transpose(0, 1)  # (freq, time)
            # deltas = torchaudio.functional.compute_deltas(fbank)
            # ddeltas = torchaudio.functional.compute_deltas(deltas)
            # concat = torch.cat([fbank, deltas, ddeltas], dim=0)
            concat = fbank
            if self.pooler is not None:
                concat = self.pooler(concat.unsqueeze(0)).squeeze(0)
            concat = concat.transpose(0, 1).contiguous()  # (freq, time)
            return concat


class W2V2FeatureReader(object):
    def __init__(
        self, ckpt_path, layer, max_chunk=1600000, fp16=False, pool_k=1, pool_s=1
    ):
        (model, cfg, task,) = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [ckpt_path]
        )
        self.model = model[0].eval().cuda()
        self.task = task
        self.layer = layer
        self.max_chunk = max_chunk
        self.fp16 = fp16
        if fp16:
            self.model.half()

        if hasattr(self.model, "encoder") and hasattr(self.model.encoder, "w2v_model"):
            self.model = self.model.encoder.w2v_model
        elif hasattr(self.model, "w2v_encoder") and hasattr(
            self.model.w2v_encoder, "w2v_model"
        ):
            self.model = self.model.w2v_encoder.w2v_model

        if isinstance(self.model, fairseq.models.hubert.HubertModel):
            self.layer_shift = 0
            self.encoder_type = "hubert"
        else:
            self.layer_shift = -1
            self.encoder_type = "w2v2"

        if pool_k == 1 and pool_s == 1:
            self.pooler = None
        else:
            self.pooler = nn.AvgPool1d(pool_k, stride=pool_s)

        logger.info(f"TASK CONFIG:\n{self.task.cfg}")
        logger.info(f" max_chunk = {self.max_chunk}")
        logger.info(f" model:\n{self.model}")
        logger.info(f" pooler: {self.pooler}")
        logger.info(f" layer_shift = {self.layer_shift}")

    def read_audio(self, path, ref_len=None):
        wav, sr = sf.read(path)
        assert sr == self.task.cfg.sample_rate, sr
        if wav.ndim == 2:
            wav = wav.mean(-1)
        assert wav.ndim == 1, wav.ndim
        if ref_len is not None and abs(ref_len - len(wav)) > 160:
            logging.warning(f"ref {ref_len} != read {len(wav)} ({path})")
        return wav

    @torch.no_grad()
    def get_feats(self, path, ref_len=None):
        x = self.read_audio(path, ref_len)
        with torch.no_grad():
            if self.fp16:
                x = torch.from_numpy(x).half().cuda()
            else:
                x = torch.from_numpy(x).float().cuda()
            if self.task.cfg.normalize:
                x = F.layer_norm(x, x.shape)
            x = x.view(1, -1)

            feat = []
            for start in range(0, x.size(1), self.max_chunk):
                x_chunk = x[:, start : start + self.max_chunk]
                if self.encoder_type == "hubert":
                    feat_chunk, _ = self.model.extract_features(
                        source=x_chunk,
                        padding_mask=None,
                        mask=False,
                        output_layer=self.layer + self.layer_shift,
                    )
                else:
                    res = self.model.extract_features(
                        source=x_chunk,
                        padding_mask=None,
                        mask=False,
                        layer=self.layer + self.layer_shift,
                    )
                    feat_chunk = res["x"]
                if self.pooler is not None:
                    feat_chunk = rearrange(feat_chunk, "b t c -> b c t")
                    feat_chunk = self.pooler(feat_chunk)
                    feat_chunk = rearrange(feat_chunk, "b c t -> b t c")
                feat.append(feat_chunk)
        if len(feat) == 0:
            return torch.zeros(0, 0)
        return torch.cat(feat, 1).squeeze(0)


class ApplyKmeans(object):
    def __init__(self, km_path):
        self.km_model = joblib.load(km_path)
        self.C_np = self.km_model.cluster_centers_.transpose()
        self.Cnorm_np = (self.C_np ** 2).sum(0, keepdims=True)

        self.C = torch.from_numpy(self.C_np)
        self.Cnorm = torch.from_numpy(self.Cnorm_np)
        if torch.cuda.is_available():
            self.C = self.C.cuda()
            self.Cnorm = self.Cnorm.cuda()

    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            self.C = self.C.to(x)
            self.Cnorm = self.Cnorm.to(x)
            dist = (
                x.pow(2).sum(1, keepdim=True) - 2 * torch.matmul(x, self.C) + self.Cnorm
            )
            return dist.argmin(dim=1).cpu().numpy()
        else:
            dist = (
                (x ** 2).sum(1, keepdims=True)
                - 2 * np.matmul(x, self.C_np)
                + self.Cnorm_np
            )
            return np.argmin(dist, axis=1)


def get_reader(ckpt_path, layer, max_chunk, fp16, pool_k, pool_s):
    if ckpt_path == "mfcc":
        return MfccFeatureReader(16000, pool_k=pool_k, pool_s=pool_s)
    if ckpt_path == "fbank":
        return FBankFeatureReader(16000, pool_k=pool_k, pool_s=pool_s)
    else:
        return W2V2FeatureReader(
            ckpt_path, layer, max_chunk, fp16=fp16, pool_k=pool_k, pool_s=pool_s
        )


class PseudoLanguageConverter:
    def __init__(self, tokenizer_path, chr_shift=33):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.chr_shift = chr_shift

    def __call__(self, cluster_ids):
        line = convert_int_to_chr(cluster_ids, self.chr_shift)
        line = merge_duplicates(line)
        token_ids = self.tokenizer.encode(line).ids
        return token_ids


def dump_hubert_features(
    manifest="manifest/librispeech/train-960/valid.tsv",
    ckpt_path="save/pretrained/hubert_base_ls960.pt",
    feat_dir="features/debug",
    layer=9,
    nshard=1,
    rank=0,
    max_chunk=1_600_000,
    fp16=False,
    pool_k=1,
    pool_s=1,
):
    reader = get_reader(
        ckpt_path, layer, max_chunk, fp16=fp16, pool_k=pool_k, pool_s=pool_s
    )
    dump_feature(rank, reader, manifest, nshard, feat_dir)


def dump_hubert_clusters(
    manifest="manifest/librispeech/train-960/valid.tsv",
    ckpt_path="save/pretrained/hubert_base_ls960.pt",
    km_path="labels/hubert-iter2-l9/librispeech/train-960/km-hubert-iter2-l9-p0.1-c500/km.pkl",
    lab_dir="labels/debug",
    output_suffix="km",
    layer=9,
    nshard=1,
    rank=0,
    max_chunk=1_600_000,
    fp16=False,
    pool_k=1,
    pool_s=1,
):
    reader = get_reader(
        ckpt_path, layer, max_chunk, fp16=fp16, pool_k=pool_k, pool_s=pool_s
    )
    apply_kmeans = ApplyKmeans(km_path)
    dump_cluster(rank, reader, apply_kmeans, manifest, nshard, lab_dir, output_suffix)


def dump_hubert_features_pl(
    manifest="manifest/librispeech/train-960/valid.tsv",
    ckpt_path="save/pretrained/hubert_base_ls960.pt",
    km_path="labels/hubert-iter2-l9/librispeech/train-960/km-hubert-iter2-l9-p0.1-c500/km.pkl",
    lab_dir="labels/debug",
    bpe_path="labels/hubert-iter2-l9/librispeech/train-960/km-hubert-iter2-l9-p0.1-c500/bpe-tokenizer-vocab30000.json",
    output_suffix="chrd_bpe30000.km",
    layer=9,
    nshard=1,
    rank=0,
    max_chunk=1_600_000,
    fp16=False,
    pool_k=1,
    pool_s=1,
):
    """dump pseudo language"""
    reader = get_reader(
        ckpt_path, layer, max_chunk, fp16=fp16, pool_k=pool_k, pool_s=pool_s
    )
    apply_kmeans = ApplyKmeans(km_path)
    pl_converter = PseudoLanguageConverter(bpe_path)
    dump_cluster(
        rank,
        reader,
        apply_kmeans,
        manifest,
        nshard,
        lab_dir,
        output_suffix,
        pl_converter=pl_converter,
    )


def dump_aligned_pl(pl_folder="", bpe_suffix="chrd_bpe30000", bpe_path=""):
    if bpe_path == "":
        bpe_path = os.path.join(pl_folder, "bpe-tokenizer-vocab30000.json")
    tokenizer = Tokenizer.from_file(bpe_path)

    suffix = f".{bpe_suffix}.km"
    splits = [
        filename.replace(suffix, "")
        for filename in os.listdir(pl_folder)
        if filename.endswith(suffix)
    ]

    for split in splits:
        with open(os.path.join(pl_folder, f"{split}.km")) as f:
            clusters = [line.strip() for line in f]

        with open(os.path.join(pl_folder, f"{split}.{bpe_suffix}.km")) as f:
            bpes = [line.strip() for line in f]

        assert len(bpes) == len(clusters)

        output_file = os.path.join(pl_folder, f"{split}.{bpe_suffix}.km.align")
        with open(output_file, "w") as f:
            for cids, bpe_ids in tqdm(
                zip(clusters, bpes), total=len(bpes), desc=f"generating {output_file}"
            ):
                aligned = get_aligned_bpe(bpe_ids, cids, tokenizer)
                print(" ".join(list(map(str, aligned))), file=f)


if __name__ == "__main__":
    fire.Fire()
