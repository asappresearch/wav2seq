# Copyright (c) ASAPP Inc.
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import logging
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

from functools import partial

from fairseq.data.audio.raw_audio_dataset import RawAudioDataset

import librosa

logger = logging.getLogger(__name__)


class FileAudioDatasetV2(RawAudioDataset):
    def __init__(
        self,
        manifest_path,
        sample_rate,
        max_sample_size=None,
        min_sample_size=0,
        shuffle=True,
        pad=False,
        normalize=False,
        use_librosa=False,
    ):
        super().__init__(
            sample_rate=sample_rate,
            max_sample_size=max_sample_size,
            min_sample_size=min_sample_size,
            shuffle=shuffle,
            pad=pad,
            normalize=normalize,
        )

        self.use_librosa = use_librosa
        self.fnames = []
        self.line_inds = set()

        skipped = 0
        with open(manifest_path, "r") as f:
            first_line = f.readline().strip()
            self.segmented = first_line == "segmented"
            if self.segmented:
                self.root_dir = ""
                for i, line in enumerate(f):
                    sid, filename, start, end = line.split()
                    start, end = float(start), float(end)
                    sz = int(sample_rate * (end - start))
                    if sz > 0:
                        self.fnames.append((sid, filename, start, end))
                        self.line_inds.add(i)
                        self.sizes.append(sz)
            else:
                self.root_dir = first_line
                for i, line in enumerate(f):
                    items = line.strip().split("\t")
                    assert len(items) == 2, line
                    sz = int(items[1])
                    if min_sample_size is not None and sz < min_sample_size:
                        skipped += 1
                        continue
                    self.fnames.append(items[0])
                    self.line_inds.add(i)
                    self.sizes.append(sz)
        logger.info(f"loaded {len(self.fnames)}, skipped {skipped} samples")

    def __getitem__(self, index):
        if self.segmented:
            import librosa

            sid, fname, start, end = self.fnames[index]
            try:
                wav, curr_sample_rate = librosa.load(
                    fname, sr=self.sample_rate, offset=start, duration=end - start
                )
            except:
                duration = librosa.get_duration(filename=fname)
                print(fname, start, end, duration)
                assert False
        else:
            fname = os.path.join(self.root_dir, self.fnames[index])
            if self.use_librosa:
                import librosa

                wav, curr_sample_rate = librosa.load(fname, sr=self.sample_rate)
            else:
                import soundfile as sf

                wav, curr_sample_rate = sf.read(fname)
        feats = torch.from_numpy(wav).float()

        out = {"id": index}
        feats = self.postprocess(feats, curr_sample_rate)
        out["source"] = feats
        return out


class FileAudioFeatDataset(FileAudioDatasetV2):
    """Extend FileAutioDataset to generate filter bank or MFCC features"""

    def __init__(
        self,
        manifest_path,
        sample_rate,
        max_sample_size=None,
        min_sample_size=0,
        shuffle=True,
        pad=False,
        normalize=False,
        use_librosa=False,
        spec_dim=0,
        fbank_dim=0,
        mfcc_dim=0,
        frame_length=25.0,
        frame_shift=10.0,
    ):
        super().__init__(
            manifest_path=manifest_path,
            sample_rate=sample_rate,
            max_sample_size=max_sample_size,
            min_sample_size=min_sample_size,
            shuffle=shuffle,
            pad=pad,
            normalize=normalize,
            use_librosa=use_librosa,
        )
        self.spec_dim = spec_dim
        self.fbank_dim = fbank_dim
        self.mfcc_dim = mfcc_dim
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.spec_func = None

    def collater(self, samples):
        samples = [s for s in samples if s["source"] is not None]
        if len(samples) == 0:
            return {}

        sources = [s["source"] for s in samples]
        sizes = [len(s) for s in sources]

        if self.pad:
            target_size = min(max(sizes), self.max_sample_size)
        else:
            target_size = min(min(sizes), self.max_sample_size)

        collated_sources = sources[0].new_zeros(len(sources), target_size)
        padding_mask = (
            torch.BoolTensor(collated_sources.shape).fill_(False) if self.pad else None
        )
        for i, (source, size) in enumerate(zip(sources, sizes)):
            diff = size - target_size
            if diff == 0:
                collated_sources[i] = source
            elif diff < 0:
                assert self.pad
                collated_sources[i] = torch.cat(
                    [source, source.new_full((-diff,), 0.0)]
                )
                if padding_mask is not None:
                    padding_mask[i, diff:] = True
            else:
                collated_sources[i] = self.crop_to_max_size(source, target_size)

        input = {"source": collated_sources}
        if self.pad:
            input["padding_mask"] = padding_mask

        audio_feats = []
        if self.spec_dim > 0:
            if self.spec_func is None:
                self.spec_func = torchaudio.transforms.Spectrogram(
                    n_fft=(self.spec_dim - 1) * 2,
                    win_length=self.frame_length,
                    hop_length=self.frame_shift,
                )
            spec = torch.stack(
                [self.spec_func(feats.view(1, -1)) for feats in collated_sources], dim=0
            )
            audio_feats.append(spec)
        if self.fbank_dim > 0:
            fbank = torch.stack(
                [
                    torchaudio.compliance.kaldi.fbank(
                        feats.view(1, -1),
                        num_mel_bins=self.fbank_dim,
                        sample_frequency=self.sample_rate,
                        frame_length=self.frame_length,
                        frame_shift=self.frame_shift,
                    )
                    for feats in collated_sources
                ],
                dim=0,
            )
            audio_feats.append(fbank)
        if self.mfcc_dim > 0:
            mfcc = torch.stack(
                [
                    torchaudio.compliance.kaldi.mfcc(
                        feats.view(1, -1),
                        num_ceps=self.mfcc_dim,
                        sample_frequency=self.sample_rate,
                        frame_length=self.frame_length,
                        frame_shift=self.frame_shift,
                    )
                    for feats in collated_sources
                ],
                dim=0,
            )
            audio_feats.append(mfcc)
        if len(audio_feats) == 0:
            audio_feats = None
        elif len(audio_feats) == 1:
            audio_feats = audio_feats[0]
        else:
            audio_feats = torch.cat(audio_feats, dim=-1)

        if audio_feats is not None:
            input["audio_feats"] = audio_feats

        return {"id": torch.LongTensor([s["id"] for s in samples]), "net_input": input}


class NCropFileAudioFeatDataset(FileAudioFeatDataset):
    def __init__(
        self,
        manifest_path,
        sample_rate,
        max_sample_size=None,
        min_sample_size=None,
        shuffle=True,
        pad=False,
        normalize=False,
        use_librosa=False,
        fbank_dim=0,
        mfcc_dim=0,
        frame_length=25.0,
        frame_shift=10.0,
        n_crops=1,
        clean_first=True,
        crop_same_position=True,
        mix_thres=0.0,
        gain_prob=0.0,
        gain_range=10,
        pitch_prob=0.0,
        pitch_range=300,
        reverb_prob=0.0,
    ):
        super().__init__(
            manifest_path=manifest_path,
            sample_rate=sample_rate,
            max_sample_size=max_sample_size,
            min_sample_size=min_sample_size,
            shuffle=shuffle,
            pad=pad,
            normalize=normalize,
            use_librosa=False,
            fbank_dim=fbank_dim,
            mfcc_dim=0,
            frame_length=frame_length,
            frame_shift=frame_shift,
        )
        self.n_crops = n_crops
        self.clean_first = clean_first
        self.crop_same_position = crop_same_position
        self.mix_thres = mix_thres
        self.pitch_prob = pitch_prob
        self.pitch_range = pitch_range
        self.gain_prob = gain_prob
        self.gain_range = gain_range
        self.reverb_prob = reverb_prob
        torchaudio.sox_effects.init_sox_effects()

    def __getitem__(self, index):
        fname = os.path.join(self.root_dir, self.fnames[index])
        # if self.use_librosa:
        #     import librosa
        #     wav, curr_sample_rate = librosa.load(fname, sr=self.sample_rate)
        # else:
        #     import soundfile as sf
        #     wav, curr_sample_rate = sf.read(fname)
        # feats = torch.from_numpy(wav).float()
        # feats = self.postprocess(feats, curr_sample_rate)
        # crop or pad
        feats, curr_sample_rate = torchaudio.load(fname)

        if self.clean_first:
            all_crops = [feats]
        else:
            all_crops = []
        start = len(all_crops)

        with torch.no_grad():
            for _ in range(start, self.n_crops):
                effects = []
                # change volumn
                if self.gain_range > 0 and np.random.uniform(0.0, 1.0) < self.gain_prob:
                    gain = np.random.uniform(-self.gain_range, self.gain_range)
                    effects.append(["gain", f"{gain}"])
                # change pitch
                if (
                    self.pitch_range > 0
                    and np.random.uniform(0.0, 1.0) < self.pitch_prob
                ):
                    pitch = np.random.uniform(-self.pitch_range, self.pitch_range)
                    effects.append(["pitch", f"{pitch}"])
                # add revert or not
                if np.random.uniform(0.0, 1.0) < self.reverb_prob:
                    effects.append(["reverb"])
                effects.append(["rate", f"{self.sample_rate}"])
                x, curr_sample_rate = torchaudio.sox_effects.apply_effects_tensor(
                    feats, curr_sample_rate, effects, channels_first=True
                )
                all_crops.append(x)

        out = {"id": index}
        # if self.fbank_dim > 0:
        #     out['fbank'] = torchaudio.compliance.kaldi.fbank(
        #         feats.view(1, -1), num_mel_bins=self.fbank_dim, sample_frequency=curr_sample_rate,
        #         frame_length=self.frame_length, frame_shift=self.frame_shift)
        # if self.mfcc_dim > 0:
        #     out['mfcc'] = torchaudio.compliance.kaldi.mfcc(
        #         feats.view(1, -1), num_ceps=self.mfcc_dim, sample_frequency=curr_sample_rate,
        #         frame_length=self.frame_length, frame_shift=self.frame_shift)
        out["source"] = all_crops
        return out

    def collater(self, samples):
        samples = [s for s in samples if s["source"] is not None]
        if len(samples) == 0:
            return {}

        sources = [
            s["source"][i].view(-1) for i in range(self.n_crops) for s in samples
        ]
        sizes = [len(s) for s in sources]
        batch_size = len(sources) // self.n_crops

        if self.pad:
            target_size = min(max(sizes), self.max_sample_size)
        else:
            target_size = min(min(sizes), self.max_sample_size)

        collated_sources = sources[0].new_zeros(len(sources), target_size)
        padding_mask = (
            torch.BoolTensor(collated_sources.shape).fill_(False) if self.pad else None
        )
        crop_ranges = {}
        for i, (source, size) in enumerate(zip(sources, sizes)):
            diff = size - target_size
            if diff == 0:
                collated_sources[i] = source
            elif diff < 0:
                assert self.pad
                collated_sources[i] = torch.cat(
                    [source, source.new_full((-diff,), 0.0)]
                )
                padding_mask[i, diff:] = True
            elif self.crop_same_position and i > batch_size:
                if i % batch_size not in crop_ranges:
                    print(i, batch_size, crop_ranges.keys())
                start, end = crop_ranges[i % batch_size]
                collated_sources[i] = source[start:end]
            else:
                collated_sources[i], crop_range = self.crop_to_max_size(
                    source, target_size, True
                )
                crop_ranges[i] = crop_range

        if self.mix_thres > 0.0:
            m = np.random.uniform(0.0, self.mix_thres)
            perm_indices = np.random.permutation(collated_sources.size(0))
            if self.clean_first:
                perm_indices[:batch_size] = np.arange(batch_size)
            collated_sources = (
                collated_sources * (1.0 - m) + collated_sources[perm_indices] * m
            )

        input = {"source": collated_sources.view(self.n_crops, -1, target_size)}
        if self.pad:
            input["padding_mask"] = padding_mask

        audio_feats = []
        if self.fbank_dim > 0:
            fbank = torch.stack(
                [
                    torchaudio.compliance.kaldi.fbank(
                        feats.view(1, -1),
                        num_mel_bins=self.fbank_dim,
                        sample_frequency=self.sample_rate,
                        frame_length=self.frame_length,
                        frame_shift=self.frame_shift,
                    )
                    for feats in collated_sources
                ],
                dim=0,
            )
            audio_feats.append(fbank)
        if self.mfcc_dim > 0:
            mfcc = torch.stack(
                [
                    torchaudio.compliance.kaldi.mfcc(
                        feats.view(1, -1),
                        num_ceps=self.mfcc_dim,
                        sample_frequency=self.sample_rate,
                        frame_length=self.frame_length,
                        frame_shift=self.frame_shift,
                    )
                    for feats in collated_sources
                ],
                dim=0,
            )
            audio_feats.append(mfcc)

        if len(audio_feats) == 0:
            audio_feats = None
        elif len(audio_feats) == 1:
            audio_feats = audio_feats
        else:
            audio_feats = torch.cat(audio_feats, dim=-1)
        if audio_feats:
            input["audio_feats"] = audio_feats

        return {"id": torch.LongTensor([s["id"] for s in samples]), "net_input": input}

    def crop_to_max_size(self, wav, target_size, return_range=False):
        size = len(wav)
        diff = size - target_size
        if diff <= 0:
            return wav

        start = np.random.randint(0, diff + 1)
        end = size - diff + start
        if return_range:
            return wav[start:end], (start, end)
        else:
            return wav[start:end]


class FileAudioFeatClassificationDataset(RawAudioDataset):
    """Extend FileAudioFeatDataset to have classification labels"""

    def __init__(
        self,
        manifest_path,
        sample_rate,
        max_sample_size=None,
        min_sample_size=None,
        shuffle=True,
        pad=False,
        normalize=False,
        use_librosa=False,
        fbank_dim=0,
        mfcc_dim=0,
        frame_length=25.0,
        frame_shift=10.0,
    ):
        super().__init__(
            sample_rate=sample_rate,
            max_sample_size=max_sample_size,
            min_sample_size=min_sample_size,
            shuffle=shuffle,
            pad=pad,
            normalize=normalize,
        )
        self.use_librosa = use_librosa
        self.fbank_dim = fbank_dim
        self.mfcc_dim = mfcc_dim
        self.frame_length = frame_length
        self.frame_shift = frame_shift

        self.fnames = []
        self.labels = []
        self.spans = []

        skipped = 0
        with open(manifest_path, "r") as f:
            first_line = f.readline().strip()
            self.manifest_type, self.root_dir = first_line.split()
            assert self.manifest_type == "cls"
            for line in f:
                items = line.strip().split("\t")
                assert len(items) == 4, line
                start, end, label = float(items[1]), float(items[2]), int(items[3])
                sz = int((end - start) * sample_rate)
                if max_sample_size is not None:
                    sz = min(sz, max_sample_size)
                if min_sample_size is not None and sz < min_sample_size:
                    skipped += 1
                    continue
                self.fnames.append(items[0])
                self.sizes.append(sz)
                self.labels.append(label)
                self.spans.append((start, end))
        logger.info(f"loaded {len(self.fnames)}, skipped {skipped} samples")

    def __getitem__(self, index):
        fname = os.path.join(self.root_dir, self.fnames[index])
        # if self.use_librosa:
        start, end = self.spans[index]
        wav, curr_sample_rate = librosa.load(
            fname, sr=self.sample_rate, offset=start, duration=end - start
        )
        # else:
        #     import soundfile as sf
        #     wav, curr_sample_rate = sf.read(fname)
        assert curr_sample_rate == self.sample_rate, curr_sample_rate
        feats = torch.from_numpy(wav).float()
        # crop or pad

        out = {"id": index, "target": self.labels[index]}
        feats = self.postprocess(feats, curr_sample_rate)
        out["source"] = feats
        return out

    def collater(self, samples):
        samples = [s for s in samples if s["source"] is not None]
        if len(samples) == 0:
            return {}

        sources = [s["source"] for s in samples]
        sizes = [len(s) for s in sources]

        if self.pad:
            target_size = min(max(sizes), self.max_sample_size)
        else:
            target_size = min(min(sizes), self.max_sample_size)

        collated_sources = sources[0].new_zeros(len(sources), target_size)
        padding_mask = (
            torch.BoolTensor(collated_sources.shape).fill_(False) if self.pad else None
        )
        for i, (source, size) in enumerate(zip(sources, sizes)):
            diff = size - target_size
            if diff == 0:
                collated_sources[i] = source
            elif diff < 0:
                assert self.pad
                collated_sources[i] = torch.cat(
                    [source, source.new_full((-diff,), 0.0)]
                )
                padding_mask[i, diff:] = True
            else:
                collated_sources[i] = self.crop_to_max_size(source, target_size)

        input = {"source": collated_sources}
        if self.pad:
            input["padding_mask"] = padding_mask

        audio_feats = []
        if self.fbank_dim > 0:
            fbank = torch.stack(
                [
                    torchaudio.compliance.kaldi.fbank(
                        feats.view(1, -1),
                        num_mel_bins=self.fbank_dim,
                        sample_frequency=self.sample_rate,
                        frame_length=self.frame_length,
                        frame_shift=self.frame_shift,
                    )
                    for feats in collated_sources
                ],
                dim=0,
            )
            audio_feats.append(fbank)
        if self.mfcc_dim > 0:
            mfcc = torch.stack(
                [
                    torchaudio.compliance.kaldi.mfcc(
                        feats.view(1, -1),
                        num_ceps=self.mfcc_dim,
                        sample_frequency=self.sample_rate,
                        frame_length=self.frame_length,
                        frame_shift=self.frame_shift,
                    )
                    for feats in collated_sources
                ],
                dim=0,
            )
            audio_feats.append(mfcc)
        if len(audio_feats) == 0:
            audio_feats = None
        elif len(audio_feats) == 1:
            audio_feats = audio_feats[0]
        else:
            audio_feats = torch.cat(audio_feats, dim=-1)

        if audio_feats is not None:
            input["audio_feats"] = audio_feats

        return {
            "id": torch.LongTensor([s["id"] for s in samples]),
            "target": torch.LongTensor([s["target"] for s in samples]),
            "net_input": input,
        }
