# Copyright (c) ASAPP Inc.
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys

from tqdm.auto import tqdm
from npy_append_array import NpyAppendArray


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("feature_utils")


def get_shard_range(tot, nshard, rank):
    assert rank < nshard and rank >= 0, f"invaid rank/nshard {rank}/{nshard}"
    start = round(tot / nshard * rank)
    end = round(tot / nshard * (rank + 1))
    assert start < end, f"start={start}, end={end}"
    logger.info(
        f"rank {rank} of {nshard}, process {end-start} " f"({start}-{end}) out of {tot}"
    )
    return start, end


def get_path_iterator(tsv, nshard, rank):
    with open(tsv, "r") as f:
        root = f.readline().rstrip()
        lines = [line.rstrip() for line in f]
        start, end = get_shard_range(len(lines), nshard, rank)
        lines = lines[start:end]

        def iterate():
            for line in lines:
                subpath, nsample = line.split("\t")
                yield f"{root}/{subpath}", int(nsample)

    return iterate, len(lines)


def dump_feature(rank, reader, manifest, nshard, feat_dir):
    generator, num = get_path_iterator(manifest, nshard, rank)
    iterator = generator()

    split = os.path.splitext(os.path.basename(manifest))[0]
    feat_path = f"{feat_dir}/{split}_{rank}_{nshard}.npy"
    leng_path = f"{feat_dir}/{split}_{rank}_{nshard}.len"

    os.makedirs(feat_dir, exist_ok=True)
    if os.path.exists(feat_path):
        os.remove(feat_path)

    feat_f = NpyAppendArray(feat_path)
    with open(leng_path, "w") as leng_f:
        for path, nsample in tqdm(iterator, total=num):
            feat = reader.get_feats(path, nsample)
            feat_f.append(feat.cpu().numpy())
            leng_f.write(f"{len(feat)}\n")
    logger.info("finished successfully")


def dump_cluster(
    rank,
    reader,
    apply_kmeans,
    manifest,
    nshard,
    lab_dir,
    output_suffix="km",
    pl_converter=None,
    use_tqdm=True,
):
    generator, num = get_path_iterator(manifest, nshard, rank)
    iterator = generator()

    split = os.path.splitext(os.path.basename(manifest))[0]
    if rank == 0 and nshard == 1:
        lab_path = f"{lab_dir}/{split}.{output_suffix}"
    else:
        lab_path = f"{lab_dir}/{split}_{rank}_{nshard}.{output_suffix}"

    os.makedirs(lab_dir, exist_ok=True)
    if os.path.exists(lab_path):
        with open(lab_path) as f:
            lines = [line.strip() for line in f]
        skip = len(lines)
    else:
        skip = 0
        # os.remove(lab_path)

    with open(lab_path, "a") as f:
        if use_tqdm:
            iterator = tqdm(iterator, total=num)
        for i, (path, nsample) in enumerate(iterator):
            if i < skip:
                continue
            feat = reader.get_feats(path, nsample)
            cluster_ids = apply_kmeans(feat).tolist()
            if pl_converter is None:
                print(" ".join(map(str, cluster_ids)), file=f)
            else:
                pl_ids = pl_converter(cluster_ids)
                print(" ".join(map(str, pl_ids)), file=f)

    logger.info("finished successfully")
