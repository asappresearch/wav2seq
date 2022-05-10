import json
import sys
import os
import fire
from colorama import Fore
import numpy as np
import glob


def get_size(filename, sr=16000):
    with open(filename) as f:
        f.readline()
        try:
            size = sum([int(line.strip().split()[1]) for line in f])
            hr = size / sr / 3600
            print(f"{filename}: {hr:.2f} hr")
        except:
            print(f"{filename}: {Fore.RED}failed{Fore.RESET}")


def list_manifest(root="manifest"):
    for dirname, dirs, files in os.walk(root):
        for filename in files:
            if filename.endswith(".tsv"):
                get_size(os.path.join(dirname, filename))


def check_exists(filename):
    with open(filename) as f:
        root = f.readline().strip()
        import pdb

        pdb.set_trace()
        for line in f:
            filename = os.path.join(root, line.split()[0])
            if not os.path.exists(filename):
                print(filename)


def subsample(input_file, output_file, keep):
    with open(input_file) as f:
        lines = [line.strip() for line in f]
    with open(output_file, "w") as f:
        for i in keep:
            print(lines[i], file=f)


def subsample_manifest(
    folder, input_split, duration=10, seed=0,
):
    prefix = os.path.join(folder, input_split)
    with open(prefix + ".tsv") as f:
        root = f.readline().strip()
        data = [line.strip().split() for line in f]

    perm = np.random.RandomState(seed=seed).permutation(len(data))
    limit = duration * 3600 * 16000
    curr_duration = 0
    keep = []
    for i in perm:
        keep.append(i)
        curr_duration += int(data[int(i)][1])
        if curr_duration >= limit:
            break
    keep.sort()

    new_prefix = prefix.replace(input_split, f"{input_split}_{duration}h")
    with open(new_prefix + ".tsv", "w") as f:
        print(root, file=f)
        for i in keep:
            print("\t".join(data[i]), file=f)

    for filename in [prefix + ".wrd", prefix + ".ltr"]:
        if os.path.exists(filename):
            subsample(filename, filename.replace(prefix, new_prefix), keep)

    print(f"{input_split}_{duration}h {curr_duration / 3600 / 16000:.4f} h")


def duplicate_manifest(
    folder, input_split, duplications,
):
    prefix = os.path.join(folder, input_split)
    filenames = glob.glob(f"{prefix}.*")
    print(filenames)

    for filename in filenames:
        output_name = filename.replace(prefix, prefix + f"x{duplications}")
        with open(filename) as fin, open(output_name, "w") as fout:
            if filename.endswith(".tsv"):
                root = fin.readline().strip()
                print(root, file=fout)
            for line in fin:
                line = line.rstrip()
                for _ in range(duplications):
                    print(line, file=fout)


def get_s2t_subset(
    split="train-10",
    w2v_folder="manifest/librispeech",
    s2t_folder="s2t-manifest/librispeech",
):
    output_file = os.path.join(s2t_folder, split + ".tsv")
    assert not os.path.exists(output_file)
    kept = []
    with open(os.path.join(w2v_folder, split + ".tsv")) as f:
        f.readline()
        for line in f:
            audio_path = os.path.basename(line.split("\t")[0])
            uid = os.path.splitext(audio_path)[0]
            uid = "-".join([str(int(i)) for i in uid.split("-")])
            kept.append(uid)

    line_map = dict()
    for filename in [
        "train-clean-100.tsv",
        "train-clean-360.tsv",
        "train-other-500.tsv",
    ]:
        with open(os.path.join(s2t_folder, filename)) as f:
            header = f.readline().rstrip()
            for line in f:
                line = line.rstrip()
                uid = line.split("\t")[0]
                line_map[uid] = line

    with open(output_file, "w") as f:
        print(header, file=f)
        for uid in kept:
            print(line_map[uid], file=f)


if __name__ == "__main__":
    fire.Fire()
