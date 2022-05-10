import os
import sys
import fire
import numpy as np


def check_bpe_ratio(split="dev-other", suffix="bpe", label_dir="manifest/librispeech"):
    with open(os.path.join(label_dir, f"{split}.tsv")) as f:
        f.readline()
        input_lengths = np.array(
            [(int(line.strip().split()[1]) - 80) // 320 for line in f]
        )
    with open(os.path.join(label_dir, f"{split}.{suffix}")) as f:
        output_lengths = np.array([len(line.strip().split()) for line in f])
    ratio = input_lengths / output_lengths
    print(f"avg input lengths: {np.mean(input_lengths)}")
    print(f"avg output lengths: {np.mean(output_lengths)}")
    print(f"min_ratio: {np.min(ratio)}")
    print(f"avg_ratio: {np.mean(ratio)}")
    print(f"max_ratio: {np.max(ratio)}")
    print(f"compression: {1 / np.mean(ratio)}")


def check_length(
    folder="labels/hubert_large_ll60k-l18-k1s1-fp16-ls0.1/c25", suffix="km"
):
    input_lengths = []
    with open("manifest/librispeech/train-960/valid.tsv") as f:
        f.readline()
        for line in f:
            line = line.strip().split()
            input_lengths.append((int(line[1]) - 80) // 320)
    input_lengths = np.array(input_lengths)
    input_l = np.mean(input_lengths)

    original_lengths, dedup_lengths = [], []
    with open(os.path.join(folder, f"valid.{suffix}")) as f:
        for line in f:
            line = line.strip().split()
            original_lengths.append(len(line))
            line = [
                line[i] for i in range(len(line)) if i == 0 or line[i] != line[i - 1]
            ]
            dedup_lengths.append(len(line))
    original_lengths = np.array(original_lengths)
    dedup_lengths = np.array(dedup_lengths)
    original_ratio = input_lengths / original_lengths
    dedup_ratio = input_lengths / dedup_lengths

    print(folder)
    print(f"input:")
    print(f"avg_len: {np.mean(input_lengths)}")
    print(f"max_len: {np.max(input_lengths)}")

    print("raw:")
    print(f"avg_len: {np.mean(original_lengths)}")
    print(f"max_len: {np.max(original_lengths)}")
    print(f"min_ratio: {np.min(original_ratio)}")
    print(f"avg_ratio: {np.mean(original_ratio)}")
    print(f"max_ratio: {np.max(original_ratio)}")
    print(f"compression: {1 / np.mean(original_ratio)}")

    print("dedup:")
    print(f"avg_len: {np.mean(dedup_lengths)}")
    print(f"max_len: {np.max(dedup_lengths)}")
    print(f"min_ratio: {np.min(dedup_ratio)}")
    print(f"avg_ratio: {np.mean(dedup_ratio)}")
    print(f"max_ratio: {np.max(dedup_ratio)}")
    print(f"compression: {1 / np.mean(dedup_ratio)}")


if __name__ == "__main__":
    fire.Fire()
