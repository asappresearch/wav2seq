import librosa
import soundfile
import fire
import os
import sys

from functools import partial
from tqdm.auto import tqdm
from multiprocessing import Pool


def get_start_ends(length, min_length, max_length):
    start = 0
    while start < length:
        if length - start <= min_length:
            return
        elif length - start <= max_length:
            end = length
        elif length - start <= min_length + max_length:
            end = length - min_length
        else:
            end = start + max_length
        yield (start, end)
        start = end


def split_file(
    files, ext="flac", sample_rate=16000, min_duration=10.0, max_duration=30.0
):
    input_file, output_prefix = files
    if not os.path.exists(f"{output_prefix}.done"):
        data, _ = librosa.load(input_file, sample_rate)
        min_length, max_length = (
            int(sample_rate * min_duration),
            int(sample_rate * max_duration),
        )

        os.makedirs(os.path.dirname(output_prefix), exist_ok=True)

        for i, (s, e) in enumerate(get_start_ends(len(data), min_length, max_length)):
            soundfile.write(
                f"{output_prefix}_{i}.{ext}", data[s:e], samplerate=sample_rate
            )
        os.system(f"touch {output_prefix}.done")


def split_folder(
    input_dir,
    output_dir,
    njobs=6,
    ext="flac",
    sample_rate=16000,
    min_duration=10.0,
    max_duration=30.0,
):
    input_dir = os.path.abspath(input_dir)
    output_dir = os.path.abspath(output_dir)

    os.makedirs(output_dir, exist_ok=True)
    input_filenames = []

    for (dirpath, dirnames, filenames) in os.walk(input_dir):
        for filename in filenames:
            if filename.endswith(".flac"):
                filename = os.path.join(dirpath, filename)
                input_filenames.append(
                    (
                        filename,
                        filename.replace(input_dir, output_dir).replace(".flac", ""),
                    )
                )

    print(input_filenames[:10])

    func = partial(
        split_file,
        ext=ext,
        sample_rate=sample_rate,
        min_duration=min_duration,
        max_duration=max_duration,
    )
    with Pool(njobs) as p:
        list(tqdm(p.imap(func, input_filenames), total=len(input_filenames)))


if __name__ == "__main__":
    fire.Fire(split_folder)
