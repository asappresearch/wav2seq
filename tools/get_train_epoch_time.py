# Copyright (c) ASAPP Inc.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import fire
import os
import sys
import time
from tqdm.auto import tqdm
import re
import json
from colorama import Fore

import numpy as np


def get_train_epoch_time(folder):
    train_log = os.path.join(folder, "hydra_train.log")
    prev_time = -1
    prev_epoch = 0
    prev_updates = 0
    epoch_times = []
    epoch = 0
    iter_times = []
    with open(train_log) as f:
        for line in f:
            if "[train]" in line:
                # try:
                epoch_log = json.loads(line.split(" - ", maxsplit=1)[1])
                if prev_time < 0:
                    prev_time = float(epoch_log["train_wall"])
                    prev_epoch = int(epoch_log["epoch"])
                    prev_updates = int(epoch_log["train_num_updates"])
                else:
                    curr_time = float(epoch_log["train_wall"])
                    curr_epoch = int(epoch_log["epoch"])
                    curr_updates = int(epoch_log["train_num_updates"])
                    if curr_epoch - prev_epoch == 1 and curr_time - prev_time > 0:
                        epoch_times.append(curr_time - prev_time)
                        iter_times.append(
                            (curr_time - prev_time) / (curr_updates - prev_updates)
                        )
                    prev_epoch, prev_time = curr_epoch, curr_time
                    prev_updates = curr_updates
                # except:
                #     pass

    epoch_time = np.median(epoch_times)
    iter_time = np.median(iter_times)
    return epoch_time, iter_time


def main(root="exp-bu", total_updates=100_000):
    exp_dirs = set()
    match_names = set()
    for dirname, dirs, files in tqdm(os.walk(root)):
        if "checkpoints" in dirs:
            exp_dirs.add(dirname)
    exp_dirs = sorted(exp_dirs)

    for folder in exp_dirs:
        epoch_time, iter_time = get_train_epoch_time(folder)
        train_hours = iter_time * total_updates / 3600
        print(f"{folder} {epoch_time} {train_hours:.2f}")


if __name__ == "__main__":
    fire.Fire(main)
