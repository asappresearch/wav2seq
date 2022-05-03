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


def get_num_updates(folder):
    num_updates = 0
    try:
        with open(os.path.join(folder, 'hydra_train.log')) as f:
            lines = [line.strip() for line in f][-50:]
            for line in lines:
                try:
                    start = line.find('num_updates": "')
                    if start >= 0:
                        start += len('num_updates": "')
                        end = line[start:].find('",') + start
                        if end >= start:
                            num_updates = int(line[start:end])
                except:
                    continue
        return num_updates
    except:
        return 0

def print_folder(folder, total_updates):
    num_updates = get_num_updates(folder)
    if num_updates >= total_updates:
        color = Fore.GREEN
    else:
        color = Fore.RED
    print(f"{folder}\t{color}{num_updates}{Fore.RESET}")

def main(root="exp-bu", total_updates=100_000):
    exp_dirs = set()
    match_names = set()
    for dirname, dirs, files in tqdm(os.walk(root)):
        if 'checkpoints' in dirs:
            exp_dirs.add(dirname)
    exp_dirs = sorted(exp_dirs)

    evaled = []
    not_evaled = []
    for x in exp_dirs:
        if os.path.exists(f"{x}/eval.log"):
            evaled.append(x)
        else:
            not_evaled.append(x)

    print("Evaluated:")
    for folder in evaled:
        print_folder(folder, total_updates)
    print("\nNot evaluated:")
    for folder in not_evaled:
        print_folder(folder, total_updates)

if __name__ == "__main__":
    fire.Fire(main)
