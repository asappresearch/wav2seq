# Copyright (c) ASAPP Inc.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import fire
import os
import sys
from tqdm.auto import tqdm


def main(input_file, output_file):
    assert not os.path.exists(output_file), f"{output_file} exists"
    with open(input_file) as fin, open(output_file, 'w') as fout:
        for line in tqdm(fin):
            s = ' '.join(line.strip().replace(' ', '|'))
            print(s, file=fout)

if __name__ == '__main__':
    fire.Fire(main)