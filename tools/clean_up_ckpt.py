# Copyright (c) ASAPP Inc.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import fire
import glob


def update_model_name(model_name):
    to_new_name = {
        "squeeze_wav2vec2_deberta": "sew-d",
        "squeeze_wav2vec2": "sew",
    }
    if model_name in to_new_name:
        return to_new_name[model_name]
    else:
        return model_name


def clean_up_ckpt(filename):
    exp_name = filename.replace(".pt", "").rsplit("-", 1)[0]
    ckpt = torch.load(filename, map_location="cpu")
    ckpt["cfg"]["common"]["user_dir"] = "/persist/git/sew/sew_asapp"
    ckpt["cfg"]["common"][
        "tensorboard_logdir"
    ] = f"/persist/git/sew/tb-logs-pre/{exp_name}"
    ckpt["cfg"]["task"]["data"] = f"/persist/git/sew/manifest/librispeech/train-960"
    # ckpt['cfg']['model']['_name'] = update_model_name(ckpt['cfg']['model']['_name'])


def main():
    for filename in glob.glob("*.pt"):
        clean_up_ckpt(filename)
        print(f"{filename} done")


if __name__ == "__main__":
    fire.Fire(main)
