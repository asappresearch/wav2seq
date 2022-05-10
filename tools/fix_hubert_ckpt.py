import sys
import os
import torch
import fire


def main(
    input_file="save/pretrained/hubert_base_ls960.pt",
    ref_file="save-pt/0821/083021/hubert-small/hubert-D256F1024H4L12-WFE-C-c128-l0-ld0.05-lr5e-4-pl_C500_km-batch1x-mt1M/checkpoints/checkpoint_last.pt",
    output_file=None,
):
    if output_file is None:
        output_file = input_file.replace(".pt", "_fix.pt")

    state1 = torch.load(input_file, map_location="cpu")
    state2 = torch.load(ref_file, map_location="cpu")
    # state2['cfg']['model']['encoder_attention_heads'] = 12

    args_old = vars(state1["args"])

    for k in args_old:
        if k in {"labels", "label_rate", "latent_temp"}:
            continue
        for group in state2["cfg"]:
            if isinstance(state2["cfg"][group], dict) and k in state2["cfg"][group]:
                # if k == 'labels':
                #     args_old[k] = eval(args_old[k])
                if type(state2["cfg"][group][k]) == type(args_old[k]):
                    state2["cfg"][group][k] = args_old[k]
                    print(f"setting cfg.{group}.{k}={args_old[k]}")

    state1["args"] = state2["args"]
    state1["cfg"] = state2["cfg"]
    torch.save(state1, output_file)


if __name__ == "__main__":
    fire.Fire(main)
