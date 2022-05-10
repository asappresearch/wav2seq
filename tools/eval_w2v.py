# Copyright (c) ASAPP Inc.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# from genericpath import exists
import os
import sys
import fire
import shlex
import subprocess

import pandas as pd

lang_pair_dict = {
    "X-high_en": ["fr_en", "de_en", "es_en", "ca_en"],
    "X-mid_en": ["fa_en", "it_en", "ru_en", "pt_en", "zh-CN_en"],
    "X-low_en": [
        "tr_en",
        "ar_en",
        "et_en",
        "mn_en",
        "nl_en",
        "sv-SE_en",
        "lv_en",
        "sl_en",
        "ta_en",
        "ja_en",
        "id_en",
        "cy_en",
    ],
}
lang_pair_dict["X_en"] = (
    lang_pair_dict["X-low_en"]
    + lang_pair_dict["X-mid_en"]
    + lang_pair_dict["X-high_en"]
)


def run_exp(
    model="save/pretrained/wav2vec_small_100h.pt",
    batch_scale=1,
    lm="nolm-argmax",
    beam_size=50,
    lm_weight=2.0,
    word_score=-1.0,
    subset="dev-other",
    data="manifest/librispeech",
    upsample=1.0,
    save_results=False,
    dump_emissions=False,
    ctc_temp=1.0,
    csv_log_file="exp-eval-logs.csv",
    fp16=False,
    batch_size=-1,
    quiet=False,
    use_bpe=False,
):
    if os.path.isdir(model):
        ckpt = os.path.join(model, "checkpoints/checkpoint_best.pt")
        if save_results:
            if "nolm" in lm:
                results_path = os.path.join(model, "decode", subset, lm)
            else:
                results_path = os.path.join(
                    model,
                    "decode",
                    subset,
                    f"{lm}-b{beam_size}-lw{lm_weight}-ws{word_score}",
                )
        else:
            results_path = None
        emission_path = (
            os.path.join(model, "decode", subset, "emissions.npy")
            if dump_emissions
            else None
        )
    else:
        ckpt = model
        if save_results:
            if "nolm" in lm:
                results_path = os.path.join(
                    os.path.basename(model) + "-decode", subset, lm
                )
            else:
                results_path = os.path.join(
                    os.path.basename(model) + "-decode",
                    subset,
                    f"{lm}-b{beam_size}-lw{lm_weight}-ws{word_score}",
                )
        else:
            results_path = None
        emission_path = (
            os.path.join(os.path.basename(model) + "-decode", subset)
            if dump_emissions
            else None
        )

    if not quiet:
        print(f"ckpt: {ckpt}")
        print(f"lm: {lm}")
        if "nolm" not in lm:
            print(
                f"lm_weight: {lm_weight} word_score: {word_score} beam_size: {beam_size}"
            )

    user_dir = os.path.abspath("pseudo_language")
    max_tokens = 4000000 * batch_scale

    cmd = (
        f"python tools/infer.py {data}"
        f" --user-dir {user_dir}"
        f" --task audio_finetuning"
        f" --nbest 1 --path {ckpt} --gen-subset {subset}"
        f" --sil-weight 0 --max-tokens {max_tokens}"
        f" --lm-weight {lm_weight} --word-score {word_score}"
        f" --criterion ctc"
        f" --beam {beam_size}"
        f" --eval-upsample {upsample}"
        # f" --task audio_pretraining"
        # f" --beam-size-token {beam_size}"
        # f" --beam-threshold {beam_size}"
    )

    if results_path is not None:
        cmd += f" --results-path {results_path}"
    if emission_path is not None:
        cmd += f" --dump-emissions {emission_path}"
    if "bpe" in ckpt or use_bpe:
        cmd += " --labels bpe --post-process sentencepiece"
    else:
        cmd += " --labels ltr --post-process letter"
    if ctc_temp != 1.0:
        cmd += f" --eval-temperature {ctc_temp}"
    if batch_size > 0:
        cmd += f" --batch-size {batch_size}"

    if lm == "nolm":
        cmd += " --w2l-decoder viterbi"
    elif lm == "nolm-argmax":
        cmd += " --w2l-decoder argmax"
    elif "s2s" in lm:
        cmd += " --w2l-decoder s2s"
        if lm == "lm-s2s":
            cmd += f" --lm-model ${lm}"
    else:
        cmd += f" --w2l-decoder kenlm --lm-model save/kenlm/{lm}/4gram.bin --lexicon save/kenlm/{lm}/lexicon.lst"

    if fp16:
        cmd += " --fp16"

    if "vox" in ckpt:
        cmd += " --normalize"

    if not quiet:
        print("cmd:")
        print(cmd)
    result = subprocess.run(
        shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    wer, bleu, time_used, model_size, extract_size = parse_result(result, quiet=quiet)

    if not quiet:
        print(
            f"WER: {wer} time_used: {time_used} model_size: {model_size} extract_size: {extract_size}"
        )
    msg = f"{subset},{model},{lm},{model_size},{extract_size},{time_used},{wer}"
    if not quiet:
        print(msg)
    if csv_log_file is not None:
        with open(csv_log_file, "a") as f:
            print(msg, file=f)

    return wer, time_used, model_size, extract_size


def parse_result(result, quiet=False):
    extract_size = 0
    model_size = 0
    wer = -1
    time_used = -1
    bleu = -1
    for line in result.stderr.decode("utf-8").split("\n"):
        if not quiet:
            print(line)
        pos = line.find("WER: ")
        if pos >= 0:
            wer = float(line[pos + 5 :].rstrip())

        pos = line.find("BLEU: ")
        if pos >= 0:
            bleu = float(line[pos + 6 :].rstrip())

        pos = line.find("time used: ")
        if pos >= 0:
            time_used = float(line[pos + 11 :].rstrip())

        query = "model 0 size: "
        pos = line.find(query)
        if pos >= 0:
            model_size = int(line[pos + len(query) :].rstrip())

        query = "w2v_encoder.w2v_model.feature_extractor size: "
        pos = line.find(query)
        if pos >= 0:
            extract_size += int(line[pos + len(query) :].rstrip())

        query = "w2v_encoder.w2v_model.spec_feature_extractor size: "
        pos = line.find(query)
        if pos >= 0:
            extract_size += int(line[pos + len(query) :].rstrip())

    return wer, bleu, time_used, model_size, extract_size


def tune_lm(
    model="save/pretrained/wav2vec_small_100h.pt",
    batch_scale=1,
    lms=["nolm-argmax", "librispeech-official"],
    beam_size=50,
    lm_weight=2.0,
    word_score=-1.0,
    subsets=["dev-other"],
    data="manifest/librispeech",
    upsample=1.0,
    save_results=False,
    dump_emissions=False,
    ctc_temp=1.0,
    csv_log_file="exp-eval-logs.csv",
    fp16=False,
    batch_size=-1,
    use_bpe=False,
):
    for lm in lms:
        for subset in subsets:
            try:
                run_exp(
                    model,
                    batch_scale,
                    lm,
                    beam_size=beam_size,
                    lm_weight=lm_weight,
                    word_score=word_score,
                    subset=subset,
                    data=data,
                    upsample=upsample,
                    save_results=save_results,
                    dump_emissions=dump_emissions,
                    ctc_temp=ctc_temp,
                    csv_log_file=csv_log_file,
                    fp16=fp16,
                    batch_size=batch_size,
                    use_bpe=use_bpe,
                )
            except:
                pass
            print("-" * 80)
            # only need to dump once per subset
            dump_emissions = False


def run_folder(
    root="save-ft-100h/example",
    batch_scale=1,
    lms=["nolm-argmax", "librispeech-official"],
    beam_size=50,
    lm_weight=2.0,
    word_score=-1.0,
    subsets=["dev-other"],
    data="manifest/librispeech",
    upsample=1.0,
    save_results=False,
    dump_emissions=False,
    ctc_temp=1.0,
    checkpoint_name="checkpoint_best.pt",
    skip=0,
    fp16=False,
    batch_size=-1,
    csv_log_file="exp-eval-logs.csv",
    use_bpe=False,
):
    exp_dirs = []
    for dirname, dirs, files in os.walk(root):
        if checkpoint_name in files:
            exp_dirs.append(os.path.join(dirname, checkpoint_name))
    print("skipped folders:", *exp_dirs[:skip], sep="\n")
    exp_dirs = exp_dirs[skip:]
    print("folders:", *exp_dirs, sep="\n")
    print("")

    for model in exp_dirs:
        tune_lm(
            model=model,
            batch_scale=batch_scale,
            lms=lms,
            beam_size=beam_size,
            lm_weight=lm_weight,
            word_score=word_score,
            subsets=subsets,
            data=data,
            upsample=upsample,
            save_results=save_results,
            dump_emissions=dump_emissions,
            ctc_temp=ctc_temp,
            fp16=fp16,
            batch_size=batch_size,
            csv_log_file=csv_log_file,
            use_bpe=use_bpe,
        )
        print("=" * 80)


def time_folder(
    file_list="tools/time_file_list.txt",
    output_file="time-eval-logs.csv",
    fp16=False,
    repeat=5,
):
    with open(file_list) as f:
        folders = [line.strip() for line in f]
    if os.path.exists(output_file):
        with open(output_file) as f:
            finished = set([line.split(",")[0].strip() for line in f])
    else:
        finished = set([])

    folders = [d for d in folders if d not in finished]
    print("finished:")
    print(*finished, sep="\n")
    print()
    print("folders:")
    print(*folders, sep="\n")
    print()

    for folder in folders:
        try:
            msg = f"{folder}"
            print(folder)
            for r in range(repeat):
                wer, time_used, model_size, extract_size = run_exp(
                    folder,
                    1,
                    lm="nolm-argmax",
                    subset="dev-other",
                    csv_log_file=None,
                    fp16=fp16,
                    quiet=True,
                )
                msg += f",{time_used}"
                print(time_used)

            with open(output_file, "a") as f:
                print(msg, file=f)
        except:
            pass

    return


def eval_s2s_asr(
    root="save-ft-100h/example",
    lms=["nolm-s2s"],
    beam_size=10,
    lm_weight=0.3,
    subsets=["dev-other"],
    csv_log_file="exp-eval-logs.csv",
    use_bpe=True,
):
    run_folder(
        root=root,
        lms=lms,
        beam_size=beam_size,
        lm_weight=lm_weight,
        subsets=subsets,
        csv_log_file=csv_log_file,
        use_bpe=use_bpe,
    )


def eval_st(
    model="save/pretrained/wav2vec_small_100h.pt",
    subset="dev",
    lm="nolm-s2s",
    beam_size=10,
    lenpen=1.0,
    labels="spm_bpe_v1000",
    lm_weight=0.2,
    word_score=-1.0,
    data="manifest/covost-v2/en_de",
    csv_log_file="",
    save_results=False,
    fp16=True,
    quiet=False,
    max_tokens=4_000_000,
    batch_size=-1,
    dump_emissions=False,
    lm_model="",
):
    lang_pair = os.path.basename(data)
    if os.path.isdir(model):
        ckpt = os.path.join(model, "checkpoints/checkpoint_best.pt")
    else:
        ckpt = model

    if csv_log_file == "":
        csv_log_file = os.path.join(
            "eval_logs",
            "-".join(data.split("/")[1:])
            + "-"
            + "-".join(model.split("/")[:2])
            + ".csv",
        )

    if save_results:
        if "nolm" in lm:
            results_path = os.path.join(
                os.path.splitext(model)[0],
                "decode",
                subset,
                f"{lm}-b{beam_size}-lp{lenpen}",
            )
        else:
            results_path = os.path.join(
                os.path.splitext(model)[0],
                "decode",
                subset,
                f"{lm}-b{beam_size}-lp{lenpen}-lw{lm_weight}-ws{word_score}",
            )
        os.makedirs(results_path, exist_ok=True)
    else:
        results_path = None
    emission_path = (
        os.path.join(model, "decode", subset, "emissions.npy")
        if dump_emissions
        else None
    )

    if not quiet:
        print(f"ckpt: {ckpt}")
        print(f"lm: {lm}")
        if "nolm" not in lm:
            print(
                f"lm_weight: {lm_weight} word_score: {word_score} beam_size: {beam_size}"
            )

    user_dir = os.path.abspath("pseudo_language")

    cmd = (
        f"python tools/infer.py {data}"
        f" --user-dir {user_dir}"
        f" --task audio_finetuning"
        f" --nbest 1 --path {ckpt} --gen-subset {subset}"
        f" --sil-weight 0 --max-tokens {max_tokens}"
        f" --lm-weight {lm_weight} --word-score {word_score}"
        f" --criterion ctc"
        f" --beam {beam_size}"
        f" --lenpen {lenpen}"
        f" --labels {labels}"
    )

    if results_path is not None:
        cmd += f" --results-path {results_path}"
    if emission_path is not None:
        cmd += f" --dump-emissions {emission_path}"
    if labels == "ltr":
        cmd += " --post-process letter"
    else:
        cmd += " --post-process sentencepiece"
    if batch_size > 0:
        cmd += f" --batch-size {batch_size}"

    if lm == "nolm":
        cmd += " --w2l-decoder viterbi"
    elif lm == "nolm-argmax":
        cmd += " --w2l-decoder argmax"
    elif "s2s" in lm:
        cmd += " --w2l-decoder s2s"
        if lm == "lm-s2s":
            cmd += f" --lm-model ${lm_model}"
    else:
        cmd += f" --w2l-decoder kenlm --lm-model save/kenlm/{lm}/4gram.bin --lexicon save/kenlm/{lm}/lexicon.lst"

    if fp16:
        cmd += " --fp16"

    # if "vox" in ckpt:
    #     cmd += " --normalize"

    if not quiet:
        print("cmd:")
        print(cmd)
    result = subprocess.run(
        shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    wer, bleu, time_used, model_size, extract_size = parse_result(result, quiet=quiet)

    if not quiet:
        print(
            f"WER: {wer} BLEU: {bleu} time_used: {time_used} model_size: {model_size} extract_size: {extract_size}"
        )
    msg = f"{lang_pair},{subset},{model},{beam_size},{lenpen},{lm},{model_size},{time_used},{wer},{bleu}"
    if not quiet:
        print(msg)
    if csv_log_file is not None:
        with open(csv_log_file, "a") as f:
            print(msg, file=f)

    return wer, bleu, time_used, model_size, extract_size, results_path


def tune_st(
    root="save-ft-100h/example",
    max_tokens=4_000_000,
    lm="nolm-s2s",
    beam_size=10,
    lm_weight=0.2,
    word_score=-1.0,
    data="manifest/covost-v2/en_de",
    save_results=False,
    checkpoint_name="checkpoint_best.pt",
    skip=0,
    batch_size=-1,
    lenpens=[1.0, 0.5, 1.5],
    labels="spm_bpe_v1000",
    fp16=True,
    csv_log_file="",
):
    exp_dirs = []
    for dirname, dirs, files in os.walk(root):
        if checkpoint_name in files:
            exp_dirs.append(os.path.join(dirname, checkpoint_name))
    print("skipped folders:", *exp_dirs[:skip], sep="\n")
    exp_dirs = exp_dirs[skip:]
    print("folders:", *exp_dirs, sep="\n")
    print("")
    lang_pair = os.path.basename(data)

    tune_log_file = "eval_logs/covost-tune_st-results.csv"
    if os.path.exists(tune_log_file):
        result_df = pd.read_csv(tune_log_file)
    else:
        with open(tune_log_file, "w") as f:
            print("model,lang_pair,best_lenpen,dev_bleu,test_bleu", file=f)

    for model in exp_dirs:

        dev_results = []
        for lenpen in lenpens:
            bleu = eval_st(
                subset="dev",
                lenpen=lenpen,
                model=model,
                lm=lm,
                beam_size=beam_size,
                lm_weight=lm_weight,
                word_score=word_score,
                data=data,
                save_results=save_results,
                fp16=fp16,
                batch_size=batch_size,
                csv_log_file=csv_log_file,
                max_tokens=max_tokens,
                labels=labels,
            )[1]
            dev_results.append((bleu, lenpen))

        best_bleu, best_lenpen = max(dev_results)

        print(f"dev_results: {dev_results}")

        test_bleu = eval_st(
            subset="test",
            lenpen=best_lenpen,
            model=model,
            lm=lm,
            beam_size=beam_size,
            lm_weight=lm_weight,
            word_score=word_score,
            data=data,
            save_results=False,
            fp16=fp16,
            batch_size=batch_size,
            csv_log_file=csv_log_file,
            max_tokens=max_tokens,
            labels=labels,
        )[1]
        print(
            f"{lang_pair}: best lenpen {best_lenpen} dev bleu: {best_bleu} test bleu: {test_bleu}"
        )
        print("=" * 80 + "\n")
        with open(tune_log_file, "a") as f:
            msg = f"{model},{lang_pair},{best_lenpen},{best_bleu},{test_bleu}"
            print(msg, file=f)


def tune_st_to_en_all(
    root="save-ft-100h/example",
    max_tokens=4_000_000,
    lm="nolm-s2s",
    beam_size=10,
    lm_weight=0.2,
    word_score=-1.0,
    data="manifest/covost-v2",
    save_results=False,
    checkpoint_name="checkpoint_best.pt",
    skip=0,
    batch_size=-1,
    lenpens=[1.0, 0.5, 1.5],
    labels="spm_bpe_v1000",
    fp16=True,
    csv_log_file="",
    skip_lang_pair="",
    lang_set="X_en",
):
    if isinstance(skip_lang_pair, str):
        skip_lang_pair = set(skip_lang_pair.split(","))
    else:
        skip_lang_pair = set(skip_lang_pair)
    print("skip_lang_pair:", skip_lang_pair)
    eval_lang_pairs = lang_pair_dict[lang_set]
    for lang_pair in eval_lang_pairs:
        if lang_pair in skip_lang_pair:
            continue
        tune_st(
            data=os.path.join(data, lang_pair),
            root=root,
            max_tokens=max_tokens,
            lm=lm,
            beam_size=beam_size,
            lm_weight=lm_weight,
            word_score=word_score,
            save_results=save_results,
            checkpoint_name=checkpoint_name,
            skip=skip,
            batch_size=batch_size,
            lenpens=lenpens,
            labels=labels,
            fp16=fp16,
            csv_log_file=csv_log_file,
        )


if __name__ == "__main__":
    fire.Fire()
