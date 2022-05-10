#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Run inference for pre-processed data with a trained model.
"""

import logging
import math
import os
import sys

import sacrebleu
import editdistance
import numpy as np
import torch
from fairseq import checkpoint_utils, options, progress_bar, tasks, utils
from fairseq.data.data_utils import post_process
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging.meters import StopwatchMeter, TimeMeter


logging.basicConfig()
logging.root.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def add_asr_eval_argument(parser):
    parser.add_argument("--kspmodel", default=None, help="sentence piece model")
    parser.add_argument(
        "--wfstlm", default=None, help="wfstlm on dictonary output units"
    )
    parser.add_argument(
        "--rnnt_decoding_type",
        default="greedy",
        help="wfstlm on dictonary\
output units",
    )
    try:
        parser.add_argument(
            "--lm-weight",
            type=float,
            default=0.2,
            help="weight for lm while interpolating with neural score",
        )
    except:
        pass
    parser.add_argument(
        "--len-penalty", type=float, default=1.0, help="length penalty on word level"
    )
    parser.add_argument(
        "--w2l-decoder",
        # choices=["viterbi", "kenlm", "fairseqlm"],
        help="use a w2l decoder",
    )
    parser.add_argument("--lexicon", help="lexicon for w2l decoder")
    parser.add_argument("--unit-lm", action="store_true", help="if using a unit lm")
    parser.add_argument("--kenlm-model", "--lm-model", help="lm model for w2l decoder")
    parser.add_argument("--beam-threshold", type=float, default=25.0)
    parser.add_argument("--beam-size-token", type=float, default=100)
    parser.add_argument("--word-score", type=float, default=1.0)
    parser.add_argument("--unk-weight", type=float, default=-math.inf)
    parser.add_argument("--sil-weight", type=float, default=0.0)
    parser.add_argument(
        "--dump-emissions",
        type=str,
        default=None,
        help="if present, dumps emissions into this file and exits",
    )
    parser.add_argument(
        "--dump-features",
        type=str,
        default=None,
        help="if present, dumps features into this file and exits",
    )
    parser.add_argument(
        "--load-emissions",
        type=str,
        default=None,
        help="if present, loads emissions from this file",
    )
    parser.add_argument(
        "--eval-temperature",
        type=float,
        default=1.0,
        help="temperature scaling of the logits",
    )
    parser.add_argument(
        "--eval-upsample", type=float, default=1.0, help="upsample factor",
    )
    parser.add_argument(
        "--ctc-weight", type=float, default=0.0, help="ctc weight",
    )
    return parser


def check_args(args):
    # assert args.path is not None, "--path required for generation!"
    # assert args.results_path is not None, "--results_path required for generation!"
    assert (
        not args.sampling or args.nbest == args.beam
    ), "--sampling requires --nbest to be equal to --beam"
    assert (
        args.replace_unk is None or args.raw_text
    ), "--replace-unk requires a raw text dataset (--raw-text)"


def get_dataset_itr(args, task, models):
    return task.get_batch_iterator(
        dataset=task.dataset(args.gen_subset),
        max_tokens=args.max_tokens,
        max_sentences=args.batch_size,
        max_positions=(sys.maxsize, sys.maxsize),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=args.required_batch_size_multiple,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        num_workers=args.num_workers,
        data_buffer_size=args.data_buffer_size,
    ).next_epoch_itr(shuffle=False)


def process_predictions(
    args, hypos, sp, tgt_dict, target_tokens, res_files, speaker, id
):
    for hypo in hypos[: min(len(hypos), args.nbest)]:
        hyp_pieces = tgt_dict.string(hypo["tokens"].int().cpu())

        if "words" in hypo:
            hyp_words = " ".join(hypo["words"])
        else:
            hyp_words = post_process(hyp_pieces, args.post_process)

        if res_files is not None:
            print(
                "{} ({}-{})".format(hyp_pieces, speaker, id),
                file=res_files["hypo.units"],
            )
            print(
                "{} ({}-{})".format(hyp_words, speaker, id),
                file=res_files["hypo.words"],
            )

        tgt_pieces = tgt_dict.string(target_tokens)
        tgt_words = post_process(tgt_pieces, args.post_process)

        if res_files is not None:
            print(
                "{} ({}-{})".format(tgt_pieces, speaker, id),
                file=res_files["ref.units"],
            )
            print(
                "{} ({}-{})".format(tgt_words, speaker, id), file=res_files["ref.words"]
            )
            # only score top hypothesis
            if not args.quiet:
                logger.debug("HYPO:" + hyp_words)
                logger.debug("TARGET:" + tgt_words)
                logger.debug("___________________")

        try:
            bleu = sacrebleu.corpus_bleu([hyp_words], [[tgt_words]])
        except:
            logger.warning("HYPO:" + hyp_words)
            logger.warning("TARGET:" + tgt_words)
            bleu = None
        hyp_words = hyp_words.split()
        tgt_words = tgt_words.split()
        return editdistance.eval(hyp_words, tgt_words), len(tgt_words), bleu


def prepare_result_files(args):
    def get_res_file(file_prefix):
        if args.num_shards > 1:
            file_prefix = f"{args.shard_id}_{file_prefix}"
        path = os.path.join(
            args.results_path,
            f"{file_prefix}-{args.gen_subset}",
            # args.results_path, f"{file_prefix}-{os.path.basename(args.path)}-{args.gen_subset}",
        )
        return open(path, "w", buffering=1)

    if not args.results_path:
        return None

    return {
        "hypo.words": get_res_file("hypo.word"),
        "hypo.units": get_res_file("hypo.units"),
        "ref.words": get_res_file("ref.word"),
        "ref.units": get_res_file("ref.units"),
    }


def load_models_and_criterions(
    filenames, data_path, arg_overrides=None, task=None, model_state=None
):
    models = []
    criterions = []

    if arg_overrides is None:
        arg_overrides = {}

    arg_overrides["wer_args"] = None
    arg_overrides["data"] = data_path
    arg_overrides["label_dir"] = data_path

    if filenames is None:
        assert model_state is not None
        filenames = [0]
    else:
        filenames = filenames.split(":")

    for filename in filenames:
        if model_state is None:
            if not os.path.exists(filename):
                raise IOError("Model file not found: {}".format(filename))
            state = checkpoint_utils.load_checkpoint_to_cpu(filename, arg_overrides)
        else:
            state = model_state

        if "cfg" in state:
            cfg = state["cfg"]
        else:
            cfg = convert_namespace_to_omegaconf(state["args"])

        if task is None:
            if hasattr(cfg.task, "data"):
                cfg.task.data = data_path
            task = tasks.setup_task(cfg.task)

        model = task.build_model(cfg.model)
        model.load_state_dict(state["model"], strict=True)
        models.append(model)

        criterion = task.build_criterion(cfg.criterion)
        if state.get("criterion", None) is not None:
            criterion.load_state_dict(state["criterion"], strict=True)
        criterions.append(criterion)
    return models, criterions, task


def optimize_models(args, use_cuda, models):
    """Optimize ensemble for generation"""
    for model in models:
        model.make_generation_fast_(
            beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
            need_attn=args.print_alignment,
        )
        if args.fp16:
            model.half()
        if use_cuda:
            model.cuda()


class ExistingEmissionsDecoder(object):
    def __init__(self, decoder, emissions):
        self.decoder = decoder
        self.emissions = emissions

    def generate(self, models, sample, **unused):
        ids = sample["id"].cpu().numpy()
        try:
            emissions = np.stack(self.emissions[ids])
        except:
            print([x.shape for x in self.emissions[ids]])
            raise Exception("invalid sizes")
        emissions = torch.from_numpy(emissions)
        return self.decoder.decode(emissions)


def get_num_param(model):
    return sum([p.numel() for p in model.parameters()])


def main(args, task=None, model_state=None):
    check_args(args)

    if args.max_tokens is None and args.batch_size is None:
        args.max_tokens = 4000000
    logger.info(args)

    use_cuda = torch.cuda.is_available() and not args.cpu

    logger.info("| decoding with criterion {}".format(args.criterion))

    # Load ensemble
    if args.load_emissions:
        models, criterions = [], []
        task = tasks.setup_task(args)
    else:
        # task = tasks.setup_task(args)
        logger.info("| loading model(s) from {}".format(args.path))
        models, criterions, task = load_models_and_criterions(
            args.path,
            data_path=args.data,
            arg_overrides=eval(args.model_overrides),  # noqa
            # task=task,
            model_state=model_state,
        )
        optimize_models(args, use_cuda, models)
        for i, model in enumerate(models):
            logger.info(f"| model {i} size: {get_num_param(model)}")
            for name, m in model.named_children():
                logger.info(f"| | model {i} {name} size: {get_num_param(m)}")
                for name2, m2 in m.named_children():
                    logger.info(
                        f"| | | model {i} {name}.{name2} size: {get_num_param(m2)}"
                    )
                    for name3, m3 in m2.named_children():
                        logger.info(
                            f"| | | | model {i} {name}.{name2}.{name3} size: {get_num_param(m3)}"
                        )

    # Load dataset splits
    task.load_dataset(args.gen_subset)

    # Set dictionary
    tgt_dict = task.target_dictionary

    logger.info(
        "| {} {} {} examples".format(
            args.data, args.gen_subset, len(task.dataset(args.gen_subset))
        )
    )

    # hack to pass transitions to W2lDecoder
    if args.criterion == "asg_loss":
        trans = criterions[0].asg.trans.data
        args.asg_transitions = torch.flatten(trans).tolist()

    # Load dataset (possibly sharded)
    itr = get_dataset_itr(args, task, models)

    # Initialize generator
    gen_timer = StopwatchMeter()

    def build_generator(args):
        w2l_decoder = getattr(args, "w2l_decoder", None)
        if w2l_decoder == "viterbi":
            # from examples.speech_recognition.w2l_decoder import W2lViterbiDecoder
            from wav2seq.decoder.w2l_decoder_old import W2lViterbiDecoder

            return W2lViterbiDecoder(args, task.target_dictionary)
        elif w2l_decoder == "kenlm":
            # from examples.speech_recognition.w2l_decoder import W2lKenLMDecoder
            from wav2seq.decoder.w2l_decoder_old import W2lKenLMDecoder

            return W2lKenLMDecoder(args, task.target_dictionary)
        elif w2l_decoder == "fairseqlm":
            from wav2seq.decoder.w2l_decoder_old import W2lFairseqLMDecoder

            return W2lFairseqLMDecoder(args, task.target_dictionary)
        elif w2l_decoder == "argmax":
            from wav2seq.decoder.ctc_decoder import CTCArgMaxDecoder

            return CTCArgMaxDecoder(args, task.target_dictionary)
        elif w2l_decoder == "s2s":
            from fairseq.dataclass.configs import GenerationConfig

            seq_gen_cls = None
            if args.ctc_weight > 0.0:
                from wav2seq.decoder.ctc_s2s_generator import CTCSeq2seqGenerator

                seq_gen_cls = CTCSeq2seqGenerator

            gen_cfg = GenerationConfig(beam=args.beam, lenpen=args.lenpen)
            if args.kenlm_model:
                overrides = dict()
                lms, _ = checkpoint_utils.load_model_ensemble(
                    [args.kenlm_model], arg_overrides=overrides, task=None
                )
                lms[0].eval()
                optimize_models(args, use_cuda, lms)
                extra_gen_cls_kwargs = {"lm_model": lms[0], "lm_weight": args.lm_weight}
                generator = task.build_generator(
                    [model],
                    gen_cfg,
                    extra_gen_cls_kwargs=extra_gen_cls_kwargs,
                    seq_gen_cls=seq_gen_cls,
                )
                print(f"lm model: {generator.lm_model}")
            else:
                generator = task.build_generator(
                    [model], gen_cfg, seq_gen_cls=seq_gen_cls
                )
            if args.ctc_weight > 0.0:
                generator.ctc_weight = args.ctc_weight
            return generator
        else:
            print(
                "only wav2letter decoders with (viterbi, kenlm, fairseqlm) options are supported at the moment"
            )

    # please do not touch this unless you test both generate.py and infer.py with audio_pretraining task
    generator = build_generator(args)

    if args.load_emissions:
        generator = ExistingEmissionsDecoder(
            generator, np.load(args.load_emissions, allow_pickle=True)
        )
        logger.info("loaded emissions from " + args.load_emissions)

    num_sentences = 0

    if args.results_path is not None and not os.path.exists(args.results_path):
        os.makedirs(args.results_path)

    max_source_pos = (
        utils.resolve_max_positions(
            task.max_positions(), *[model.max_positions() for model in models]
        ),
    )

    if max_source_pos is not None:
        max_source_pos = max_source_pos[0]
        if max_source_pos is not None:
            max_source_pos = max_source_pos[0] - 1

    if args.dump_emissions:
        emissions = {}
    if args.dump_features:
        features = {}
        models[0].bert.proj = None
    else:
        res_files = prepare_result_files(args)
    errs_t = 0
    lengths_t = 0
    bleu_stats = {
        "correct": [0, 0, 0, 0],
        "total": [0, 0, 0, 0],
        "sys_len": 0,
        "ref_len": 0,
    }
    with progress_bar.build_progress_bar(args, itr) as t:
        wps_meter = TimeMeter()
        full_timer = StopwatchMeter()
        full_timer.start()
        for sample in t:
            sample = utils.move_to_cuda(sample) if use_cuda else sample

            def apply_half(t):
                if t.dtype is torch.float32:
                    return t.half()
                return t

            if args.fp16:
                sample = utils.apply_to_sample(apply_half, sample)

            if "net_input" not in sample:
                continue

            prefix_tokens = None
            if args.prefix_size > 0:
                prefix_tokens = sample["target"][:, : args.prefix_size]

            gen_timer.start()
            if args.dump_emissions:
                with torch.no_grad():
                    encoder_out = models[0](**sample["net_input"])
                    emm = models[0].get_normalized_probs(encoder_out, log_probs=True)
                    emm = emm.transpose(0, 1).cpu().numpy()
                    for i, id in enumerate(sample["id"]):
                        emissions[id.item()] = emm[i]
                    continue
            elif args.dump_features:
                with torch.no_grad():
                    encoder_out = models[0](**sample["net_input"])
                    feat = encoder_out["encoder_out"].transpose(0, 1).cpu().numpy()
                    for i, id in enumerate(sample["id"]):
                        padding = (
                            encoder_out["encoder_padding_mask"][i].cpu().numpy()
                            if encoder_out["encoder_padding_mask"] is not None
                            else None
                        )
                        features[id.item()] = (feat[i], padding)
                    continue
            hypos = task.inference_step(generator, models, sample, prefix_tokens)
            num_generated_tokens = sum(len(h[0]["tokens"]) for h in hypos)
            gen_timer.stop(num_generated_tokens)

            for i, sample_id in enumerate(sample["id"].tolist()):
                speaker = None
                # id = task.dataset(args.gen_subset).ids[int(sample_id)]
                id = sample_id
                toks = (
                    sample["target"][i, :]
                    if "target_label" not in sample
                    else sample["target_label"][i, :]
                )
                target_tokens = utils.strip_pad(toks, tgt_dict.pad()).int().cpu()
                # Process top predictions
                errs, length, bleu = process_predictions(
                    args,
                    hypos[i],
                    None,
                    tgt_dict,
                    target_tokens,
                    res_files,
                    speaker,
                    id,
                )
                errs_t += errs
                lengths_t += length

                if bleu is not None:
                    for i in range(4):
                        bleu_stats["correct"][i] += bleu.counts[i]
                        bleu_stats["total"][i] += bleu.totals[i]
                    bleu_stats["ref_len"] += bleu.ref_len
                    bleu_stats["sys_len"] += bleu.sys_len

            wps_meter.update(num_generated_tokens)
            t.log({"wps": round(wps_meter.avg)})
            num_sentences += (
                sample["nsentences"] if "nsentences" in sample else sample["id"].numel()
            )

        full_timer.stop()

    bleu = sacrebleu.compute_bleu(**bleu_stats)
    wer = None
    if args.dump_emissions:
        emm_arr = []
        for i in range(len(emissions)):
            emm_arr.append(emissions[i])
        np.save(args.dump_emissions, np.array(emm_arr, dtype="object"))
        logger.info(f"saved {len(emissions)} emissions to {args.dump_emissions}")
    elif args.dump_features:
        feat_arr = []
        for i in range(len(features)):
            feat_arr.append(features[i])
        np.save(args.dump_features, np.array(feat_arr, dtype="object"))
        logger.info(f"saved {len(features)} emissions to {args.dump_features}")
    else:
        if lengths_t > 0:
            wer = errs_t * 100.0 / lengths_t
            logger.info(f"WER: {wer}")
        logger.info(f"BLEU: {bleu.score}")
        logger.info(f"{bleu}")
        logger.info(f"full time used: {full_timer.sum}")
        logger.info(f"time used: {gen_timer.sum}")

        logger.info(
            "| Processed {} sentences ({} tokens) in {:.2f} s ({:.2f}"
            " sentences/s, {:.2f} tokens/s)".format(
                num_sentences,
                gen_timer.n,
                gen_timer.sum,
                num_sentences / gen_timer.sum,
                1.0 / gen_timer.avg,
            )
        )
        logger.info("| Generate {} with beam={}".format(args.gen_subset, args.beam))
        # write the WER info into file in case it's called by infer_multiprocess
        if args.results_path is not None and not os.path.exists(args.results_path):
            with open(os.path.join(args.results_path, "wer"), "w") as fw_wer:
                fw_wer.write(f"{errs_t} {lengths_t}")
    return task, wer


def make_parser():
    parser = options.get_generation_parser()
    parser = add_asr_eval_argument(parser)
    return parser


def cli_main():
    parser = make_parser()
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == "__main__":
    cli_main()
