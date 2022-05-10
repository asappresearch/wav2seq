# Copyright (c) ASAPP Inc.
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, List, Dict
from torch import Tensor

import torch
import torch.nn as nn
import torch.nn.functional as F


class CTCArgMaxDecoder(object):
    def __init__(self, args, tgt_dict):
        self.tgt_dict = tgt_dict
        self.vocab_size = len(tgt_dict)

        from fairseq import search
        from fairseq.data.dictionary import Dictionary

        assert isinstance(tgt_dict, Dictionary)
        self.search = search.BeamSearch(tgt_dict)
        self.normalize_scores = True

        self.blank = (
            tgt_dict.index("<ctc_blank>")
            if "<ctc_blank>" in tgt_dict.indices
            else tgt_dict.bos()
        )

    def generate(self, models, sample, **unused):
        """Generate a batch of inferences."""
        # model.forward normally channels prev_output_tokens into the decoder
        # separately, but SequenceGenerator directly calls model.encoder
        encoder_input = {
            k: v for k, v in sample["net_input"].items() if k != "prev_output_tokens"
        }
        emissions = self.get_emissions(models, encoder_input)
        return self.decode(emissions)

    def get_emissions(self, models, encoder_input):
        """Run encoder and normalize emissions"""
        encoder_out = models[0](**encoder_input)
        emissions = encoder_out["encoder_out"].transpose(0, 1).contiguous()
        if encoder_out["padding_mask"] is not None:
            emissions[encoder_out["padding_mask"]] = 0.0
            emissions[encoder_out["padding_mask"]][:, self.blank] = 1.0
        return emissions

    def get_tokens(self, idxs):
        """Normalize tokens by handling CTC blank, ASG replabels, etc."""
        idxs = (g[0] for g in it.groupby(idxs))
        idxs = filter(lambda x: x != self.blank, idxs)
        return torch.LongTensor(list(idxs))

    def decode(self, emissions):
        B, T, N = emissions.size()

        # set length bounds
        toks = emissions.argmax(dim=-1)
        hypos = []
        for i in range(B):
            hypos.append(
                [
                    {
                        "tokens": toks[i].unique_consecutive().cpu(),
                        "score": emissions[i].sum().item(),
                    }
                ]
            )
        return hypos


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


@torch.jit.script
def forward_ctc_prefix(
    y: torch.Tensor, state: torch.Tensor, logp: torch.Tensor, blank: int, eos: int
):
    """
    Partial CTC algorithm. PyTorch version of espnet.nets.ctc_prefix_score.CTCPrefixScore but 

    which is based on Algorithm 2 in WATANABE et al.
    "HYBRID CTC/ATTENTION ARCHITECTURE FOR END-TO-END SPEECH RECOGNITION,"
    but extended to efficiently compute the probablities of multiple labels
    simultaneously

    state[:, 0] are the non-blank log_prob
    state[:, 1] are the blank log_prob
    """
    state_prev = state
    output_length = y.size(0) - 1
    start = max(output_length, 1)
    T, V = logp.shape
    device = logp.device

    # line 6
    state = torch.full((T, 2, V), -np.inf, device=device)
    if output_length == 0:
        state[0, 0] = logp[0]  # non-blank

    # line 10
    state_sum = state_prev.logsumexp(dim=1)

    # line 7-8
    last = y[:, -1]
    log_phi = state_sum.view(-1, 1).repeat(1, V)
    log_phi[:, last] = state_prev[:, 1]

    # ling 13
    log_psi = log_phi.clone()
    log_psi[start:] = log_phi[start - 1 : -1] + logp[start:]

    log_psi[start - 1, :] = state[start - 1, 0]  # line 8
    log_psi = log_psi[start - 1 :].logsumexp(dim=0)

    # ignore generating blank
    log_psi[blank] = -np.inf

    # line 3-4
    log_psi[eos] = state_sum[-1]

    # line 11-12
    for t in range(start, T):
        state[t, 0] = (
            torch.logsumexp(
                torch.stack([state[t - 1, 0], log_phi[t - 1]], dim=0), dim=0
            )
            + logp[t]
        )
        state[t, 1] = (
            torch.logsumexp(
                torch.stack([state[t - 1, 0], state[t - 1, 1]], dim=0), dim=0
            )
            + logp[t, blank]
        )

    return log_psi, state.permute(2, 0, 1)


@torch.jit.script
def batch_forward_ctc_prefix(
    y: torch.Tensor, state: torch.Tensor, logp: torch.Tensor, blank: int, eos: int
):
    # y: B x T_out
    # r: B x T x 2
    # logp: B x T x V
    state_prev = state
    output_length = y.size(1) - 1
    start = max(output_length, 1)
    B, T, V = logp.shape
    device = logp.device

    # line 6
    state = torch.full((B, T, 2, V), -np.inf, device=device)
    if output_length == 0:
        state[:, 0, 0] = logp[:, 0]  # non-blank

    # line 10
    state_sum = state_prev.logsumexp(dim=2)

    # line 7-8
    last = y[:, -1]
    log_phi = state_sum.view(B, -1, 1).repeat(1, 1, V)
    log_phi.scatter_(2, last.view(B, 1, 1).expand(B, T, 1), state_prev[:, :, 1:])

    # ling 13
    log_psi = log_phi.clone()
    log_psi[:, start:] = log_phi[:, start - 1 : -1] + logp[:, start:]

    log_psi[:, start - 1, :] = state[:, start - 1, 0]  # line 8
    log_psi = log_psi[:, start - 1 :].logsumexp(dim=1)

    # ignore generating blank
    log_psi[:, blank] = -np.inf

    # line 3-4
    log_psi[:, eos] = state_sum[:, -1]

    # line 11-12
    for t in range(start, T):
        state[:, t, 0] = (
            torch.logsumexp(
                torch.stack([state[:, t - 1, 0], log_phi[:, t - 1]], dim=0), dim=0
            )
            + logp[:, t]
        )
        state[:, t, 1] = torch.logsumexp(state[:, t - 1], dim=1) + logp[
            :, t, blank
        ].view(-1, 1)

    return log_psi, state.permute(0, 3, 1, 2)


@torch.jit.script
def batch_forward_ctc_prefix_low_memory(
    y: torch.Tensor, state: torch.Tensor, logp: torch.Tensor, blank: int, eos: int
):
    """
    This implementation uses less memory, but is slower
    """
    # y: B x T_out
    # r: B x T x 2
    # logp: B x T x V
    state_prev = state
    output_length = y.size(1) - 1
    start = max(output_length, 1)
    B, T, V = logp.shape
    device = logp.device

    # line 6
    state = torch.full((B, T, 2, V), -np.inf, device=device)
    if output_length == 0:
        state[:, 0, 0] = logp[:, 0]  # non-blank

    # line 10
    state_sum = state_prev.logsumexp(dim=2)

    # line 7-8
    last = y[:, -1]
    log_phi = state_sum.view(B, -1, 1).repeat(1, 1, V)
    log_phi.scatter_(2, last.view(B, 1, 1).expand(B, T, 1), state_prev[:, :, 1:])

    # line 11-12
    log_psi = state[:, start - 1, 0]
    buffer = torch.zeros((2, B, V), device=device, dtype=torch.float)
    for t in range(start, T):
        buffer[0] = state[:, t - 1, 0]
        buffer[1] = log_phi[:, t - 1]
        state[:, t, 0].copy_(torch.logsumexp(buffer, dim=0).add_(logp[:, t]))
        state[:, t, 1].copy_(
            torch.logsumexp(state[:, t - 1], dim=1).add_(
                logp[:, t, blank].view(-1, 1).expand(-1, V)
            )
        )
        log_psi = torch.logsumexp(
            torch.stack([log_psi, log_phi[:, t - 1].add_(logp[:, t])], dim=0), dim=0
        )

    return log_psi, state.permute(0, 3, 1, 2)


class BatchCTCPrefixScore(object):
    """Compute CTC label sequence scores

    which is based on Algorithm 2 in WATANABE et al.
    "HYBRID CTC/ATTENTION ARCHITECTURE FOR END-TO-END SPEECH RECOGNITION,"
    but extended to efficiently compute the probablities of multiple labels
    simultaneously
    """

    def __init__(self, blank, eos):
        self.blank = blank
        self.eos = eos

    def initial_state(self, logp):
        """Obtain an initial CTC state

        :return: CTC state
        """
        B, T = logp.shape[:2]
        state = torch.full((B, T, 2), -np.inf, device=logp.device)
        state[:, :, 1] = logp[:, :, self.blank].cumsum(dim=1)
        score = logp.new_zeros(B)

        return score, state

    def __call__(self, y, state, logp):
        return batch_forward_ctc_prefix(y, state, logp, self.blank, self.eos)

    def reorder_state(self, score, state, logp, new_order, last_token=None):
        score = score.index_select(0, new_order)
        state = state.index_select(0, new_order)
        logp = logp.index_select(0, new_order)
        if last_token is not None:
            score = score.gather(1, last_token.view(-1, 1)).view(
                -1
            )  # the score corresponds to the last token
            state = state.gather(
                1, last_token.view(-1, 1, 1, 1).expand(-1, 1, *state.shape[2:])
            ).squeeze(1)
        assert score.dim() == 1
        assert state.dim() == 3
        return score, state, logp
