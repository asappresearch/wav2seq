import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Callable
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from fairseq.modules import (
    Fp32GroupNorm,
    Fp32LayerNorm,
    LayerNorm,
    TransposeLast,
)

from .activations import act_module_dict
from .fp32_batch_norm import FP32BatchNorm1d

logger = logging.getLogger(__name__)


def get_act(act_type):
    if act_type == "relu":
        return act_module_dict[act_type](inplace=True)
    else:
        return act_module_dict[act_type]()


class ConvFeatureExtractionModelV2(nn.Module):
    def __init__(
        self,
        conv_layers: List[Tuple[int, int, int]],
        dropout: float = 0.0,
        mode: str = "default",
        conv_bias: bool = False,
    ):
        super().__init__()

        assert mode in {"default", "layer_norm", "batch_norm"}

        def block(
            n_in,
            n_out,
            k,
            stride,
            layer_idx,
            mode="default",
            # is_layer_norm=False,
            # is_group_norm=False,
            conv_bias=False,
        ):
            def make_conv():
                conv = nn.Conv1d(
                    n_in,
                    n_out,
                    k,
                    stride=stride,
                    bias=conv_bias and mode != "batch_norm",
                )
                nn.init.kaiming_normal_(conv.weight)
                return conv

            # assert (
            #     is_layer_norm and is_group_norm
            # ) == False, "layer norm and group norm are exclusive"

            if mode == "layer_norm":
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    nn.Sequential(
                        TransposeLast(),
                        Fp32LayerNorm(dim, elementwise_affine=True),
                        TransposeLast(),
                    ),
                    nn.GELU(),
                )
            elif mode == "default" and layer_idx == 0:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    Fp32GroupNorm(dim, dim, affine=True),
                    nn.GELU(),
                )
            elif mode == "batch_norm":
                return nn.Sequential(
                    make_conv(), FP32BatchNorm1d(dim, affine=True), nn.GELU(),
                )
            else:
                return nn.Sequential(make_conv(), nn.Dropout(p=dropout), nn.GELU())

        in_d = 1
        self.conv_cfg_list = conv_layers
        self.conv_layers = nn.ModuleList()
        self.kernel_size = 1
        self.stride = 1
        for i, cl in enumerate(conv_layers):
            assert len(cl) == 3, "invalid conv definition: " + str(cl)
            (dim, k, stride) = cl

            self.conv_layers.append(
                block(
                    in_d,
                    dim,
                    k,
                    stride,
                    layer_idx=i,
                    mode=mode,
                    # is_layer_norm=mode == "layer_norm",
                    # is_group_norm=mode == "default" and i == 0,
                    conv_bias=conv_bias,
                )
            )
            in_d = dim
            self.kernel_size += (k - 1) * self.stride
            self.stride *= stride
            self.output_dim = dim

    def forward(self, x):

        # BxT -> BxCxT
        x = x.unsqueeze(1)

        for conv in self.conv_layers:
            x = conv(x)

        return x

    def print_complexity(self, x):
        def conv1d_cx(m, x, cx=None):
            if cx is None:
                cx = {"flops": 0, "params": 0, "acts": 0}
            w_in, w_out = m.weight.shape[1], m.weight.shape[0]
            k, stride, padding = m.weight.shape[2], m.stride[0], m.padding[0]
            bias = m.bias is not None

            t = x.size(2)
            new_t = (t - k + 2 * padding) // stride + 1

            cx["flops"] += k * w_in * w_out * new_t + (w_out if bias else 0)
            cx["params"] += sum([p.numel() for p in m.parameters()])
            cx["acts"] += w_out * new_t
            return cx

        def complexity(block, x, cx=None):
            if cx is None:
                cx = {"flops": 0, "params": 0, "acts": 0}
            x = block(x)
            cx = conv1d_cx(block[0], x, cx)
            if isinstance(block[-1], torch.nn.GELU):
                cx["acts"] += x.size(1) * x.size(2)
                cx["flops"] += x.size(1) * x.size(2) * 8
            else:
                raise NotImplementedError
            cx["t"] = x.size(2)
            return x, cx

        for i, block in enumerate(self.conv_layers):
            x, cx = complexity(block, x)
            flops, acts = cx["flops"], cx["acts"]
            print(f"{i} flops: {flops / 1e6:.2f} M acts: {acts / 1e3:.2f} K")

    def make_generation_fast_(self, **kwargs):
        for i in range(len(self.conv_layers)):
            layer = self.conv_layers[i]
            if isinstance(layer[1], nn.BatchNorm1d):
                conv, bn = layer[0], layer[1]
                merge_bn_to_conv(conv, bn)
                self.conv_layers[i] = nn.Sequential(conv, layer[-1])

    def get_output_lengths(self, input_lengths: torch.LongTensor):
        """
        Computes the output length of the convolutional layers
        """

        def _conv_out_length(input_length, kernel_size, stride):
            return torch.floor((input_length - kernel_size) / stride + 1)

        for i in range(len(self.conv_cfg_list)):
            input_lengths = _conv_out_length(
                input_lengths, self.conv_cfg_list[i][1], self.conv_cfg_list[i][2]
            )

        return input_lengths.to(torch.long)


@torch.no_grad()
def merge_bn_to_conv(conv, bn):
    scale = bn.weight / (bn.running_var + bn.eps) ** 0.5
    shift = bn.bias - bn.running_mean * scale
    conv.weight.data.mul_(scale.view(-1, 1, 1))
    conv.bias = nn.Parameter(shift)
