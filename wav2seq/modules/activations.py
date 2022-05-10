# Copyright (c) ASAPP Inc.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.modules.gelu import gelu, gelu_accurate


@torch.jit.script
def swish(x):
    return x * torch.sigmoid(x)


@torch.jit.script
def mish(x):
    return x * torch.tanh(F.softplus(x))


class Swish(nn.Module):
    def forward(self, x):
        return swish(x)


class Mish(nn.Module):
    def forward(self, x):
        return mish(x)


act_func_dict = {
    "sigmoid": torch.sigmoid,
    "tanh": torch.tanh,
    "elu": F.elu,
    "relu": F.relu,
    "gelu": gelu,
    "gelu_accurate": gelu_accurate,
    "glu": F.glu,
    "swish": F.silu,
    "mish": mish,
    "none": lambda x: x,
    "identity": lambda x: x,
    None: lambda x: x,
}

act_module_dict = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "mish": Mish,
    "swish": nn.SiLU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
    "hardswish": nn.Hardswish,
}
