import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq.modules import (
    Fp32GroupNorm,
    Fp32LayerNorm,
)

from .fp32_batch_norm import FP32BatchNorm1d
from .wav_extracter import get_act
from einops.layers.torch import Rearrange
from einops import rearrange


# for (B, T, C) inputs
class MLPv2(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 4096,
        num_layers=2,
        norm_type: str = "bn",
        act_type: str = "relu",
    ):
        super().__init__()
        net = []
        for i in range(num_layers - 1):
            net += [
                nn.Linear(input_dim, hidden_dim, bias=norm_type == "none"),
                self.get_norm(norm_type, hidden_dim),
                get_act(act_type),
            ]
            input_dim = hidden_dim
        net += [nn.Linear(input_dim, output_dim)]

        self.net = nn.Sequential(*net)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

    def get_norm(self, norm_type, dim):
        if norm_type == "in":
            return nn.Sequential(
                Rearrange("b t c -> b c t"),
                Fp32GroupNorm(dim, dim),
                Rearrange("b c t -> b t c"),
            )
        elif norm_type.startswith("gn"):
            ngroups = int(norm_type[2:])
            return nn.Sequential(
                Rearrange("b t c -> b c t"),
                Fp32GroupNorm(ngroups, dim),
                Rearrange("b c t -> b t c"),
            )
        elif norm_type == "bn":
            return nn.Sequential(
                Rearrange("b t c -> b c t"),
                FP32BatchNorm1d(dim),
                Rearrange("b c t -> b t c"),
            )
        elif norm_type == "ln":
            return Fp32LayerNorm(dim, elementwise_affine=True)
        elif norm_type in {"none", "-", ""}:
            return nn.Identity()
        else:
            raise ValueError(f"norm_type={norm_type}")


# for (B, T, C) inputs but operations in (B, C, T)
class MLPv3(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 4096,
        num_layers=2,
        norm_type: str = "bn",
        act_type: str = "relu",
    ):
        super().__init__()
        net = []
        for i in range(num_layers - 1):
            net += [
                nn.Conv1d(input_dim, hidden_dim, 1, 1, 0, bias=norm_type == "none"),
                self.get_norm(norm_type, hidden_dim),
                get_act(act_type),
            ]
            input_dim = hidden_dim
        net += [
            nn.Conv1d(input_dim, output_dim, 1, 1, 0),
        ]

        self.net = nn.Sequential(*net)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        is2d = x.ndim == 2
        if is2d:
            x = x.unsqueeze(-1)
        else:
            x = x.transpose(1, 2)
        x = self.net(x)
        if is2d:
            x = x.squeeze(-1)
        else:
            x = x.transpose(1, 2).contiguous()
        return x

    def get_norm(self, norm_type, dim):
        if norm_type == "in":
            return Fp32GroupNorm(dim, dim)
        elif norm_type.startswith("gn"):
            ngroups = int(norm_type[2:])
            return Fp32GroupNorm(ngroups, dim)
        elif norm_type == "bn":
            return FP32BatchNorm1d(dim)
        elif norm_type == "ln":
            return nn.Sequential(
                Rearrange("b c t -> b t c"),
                Fp32LayerNorm(dim, elementwise_affine=True),
                Rearrange("b t c -> b c t"),
            )
        elif norm_type in {"none", "-", ""}:
            return nn.Identity()
        else:
            raise ValueError(f"norm_type={norm_type}")
