# Copyright (c) ASAPP Inc.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field
from omegaconf import II

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.models import register_model
from fairseq.models.wav2vec.wav2vec2 import TransformerEncoder
from fairseq.modules import SamePad

from einops.layers.torch import Rearrange

from .feat_mlp_wav2vec2 import FeatMLPWav2Vec2Config, MLP, FeatMLPWav2Vec2Model
from ..modules.fp32_batch_norm import FP32BatchNorm1d


@torch.jit.script
def make_pad_mask(lengths: torch.Tensor) -> torch.Tensor:
    return torch.arange(0, lengths.max(), device=lengths.device).view(1, -1).expand(
        lengths.size(0), -1
    ) >= lengths.view(-1, 1)


@dataclass
class SqueezeWav2Vec2Config(FeatMLPWav2Vec2Config):
    squeeze_factor: int = field(
        default=2,
        metadata={
            "help": "downsample the sequece length by this factor in pos_conv and upsample after transformer"
        },
    )
    squeeze_method: str = field(
        default="default", metadata={"help": "method to squeeze the temporal dimension"}
    )


@register_model("squeeze_wav2vec2", dataclass=SqueezeWav2Vec2Config)
class SqueezeWav2Vec2Model(FeatMLPWav2Vec2Model):
    def __init__(self, cfg: SqueezeWav2Vec2Config):
        super().__init__(cfg)
        self.encoder = SqueezeTransformerEncoder(cfg)


class SqueezeTransformerEncoder(TransformerEncoder):
    def __init__(self, cfg: SqueezeWav2Vec2Config):
        super().__init__(cfg)
        self.cfg = cfg
        self.pos_conv = self.get_pos_conv(cfg.squeeze_factor)
        self.pool = self.get_pool(cfg.squeeze_factor)
        self.upsample = self.get_upsample(cfg.squeeze_factor)

    def get_pool(self, squeeze_factor: int):
        if squeeze_factor == 1:
            return nn.Identity()
        if self.cfg.squeeze_method in {"default", "default-v2"}:
            pool = nn.AvgPool1d(squeeze_factor, squeeze_factor)
        elif self.cfg.squeeze_method in {
            "multi-layer",
            "multi-layer-k4",
            "multi-layer-k4-bn",
        }:
            pool = nn.AvgPool1d(3, 2)
        else:
            raise ValueError(f"squeeze_method={self.cfg.squeeze_method}")
        return pool

    def get_pos_conv(self, squeeze_factor: int):
        if self.cfg.squeeze_method in {"default", "default-v2"}:
            pos_conv = nn.Conv1d(
                self.embedding_dim,
                self.embedding_dim,
                kernel_size=self.cfg.conv_pos,
                padding=self.cfg.conv_pos // 2,
                groups=self.cfg.conv_pos_groups,
                stride=squeeze_factor,
            )
            dropout = 0
            std = math.sqrt(
                (4 * (1.0 - dropout)) / (self.cfg.conv_pos * self.embedding_dim)
            )
            nn.init.normal_(pos_conv.weight, mean=0, std=std)
            nn.init.constant_(pos_conv.bias, 0)
            pos_conv = nn.utils.weight_norm(pos_conv, name="weight", dim=2)
            pos_conv = nn.Sequential(pos_conv, SamePad(self.cfg.conv_pos), nn.GELU())
        elif self.cfg.squeeze_method in {"multi-layer", "multi-layer-k4"}:
            layers = []
            for i in range(int(np.log2(squeeze_factor))):
                conv = nn.Conv1d(
                    self.embedding_dim,
                    self.embedding_dim,
                    kernel_size=self.cfg.conv_pos,
                    padding=self.cfg.conv_pos // 2,
                    groups=self.cfg.conv_pos_groups,
                    stride=2,
                )
                dropout = 0
                std = math.sqrt(
                    (4 * (1.0 - dropout)) / (self.cfg.conv_pos * self.embedding_dim)
                )
                nn.init.normal_(conv.weight, mean=0, std=std)
                nn.init.constant_(conv.bias, 0)
                conv = nn.utils.weight_norm(conv, name="weight", dim=2)
                layers += [nn.Sequential(conv, nn.GELU())]
            pos_conv = nn.ModuleList(layers)
        elif self.cfg.squeeze_method in {"multi-layer-k4-bn"}:
            layers = []
            for i in range(int(np.log2(squeeze_factor))):
                conv = nn.Conv1d(
                    self.embedding_dim,
                    self.embedding_dim,
                    kernel_size=self.cfg.conv_pos,
                    padding=self.cfg.conv_pos // 2,
                    groups=self.cfg.conv_pos_groups,
                    stride=2,
                )
                dropout = 0
                std = math.sqrt(
                    (4 * (1.0 - dropout)) / (self.cfg.conv_pos * self.embedding_dim)
                )
                nn.init.normal_(conv.weight, mean=0, std=std)
                nn.init.constant_(conv.bias, 0)
                conv = nn.utils.weight_norm(conv, name="weight", dim=2)
                layers += [
                    nn.Sequential(conv, FP32BatchNorm1d(self.embedding_dim), nn.GELU())
                ]
            pos_conv = nn.ModuleList(layers)
        else:
            raise ValueError(f"squeeze_method={self.cfg.squeeze_method}")
        return pos_conv

    def get_upsample(self, squeeze_factor: int):
        if self.cfg.squeeze_method == "default":
            layers = [
                nn.Linear(self.embedding_dim, self.embedding_dim * squeeze_factor),
                nn.GELU(),
                Rearrange(
                    "b t (s c) -> b (t s) c", s=squeeze_factor, c=self.embedding_dim
                ),
            ]
            upsample = nn.Sequential(*layers)
        elif self.cfg.squeeze_method == "default-v2":
            layers = []
            for _ in range(int(np.log2(squeeze_factor))):
                layers += [
                    nn.Linear(self.embedding_dim, self.embedding_dim * 2),
                    nn.GELU(),
                    Rearrange("b t (s c) -> b (t s) c", s=2, c=self.embedding_dim),
                ]
            upsample = nn.Sequential(*layers)
        elif self.cfg.squeeze_method == "multi-layer":
            upsample = [Rearrange("b t c -> b c t")]
            for i in range(int(np.log2(squeeze_factor))):
                upsample += [
                    nn.ConvTranspose1d(
                        self.embedding_dim, self.embedding_dim, 2, 2, 0, bias=False
                    ),
                    nn.GELU(),
                ]
            upsample.append(Rearrange("b c t -> b t c"))
            upsample = nn.Sequential(*upsample)
        elif self.cfg.squeeze_method == "multi-layer-k4":
            upsample = [Rearrange("b t c -> b c t")]
            for i in range(int(np.log2(squeeze_factor))):
                upsample += [
                    nn.ConvTranspose1d(
                        self.embedding_dim, self.embedding_dim, 4, 2, 1, bias=False
                    ),
                    nn.GELU(),
                ]
            upsample.append(Rearrange("b c t -> b t c"))
            upsample = nn.Sequential(*upsample)
        elif self.cfg.squeeze_method == "multi-layer-k4-bn":
            upsample = [Rearrange("b t c -> b c t")]
            for i in range(int(np.log2(squeeze_factor))):
                upsample += [
                    nn.ConvTranspose1d(
                        self.embedding_dim, self.embedding_dim, 4, 2, 1, bias=False
                    ),
                    FP32BatchNorm1d(self.embedding_dim),
                    nn.GELU(),
                ]
            upsample.append(Rearrange("b c t -> b t c"))
            upsample = nn.Sequential(*upsample)
        else:
            raise ValueError(f"squeeze_method={self.cfg.squeeze_method}")
        for m in upsample.modules():
            if isinstance(m, (nn.ConvTranspose1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        return upsample

    def forward(self, x, padding_mask=None, layer=None):
        x = self.extract_features(x, padding_mask, layer=layer)

        if self.layer_norm_first and self.upsample is None:
            x = self.layer_norm(x)

        return x

    def extract_features(self, x, padding_mask=None, layer=None):

        if padding_mask is not None:
            x[padding_mask] = 0

        T = x.shape[1]

        x = x.transpose(1, 2)  # B, T, C to B, C, T

        if isinstance(self.pos_conv, nn.Sequential):
            x_conv = self.pos_conv(x)
            x_pool = self.pool(x)
            min_length = min(x_conv.size(-1), x_pool.size(-1))
            x = x_pool[..., :min_length] + x_conv[..., :min_length]
        elif isinstance(self.pos_conv, nn.ModuleList):
            for conv in self.pos_conv:
                x_conv = conv(x)
                x_pool = self.pool(x)
                min_length = min(x_conv.size(-1), x_pool.size(-1))
                x = x_pool[..., :min_length] + x_conv[..., :min_length]
        else:
            raise NotImplementedError

        x = x.transpose(1, 2)

        # adjust the padding_mask
        if padding_mask is not None:
            input_lengths = (1 - padding_mask.long()).sum(-1)
            # apply conv formula to get real output_lengths
            output_lengths = input_lengths // self.cfg.squeeze_factor
            output_lengths += x.size(1) - output_lengths.max().item()
            padding_mask = make_pad_mask(output_lengths).to(x.device)  # 1 at padding

        if not self.layer_norm_first:
            x = self.layer_norm(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        layer_results = []
        for i, layer in enumerate(self.layers):
            dropout_probability = np.random.random()
            if not self.training or (dropout_probability > self.layerdrop):
                x, z = layer(x, self_attn_padding_mask=padding_mask, need_weights=False)
                layer_results.append(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.upsample is not None:
            if self.layer_norm_first:
                x = self.layer_norm(x)
            x = self.upsample(x)

        if x.size(1) < T:
            x = F.pad(x, (0, 0, 0, T - x.size(1)))

        return x, layer_results
