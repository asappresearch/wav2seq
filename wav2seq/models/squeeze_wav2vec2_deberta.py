# Copyright (c) ASAPP Inc.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field
from omegaconf import II

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.dataclass import ChoiceEnum
from fairseq.models import register_model
from fairseq.modules import (
    SamePad,
    LayerDropModuleList,
)


EXTRACTOR_MODE_CHOICES = ChoiceEnum(["default", "layer_norm"])
from einops.layers.torch import Rearrange

from .squeeze_wav2vec2 import SqueezeWav2Vec2Config, make_pad_mask
from .feat_mlp_wav2vec2 import FeatMLPWav2Vec2Model


try:
    from DeBERTa import deberta
except ImportError:
    print("Please install deberta")


@dataclass
class SqueezeWav2Vec2DebertaConfig(SqueezeWav2Vec2Config):
    # deberta
    norm_rel_ebd: str = field(
        default="layer_norm", metadata={"help": "use layer norm in relative embedding"}
    )
    max_position_embeddings: int = field(
        default=512, metadata={"help": "max size of positional embedding"}
    )
    max_relative_positions: int = field(
        default=-1, metadata={"help": "max size of rel positional embedding"}
    )
    position_biased_input: bool = field(
        default=False, metadata={"help": "max size of rel positional embedding"}
    )
    pos_att_type: str = field(
        default="p2c|c2p", metadata={"help": "max size of rel positional embedding"}
    )
    position_buckets: int = field(
        default=256, metadata={"help": "max size of rel positional embedding"}
    )
    initializer_range: float = field(
        default=0.02, metadata={"help": "initialization range of bert"}
    )
    relative_attention: bool = field(
        default=True, metadata={"help": "use relative positional embedding or not"}
    )
    share_att_key: bool = field(
        default=True, metadata={"help": "share attention key with c2p and p2c"}
    )
    # deberta_conv_kernel_size: int = field(
    #     default=3, metadata={"help": "conv size at the beginnning"}
    # )
    # deberta_conv_groups: int = field(
    #     default=1, metadata={"help": "group size at the beginnning"}
    # )
    # deberta_conv_act: str = field(
    #     default="gelu", metadata={"help": "activation of the first conv layer"}
    # )
    # new
    cross_layer_param_share: bool = field(
        default=False, metadata={"help": "tie transformer layers like ALBERT"}
    )


@register_model("squeeze_wav2vec2_deberta", dataclass=SqueezeWav2Vec2DebertaConfig)
class SqueezeWav2Vec2DebertaModel(FeatMLPWav2Vec2Model):
    def __init__(self, cfg: SqueezeWav2Vec2DebertaConfig):
        super().__init__(cfg)
        self.encoder = SqueezeDebertaEncoder(cfg)


class SqueezeDebertaEncoder(nn.Module):
    def __init__(self, cfg: SqueezeWav2Vec2DebertaConfig):
        super().__init__()
        self.cfg = cfg
        self.embedding_dim = cfg.encoder_embed_dim
        deberta_cfg = deberta.config.ModelConfig()

        deberta_cfg.num_hidden_layers = cfg.encoder_layers
        deberta_cfg.hidden_size = cfg.encoder_embed_dim
        deberta_cfg.intermediate_size = cfg.encoder_ffn_embed_dim
        deberta_cfg.num_attention_heads = cfg.encoder_attention_heads
        deberta_cfg.attention_head_size = 64
        deberta_cfg.hidden_act = str(cfg.activation_fn)
        deberta_cfg.hidden_dropout_prob = cfg.activation_dropout
        deberta_cfg.attention_probs_dropout_prob = cfg.attention_dropout

        # don't use conv in deberta
        deberta_cfg.conv_kernel_size = 0
        # deberta_cfg.conv_act = cfg.deberta_conv_act
        # deberta_cfg.conv_kernel_size = cfg.deberta_conv_kernel_size
        # deberta_cfg.conv_groups = cfg.deberta_conv_groups

        deberta_cfg.layer_norm_eps = 1e-7
        deberta_cfg.norm_rel_ebd = cfg.norm_rel_ebd
        deberta_cfg.max_position_embeddings = cfg.max_position_embeddings
        deberta_cfg.max_relative_positions = cfg.max_relative_positions
        deberta_cfg.position_biased_input = cfg.position_biased_input
        deberta_cfg.pos_att_type = cfg.pos_att_type
        deberta_cfg.position_buckets = cfg.position_buckets
        deberta_cfg.initializer_range = cfg.initializer_range
        deberta_cfg.relative_attention = cfg.relative_attention
        deberta_cfg.share_att_key = cfg.share_att_key

        self.encoder = deberta.bert.BertEncoder(deberta_cfg)
        if cfg.cross_layer_param_share:
            self.encoder.layer = LayerDropModuleList(
                cfg.encoder_layerdrop, [self.encoder.layer[0]] * len(self.encoder.layer)
            )
        else:
            self.encoder.layer = LayerDropModuleList(
                cfg.encoder_layerdrop, [l for l in self.encoder.layer]
            )

        # squeezing
        self.pos_conv = self.get_pos_conv(cfg.squeeze_factor)
        self.pool = nn.AvgPool1d(cfg.squeeze_factor, cfg.squeeze_factor)
        self.upsample = self.get_upsample(cfg.squeeze_factor)

    def get_pos_conv(self, squeeze_factor: int):
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
        return pos_conv

    def get_upsample(self, squeeze_factor: int):
        upsample = nn.Linear(self.embedding_dim, self.embedding_dim * squeeze_factor)
        nn.init.kaiming_normal_(upsample.weight, mode="fan_out", nonlinearity="relu")
        nn.init.zeros_(upsample.bias)
        upsample = nn.Sequential(
            upsample,
            nn.GELU(),
            Rearrange("b t (s c) -> b (t s) c", s=squeeze_factor, c=self.embedding_dim),
        )
        return upsample

    def forward(self, x, padding_mask=None, layer=None):

        if padding_mask is not None:
            x[padding_mask] = 0

        T = x.shape[1]

        # down sample
        x = x.transpose(1, 2)  # (B, T, C) to (B, C, T)
        x_conv = self.pos_conv(x)
        x_pool = self.pool(x)
        min_length = min(x_conv.size(-1), x_pool.size(-1))
        x = (x_pool[..., :min_length] + x_conv[..., :min_length]).transpose(
            1, 2
        )  # back to (B, T, C)

        if padding_mask is None:
            attention_mask = torch.ones(x.shape[:2], dtype=torch.long, device=x.device)
        else:
            input_lengths = (1 - padding_mask.long()).sum(-1)
            # apply conv formula to get real output_lengths
            output_lengths = input_lengths // self.cfg.squeeze_factor
            padding_mask = make_pad_mask(output_lengths).to(x.device)  # 1 at padding
            attention_mask = padding_mask.eq(0).long()

        layer_results = self.encoder(x, attention_mask=attention_mask)
        if isinstance(layer_results, dict):
            layer_results = layer_results["hidden_states"]
        if layer is None:
            x = layer_results[-1]
        else:
            x = layer_results[layer]

        if self.upsample is not None:
            x = self.upsample(x)

        if x.size(1) < T:
            x = F.pad(x, (0, 0, 0, T - x.size(1)))

        return x, layer_results
