# Copyright (c) ASAPP Inc.
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import List, Tuple
from omegaconf import II

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.data.data_utils import compute_mask_indices
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.models import BaseFairseqModel, register_model
from fairseq.modules import (
    Fp32GroupNorm,
    GradMultiply,
    GumbelVectorQuantizer,
    LayerNorm,
)
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.utils import buffered_arange


EXTRACTOR_MODE_CHOICES = ChoiceEnum(["default", "layer_norm"])
MASKING_DISTRIBUTION_CHOICES = ChoiceEnum(["static", "uniform", "normal", "poisson"])

from fairseq.models.wav2vec.wav2vec2 import TransformerEncoder
from ..modules.fp32_batch_norm import FP32BatchNorm1d
from ..modules.mlp import MLPv2, MLPv3


@dataclass
class FeatMLPWav2Vec2Config(FairseqDataclass):
    extractor_mode: str = field(
        default="default",
        metadata={
            "help": "mode for feature extractor. default has a single group norm with d "
            "groups in the first conv block, whereas layer_norm has layer norms in "
            "every block (meant to use with normalize=True)"
        },
    )
    encoder_layers: int = field(
        default=12, metadata={"help": "num encoder layers in the transformer"}
    )
    encoder_embed_dim: int = field(
        default=768, metadata={"help": "encoder embedding dimension"}
    )
    encoder_ffn_embed_dim: int = field(
        default=3072, metadata={"help": "encoder embedding dimension for FFN"}
    )
    encoder_attention_heads: int = field(
        default=12, metadata={"help": "num encoder attention heads"}
    )
    activation_fn: ChoiceEnum(utils.get_available_activation_fns()) = field(
        default="gelu", metadata={"help": "activation function to use"}
    )

    # dropouts
    dropout: float = field(
        default=0.1, metadata={"help": "dropout probability for the transformer"}
    )
    attention_dropout: float = field(
        default=0.1, metadata={"help": "dropout probability for attention weights"}
    )
    activation_dropout: float = field(
        default=0.0, metadata={"help": "dropout probability after activation in FFN"}
    )
    encoder_layerdrop: float = field(
        default=0.0, metadata={"help": "probability of dropping a tarnsformer layer"}
    )
    dropout_input: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the input (after feat extr)"},
    )
    dropout_features: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the features (after feat extr)"},
    )

    final_dim: int = field(
        default=0,
        metadata={
            "help": "project final representations and targets to this many dimensions."
            "set to encoder_embed_dim is <= 0"
        },
    )
    layer_norm_first: bool = field(
        default=False, metadata={"help": "apply layernorm first in the transformer"}
    )
    conv_feature_layers: str = field(
        default="[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] + [(512,2,2)]",
        metadata={
            "help": "string describing convolutional feature extraction layers in form of a python list that contains "
            "[(dim, kernel_size, stride), ...]"
        },
    )
    conv_bias: bool = field(
        default=False, metadata={"help": "include bias in conv encoder"}
    )
    logit_temp: float = field(
        default=0.1, metadata={"help": "temperature to divide logits by"}
    )
    quantize_targets: bool = field(
        default=False, metadata={"help": "use quantized targets"}
    )
    quantize_input: bool = field(
        default=False, metadata={"help": "use quantized inputs"}
    )
    same_quantizer: bool = field(
        default=False, metadata={"help": "use same quantizer for inputs and targets"}
    )
    target_glu: bool = field(
        default=False, metadata={"help": "adds projection + glu to targets"}
    )
    feature_grad_mult: float = field(
        default=1.0, metadata={"help": "multiply feature extractor var grads by this"}
    )
    latent_vars: int = field(
        default=320,
        metadata={"help": "number of latent variables V in each group of the codebook"},
    )
    latent_groups: int = field(
        default=2,
        metadata={"help": "number of groups G of latent variables in the codebook"},
    )
    latent_dim: int = field(
        default=0,
        metadata={
            "help": "if > 0, uses this dimensionality for latent variables. "
            "otherwise uses final_dim / latent_groups"
        },
    )

    # masking
    mask_length: int = field(default=10, metadata={"help": "mask length"})
    mask_prob: float = field(
        default=0.65, metadata={"help": "probability of replacing a token with mask"}
    )
    mask_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static", metadata={"help": "how to choose mask length"}
    )
    mask_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument (used for more complex distributions), "
            "see help in compute_mask_indices"
        },
    )
    no_mask_overlap: bool = field(
        default=False, metadata={"help": "whether to allow masks to overlap"}
    )
    mask_min_space: int = field(
        default=1,
        metadata={"help": "min space between spans (if no overlap is enabled)"},
    )

    # channel masking
    mask_channel_length: int = field(
        default=10, metadata={"help": "length of the mask for features (channels)"}
    )
    mask_channel_prob: float = field(
        default=0.0, metadata={"help": "probability of replacing a feature with 0"}
    )
    mask_channel_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static",
        metadata={"help": "how to choose mask length for channel masking"},
    )
    mask_channel_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument (used for more complex distributions), "
            "see help in compute_mask_indicesh"
        },
    )
    no_mask_channel_overlap: bool = field(
        default=False, metadata={"help": "whether to allow channel masks to overlap"}
    )
    mask_channel_min_space: int = field(
        default=1,
        metadata={"help": "min space between spans (if no overlap is enabled)"},
    )

    # negative selection
    num_negatives: int = field(
        default=100,
        metadata={"help": "number of negative examples from the same sample"},
    )
    negatives_from_everywhere: bool = field(
        default=False,
        metadata={"help": "sample negatives from everywhere, not just masked states"},
    )
    cross_sample_negatives: int = field(
        default=0, metadata={"help": "number of negative examples from the any sample"}
    )
    codebook_negatives: int = field(
        default=0, metadata={"help": "number of negative examples codebook"}
    )

    # positional embeddings
    conv_pos: int = field(
        default=128,
        metadata={"help": "number of filters for convolutional positional embeddings"},
    )
    conv_pos_groups: int = field(
        default=16,
        metadata={"help": "number of groups for convolutional positional embedding"},
    )

    latent_temp: Tuple[float, float, float] = field(
        default=(2, 0.5, 0.999995),
        metadata={
            "help": "temperature for latent variable sampling. "
            "can be tuple of 3 values (start, end, decay)"
        },
    )

    # mlp
    use_mlp: bool = field(
        default=False, metadata={"help": "use MLP head"},
    )
    mlp_version: str = field(
        default="v1", metadata={"help": "version of mlp"},
    )
    mlp_num_layers: int = field(
        default=2, metadata={"help": "number of layers in mlp"},
    )
    mlp_hidden_size: int = field(
        default=4096, metadata={"help": "hidden size in mlp"},
    )
    mlp_act_type: str = field(
        default="relu", metadata={"help": "activation type in mlp"},
    )
    proj_mlp_norm_type: str = field(
        default="in", metadata={"help": "normalization method in the MLP"},
    )
    share_final_proj: bool = field(
        default=False, metadata={"help": "share the last linear layer of projection"},
    )
    final_proj_momentum: float = field(
        default=-1.0,
        metadata={"help": "momentum used when using a momentum projector"},
    )

    # feature
    spec_feature_only: bool = field(
        default=False, metadata={"help": "only use spectrogram features (fbank / MFCC)"}
    )
    fbank_dim: int = field(default=II("task.fbank_dim"),)
    mfcc_dim: int = field(default=II("task.mfcc_dim"),)
    conv_hidden_sizes: str = field(
        default="[32, 32]", metadata={"help": "hidden size of conv2d"}
    )
    conv_kernel_sizes: str = field(
        default="[3, 3]", metadata={"help": "kernel size of conv2d"}
    )
    conv_strides: str = field(
        default="[2, 2]", metadata={"help": "stride size of conv2d"}
    )
    conv_exp_ratios: str = field(
        default="[1., 4.]", metadata={"help": "exp ratio size of mbconv2d"}
    )
    conv_use_pad: bool = field(
        default=False, metadata={"help": "use padding in conv2d"}
    )
    encoder_input_embed_dim: int = field(
        default=-1,
        metadata={
            "help": "dim of the inputs of the contextual network (the same as encoder_embed_dim if <= 0)"
        },
    )
    conv_flatten_dim: int = field(
        default=-1,
        metadata={
            "help": "dim of the flattened spec features (the same as encoder_input_embed_dim if <= 0)"
        },
    )
    subsample_type: str = field(
        default="conv2dv4", metadata={"help": "subsample module type"}
    )
    conv_norm_type: str = field(
        default="none", metadata={"help": "subsample normalization type"}
    )
    conv_act_type: str = field(
        default="relu", metadata={"help": "subsample activation type"}
    )
    conv_kaiming_init: bool = field(
        default=False,
        metadata={"help": "use kaiming_normal init for subsampling layers"},
    )
    feature_extractor_type: str = field(
        default="default", metadata={"help": "name of the wav feature extractor"}
    )
    chunk_mode: str = field(
        default="chunk", metadata={"help": "mode of chunkwise feature extractor"}
    )


class MLP(nn.Module):
    def __init__(
        self,
        dim: int,
        projection_size: int,
        hidden_size: int = 4096,
        norm_type: str = "in",
    ):
        super().__init__()
        self.fc1 = nn.Linear(
            dim, hidden_size, bias=norm_type == "none"
        )  # bias term will be removed by norm
        if norm_type == "in":
            self.norm = Fp32GroupNorm(hidden_size, hidden_size, affine=True)
        elif norm_type == "bn":
            self.norm = FP32BatchNorm1d(hidden_size, affine=True)
        elif norm_type == "none":
            self.norm = lambda x: x
        else:
            raise NotImplementedError
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_size, projection_size)

        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = x.transpose(1, 2)  # B x T x C to B x C x T
        x = self.act(self.norm(x))
        x = x.transpose(1, 2)
        x = self.fc2(x)
        return x


@register_model("feat_mlp_wav2vec2", dataclass=FeatMLPWav2Vec2Config)
class FeatMLPWav2Vec2Model(BaseFairseqModel):
    def __init__(self, cfg: FeatMLPWav2Vec2Config):
        super().__init__()
        self.cfg = cfg
        if cfg.encoder_input_embed_dim <= 0:
            cfg.encoder_input_embed_dim = cfg.encoder_embed_dim
        if cfg.conv_flatten_dim <= 0:
            cfg.conv_flatten_dim = cfg.encoder_input_embed_dim

        if cfg.spec_feature_only:
            self.feature_extractor = None
            self.embed = cfg.conv_flatten_dim
        else:
            self.feature_extractor, self.embed = self.build_feature_extractor(cfg)

        self.spec_feature_dim = cfg.fbank_dim + cfg.mfcc_dim
        if self.spec_feature_dim > 0:
            assert (
                cfg.conv_flatten_dim == self.embed
            ), f"{cfg.conv_flatten_dim} != {self.embed}"
            self.spec_feature_extractor = self.build_spec_feature_extractor(cfg)
        else:
            self.spec_feature_extractor = None

        self.post_extract_proj = (
            nn.Linear(self.embed, cfg.encoder_input_embed_dim)
            if self.embed != cfg.encoder_input_embed_dim and not cfg.quantize_input
            else None
        )

        self.input_proj = (
            nn.Linear(cfg.encoder_input_embed_dim, cfg.encoder_embed_dim)
            if cfg.encoder_input_embed_dim != cfg.encoder_embed_dim
            else None
        )

        self.mask_prob = cfg.mask_prob
        self.mask_selection = cfg.mask_selection
        self.mask_other = cfg.mask_other
        self.mask_length = cfg.mask_length
        self.no_mask_overlap = cfg.no_mask_overlap
        self.mask_min_space = cfg.mask_min_space

        self.mask_channel_prob = cfg.mask_channel_prob
        self.mask_channel_selection = cfg.mask_channel_selection
        self.mask_channel_other = cfg.mask_channel_other
        self.mask_channel_length = cfg.mask_channel_length
        self.no_mask_channel_overlap = cfg.no_mask_channel_overlap
        self.mask_channel_min_space = cfg.mask_channel_min_space

        self.dropout_input = nn.Dropout(cfg.dropout_input)
        self.dropout_features = nn.Dropout(cfg.dropout_features)

        self.feature_grad_mult = cfg.feature_grad_mult

        self.quantizer = None
        self.input_quantizer = None

        self.n_negatives = cfg.num_negatives
        self.cross_sample_negatives = cfg.cross_sample_negatives
        self.codebook_negatives = cfg.codebook_negatives
        self.negatives_from_everywhere = cfg.negatives_from_everywhere

        self.logit_temp = cfg.logit_temp

        final_dim = cfg.final_dim if cfg.final_dim > 0 else cfg.encoder_embed_dim

        if cfg.quantize_targets:
            vq_dim = cfg.latent_dim if cfg.latent_dim > 0 else final_dim
            self.quantizer = GumbelVectorQuantizer(
                dim=self.embed,
                num_vars=cfg.latent_vars,
                temp=cfg.latent_temp,
                groups=cfg.latent_groups,
                combine_groups=False,
                vq_dim=vq_dim,
                time_first=True,
            )
            self.project_q = nn.Linear(vq_dim, final_dim)
        else:
            self.project_q = nn.Linear(self.embed, final_dim)

        if cfg.quantize_input:
            if cfg.same_quantizer and self.quantizer is not None:
                vq_dim = final_dim
                self.input_quantizer = self.quantizer
            else:
                vq_dim = cfg.latent_dim if cfg.latent_dim > 0 else cfg.encoder_embed_dim
                self.input_quantizer = GumbelVectorQuantizer(
                    dim=self.embed,
                    num_vars=cfg.latent_vars,
                    temp=cfg.latent_temp,
                    groups=cfg.latent_groups,
                    combine_groups=False,
                    vq_dim=vq_dim,
                    time_first=True,
                )
            self.project_inp = nn.Linear(vq_dim, cfg.encoder_embed_dim)

        self.mask_emb = nn.Parameter(
            torch.FloatTensor(cfg.encoder_input_embed_dim).uniform_()
        )

        self.encoder = TransformerEncoder(cfg)
        self.layer_norm = LayerNorm(self.embed)

        self.target_glu = None
        if cfg.target_glu:
            self.target_glu = nn.Sequential(
                nn.Linear(final_dim, final_dim * 2), nn.GLU()
            )

        if cfg.use_mlp:
            if cfg.quantize_targets:
                vq_dim = cfg.latent_dim if cfg.latent_dim > 0 else final_dim
                self.project_q = self.build_mlp(vq_dim, final_dim)
                self.qk_same_dim = vq_dim == cfg.encoder_embed_dim
            else:
                self.project_q = self.build_mlp(self.embed, final_dim)
                self.qk_same_dim = self.embed == cfg.encoder_embed_dim

            self.final_proj = self.build_mlp(cfg.encoder_embed_dim, final_dim)

            if cfg.share_final_proj:
                if self.qk_same_dim:
                    self.final_proj = self.project_q
                else:
                    # don't share the first layer
                    if cfg.mlp_version == "v1":
                        self.final_proj.fc2 = self.project_q.fc2
                    elif cfg.mlp_version in {"v2", "v3"}:
                        for i in range(3, len(self.final_proj.net)):
                            self.final_proj.net[i] = self.project_q.net[i]
                    else:
                        raise ValueError(f"cfg.mlp_version={cfg.mlp_version}")
        else:
            self.final_proj = nn.Linear(cfg.encoder_embed_dim, final_dim)

        self.final_proj_momentum = cfg.final_proj_momentum
        if cfg.final_proj_momentum >= 0.0:
            assert not cfg.share_final_proj
            self.initialize_momentum()

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        return state_dict

    @classmethod
    def build_model(cls, args, task=None):
        """Build a new model instance."""

        return cls(args)

    def build_mlp(self, input_dim, output_dim):
        cfg = self.cfg
        if cfg.mlp_version == "v1":
            return MLP(
                input_dim,
                output_dim,
                hidden_size=cfg.mlp_hidden_size,
                norm_type=cfg.proj_mlp_norm_type,
            )
        elif cfg.mlp_version == "v2":
            return MLPv2(
                input_dim,
                output_dim,
                hidden_dim=cfg.mlp_hidden_size,
                num_layers=cfg.mlp_num_layers,
                norm_type=cfg.proj_mlp_norm_type,
                act_type=cfg.mlp_act_type,
            )
        elif cfg.mlp_version == "v3":
            return MLPv3(
                input_dim,
                output_dim,
                hidden_dim=cfg.mlp_hidden_size,
                num_layers=cfg.mlp_num_layers,
                norm_type=cfg.proj_mlp_norm_type,
                act_type=cfg.mlp_act_type,
            )
        else:
            raise ValueError(f"cfg.mlp_version={cfg.mlp_version}")

    @torch.no_grad()
    def initialize_momentum(self):
        if self.final_proj_momentum < 0.0:
            if self.qk_same_dim:
                src, tgt = self.final_proj, self.project_q
            else:
                src, tgt = self.final_proj.fc2, self.project_q.fc2
            for ps, pt in zip(src.parameters(), tgt.parameters()):
                pt.data.copy_(ps.data)
                pt.requires_grad = False

    @torch.no_grad()
    def update_momentum(self):
        if self.final_proj_momentum >= 0.0 and self.training:
            if self.qk_same_dim:
                src, tgt = self.final_proj, self.project_q
            else:
                src, tgt = self.final_proj.fc2, self.project_q.fc2

            m = self.final_proj_momentum
            for ps, pt in zip(src.parameters(), tgt.parameters()):
                dtype = pt.dtype
                new_p = pt.data.float().mul_(m).add_(ps.data.float(), alpha=1.0 - m)
                pt.data.copy_(new_p.to(dtype))

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        if num_updates != getattr(self, "num_updates", -1):
            self.update_momentum()
        self.num_updates = num_updates
        super().set_num_updates(num_updates)

    def apply_mask(self, x, padding_mask):
        B, T, C = x.shape
        if self.mask_prob > 0:
            mask_indices = compute_mask_indices(
                (B, T),
                padding_mask,
                self.mask_prob,
                self.mask_length,
                self.mask_selection,
                self.mask_other,
                min_masks=2,
                no_overlap=self.no_mask_overlap,
                min_space=self.mask_min_space,
            )
            mask_indices = torch.from_numpy(mask_indices).to(x.device)
            x[mask_indices] = self.mask_emb
        else:
            mask_indices = None

        if self.mask_channel_prob > 0:
            mask_channel_indices = compute_mask_indices(
                (B, C),
                None,
                self.mask_channel_prob,
                self.mask_channel_length,
                self.mask_channel_selection,
                self.mask_channel_other,
                no_overlap=self.no_mask_channel_overlap,
                min_space=self.mask_channel_min_space,
            )
            mask_channel_indices = (
                torch.from_numpy(mask_channel_indices)
                .to(x.device)
                .unsqueeze(1)
                .expand(-1, T, -1)
            )
            x[mask_channel_indices] = 0

        return x, mask_indices

    def sample_negatives(self, y, num):

        if self.n_negatives == 0 and self.cross_sample_negatives == 0:
            return y.new(0)

        bsz, tsz, fsz = y.shape
        y = y.view(-1, fsz)  # BTC => (BxT)C

        cross_high = tsz * bsz
        high = tsz
        with torch.no_grad():
            assert high > 1, f"{bsz,tsz,fsz}"

            if self.n_negatives > 0:
                tszs = (
                    buffered_arange(num)
                    .unsqueeze(-1)
                    .expand(-1, self.n_negatives)
                    .flatten()
                )

                neg_idxs = torch.randint(
                    low=0, high=high - 1, size=(bsz, self.n_negatives * num)
                )
                neg_idxs[neg_idxs >= tszs] += 1

            if self.cross_sample_negatives > 0:
                tszs = (
                    buffered_arange(num)
                    .unsqueeze(-1)
                    .expand(-1, self.cross_sample_negatives)
                    .flatten()
                )

                cross_neg_idxs = torch.randint(
                    low=0,
                    high=cross_high - 1,
                    size=(bsz, self.cross_sample_negatives * num),
                )
                cross_neg_idxs[cross_neg_idxs >= tszs] += 1

        if self.n_negatives > 0:
            for i in range(1, bsz):
                neg_idxs[i] += i * high
        else:
            neg_idxs = cross_neg_idxs

        if self.cross_sample_negatives > 0 and self.n_negatives > 0:
            neg_idxs = torch.cat([neg_idxs, cross_neg_idxs], dim=1)

        negs = y[neg_idxs.view(-1)]
        negs = negs.view(
            bsz, num, self.n_negatives + self.cross_sample_negatives, fsz
        ).permute(
            2, 0, 1, 3
        )  # to NxBxTxC
        return negs, neg_idxs

    def compute_preds(self, x, y, negatives):

        neg_is_pos = (y == negatives).all(-1)
        y = y.unsqueeze(0)
        targets = torch.cat([y, negatives], dim=0)

        logits = torch.cosine_similarity(x.float(), targets.float(), dim=-1).type_as(x)

        logits /= self.logit_temp

        if neg_is_pos.any():
            logits[1:][neg_is_pos] = float("-inf")

        return logits

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """
        Computes the output length of the convolutional layers
        """

        def _conv_out_length(input_length, kernel_size, stride):
            return torch.floor((input_length - kernel_size) / stride + 1)

        conv_cfg_list = eval(self.cfg.conv_feature_layers)

        for i in range(len(conv_cfg_list)):
            input_lengths = _conv_out_length(
                input_lengths, conv_cfg_list[i][1], conv_cfg_list[i][2]
            )

        return input_lengths.to(torch.long)

    def forward(
        self,
        source,
        padding_mask=None,
        audio_feats=None,
        mask=True,
        features_only=False,
        embed_inputs=False,
        before_context=False,
        layer=None,
    ):

        if embed_inputs:
            features = source
        else:
            if self.feature_extractor is not None:
                if self.feature_grad_mult > 0:
                    features = self.feature_extractor(source)
                    if self.feature_grad_mult != 1.0:
                        features = GradMultiply.apply(features, self.feature_grad_mult)
                else:
                    with torch.no_grad():
                        features = self.feature_extractor(source)

                features = features.transpose(1, 2)  # B x T x C
            else:
                features = None

            if self.spec_feature_extractor is not None:
                assert audio_feats is not None
                if self.feature_grad_mult > 0:
                    spec_features = self.spec_feature_extractor(audio_feats)
                    if self.feature_grad_mult != 1.0:
                        spec_features = GradMultiply.apply(
                            spec_features, self.feature_grad_mult
                        )
                    features = (
                        spec_features if features is None else features + spec_features
                    )
                else:
                    with torch.no_grad():
                        spec_features = self.spec_feature_extractor(audio_feats)
                        features = (
                            spec_features
                            if features is None
                            else features + spec_features
                        )

        features_pen = features.float().pow(2).mean()

        features = self.layer_norm(features)
        unmasked_features = features.clone()

        if padding_mask is not None:
            extra = padding_mask.size(1) % features.size(1)
            if extra > 0:
                padding_mask = padding_mask[:, :-extra]
            padding_mask = padding_mask.view(padding_mask.size(0), features.size(1), -1)
            padding_mask = padding_mask.all(-1)

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        if features_only and before_context:
            return {"x": features, "padding_mask": padding_mask}

        features = self.dropout_input(features)
        unmasked_features = self.dropout_features(unmasked_features)

        num_vars = None
        code_ppl = None
        prob_ppl = None
        curr_temp = None

        if self.input_quantizer:
            q = self.input_quantizer(features, produce_targets=False)
            features = q["x"]
            num_vars = q["num_vars"]
            code_ppl = q["code_perplexity"]
            prob_ppl = q["prob_perplexity"]
            curr_temp = q["temp"]
            features = self.project_inp(features)

        if mask:
            x, mask_indices = self.apply_mask(features, padding_mask)
            if mask_indices is not None:
                y = unmasked_features[mask_indices].view(
                    unmasked_features.size(0), -1, unmasked_features.size(-1)
                )
            else:
                y = unmasked_features
        else:
            x = features
            y = unmasked_features
            mask_indices = None

        if self.input_proj is not None:
            x = self.input_proj(x)

        x, layer_results = self.encoder(x, padding_mask=padding_mask, layer=layer)

        if features_only:
            return {
                "x": x,
                "padding_mask": padding_mask,
                "features": unmasked_features,
                "layer_results": layer_results,
            }

        if self.quantizer:
            q = self.quantizer(y, produce_targets=False)
            y = q["x"]
            num_vars = q["num_vars"]
            code_ppl = q["code_perplexity"]
            prob_ppl = q["prob_perplexity"]
            curr_temp = q["temp"]

            y = self.project_q(y)

            if self.negatives_from_everywhere:
                neg_cands, *_ = self.quantizer(unmasked_features, produce_targets=False)
                negs, _ = self.sample_negatives(neg_cands, y.size(1))
                negs = self.project_q(negs)

            else:
                negs, _ = self.sample_negatives(y, y.size(1))

            if self.codebook_negatives > 0:
                cb_negs = self.quantizer.sample_from_codebook(
                    y.size(0) * y.size(1), self.codebook_negatives
                )
                cb_negs = cb_negs.view(
                    self.codebook_negatives, y.size(0), y.size(1), -1
                )  # order doesnt matter
                cb_negs = self.project_q(cb_negs)
                negs = torch.cat([negs, cb_negs], dim=0)
        else:
            y = self.project_q(y)

            if self.negatives_from_everywhere:
                negs, _ = self.sample_negatives(unmasked_features, y.size(1))
                negs = self.project_q(negs)
            else:
                negs, _ = self.sample_negatives(y, y.size(1))

        x = x[mask_indices].view(x.size(0), -1, x.size(-1))

        if self.target_glu:
            y = self.target_glu(y)
            negs = self.target_glu(negs)

        x = self.final_proj(x)

        x = self.compute_preds(x, y, negs)

        result = {"x": x, "padding_mask": padding_mask, "features_pen": features_pen}

        if prob_ppl is not None:
            result["prob_perplexity"] = prob_ppl
            result["code_perplexity"] = code_ppl
            result["num_vars"] = num_vars
            result["temp"] = curr_temp

        return result

    def quantize(self, x):
        assert self.quantizer is not None
        x = self.feature_extractor(x)
        x = x.transpose(1, 2)
        x = self.layer_norm(x)
        return self.quantizer.forward_idx(x)

    def extract_features(
        self,
        source,
        padding_mask,
        audio_feats=None,
        mask=False,
        embed_inputs=False,
        layer=None,
    ):
        res = self.forward(
            source,
            padding_mask,
            audio_feats=audio_feats,
            mask=mask,
            features_only=True,
            embed_inputs=embed_inputs,
            layer=layer,
        )
        return res

    def get_logits(self, net_output):
        logits = net_output["x"]
        logits = logits.transpose(0, 2)
        logits = logits.reshape(-1, logits.size(-1))
        return logits

    def get_targets(self, sample, net_output, expand_steps=True):
        x = net_output["x"]
        return x.new_zeros(x.size(1) * x.size(2), dtype=torch.long)

    def get_extra_losses(self, net_output):
        pen = []

        if "prob_perplexity" in net_output:
            pen.append(
                (net_output["num_vars"] - net_output["prob_perplexity"])
                / net_output["num_vars"]
            )

        if "features_pen" in net_output:
            pen.append(net_output["features_pen"])

        return pen

    def remove_pretraining_modules(self):
        self.quantizer = None
        self.project_q = None
        self.target_glu = None
        self.final_proj = None

    def build_feature_extractor(self, cfg):
        if cfg.feature_extractor_type == "default":
            # from fairseq.models.wav2vec.wav2vec2 import ConvFeatureExtractionModel
            from ..modules.wav_extracter import ConvFeatureExtractionModelV2

            feature_enc_layers = eval(cfg.conv_feature_layers)
            extractor_output_dim = feature_enc_layers[-1][0]
            feature_extractor = ConvFeatureExtractionModelV2(
                conv_layers=feature_enc_layers,
                dropout=0.0,
                mode=cfg.extractor_mode,
                conv_bias=cfg.conv_bias,
            )
        else:
            raise ValueError(f"feature_extractor_type={cfg.feature_extractor_type}")
        return feature_extractor, extractor_output_dim

    def build_spec_feature_extractor(self, cfg):
        if cfg.subsample_type == "conv2d":
            # remove from SEW open source
            raise NotImplementedError
        else:
            raise ValueError(f"subsample_type={cfg.subsample_type}")
        return spec_feature_extractor

    def extra_repr(self):
        print_attrs = [k for k in self.__dict__ if "mask" in k or "drop" in k] + [
            "feature_grad_mult"
        ]
        s = ", ".join([f"{k}={{{k}}}" for k in print_attrs])
        return s.format(**self.__dict__)
