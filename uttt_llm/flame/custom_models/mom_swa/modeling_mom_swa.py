from __future__ import annotations

import torch
import torch.nn as nn

from fla.models.mom.modeling_mom import MomBlock, MomForCausalLM, MomModel, MomPreTrainedModel
from fla.modules import GatedMLP as MomMLP
from fla.modules import RMSNorm

from flame.custom_models.fla_window_mix_utils import cast_floating_point_parameters

from .configuration_mom_swa import MomWindowMixConfig
from .layer_mom_swa import MomWindowMix


class MomWindowMixBlock(MomBlock):
    """MomBlock with MomWindowMix as the mixer. Built here because MomBlock.__init__ passes use_short_conv straight
    to MomAttention, which refuses False; forward is MomBlock's."""

    def __init__(self, config: MomWindowMixConfig, layer_idx: int):
        super(MomBlock, self).__init__()   # GradientCheckpointingLayer.__init__, skipping MomBlock's
        self.hidden_size = config.hidden_size
        self.config = config
        self.layer_idx = layer_idx
        self.attn_norm = RMSNorm(hidden_size=config.hidden_size, eps=config.norm_eps)
        if config.mom_backend != "gated_deltanet":
            raise NotImplementedError(f"MoM backend {config.mom_backend} is not supported")
        self.attn = MomWindowMix(
            mode=config.attn_mode,
            hidden_size=config.hidden_size,
            expand_v=config.expand_v,
            head_dim=config.head_dim,
            num_heads=config.num_heads,
            use_output_gate=config.use_output_gate,
            use_short_conv=config.use_short_conv,
            conv_size=config.conv_size,
            norm_eps=config.norm_eps,
            layer_idx=layer_idx,
            num_memories=config.num_memories,
            topk=config.topk,
            capacity=config.capacity,
            shared_mem=config.shared_mem,
            single_kv_proj=config.single_kv_proj,
            attn_head_dim=config.attn_head_dim,
            window_size=config.window_size,
            rope_theta=config.rope_theta,
            use_window_attention=config.use_window_attention,
        )
        self.mlp_norm = RMSNorm(hidden_size=config.hidden_size, eps=config.norm_eps)
        self.mlp = MomMLP(
            hidden_size=config.hidden_size,
            hidden_ratio=config.hidden_ratio,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            fuse_swiglu=config.fuse_swiglu,
        )
        # fla's MomBlock pins its norms to fp32; keep one dtype per FSDP unit, as the GDN block does.
        cast_floating_point_parameters(self, torch.get_default_dtype())


class MomWindowMixPreTrainedModel(MomPreTrainedModel):
    config_class = MomWindowMixConfig
    # fla's MomPreTrainedModel leaves HF's empty base_model_prefix, so flame's get_blocks() finds no "layers": no
    # activation checkpointing and no per-block FSDP (smoke 7971aedf held every layer's activations and OOMed at
    # batch 2 on 40 GB). The body sits under .model, as in the other fla models.
    base_model_prefix = "model"
    _no_split_modules = ["MomWindowMixBlock"]


class MomWindowMixModel(MomWindowMixPreTrainedModel, MomModel):
    def __init__(self, config: MomWindowMixConfig):
        MomWindowMixPreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([MomWindowMixBlock(config, idx) for idx in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.gradient_checkpointing = False
        self.post_init()


class MomWindowMixForCausalLM(MomWindowMixPreTrainedModel, MomForCausalLM):
    def __init__(self, config: MomWindowMixConfig):
        MomWindowMixPreTrainedModel.__init__(self, config)
        self.model = MomWindowMixModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.num_memories = config.num_memories
        self.topk = config.topk
        self.aux_loss_scale = config.aux_loss_scale
        self.post_init()
