from __future__ import annotations

import torch
import torch.nn as nn

from fla.layers.attn import Attention
from fla.models.kda.modeling_kda import (
    KDAForCausalLM,
    KDAModel,
    KDAPreTrainedModel,
)
from fla.modules import GatedMLP as KDAMLP
from fla.modules import RMSNorm

from .configuration_kda_swa import KDAWindowMixConfig
from .layer_kda_swa import KDAWindowMix

try:
    from transformers.modeling_layers import GradientCheckpointingLayer
except ImportError:
    from fla.models.modeling_layers import GradientCheckpointingLayer


class KDAWindowMixBlock(GradientCheckpointingLayer):
    def __init__(self, config: KDAWindowMixConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.attn_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)
        if config.attn is not None and layer_idx in config.attn["layers"]:
            self.attn = Attention(
                hidden_size=config.hidden_size,
                num_heads=config.attn["num_heads"],
                num_kv_heads=config.attn["num_kv_heads"],
                qkv_bias=config.attn["qkv_bias"],
                window_size=config.attn["window_size"],
                rope_theta=config.attn["rope_theta"],
                max_position_embeddings=config.max_position_embeddings,
                layer_idx=layer_idx,
            )
        else:
            self.attn = KDAWindowMix(
                mode=config.attn_mode,
                hidden_size=config.hidden_size,
                expand_v=config.expand_v,
                head_dim=config.head_dim,
                num_heads=config.num_heads,
                num_v_heads=config.num_v_heads,
                use_short_conv=config.use_short_conv,
                allow_neg_eigval=config.allow_neg_eigval,
                conv_size=config.conv_size,
                norm_eps=config.norm_eps,
                layer_idx=layer_idx,
                attn_head_dim=config.attn_head_dim,
                window_size=config.window_size,
                rope_theta=config.rope_theta,
                use_window_attention=getattr(config, "use_window_attention", True),
            )
        self.mlp_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)
        self.mlp = KDAMLP(
            hidden_size=config.hidden_size,
            hidden_ratio=config.hidden_ratio,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            fuse_swiglu=config.fuse_swiglu,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        hidden_states, attentions, past_key_values = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            **kwargs,
        )
        if self.config.fuse_norm:
            hidden_states, residual = self.mlp_norm(hidden_states, residual, True)
        else:
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.mlp_norm(hidden_states)
        hidden_states = self.mlp(hidden_states, **kwargs)
        hidden_states = residual + hidden_states
        return hidden_states, attentions, past_key_values


class KDAWindowMixPreTrainedModel(KDAPreTrainedModel):
    config_class = KDAWindowMixConfig
    _no_split_modules = ["KDAWindowMixBlock"]

    def _init_weights(self, module: nn.Module, *args, **kwargs):
        super()._init_weights(module, *args, **kwargs)
        if hasattr(module, "qk_scale"):
            nn.init.ones_(module.qk_scale)
            nn.init.zeros_(module.qk_offset)


class KDAWindowMixModel(KDAWindowMixPreTrainedModel, KDAModel):
    config_class = KDAWindowMixConfig

    def __init__(self, config: KDAWindowMixConfig):
        KDAWindowMixPreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([KDAWindowMixBlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)
        self.gradient_checkpointing = False
        self.post_init()


class KDAWindowMixForCausalLM(KDAWindowMixPreTrainedModel, KDAForCausalLM):
    config_class = KDAWindowMixConfig

    def __init__(self, config: KDAWindowMixConfig):
        KDAWindowMixPreTrainedModel.__init__(self, config)
        self.model = KDAWindowMixModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.criterion = None
        self.post_init()
