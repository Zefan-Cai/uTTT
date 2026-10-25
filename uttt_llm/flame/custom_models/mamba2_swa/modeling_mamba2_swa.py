from __future__ import annotations

import torch
import torch.nn as nn

from fla.layers.attn import Attention
from fla.models.mamba2.modeling_mamba2 import (
    Mamba2Block,
    Mamba2ForCausalLM,
    Mamba2Model,
    Mamba2PreTrainedModel,
)
from fla.modules import RMSNorm
from fla.modules import GatedMLP as Mamba2WindowMixMLP

from .configuration_mamba2_swa import Mamba2WindowMixConfig
from .layer_mamba2_swa import Mamba2WindowMix


class Mamba2WindowMixBlock(Mamba2Block):
    def __init__(self, config: Mamba2WindowMixConfig, layer_idx: int):
        super(Mamba2Block, self).__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.residual_in_fp32 = config.residual_in_fp32
        self.norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        if config.attn is not None and layer_idx in config.attn["layers"]:
            self.mixer = Attention(
                hidden_size=config.hidden_size,
                num_heads=config.attn["num_heads"],
                num_kv_heads=config.attn["num_kv_heads"],
                qkv_bias=config.attn["qkv_bias"],
                window_size=config.attn["window_size"],
                rope_theta=config.attn["rope_theta"],
                max_position_embeddings=getattr(config, "max_position_embeddings", None),
                layer_idx=layer_idx,
            )
        else:
            self.mixer = Mamba2WindowMix(
                num_heads=config.num_heads,
                head_dim=config.head_dim,
                hidden_size=config.hidden_size,
                state_size=config.state_size,
                expand=config.expand,
                n_groups=config.n_groups,
                conv_kernel=config.conv_kernel,
                use_conv_bias=config.use_conv_bias,
                hidden_act=config.hidden_act,
                rms_norm=config.rms_norm,
                chunk_size=config.chunk_size,
                time_step_rank=config.time_step_rank,
                time_step_limit=config.time_step_limit,
                time_step_min=config.time_step_min,
                time_step_max=config.time_step_max,
                use_bias=config.use_bias,
                norm_eps=config.norm_eps,
                layer_idx=layer_idx,
                attn_head_dim=config.attn_head_dim,
                attn_num_heads=getattr(config, "attn_num_heads", None),
                window_size=config.window_size,
                rope_theta=config.rope_theta,
                use_window_attention=getattr(config, "use_window_attention", True),
            )
        # Optional SwiGLU MLP after the mixer (same block layout as the DeltaNet/GatedDeltaNet window-mix
        # baselines and TTT-Dense). Legacy configs (hidden_ratio=None) keep the mixer-only Mamba-2 block.
        self.hidden_ratio = getattr(config, "hidden_ratio", None)
        if self.hidden_ratio is not None:
            self.mlp_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
            self.mlp = Mamba2WindowMixMLP(
                hidden_size=config.hidden_size,
                hidden_ratio=self.hidden_ratio,
                intermediate_size=getattr(config, "intermediate_size", None),
                hidden_act="swish",
                fuse_swiglu=getattr(config, "fuse_swiglu", True),
            )
        else:
            self.mlp_norm = None
            self.mlp = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs,
    ):
        if self.mlp is None:
            return super().forward(
                hidden_states,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                **kwargs,
            )
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)
        hidden_states, attentions, past_key_values = self.mixer(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            **kwargs,
        )
        compute_dtype = hidden_states.dtype  # mixer output dtype (bf16 under mixed precision)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.mlp_norm(hidden_states.to(dtype=compute_dtype))
        hidden_states = residual + self.mlp(hidden_states, **kwargs)
        if self.residual_in_fp32:
            hidden_states = hidden_states.to(dtype=self.norm.weight.dtype)
        return hidden_states, attentions, past_key_values


class Mamba2WindowMixPreTrainedModel(Mamba2PreTrainedModel):
    config_class = Mamba2WindowMixConfig
    _no_split_modules = ["Mamba2WindowMixBlock"]

    def _init_weights(self, module: nn.Module, *args, **kwargs):
        # two residual branches per layer (mixer + MLP) when the MLP is enabled, as in GatedDeltaNet
        if getattr(self.config, "hidden_ratio", None) is not None and "num_residuals_per_layer" not in kwargs and not args:
            kwargs["num_residuals_per_layer"] = 2
        super()._init_weights(module, *args, **kwargs)
        if hasattr(module, "qk_scale"):
            nn.init.ones_(module.qk_scale)
            nn.init.zeros_(module.qk_offset)


class Mamba2WindowMixModel(Mamba2WindowMixPreTrainedModel, Mamba2Model):
    config_class = Mamba2WindowMixConfig

    def __init__(self, config: Mamba2WindowMixConfig):
        Mamba2WindowMixPreTrainedModel.__init__(self, config)
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Mamba2WindowMixBlock(config, idx) for idx in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False
        self.norm_f = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self._register_load_state_dict_pre_hook(self.load_hook)
        self.post_init()


class Mamba2WindowMixForCausalLM(Mamba2WindowMixPreTrainedModel, Mamba2ForCausalLM):
    config_class = Mamba2WindowMixConfig

    def __init__(self, config: Mamba2WindowMixConfig):
        Mamba2WindowMixPreTrainedModel.__init__(self, config)
        self.backbone = Mamba2WindowMixModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.criterion = None
        self.post_init()
