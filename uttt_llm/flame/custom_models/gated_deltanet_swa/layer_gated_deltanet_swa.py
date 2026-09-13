from __future__ import annotations

import math

import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch.nn import functional as F

from fla.layers.gated_deltanet import GatedDeltaNet
from fla.layers.utils import get_layer_cache, get_unpad_data, index_first_axis, pad_input, update_layer_cache
from fla.modules import RotaryEmbedding
from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule

from flame.custom_models.fla_window_mix_utils import (
    cast_floating_point_parameters,
    normalize_cu_seqlens,
    sliding_window_attention_from_projections,
)


class GatedDeltaNetWindowMix(GatedDeltaNet):
    def __init__(
        self,
        *args,
        window_size: int | None = 4096,
        rope_theta: float = 1000000.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.attn_head_dim = self.head_dim
        self.window_size = window_size
        self.rope_theta = rope_theta
        parameter_dtype = torch.get_default_dtype()
        self.qk_scale = nn.Parameter(torch.ones(self.key_dim, 2, dtype=parameter_dtype))
        self.qk_offset = nn.Parameter(torch.zeros(self.key_dim, 2, dtype=parameter_dtype))
        self.rotary = RotaryEmbedding(dim=self.attn_head_dim, base=self.rope_theta)
        cast_floating_point_parameters(self, parameter_dtype)

    def reset_parameters(self) -> None:
        nn.init.ones_(self.qk_scale)
        nn.init.zeros_(self.qk_offset)
        with torch.no_grad():
            A = torch.empty_like(self.A_log, dtype=torch.float32).uniform_(0, 16)
            self.A_log.copy_(torch.log(A).to(dtype=self.A_log.dtype))
            dt = torch.exp(
                torch.rand_like(self.dt_bias, dtype=torch.float32) * (math.log(0.1) - math.log(0.001))
                + math.log(0.001)
            ).clamp(min=1e-4)
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            self.dt_bias.copy_(inv_dt.to(dtype=self.dt_bias.dtype))
            self.A_log._no_weight_decay = True
            self.dt_bias._no_weight_decay = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs,
    ):
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        batch_size, q_len, _ = hidden_states.shape
        mode = "fused_recurrent" if (q_len <= 64 and not self.training) else self.mode
        if self.training:
            assert mode == "chunk", "Only chunk mode is supported in training."

        last_state = get_layer_cache(self, past_key_values)

        cu_seqlens = normalize_cu_seqlens(kwargs.get("cu_seqlens"))
        max_seqlen = q_len
        if attention_mask is not None:
            indices, cu_seqlens, max_seqlen = get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices).unsqueeze(0)

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        attn_o = sliding_window_attention_from_projections(
            q,
            k,
            v,
            qk_scale=self.qk_scale,
            qk_offset=self.qk_offset,
            attn_head_dim=self.attn_head_dim,
            rotary=self.rotary,
            window_size=self.window_size,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        if self.use_short_conv:
            conv_state_q, conv_state_k, conv_state_v = None, None, None
            if last_state is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state["conv_state"]
            q, conv_state_q = self.q_conv1d(
                x=q,
                cache=conv_state_q,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
            k, conv_state_k = self.k_conv1d(
                x=k,
                cache=conv_state_k,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
            v, conv_state_v = self.v_conv1d(
                x=v,
                cache=conv_state_v,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
        else:
            q = F.silu(q)
            k = F.silu(k)
            v = F.silu(v)

        q, k = map(lambda x: rearrange(x, "... (h d) -> ... h d", d=self.head_k_dim), (q, k))
        v = rearrange(v, "... (h d) -> ... h d", d=self.head_v_dim)

        if self.num_v_heads > self.num_heads:
            q, k = map(lambda x: repeat(x, "... h d -> ... (h g) d", g=self.num_v_heads // self.num_heads), (q, k))

        beta = self.b_proj(hidden_states).sigmoid()
        if self.allow_neg_eigval:
            beta = beta * 2.0

        g = -self.A_log.float().exp() * F.softplus(self.a_proj(hidden_states).float() + self.dt_bias.float())

        recurrent_state = last_state["recurrent_state"] if last_state is not None else None
        if mode == "chunk":
            o, recurrent_state = chunk_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=True,
            )
        elif mode == "fused_recurrent":
            o, recurrent_state = fused_recurrent_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        update_layer_cache(
            self,
            past_key_values,
            recurrent_state=recurrent_state,
            conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
            offset=q_len,
        )

        if self.use_gate:
            g = rearrange(self.g_proj(hidden_states), "... (h d) -> ... h d", d=self.head_v_dim)
            o = self.o_norm(o, g)
        else:
            o = self.o_norm(o)
        o = rearrange(o, "b t h d -> b t (h d)")
        o = self.o_proj(o + attn_o)
        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, q_len)

        return o, None, past_key_values
