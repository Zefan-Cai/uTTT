from __future__ import annotations

import math

import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch.nn import functional as F

from fla.layers.kda import KimiDeltaAttention
from fla.layers.utils import (
    get_layer_cache,
    get_unpad_data,
    index_first_axis,
    pad_input,
    prepare_lens_from_mask,
    unpad_input,
    update_layer_cache,
)
from fla.modules import RotaryEmbedding
from fla.ops.kda import chunk_kda, fused_recurrent_kda
from fla.ops.kda.gate import fused_kda_gate

from flame.custom_models.fla_window_mix_utils import (
    cast_floating_point_parameters,
    normalize_cu_seqlens,
    sliding_window_attention_from_projections,
)

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
except ImportError:  # pragma: no cover - the training image ships flash-attn
    flash_attn_func = flash_attn_varlen_func = None


class KDAWindowMix(KimiDeltaAttention):
    def __init__(
        self,
        *args,
        attn_head_dim: int | None = None,
        window_size: int | None = 4096,
        rope_theta: float = 1000000.0,
        use_window_attention: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        # None -> SWA head dim tied to the KDA head_dim. The SWA branch re-splits the
        # shared q/k (key_dim) and v (value_dim) projections into
        # key_dim // attn_head_dim q-heads and value_dim // attn_head_dim v-heads.
        self.attn_head_dim = attn_head_dim if attn_head_dim is not None else self.head_dim
        self.window_size = window_size
        self.rope_theta = rope_theta
        self.use_window_attention = use_window_attention
        parameter_dtype = torch.get_default_dtype()
        if self.use_window_attention:
            self.qk_scale = nn.Parameter(torch.ones(self.key_dim, 2, dtype=parameter_dtype))
            self.qk_offset = nn.Parameter(torch.zeros(self.key_dim, 2, dtype=parameter_dtype))
            self.rotary = RotaryEmbedding(dim=self.attn_head_dim, base=self.rope_theta)
        cast_floating_point_parameters(self, parameter_dtype)

    def reset_parameters(self) -> None:
        if self.use_window_attention:
            nn.init.ones_(self.qk_scale)
            nn.init.zeros_(self.qk_offset)
        with torch.no_grad():
            # mirror fla KimiDeltaAttention.__init__ (kda.py): A_log ~ log(U(1, 16)),
            # dt_bias = inv-softplus of dt ~ exp(U(log 1e-3, log 1e-1)).clamp(min=1e-4)
            A = torch.empty_like(self.A_log, dtype=torch.float32).uniform_(1, 16)
            self.A_log.copy_(torch.log(A).to(dtype=self.A_log.dtype))
            dt = torch.exp(
                torch.rand_like(self.dt_bias, dtype=torch.float32) * (math.log(0.1) - math.log(0.001))
                + math.log(0.001)
            ).clamp(min=1e-4)
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            self.dt_bias.copy_(inv_dt.to(dtype=self.dt_bias.dtype))
            self.A_log._no_weight_decay = True
            self.dt_bias._no_weight_decay = True

    def _window_attention_cached(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_values,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Sliding-window branch with a KV cache (decoding with use_cache=True), ported from the MoM window-mix layer
        (eae84aec). Same projections, scale/offset, head split and RoPE base as the training path in
        `sliding_window_attention_from_projections`, but RoPE is applied at the absolute positions the cache has
        seen, attention runs over the cached keys plus the new tokens, and the last `window_size` keys/values are
        returned for the caller to store as this layer's `attn_state`. Without a cache the training path runs."""
        if flash_attn_func is None or flash_attn_varlen_func is None:
            raise ImportError("Please install Flash Attention via `pip install flash-attn --no-build-isolation` first")
        batch_size, q_len, _ = hidden_states.shape
        q, k, v = self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)
        key_dim, value_dim = q.shape[-1], v.shape[-1]
        scale = self.qk_scale.to(dtype=q.dtype).view(1, 1, key_dim, 2)
        offset = self.qk_offset.to(dtype=q.dtype).view(1, 1, key_dim, 2)
        q = q * scale[..., 0] + offset[..., 0]
        k = k * scale[..., 1] + offset[..., 1]
        q = rearrange(q, "... (h d) -> ... h d", d=self.attn_head_dim)
        k = rearrange(k, "... (h d) -> ... h d", d=self.attn_head_dim)
        v = rearrange(v, "... (h d) -> ... h d", d=self.attn_head_dim)
        head_repeat = v.shape[-2] // q.shape[-2]
        if head_repeat != 1:
            q = torch.repeat_interleave(q, head_repeat, dim=-2)
            k = torch.repeat_interleave(k, head_repeat, dim=-2)

        # absolute positions, as fla's Attention layer: the cache counts every token seen, padding included,
        # and the per-sample correction removes the left padding again
        seqlen_offset = past_key_values.get_seq_length(self.layer_idx)
        max_seqlen = q_len + seqlen_offset
        mask = attention_mask.bool() if attention_mask is not None else None
        if mask is not None:
            seqlen_offset = seqlen_offset + prepare_lens_from_mask(mask) - mask.shape[-1]
            max_seqlen = q_len + int(max(seqlen_offset))
        q, k = self.rotary(q, k, seqlen_offset=seqlen_offset, max_seqlen=max_seqlen)

        last_state = get_layer_cache(self, past_key_values)
        if last_state is not None and last_state.get("attn_state") is not None:
            k_old, v_old = last_state["attn_state"]
            k = torch.cat([k_old, k], dim=1)
            v = torch.cat([v_old, v], dim=1)
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        # attention runs over every cached key plus the new ones; flash's window bound does the truncation,
        # and only the stored copy is cut to the window
        window = (-1, -1) if self.window_size is None else (self.window_size - 1, 0)
        if mask is not None:
            # the cached keys are the last k.shape[1] positions of the padded sequence
            mask = mask[:, -k.shape[1]:]
            q_u, (k_u, v_u), indices_q, cu_seqlens, max_seq_lens = unpad_input(q, (k, v), mask, q_len)
            o = flash_attn_varlen_func(
                q_u, k_u, v_u,
                cu_seqlens_q=cu_seqlens[0],
                cu_seqlens_k=cu_seqlens[1],
                max_seqlen_q=max_seq_lens[0],
                max_seqlen_k=max_seq_lens[1],
                causal=True,
                window_size=window,
            )
            o = pad_input(o, indices_q, batch_size, q_len)
        else:
            o = flash_attn_func(q, k, v, causal=True, window_size=window)
        if self.window_size is not None and k.shape[1] > self.window_size:
            k, v = k[:, -self.window_size:].contiguous(), v[:, -self.window_size:].contiguous()
        return o.reshape(batch_size, q_len, value_dim), (k, v)

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

        padded_hidden_states = hidden_states
        cu_seqlens = normalize_cu_seqlens(kwargs.get("cu_seqlens"))
        max_seqlen = q_len
        indices = None
        if attention_mask is not None:
            indices, cu_seqlens, max_seqlen = get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices).unsqueeze(0)

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        attn_o, attn_kv = None, None
        if self.use_window_attention:
            if past_key_values is None:
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
            else:
                # cached decoding: the window branch works on the padded [batch, q_len] layout and is brought
                # back to the unpadded layout of the KDA path before the two are added
                attn_o, attn_kv = self._window_attention_cached(padded_hidden_states, attention_mask, past_key_values)
                if indices is not None:
                    attn_o = index_first_axis(rearrange(attn_o, "b s ... -> (b s) ..."), indices).unsqueeze(0)

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

        g = self.f_proj(hidden_states)
        beta = self.b_proj(hidden_states).sigmoid()

        q, k, g = (rearrange(x, "... (h d) -> ... h d", d=self.head_k_dim) for x in (q, k, g))
        v = rearrange(v, "... (h d) -> ... h d", d=self.head_v_dim)

        if self.num_v_heads > self.num_heads:
            q, k, g = (repeat(x, "... h d -> ... (h g) d", g=self.num_v_heads // self.num_heads) for x in (q, k, g))
            beta = repeat(beta, "... h -> ... (h g)", g=self.num_v_heads // self.num_heads)

        if self.allow_neg_eigval:
            beta = beta * 2.0

        recurrent_state = last_state["recurrent_state"] if last_state is not None else None
        if mode == "chunk":
            o, recurrent_state = chunk_kda(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                A_log=self.A_log,
                dt_bias=self.dt_bias,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
                cu_seqlens=cu_seqlens,
            )
        elif mode == "fused_recurrent":
            g = fused_kda_gate(g=g, A_log=self.A_log, dt_bias=self.dt_bias)
            o, recurrent_state = fused_recurrent_kda(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seqlens,
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
        if attn_kv is not None:
            # stored after the KDA update so the cache entry exists on the first call; the window branch is the
            # only reader, so it need not go through Cache.update's append/roll
            past_key_values[self.layer_idx]["attn_state"] = attn_kv

        o = self.o_norm(o, rearrange(self.g_proj(hidden_states), "... (h d) -> ... h d", d=self.head_v_dim))
        o = rearrange(o, "b t h d -> b t (h d)")
        o = self.o_proj(o if attn_o is None else o + attn_o)
        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, q_len)

        return o, None, past_key_values
