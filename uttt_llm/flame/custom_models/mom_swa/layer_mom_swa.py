from __future__ import annotations

import math

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F

from fla.layers.mom import MomAttention, _upad_input, reconstruct, transform
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
from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule

from flame.custom_models.fla_window_mix_utils import (
    cast_floating_point_parameters,
    sliding_window_attention_from_projections,
)

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
except ImportError:  # pragma: no cover - the training image ships flash-attn
    flash_attn_func = flash_attn_varlen_func = None


class MomWindowMix(MomAttention):
    """fla MomAttention (Mixture-of-Memories over gated delta rule) plus the lactcmp sliding-window branch.

    Derived from fla 0.5.0's MomAttention.forward with three edits (the sliding-window branch, its cache,
    and the padded-length handling). The sliding-window branch keeps a KV cache of its last `window_size`
    keys/values in the layer's cache entry (`attn_state`), the same device as fla's Attention layer, so the model
    decodes with `use_cache=True` (2026-09-20); without a cache the branch runs the training path.
    """

    def __init__(
        self,
        *args,
        use_short_conv: bool = False,
        attn_head_dim: int | None = None,
        window_size: int | None = 4096,
        rope_theta: float = 1000000.0,
        use_window_attention: bool = True,
        **kwargs,
    ) -> None:
        # MomAttention raises on use_short_conv=False; build it with the convolutions and drop them afterwards.
        super().__init__(*args, use_short_conv=True, **kwargs)
        if not use_short_conv:
            del self.q_conv1d, self.k_conv1d, self.v_conv1d
            self.use_short_conv = False
        self.use_window_attention = use_window_attention
        if use_window_attention and not (self.single_kv_proj or self.shared_mem):
            raise ValueError("the SWA branch reuses the shared k/v projections: set single_kv_proj or shared_mem")
        self.attn_head_dim = attn_head_dim if attn_head_dim is not None else self.head_dim
        self.window_size = window_size
        self.rope_theta = rope_theta
        parameter_dtype = torch.get_default_dtype()
        if use_window_attention:
            self.qk_scale = nn.Parameter(torch.ones(self.key_dim, 2, dtype=parameter_dtype))
            self.qk_offset = nn.Parameter(torch.zeros(self.key_dim, 2, dtype=parameter_dtype))
            self.rotary = RotaryEmbedding(dim=self.attn_head_dim, base=self.rope_theta)
        # one dtype per FSDP unit (fla builds A_log/dt_bias in fp32)
        cast_floating_point_parameters(self, parameter_dtype)

    def reset_parameters(self) -> None:
        # flame builds the model on the meta device and initialises it in post_init, so A_log/dt_bias need this
        # (same draws as fla's MomAttention.__init__ and the GDN window-mix baseline).
        if self.use_window_attention:
            nn.init.ones_(self.qk_scale)
            nn.init.zeros_(self.qk_offset)
        with torch.no_grad():
            A = torch.empty_like(self.A_log, dtype=torch.float32).uniform_(0, 16)
            self.A_log.copy_(torch.log(A).to(dtype=self.A_log.dtype))
            dt = torch.exp(
                torch.rand_like(self.dt_bias, dtype=torch.float32) * (math.log(0.1) - math.log(0.001))
                + math.log(0.001)
            ).clamp(min=1e-4)
            self.dt_bias.copy_((dt + torch.log(-torch.expm1(-dt))).to(dtype=self.dt_bias.dtype))
        self.A_log._no_weight_decay = True
        self.dt_bias._no_weight_decay = True

    def prepare_recurrent_state(self, recurrent_state, cu_seqlens, cu_seqlen_all, reverse_indices, batch_size):
        """States of the memories that received tokens in this call, in kernel sequence order. fla's version
        assumes exactly topk*batch active memories (single-token decode) and fails on longer continuations."""
        if recurrent_state is None or cu_seqlens is None:
            return recurrent_state
        active = cu_seqlen_all[1:] != cu_seqlen_all[:-1]
        if bool(active.all()):
            return recurrent_state
        return recurrent_state[active.nonzero(as_tuple=True)[0]]

    def handle_recurrent_state(self, recurrent_state, recurrent_state_new, cu_seqlens, cu_seqlen_all, reverse_indices):
        if recurrent_state_new is None:
            return None
        if cu_seqlens is None:
            return recurrent_state_new
        active = cu_seqlen_all[1:] != cu_seqlen_all[:-1]
        if recurrent_state is None:
            recurrent_state = recurrent_state_new.new_zeros((active.numel(),) + tuple(recurrent_state_new.shape[1:]))
        elif bool(active.all()):
            return recurrent_state_new
        recurrent_state = recurrent_state.clone()
        recurrent_state[active.nonzero(as_tuple=True)[0]] = recurrent_state_new
        return recurrent_state

    def _window_attention(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        """Sliding-window branch. Without a cache: the training path (packed varlen flash attention). With a cache:
        RoPE at the absolute positions, attention over the cached window plus the new tokens, and the last
        `window_size` keys/values returned for the caller to store as this layer's `attn_state`."""
        if past_key_values is None:
            return self._window_attention_train(hidden_states, attention_mask), None
        return self._window_attention_cached(hidden_states, attention_mask, past_key_values)

    def _window_attention_train(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        batch_size, q_len, _ = hidden_states.shape
        x, cu_seqlens, max_seqlen = hidden_states, None, q_len
        if attention_mask is not None:
            indices, cu_seqlens, max_seqlen = get_unpad_data(attention_mask[:, -q_len:])
            x = index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices).unsqueeze(0)
        o = sliding_window_attention_from_projections(
            self.q_proj(x),
            self.shared_k(x),
            self.shared_v(x),
            qk_scale=self.qk_scale,
            qk_offset=self.qk_offset,
            attn_head_dim=self.attn_head_dim,
            rotary=self.rotary,
            window_size=self.window_size,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, q_len)
        return o

    def _window_attention_cached(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_values: Cache,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if flash_attn_func is None or flash_attn_varlen_func is None:
            raise ImportError("Please install Flash Attention via `pip install flash-attn --no-build-isolation` first")
        batch_size, q_len, _ = hidden_states.shape
        q, k, v = self.q_proj(hidden_states), self.shared_k(hidden_states), self.shared_v(hidden_states)
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
        if attention_mask is not None:
            seqlen_offset = seqlen_offset + prepare_lens_from_mask(attention_mask) - attention_mask.shape[-1]
            max_seqlen = q_len + int(max(seqlen_offset))
        q, k = self.rotary(q, k, seqlen_offset=seqlen_offset, max_seqlen=max_seqlen)

        last_state = get_layer_cache(self, past_key_values)
        if last_state is not None and last_state.get("attn_state") is not None:
            k_old, v_old = last_state["attn_state"]
            k = torch.cat([k_old, k], dim=1)
            v = torch.cat([v_old, v], dim=1)
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        # attention runs over every cached key plus the new ones (fla's unpad_input needs q_len == k_len at
        # prefill); flash's window bound does the truncation, and only the stored copy is cut to the window
        window = (-1, -1) if self.window_size is None else (self.window_size - 1, 0)
        if attention_mask is not None:
            # the cached keys are the last k.shape[1] positions of the padded sequence
            mask = attention_mask[:, -k.shape[1]:]
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
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs: Unpack[dict],
    ) -> tuple[torch.Tensor, torch.Tensor | None, Cache | None]:
        if attention_mask is not None:
            attention_mask = (attention_mask == 1)
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        origin_cu_seqlens = kwargs.get('cu_seqlens')
        if origin_cu_seqlens is not None:
            hidden_states, attention_mask = self.cu2pad(hidden_states, origin_cu_seqlens)

        mode = 'fused_recurrent' if (hidden_states.shape[1] <= 64 and not self.training) else self.mode
        if self.training:
            assert mode == 'chunk', "Only chunk mode is supported in training."

        last_state = get_layer_cache(self, past_key_values)
        # _, q_len = hidden_states.shape[0], hidden_states.shape[1]

        # 🔍 topk gating
        router_logits = self.gate(hidden_states)  # (bsz, q_len, num_memories)
        scores = F.softmax(router_logits, dim=2, dtype=torch.float)
        routing_weights, selected_memories = torch.topk(scores, self.topk, dim=-1)  # (bsz, seq, topk)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)  # we cast back to the input dtype
        routing_weights_full = torch.zeros(
            routing_weights.shape[0],
            routing_weights.shape[1],
            self.num_memories,
            dtype=routing_weights.dtype,
            device=routing_weights.device,
        ).scatter(-1, selected_memories, routing_weights)
        routing_mask = routing_weights_full.bool().int()

        # if self.use_output_gate:
        #     o_g = self.g_proj(hidden_states)

        batch_size, seq_len = hidden_states.shape[0], hidden_states.shape[1]

        shared_hidden_states = hidden_states
        attn_o, attn_kv = (
            self._window_attention(shared_hidden_states, attention_mask, past_key_values)
            if self.use_window_attention else (None, None)
        )
        hidden_states, indices, sorted_indices, max_len, mask, mask_2 = transform(
            hidden_states, routing_mask, self.num_memories, selected_memories, attention_mask)

        q = self.q_proj(hidden_states)
        if self.single_kv_proj:
            k = self.shared_k(hidden_states)
            v = self.shared_v(hidden_states)
            beta = self.shared_b(hidden_states).sigmoid()
            g = -self.A_log.float().exp() * F.softplus(self.shared_a(hidden_states).float() + self.dt_bias)
        else:
            k = torch.stack([k_expert(hidden_states[i]) for i, k_expert in enumerate(self.k_proj)], dim=0)
            v = torch.stack([v_expert(hidden_states[i]) for i, v_expert in enumerate(self.v_proj)], dim=0)
            beta = torch.stack([b_expert(hidden_states[i]).sigmoid() for i, b_expert in enumerate(self.b_proj)], dim=0)
            g = torch.stack([-self.A_log.float().exp() * F.softplus(a_expert(hidden_states[i]).float() + self.dt_bias)
                            for i, a_expert in enumerate(self.a_proj)], dim=0)

        q, k, v, g, beta, mask_2 = (rearrange(x, 'e b l ... ->  (e b) l ...') for x in (q, k, v, g, beta, mask_2))
        cu_q, cu_k, cu_v, cu_g, cu_beta, indices_q, cu_seqlen_all, max_seq_lens = _upad_input(q, k, v, g, beta, mask_2)
        cu_seqlens, reverse_indices = cu_seqlen_all[0].to(torch.long).unique(return_inverse=True)
        cu_q, cu_k, cu_v, cu_g, cu_beta = (x.unsqueeze(0).contiguous() for x in (cu_q, cu_k, cu_v, cu_g, cu_beta))

        if self.use_short_conv:
            conv_state_q, conv_state_k, conv_state_v = [None, None], [None, None], [None, None]
            if last_state is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state['conv_state']

            conv_cu_seqlens = cu_seqlens
            padded = False
            if self.training:
                conv_cu_seqlens = None
            elif seq_len != 1 and (cu_seqlens[1:] - cu_seqlens[:-1]).min().item() < self.conv_size:
                padded = True
                conv_cu_seqlens, cu_q, cu_k, cu_v, pad_lengths = self.pad_for_conv(cu_seqlens, cu_q, cu_k, cu_v)

            conv_q = self.prepare_recurrent_state(
                conv_state_q[0],
                conv_cu_seqlens,
                cu_seqlen_all[0],
                reverse_indices,
                batch_size,
            )
            cu_q, conv_q_new = self.q_conv1d(
                x=cu_q,
                cache=conv_q,
                output_final_state=use_cache,
                cu_seqlens=conv_cu_seqlens,
            )
            conv_state_q[0] = self.handle_recurrent_state(
                conv_state_q[0],
                conv_q_new,
                conv_cu_seqlens,
                cu_seqlen_all[0],
                reverse_indices,
            )
            conv_k = self.prepare_recurrent_state(
                conv_state_k[0],
                conv_cu_seqlens,
                cu_seqlen_all[0],
                reverse_indices,
                batch_size,
            )
            cu_k, conv_k_new = self.k_conv1d(
                x=cu_k,
                cache=conv_k,
                output_final_state=use_cache,
                cu_seqlens=conv_cu_seqlens,
            )
            conv_state_k[0] = self.handle_recurrent_state(
                conv_state_k[0],
                conv_k_new,
                conv_cu_seqlens,
                cu_seqlen_all[0],
                reverse_indices,
            )
            conv_v = self.prepare_recurrent_state(
                conv_state_v[0],
                conv_cu_seqlens,
                cu_seqlen_all[0],
                reverse_indices,
                batch_size,
            )
            cu_v, conv_v_new = self.v_conv1d(
                x=cu_v,
                cache=conv_v,
                output_final_state=use_cache,
                cu_seqlens=conv_cu_seqlens,
            )
            conv_state_v[0] = self.handle_recurrent_state(
                conv_state_v[0],
                conv_v_new, conv_cu_seqlens,
                cu_seqlen_all[0],
                reverse_indices,
            )

            if padded:
                cu_q, cu_k, cu_v = self.unpad_after_conv(conv_cu_seqlens, cu_seqlens, cu_q, cu_k, cu_v, pad_lengths)

        else:
            # fla activates q/k/v here, after cu_q/cu_k/cu_v were split off, so it never took effect (the branch
            # never ran upstream: MomAttention refuses use_short_conv=False). Activate what the recurrence reads.
            cu_q, cu_k, cu_v = self.silu(cu_q), self.silu(cu_k), self.silu(cu_v)
            # shared_o below takes the conv states, which fla only defines in the conv branch.
            conv_state_q, conv_state_k, conv_state_v = [None, None], [None, None], [None, None]

        cu_q, cu_k, cu_v = map(lambda x: rearrange(x, 'b t (h d) -> b t h d', h=self.num_heads), (cu_q, cu_k, cu_v))

        recurrent_state = last_state['recurrent_state'] if last_state is not None else [
            None for _ in range(1 + self.shared_mem)]
        if mode == 'chunk':
            o, recurrent_state_ = chunk_gated_delta_rule(
                q=cu_q,
                k=cu_k,
                v=cu_v,
                g=cu_g,
                beta=cu_beta,
                # only the memories that received tokens are sequences of this call (fla passed every state,
                # which only holds for a first prefill)
                initial_state=self.prepare_recurrent_state(
                    recurrent_state[0], cu_seqlens, cu_seqlen_all[0], reverse_indices, batch_size),
                output_final_state=use_cache,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seqlens,
            )
            recurrent_state[0] = self.handle_recurrent_state(
                recurrent_state[0],
                recurrent_state_,
                cu_seqlens,
                cu_seqlen_all[0],
                reverse_indices,
            )

        elif mode == 'fused_recurrent':
            memories = self.prepare_recurrent_state(
                recurrent_state[0],
                cu_seqlens, cu_seqlen_all[0],
                reverse_indices, batch_size,
            )
            o, recurrent_state_ = fused_recurrent_gated_delta_rule(
                q=cu_q,
                k=cu_k,
                v=cu_v,
                g=cu_g,
                beta=cu_beta,
                initial_state=memories,
                output_final_state=use_cache,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seqlens,
            )
            recurrent_state[0] = self.handle_recurrent_state(
                recurrent_state[0],
                recurrent_state_,
                cu_seqlens,
                cu_seqlen_all[0],
                reverse_indices,
            )

        o = o.squeeze(0).contiguous()
        o = pad_input(o, indices_q, batch_size*self.num_memories, max_len)
        o = rearrange(o, '(e b) l h d -> e b l (h d)', b=batch_size)
        o = reconstruct(o, indices=indices, sorted_indices=sorted_indices, batch_size=batch_size,
                        seq_len=seq_len, topk=self.topk, routing_weights=routing_weights, mask=mask)
        o = rearrange(o, 'b l (h d) -> b l h d', h=self.num_heads)

        if self.shared_mem:
            shared_o = self.shared_o(shared_hidden_states, attention_mask, recurrent_state,
                                     use_cache, conv_state_q, conv_state_k, conv_state_v)
            o += shared_o

        update_layer_cache(
            self,
            past_key_values,
            recurrent_state=recurrent_state,
            conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
            # the number of NEW tokens: fla's q.shape[2] is the longest per-memory slice after `transform`, which
            # under-counted the cache's seen tokens and shifted the window branch's RoPE positions at decode
            offset=seq_len,
        )
        if attn_kv is not None:
            # stored after the memory update so the cache entry exists on the first call; the window branch
            # is the only reader, so it need not go through Cache.update's append/roll.
            past_key_values[self.layer_idx]["attn_state"] = attn_kv

        if self.use_output_gate:
            g = rearrange(self.g_proj(shared_hidden_states), '... (h d) -> ... h d', d=self.head_v_dim)
            o = self.o_norm(o, g)
        else:
            o = self.o_norm(o)
        o = rearrange(o, 'b t h d -> b t (h d)')
        o = self.o_proj(o if attn_o is None else o + attn_o)

        if origin_cu_seqlens is not None:
            indices, _, _ = get_unpad_data(attention_mask[:, -seq_len:])
            o = index_first_axis(rearrange(o, "b s ... -> (b s) ..."), indices).unsqueeze(0)

        return o, None, past_key_values, router_logits.view(-1, self.num_memories)
