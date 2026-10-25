from __future__ import annotations

import math

import torch
import torch.nn as nn
from einops import rearrange

from fla.layers.mamba2 import (
    Mamba2,
    apply_mask_to_padding_states,
    is_fast_path_available,
    mamba_chunk_scan_combined,
    pad_tensor_by_size,
    reshape_into_chunks,
    segment_sum,
    selective_state_update,
)
from fla.layers.utils import (
    get_layer_cache,
    get_unpad_data,
    index_first_axis,
    pad_input,
    prepare_lens_from_mask,
    unpad_input,
    update_layer_cache,
)

if not is_fast_path_available:
    # mamba_ssm is missing (its __init__ needs the compiled selective_scan_cuda), which would send training to
    # torch_inner_forward (864 GiB at 32K). Its SSD kernels are pure Triton, so use the vendored copies.
    from .mamba_ssm_triton.selective_state_update import selective_state_update
    from .mamba_ssm_triton.ssd_combined import mamba_chunk_scan_combined

    is_fast_path_available = True
from fla.modules import RotaryEmbedding

from flame.custom_models.fla_window_mix_utils import (
    cast_floating_point_parameters,
    copy_tensor_to_parameter,
    normalize_cu_seqlens,
    sliding_window_attention_from_projections,
)

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
except ImportError:  # pragma: no cover - the training image ships flash-attn
    flash_attn_func = flash_attn_varlen_func = None


class Mamba2WindowMix(Mamba2):
    def __init__(
        self,
        *args,
        attn_head_dim: int = 64,
        attn_num_heads: int | None = None,
        window_size: int | None = 4096,
        rope_theta: float = 1000000.0,
        use_window_attention: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.attn_head_dim = attn_head_dim
        # SWA width: attn_num_heads * attn_head_dim; legacy default ties it to the Mamba-2 expand width.
        self.attn_dim = self.intermediate_size if attn_num_heads is None else attn_num_heads * attn_head_dim
        if self.attn_dim % attn_head_dim != 0:
            raise ValueError(f"SWA width {self.attn_dim} must be divisible by attn_head_dim={attn_head_dim}")
        self.attn_num_heads = self.attn_dim // attn_head_dim
        self.window_size = window_size
        self.rope_theta = rope_theta
        self.use_window_attention = use_window_attention
        parameter_dtype = torch.get_default_dtype()
        if self.use_window_attention:
            self.attn_q_proj = nn.Linear(self.hidden_size, self.attn_dim, bias=self.use_bias)
            self.attn_k_proj = nn.Linear(self.hidden_size, self.attn_dim, bias=self.use_bias)
            self.attn_v_proj = nn.Linear(self.hidden_size, self.attn_dim, bias=self.use_bias)
            self.qk_scale = nn.Parameter(torch.ones(self.attn_dim, 2, dtype=parameter_dtype))
            self.qk_offset = nn.Parameter(torch.zeros(self.attn_dim, 2, dtype=parameter_dtype))
            # When the SWA width equals the SSM width the two branches are summed and share `out_proj`
            # (same as the DeltaNet window-mix); otherwise the SWA branch gets its own output projection.
            self.attn_o_proj = (
                None if self.attn_dim == self.intermediate_size
                else nn.Linear(self.attn_dim, self.hidden_size, bias=self.use_bias)
            )
            self.rotary = RotaryEmbedding(dim=self.attn_head_dim, base=self.rope_theta)
        else:
            self.attn_o_proj = None
        cast_floating_point_parameters(self, parameter_dtype)

    def reset_parameters(self) -> None:
        with torch.no_grad():
            if self.use_window_attention:
                copy_tensor_to_parameter(self.qk_scale, torch.ones(tuple(self.qk_scale.shape), dtype=torch.float32))
                copy_tensor_to_parameter(self.qk_offset, torch.zeros(tuple(self.qk_offset.shape), dtype=torch.float32))

            A = torch.empty(tuple(self.A_log.shape), dtype=torch.float32).uniform_(0, 16)
            copy_tensor_to_parameter(self.A_log, torch.log(A))
            copy_tensor_to_parameter(self.D, torch.ones(tuple(self.D.shape), dtype=torch.float32))

            dt = torch.exp(
                torch.rand(tuple(self.dt_bias.shape), dtype=torch.float32)
                * (math.log(self.time_step_max) - math.log(self.time_step_min))
                + math.log(self.time_step_min)
            ).clamp(min=1e-4)
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            copy_tensor_to_parameter(self.dt_bias, inv_dt)

            self.A_log._no_weight_decay = True
            self.D._no_weight_decay = True
            self.dt_bias._no_reinit = True

    def _get_D(self, expand_to_head_dim: bool = False) -> torch.Tensor:
        if expand_to_head_dim:
            return self.D[..., None].expand(self.D.shape[0], self.head_dim)
        return self.D

    def _window_attention(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
        cu_seqlens: torch.Tensor | None,
        max_seqlen: int | None,
    ) -> torch.Tensor:
        batch_size, q_len, _ = hidden_states.shape
        attn_states = hidden_states
        if attention_mask is not None:
            indices, cu_seqlens, max_seqlen = get_unpad_data(attention_mask[:, -q_len:])
            attn_states = index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices).unsqueeze(0)

        q = self.attn_q_proj(attn_states)
        k = self.attn_k_proj(attn_states)
        v = self.attn_v_proj(attn_states)
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
        if attention_mask is not None:
            attn_o = pad_input(attn_o.squeeze(0), indices, batch_size, q_len)
        return attn_o

    def _window_attention_cached(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_values,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Sliding-window branch with a KV cache (decoding with use_cache=True), the KDA window-mix port (5126b7cc)
        of the MoM layer's cached branch (eae84aec). Same projections, scale/offset, head split and RoPE base as the
        training path in `sliding_window_attention_from_projections`, but RoPE is applied at the absolute positions
        the cache has seen, attention runs over the cached keys plus the new tokens, and the last `window_size`
        keys/values are returned for the caller to store as this layer's `attn_state`. Without a cache the training
        path runs. Works on the padded [batch, q_len] layout, like the Mamba-2 branch of this layer."""
        if flash_attn_func is None or flash_attn_varlen_func is None:
            raise ImportError("Please install Flash Attention via `pip install flash-attn --no-build-isolation` first")
        batch_size, q_len, _ = hidden_states.shape
        q, k, v = self.attn_q_proj(hidden_states), self.attn_k_proj(hidden_states), self.attn_v_proj(hidden_states)
        key_dim, value_dim = q.shape[-1], v.shape[-1]
        scale = self.qk_scale.to(dtype=q.dtype).view(1, 1, key_dim, 2)
        offset = self.qk_offset.to(dtype=q.dtype).view(1, 1, key_dim, 2)
        q = q * scale[..., 0] + offset[..., 0]
        k = k * scale[..., 1] + offset[..., 1]
        q = rearrange(q, "... (h d) -> ... h d", d=self.attn_head_dim)
        k = rearrange(k, "... (h d) -> ... h d", d=self.attn_head_dim)
        v = rearrange(v, "... (h d) -> ... h d", d=self.attn_head_dim)
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

    def cuda_inner_forward(
        self,
        hidden_states: torch.Tensor,
        last_state: dict | None = None,
        use_cache: bool = False,
        attention_mask: torch.Tensor | None = None,
    ):
        projected_states = self.in_proj(hidden_states)
        batch_size, seq_len, _ = hidden_states.shape
        groups_time_state_size = self.n_groups * self.ssm_state_size
        d_mlp = (
            projected_states.shape[-1]
            - 2 * self.intermediate_size
            - 2 * self.n_groups * self.ssm_state_size
            - self.num_heads
        ) // 2

        if last_state is not None:
            if hidden_states.shape[1] != 1:
                raise ValueError("Mamba2 cached decoding only supports a single new token per step.")
            conv_state = last_state["conv_state"]
            ssm_state = last_state["recurrent_state"]
            _, _, gate, hidden_states_B_C, dt = projected_states.squeeze(1).split(
                [d_mlp, d_mlp, self.intermediate_size, self.conv_dim, self.num_heads],
                dim=-1,
            )
            if self.backend == "cuda":
                hidden_states_B_C = self.causal_conv1d_update(
                    hidden_states_B_C.contiguous(),
                    conv_state,
                    self.conv1d.weight.squeeze(1),
                    self.conv1d.bias,
                    self.activation,
                )
            else:
                # Same rolling [B, D, W] update the CUDA kernel does in place (see the full-sequence branch below).
                conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))
                conv_state[:, :, -1] = hidden_states_B_C.to(conv_state.dtype)
                hidden_states_B_C = torch.sum(conv_state * self.conv1d.weight.squeeze(1), dim=-1)
                if self.conv1d.bias is not None:
                    hidden_states_B_C = hidden_states_B_C + self.conv1d.bias
                hidden_states_B_C = self.act(hidden_states_B_C).to(projected_states.dtype)
            hidden_states, B, C = torch.split(
                hidden_states_B_C,
                [self.intermediate_size, groups_time_state_size, groups_time_state_size],
                dim=-1,
            )
            A = -torch.exp(self.A_log.float())
            A = A[:, None, ...][:, :, None].expand(-1, self.head_dim, self.ssm_state_size).to(dtype=torch.float32)
            dt = dt[:, :, None].expand(-1, -1, self.head_dim)
            dt_bias = self.dt_bias[:, None, ...].expand(-1, self.head_dim)
            D = self._get_D(expand_to_head_dim=True)
            B = B.view(batch_size, self.n_groups, B.shape[1] // self.n_groups)
            C = C.view(batch_size, self.n_groups, C.shape[1] // self.n_groups)
            hidden_states_reshaped = hidden_states.view(batch_size, self.num_heads, self.head_dim)
            hidden_states = selective_state_update(
                ssm_state,
                hidden_states_reshaped,
                dt,
                A,
                B,
                C,
                D,
                z=None if self.rms_norm else gate.view(batch_size, self.num_heads, self.head_dim),
                dt_bias=dt_bias,
                dt_softplus=True,
            )
            hidden_states = hidden_states.view(batch_size, self.num_heads * self.head_dim)
            if self.rms_norm:
                hidden_states = self.norm(hidden_states, gate)
            return hidden_states[:, None, ...], conv_state, ssm_state

        A = -torch.exp(self.A_log.float())
        dt_limit_kwargs = {} if self.time_step_limit == (0.0, float("inf")) else {"dt_limit": self.time_step_limit}
        _, _, gate, hidden_states_B_C, dt = projected_states.split(
            [d_mlp, d_mlp, self.intermediate_size, self.conv_dim, self.num_heads],
            dim=-1,
        )
        hidden_states_B_C = apply_mask_to_padding_states(hidden_states_B_C, attention_mask)
        new_conv_state = None
        if use_cache:
            hidden_states_B_C_transposed = hidden_states_B_C.transpose(1, 2)
            new_conv_state = nn.functional.pad(
                hidden_states_B_C_transposed,
                (self.conv_kernel_size - hidden_states_B_C_transposed.shape[-1], 0),
            )

        if self.activation not in ["silu", "swish"]:
            hidden_states_B_C = self.act(
                self.conv1d(hidden_states_B_C.transpose(1, 2))[..., :seq_len].transpose(1, 2),
            )
        else:
            if self.backend == "cuda":
                hidden_states_B_C = self.causal_conv1d_fn(
                    x=hidden_states_B_C.transpose(1, 2).contiguous(),
                    weight=self.conv1d.weight.squeeze(1),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                ).transpose(1, 2)
            else:
                # No causal_conv1d CUDA build: fla's Triton conv takes [B, T, D], not this [B, D, T] layout (it reads
                # past the weight and crashes in backward), so use the depthwise conv of torch_inner_forward.
                hidden_states_B_C = self.act(
                    self.conv1d(hidden_states_B_C.transpose(1, 2))[..., :seq_len].transpose(1, 2),
                )

        if attention_mask is not None and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
            hidden_states_B_C = (hidden_states_B_C * attention_mask[:, :, None]).to(hidden_states_B_C.dtype)
        hidden_states, B, C = torch.split(
            hidden_states_B_C,
            [self.intermediate_size, groups_time_state_size, groups_time_state_size],
            dim=-1,
        )
        scan_output, ssm_state = mamba_chunk_scan_combined(
            hidden_states.view(batch_size, seq_len, -1, self.head_dim),
            dt,
            A,
            B.view(batch_size, seq_len, self.n_groups, -1),
            C.view(batch_size, seq_len, self.n_groups, -1),
            chunk_size=self.chunk_size,
            D=self._get_D(),
            z=None if self.rms_norm else gate.view(batch_size, seq_len, self.num_heads, self.head_dim),
            seq_idx=None,
            return_final_states=True,
            dt_bias=self.dt_bias,
            dt_softplus=True,
            **dt_limit_kwargs,
        )
        scan_output = scan_output.view(batch_size, seq_len, -1)
        if self.rms_norm:
            scan_output = self.norm(scan_output, gate)
        return scan_output, new_conv_state, ssm_state

    def torch_inner_forward(
        self,
        input_states: torch.Tensor,
        last_state: dict | None = None,
        use_cache: bool = False,
        attention_mask: torch.Tensor | None = None,
    ):
        batch_size, seq_len, _ = input_states.shape
        dtype = input_states.dtype
        projected_states = self.in_proj(input_states)
        d_mlp = (
            projected_states.shape[-1]
            - 2 * self.intermediate_size
            - 2 * self.n_groups * self.ssm_state_size
            - self.num_heads
        ) // 2
        _, _, gate, hidden_states_B_C, dt = projected_states.split(
            [d_mlp, d_mlp, self.intermediate_size, self.conv_dim, self.num_heads],
            dim=-1,
        )

        if last_state is not None:
            if input_states.shape[1] != 1:
                raise ValueError("Mamba2 cached decoding only supports a single new token per step.")
            conv_state = last_state["conv_state"]
            ssm_state = last_state["recurrent_state"]
            conv_state = conv_state.roll(shifts=-1, dims=-1)
            conv_state[:, :, -1] = hidden_states_B_C[:, 0, :].to(conv_state.device)
            conv_states_for_compute = conv_state.to(device=self.conv1d.weight.device)
            hidden_states_B_C = torch.sum(
                conv_states_for_compute * self.conv1d.weight.squeeze(1),
                dim=-1,
            )
            if self.use_conv_bias:
                hidden_states_B_C = hidden_states_B_C + self.conv1d.bias
            hidden_states_B_C = self.act(hidden_states_B_C)
        else:
            hidden_states_B_C = apply_mask_to_padding_states(hidden_states_B_C, attention_mask)
            new_conv_state = None
            if use_cache:
                hidden_states_B_C_transposed = hidden_states_B_C.transpose(1, 2)
                new_conv_state = nn.functional.pad(
                    hidden_states_B_C_transposed,
                    (self.conv_kernel_size - hidden_states_B_C_transposed.shape[-1], 0),
                )
            hidden_states_B_C = self.act(self.conv1d(hidden_states_B_C.transpose(1, 2))[..., :seq_len].transpose(1, 2))

        if last_state is None and attention_mask is not None and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
            hidden_states_B_C = (hidden_states_B_C * attention_mask[:, :, None]).to(hidden_states_B_C.dtype)
        hidden_states, B, C = torch.split(
            hidden_states_B_C,
            [self.intermediate_size, self.n_groups * self.ssm_state_size, self.n_groups * self.ssm_state_size],
            dim=-1,
        )

        A = -torch.exp(self.A_log.float())
        if last_state is not None:
            cache_device = ssm_state.device
            dt = dt[:, 0, :][:, None, ...]
            dt = dt.transpose(1, 2).expand(batch_size, dt.shape[-1], self.head_dim)
            dt_bias = self.dt_bias[..., None].expand(self.dt_bias.shape[0], self.head_dim)
            dt = torch.nn.functional.softplus(dt + dt_bias.to(dt.dtype))
            dt = torch.clamp(dt, self.time_step_limit[0], self.time_step_limit[1])
            A = A[..., None, None].expand(self.num_heads, self.head_dim, self.ssm_state_size).to(dtype=torch.float32)
            dA = (torch.exp(dt[..., None] * A)).to(device=cache_device)
            B = B.reshape(batch_size, self.n_groups, -1)[..., None, :]
            B = B.expand(batch_size, self.n_groups, self.num_heads // self.n_groups, B.shape[-1]).contiguous()
            B = B.reshape(batch_size, -1, B.shape[-1])
            dB = dt[..., None] * B[..., None, :]
            hidden_states = hidden_states.reshape(batch_size, -1, self.head_dim)
            dBx = (dB * hidden_states[..., None]).to(device=cache_device)
            ssm_state = ssm_state * dA + dBx
            C = C.reshape(batch_size, self.n_groups, -1)[..., None, :]
            C = C.expand(batch_size, self.n_groups, self.num_heads // self.n_groups, C.shape[-1]).contiguous()
            C = C.reshape(batch_size, -1, C.shape[-1])
            ssm_states_for_compute = ssm_state.to(device=C.device, dtype=C.dtype)
            ssm_states_reshaped = ssm_states_for_compute.view(
                batch_size * self.num_heads,
                self.head_dim,
                self.ssm_state_size,
            )
            C_reshaped = C.view(batch_size * self.num_heads, self.ssm_state_size, 1)
            y = torch.bmm(ssm_states_reshaped, C_reshaped)
            y = y.view(batch_size, self.num_heads, self.head_dim)
            y = (y + hidden_states * self._get_D(expand_to_head_dim=True)).to(y.dtype)
            y = y.reshape(batch_size, -1)[:, None, ...]
            scan_output = self.norm(y, gate) if self.rms_norm else y * self.act(gate)
            return scan_output.to(dtype), conv_state, ssm_state

        dt = nn.functional.softplus(dt + self.dt_bias)
        dt = torch.clamp(dt, self.time_step_limit[0], self.time_step_limit[1])
        hidden_states = hidden_states.reshape(batch_size, seq_len, -1, self.head_dim).float()
        B = B.reshape(batch_size, seq_len, -1, self.ssm_state_size).float()
        C = C.reshape(batch_size, seq_len, -1, self.ssm_state_size).float()
        B = B.repeat(1, 1, self.num_heads // self.n_groups, 1)
        C = C.repeat(1, 1, self.num_heads // self.n_groups, 1)
        pad_size = (self.chunk_size - seq_len % self.chunk_size) % self.chunk_size
        D_residual = self._get_D(expand_to_head_dim=True) * pad_tensor_by_size(hidden_states, pad_size)
        hidden_states = hidden_states * dt[..., None]
        A = A.to(hidden_states.dtype) * dt
        hidden_states, A, B, C = [reshape_into_chunks(t, pad_size, self.chunk_size) for t in (hidden_states, A, B, C)]
        A = A.permute(0, 3, 1, 2)
        A_cumsum = torch.cumsum(A, dim=-1)
        L = torch.exp(segment_sum(A))
        G_intermediate = C[:, :, :, None, :, :] * B[:, :, None, :, :, :]
        G = G_intermediate.sum(dim=-1)
        M_intermediate = G[..., None] * L.permute(0, 2, 3, 4, 1)[..., None]
        M = M_intermediate.sum(dim=-1)
        Y_diag = (M[..., None] * hidden_states[:, :, None]).sum(dim=3)
        decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
        B_decay = B * decay_states.permute(0, -2, -1, 1)[..., None]
        states = (B_decay[..., None, :] * hidden_states[..., None]).sum(dim=2)
        previous_states = torch.zeros_like(states[:, :1])
        states = torch.cat([previous_states, states], dim=1)
        decay_chunk = torch.exp(segment_sum(nn.functional.pad(A_cumsum[:, :, :, -1], (1, 0))))
        decay_chunk = decay_chunk.transpose(1, 3)
        new_states = (decay_chunk[..., None, None] * states[:, :, None, ...]).sum(dim=1)
        states, ssm_state = new_states[:, :-1], new_states[:, -1]
        state_decay_out = torch.exp(A_cumsum)
        C_times_states = C[..., None, :] * states[:, :, None, ...]
        state_decay_out_permuted = state_decay_out.permute(0, 2, 3, 1)
        Y_off = C_times_states.sum(-1) * state_decay_out_permuted[..., None]
        y = Y_diag + Y_off
        y = y.reshape(batch_size, -1, self.num_heads, self.head_dim)
        y = y + D_residual
        if pad_size > 0:
            y = y[:, :seq_len, :, :]
        y = y.reshape(batch_size, seq_len, -1)
        scan_output = self.norm(y, gate) if self.rms_norm else y * self.act(gate)
        return scan_output.to(dtype), new_conv_state if use_cache else None, ssm_state

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs,
    ):
        last_state = get_layer_cache(self, past_key_values)
        cu_seqlens = normalize_cu_seqlens(kwargs.get("cu_seqlens"))
        max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item()) if cu_seqlens is not None else hidden_states.shape[1]
        attn_o, attn_kv = None, None
        if self.use_window_attention:
            if past_key_values is None:
                attn_o = self._window_attention(
                    hidden_states,
                    attention_mask=attention_mask,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                )
            else:
                # cached decoding (use_cache=True, 2026-09-24): the window branch keeps the last `window_size`
                # keys/values per layer next to the Mamba-2 conv/SSM states
                attn_o, attn_kv = self._window_attention_cached(hidden_states, attention_mask, past_key_values)

        if is_fast_path_available and "cuda" in self.in_proj.weight.device.type:
            inner_states, conv_state, ssm_state = self.cuda_inner_forward(hidden_states, last_state, use_cache, attention_mask)
        else:
            dtype = hidden_states.dtype
            if last_state is None and attention_mask is not None and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
                hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)
            inner_states, conv_state, ssm_state = self.torch_inner_forward(hidden_states, last_state, use_cache, attention_mask)

        if attn_o is None:
            output = self.out_proj(inner_states)
        elif self.attn_o_proj is None:
            output = self.out_proj(inner_states + attn_o)
        else:
            output = self.out_proj(inner_states) + self.attn_o_proj(attn_o)

        update_layer_cache(
            self,
            past_key_values,
            recurrent_state=ssm_state,
            conv_state=conv_state,
            offset=hidden_states.shape[1],
        )
        if attn_kv is not None:
            # stored after the Mamba-2 update so the cache entry exists on the first call; the window branch is the
            # only reader, so it need not go through Cache.update's append/roll
            past_key_values[self.layer_idx]["attn_state"] = attn_kv
        return output, None, past_key_values
