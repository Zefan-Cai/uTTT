from __future__ import annotations

import math
import os
import warnings
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from .configuration_recurrent_lact import RecurrentLactRefMoeV3Config
from .moe_routing import create_router_mask_sizes_probs
from . import ref_base as base

try:
    from .lact_moe_triton.triton_permute import (
        permute_kv_and_lrs as _permute_kv_and_lrs,
        permute_with_expert_mask as _permute_with_expert_mask,
        unpermute_and_merge_with_probs as _unpermute_and_merge_with_probs,
    )
    from .lact_moe_triton.lact_fw_grad import (
        grouped_lact_swiglu_ffn_fast_weight_grads as _grouped_lact_swiglu_ffn_fast_weight_grads,
    )
    from .lact_moe_triton.lact_swiglu_ffn import (
        grouped_swiglu_ffn_fwd as _grouped_swiglu_ffn_fwd,
    )
except ImportError:
    _permute_kv_and_lrs = None
    _permute_with_expert_mask = None
    _unpermute_and_merge_with_probs = None
    _grouped_lact_swiglu_ffn_fast_weight_grads = None
    _grouped_swiglu_ffn_fwd = None


@dataclass
class MoeModelOutput(BaseModelOutputWithPast):
    lb_loss: Optional[torch.Tensor] = None
    expert_freq_min: Optional[float] = None
    expert_freq_max: Optional[float] = None
    expert_freq_std: Optional[float] = None


@dataclass
class CausalLMOutputWithMoeStats(CausalLMOutputWithPast):
    ce_loss: Optional[torch.Tensor] = None
    lb_loss: Optional[torch.Tensor] = None
    expert_freq_min: Optional[float] = None
    expert_freq_max: Optional[float] = None
    expert_freq_std: Optional[float] = None


_MOE_V3_EXPERT_CHUNK_SIZE = max(
    1,
    int(
        os.environ.get(
            "LACT_MOE_V3_EXPERT_CHUNK_SIZE",
            os.environ.get(
                "LACT_MOE_V2_EXPERT_CHUNK_SIZE",
                os.environ.get("LACT_GLOBAL_MOE_EXPERT_CHUNK_SIZE", "100"),
            ),
        ),
    ),
)
_ROUTER_TOKEN_CHUNK_SIZE = max(
    0,
    int(os.environ.get("LACT_ROUTER_TOKEN_CHUNK_SIZE", "0")),
)
_QK_NORM_TOKEN_CHUNK_SIZE = max(
    0,
    int(os.environ.get("LACT_QK_NORM_TOKEN_CHUNK_SIZE", "0")),
)
_CHECKPOINT_APPLY_UPDATE = os.environ.get("LACT_CHECKPOINT_APPLY_UPDATE", "0") == "1"
_MOE_V3_UPDATE_GRAD_DTYPE = os.environ.get(
    "LACT_MOE_V3_UPDATE_GRAD_DTYPE",
    os.environ.get(
        "LACT_MOE_V2_UPDATE_GRAD_DTYPE",
        os.environ.get("LACT_STREAM_SHARED_UPDATE_GRAD_DTYPE", "bf16"),
    ),
).lower()


def _moe_v3_update_grad_dtype() -> torch.dtype:
    if _MOE_V3_UPDATE_GRAD_DTYPE in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if _MOE_V3_UPDATE_GRAD_DTYPE in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(
        "LACT_MOE_V3_UPDATE_GRAD_DTYPE must be 'bf16' or 'fp32', "
        f"got {_MOE_V3_UPDATE_GRAD_DTYPE!r}",
    )


def _zero_lb_accumulators(
    num_layers: int,
) -> tuple[list[torch.Tensor | None], list[torch.Tensor | None]]:
    return [None] * num_layers, [None] * num_layers


def _accumulate_lb_info(
    probs_per_layer: list[torch.Tensor | None],
    freqs_per_layer: list[torch.Tensor | None],
    layer_idx: int,
    lb_info: tuple[torch.Tensor, torch.Tensor] | None,
) -> None:
    if lb_info is None:
        return
    probs_chunk, freqs_chunk = lb_info
    p = probs_chunk.float()
    f = freqs_chunk.float().detach()
    probs_per_layer[layer_idx] = (
        p if probs_per_layer[layer_idx] is None else probs_per_layer[layer_idx] + p
    )
    freqs_per_layer[layer_idx] = (
        f if freqs_per_layer[layer_idx] is None else freqs_per_layer[layer_idx] + f
    )


class FastWeightMLPSubNNMoeV3(base.FastWeightMLPSubNN):
    """Per-layer MoE fast-weight memory using the grouped flash FFN path."""

    def __init__(self, config: RecurrentLactRefMoeV3Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.memory_num_experts = config.memory_num_experts
        self.memory_num_active_experts = config.memory_num_active_experts
        self.memory_router_alpha = config.memory_router_alpha
        self.memory_router_use_sigmoid = config.memory_router_use_sigmoid
        self.memory_router_from_v = config.memory_router_from_v
        self.use_grouped_forward = os.environ.get("LACT_GLOBAL_MOE_GROUPED_FWD", "1") == "1"

        if self.w0_w2_low_rank > 0:
            self.w0 = base.LowRankFastWeight(
                self.memory_num_experts,
                self.inter_dim,
                self.head_dim,
                rank=self.w0_w2_low_rank,
                init_gain=self.fw_init_gain,
                add_identity=True,
            )
            self.w2 = base.LowRankFastWeight(
                self.memory_num_experts,
                self.inter_dim,
                self.head_dim,
                rank=self.w0_w2_low_rank,
                init_gain=self.fw_init_gain,
                add_identity=True,
            )
        else:
            self.w0 = nn.Parameter(torch.empty(self.memory_num_experts, self.inter_dim, self.head_dim))
            self.w2 = nn.Parameter(torch.empty(self.memory_num_experts, self.inter_dim, self.head_dim))
        self.w1 = nn.Parameter(torch.empty(self.memory_num_experts, self.head_dim, self.inter_dim))
        self.router_proj_weights = nn.Parameter(torch.empty(1, self.memory_num_experts, self.head_dim))

        if self.use_momentum:
            if hasattr(self, "momentum_proj"):
                del self.momentum_proj
            self.momentum_proj = nn.Sequential(
                nn.Linear(self.hidden_size, 1),
                nn.Sigmoid(),
            )

    def init_fast_weights(self, batch_size: int = 1):
        base_w0 = self.w0() if self.w0_w2_low_rank > 0 else self.w0
        base_w2 = self.w2() if self.w0_w2_low_rank > 0 else self.w2
        master_weight = (
            base_w0.unsqueeze(0).repeat(batch_size, 1, 1, 1),
            self.w1.unsqueeze(0).repeat(batch_size, 1, 1, 1),
            base_w2.unsqueeze(0).repeat(batch_size, 1, 1, 1),
        )
        weight = tuple(w.to(torch.bfloat16) for w in master_weight)
        momentum_buf = None
        if self.use_momentum:
            momentum_buf = (
                torch.zeros_like(master_weight[0]),
                torch.zeros_like(master_weight[1]),
                torch.zeros_like(master_weight[2]),
            )
        return weight, master_weight, momentum_buf

    def reset_parameters(self):
        if self.qk_rescale:
            nn.init.ones_(self.qk_scale)
            nn.init.zeros_(self.qk_offset)
        if self.w0_w2_low_rank > 0:
            self.w0.reset_parameters()
            self.w2.reset_parameters()
        else:
            nn.init.normal_(self.w0, mean=0.0, std=1.0 / math.sqrt(self.head_dim))
            nn.init.normal_(self.w2, mean=0.0, std=1.0 / math.sqrt(self.head_dim))
        nn.init.normal_(self.w1, mean=0.0, std=1.0 / math.sqrt(self.inter_dim))
        nn.init.normal_(self.router_proj_weights, mean=0.0, std=1.0 / math.sqrt(self.head_dim))

    def _reshape_head_tokens(self, x: torch.Tensor) -> torch.Tensor:
        return base.rearrange(
            x,
            "b l (h hd) -> b (l h) hd",
            h=self.fw_num_heads,
        ).contiguous()

    def _merge_head_tokens(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        return base.rearrange(
            x,
            "b (l h) hd -> b l (h hd)",
            h=self.fw_num_heads,
            l=seq_len,
        )

    def _qk_norm_chunked(self, x: torch.Tensor) -> torch.Tensor:
        if _QK_NORM_TOKEN_CHUNK_SIZE and x.shape[1] > _QK_NORM_TOKEN_CHUNK_SIZE:
            return torch.cat(
                [
                    self.qk_norm_fn(x[:, start : start + _QK_NORM_TOKEN_CHUNK_SIZE])
                    for start in range(0, x.shape[1], _QK_NORM_TOKEN_CHUNK_SIZE)
                ],
                dim=1,
            )
        return self.qk_norm_fn(x)

    def _route_tokens(
        self,
        x: torch.Tensor,
        router_proj_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        with torch.autocast(device_type=x.device.type, enabled=False):
            if router_proj_weights is None:
                router_proj_weights = self.router_proj_weights
            router_proj = router_proj_weights.float()
            if _ROUTER_TOKEN_CHUNK_SIZE and x.shape[1] > _ROUTER_TOKEN_CHUNK_SIZE:
                logits = torch.cat(
                    [
                        torch.matmul(
                            router_proj,
                            x[:, start : start + _ROUTER_TOKEN_CHUNK_SIZE].transpose(1, 2).float(),
                        )
                        for start in range(0, x.shape[1], _ROUTER_TOKEN_CHUNK_SIZE)
                    ],
                    dim=-1,
                )
            else:
                logits = torch.matmul(router_proj, x.transpose(1, 2).float())

        expert_mask, group_sizes, router_probs, _ = create_router_mask_sizes_probs(
            logits,
            topk=self.memory_num_active_experts,
            alpha=self.memory_router_alpha,
            use_sigmoid=self.memory_router_use_sigmoid,
        )
        lb_info = (router_probs.sum(dim=-1), group_sizes)
        return expert_mask, group_sizes, router_probs, lb_info

    def _apply_moe_fast_weights_slow(
        self,
        x: torch.Tensor,
        w0: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        router_proj_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        seq_len = x.shape[1]
        x = self._reshape_head_tokens(x)
        top1_dispatch = self.memory_num_active_experts == 1
        out = torch.empty_like(x) if top1_dispatch else torch.zeros_like(x, dtype=torch.float32)
        expert_mask, _, router_probs, lb_info = self._route_tokens(x, router_proj_weights)

        for batch_idx in range(x.shape[0]):
            for expert_idx in range(self.memory_num_experts):
                token_mask = expert_mask[batch_idx, expert_idx]
                if not token_mask.any().item():
                    continue
                x_e = x[batch_idx, token_mask].to(w0.dtype)
                w0_e = w0[batch_idx, expert_idx]
                w1_e = w1[batch_idx, expert_idx]
                w2_e = w2[batch_idx, expert_idx]
                gate = torch.matmul(x_e, w0_e.transpose(0, 1))
                up = torch.matmul(x_e, w2_e.transpose(0, 1))
                hidden = F.silu(gate, inplace=False) * up
                y = torch.matmul(hidden, w1_e.transpose(0, 1))
                probs = router_probs[batch_idx, expert_idx, token_mask].unsqueeze(-1)
                weighted = y.to(out.dtype) * probs.to(out.dtype)
                if top1_dispatch:
                    out[batch_idx, token_mask] = weighted
                else:
                    out[batch_idx, token_mask] += weighted

        return self._merge_head_tokens(out.to(x.dtype), seq_len), lb_info

    def _apply_moe_fast_weights_grouped(
        self,
        x: torch.Tensor,
        w0: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        router_proj_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if (
            _permute_with_expert_mask is None
            or _unpermute_and_merge_with_probs is None
            or _grouped_swiglu_ffn_fwd is None
        ):
            return self._apply_moe_fast_weights_slow(x, w0, w1, w2, router_proj_weights)

        seq_len = x.shape[1]
        out_dtype = x.dtype
        x = self._reshape_head_tokens(x)
        target_dtype = w0.dtype
        expert_mask, group_sizes_raw, router_probs, lb_info = self._route_tokens(x, router_proj_weights)
        group_sizes = group_sizes_raw.to(torch.int32).contiguous()

        x_perm, _, row_id_map = _permute_with_expert_mask(
            x.to(target_dtype).contiguous(),
            expert_mask,
            router_probs,
            self.memory_num_active_experts,
        )
        w0_w2 = torch.cat([w0, w2], dim=2).contiguous()
        y_perm = _grouped_swiglu_ffn_fwd(
            w0_w2,
            w1.contiguous(),
            x_perm.contiguous(),
            group_sizes,
        )
        out = _unpermute_and_merge_with_probs(y_perm, row_id_map, router_probs)
        return self._merge_head_tokens(out.to(out_dtype), seq_len), lb_info

    def _apply_moe_fast_weights(
        self,
        x: torch.Tensor,
        w0: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        router_proj_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if self.use_grouped_forward:
            return self._apply_moe_fast_weights_grouped(x, w0, w1, w2, router_proj_weights)
        return self._apply_moe_fast_weights_slow(x, w0, w1, w2, router_proj_weights)

    def forward(
        self,
        q: torch.Tensor,
        fastw: tuple[torch.Tensor, ...],
        router_proj_weights: torch.Tensor | None = None,
        seqlen_offset: int = 0,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        w0, w1, w2 = fastw
        if self.qk_rescale:
            q, _ = self._rescale_qk(q, q)
        if self.qkv_silu:
            q = F.silu(q)
        if self.qk_norm:
            q = base.rearrange(q, "b l (h d) -> (b h) l d", h=self.fw_num_heads)
            q = self._qk_norm_chunked(q)
            q = base.rearrange(q, "(b h) l d -> b l (h d)", h=self.fw_num_heads)
        if self.ttt_rotary is not None:
            q = base.rearrange(q, "b l (nh d) -> b l nh d", nh=self.n_rope_heads)
            q = q.to(self.ttt_rotary._cos_cached.dtype)
            _, q = self.ttt_rotary(None, q, seqlen_offset=seqlen_offset)
            q = base.rearrange(q, "b l nh d -> b l (nh d)")

        out, lb_info = self._apply_moe_fast_weights(q, w0, w1, w2, router_proj_weights)
        if not self.learnable_ttt_scale:
            out = base.rearrange(out, "b s (n_h d) -> (b n_h) s d", n_h=self.fw_num_heads)
            out = self.ttt_norm(out)
            out = base.rearrange(out, "(b n_h) s d -> b s (n_h d)", n_h=self.fw_num_heads)
        if self.enable_memory_output_proj:
            out = self.output_proj(out)
        return out, lb_info

    def compute_raw_grad(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        lr: torch.Tensor,
        fastw: tuple[torch.Tensor, ...],
        router_proj_weights: torch.Tensor | None = None,
        pre_vi: torch.Tensor | None = None,
        seqlen_offset: int = 0,
        grad_dtype: torch.dtype = torch.float32,
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        torch.Tensor | None,
        tuple[torch.Tensor, torch.Tensor],
    ]:
        w0, w1, w2 = fastw
        batch_size, seq_len, _ = k.shape

        if self.qk_rescale:
            _, k = self._rescale_qk(k, k)
        if self.qkv_silu:
            k = F.silu(k)
            v = F.silu(v)
        if self.v_norm:
            v = self.norm_fn(v)
        if self.qk_norm:
            k = base.rearrange(k, "b l (h d) -> (b h) l d", h=self.fw_num_heads)
            k = self._qk_norm_chunked(k)
            k = base.rearrange(k, "(b h) l d -> b l (h d)", h=self.fw_num_heads)
        if self.ttt_rotary is not None:
            k = base.rearrange(k, "b l (nh d) -> b l nh d", nh=self.n_rope_heads)
            k = k.to(self.ttt_rotary._cos_cached.dtype)
            _, k = self.ttt_rotary(None, k, seqlen_offset=seqlen_offset)
            k = base.rearrange(k, "b l nh d -> b l (nh d)")

        k_tokens = self._reshape_head_tokens(k)
        v_tokens = self._reshape_head_tokens(v)

        if lr.shape[2] == 3:
            lr_mh = lr.unsqueeze(2).expand(-1, -1, self.fw_num_heads, -1)
        else:
            lr_mh = lr.view(batch_size, seq_len, self.fw_num_heads, 3)
        lr_tokens = base.rearrange(lr_mh, "b l h c -> b (l h) c").contiguous()

        router_input = v_tokens if self.memory_router_from_v else k_tokens
        expert_mask, _, router_probs, update_lb_info = self._route_tokens(
            router_input,
            router_proj_weights,
        )

        if (
            k_tokens.device.type == "cuda"
            and _permute_kv_and_lrs is not None
            and _grouped_lact_swiglu_ffn_fast_weight_grads is not None
        ):
            group_sizes = expert_mask.sum(dim=-1).to(torch.int32).contiguous()
            lr0 = lr_tokens[..., 0].contiguous()
            lr1 = lr_tokens[..., 1].contiguous()
            lr2 = lr_tokens[..., 2].contiguous()
            k_perm, v_perm, lr0_perm, lr1_perm, lr2_perm = _permute_kv_and_lrs(
                k_tokens.to(w0.dtype).contiguous(),
                v_tokens.to(w1.dtype).contiguous(),
                lr0,
                lr1,
                lr2,
                expert_mask,
                router_probs,
                self.memory_num_active_experts,
            )
            w0_w2 = torch.cat([w0, w2], dim=2).contiguous()
            w0_w2_grad_raw, w1_grad_raw = _grouped_lact_swiglu_ffn_fast_weight_grads(
                w0_w2,
                w1.contiguous(),
                k_perm.contiguous(),
                v_perm.contiguous(),
                lr0_perm.contiguous(),
                lr1_perm.contiguous(),
                lr2_perm.contiguous(),
                group_sizes,
            )
            w0_grad_raw, w2_grad_raw = w0_w2_grad_raw.split(self.inter_dim, dim=2)
            w0_grad_raw = w0_grad_raw.to(grad_dtype).contiguous()
            w1_grad_raw = w1_grad_raw.to(grad_dtype).contiguous()
            w2_grad_raw = w2_grad_raw.to(grad_dtype).contiguous()

            m_coeff = None
            if self.use_momentum and pre_vi is not None:
                m_coeff = self.momentum_proj(
                    pre_vi.to(self.momentum_proj[0].weight.dtype),
                ).mean(dim=1).view(batch_size, -1, 1, 1)

            return (w0_grad_raw, w1_grad_raw, w2_grad_raw), m_coeff, update_lb_info

        w0_grad_raw = torch.zeros(w0.shape, dtype=grad_dtype, device=w0.device)
        w1_grad_raw = torch.zeros(w1.shape, dtype=grad_dtype, device=w1.device)
        w2_grad_raw = torch.zeros(w2.shape, dtype=grad_dtype, device=w2.device)

        for batch_idx in range(batch_size):
            for expert_idx in range(self.memory_num_experts):
                token_mask = expert_mask[batch_idx, expert_idx]
                if not token_mask.any().item():
                    continue

                k_e = k_tokens[batch_idx, token_mask]
                v_e = v_tokens[batch_idx, token_mask]
                lr_e = lr_tokens[batch_idx, token_mask]
                probs_e = router_probs[batch_idx, expert_idx, token_mask].unsqueeze(-1)

                w0_e = w0[batch_idx, expert_idx]
                w1_e = w1[batch_idx, expert_idx]
                w2_e = w2[batch_idx, expert_idx]
                k_e = k_e.to(w0_e.dtype)
                v_e = (v_e * probs_e.to(v_e.dtype)).to(w1_e.dtype)

                lr0_e = lr_e[:, 0:1].to(k_e.dtype)
                lr1_e = lr_e[:, 1:2].to(k_e.dtype)
                lr2_e = lr_e[:, 2:3].to(k_e.dtype)

                gate = torch.matmul(k_e, w0_e.transpose(0, 1))
                up = torch.matmul(k_e, w2_e.transpose(0, 1))
                hidden = F.silu(gate, inplace=False) * up

                dhidden = torch.matmul(v_e, w1_e)
                dhidden_before_mul = dhidden * F.silu(gate, inplace=False)
                dgate = dhidden * up
                dgate_before_act = base.silu_backprop(dgate, gate)

                w1_grad_raw[batch_idx, expert_idx] = torch.matmul(
                    v_e.transpose(0, 1),
                    hidden * lr1_e,
                ).to(grad_dtype)
                w0_grad_raw[batch_idx, expert_idx] = torch.matmul(
                    dgate_before_act.transpose(0, 1),
                    k_e * lr0_e,
                ).to(grad_dtype)
                w2_grad_raw[batch_idx, expert_idx] = torch.matmul(
                    dhidden_before_mul.transpose(0, 1),
                    k_e * lr2_e,
                ).to(grad_dtype)

        m_coeff = None
        if self.use_momentum and pre_vi is not None:
            m_coeff = self.momentum_proj(
                pre_vi.to(self.momentum_proj[0].weight.dtype),
            ).mean(dim=1).view(batch_size, -1, 1, 1)

        return (w0_grad_raw, w1_grad_raw, w2_grad_raw), m_coeff, update_lb_info

    def _apply_update(
        self,
        grads: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        masterw: tuple[torch.Tensor, ...],
        momentum_buf: tuple[torch.Tensor, ...] | None,
        avg_m_coeff: torch.Tensor | None,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], tuple[torch.Tensor, ...] | None]:
        w0_grad_raw, w1_grad_raw, w2_grad_raw = grads
        momentum_inputs = momentum_buf if momentum_buf is not None else (None, None, None)
        update_momentum = (
            self.use_momentum
            and momentum_buf is not None
            and avg_m_coeff is not None
        )

        def _apply_one_weight(
            grad_raw: torch.Tensor,
            master: torch.Tensor,
            momentum: torch.Tensor | None,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
            chunk_size = _MOE_V3_EXPERT_CHUNK_SIZE
            n_group = grad_raw.shape[1]
            new_master = torch.empty_like(master)
            new_weight = torch.empty_like(master, dtype=torch.bfloat16)
            new_momentum = torch.empty_like(momentum) if update_momentum else None

            for start in range(0, n_group, chunk_size):
                end = min(start + chunk_size, n_group)
                grad_chunk = grad_raw[:, start:end]
                master_chunk = master[:, start:end]
                old_norm = master_chunk.norm(dim=3, keepdim=True)

                if update_momentum:
                    grad_chunk = grad_chunk + momentum[:, start:end] * avg_m_coeff
                    new_momentum[:, start:end] = grad_chunk

                if self.use_moun:
                    grad_flat = base.rearrange(grad_chunk, "b g d1 d2 -> (b g) d1 d2")
                    grad_ns = base.zeropower_via_newtonschulz5(grad_flat, 5)
                    grad_chunk = base.rearrange(
                        grad_ns,
                        "(b g) d1 d2 -> b g d1 d2",
                        b=grad_raw.shape[0],
                        g=end - start,
                    )

                updated_master = master_chunk + grad_chunk
                updated_master = (
                    updated_master
                    / (updated_master.norm(dim=3, keepdim=True) + 1e-5)
                    * old_norm
                )
                new_master[:, start:end] = updated_master
                new_weight[:, start:end] = updated_master.to(torch.bfloat16)

            return new_weight, new_master, new_momentum

        w0_weight, w0_master, w0_momentum = _apply_one_weight(
            w0_grad_raw,
            masterw[0],
            momentum_inputs[0],
        )
        w1_weight, w1_master, w1_momentum = _apply_one_weight(
            w1_grad_raw,
            masterw[1],
            momentum_inputs[1],
        )
        w2_weight, w2_master, w2_momentum = _apply_one_weight(
            w2_grad_raw,
            masterw[2],
            momentum_inputs[2],
        )

        if update_momentum:
            momentum_buf = (w0_momentum, w1_momentum, w2_momentum)

        return (w0_weight, w1_weight, w2_weight), (w0_master, w1_master, w2_master), momentum_buf

    def apply_update(
        self,
        grads: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        masterw: tuple[torch.Tensor, ...],
        momentum_buf: tuple[torch.Tensor, ...] | None,
        avg_m_coeff: torch.Tensor | None,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], tuple[torch.Tensor, ...] | None]:
        if _CHECKPOINT_APPLY_UPDATE:
            return checkpoint(
                self._apply_update,
                grads,
                masterw,
                momentum_buf,
                avg_m_coeff,
                preserve_rng_state=False,
                use_reentrant=False,
            )
        return self._apply_update(grads, masterw, momentum_buf, avg_m_coeff)

    def update(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        lr: torch.Tensor,
        fastw: tuple[torch.Tensor, ...],
        masterw: tuple[torch.Tensor, ...],
        momentum_buf: tuple[torch.Tensor, ...] | None = None,
        pre_vi: torch.Tensor | None = None,
        seqlen_offset: int = 0,
    ) -> tuple[
        tuple[torch.Tensor, ...],
        tuple[torch.Tensor, ...],
        tuple[torch.Tensor, ...] | None,
        tuple[torch.Tensor, torch.Tensor],
    ]:
        grads, m_coeff, update_lb_info = self.compute_raw_grad(
            k,
            v,
            lr,
            fastw,
            pre_vi=pre_vi,
            seqlen_offset=seqlen_offset,
            grad_dtype=_moe_v3_update_grad_dtype(),
        )
        fastw, masterw, momentum_buf = self.apply_update(
            grads,
            masterw,
            momentum_buf,
            m_coeff,
        )
        return fastw, masterw, momentum_buf, update_lb_info


class RecurrentLactRefMoeV3Block(base.RecurrentLactBlock):
    def __init__(self, config: RecurrentLactRefMoeV3Config, layer_idx: int):
        super().__init__(config, layer_idx)
        if config.use_memory_block:
            self.memory = FastWeightMLPSubNNMoeV3(config, layer_idx)

    def init_fast_weight(self, batch_size: int = 1):
        if self.memory is not None:
            return self.memory.init_fast_weights(batch_size=batch_size)
        return None, None, None

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        fastw: tuple[torch.Tensor, ...] | None = None,
        seqlen_offset: int = 0,
        **kwargs: Any,
    ):
        masterw = None
        momentum_buf = None
        x_memory_in = None
        if self.memory is not None and fastw is None:
            fastw, masterw, momentum_buf = self.memory.init_fast_weights(
                batch_size=hidden_states.size(0),
            )

        if self.config.residual_style == "parallel":
            residual = hidden_states
            x_attn_in = self.attn_norm(hidden_states)
            attn_out, attn_k, attn_v, attn_q = self.attn(
                hidden_states=x_attn_in,
                kv_cache=kv_cache,
            )

            if self.memory is None or fastw is None:
                raise ValueError("Memory block is not enabled")
            x_memory_in = x_attn_in
            if self.config.memory_kv_mode == "reuse_kv":
                q = attn_q
            elif self.config.memory_kv_proj == "reuse_attn_kv_proj":
                q = self.attention_toq(x_memory_in)
            else:
                q = self.memory_toq(x_memory_in)

            memory_out, apply_lb_info = self.memory(
                q,
                fastw,
                seqlen_offset=seqlen_offset,
            )
            if self.memory.learnable_ttt_scale:
                memory_out = self.memory.apply_ttt_scale(memory_out, x_attn_in)

            hidden_states = memory_out + attn_out
            if self.enable_attention_memory_output_proj:
                hidden_states = self.memory_output_proj(hidden_states)
                hidden_states.add_(residual)
            else:
                hidden_states = residual + hidden_states

        elif self.config.residual_style == "sequential":
            residual = hidden_states
            x_attn_in = self.attn_norm(hidden_states)
            attn_out, attn_k, attn_v, attn_q = self.attn(
                hidden_states=x_attn_in,
                kv_cache=kv_cache,
            )
            hidden_states = residual + attn_out

            if self.memory is None or fastw is None:
                raise ValueError("Memory block is not enabled")
            residual = hidden_states
            x_memory_in = self.memory_norm(hidden_states)
            if self.config.memory_kv_mode == "reuse_kv":
                q = attn_q
            elif self.config.memory_kv_proj == "reuse_attn_kv_proj":
                q = self.attention_toq(x_memory_in)
            else:
                q = self.memory_toq(x_memory_in)

            memory_out, apply_lb_info = self.memory(
                q,
                fastw,
                seqlen_offset=seqlen_offset,
            )
            if self.memory.learnable_ttt_scale:
                memory_out = self.memory.apply_ttt_scale(memory_out, x_memory_in)

            hidden_states = memory_out
            if self.enable_attention_memory_output_proj:
                hidden_states = self.memory_output_proj(hidden_states)
                hidden_states.add_(residual)
            else:
                hidden_states = residual + hidden_states
        else:
            raise ValueError(f"Invalid residual_style: {self.config.residual_style}")

        residual = hidden_states
        hidden_states = self.mlp_norm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states, **kwargs)

        return hidden_states, x_attn_in, x_memory_in, attn_k, attn_v, apply_lb_info, fastw, masterw, momentum_buf

    def _update_fast_weight(
        self,
        pre_vi: torch.Tensor,
        fastw: tuple[torch.Tensor, ...],
        masterw: tuple[torch.Tensor, ...],
        momentum_buf: tuple[torch.Tensor, ...] | None = None,
        ki: torch.Tensor | None = None,
        vi: torch.Tensor | None = None,
        seqlen_offset: int = 0,
    ):
        if self.memory is None:
            return fastw, masterw, momentum_buf, None
        if ki is None or vi is None:
            raise ValueError("`ki` and `vi` are required for MoE v2 memory update.")
        lri = self.to_lr(pre_vi)
        lri = F.softplus(lri.float() + self.base_lr_inv)
        return self.memory.update(
            ki,
            vi,
            lri,
            fastw,
            masterw,
            momentum_buf=momentum_buf,
            pre_vi=pre_vi,
            seqlen_offset=seqlen_offset,
        )

    def update_fast_weight(self, *args, **kwargs):
        return checkpoint(
            self._update_fast_weight,
            *args,
            **kwargs,
            preserve_rng_state=False,
            use_reentrant=False,
        )


class RecurrentLactRefMoeV3Model(base.RecurrentLactRefModel):
    config_class = RecurrentLactRefMoeV3Config
    _no_split_modules = ["RecurrentLactRefMoeV3Block"]

    def __init__(self, config: RecurrentLactRefMoeV3Config):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [
                RecurrentLactRefMoeV3Block(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        num_kv_heads = config.num_kv_heads if config.num_kv_heads is not None else config.num_heads
        fw_num_heads = config.fw_num_heads if config.fw_num_heads is not None else num_kv_heads
        if config.hidden_size % fw_num_heads != 0:
            raise ValueError(
                f"`hidden_size` ({config.hidden_size}) must be divisible by "
                f"`fw_num_heads` ({fw_num_heads}).",
            )

        self.memory_num_experts = config.memory_num_experts
        self.memory_num_active_experts = config.memory_num_active_experts
        self.memory_lb_loss_alpha = config.memory_lb_loss_alpha
        self.w0_w2_low_rank = config.w0_w2_low_rank
        self.layers.apply(self._init_weights)

    def _compose_memory_update_inputs(
        self,
        layer: RecurrentLactRefMoeV3Block,
        layer_idx: int,
        caches: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cache_for_ki = (
            caches[self.memory_k_layer_idxs[layer_idx]]
            if self.memory_k_gap is not None
            else caches[layer_idx]
        )
        if self.memory_kv_mode == "reuse_kv":
            ki = cache_for_ki["attn_k"]
        elif self.memory_kv_mode == "to_kv":
            feat = cache_for_ki[self.memory_kv_feature]
            if self.memory_kv_proj == "reuse_attn_kv_proj":
                ki, _ = layer.attn.attention_tokv(feat)
            elif self.memory_kv_proj == "ttt_kv_proj":
                ki, _ = layer.memory_tokv(feat)
            elif self.memory_kv_proj == "model_kv_proj":
                ki, _ = self.memory_tokv(feat)
            else:
                raise ValueError(f"Invalid memory_kv_proj: {self.memory_kv_proj}")
        else:
            raise ValueError(f"Invalid memory_kv_mode: {self.memory_kv_mode}")

        cache_for_vi = (
            caches[self.memory_v_layer_idxs[layer_idx]]
            if self.memory_v_gap is not None
            else caches[layer_idx]
        )
        if self.memory_kv_mode == "reuse_kv":
            vi = cache_for_vi["attn_v"]
        elif self.memory_kv_mode == "to_kv":
            feat = cache_for_vi[self.memory_kv_feature]
            if self.memory_kv_proj == "reuse_attn_kv_proj":
                _, vi = layer.attn.attention_tokv(feat)
            elif self.memory_kv_proj == "ttt_kv_proj":
                _, vi = layer.memory_tokv(feat)
            elif self.memory_kv_proj == "model_kv_proj":
                _, vi = self.memory_tokv(feat)
            else:
                raise ValueError(f"Invalid memory_kv_proj: {self.memory_kv_proj}")
        else:
            raise ValueError(f"Invalid memory_kv_mode: {self.memory_kv_mode}")

        pre_update = cache_for_ki[self.memory_kv_feature]
        return pre_update, ki, vi

    def _update_generation_memory_blocks(
        self,
        caches: list[dict[str, Any]],
        chunk_start: int,
        update_probs_per_layer: list[torch.Tensor | None] | None = None,
        update_freqs_per_layer: list[torch.Tensor | None] | None = None,
    ) -> None:
        if not self.use_memory_block:
            return
        for layer_idx, layer in enumerate(self.layers):
            cache = caches[layer_idx]
            pre_update, ki, vi = self._compose_memory_update_inputs(layer, layer_idx, caches)
            fastw, masterw, momentum_buf, update_lb_info = layer.update_fast_weight(
                pre_update,
                cache["fastw"],
                cache["masterw"],
                cache["momentum_buf"],
                ki=ki,
                vi=vi,
                seqlen_offset=chunk_start,
            )
            cache["fastw"] = fastw
            cache["masterw"] = masterw
            cache["momentum_buf"] = momentum_buf
            if update_probs_per_layer is not None and update_freqs_per_layer is not None:
                _accumulate_lb_info(
                    update_probs_per_layer,
                    update_freqs_per_layer,
                    layer_idx,
                    update_lb_info,
                )

    def _forward_with_generation_cache(
        self,
        *,
        inputs_embeds: torch.FloatTensor,
        past_key_values: Any | None,
        output_hidden_states: bool,
        return_dict: bool,
    ) -> tuple | MoeModelOutput:
        batch_size, seq_len, _ = inputs_embeds.shape
        max_len = base.recurrent_lact_cache_seen_tokens(past_key_values) + seq_len
        caches, seen_tokens, pending_start = self._init_or_load_generation_caches(
            past_key_values,
            batch_size,
            max_len,
            inputs_embeds.device,
        )
        pending_len = self._pending_len(caches, seen_tokens, pending_start)

        L = len(self.layers)
        num_experts = self.memory_num_experts
        apply_probs_per_layer, apply_freqs_per_layer = _zero_lb_accumulators(L)
        update_probs_per_layer, update_freqs_per_layer = _zero_lb_accumulators(L)

        all_hidden_states = () if output_hidden_states else None
        chunk_outputs = []
        local_start = 0
        current_seen = seen_tokens

        while local_start < seq_len:
            if pending_len >= self.chunk_size:
                self._update_generation_memory_blocks(
                    caches,
                    pending_start,
                    update_probs_per_layer,
                    update_freqs_per_layer,
                )
                self._clear_pending_generation_features(caches)
                pending_start = current_seen
                pending_len = 0

            take = min(seq_len - local_start, self.chunk_size - pending_len)
            chunk_inputs = inputs_embeds[:, local_start:local_start + take]
            append_pending = pending_len > 0
            hidden_states = chunk_inputs

            for layer_idx, layer in enumerate(self.layers):
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)
                (
                    hidden_states,
                    x_attn_in,
                    x_memory_in,
                    attn_k,
                    attn_v,
                    apply_lb_info,
                    fastw_out,
                    masterw_out,
                    momentum_buf_out,
                ) = layer(
                    hidden_states,
                    caches[layer_idx]["kv_cache"],
                    caches[layer_idx]["fastw"],
                    seqlen_offset=current_seen,
                )
                self._store_generation_layer_features(
                    caches[layer_idx],
                    xi_out=hidden_states,
                    x_attn_in=x_attn_in,
                    x_memory_in=x_memory_in,
                    attn_k=attn_k,
                    attn_v=attn_v,
                    append=append_pending,
                )
                if masterw_out is not None:
                    caches[layer_idx]["fastw"] = fastw_out
                    caches[layer_idx]["masterw"] = masterw_out
                    caches[layer_idx]["momentum_buf"] = momentum_buf_out
                _accumulate_lb_info(
                    apply_probs_per_layer,
                    apply_freqs_per_layer,
                    layer_idx,
                    apply_lb_info,
                )

            chunk_outputs.append(hidden_states)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            for layer_idx, layer in enumerate(self.layers):
                if self.attention_v_gap is not None:
                    pre_vi = self.pre_attention_norm(
                        caches[self.attention_v_layer_idxs[layer_idx]]["xi_out"],
                    )
                    if self.shared_kv_cache:
                        pre_vi = self.to_kv_cache(pre_vi)
                    pre_vi = pre_vi[:, -take:]
                else:
                    pre_vi = caches[layer_idx]["x_attn_in"][:, -take:]
                caches[layer_idx]["kv_cache"] = layer.update_kv(
                    pre_vi,
                    caches[layer_idx]["kv_cache"],
                    apply_layer_kv=not self.shared_kv_cache,
                )

            local_start += take
            current_seen += take
            pending_len += take

        hidden_states = torch.cat(chunk_outputs, dim=1)
        hidden_states = self.norm(hidden_states)
        next_cache = base.make_recurrent_lact_cache(
            layers=caches,
            seen_tokens=current_seen,
            pending_start=pending_start,
            batch_size=batch_size,
        )

        zero_be = inputs_embeds.new_zeros(batch_size, num_experts)
        apply_accum_probs = torch.stack(
            [p if p is not None else zero_be for p in apply_probs_per_layer],
            dim=0,
        )
        apply_accum_freqs = torch.stack(
            [f if f is not None else zero_be.detach() for f in apply_freqs_per_layer],
            dim=0,
        )
        apply_total_probs = apply_accum_probs.sum(dim=1)
        apply_total_freqs = apply_accum_freqs.sum(dim=1)
        apply_probs_mean = apply_total_probs / apply_total_probs.sum(dim=1, keepdim=True).clamp(min=1e-8)
        apply_freqs_mean = apply_total_freqs / apply_total_freqs.sum(dim=1, keepdim=True).clamp(min=1e-8)
        apply_lb_loss = ((apply_probs_mean * apply_freqs_mean).sum(dim=1) * num_experts).sum()

        zero_be_d = zero_be.detach()
        update_accum_probs = torch.stack(
            [p if p is not None else zero_be_d for p in update_probs_per_layer],
            dim=0,
        )
        update_accum_freqs = torch.stack(
            [f if f is not None else zero_be_d for f in update_freqs_per_layer],
            dim=0,
        )
        update_total_probs = update_accum_probs.sum(dim=1)
        update_total_freqs = update_accum_freqs.sum(dim=1)
        update_probs_mean = update_total_probs / update_total_probs.sum(dim=1, keepdim=True).clamp(min=1e-8)
        update_freqs_mean = update_total_freqs / update_total_freqs.sum(dim=1, keepdim=True).clamp(min=1e-8)
        update_lb_loss = (
            ((update_probs_mean * update_freqs_mean).sum(dim=1) * num_experts).sum()
            if update_total_freqs.sum() > 0
            else apply_lb_loss.new_zeros(())
        )
        lb_loss = (apply_lb_loss + update_lb_loss) * self.memory_lb_loss_alpha

        with torch.no_grad():
            expert_freq_min = apply_freqs_mean.detach().min(dim=1).values.mean().item()
            expert_freq_max = apply_freqs_mean.detach().max(dim=1).values.mean().item()
            expert_freq_std = apply_freqs_mean.detach().std(dim=1).mean().item()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, None] if v is not None)

        return MoeModelOutput(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=None,
            lb_loss=lb_loss,
            expert_freq_min=expert_freq_min,
            expert_freq_max=expert_freq_max,
            expert_freq_std=expert_freq_std,
        )

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        past_key_values: Any | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs: Any,
    ) -> tuple | MoeModelOutput:
        if output_attentions:
            warnings.warn(
                "`RecurrentLactRefMoeV3Model` does not support output attention weights, "
                "so `output_attentions` is set to `False`.",
            )
            output_attentions = False
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)
        if not self.use_memory_block:
            raise ValueError("ttt_moe_no_lb requires use_memory_block=True")

        batch_size, seq_len, _ = inputs_embeds.shape
        if use_cache:
            return self._forward_with_generation_cache(
                inputs_embeds=inputs_embeds,
                past_key_values=past_key_values,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        caches = []
        for layer in self.layers:
            k_cache, v_cache = layer.attn.cache_init(
                batch_size,
                seq_len,
                device=inputs_embeds.device,
                dtype=torch.bfloat16,
            )
            if layer.memory is not None:
                layer.memory.init_rotary_cache(seq_len, device=inputs_embeds.device, dtype=torch.bfloat16)
            caches.append(
                {
                    "kv_cache": (k_cache, v_cache),
                    "fastw": None,
                    "masterw": None,
                    "momentum_buf": None,
                    "xi_out": None,
                    "x_attn_in": None,
                    "x_memory_in": None,
                    "attn_k": None,
                    "attn_v": None,
                },
            )

        L = len(self.layers)
        num_experts = self.memory_num_experts
        apply_probs_per_layer, apply_freqs_per_layer = _zero_lb_accumulators(L)
        update_probs_per_layer, update_freqs_per_layer = _zero_lb_accumulators(L)

        all_hidden_states = () if output_hidden_states else None
        chunk_outputs = []

        for chunk_start in range(0, seq_len, self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, seq_len)
            chunk_inputs = inputs_embeds[:, chunk_start:chunk_end]
            is_last_chunk = chunk_start + self.chunk_size >= seq_len

            hidden_states = chunk_inputs
            for layer_idx, layer in enumerate(self.layers):
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                target = layer if chunk_start == 0 else layer.forward
                (
                    hidden_states,
                    x_attn_in,
                    x_memory_in,
                    attn_k,
                    attn_v,
                    apply_lb_info,
                    fastw_out,
                    masterw_out,
                    momentum_buf_out,
                ) = checkpoint(
                    target,
                    hidden_states,
                    caches[layer_idx]["kv_cache"],
                    caches[layer_idx]["fastw"],
                    seqlen_offset=chunk_start,
                    preserve_rng_state=False,
                    use_reentrant=False,
                )
                caches[layer_idx]["xi_out"] = hidden_states
                caches[layer_idx]["x_attn_in"] = x_attn_in
                caches[layer_idx]["x_memory_in"] = x_memory_in
                caches[layer_idx]["attn_k"] = attn_k
                caches[layer_idx]["attn_v"] = attn_v
                if masterw_out is not None:
                    caches[layer_idx]["fastw"] = fastw_out
                    caches[layer_idx]["masterw"] = masterw_out
                    caches[layer_idx]["momentum_buf"] = momentum_buf_out

                _accumulate_lb_info(
                    apply_probs_per_layer,
                    apply_freqs_per_layer,
                    layer_idx,
                    apply_lb_info,
                )

            chunk_outputs.append(hidden_states)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            for layer_idx, layer in enumerate(self.layers):
                if self.attention_v_gap is not None:
                    pre_vi = self.pre_attention_norm(
                        caches[self.attention_v_layer_idxs[layer_idx]]["xi_out"],
                    )
                    if self.shared_kv_cache:
                        pre_vi = self.to_kv_cache(pre_vi)
                else:
                    pre_vi = caches[layer_idx]["x_attn_in"]
                caches[layer_idx]["kv_cache"] = layer.update_kv(
                    pre_vi,
                    caches[layer_idx]["kv_cache"],
                    apply_layer_kv=not self.shared_kv_cache,
                )

            if not is_last_chunk:
                for layer_idx, layer in enumerate(self.layers):
                    cache = caches[layer_idx]
                    pre_update, ki, vi = self._compose_memory_update_inputs(
                        layer,
                        layer_idx,
                        caches,
                    )
                    fastw, masterw, momentum_buf, update_lb_info = layer.update_fast_weight(
                        pre_update,
                        cache["fastw"],
                        cache["masterw"],
                        cache["momentum_buf"],
                        ki=ki,
                        vi=vi,
                        seqlen_offset=chunk_start,
                    )
                    cache["fastw"] = fastw
                    cache["masterw"] = masterw
                    cache["momentum_buf"] = momentum_buf
                    _accumulate_lb_info(
                        update_probs_per_layer,
                        update_freqs_per_layer,
                        layer_idx,
                        update_lb_info,
                    )

        hidden_states = torch.cat(chunk_outputs, dim=1)
        hidden_states = self.norm(hidden_states)

        zero_be = inputs_embeds.new_zeros(batch_size, num_experts)
        apply_accum_probs = torch.stack(
            [p if p is not None else zero_be for p in apply_probs_per_layer],
            dim=0,
        )
        apply_accum_freqs = torch.stack(
            [f if f is not None else zero_be.detach() for f in apply_freqs_per_layer],
            dim=0,
        )
        apply_total_probs = apply_accum_probs.sum(dim=1)
        apply_total_freqs = apply_accum_freqs.sum(dim=1)
        apply_probs_mean = apply_total_probs / apply_total_probs.sum(dim=1, keepdim=True).clamp(min=1e-8)
        apply_freqs_mean = apply_total_freqs / apply_total_freqs.sum(dim=1, keepdim=True).clamp(min=1e-8)
        apply_lb_loss = ((apply_probs_mean * apply_freqs_mean).sum(dim=1) * num_experts).sum()

        zero_be_d = zero_be.detach()
        update_accum_probs = torch.stack(
            [p if p is not None else zero_be_d for p in update_probs_per_layer],
            dim=0,
        )
        update_accum_freqs = torch.stack(
            [f if f is not None else zero_be_d for f in update_freqs_per_layer],
            dim=0,
        )
        update_total_probs = update_accum_probs.sum(dim=1)
        update_total_freqs = update_accum_freqs.sum(dim=1)
        update_probs_mean = update_total_probs / update_total_probs.sum(dim=1, keepdim=True).clamp(min=1e-8)
        update_freqs_mean = update_total_freqs / update_total_freqs.sum(dim=1, keepdim=True).clamp(min=1e-8)
        update_lb_loss = (
            ((update_probs_mean * update_freqs_mean).sum(dim=1) * num_experts).sum()
            if update_total_freqs.sum() > 0
            else apply_lb_loss.new_zeros(())
        )
        lb_loss = (apply_lb_loss + update_lb_loss) * self.memory_lb_loss_alpha

        with torch.no_grad():
            expert_freq_min = apply_freqs_mean.detach().min(dim=1).values.mean().item()
            expert_freq_max = apply_freqs_mean.detach().max(dim=1).values.mean().item()
            expert_freq_std = apply_freqs_mean.detach().std(dim=1).mean().item()

        if not return_dict:
            return tuple(v for v in [hidden_states, None, all_hidden_states, None] if v is not None)

        return MoeModelOutput(
            last_hidden_state=hidden_states,
            past_key_values=None,
            hidden_states=all_hidden_states,
            attentions=None,
            lb_loss=lb_loss,
            expert_freq_min=expert_freq_min,
            expert_freq_max=expert_freq_max,
            expert_freq_std=expert_freq_std,
        )

    def extra_repr(self) -> str:
        parent_repr = super().extra_repr()
        if not self.use_memory_block:
            return parent_repr
        memory = self.layers[0].memory if self.layers and self.layers[0].memory is not None else None
        if memory is None:
            return parent_repr
        lines = [
            parent_repr,
            "memory_pool_mode: per_layer_moe_v3",
            f"memory_num_experts: {self.memory_num_experts}",
            f"memory_num_active_experts: {self.memory_num_active_experts}",
            f"expert_w1_per_layer: {tuple(memory.w1.shape)}",
            f"router_proj_per_layer: {tuple(memory.router_proj_weights.shape)}",
        ]
        return "\n".join(line for line in lines if line)


def _init_moe_v3_weights(
    self,
    module: nn.Module,
    rescale_prenorm_residual: bool = False,
    num_residuals_per_layer: int = 2,
):
    if isinstance(module, FastWeightMLPSubNNMoeV3):
        module.reset_parameters()
        return
    if isinstance(module, RecurrentLactRefMoeV3Model):
        return
    return base.RecurrentLactRefPreTrainedModel._init_weights(
        self,
        module,
        rescale_prenorm_residual=rescale_prenorm_residual,
        num_residuals_per_layer=num_residuals_per_layer,
    )


RecurrentLactRefMoeV3Model._init_weights = _init_moe_v3_weights


class RecurrentLactRefMoeV3ForCausalLM(base.RecurrentLactRefForCausalLM):
    config_class = RecurrentLactRefMoeV3Config
    _no_split_modules = ["RecurrentLactRefMoeV3Block"]
    _init_weights = _init_moe_v3_weights

    def __init__(self, config: RecurrentLactRefMoeV3Config):
        base.RecurrentLactRefPreTrainedModel.__init__(self, config)
        self.model = RecurrentLactRefMoeV3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.criterion = None
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        past_key_values: Any | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        logits_to_keep: int | None = 0,
        **kwargs: Any,
    ):
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)

        outputs = self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            # Never forward output_hidden_states into the inner model: it
            # accumulates per-chunk fragments, and consumers such as
            # flame.eval_loss's chunked lm_head path expect [-1] to be the
            # full sequence. Expose the concatenated last_hidden_state below.
            output_hidden_states=False,
            return_dict=True,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        fuse_lce = self.config.fuse_linear_cross_entropy
        logits = None if fuse_lce else self.lm_head(hidden_states[:, -logits_to_keep:])

        loss = None
        ce_loss = None
        if labels is not None:
            if getattr(self, "criterion", None) is None:
                if fuse_lce:
                    criterion = base.FusedLinearCrossEntropyLoss(use_l2warp=self.config.use_l2warp)
                elif self.config.fuse_cross_entropy:
                    criterion = base.FusedCrossEntropyLoss(inplace_backward=True)
                else:
                    criterion = nn.CrossEntropyLoss()
            else:
                criterion = self.criterion
            labels = labels.to(hidden_states.device)
            labels = torch.cat(
                (labels[..., 1:], torch.full_like(labels[:, :1], criterion.ignore_index)),
                dim=1,
            )
            if fuse_lce:
                ce_loss = criterion(hidden_states, labels, self.lm_head.weight, self.lm_head.bias)
            else:
                ce_loss = criterion(logits.view(labels.numel(), -1), labels.view(-1))
                ce_loss = base.l2_warp(ce_loss, logits) if self.config.use_l2warp else ce_loss
            loss = ce_loss + outputs.lb_loss if outputs.lb_loss is not None else ce_loss

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if not return_dict:
            output = (logits,) + (outputs.past_key_values,) + (outputs.hidden_states,)
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithMoeStats(
            loss=loss,
            ce_loss=ce_loss.detach() if ce_loss is not None else None,
            lb_loss=outputs.lb_loss.detach() if outputs.lb_loss is not None else None,
            expert_freq_min=outputs.expert_freq_min,
            expert_freq_max=outputs.expert_freq_max,
            expert_freq_std=outputs.expert_freq_std,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=(hidden_states,) if output_hidden_states else None,
            attentions=outputs.attentions,
        )


RecurrentLactRefModel = RecurrentLactRefMoeV3Model
RecurrentLactRefForCausalLM = RecurrentLactRefMoeV3ForCausalLM
