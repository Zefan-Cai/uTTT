from __future__ import annotations

from dataclasses import dataclass
import math
import os
import warnings
from typing import Any, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from .configuration_recurrent_lact import RecurrentLactRefMoeGlobalV45Config
from .moe_routing import create_router_mask_sizes_probs
from . import ref_base as base
from ..lact_cache_utils import (
    is_recurrent_lact_cache,
    make_recurrent_lact_cache,
    prepare_recurrent_lact_generation_inputs,
    recurrent_lact_cache_seen_tokens,
    reorder_recurrent_lact_cache,
)

try:
    from .lact_moe_triton.triton_permute import (
        permute_kv_and_lrs as _permute_kv_and_lrs,
        permute_with_expert_mask as _permute_with_expert_mask,
        unpermute_and_merge_with_probs as _unpermute_and_merge_with_probs,
    )
    from .lact_moe_triton.lact_swiglu_ffn import (
        grouped_swiglu_ffn_fwd as _grouped_swiglu_ffn_fwd,
    )
    from .lact_moe_triton.lact_fw_grad import (
        grouped_lact_swiglu_ffn_fast_weight_grads as _grouped_lact_swiglu_ffn_fast_weight_grads,
    )
except ImportError:
    _permute_kv_and_lrs = None
    _permute_with_expert_mask = None
    _unpermute_and_merge_with_probs = None
    _grouped_swiglu_ffn_fwd = None
    _grouped_lact_swiglu_ffn_fast_weight_grads = None


def _env_flag(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).lower() not in {"0", "false", "no", "off"}


_PRECOMPUTE_W0_W2 = _env_flag("LACT_GLOBAL_MOE_V45_PRECOMPUTE_W0W2", "1")
_AGG_INNER_CHECKPOINT = _env_flag("LACT_GLOBAL_MOE_V45_AGG_INNER_CKPT", "1")
_ROUTE_STATS = _env_flag("LACT_GLOBAL_MOE_V45_ROUTE_STATS", "1")
_MOE_STATS_INTERVAL = max(
    1,
    int(
        os.environ.get(
            "LACT_GLOBAL_MOE_V45_MOE_STATS_INTERVAL",
            os.environ.get("LACT_GLOBAL_MOE_V45_ROUTE_STATS_INTERVAL", "1000"),
        ),
    ),
)


def _moe_stats_enabled() -> bool:
    if _MOE_STATS_INTERVAL <= 1:
        return True
    step = int(os.environ.get("FLAME_TRAIN_STEP", "0") or "0")
    return step > 0 and step % _MOE_STATS_INTERVAL == 0


def _route_stats_enabled() -> bool:
    return _ROUTE_STATS and _moe_stats_enabled()


@dataclass
class MoeModelOutput(BaseModelOutputWithPast):
    lb_loss: Optional[torch.Tensor] = None
    moe_stats: Optional[Any] = None


@dataclass
class CausalLMOutputWithMoeStats(CausalLMOutputWithPast):
    ce_loss: Optional[torch.Tensor] = None
    lb_loss: Optional[torch.Tensor] = None
    moe_stats: Optional[Any] = None


class FastWeightMLPSubNN(base.FastWeightMLPSubNN):
    def __init__(
        self,
        config: RecurrentLactRefMoeGlobalV45Config,
        layer_idx: int,
    ):
        self.memory_num_experts = config.memory_num_experts
        self.memory_num_active_experts = config.memory_num_active_experts
        self.memory_router_alpha = config.memory_router_alpha
        self.memory_router_use_sigmoid = config.memory_router_use_sigmoid
        self.memory_router_from_v = config.memory_router_from_v
        self.memory_router_type = config.memory_router_type
        self.memory_router_combine_mode = config.memory_router_combine_mode
        self.memory_router_scale = config.memory_router_scale
        self.memory_router_norm_eps = config.memory_router_norm_eps
        self.use_moe = self.memory_num_experts > 1
        self.use_grouped_moe = self.use_moe
        self.use_dense_fused_moe = False
        super().__init__(config, layer_idx)

        if self.use_moe:
            for attr in ("w0", "w1", "w2"):
                if hasattr(self, attr):
                    delattr(self, attr)
            if self.use_momentum:
                self.momentum_proj = nn.Sequential(
                    nn.Linear(self.hidden_size, 1),
                    nn.Sigmoid(),
                )
            self._last_apply_lb_info = None
            self._last_update_lb_info = None
            self._last_apply_route_stats = None
            self._last_update_route_stats = None

    def reset_parameters(self):
        if not self.use_moe:
            super().reset_parameters()
            return

        if self.qk_rescale:
            nn.init.ones_(self.qk_scale)
            nn.init.zeros_(self.qk_offset)

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

    def _route_tokens(
        self,
        x: torch.Tensor,
        router_proj_weights: torch.Tensor,
        routing_bias: torch.Tensor | None = None,
    ):
        with torch.autocast(device_type=x.device.type, enabled=False):
            logits = torch.matmul(
                router_proj_weights.float(),
                x.transpose(1, 2).float(),
            )
        expert_mask, group_sizes, router_probs, balance_probs, _ = create_router_mask_sizes_probs(
            logits,
            topk=self.memory_num_active_experts,
            alpha=self.memory_router_alpha,
            use_sigmoid=self.memory_router_use_sigmoid,
            router_type=self.memory_router_type,
            combine_mode=self.memory_router_combine_mode,
            router_scale=self.memory_router_scale,
            norm_eps=self.memory_router_norm_eps,
            routing_bias=routing_bias,
        )
        return expert_mask, group_sizes, router_probs, balance_probs

    @staticmethod
    def _make_lb_info(
        balance_probs: torch.Tensor,
        group_sizes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return balance_probs.sum(dim=-1), group_sizes

    @staticmethod
    def _make_route_stats(
        combine_weights: torch.Tensor,
        balance_probs: torch.Tensor,
        expert_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor] | None:
        if not _route_stats_enabled():
            return None
        with torch.no_grad():
            zero = balance_probs.new_zeros((), dtype=torch.float32)
            selected = combine_weights.detach()[expert_mask].float()
            token_count = balance_probs.new_tensor(
                float(balance_probs.numel() // balance_probs.shape[-2]),
                dtype=torch.float32,
            )
            probs = balance_probs.detach().float()
            probs = probs / probs.sum(dim=-2, keepdim=True).clamp_min(1e-12)
            entropy_input = probs.clamp_min(1e-12)
            router_entropy = -(entropy_input * entropy_input.log()).sum(dim=-2).mean()
            if selected.numel() == 0:
                return {
                    "selected_weight_sum": zero,
                    "selected_weight_count": zero,
                    "selected_weight_min": zero,
                    "selected_weight_max": zero,
                    "selected_weight_p50_sum": zero,
                    "selected_weight_p95_sum": zero,
                    "router_entropy_sum": router_entropy * token_count,
                    "router_entropy_count": token_count,
                }

            count = selected.new_tensor(float(selected.numel()))
            return {
                "selected_weight_sum": selected.sum(),
                "selected_weight_count": count,
                "selected_weight_min": selected.min(),
                "selected_weight_max": selected.max(),
                "selected_weight_p50_sum": torch.quantile(selected, 0.50) * count,
                "selected_weight_p95_sum": torch.quantile(selected, 0.95) * count,
                "router_entropy_sum": router_entropy * token_count,
                "router_entropy_count": token_count,
            }

    def _apply_moe_fast_weights_loop(
        self,
        x: torch.Tensor,
        w0: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        router_proj_weights: torch.Tensor,
        routing_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        seq_len = x.shape[1]
        x = self._reshape_head_tokens(x)
        out = torch.zeros_like(x, dtype=torch.float32)
        expert_mask, group_sizes, router_probs, balance_probs = self._route_tokens(
            x,
            router_proj_weights,
            routing_bias=routing_bias,
        )
        self._last_apply_lb_info = self._make_lb_info(balance_probs, group_sizes)
        self._last_apply_route_stats = self._make_route_stats(router_probs, balance_probs, expert_mask)

        for batch_idx in range(x.shape[0]):
            for expert_idx in range(self.memory_num_experts):
                token_mask = expert_mask[batch_idx, expert_idx]
                if not token_mask.any().item():
                    continue

                q_tokens = x[batch_idx, token_mask]
                w0_e = w0[batch_idx, expert_idx]
                w1_e = w1[batch_idx, expert_idx]
                w2_e = w2[batch_idx, expert_idx]
                q_tokens = q_tokens.to(w0_e.dtype)

                gate_before_act = torch.matmul(q_tokens, w0_e.transpose(0, 1))
                hidden_before_mul = torch.matmul(q_tokens, w2_e.transpose(0, 1))
                hidden = F.silu(gate_before_act, inplace=False) * hidden_before_mul
                out_tokens = torch.matmul(hidden, w1_e.transpose(0, 1))

                probs = router_probs[batch_idx, expert_idx, token_mask].unsqueeze(-1)
                out[batch_idx, token_mask] += out_tokens.to(out.dtype) * probs

        return self._merge_head_tokens(out.to(x.dtype), seq_len)

    def _apply_moe_fast_weights_dense(
        self,
        x: torch.Tensor,
        w0: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        router_proj_weights: torch.Tensor,
        routing_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        seq_len = x.shape[1]
        input_dtype = x.dtype
        x = self._reshape_head_tokens(x)
        expert_mask, group_sizes, router_probs, balance_probs = self._route_tokens(
            x,
            router_proj_weights,
            routing_bias=routing_bias,
        )
        self._last_apply_lb_info = self._make_lb_info(balance_probs, group_sizes)
        self._last_apply_route_stats = self._make_route_stats(router_probs, balance_probs, expert_mask)
        route_weights = router_probs * expert_mask.to(router_probs.dtype)

        x = x.to(w0.dtype)
        gate_before_act = torch.einsum("btd,beid->beti", x, w0)
        hidden_before_mul = torch.einsum("btd,beid->beti", x, w2)
        hidden = F.silu(gate_before_act, inplace=False) * hidden_before_mul
        out_by_expert = torch.einsum("beti,bedi->betd", hidden, w1)

        out = (
            out_by_expert.to(route_weights.dtype)
            * route_weights.unsqueeze(-1)
        ).sum(dim=1)
        return self._merge_head_tokens(out.to(input_dtype), seq_len)

    def _apply_moe_fast_weights_grouped(
        self,
        x: torch.Tensor,
        w0: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        router_proj_weights: torch.Tensor,
        routing_bias: torch.Tensor | None = None,
        precomputed_w0_w2: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (
            _permute_with_expert_mask is None
            or _unpermute_and_merge_with_probs is None
            or _grouped_swiglu_ffn_fwd is None
        ):
            return self._apply_moe_fast_weights_loop(
                x,
                w0,
                w1,
                w2,
                router_proj_weights,
                routing_bias=routing_bias,
            )

        seq_len = x.shape[1]
        input_dtype = x.dtype
        x = self._reshape_head_tokens(x)
        expert_mask, group_sizes, router_probs, balance_probs = self._route_tokens(
            x,
            router_proj_weights,
            routing_bias=routing_bias,
        )
        self._last_apply_lb_info = self._make_lb_info(balance_probs, group_sizes)
        self._last_apply_route_stats = self._make_route_stats(router_probs, balance_probs, expert_mask)

        x_perm, _, row_id_map = _permute_with_expert_mask(
            x.to(w0.dtype).contiguous(),
            expert_mask,
            router_probs,
            self.memory_num_active_experts,
        )
        w0_w2 = (
            precomputed_w0_w2
            if precomputed_w0_w2 is not None
            else torch.cat([w0, w2], dim=2).contiguous()
        )
        y_perm = _grouped_swiglu_ffn_fwd(
            w0_w2,
            w1.contiguous(),
            x_perm.contiguous(),
            group_sizes.to(torch.int32).contiguous(),
        )
        out = _unpermute_and_merge_with_probs(y_perm, row_id_map, router_probs)
        return self._merge_head_tokens(out.to(input_dtype), seq_len)

    def _apply_moe_fast_weights(
        self,
        x: torch.Tensor,
        w0: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        router_proj_weights: torch.Tensor,
        routing_bias: torch.Tensor | None = None,
        precomputed_w0_w2: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.use_grouped_moe:
            return self._apply_moe_fast_weights_grouped(
                x,
                w0,
                w1,
                w2,
                router_proj_weights,
                routing_bias=routing_bias,
                precomputed_w0_w2=precomputed_w0_w2,
            )
        if self.use_dense_fused_moe:
            return self._apply_moe_fast_weights_dense(
                x,
                w0,
                w1,
                w2,
                router_proj_weights,
                routing_bias=routing_bias,
            )
        return self._apply_moe_fast_weights_loop(
            x,
            w0,
            w1,
            w2,
            router_proj_weights,
            routing_bias=routing_bias,
        )

    def _compute_moe_update_grads_loop(
        self,
        k_tokens: torch.Tensor,
        v_tokens: torch.Tensor,
        lr_tokens: torch.Tensor,
        w0: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        expert_mask: torch.Tensor,
        router_probs: torch.Tensor,
        w0_master: torch.Tensor,
        w1_master: torch.Tensor,
        w2_master: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = k_tokens.shape[0]
        w0_grad_raw = torch.zeros_like(w0_master)
        w1_grad_raw = torch.zeros_like(w1_master)
        w2_grad_raw = torch.zeros_like(w2_master)

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

                gate_before_act = torch.matmul(k_e, w0_e.transpose(0, 1))
                hidden_before_mul = torch.matmul(k_e, w2_e.transpose(0, 1))
                hidden = F.silu(gate_before_act, inplace=False) * hidden_before_mul

                dhidden = torch.matmul(v_e, w1_e)
                dhidden_before_mul = dhidden * F.silu(gate_before_act, inplace=False)
                dgate = dhidden * hidden_before_mul
                dgate_before_act = base.silu_backprop(dgate, gate_before_act)

                w1_grad_raw[batch_idx, expert_idx] = torch.matmul(
                    v_e.transpose(0, 1),
                    hidden * lr1_e,
                ).to(w1_grad_raw.dtype)
                w0_grad_raw[batch_idx, expert_idx] = torch.matmul(
                    dgate_before_act.transpose(0, 1),
                    k_e * lr0_e,
                ).to(w0_grad_raw.dtype)
                w2_grad_raw[batch_idx, expert_idx] = torch.matmul(
                    dhidden_before_mul.transpose(0, 1),
                    k_e * lr2_e,
                ).to(w2_grad_raw.dtype)

        return w0_grad_raw, w1_grad_raw, w2_grad_raw

    def _compute_moe_update_grads_grouped(
        self,
        k_tokens: torch.Tensor,
        v_tokens: torch.Tensor,
        lr_tokens: torch.Tensor,
        w0: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        expert_mask: torch.Tensor,
        group_sizes: torch.Tensor,
        router_probs: torch.Tensor,
        w0_master: torch.Tensor,
        w1_master: torch.Tensor,
        w2_master: torch.Tensor,
        precomputed_w0_w2: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if _permute_kv_and_lrs is None or _grouped_lact_swiglu_ffn_fast_weight_grads is None:
            return self._compute_moe_update_grads_loop(
                k_tokens,
                v_tokens,
                lr_tokens,
                w0,
                w1,
                w2,
                expert_mask,
                router_probs,
                w0_master,
                w1_master,
                w2_master,
            )

        target_dtype = w0.dtype
        lr0 = lr_tokens[:, :, 0].contiguous().to(target_dtype)
        lr1 = lr_tokens[:, :, 1].contiguous().to(target_dtype)
        lr2 = lr_tokens[:, :, 2].contiguous().to(target_dtype)
        k_perm, v_perm, lr0_perm, lr1_perm, lr2_perm = _permute_kv_and_lrs(
            k_tokens.contiguous().to(target_dtype),
            v_tokens.contiguous(),
            lr0,
            lr1,
            lr2,
            expert_mask.to(torch.int32),
            router_probs,
            self.memory_num_active_experts,
        )
        w0_w2 = (
            precomputed_w0_w2
            if precomputed_w0_w2 is not None
            else torch.cat([w0, w2], dim=2).contiguous()
        )
        dw0_w2, dw1 = _grouped_lact_swiglu_ffn_fast_weight_grads(
            w0_w2,
            w1.contiguous(),
            k_perm,
            v_perm,
            lr0_perm,
            lr1_perm,
            lr2_perm,
            group_sizes.to(torch.int32).contiguous(),
        )
        inter_dim = w0.shape[2]
        return (
            dw0_w2[:, :, :inter_dim, :].to(w0_master.dtype),
            dw1.to(w1_master.dtype),
            dw0_w2[:, :, inter_dim:, :].to(w2_master.dtype),
        )

    def _compute_moe_update_grads_dense(
        self,
        k_tokens: torch.Tensor,
        v_tokens: torch.Tensor,
        lr_tokens: torch.Tensor,
        w0: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        expert_mask: torch.Tensor,
        router_probs: torch.Tensor,
        w0_master: torch.Tensor,
        w1_master: torch.Tensor,
        w2_master: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        route_weights = router_probs * expert_mask.to(router_probs.dtype)
        k_tokens = k_tokens.to(w0.dtype)
        v_weighted = (
            v_tokens.unsqueeze(1)
            * route_weights.to(v_tokens.dtype).unsqueeze(-1)
        ).to(w1.dtype)
        lr_tokens = lr_tokens.to(k_tokens.dtype)
        lr0 = lr_tokens[:, None, :, 0:1]
        lr1 = lr_tokens[:, None, :, 1:2]
        lr2 = lr_tokens[:, None, :, 2:3]

        gate_before_act = torch.einsum("btd,beid->beti", k_tokens, w0)
        hidden_before_mul = torch.einsum("btd,beid->beti", k_tokens, w2)
        hidden = F.silu(gate_before_act, inplace=False) * hidden_before_mul

        dhidden = torch.einsum("betd,bedi->beti", v_weighted, w1)
        dhidden_before_mul = dhidden * F.silu(gate_before_act, inplace=False)
        dgate = dhidden * hidden_before_mul
        dgate_before_act = base.silu_backprop(dgate, gate_before_act)

        w1_grad_raw = torch.einsum(
            "betd,beti->bedi",
            v_weighted,
            (hidden * lr1).to(v_weighted.dtype),
        ).to(w1_master.dtype)
        w0_grad_raw = torch.einsum(
            "beti,betd->beid",
            dgate_before_act,
            (k_tokens[:, None] * lr0).to(dgate_before_act.dtype),
        ).to(w0_master.dtype)
        w2_grad_raw = torch.einsum(
            "beti,betd->beid",
            dhidden_before_mul,
            (k_tokens[:, None] * lr2).to(dhidden_before_mul.dtype),
        ).to(w2_master.dtype)

        return w0_grad_raw, w1_grad_raw, w2_grad_raw

    def forward(
        self,
        q: torch.Tensor,
        fastw: Tuple[torch.Tensor, ...],
        seqlen_offset: int = 0,
        router_proj_weights: torch.Tensor | None = None,
        routing_bias: torch.Tensor | None = None,
        precomputed_w0_w2: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.use_moe:
            return super().forward(q, fastw, seqlen_offset=seqlen_offset)
        if router_proj_weights is None:
            raise ValueError("global-v4.5 MoE memory requires model-owned router weights.")

        w0, w1, w2 = fastw

        if self.qk_rescale:
            q, _ = self._rescale_qk(q, q)
        if self.qkv_silu:
            q = F.silu(q)
        if self.qk_norm:
            q = base.rearrange(q, "b l (h d) -> (b h) l d", h=self.fw_num_heads)
            q = self.qk_norm_fn(q)
            q = base.rearrange(q, "(b h) l d -> b l (h d)", h=self.fw_num_heads)

        if self.ttt_rotary is not None:
            q = base.rearrange(q, "b l (nh d) -> b l nh d", nh=self.n_rope_heads)
            q = q.to(self.ttt_rotary._cos_cached.dtype)
            _, q = self.ttt_rotary(None, q, seqlen_offset=seqlen_offset)
            q = base.rearrange(q, "b l nh d -> b l (nh d)")

        out = self._apply_moe_fast_weights(
            q,
            w0,
            w1,
            w2,
            router_proj_weights,
            routing_bias=routing_bias,
            precomputed_w0_w2=precomputed_w0_w2,
        )

        if not self.learnable_ttt_scale:
            out = base.rearrange(
                out,
                "b s (n_h d) -> (b n_h) s d",
                n_h=self.fw_num_heads,
            )
            out = self.ttt_norm(out)
            out = base.rearrange(
                out,
                "(b n_h) s d -> b s (n_h d)",
                n_h=self.fw_num_heads,
            )
        if self.enable_memory_output_proj:
            out = self.output_proj(out)

        return out

    def _preprocess_update_inputs(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        lr: torch.Tensor,
        seqlen_offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
            k = self.qk_norm_fn(k)
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
        return k_tokens, v_tokens, lr_tokens

    def compute_raw_grad(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        lr: torch.Tensor,
        fastw: Tuple[torch.Tensor, ...],
        masterw: Tuple[torch.Tensor, ...],
        router_proj_weights: torch.Tensor,
        routing_bias: torch.Tensor | None = None,
        pre_vi: torch.Tensor = None,
        seqlen_offset: int = 0,
        precomputed_w0_w2: torch.Tensor | None = None,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor | None]:
        if not self.use_moe:
            raise ValueError("`compute_raw_grad` is only defined for MoE memory.")

        w0_master, w1_master, w2_master = masterw
        w0, w1, w2 = fastw
        k_tokens, v_tokens, lr_tokens = self._preprocess_update_inputs(
            k,
            v,
            lr,
            seqlen_offset=seqlen_offset,
        )

        router_input = v_tokens if self.memory_router_from_v else k_tokens
        expert_mask, group_sizes, router_probs, balance_probs = self._route_tokens(
            router_input,
            router_proj_weights,
            routing_bias=routing_bias,
        )
        self._last_update_lb_info = self._make_lb_info(balance_probs, group_sizes)
        self._last_update_route_stats = self._make_route_stats(router_probs, balance_probs, expert_mask)

        if self.use_grouped_moe:
            grads = self._compute_moe_update_grads_grouped(
                k_tokens,
                v_tokens,
                lr_tokens,
                w0,
                w1,
                w2,
                expert_mask,
                group_sizes,
                router_probs,
                w0_master,
                w1_master,
                w2_master,
                precomputed_w0_w2=precomputed_w0_w2,
            )
        elif self.use_dense_fused_moe:
            grads = self._compute_moe_update_grads_dense(
                k_tokens,
                v_tokens,
                lr_tokens,
                w0,
                w1,
                w2,
                expert_mask,
                router_probs,
                w0_master,
                w1_master,
                w2_master,
            )
        else:
            grads = self._compute_moe_update_grads_loop(
                k_tokens,
                v_tokens,
                lr_tokens,
                w0,
                w1,
                w2,
                expert_mask,
                router_probs,
                w0_master,
                w1_master,
                w2_master,
            )

        m_coeff = None
        if self.use_momentum and pre_vi is not None:
            m_coeff = self.momentum_proj(
                pre_vi.to(self.momentum_proj[0].weight.dtype),
            ).mean(dim=1).view(k.shape[0], -1, 1, 1)

        return grads, m_coeff

    def _apply_moe_update_dense(
        self,
        grads: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        fastw: Tuple[torch.Tensor, ...],
        masterw: Tuple[torch.Tensor, ...],
        momentum_buf: Tuple[torch.Tensor, ...] | None = None,
        m_coeff: torch.Tensor | None = None,
        lr_scale: float = 1.0,
    ) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...] | None]:
        w0_master, w1_master, w2_master = masterw
        w0, w1, w2 = fastw
        w0_grad_raw, w1_grad_raw, w2_grad_raw = grads

        w0_norm = w0_master.norm(dim=3, keepdim=True)
        w1_norm = w1_master.norm(dim=3, keepdim=True)
        w2_norm = w2_master.norm(dim=3, keepdim=True)

        if self.use_momentum and momentum_buf is not None and m_coeff is not None:
            dw0_mom, dw1_mom, dw2_mom = momentum_buf
            w0_grad_raw = w0_grad_raw + dw0_mom * m_coeff
            w1_grad_raw = w1_grad_raw + dw1_mom * m_coeff
            w2_grad_raw = w2_grad_raw + dw2_mom * m_coeff
            momentum_buf = (
                w0_grad_raw.clone(),
                w1_grad_raw.clone(),
                w2_grad_raw.clone(),
            )

        if self.use_moun:
            b_size, n_group = w0_grad_raw.shape[:2]
            w0_grad_flat = base.rearrange(w0_grad_raw, "b g d1 d2 -> (b g) d1 d2")
            w1_grad_flat = base.rearrange(w1_grad_raw, "b g d1 d2 -> (b g) d1 d2")
            w2_grad_flat = base.rearrange(w2_grad_raw, "b g d1 d2 -> (b g) d1 d2")

            w0_grad_flat = base.zeropower_via_newtonschulz5(w0_grad_flat, 5)
            w1_grad_flat = base.zeropower_via_newtonschulz5(w1_grad_flat, 5)
            w2_grad_flat = base.zeropower_via_newtonschulz5(w2_grad_flat, 5)

            w0_grad = base.rearrange(
                w0_grad_flat,
                "(b g) d1 d2 -> b g d1 d2",
                b=b_size,
                g=n_group,
            )
            w1_grad = base.rearrange(
                w1_grad_flat,
                "(b g) d1 d2 -> b g d1 d2",
                b=b_size,
                g=n_group,
            )
            w2_grad = base.rearrange(
                w2_grad_flat,
                "(b g) d1 d2 -> b g d1 d2",
                b=b_size,
                g=n_group,
            )
        else:
            w0_grad = w0_grad_raw
            w1_grad = w1_grad_raw
            w2_grad = w2_grad_raw

        w0_master = w0_master + w0_grad * lr_scale
        w1_master = w1_master + w1_grad * lr_scale
        w2_master = w2_master + w2_grad * lr_scale

        w0_master = w0_master / (w0_master.norm(dim=3, keepdim=True) + 1e-5) * w0_norm
        w1_master = w1_master / (w1_master.norm(dim=3, keepdim=True) + 1e-5) * w1_norm
        w2_master = w2_master / (w2_master.norm(dim=3, keepdim=True) + 1e-5) * w2_norm

        masterw = (w0_master, w1_master, w2_master)
        weight = tuple(w.to(torch.bfloat16) for w in masterw)
        return weight, masterw, momentum_buf

    def apply_update(
        self,
        grads: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        fastw: Tuple[torch.Tensor, ...],
        masterw: Tuple[torch.Tensor, ...],
        momentum_buf: Tuple[torch.Tensor, ...] | None = None,
        m_coeff: torch.Tensor | None = None,
        lr_scale: float = 1.0,
    ) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...] | None]:
        return self._apply_moe_update_dense(
            grads,
            fastw,
            masterw,
            momentum_buf=momentum_buf,
            m_coeff=m_coeff,
            lr_scale=lr_scale,
        )

    def update(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        lr: torch.Tensor,
        fastw: Tuple[torch.Tensor, ...],
        masterw: Tuple[torch.Tensor, ...],
        momentum_buf: Tuple[torch.Tensor, ...] | None = None,
        pre_vi: torch.Tensor = None,
        seqlen_offset: int = 0,
        router_proj_weights: torch.Tensor | None = None,
        routing_bias: torch.Tensor | None = None,
        precomputed_w0_w2: torch.Tensor | None = None,
    ) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...] | None]:
        if not self.use_moe:
            return super().update(
                k,
                v,
                lr,
                fastw,
                masterw,
                momentum_buf=momentum_buf,
                pre_vi=pre_vi,
                seqlen_offset=seqlen_offset,
            )
        if router_proj_weights is None:
            raise ValueError("global-v4.5 MoE update requires model-owned router weights.")

        grads, m_coeff = self.compute_raw_grad(
            k,
            v,
            lr,
            fastw,
            masterw,
            router_proj_weights,
            routing_bias=routing_bias,
            pre_vi=pre_vi,
            seqlen_offset=seqlen_offset,
            precomputed_w0_w2=precomputed_w0_w2,
        )
        return self.apply_update(
            grads,
            fastw,
            masterw,
            momentum_buf=momentum_buf,
            m_coeff=m_coeff,
        )

    def extra_repr(self) -> str:
        if not self.use_moe:
            return super().extra_repr()
        return (
            f"head_dim={self.head_dim}, inter_dim={self.inter_dim}, "
            f"fw_num_heads={self.fw_num_heads}, low_rank={self.w0_w2_low_rank}, "
            f"use_moe={self.use_moe}, num_experts={self.memory_num_experts}, "
            f"topk={self.memory_num_active_experts}, "
            f"dense_fused={self.use_dense_fused_moe}"
        )


class RecurrentLactRefMoeGlobalV45Block(base.RecurrentLactBlock):
    def __init__(self, config: RecurrentLactRefMoeGlobalV45Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.memory_update_backward_mode = config.memory_update_backward_mode
        self._detach_update_grads = self.memory_update_backward_mode in {
            "first_order_state",
            "stop_state",
        }
        self._detach_updated_state = self.memory_update_backward_mode == "stop_state"
        self._detach_m_coeff = config.memory_detach_m_coeff
        if self.memory is not None:
            self.memory = FastWeightMLPSubNN(config, layer_idx)

    def _apply_grad_detach_policy(
        self,
        grads: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        m_coeff: torch.Tensor | None,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor | None]:
        if self._detach_update_grads:
            grads = tuple(grad.detach() for grad in grads)
        if (self._detach_m_coeff or self._detach_updated_state) and m_coeff is not None:
            m_coeff = m_coeff.detach()
        return grads, m_coeff

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        fastw: Tuple[torch.Tensor, ...] | None = None,
        seqlen_offset: int = 0,
        router_proj_weights: torch.Tensor | None = None,
        routing_bias: torch.Tensor | None = None,
        precomputed_w0_w2: torch.Tensor | None = None,
        **kwargs: Any,
    ):
        if self.memory is None or fastw is None:
            raise ValueError("Global-v4.5 requires an initialized model-owned memory pool.")

        if self.config.residual_style == "parallel":
            residual = hidden_states
            x_attn_in = self.attn_norm(hidden_states)
            attn_out, attn_k, attn_v, attn_q = self.attn(
                hidden_states=x_attn_in,
                kv_cache=kv_cache,
            )

            x_memory_in = x_attn_in
            if self.config.memory_kv_mode == "reuse_kv":
                q = attn_q
            elif self.config.memory_kv_proj == "reuse_attn_kv_proj":
                q = self.attention_toq(x_memory_in)
            else:
                q = self.memory_toq(x_memory_in)

            memory_out = self.memory(
                q,
                fastw,
                seqlen_offset=seqlen_offset,
                router_proj_weights=router_proj_weights,
                routing_bias=routing_bias,
                precomputed_w0_w2=precomputed_w0_w2,
            )
            if self.memory.learnable_ttt_scale:
                memory_out = self.memory.apply_ttt_scale(memory_out, x_attn_in)

            hidden_states = memory_out + attn_out
            if self.enable_attention_memory_output_proj:
                hidden_states = residual + self.memory_output_proj(hidden_states)
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

            residual = hidden_states
            x_memory_in = self.memory_norm(hidden_states)
            if self.config.memory_kv_mode == "reuse_kv":
                q = attn_q
            elif self.config.memory_kv_proj == "reuse_attn_kv_proj":
                q = self.attention_toq(x_memory_in)
            else:
                q = self.memory_toq(x_memory_in)

            memory_out = self.memory(
                q,
                fastw,
                seqlen_offset=seqlen_offset,
                router_proj_weights=router_proj_weights,
                routing_bias=routing_bias,
                precomputed_w0_w2=precomputed_w0_w2,
            )
            if self.memory.learnable_ttt_scale:
                memory_out = self.memory.apply_ttt_scale(memory_out, x_memory_in)

            hidden_states = memory_out
            if self.enable_attention_memory_output_proj:
                hidden_states = residual + self.memory_output_proj(hidden_states)
            else:
                hidden_states = residual + hidden_states
        else:
            raise ValueError(f"Invalid residual_style: {self.config.residual_style}")

        residual = hidden_states
        hidden_states = self.mlp_norm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states, **kwargs)

        return hidden_states, x_attn_in, x_memory_in, attn_k, attn_v, fastw, None, None

    def _update_fast_weight(
        self,
        pre_vi: torch.Tensor,
        fastw: Tuple[torch.Tensor, ...],
        masterw: Tuple[torch.Tensor, ...],
        momentum_buf: Tuple[torch.Tensor, ...] | None = None,
        ki: torch.Tensor = None,
        vi: torch.Tensor = None,
        seqlen_offset: int = 0,
        router_proj_weights: torch.Tensor | None = None,
        routing_bias: torch.Tensor | None = None,
        precomputed_w0_w2: torch.Tensor | None = None,
    ) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...] | None]:
        if self.memory is None:
            return fastw, masterw, momentum_buf

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
            router_proj_weights=router_proj_weights,
            routing_bias=routing_bias,
            precomputed_w0_w2=precomputed_w0_w2,
        )

    def update_fast_weight(
        self,
        pre_vi: torch.Tensor,
        fastw: Tuple[torch.Tensor, ...],
        masterw: Tuple[torch.Tensor, ...],
        momentum_buf: Tuple[torch.Tensor, ...] | None = None,
        ki: torch.Tensor = None,
        vi: torch.Tensor = None,
        seqlen_offset: int = 0,
        router_proj_weights: torch.Tensor | None = None,
        routing_bias: torch.Tensor | None = None,
        precomputed_w0_w2: torch.Tensor | None = None,
    ):
        if self.memory is None:
            return fastw, masterw, momentum_buf
        if not isinstance(self.memory, FastWeightMLPSubNN) or not self.memory.use_moe:
            return checkpoint(
                self._update_fast_weight,
                pre_vi,
                fastw,
                masterw,
                momentum_buf=momentum_buf,
                ki=ki,
                vi=vi,
                seqlen_offset=seqlen_offset,
                router_proj_weights=router_proj_weights,
                routing_bias=routing_bias,
                preserve_rng_state=False,
                use_reentrant=False,
            )
        if router_proj_weights is None:
            raise ValueError("global-v4.5 MoE update requires model-owned router weights.")

        lri = self.to_lr(pre_vi)
        lri = F.softplus(lri.float() + self.base_lr_inv)
        k_tokens, v_tokens, lr_tokens = self.memory._preprocess_update_inputs(
            ki,
            vi,
            lri,
            seqlen_offset=seqlen_offset,
        )
        router_input = v_tokens if self.memory.memory_router_from_v else k_tokens
        expert_mask, group_sizes, router_probs, balance_probs = self.memory._route_tokens(
            router_input,
            router_proj_weights,
            routing_bias=routing_bias,
        )
        self.memory._last_update_lb_info = self.memory._make_lb_info(balance_probs, group_sizes)
        self.memory._last_update_route_stats = self.memory._make_route_stats(
            router_probs,
            balance_probs,
            expert_mask,
        )

        m_coeff = None
        if self.memory.use_momentum and pre_vi is not None:
            m_coeff = self.memory.momentum_proj(
                pre_vi.to(self.memory.momentum_proj[0].weight.dtype),
            ).mean(dim=1).view(ki.shape[0], -1, 1, 1)

        def _inner(
            k_tokens,
            v_tokens,
            lr_tokens,
            expert_mask,
            router_probs,
            precomputed_w0_w2,
            m_coeff,
        ):
            w0_master, w1_master, w2_master = masterw
            w0, w1, w2 = fastw
            if self.memory.use_grouped_moe:
                grads = self.memory._compute_moe_update_grads_grouped(
                    k_tokens,
                    v_tokens,
                    lr_tokens,
                    w0,
                    w1,
                    w2,
                    expert_mask,
                    group_sizes,
                    router_probs,
                    w0_master,
                    w1_master,
                    w2_master,
                    precomputed_w0_w2=precomputed_w0_w2,
                )
            elif self.memory.use_dense_fused_moe:
                grads = self.memory._compute_moe_update_grads_dense(
                    k_tokens,
                    v_tokens,
                    lr_tokens,
                    w0,
                    w1,
                    w2,
                    expert_mask,
                    router_probs,
                    w0_master,
                    w1_master,
                    w2_master,
                )
            else:
                grads = self.memory._compute_moe_update_grads_loop(
                    k_tokens,
                    v_tokens,
                    lr_tokens,
                    w0,
                    w1,
                    w2,
                    expert_mask,
                    router_probs,
                    w0_master,
                    w1_master,
                    w2_master,
                )
            grads, update_m_coeff = self._apply_grad_detach_policy(grads, m_coeff)
            return self.memory.apply_update(
                grads,
                fastw,
                masterw,
                momentum_buf=momentum_buf,
                m_coeff=update_m_coeff,
            )

        if self._detach_updated_state:
            new_fastw, new_masterw, new_momentum_buf = _inner(
                k_tokens,
                v_tokens,
                lr_tokens,
                expert_mask,
                router_probs,
                precomputed_w0_w2,
                m_coeff,
            )
            return (
                tuple(weight.detach() for weight in new_fastw),
                tuple(weight.detach() for weight in new_masterw),
                None
                if new_momentum_buf is None
                else tuple(weight.detach() for weight in new_momentum_buf),
            )

        return checkpoint(
            _inner,
            k_tokens,
            v_tokens,
            lr_tokens,
            expert_mask,
            router_probs,
            precomputed_w0_w2,
            m_coeff,
            preserve_rng_state=False,
            use_reentrant=False,
        )

    def compute_raw_grad_only(
        self,
        pre_vi: torch.Tensor,
        fastw: Tuple[torch.Tensor, ...],
        masterw: Tuple[torch.Tensor, ...],
        ki: torch.Tensor,
        vi: torch.Tensor,
        seqlen_offset: int,
        router_proj_weights: torch.Tensor,
        routing_bias: torch.Tensor | None = None,
        precomputed_w0_w2: torch.Tensor | None = None,
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        torch.Tensor | None,
        tuple[torch.Tensor, torch.Tensor],
    ]:
        if self.memory is None:
            raise ValueError("global-v4.5 aggregate update requires a memory module.")
        if not isinstance(self.memory, FastWeightMLPSubNN) or not self.memory.use_moe:
            raise ValueError("global-v4.5 aggregate update is only implemented for MoE memory.")
        if router_proj_weights is None:
            raise ValueError("global-v4.5 MoE update requires model-owned router weights.")

        lri = self.to_lr(pre_vi)
        lri = F.softplus(lri.float() + self.base_lr_inv)
        k_tokens, v_tokens, lr_tokens = self.memory._preprocess_update_inputs(
            ki,
            vi,
            lri,
            seqlen_offset=seqlen_offset,
        )
        router_input = v_tokens if self.memory.memory_router_from_v else k_tokens
        expert_mask, group_sizes, router_probs, balance_probs = self.memory._route_tokens(
            router_input,
            router_proj_weights,
            routing_bias=routing_bias,
        )
        update_lb_info = self.memory._make_lb_info(balance_probs, group_sizes)
        update_route_stats = self.memory._make_route_stats(
            router_probs,
            balance_probs,
            expert_mask,
        )

        m_coeff = None
        if self.memory.use_momentum and pre_vi is not None:
            m_coeff = self.memory.momentum_proj(
                pre_vi.to(self.memory.momentum_proj[0].weight.dtype),
            ).mean(dim=1).view(ki.shape[0], -1, 1, 1)

        def _inner(k_tokens, v_tokens, lr_tokens, expert_mask, router_probs, precomputed_w0_w2):
            w0_master, w1_master, w2_master = masterw
            w0, w1, w2 = fastw
            if self.memory.use_grouped_moe:
                return self.memory._compute_moe_update_grads_grouped(
                    k_tokens,
                    v_tokens,
                    lr_tokens,
                    w0,
                    w1,
                    w2,
                    expert_mask,
                    group_sizes,
                    router_probs,
                    w0_master,
                    w1_master,
                    w2_master,
                    precomputed_w0_w2=precomputed_w0_w2,
                )
            if self.memory.use_dense_fused_moe:
                return self.memory._compute_moe_update_grads_dense(
                    k_tokens,
                    v_tokens,
                    lr_tokens,
                    w0,
                    w1,
                    w2,
                    expert_mask,
                    router_probs,
                    w0_master,
                    w1_master,
                    w2_master,
                )
            return self.memory._compute_moe_update_grads_loop(
                k_tokens,
                v_tokens,
                lr_tokens,
                w0,
                w1,
                w2,
                expert_mask,
                router_probs,
                w0_master,
                w1_master,
                w2_master,
            )

        if self._detach_update_grads:
            grads = _inner(
                k_tokens,
                v_tokens,
                lr_tokens,
                expert_mask,
                router_probs,
                precomputed_w0_w2,
            )
        else:
            grads = checkpoint(
                _inner,
                k_tokens,
                v_tokens,
                lr_tokens,
                expert_mask,
                router_probs,
                precomputed_w0_w2,
                preserve_rng_state=False,
                use_reentrant=False,
            )
        grads, m_coeff = self._apply_grad_detach_policy(grads, m_coeff)

        self.memory._last_update_route_stats = update_route_stats
        return grads, m_coeff, update_lb_info


def _init_global_v4_5_weights(
    self,
    module: nn.Module,
    rescale_prenorm_residual: bool = False,
    num_residuals_per_layer: int = 2,
):
    if isinstance(module, FastWeightMLPSubNN):
        if module.qk_rescale:
            nn.init.ones_(module.qk_scale)
            nn.init.zeros_(module.qk_offset)
        return

    base.RecurrentLactRefPreTrainedModel._init_weights(
        self,
        module,
        rescale_prenorm_residual=rescale_prenorm_residual,
        num_residuals_per_layer=num_residuals_per_layer,
    )


class RecurrentLactRefMoeGlobalV45Model(base.RecurrentLactRefModel):
    config_class = RecurrentLactRefMoeGlobalV45Config
    _no_split_modules = ["RecurrentLactRefMoeGlobalV45Block"]
    _init_weights = _init_global_v4_5_weights

    def __init__(self, config: RecurrentLactRefMoeGlobalV45Config):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [
                RecurrentLactRefMoeGlobalV45Block(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ],
        )
        first_memory = self.layers[0].memory
        self.memory_num_experts = config.memory_num_experts
        self.memory_router_share_mode = config.memory_router_share_mode
        self.memory_lb_loss_alpha = config.memory_lb_loss_alpha
        self.memory_lb_scope = config.memory_lb_scope
        self.memory_lb_mode = config.memory_lb_mode
        self.memory_loss_free_update_rate = config.memory_loss_free_update_rate
        self.memory_loss_free_bias_max = config.memory_loss_free_bias_max
        self.memory_loss_free_warmup_steps = config.memory_loss_free_warmup_steps
        self.memory_loss_free_update_interval = config.memory_loss_free_update_interval
        self.memory_update_backward_mode = config.memory_update_backward_mode
        self._detach_update_grads = self.memory_update_backward_mode in {
            "first_order_state",
            "stop_state",
        }
        self._detach_updated_state = self.memory_update_backward_mode == "stop_state"
        self._detach_m_coeff = config.memory_detach_m_coeff
        self.memory_update_aggregate_write = config.memory_update_aggregate_write
        self.memory_aggregate_momentum_mode = config.memory_aggregate_momentum_mode
        self.memory_aggregate_lr_scale = config.memory_aggregate_lr_scale
        self.w0_w2_low_rank = config.w0_w2_low_rank
        self.fw_init_gain = config.fw_init_gain
        self.pool_head_dim = first_memory.head_dim
        self.pool_inter_dim = first_memory.inter_dim
        self._init_pool_parameters()
        self.reset_parameters()
        self.layers.apply(self._init_weights)
        self._init_loss_free_buffers()

    def _init_loss_free_buffers(self) -> None:
        if self.memory_lb_scope == "pool":
            bias_shape = (self.memory_num_experts,)
        else:
            bias_shape = (len(self.layers), self.memory_num_experts)
        self.register_buffer(
            "memory_loss_free_bias",
            torch.zeros(bias_shape, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "memory_loss_free_pending_counts",
            torch.zeros(bias_shape, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "memory_loss_free_steps",
            torch.zeros((), dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "memory_loss_free_num_updates",
            torch.zeros((), dtype=torch.long),
            persistent=True,
        )

    def _keep_loss_free_float_buffers_fp32(self) -> None:
        for name in ("memory_loss_free_bias", "memory_loss_free_pending_counts"):
            buffer = self._buffers.get(name)
            if buffer is not None and buffer.dtype != torch.float32:
                self._buffers[name] = buffer.float()

    def _apply(self, fn):
        module = super()._apply(fn)
        self._keep_loss_free_float_buffers_fp32()
        return module

    def _loss_free_enabled(self) -> bool:
        return (
            self.memory_lb_mode == "loss_free"
            and self.memory_loss_free_update_rate > 0
            and self.memory_num_experts > 1
        )

    def _get_router_bias(self, layer_idx: int) -> torch.Tensor | None:
        if not self._loss_free_enabled():
            return None
        if self.memory_lb_scope == "pool":
            bias = self.memory_loss_free_bias
        else:
            bias = self.memory_loss_free_bias[layer_idx]
        return bias.view(1, self.memory_num_experts, 1)

    def _loss_free_counts_for_scope(
        self,
        apply_accum_freqs: torch.Tensor,
        update_accum_freqs: torch.Tensor,
    ) -> torch.Tensor:
        counts = apply_accum_freqs.sum(dim=1) + update_accum_freqs.sum(dim=1)
        if self.memory_lb_scope == "pool":
            counts = counts.sum(dim=0)
        return counts.to(dtype=torch.float32)

    @staticmethod
    def _sync_loss_free_counts(counts: torch.Tensor) -> torch.Tensor:
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        return counts

    def _maybe_update_loss_free_bias(self) -> None:
        if not self.training or not self._loss_free_enabled():
            return
        with torch.no_grad():
            counts = self.memory_loss_free_pending_counts
            if counts.sum() <= 0:
                return
            steps = int(self.memory_loss_free_steps.item())
            if steps <= self.memory_loss_free_warmup_steps:
                counts.zero_()
                return
            effective_step = steps - self.memory_loss_free_warmup_steps
            if effective_step % self.memory_loss_free_update_interval != 0:
                counts.zero_()
                return

            bias = self.memory_loss_free_bias
            target = counts.mean() if self.memory_lb_scope == "pool" else counts.mean(dim=-1, keepdim=True)
            bias.add_(self.memory_loss_free_update_rate * torch.sign(target - counts))
            if self.memory_lb_scope == "pool":
                bias.sub_(bias.mean())
            else:
                bias.sub_(bias.mean(dim=-1, keepdim=True))
            if self.memory_loss_free_bias_max >= 0:
                bias.clamp_(
                    min=-self.memory_loss_free_bias_max,
                    max=self.memory_loss_free_bias_max,
                )
            counts.zero_()
            self.memory_loss_free_num_updates.add_(1)

    def _store_loss_free_counts(
        self,
        apply_accum_freqs: torch.Tensor,
        update_accum_freqs: torch.Tensor,
    ) -> None:
        if not self.training or not self._loss_free_enabled():
            return
        with torch.no_grad():
            counts = self._loss_free_counts_for_scope(apply_accum_freqs, update_accum_freqs)
            counts = counts.to(
                device=self.memory_loss_free_pending_counts.device,
                dtype=self.memory_loss_free_pending_counts.dtype,
            )
            self._sync_loss_free_counts(counts)
            self.memory_loss_free_pending_counts.copy_(counts)
            self.memory_loss_free_steps.add_(1)

    def _init_pool_parameters(self):
        if self.w0_w2_low_rank > 0:
            self.pool_w0 = base.LowRankFastWeight(
                self.memory_num_experts,
                self.pool_inter_dim,
                self.pool_head_dim,
                rank=self.w0_w2_low_rank,
                init_gain=self.fw_init_gain,
                add_identity=True,
            )
            self.pool_w2 = base.LowRankFastWeight(
                self.memory_num_experts,
                self.pool_inter_dim,
                self.pool_head_dim,
                rank=self.w0_w2_low_rank,
                init_gain=self.fw_init_gain,
                add_identity=True,
            )
        else:
            self.pool_w0 = nn.Parameter(
                torch.empty(self.memory_num_experts, self.pool_inter_dim, self.pool_head_dim),
            )
            self.pool_w2 = nn.Parameter(
                torch.empty(self.memory_num_experts, self.pool_inter_dim, self.pool_head_dim),
            )
        self.pool_w1 = nn.Parameter(
            torch.empty(self.memory_num_experts, self.pool_head_dim, self.pool_inter_dim),
        )

        if self.memory_router_share_mode == "share_all":
            self.shared_router_proj_weights = nn.Parameter(
                torch.empty(1, self.memory_num_experts, self.pool_head_dim),
            )
            self.per_layer_router_proj_weights = None
        else:
            self.shared_router_proj_weights = None
            self.per_layer_router_proj_weights = nn.Parameter(
                torch.empty(len(self.layers), 1, self.memory_num_experts, self.pool_head_dim),
            )

    def reset_parameters(self):
        if not hasattr(self, "pool_w1"):
            return

        if self.w0_w2_low_rank > 0:
            self.pool_w0.reset_parameters()
            self.pool_w2.reset_parameters()
        else:
            nn.init.normal_(self.pool_w0, mean=0.0, std=1.0 / math.sqrt(self.pool_head_dim))
            nn.init.normal_(self.pool_w2, mean=0.0, std=1.0 / math.sqrt(self.pool_head_dim))

        nn.init.normal_(self.pool_w1, mean=0.0, std=1.0 / math.sqrt(self.pool_inter_dim))

        if self.memory_router_share_mode == "share_all":
            nn.init.normal_(
                self.shared_router_proj_weights,
                mean=0.0,
                std=1.0 / math.sqrt(self.pool_head_dim),
            )
        else:
            nn.init.normal_(
                self.per_layer_router_proj_weights,
                mean=0.0,
                std=1.0 / math.sqrt(self.pool_head_dim),
            )

    def _get_router(self, layer_idx: int) -> torch.Tensor:
        if self.memory_router_share_mode == "share_all":
            return self.shared_router_proj_weights
        return self.per_layer_router_proj_weights[layer_idx]

    def _init_pool_fast_weights(self, batch_size: int, device: torch.device):
        base_w0 = self.pool_w0() if self.w0_w2_low_rank > 0 else self.pool_w0
        base_w2 = self.pool_w2() if self.w0_w2_low_rank > 0 else self.pool_w2
        masterw = (
            base_w0.to(device=device, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1, 1, 1),
            self.pool_w1.to(device=device, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1, 1, 1),
            base_w2.to(device=device, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1, 1, 1),
        )
        fastw = tuple(w.to(torch.bfloat16) for w in masterw)
        if self.layers[0].memory.use_momentum:
            momentum_buf = tuple(torch.zeros_like(w) for w in masterw)
        else:
            momentum_buf = None
        return fastw, masterw, momentum_buf

    @staticmethod
    def _precompute_w0_w2(
        fastw: Tuple[torch.Tensor, ...],
    ) -> torch.Tensor | None:
        if not _PRECOMPUTE_W0_W2:
            return None
        return torch.cat([fastw[0], fastw[2]], dim=2).contiguous()

    @staticmethod
    def _zero_lb_accumulators(
        num_layers: int,
        batch_size: int,
        num_experts: int,
        device: torch.device,
    ):
        shape = (num_layers, batch_size, num_experts)
        return (
            torch.zeros(shape, device=device, dtype=torch.float32),
            torch.zeros(shape, device=device, dtype=torch.float32),
        )

    @staticmethod
    def _accumulate_lb_info(
        accum_probs: torch.Tensor,
        accum_freqs: torch.Tensor,
        layer_idx: int,
        lb_info: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> None:
        if lb_info is None:
            return
        probs, freqs = lb_info
        accum_probs[layer_idx] = accum_probs[layer_idx] + probs.float()
        accum_freqs[layer_idx] = accum_freqs[layer_idx] + freqs.float()

    @staticmethod
    def _global_lb_loss(
        accum_probs: torch.Tensor,
        accum_freqs: torch.Tensor,
        num_experts: int,
    ) -> torch.Tensor:
        if accum_freqs.sum() <= 0:
            return accum_probs.new_zeros(())
        total_probs = accum_probs.sum(dim=1)
        total_freqs = accum_freqs.sum(dim=1)
        probs_denom = total_probs.sum(dim=1, keepdim=True).clamp_min(1e-9)
        freqs_denom = total_freqs.sum(dim=1, keepdim=True).clamp_min(1e-9)
        probs_mean = total_probs / probs_denom
        freqs_mean = total_freqs / freqs_denom
        return ((probs_mean * freqs_mean).sum(dim=1) * num_experts).sum()

    @staticmethod
    def _pool_lb_loss(
        accum_probs: torch.Tensor,
        accum_freqs: torch.Tensor,
        num_experts: int,
    ) -> torch.Tensor:
        if accum_freqs.sum() <= 0:
            return accum_probs.new_zeros(())
        total_probs = accum_probs.sum(dim=(0, 1))
        total_freqs = accum_freqs.sum(dim=(0, 1))
        probs = total_probs / total_probs.sum().clamp_min(1e-9)
        freqs = total_freqs / total_freqs.sum().clamp_min(1e-9)
        return (probs * freqs).sum() * num_experts

    def _lb_loss_for_scope(
        self,
        accum_probs: torch.Tensor,
        accum_freqs: torch.Tensor,
    ) -> torch.Tensor:
        if self.memory_lb_scope == "pool":
            return self._pool_lb_loss(accum_probs, accum_freqs, self.memory_num_experts)
        return self._global_lb_loss(accum_probs, accum_freqs, self.memory_num_experts)

    @staticmethod
    def _zero_route_stats(device: torch.device) -> dict[str, torch.Tensor]:
        zero = torch.zeros((), device=device, dtype=torch.float32)
        return {
            "selected_weight_sum": zero.clone(),
            "selected_weight_count": zero.clone(),
            "selected_weight_min": torch.full((), float("inf"), device=device, dtype=torch.float32),
            "selected_weight_max": torch.full((), float("-inf"), device=device, dtype=torch.float32),
            "selected_weight_p50_sum": zero.clone(),
            "selected_weight_p95_sum": zero.clone(),
            "router_entropy_sum": zero.clone(),
            "router_entropy_count": zero.clone(),
        }

    @staticmethod
    def _accumulate_route_stats(
        accum: dict[str, torch.Tensor] | None,
        stats: dict[str, torch.Tensor] | None,
    ) -> None:
        if accum is None or stats is None:
            return
        count = stats["selected_weight_count"].to(accum["selected_weight_count"].device)
        accum["selected_weight_sum"] = accum["selected_weight_sum"] + stats["selected_weight_sum"].to(
            accum["selected_weight_sum"].device,
        )
        accum["selected_weight_count"] = accum["selected_weight_count"] + count
        has_selected = count > 0
        candidate_min = torch.minimum(
            accum["selected_weight_min"],
            stats["selected_weight_min"].to(accum["selected_weight_min"].device),
        )
        candidate_max = torch.maximum(
            accum["selected_weight_max"],
            stats["selected_weight_max"].to(accum["selected_weight_max"].device),
        )
        accum["selected_weight_min"] = torch.where(has_selected, candidate_min, accum["selected_weight_min"])
        accum["selected_weight_max"] = torch.where(has_selected, candidate_max, accum["selected_weight_max"])
        accum["selected_weight_p50_sum"] = (
            accum["selected_weight_p50_sum"]
            + stats["selected_weight_p50_sum"].to(accum["selected_weight_p50_sum"].device)
        )
        accum["selected_weight_p95_sum"] = (
            accum["selected_weight_p95_sum"]
            + stats["selected_weight_p95_sum"].to(accum["selected_weight_p95_sum"].device)
        )
        accum["router_entropy_sum"] = accum["router_entropy_sum"] + stats["router_entropy_sum"].to(
            accum["router_entropy_sum"].device,
        )
        accum["router_entropy_count"] = (
            accum["router_entropy_count"]
            + stats["router_entropy_count"].to(accum["router_entropy_count"].device)
        )

    @staticmethod
    def _finalize_route_stats(
        accum: dict[str, torch.Tensor],
        prefix: str,
    ) -> dict[str, torch.Tensor]:
        zero = accum["selected_weight_sum"].new_zeros(())
        count = accum["selected_weight_count"]
        entropy_count = accum["router_entropy_count"]
        has_selected = count > 0
        return {
            f"{prefix}_selected_weight_mean": torch.where(
                has_selected,
                accum["selected_weight_sum"] / count.clamp_min(1.0),
                zero,
            ),
            f"{prefix}_selected_weight_min": torch.where(has_selected, accum["selected_weight_min"], zero),
            f"{prefix}_selected_weight_max": torch.where(has_selected, accum["selected_weight_max"], zero),
            f"{prefix}_selected_weight_p50": torch.where(
                has_selected,
                accum["selected_weight_p50_sum"] / count.clamp_min(1.0),
                zero,
            ),
            f"{prefix}_selected_weight_p95": torch.where(
                has_selected,
                accum["selected_weight_p95_sum"] / count.clamp_min(1.0),
                zero,
            ),
            f"{prefix}_router_entropy": torch.where(
                entropy_count > 0,
                accum["router_entropy_sum"] / entropy_count.clamp_min(1.0),
                zero,
            ),
        }

    @staticmethod
    def _expert_usage_stats(
        accum_freqs: torch.Tensor,
        num_experts: int,
        prefix: str,
    ) -> dict[str, torch.Tensor]:
        zero = accum_freqs.new_zeros(())
        total_freqs = accum_freqs.sum(dim=1)
        if total_freqs.sum() <= 0:
            return {
                f"{prefix}_freq_min": zero,
                f"{prefix}_freq_max": zero,
                f"{prefix}_freq_std": zero,
                f"{prefix}_freq_cv": zero,
                f"{prefix}_freq_max_over_uniform": zero,
                f"{prefix}_dead_frac_0p25": zero,
                f"{prefix}_pool_freq_entropy": zero,
                f"{prefix}_entropy_ratio": zero,
                f"{prefix}_layer_freq_cv_mean": zero,
                f"{prefix}_layer_freq_cv_max": zero,
                f"{prefix}_layer_dead_frac_mean_0p25": zero,
                f"{prefix}_layer_dead_frac_max_0p25": zero,
            }

        uniform = 1.0 / float(num_experts)
        dead_threshold = 0.25 * uniform

        layer_denoms = total_freqs.sum(dim=1, keepdim=True).clamp_min(1e-9)
        layer_freqs = total_freqs / layer_denoms
        global_freqs = total_freqs.sum(dim=0)
        global_freqs = global_freqs / global_freqs.sum().clamp_min(1e-9)

        global_std = global_freqs.std(unbiased=False)
        global_mean = global_freqs.mean().clamp_min(1e-9)
        entropy_input = global_freqs.clamp_min(1e-12)
        entropy = -(entropy_input * entropy_input.log()).sum()
        max_entropy = math.log(num_experts) if num_experts > 1 else 1.0
        entropy_ratio = entropy / max_entropy

        layer_cv = layer_freqs.std(dim=1, unbiased=False) / layer_freqs.mean(
            dim=1,
        ).clamp_min(1e-9)
        layer_dead_frac = (layer_freqs < dead_threshold).float().mean(dim=1)

        return {
            f"{prefix}_freq_min": global_freqs.min(),
            f"{prefix}_freq_max": global_freqs.max(),
            f"{prefix}_freq_std": global_std,
            f"{prefix}_freq_cv": global_std / global_mean,
            f"{prefix}_freq_max_over_uniform": global_freqs.max() / uniform,
            f"{prefix}_dead_frac_0p25": (global_freqs < dead_threshold).float().mean(),
            f"{prefix}_pool_freq_entropy": entropy,
            f"{prefix}_entropy_ratio": entropy_ratio,
            f"{prefix}_layer_freq_cv_mean": layer_cv.mean(),
            f"{prefix}_layer_freq_cv_max": layer_cv.max(),
            f"{prefix}_layer_dead_frac_mean_0p25": layer_dead_frac.mean(),
            f"{prefix}_layer_dead_frac_max_0p25": layer_dead_frac.max(),
        }

    def _can_use_batched_per_layer_moe_update(
        self,
        caches: list[dict[str, Any]],
    ) -> bool:
        return False
        if getattr(self, "_disable_batched_moe_update", False):
            return False
        if not self.use_memory_block:
            return False
        if not self.layers:
            return False

        first_memory = self.layers[0].memory
        if not isinstance(first_memory, FastWeightMLPSubNN):
            return False
        if not first_memory.use_moe or not first_memory.use_dense_fused_moe:
            return False
        if first_memory.ttt_rotary is not None:
            return False

        keys = ("fastw", "masterw")
        for layer, cache in zip(self.layers, caches):
            memory = layer.memory
            if not isinstance(memory, FastWeightMLPSubNN):
                return False
            if not memory.use_moe or not memory.use_dense_fused_moe:
                return False
            if memory.ttt_rotary is not None:
                return False
            if any(cache[key] is None for key in keys):
                return False
            if memory.use_momentum and cache["momentum_buf"] is None:
                return False

        return True

    def _batched_preprocess_update_inputs(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        lr: torch.Tensor,
        memories: list[FastWeightMLPSubNN],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        first = memories[0]
        num_layers, batch_size, seq_len, _ = k.shape

        if first.qk_rescale:
            qk_scale = torch.stack([memory.qk_scale for memory in memories], dim=0)
            qk_offset = torch.stack([memory.qk_offset for memory in memories], dim=0)
            k = k * qk_scale[:, None, None, :, 1] + qk_offset[:, None, None, :, 1]

        if first.qkv_silu:
            k = F.silu(k)
            v = F.silu(v)

        if first.v_norm:
            v_shape = v.shape
            v = first.norm_fn(v.reshape(num_layers * batch_size, seq_len, v_shape[-1]))
            v = v.reshape(v_shape)

        if first.qk_norm:
            k = base.rearrange(
                k,
                "l b s (h d) -> (l b h) s d",
                h=first.fw_num_heads,
            )
            k = first.qk_norm_fn(k)
            k = base.rearrange(
                k,
                "(l b h) s d -> l b s (h d)",
                l=num_layers,
                b=batch_size,
                h=first.fw_num_heads,
            )

        k_tokens = base.rearrange(
            k,
            "l b s (h d) -> l b (s h) d",
            h=first.fw_num_heads,
        ).contiguous()
        v_tokens = base.rearrange(
            v,
            "l b s (h d) -> l b (s h) d",
            h=first.fw_num_heads,
        ).contiguous()

        if lr.shape[-1] == 3:
            lr_mh = lr.unsqueeze(3).expand(-1, -1, -1, first.fw_num_heads, -1)
        else:
            lr_mh = lr.view(num_layers, batch_size, seq_len, first.fw_num_heads, 3)
        lr_tokens = base.rearrange(
            lr_mh,
            "l b s h c -> l b (s h) c",
        ).contiguous()
        return k_tokens, v_tokens, lr_tokens

    def _batched_route_tokens(
        self,
        router_input: torch.Tensor,
        memories: list[FastWeightMLPSubNN],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        first = memories[0]
        router_weights = torch.stack(
            [memory.router_proj_weights.squeeze(0) for memory in memories],
            dim=0,
        )
        with torch.autocast(device_type=router_input.device.type, enabled=False):
            logits = torch.einsum(
                "led,lbtd->lbet",
                router_weights.float(),
                router_input.float(),
            )
        expert_mask, _, router_probs, balance_probs, _ = create_router_mask_sizes_probs(
            logits,
            topk=first.memory_num_active_experts,
            alpha=first.memory_router_alpha,
            use_sigmoid=first.memory_router_use_sigmoid,
            router_type=first.memory_router_type,
            combine_mode=first.memory_router_combine_mode,
            router_scale=first.memory_router_scale,
            norm_eps=first.memory_router_norm_eps,
        )
        return expert_mask, router_probs, balance_probs

    def _batched_compute_raw_grads(
        self,
        k_tokens: torch.Tensor,
        v_tokens: torch.Tensor,
        lr_tokens: torch.Tensor,
        fastw: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        masterw: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        memories: list[FastWeightMLPSubNN],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        first = memories[0]
        w0, w1, w2 = fastw
        w0_master, w1_master, w2_master = masterw
        router_input = v_tokens if first.memory_router_from_v else k_tokens
        expert_mask, router_probs, _ = self._batched_route_tokens(router_input, memories)
        route_weights = router_probs * expert_mask.to(router_probs.dtype)

        k_tokens = k_tokens.to(w0.dtype)
        v_weighted = (
            v_tokens.unsqueeze(2)
            * route_weights.to(v_tokens.dtype).unsqueeze(-1)
        ).to(w1.dtype)
        lr_tokens = lr_tokens.to(k_tokens.dtype)
        lr0 = lr_tokens[:, :, None, :, 0:1]
        lr1 = lr_tokens[:, :, None, :, 1:2]
        lr2 = lr_tokens[:, :, None, :, 2:3]

        gate_before_act = torch.einsum("lbtd,lbeid->lbeti", k_tokens, w0)
        hidden_before_mul = torch.einsum("lbtd,lbeid->lbeti", k_tokens, w2)
        hidden = F.silu(gate_before_act, inplace=False) * hidden_before_mul

        dhidden = torch.einsum("lbetd,lbedi->lbeti", v_weighted, w1)
        dhidden_before_mul = dhidden * F.silu(gate_before_act, inplace=False)
        dgate = dhidden * hidden_before_mul
        dgate_before_act = base.silu_backprop(dgate, gate_before_act)

        w1_grad_raw = torch.einsum(
            "lbetd,lbeti->lbedi",
            v_weighted,
            (hidden * lr1).to(v_weighted.dtype),
        ).to(w1_master.dtype)
        w0_grad_raw = torch.einsum(
            "lbeti,lbetd->lbeid",
            dgate_before_act,
            (k_tokens[:, :, None] * lr0).to(dgate_before_act.dtype),
        ).to(w0_master.dtype)
        w2_grad_raw = torch.einsum(
            "lbeti,lbetd->lbeid",
            dhidden_before_mul,
            (k_tokens[:, :, None] * lr2).to(dhidden_before_mul.dtype),
        ).to(w2_master.dtype)
        return w0_grad_raw, w1_grad_raw, w2_grad_raw

    def _batched_apply_moe_update_dense(
        self,
        grads: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        fastw: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        masterw: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        momentum_buf: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
        m_coeff: torch.Tensor | None,
        memories: list[FastWeightMLPSubNN],
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
    ]:
        first = memories[0]
        w0_master, w1_master, w2_master = masterw
        w0, w1, w2 = fastw
        w0_grad_raw, w1_grad_raw, w2_grad_raw = grads

        w0_norm = w0_master.norm(dim=4, keepdim=True)
        w1_norm = w1_master.norm(dim=4, keepdim=True)
        w2_norm = w2_master.norm(dim=4, keepdim=True)

        if first.use_momentum and momentum_buf is not None and m_coeff is not None:
            dw0_mom, dw1_mom, dw2_mom = momentum_buf
            w0_grad_raw = w0_grad_raw + dw0_mom * m_coeff
            w1_grad_raw = w1_grad_raw + dw1_mom * m_coeff
            w2_grad_raw = w2_grad_raw + dw2_mom * m_coeff
            momentum_buf = (
                w0_grad_raw.clone(),
                w1_grad_raw.clone(),
                w2_grad_raw.clone(),
            )

        if first.use_moun:
            num_layers, batch_size, num_experts = w0_grad_raw.shape[:3]
            w0_grad_flat = base.rearrange(
                w0_grad_raw,
                "l b e d1 d2 -> (l b e) d1 d2",
            )
            w1_grad_flat = base.rearrange(
                w1_grad_raw,
                "l b e d1 d2 -> (l b e) d1 d2",
            )
            w2_grad_flat = base.rearrange(
                w2_grad_raw,
                "l b e d1 d2 -> (l b e) d1 d2",
            )
            w0_grad_flat = base.zeropower_via_newtonschulz5(w0_grad_flat, 5)
            w1_grad_flat = base.zeropower_via_newtonschulz5(w1_grad_flat, 5)
            w2_grad_flat = base.zeropower_via_newtonschulz5(w2_grad_flat, 5)
            w0_grad = base.rearrange(
                w0_grad_flat,
                "(l b e) d1 d2 -> l b e d1 d2",
                l=num_layers,
                b=batch_size,
                e=num_experts,
            )
            w1_grad = base.rearrange(
                w1_grad_flat,
                "(l b e) d1 d2 -> l b e d1 d2",
                l=num_layers,
                b=batch_size,
                e=num_experts,
            )
            w2_grad = base.rearrange(
                w2_grad_flat,
                "(l b e) d1 d2 -> l b e d1 d2",
                l=num_layers,
                b=batch_size,
                e=num_experts,
            )
        else:
            w0_grad = w0_grad_raw
            w1_grad = w1_grad_raw
            w2_grad = w2_grad_raw

        w0_master = w0_master + w0_grad
        w1_master = w1_master + w1_grad
        w2_master = w2_master + w2_grad

        w0_master = w0_master / (w0_master.norm(dim=4, keepdim=True) + 1e-5) * w0_norm
        w1_master = w1_master / (w1_master.norm(dim=4, keepdim=True) + 1e-5) * w1_norm
        w2_master = w2_master / (w2_master.norm(dim=4, keepdim=True) + 1e-5) * w2_norm

        masterw = (w0_master, w1_master, w2_master)
        fastw = tuple(w.to(torch.bfloat16) for w in masterw)
        return fastw, masterw, momentum_buf

    def _batched_update_memory_blocks_for_chunk(
        self,
        caches: list[dict[str, Any]],
        chunk_start: int,
    ) -> None:
        memories = [layer.memory for layer in self.layers]
        pre_updates = []
        ki_values = []
        vi_values = []
        lr_values = []

        for layer_idx, layer in enumerate(self.layers):
            pre_update, ki, vi = self._compose_memory_update_inputs(layer, layer_idx, caches)
            pre_updates.append(pre_update)
            ki_values.append(ki)
            vi_values.append(vi)
            lri = layer.to_lr(pre_update)
            lr_values.append(F.softplus(lri.float() + layer.base_lr_inv))

        pre_updates_stacked = torch.stack(pre_updates, dim=0)
        ki_stacked = torch.stack(ki_values, dim=0)
        vi_stacked = torch.stack(vi_values, dim=0)
        lr_stacked = torch.stack(lr_values, dim=0)

        fastw = tuple(
            torch.stack([cache["fastw"][idx] for cache in caches], dim=0)
            for idx in range(3)
        )
        masterw = tuple(
            torch.stack([cache["masterw"][idx] for cache in caches], dim=0)
            for idx in range(3)
        )
        if caches[0]["momentum_buf"] is None:
            momentum_buf = None
        else:
            momentum_buf = tuple(
                torch.stack([cache["momentum_buf"][idx] for cache in caches], dim=0)
                for idx in range(3)
            )

        k_tokens, v_tokens, lr_tokens = self._batched_preprocess_update_inputs(
            ki_stacked,
            vi_stacked,
            lr_stacked,
            memories,
        )
        grads = self._batched_compute_raw_grads(
            k_tokens,
            v_tokens,
            lr_tokens,
            fastw,
            masterw,
            memories,
        )

        m_coeff = None
        if memories[0].use_momentum and momentum_buf is not None:
            m_coeff = torch.stack(
                [
                    memory.momentum_proj(
                        pre_update.to(memory.momentum_proj[0].weight.dtype),
                    ).mean(dim=1).view(pre_update.shape[0], -1, 1, 1)
                    for memory, pre_update in zip(memories, pre_updates_stacked)
                ],
                dim=0,
            )

        fastw, masterw, momentum_buf = self._batched_apply_moe_update_dense(
            grads,
            fastw,
            masterw,
            momentum_buf,
            m_coeff,
            memories,
        )

        for layer_idx, cache in enumerate(caches):
            cache["fastw"] = tuple(weight[layer_idx].contiguous() for weight in fastw)
            cache["masterw"] = tuple(weight[layer_idx].contiguous() for weight in masterw)
            cache["momentum_buf"] = (
                None
                if momentum_buf is None
                else tuple(weight[layer_idx].contiguous() for weight in momentum_buf)
            )

    def _update_memory_blocks_for_chunk(
        self,
        caches: list[dict[str, Any]],
        chunk_start: int,
        seq_len: int,
    ) -> None:
        is_last_chunk = (chunk_start + self.chunk_size >= seq_len)
        if not self.use_memory_block or is_last_chunk:
            return
        if not self._can_use_batched_per_layer_moe_update(caches):
            return super()._update_memory_blocks_for_chunk(caches, chunk_start, seq_len)
        self._batched_update_memory_blocks_for_chunk(caches, chunk_start)

    def _update_global_memory_pool_for_chunk_aggregate_inner(
        self,
        caches: list[dict[str, Any]],
        chunk_start: int,
        fastw: Tuple[torch.Tensor, ...],
        masterw: Tuple[torch.Tensor, ...],
        momentum_buf: Tuple[torch.Tensor, ...] | None,
        update_accum_probs: torch.Tensor,
        update_accum_freqs: torch.Tensor,
        update_route_stats: dict[str, torch.Tensor] | None,
    ) -> tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...] | None]:
        num_layers = len(self.layers)
        pre_updates = []
        ki_values = []
        vi_values = []
        for layer_idx, layer in enumerate(self.layers):
            pre_update, ki, vi = self._compose_memory_update_inputs(layer, layer_idx, caches)
            pre_updates.append(pre_update)
            ki_values.append(ki)
            vi_values.append(vi)

        memory = self.layers[0].memory
        include_momentum_buf = momentum_buf is not None

        def _aggregate_update_inner(*args):
            arg_idx = 0
            inner_pre_updates = args[arg_idx : arg_idx + num_layers]
            arg_idx += num_layers
            inner_ki_values = args[arg_idx : arg_idx + num_layers]
            arg_idx += num_layers
            inner_vi_values = args[arg_idx : arg_idx + num_layers]
            arg_idx += num_layers
            inner_fastw = tuple(args[arg_idx : arg_idx + 3])
            arg_idx += 3
            inner_masterw = tuple(args[arg_idx : arg_idx + 3])
            arg_idx += 3
            inner_momentum_buf = None
            if include_momentum_buf:
                inner_momentum_buf = tuple(args[arg_idx : arg_idx + 3])

            agg_grads = [torch.zeros_like(weight) for weight in inner_masterw]
            m_coeffs = []
            update_probs = []
            update_freqs = []
            precomputed_w0_w2 = self._precompute_w0_w2(inner_fastw)

            for inner_layer_idx, inner_layer in enumerate(self.layers):
                raw_grads, m_coeff, update_lb_info = inner_layer.compute_raw_grad_only(
                    inner_pre_updates[inner_layer_idx],
                    inner_fastw,
                    inner_masterw,
                    ki=inner_ki_values[inner_layer_idx],
                    vi=inner_vi_values[inner_layer_idx],
                    seqlen_offset=chunk_start,
                    router_proj_weights=self._get_router(inner_layer_idx),
                    routing_bias=self._get_router_bias(inner_layer_idx),
                    precomputed_w0_w2=precomputed_w0_w2,
                )
                for grad_idx, grad in enumerate(raw_grads):
                    agg_grads[grad_idx] = agg_grads[grad_idx] + grad
                if m_coeff is not None:
                    m_coeffs.append(m_coeff)
                probs, freqs = update_lb_info
                update_probs.append(probs.float())
                update_freqs.append(freqs.float())

            if self.memory_aggregate_momentum_mode == "none":
                pool_m_coeff = None
            elif self.memory_aggregate_momentum_mode == "pool_mean":
                pool_m_coeff = torch.stack(m_coeffs, dim=0).mean(dim=0) if m_coeffs else None
            else:
                raise ValueError(
                    f"Unsupported aggregate momentum mode: {self.memory_aggregate_momentum_mode}",
                )

            new_fastw, new_masterw, new_momentum_buf = memory.apply_update(
                tuple(agg_grads),
                inner_fastw,
                inner_masterw,
                momentum_buf=inner_momentum_buf,
                m_coeff=pool_m_coeff,
                lr_scale=self.memory_aggregate_lr_scale,
            )

            outputs = [
                *new_fastw,
                *new_masterw,
            ]
            if include_momentum_buf:
                outputs.extend(
                    new_momentum_buf if new_momentum_buf is not None else inner_momentum_buf,
                )
            outputs.extend(
                [
                    torch.stack(update_probs, dim=0),
                    torch.stack(update_freqs, dim=0),
                ],
            )
            return tuple(outputs)

        checkpoint_args = [
            *pre_updates,
            *ki_values,
            *vi_values,
            *fastw,
            *masterw,
        ]
        if include_momentum_buf:
            checkpoint_args.extend(momentum_buf)

        if self._detach_updated_state:
            outputs = _aggregate_update_inner(*checkpoint_args)
        else:
            outputs = checkpoint(
                _aggregate_update_inner,
                *checkpoint_args,
                preserve_rng_state=False,
                use_reentrant=False,
            )

        output_idx = 0
        new_fastw = tuple(outputs[output_idx : output_idx + 3])
        output_idx += 3
        new_masterw = tuple(outputs[output_idx : output_idx + 3])
        output_idx += 3
        new_momentum_buf = None
        if include_momentum_buf:
            new_momentum_buf = tuple(outputs[output_idx : output_idx + 3])
            output_idx += 3
        update_probs = outputs[output_idx]
        update_freqs = outputs[output_idx + 1]

        for layer_idx in range(num_layers):
            self._accumulate_lb_info(
                update_accum_probs,
                update_accum_freqs,
                layer_idx,
                (update_probs[layer_idx], update_freqs[layer_idx]),
            )
            self._accumulate_route_stats(
                update_route_stats,
                self.layers[layer_idx].memory._last_update_route_stats,
            )

        if self._detach_updated_state:
            new_fastw = tuple(weight.detach() for weight in new_fastw)
            new_masterw = tuple(weight.detach() for weight in new_masterw)
            new_momentum_buf = (
                None
                if new_momentum_buf is None
                else tuple(weight.detach() for weight in new_momentum_buf)
            )

        return new_fastw, new_masterw, new_momentum_buf

    def _update_global_memory_pool_for_chunk(
        self,
        caches: list[dict[str, Any]],
        chunk_start: int,
        seq_len: int,
        fastw: Tuple[torch.Tensor, ...],
        masterw: Tuple[torch.Tensor, ...],
        momentum_buf: Tuple[torch.Tensor, ...] | None,
        update_accum_probs: torch.Tensor,
        update_accum_freqs: torch.Tensor,
        update_route_stats: dict[str, torch.Tensor] | None,
    ) -> tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...] | None]:
        is_last_chunk = chunk_start + self.chunk_size >= seq_len
        if not self.use_memory_block or is_last_chunk:
            return fastw, masterw, momentum_buf

        if self.memory_update_aggregate_write:
            if _AGG_INNER_CHECKPOINT:
                return self._update_global_memory_pool_for_chunk_aggregate_inner(
                    caches,
                    chunk_start,
                    fastw,
                    masterw,
                    momentum_buf,
                    update_accum_probs,
                    update_accum_freqs,
                    update_route_stats,
                )

            agg_grads = [torch.zeros_like(weight) for weight in masterw]
            m_coeffs = []
            precomputed_w0_w2 = self._precompute_w0_w2(fastw)

            for layer_idx, layer in enumerate(self.layers):
                pre_update, ki, vi = self._compose_memory_update_inputs(layer, layer_idx, caches)
                raw_grads, m_coeff, update_lb_info = layer.compute_raw_grad_only(
                    pre_update,
                    fastw,
                    masterw,
                    ki=ki,
                    vi=vi,
                    seqlen_offset=chunk_start,
                    router_proj_weights=self._get_router(layer_idx),
                    routing_bias=self._get_router_bias(layer_idx),
                    precomputed_w0_w2=precomputed_w0_w2,
                )
                for grad_idx, grad in enumerate(raw_grads):
                    agg_grads[grad_idx] = agg_grads[grad_idx] + grad
                if m_coeff is not None:
                    m_coeffs.append(m_coeff)
                self._accumulate_lb_info(
                    update_accum_probs,
                    update_accum_freqs,
                    layer_idx,
                    update_lb_info,
                )
                self._accumulate_route_stats(update_route_stats, layer.memory._last_update_route_stats)

            if self.memory_aggregate_momentum_mode == "none":
                pool_m_coeff = None
            elif self.memory_aggregate_momentum_mode == "pool_mean":
                pool_m_coeff = torch.stack(m_coeffs, dim=0).mean(dim=0) if m_coeffs else None
            else:
                raise ValueError(
                    f"Unsupported aggregate momentum mode: {self.memory_aggregate_momentum_mode}",
                )

            memory = self.layers[0].memory
            if self._detach_updated_state:
                new_fastw, new_masterw, new_momentum_buf = memory.apply_update(
                    tuple(agg_grads),
                    fastw,
                    masterw,
                    momentum_buf=momentum_buf,
                    m_coeff=pool_m_coeff,
                    lr_scale=self.memory_aggregate_lr_scale,
                )
                new_fastw = tuple(weight.detach() for weight in new_fastw)
                new_masterw = tuple(weight.detach() for weight in new_masterw)
                new_momentum_buf = (
                    None
                    if new_momentum_buf is None
                    else tuple(weight.detach() for weight in new_momentum_buf)
                )
            else:
                include_pool_m_coeff = pool_m_coeff is not None
                include_momentum_buf = momentum_buf is not None

                def _apply_aggregate(agg_w0, agg_w1, agg_w2, *extra):
                    extra_idx = 0
                    ckpt_pool_m_coeff = None
                    if include_pool_m_coeff:
                        ckpt_pool_m_coeff = extra[extra_idx]
                        extra_idx += 1
                    ckpt_momentum_buf = None
                    if include_momentum_buf:
                        ckpt_momentum_buf = tuple(extra[extra_idx:extra_idx + 3])
                    return memory.apply_update(
                        (agg_w0, agg_w1, agg_w2),
                        fastw,
                        masterw,
                        momentum_buf=ckpt_momentum_buf,
                        m_coeff=ckpt_pool_m_coeff,
                        lr_scale=self.memory_aggregate_lr_scale,
                    )

                checkpoint_args = [agg_grads[0], agg_grads[1], agg_grads[2]]
                if include_pool_m_coeff:
                    checkpoint_args.append(pool_m_coeff)
                if include_momentum_buf:
                    checkpoint_args.extend(momentum_buf)
                new_fastw, new_masterw, new_momentum_buf = checkpoint(
                    _apply_aggregate,
                    *checkpoint_args,
                    preserve_rng_state=False,
                    use_reentrant=False,
                )
            return new_fastw, new_masterw, new_momentum_buf

        for layer_idx, layer in enumerate(self.layers):
            cache = caches[layer_idx]
            pre_update, ki, vi = self._compose_memory_update_inputs(layer, layer_idx, caches)
            fastw, masterw, momentum_buf = layer.update_fast_weight(
                pre_update,
                fastw,
                masterw,
                momentum_buf,
                ki=ki,
                vi=vi,
                seqlen_offset=chunk_start,
                router_proj_weights=self._get_router(layer_idx),
                routing_bias=self._get_router_bias(layer_idx),
                precomputed_w0_w2=self._precompute_w0_w2(fastw),
            )
            self._accumulate_lb_info(
                update_accum_probs,
                update_accum_freqs,
                layer_idx,
                layer.memory._last_update_lb_info,
            )
            self._accumulate_route_stats(update_route_stats, layer.memory._last_update_route_stats)
        return fastw, masterw, momentum_buf

    def _init_or_load_generation_caches(
        self,
        past_key_values: Any | None,
        batch_size: int,
        max_len: int,
        device: torch.device,
    ) -> tuple[list[dict[str, Any]], int, int]:
        cache_state = past_key_values if is_recurrent_lact_cache(past_key_values) else None
        seen_tokens = recurrent_lact_cache_seen_tokens(cache_state)
        pending_start = (
            int(cache_state.get("pending_start", seen_tokens))
            if cache_state is not None
            else seen_tokens
        )
        max_len = max(max_len, 1)

        if cache_state is None:
            caches = []
            for layer in self.layers:
                k_cache, v_cache = layer.attn.cache_init(
                    batch_size,
                    max_len,
                    device=device,
                    dtype=torch.bfloat16,
                )
                if layer.memory is not None:
                    layer.memory.init_rotary_cache(max_len, device=device, dtype=torch.bfloat16)
                caches.append(
                    {
                        "kv_cache": (k_cache, v_cache),
                        "xi_out": None,
                        "x_attn_in": None,
                        "x_memory_in": None,
                        "attn_k": None,
                        "attn_v": None,
                    },
                )
            return caches, seen_tokens, pending_start

        caches = [dict(layer_cache) for layer_cache in cache_state["layers"]]
        for layer, cache in zip(self.layers, caches, strict=True):
            k_cache, v_cache = cache["kv_cache"]
            cache_len = k_cache.size(1)
            layer.attn._ensure_storage(batch_size, max_len, device=device, dtype=torch.bfloat16)
            if cache_len > 0:
                layer.attn.k_storage.data[:batch_size, :cache_len] = k_cache.to(
                    device=device,
                    dtype=torch.bfloat16,
                )
                layer.attn.v_storage.data[:batch_size, :cache_len] = v_cache.to(
                    device=device,
                    dtype=torch.bfloat16,
                )
            cache["kv_cache"] = (
                layer.attn.k_storage[:batch_size, :cache_len],
                layer.attn.v_storage[:batch_size, :cache_len],
            )
            layer.attn.rotary._update_cos_sin_cache(max_len * 2, device=device, dtype=torch.bfloat16)
            if layer.memory is not None:
                layer.memory.init_rotary_cache(max_len, device=device, dtype=torch.bfloat16)
        return caches, seen_tokens, pending_start

    def _init_or_load_generation_global_state(
        self,
        past_key_values: Any | None,
        batch_size: int,
        device: torch.device,
    ) -> dict[str, Any]:
        cache_state = past_key_values if is_recurrent_lact_cache(past_key_values) else None
        if cache_state is not None and "global_state" in cache_state:
            return dict(cache_state["global_state"])

        fastw, masterw, momentum_buf = self._init_pool_fast_weights(batch_size, device)
        return {
            "fastw": fastw,
            "masterw": masterw,
            "momentum_buf": momentum_buf,
        }

    @staticmethod
    def _pending_len(caches: list[dict[str, Any]], seen_tokens: int, pending_start: int) -> int:
        if caches and caches[0].get("xi_out") is not None:
            return int(caches[0]["xi_out"].size(1))
        return max(0, int(seen_tokens - pending_start))

    @staticmethod
    def _clear_pending_generation_features(caches: list[dict[str, Any]]) -> None:
        for cache in caches:
            cache["xi_out"] = None
            cache["x_attn_in"] = None
            cache["x_memory_in"] = None
            cache["attn_k"] = None
            cache["attn_v"] = None

    @staticmethod
    def _store_generation_layer_features(
        cache: dict[str, Any],
        *,
        xi_out: torch.Tensor,
        x_attn_in: torch.Tensor,
        x_memory_in: torch.Tensor | None,
        attn_k: torch.Tensor,
        attn_v: torch.Tensor,
        append: bool,
    ) -> None:
        values = {
            "xi_out": xi_out,
            "x_attn_in": x_attn_in,
            "x_memory_in": x_memory_in,
            "attn_k": attn_k,
            "attn_v": attn_v,
        }
        for key, value in values.items():
            if value is None:
                cache[key] = None
            elif append and cache.get(key) is not None:
                cache[key] = torch.cat((cache[key], value), dim=1)
            else:
                cache[key] = value

    def _build_moe_stats(
        self,
        apply_lb_loss: torch.Tensor,
        update_lb_loss: torch.Tensor,
        lb_loss: torch.Tensor,
        apply_accum_freqs: torch.Tensor,
        update_accum_freqs: torch.Tensor,
        apply_route_stats: dict[str, torch.Tensor] | None,
        update_route_stats: dict[str, torch.Tensor] | None,
    ) -> dict[str, torch.Tensor]:
        if not _moe_stats_enabled():
            return {}

        moe_stats = {
            "apply_lb_loss": apply_lb_loss.detach(),
            "update_lb_loss": update_lb_loss.detach(),
            "lb_loss": lb_loss.detach(),
            "num_experts": apply_lb_loss.new_tensor(float(self.memory_num_experts)),
            "memory_lb_scope_pool": apply_lb_loss.new_tensor(float(self.memory_lb_scope == "pool")),
            "memory_lb_mode_loss": apply_lb_loss.new_tensor(float(self.memory_lb_mode == "loss")),
            "memory_lb_mode_loss_free": apply_lb_loss.new_tensor(float(self.memory_lb_mode == "loss_free")),
            "memory_lb_mode_none": apply_lb_loss.new_tensor(float(self.memory_lb_mode == "none")),
            "loss_free_update_rate": apply_lb_loss.new_tensor(float(self.memory_loss_free_update_rate)),
            "loss_free_bias_abs_max": self.memory_loss_free_bias.detach().abs().max(),
            "loss_free_bias_std": self.memory_loss_free_bias.detach().std(unbiased=False),
            "loss_free_pending_count_sum": self.memory_loss_free_pending_counts.detach().sum(),
            "loss_free_steps": self.memory_loss_free_steps.detach().to(dtype=torch.float32),
            "loss_free_num_updates": self.memory_loss_free_num_updates.detach().to(dtype=torch.float32),
        }
        moe_stats.update(
            {
                key: value.detach()
                for key, value in self._expert_usage_stats(
                    apply_accum_freqs,
                    self.memory_num_experts,
                    "apply",
                ).items()
            },
        )
        moe_stats.update(
            {
                key: value.detach()
                for key, value in self._expert_usage_stats(
                    update_accum_freqs,
                    self.memory_num_experts,
                    "update",
                ).items()
            },
        )
        if apply_route_stats is not None:
            moe_stats.update(
                {
                    key: value.detach()
                    for key, value in self._finalize_route_stats(
                        apply_route_stats,
                        "apply",
                    ).items()
                },
            )
        if update_route_stats is not None:
            moe_stats.update(
                {
                    key: value.detach()
                    for key, value in self._finalize_route_stats(
                        update_route_stats,
                        "update",
                    ).items()
                },
            )
        return moe_stats

    def _forward_with_generation_cache(
        self,
        *,
        inputs_embeds: torch.FloatTensor,
        past_key_values: Any | None,
        output_hidden_states: bool,
        return_dict: bool,
    ) -> tuple | MoeModelOutput:
        batch_size, seq_len, _ = inputs_embeds.shape
        total_target_len = recurrent_lact_cache_seen_tokens(past_key_values) + seq_len
        caches, seen_tokens, pending_start = self._init_or_load_generation_caches(
            past_key_values,
            batch_size,
            total_target_len,
            inputs_embeds.device,
        )
        pending_len = self._pending_len(caches, seen_tokens, pending_start)
        global_state = self._init_or_load_generation_global_state(
            past_key_values,
            batch_size,
            inputs_embeds.device,
        )
        fastw = global_state["fastw"]
        masterw = global_state["masterw"]
        momentum_buf = global_state["momentum_buf"]

        apply_accum_probs, apply_accum_freqs = self._zero_lb_accumulators(
            len(self.layers),
            batch_size,
            self.memory_num_experts,
            inputs_embeds.device,
        )
        update_accum_probs, update_accum_freqs = self._zero_lb_accumulators(
            len(self.layers),
            batch_size,
            self.memory_num_experts,
            inputs_embeds.device,
        )
        route_stats_enabled = _route_stats_enabled()
        apply_route_stats = self._zero_route_stats(inputs_embeds.device) if route_stats_enabled else None
        update_route_stats = self._zero_route_stats(inputs_embeds.device) if route_stats_enabled else None

        all_hidden_states = () if output_hidden_states else None
        chunk_outputs = []
        local_start = 0
        current_seen = seen_tokens

        while local_start < seq_len:
            if pending_len >= self.chunk_size:
                fastw, masterw, momentum_buf = self._update_global_memory_pool_for_chunk(
                    caches,
                    pending_start,
                    total_target_len,
                    fastw,
                    masterw,
                    momentum_buf,
                    update_accum_probs,
                    update_accum_freqs,
                    update_route_stats,
                )
                self._clear_pending_generation_features(caches)
                pending_start = current_seen
                pending_len = 0

            take = min(seq_len - local_start, self.chunk_size - pending_len)
            chunk_inputs = inputs_embeds[:, local_start:local_start + take]
            append_pending = pending_len > 0
            hidden_states = chunk_inputs
            precomputed_w0_w2 = self._precompute_w0_w2(fastw)

            for layer_idx, layer in enumerate(self.layers):
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                hidden_states, x_attn_in, x_memory_in, attn_k, attn_v, _, _, _ = layer(
                    hidden_states,
                    caches[layer_idx]["kv_cache"],
                    fastw,
                    current_seen,
                    self._get_router(layer_idx),
                    self._get_router_bias(layer_idx),
                    precomputed_w0_w2,
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
                self._accumulate_lb_info(
                    apply_accum_probs,
                    apply_accum_freqs,
                    layer_idx,
                    layer.memory._last_apply_lb_info,
                )
                self._accumulate_route_stats(apply_route_stats, layer.memory._last_apply_route_stats)

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

        global_state["fastw"] = fastw
        global_state["masterw"] = masterw
        global_state["momentum_buf"] = momentum_buf
        next_cache = make_recurrent_lact_cache(
            layers=caches,
            seen_tokens=current_seen,
            pending_start=pending_start,
            batch_size=batch_size,
            global_state=global_state,
        )

        apply_lb_loss = self._lb_loss_for_scope(apply_accum_probs, apply_accum_freqs)
        update_lb_loss = self._lb_loss_for_scope(update_accum_probs, update_accum_freqs)
        if self.memory_lb_mode == "loss":
            lb_loss = (apply_lb_loss + update_lb_loss) * self.memory_lb_loss_alpha
        else:
            lb_loss = apply_lb_loss.new_zeros(())
        self._store_loss_free_counts(apply_accum_freqs, update_accum_freqs)
        moe_stats = self._build_moe_stats(
            apply_lb_loss,
            update_lb_loss,
            lb_loss,
            apply_accum_freqs,
            update_accum_freqs,
            apply_route_stats,
            update_route_stats,
        )

        if not return_dict:
            values = [hidden_states, next_cache, all_hidden_states, None]
            return tuple(v for v in values if v is not None)

        return MoeModelOutput(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=None,
            lb_loss=lb_loss,
            moe_stats=moe_stats,
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
                "`RecurrentLactRefMoeGlobalV45Model` does not support output attention weights, "
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

        self._maybe_update_loss_free_bias()

        batch_size, seq_len, _ = inputs_embeds.shape
        if use_cache and not self.training:
            return self._forward_with_generation_cache(
                inputs_embeds=inputs_embeds,
                past_key_values=past_key_values,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        fastw, masterw, momentum_buf = self._init_pool_fast_weights(
            batch_size,
            inputs_embeds.device,
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
                    "xi_out": None,
                    "x_attn_in": None,
                    "x_memory_in": None,
                    "attn_k": None,
                    "attn_v": None,
                },
            )

        apply_accum_probs, apply_accum_freqs = self._zero_lb_accumulators(
            len(self.layers),
            batch_size,
            self.memory_num_experts,
            inputs_embeds.device,
        )
        update_accum_probs, update_accum_freqs = self._zero_lb_accumulators(
            len(self.layers),
            batch_size,
            self.memory_num_experts,
            inputs_embeds.device,
        )
        route_stats_enabled = _route_stats_enabled()
        apply_route_stats = self._zero_route_stats(inputs_embeds.device) if route_stats_enabled else None
        update_route_stats = self._zero_route_stats(inputs_embeds.device) if route_stats_enabled else None

        all_hidden_states = () if output_hidden_states else None
        chunk_outputs = []

        for chunk_start in range(0, seq_len, self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, seq_len)
            chunk_inputs = inputs_embeds[:, chunk_start:chunk_end]
            hidden_states = chunk_inputs
            precomputed_w0_w2 = self._precompute_w0_w2(fastw)

            for layer_idx, layer in enumerate(self.layers):
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                target = layer if chunk_start == 0 else layer.forward
                hidden_states, x_attn_in, x_memory_in, attn_k, attn_v, _, _, _ = checkpoint(
                    target,
                    hidden_states,
                    caches[layer_idx]["kv_cache"],
                    fastw,
                    chunk_start,
                    self._get_router(layer_idx),
                    self._get_router_bias(layer_idx),
                    precomputed_w0_w2,
                    preserve_rng_state=False,
                    use_reentrant=False,
                )
                caches[layer_idx]["xi_out"] = hidden_states
                caches[layer_idx]["x_attn_in"] = x_attn_in
                caches[layer_idx]["x_memory_in"] = x_memory_in
                caches[layer_idx]["attn_k"] = attn_k
                caches[layer_idx]["attn_v"] = attn_v
                self._accumulate_lb_info(
                    apply_accum_probs,
                    apply_accum_freqs,
                    layer_idx,
                    layer.memory._last_apply_lb_info,
                )
                self._accumulate_route_stats(apply_route_stats, layer.memory._last_apply_route_stats)

            chunk_outputs.append(hidden_states)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            for layer_idx, layer in enumerate(self.layers):
                if self.attention_v_gap is not None:
                    pre_vi = self.pre_attention_norm(caches[self.attention_v_layer_idxs[layer_idx]]["xi_out"])
                    if self.shared_kv_cache:
                        pre_vi = self.to_kv_cache(pre_vi)
                else:
                    pre_vi = caches[layer_idx]["x_attn_in"]

                caches[layer_idx]["kv_cache"] = layer.update_kv(
                    pre_vi,
                    caches[layer_idx]["kv_cache"],
                    apply_layer_kv=not self.shared_kv_cache,
                )

            fastw, masterw, momentum_buf = self._update_global_memory_pool_for_chunk(
                caches,
                chunk_start,
                seq_len,
                fastw,
                masterw,
                momentum_buf,
                update_accum_probs,
                update_accum_freqs,
                update_route_stats,
            )

        hidden_states = torch.cat(chunk_outputs, dim=1)
        hidden_states = self.norm(hidden_states)

        apply_lb_loss = self._lb_loss_for_scope(apply_accum_probs, apply_accum_freqs)
        update_lb_loss = self._lb_loss_for_scope(update_accum_probs, update_accum_freqs)
        if self.memory_lb_mode == "loss":
            lb_loss = (apply_lb_loss + update_lb_loss) * self.memory_lb_loss_alpha
        else:
            lb_loss = apply_lb_loss.new_zeros(())
        self._store_loss_free_counts(apply_accum_freqs, update_accum_freqs)
        moe_stats = self._build_moe_stats(
            apply_lb_loss,
            update_lb_loss,
            lb_loss,
            apply_accum_freqs,
            update_accum_freqs,
            apply_route_stats,
            update_route_stats,
        )

        if not return_dict:
            values = [hidden_states, None, all_hidden_states, None]
            return tuple(v for v in values if v is not None)

        return MoeModelOutput(
            last_hidden_state=hidden_states,
            past_key_values=None,
            hidden_states=all_hidden_states,
            attentions=None,
            lb_loss=lb_loss,
            moe_stats=moe_stats,
        )


class RecurrentLactRefMoeGlobalV45ForCausalLM(base.RecurrentLactRefForCausalLM):
    config_class = RecurrentLactRefMoeGlobalV45Config
    _init_weights = _init_global_v4_5_weights

    def __init__(self, config: RecurrentLactRefMoeGlobalV45Config):
        base.RecurrentLactRefPreTrainedModel.__init__(self, config)
        self.model = RecurrentLactRefMoeGlobalV45Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.criterion = None
        self.post_init()

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Any | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return prepare_recurrent_lact_generation_inputs(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            kwargs=kwargs,
        )

    @staticmethod
    def _reorder_cache(past_key_values: Any, beam_idx: torch.LongTensor) -> Any:
        return reorder_recurrent_lact_cache(past_key_values, beam_idx)

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
    ) -> tuple | CausalLMOutputWithMoeStats:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # The inner model accumulates hidden states per layer *per 4k chunk*,
        # so its tuple is fragments whose last entry is only the final chunk.
        # Consumers (e.g. flame.eval_loss's chunked lm_head path) expect
        # [-1] to be the full sequence, so never request the fragment tuple;
        # return the concatenated last_hidden_state instead.
        outputs = self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=False,
            return_dict=True,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        lb_loss = outputs.lb_loss
        moe_stats = outputs.moe_stats
        logits = None if self.config.fuse_linear_cross_entropy else self.lm_head(hidden_states[:, -logits_to_keep:])

        ce_loss = None
        loss = None
        if labels is not None:
            if getattr(self, "criterion", None) is None:
                if self.config.fuse_linear_cross_entropy:
                    criterion = base.FusedLinearCrossEntropyLoss(use_l2warp=self.config.use_l2warp)
                elif self.config.fuse_cross_entropy:
                    criterion = base.FusedCrossEntropyLoss(inplace_backward=True)
                else:
                    criterion = nn.CrossEntropyLoss()
            else:
                criterion = self.criterion
            labels = labels.to(hidden_states.device)
            labels = torch.cat((labels[..., 1:], torch.full_like(labels[:, :1], criterion.ignore_index)), 1)
            if self.config.fuse_linear_cross_entropy:
                ce_loss = criterion(hidden_states, labels, self.lm_head.weight, self.lm_head.bias)
            else:
                ce_loss = criterion(logits.view(labels.numel(), -1), labels.view(-1))
                ce_loss = base.l2_warp(ce_loss, logits) if self.config.use_l2warp else ce_loss
            loss = ce_loss + lb_loss if lb_loss is not None else ce_loss

        if not return_dict:
            output = (logits,) + tuple(
                value
                for value in (
                    outputs.past_key_values,
                    outputs.hidden_states,
                    outputs.attentions,
                )
                if value is not None
            )
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithMoeStats(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=(hidden_states,) if output_hidden_states else None,
            attentions=outputs.attentions,
            ce_loss=ce_loss.detach() if ce_loss is not None else None,
            lb_loss=lb_loss.detach() if lb_loss is not None else None,
            moe_stats=moe_stats,
        )


RecurrentLactRefModel = RecurrentLactRefMoeGlobalV45Model
RecurrentLactRefForCausalLM = RecurrentLactRefMoeGlobalV45ForCausalLM
RecurrentLactRefMoeModel = RecurrentLactRefMoeGlobalV45Model
RecurrentLactRefMoeForCausalLM = RecurrentLactRefMoeGlobalV45ForCausalLM
