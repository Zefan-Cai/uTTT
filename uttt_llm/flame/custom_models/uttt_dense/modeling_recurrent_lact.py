from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils.deprecation import deprecate_kwarg

from einops import rearrange

from flash_attn import flash_attn_func
from fla.modules import FusedCrossEntropyLoss, FusedLinearCrossEntropyLoss, RMSNorm
from fla.modules import GatedMLP as TransformerMLP
from fla.modules.l2warp import l2_warp
from .ttt_operation import (
    silu_backprop as lact_silu_backprop,
    zeropower_via_newtonschulz5 as lact_zeropower_via_newtonschulz5,
)

from .configuration_recurrent_lact import RecurrentLactRefUtttDenseV3Config
from .rotary import RotaryEmbedding
from ..lact_cache_utils import (
    is_recurrent_lact_cache,
    make_recurrent_lact_cache,
    prepare_recurrent_lact_generation_inputs,
    recurrent_lact_cache_seen_tokens,
    reorder_recurrent_lact_cache,
)

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

try:
    from transformers.modeling_layers import GradientCheckpointingLayer
except ImportError:
    from fla.models.modeling_layers import GradientCheckpointingLayer

# ============================================================================
# Memory-efficient KV cache append with gradient support
# ============================================================================

class CacheAppend(torch.autograd.Function):
    """
    Appends new K/V to the cache storage in a memory-efficient way with gradient support.
    """

    @staticmethod
    def forward(ctx, storage, active_cache, x):
        bs = active_cache.size(0)
        start = active_cache.size(1)
        end = start + x.size(1)
        storage.data[:bs, start:end] = x.to(storage.dtype)
        ctx.bs = bs
        ctx.start = start
        ctx.end = end
        return storage[:bs, :end]

    @staticmethod
    def backward(ctx, grad_output):
        bs = ctx.bs
        start = ctx.start
        end = ctx.end
        return None, grad_output[:bs, :start], grad_output[:bs, start:end]


cache_append = CacheAppend.apply


# ============================================================================
# Helper Functions
# ============================================================================

def inv_softplus(x):
    """Inverse of softplus function."""
    return x + math.log(-math.expm1(-x))


def silu_backprop(dy: torch.Tensor, x: torch.Tensor):
    return lact_silu_backprop(dy, x)


def zeropower_via_newtonschulz5(G, steps):
    if steps == 0:
        return G
    if steps == 5:
        return lact_zeropower_via_newtonschulz5(G)

    assert len(G.shape) == 3
    X = G.bfloat16()
    if G.size(1) > G.size(2):
        X = X.transpose(1, 2)
    X = X / (X.norm(dim=(1, 2), keepdim=True) + 1e-7)
    coeffs = [
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ]
    for a, b, c in coeffs[:steps]:
        A = X @ X.transpose(1, 2)
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(1) > G.size(2):
        X = X.transpose(1, 2)
    return X


@torch.compile()
def l2_norm(x: torch.Tensor):
    """x: [b, l, d]"""
    x_type = x.dtype
    ret = x / (x.norm(dim=-1, keepdim=True) + 1e-5)
    return ret.type(x_type)


@torch.compile()
def rms_norm(x: torch.Tensor):
    """x: [b, l, d]"""
    x_type = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5)
    return x.type(x_type)


# ============================================================================
# Fast Weight MLP Sub-Network
# ============================================================================




class FastWeightMLPSubNN(nn.Module):
    """
    Fast Weight MLP that updates online during forward pass.
    Closely follows the reference LaCT implementation.
    """

    def __init__(
        self,
        config: RecurrentLactRefUtttDenseV3Config,
        layer_idx: int,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # Dimensions
        self.hidden_size = config.hidden_size
        num_kv_heads = config.num_kv_heads if config.num_kv_heads is not None else config.num_heads
        self.fw_num_heads = config.fw_num_heads if config.fw_num_heads is not None else num_kv_heads
        assert self.hidden_size % self.fw_num_heads == 0
        self.head_dim = self.hidden_size // self.fw_num_heads
        self.inter_multi = config.fw_inter_multi
        self.inter_dim = int(self.head_dim * self.inter_multi)
        self.w0_w2_low_rank = config.w0_w2_low_rank
        self.fw_init_gain = config.fw_init_gain

        self.repeat_function = config.repeat_function
        self.qkv_silu = config.memory_qkv_silu
        self.qk_norm = config.memory_qk_norm
        self.qk_norm_fn = l2_norm if config.memory_qk_norm_type == "l2" else rms_norm
        self.use_momentum = config.use_momentum
        self.use_moun = config.use_moun
        self.norm_fn = l2_norm if config.memory_norm_type == "l2" else rms_norm
        self.v_norm = config.memory_v_norm
        self.ttt_prenorm = config.ttt_prenorm

        # QKV projections (only created when using memory's own projections)
        if config.memory_kv_proj == "ttt_kv_proj" and config.memory_kv_mode != "reuse_kv":
            self.to_q = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
            self.to_k = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
            self.to_v = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        # Lane B owns the slow FW bank at Model scope.  A block keeps only the
        # projections and the numerics needed to read the shared recurrent
        # state and to produce one layer's update contribution.

        # Q/K Rescale (optional learnable affine transform on memory q/k)

        self.qk_rescale = config.memory_qk_rescale
        if self.qk_rescale:
            self.qk_scale = nn.Parameter(torch.ones(self.hidden_size, 2))
            self.qk_offset = nn.Parameter(torch.zeros(self.hidden_size, 2))

        # TTT RoPE (optional)
        self.ttt_rope_theta = config.ttt_rope_theta
        if config.same_rope:
            self.ttt_rope_theta = config.rope_theta
        if self.ttt_rope_theta > 0:
            if config.same_rope:
                attn_head_dim = config.hidden_size // config.num_heads
                self.rope_dim = attn_head_dim
            else:
                self.rope_dim = min(self.head_dim, 256)
            self.n_rope_heads = self.hidden_size // self.rope_dim
            self.ttt_rotary = RotaryEmbedding(dim=self.rope_dim, base=self.ttt_rope_theta)
        else:
            self.ttt_rotary = None

        # Output projection
        self.enable_memory_output_proj = config.enable_memory_output_proj
        if self.enable_memory_output_proj:
            self.output_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        # Momentum projection (Sigmoid mode)
        if self.use_momentum:
            self.momentum_proj = nn.Sequential(
                nn.Linear(self.hidden_size, self.fw_num_heads),
                nn.Sigmoid(),
            )

        # Learnable per-head TTT scale (optional)
        self.learnable_ttt_scale = config.learnable_ttt_scale
        self.ttt_norm = RMSNorm(self.head_dim, elementwise_affine=True)
        if self.learnable_ttt_scale:
            self.ttt_scale_proj = nn.Linear(self.hidden_size, self.fw_num_heads)
            fan_in = self.hidden_size
            init_alpha = 0.1
            nn.init.normal_(self.ttt_scale_proj.weight, mean=0.0, std=init_alpha / math.sqrt(fan_in))
            if self.ttt_scale_proj.bias is not None:
                nn.init.zeros_(self.ttt_scale_proj.bias)

    def reset_parameters(self):
        if self.qk_rescale:
            nn.init.ones_(self.qk_scale)
            nn.init.zeros_(self.qk_offset)

    def init_rotary_cache(self, max_len: int, device: torch.device, dtype: torch.dtype):
        """Pre-compute cos/sin cache for TTT RoPE."""
        if self.ttt_rotary is not None:
            self.ttt_rotary._update_cos_sin_cache(max_len, device=device, dtype=dtype)

    def _apply_fast_weights(
        self,
        x: torch.Tensor,
        w0: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply fast weights using the same bmm contraction order as lact."""
        b, l, d = x.shape

        fw_input_mh = rearrange(x, 'b l (h hd) -> (b h) l hd', h=self.fw_num_heads)
        fw_input_mh = fw_input_mh.to(dtype=w0.dtype)
        w0_flat = rearrange(w0, 'b h k d -> (b h) k d')
        w1_flat = rearrange(w1, 'b h d k -> (b h) d k')
        w2_flat = rearrange(w2, 'b h k d -> (b h) k d')

        gate_before_act = torch.bmm(w0_flat, fw_input_mh.transpose(1, 2))
        hidden_before_mul = torch.bmm(w2_flat, fw_input_mh.transpose(1, 2))
        hidden = F.silu(gate_before_act, inplace=False) * hidden_before_mul
        out_mh = torch.bmm(w1_flat, hidden).transpose(1, 2)
        out = rearrange(out_mh, '(b h) l hd -> b l (h hd)', b=b, h=self.fw_num_heads)

        gate_before_act = rearrange(gate_before_act, '(b h) k l -> b l h k', b=b, h=self.fw_num_heads)
        hidden_before_mul = rearrange(hidden_before_mul, '(b h) k l -> b l h k', b=b, h=self.fw_num_heads)
        hidden = rearrange(hidden, '(b h) k l -> b l h k', b=b, h=self.fw_num_heads)
        fw_input_mh = rearrange(fw_input_mh, '(b h) l hd -> b l h hd', b=b, h=self.fw_num_heads)

        return out, gate_before_act, hidden_before_mul, hidden, fw_input_mh

    def _rescale_qk(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply learnable affine transform to q and k. q/k: [b, l, d]."""
        qk_scale = self.qk_scale.view(1, 1, -1, 2)
        qk_offset = self.qk_offset.view(1, 1, -1, 2)
        q = q * qk_scale[:, :, :, 0] + qk_offset[:, :, :, 0]
        k = k * qk_scale[:, :, :, 1] + qk_offset[:, :, :, 1]
        return q, k

    def _apply_ttt_norm(self, x: torch.Tensor) -> torch.Tensor:
        if x.device.type == "cpu":
            x = rms_norm(x)
            weight = getattr(self.ttt_norm, "weight", None)
            return x if weight is None else x * weight
        return self.ttt_norm(x)

    def forward(self, q: torch.Tensor, fastw: Tuple[torch.Tensor, ...], seqlen_offset: int = 0) -> torch.Tensor:
        """
        Forward pass (memory read) with fast weights.
        Args:
            q: [b, l, d] - pre-projected query (from to_q)
            fastw: (w0, w1, w2) BF16 fast weights
            seqlen_offset: position offset for TTT RoPE
        """
        w0, w1, w2 = fastw

        if self.qk_rescale:
            q, _ = self._rescale_qk(q, q)  # only apply q channel for read
        if self.qkv_silu:
            q = F.silu(q)
        if self.qk_norm:
            q = rearrange(q, 'b l (h d) -> (b h) l d', h=self.fw_num_heads)
            q = self.qk_norm_fn(q)
            q = rearrange(q, '(b h) l d -> b l (h d)', h=self.fw_num_heads)

        if self.ttt_rotary is not None:
            q = rearrange(q, 'b l (nh d) -> b l nh d', nh=self.n_rope_heads)
            q = q.to(self.ttt_rotary._cos_cached.dtype)
            _, q = self.ttt_rotary(None, q, seqlen_offset=seqlen_offset)
            q = rearrange(q, 'b l nh d -> b l (nh d)')

        out, _, _, _, _ = self._apply_fast_weights(q, w0, w1, w2)

        if not self.learnable_ttt_scale:
            out = rearrange(out, 'b s (n_h d) -> (b n_h) s d', n_h=self.fw_num_heads)
            out = self._apply_ttt_norm(out)
            out = rearrange(out, '(b n_h) s d -> b s (n_h d)', n_h=self.fw_num_heads)
        if self.enable_memory_output_proj:
            out = self.output_proj(out)

        return out

    def apply_ttt_scale(self, memory_out: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply per-head learnable scale to memory output."""
        memory_out = rearrange(memory_out, 'b s (n_h d) -> (b n_h) s d', n_h=self.fw_num_heads)
        ttt_x_normed = self._apply_ttt_norm(memory_out)
        # scalar scale per head
        ttt_scale = F.silu(self.ttt_scale_proj(hidden_states), inplace=False)
        ttt_scale = rearrange(ttt_scale, 'b s (n_h d) -> (b n_h) s d', n_h=self.fw_num_heads)
        ttt_x_normed = ttt_x_normed * ttt_scale
        ttt_x_normed = rearrange(ttt_x_normed, '(b n_h) s d -> b s (n_h d)', n_h=self.fw_num_heads)
        return ttt_x_normed


    def compute_update(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        lr: torch.Tensor,
        fastw: Tuple[torch.Tensor, ...],
        pre_vi: torch.Tensor | None = None,
        seqlen_offset: int = 0,
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor | None]:
        """Compute one layer's raw wide update against a shared snapshot.

        The model, rather than this block, owns momentum, Muon, normalization,
        and the recurrent write.  Returning the raw contribution here is what
        lets every layer read the same chunk-start snapshot before one model-
        level aggregate writeback.
        """
        w0, w1, w2 = fastw
        b, l, _ = k.shape

        if self.qk_rescale:
            _, k = self._rescale_qk(k, k)
        if self.qkv_silu:
            k = F.silu(k)
            v = F.silu(v)
        if self.v_norm:
            v = self.norm_fn(v)
        if self.qk_norm:
            k = rearrange(k, "b l (h d) -> (b h) l d", h=self.fw_num_heads)
            k = self.qk_norm_fn(k)
            k = rearrange(k, "(b h) l d -> b l (h d)", h=self.fw_num_heads)

        if self.ttt_rotary is not None:
            k = rearrange(k, "b l (nh d) -> b l nh d", nh=self.n_rope_heads)
            k = k.to(self.ttt_rotary._cos_cached.dtype)
            _, k = self.ttt_rotary(None, k, seqlen_offset=seqlen_offset)
            k = rearrange(k, "b l nh d -> b l (nh d)")

        k_mh = rearrange(k, "b l (h hd) -> (b h) l hd", h=self.fw_num_heads)
        v_mh = rearrange(v, "b l (h hd) -> (b h) l hd", h=self.fw_num_heads)
        # ref_v4's compiled bmm promotes mixed bf16/float32 inputs on CPU;
        # make that promotion explicit for the eager aggregate contribution.
        # CUDA autocast still lowers these contractions to bf16.
        compute_dtype = torch.promote_types(w0.dtype, k_mh.dtype)
        k_mh = k_mh.to(dtype=compute_dtype)
        v_mh = v_mh.to(dtype=compute_dtype)
        w0_flat = rearrange(w0, "b h k d -> (b h) k d").to(dtype=compute_dtype)
        w1_flat = rearrange(w1, "b h d k -> (b h) d k").to(dtype=compute_dtype)
        w2_flat = rearrange(w2, "b h k d -> (b h) k d").to(dtype=compute_dtype)

        if lr.shape[2] == 3:
            lr0, lr1, lr2 = lr.split(1, dim=2)
            lr0_mh = rearrange(
                lr0.expand(-1, -1, self.fw_num_heads),
                "b l h -> (b h) l 1",
            )
            lr1_mh = rearrange(
                lr1.expand(-1, -1, self.fw_num_heads),
                "b l h -> (b h) l 1",
            )
            lr2_mh = rearrange(
                lr2.expand(-1, -1, self.fw_num_heads),
                "b l h -> (b h) l 1",
            )
        else:
            lr = lr.view(lr.shape[0], lr.shape[1], self.fw_num_heads, 3)
            lr0_mh = rearrange(lr[:, :, :, 0:1], "b l h d -> (b h) l d")
            lr1_mh = rearrange(lr[:, :, :, 1:2], "b l h d -> (b h) l d")
            lr2_mh = rearrange(lr[:, :, :, 2:3], "b l h d -> (b h) l d")

        with torch.autocast(
            device_type=k_mh.device.type,
            enabled=k_mh.device.type == "cuda",
            dtype=torch.bfloat16,
        ):
            vi = v_mh.transpose(1, 2)
            gate_dtype = torch.promote_types(w0_flat.dtype, k_mh.dtype)
            gate_before_act = torch.bmm(
                w0_flat.to(dtype=gate_dtype), k_mh.transpose(1, 2).to(dtype=gate_dtype)
            )
            hidden_before_mul = torch.bmm(
                w2_flat.to(dtype=gate_dtype), k_mh.transpose(1, 2).to(dtype=gate_dtype)
            )
            hidden = F.silu(gate_before_act, inplace=False) * hidden_before_mul

            dhid_dtype = torch.promote_types(w1_flat.dtype, vi.dtype)
            dhidden = torch.bmm(
                w1_flat.transpose(1, 2).to(dtype=dhid_dtype), vi.to(dtype=dhid_dtype)
            )
            dhidden_before_mul = dhidden * F.silu(gate_before_act, inplace=False)
            dgate = dhidden * hidden_before_mul
            dgate_before_act = silu_backprop(dgate, gate_before_act)

            dw1 = torch.bmm(vi, (hidden.transpose(1, 2) * lr1_mh).type_as(vi))
            dw0 = torch.bmm(
                dgate_before_act,
                (k_mh * lr0_mh).type_as(dgate_before_act),
            )
            dw2 = torch.bmm(
                dhidden_before_mul,
                (k_mh * lr2_mh).type_as(dhidden_before_mul),
            )

        momentum_coeff = None
        if self.use_momentum and pre_vi is not None:
            # Match ref_v4's per-head, token-mean coefficient.  The shared
            # model aggregates these coefficients across layers at writeback.
            momentum_input = pre_vi.to(dtype=self.momentum_proj[0].weight.dtype)
            momentum_coeff = self.momentum_proj(momentum_input).mean(dim=1).unsqueeze(-1).unsqueeze(-1)

        return (
            (
                rearrange(dw0, "(b h) k d -> b h k d", b=b, h=self.fw_num_heads).to(torch.float32),
                rearrange(dw1, "(b h) d k -> b h d k", b=b, h=self.fw_num_heads).to(torch.float32),
                rearrange(dw2, "(b h) k d -> b h k d", b=b, h=self.fw_num_heads).to(torch.float32),
            ),
            momentum_coeff,
        )


    def extra_repr(self) -> str:
        return (
            f"head_dim={self.head_dim}, inter_dim={self.inter_dim}, "
            f"fw_num_heads={self.fw_num_heads}, low_rank={self.w0_w2_low_rank}"
        )


# ============================================================================
# Recurrent Attention
# ============================================================================

class RecurrentAttention(nn.Module):
    def __init__(self, config: RecurrentLactRefUtttDenseV3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads if config.num_kv_heads is not None else config.num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.kv_dim = self.num_kv_heads * self.head_dim

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.qkv_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=config.qkv_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=config.qkv_bias)
        self.enable_attention_output_proj = config.enable_attention_output_proj
        if config.enable_attention_output_proj:
            self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.rotary = RotaryEmbedding(dim=self.head_dim, base=config.rope_theta)
        self.window_size = config.window_size

        self.k_storage = torch.empty(0, 0, self.num_kv_heads, self.head_dim)
        self.v_storage = torch.empty(0, 0, self.num_kv_heads, self.head_dim)

    def _ensure_storage(self, batch_size, needed_len, device, dtype):
        b, l, h, d = self.k_storage.shape
        batch_size = max(batch_size, b)
        needed_len = max(needed_len, l)
        if b < batch_size or l < needed_len or self.k_storage.device != device or self.k_storage.dtype != dtype:
            self.k_storage = self.k_storage.new_empty(batch_size, needed_len, h, d, device=device, dtype=dtype)
            self.v_storage = self.v_storage.new_empty(batch_size, needed_len, h, d, device=device, dtype=dtype)

    def cache_init(self, batch_size, max_len, device, dtype):
        self._ensure_storage(batch_size, max_len, device=device, dtype=dtype)
        self.rotary._update_cos_sin_cache(max_len * 2, device=device, dtype=dtype)
        k_active = self.k_storage[:batch_size, :0]
        v_active = self.v_storage[:batch_size, :0]
        return k_active, v_active

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, q_len, _ = hidden_states.size()

        q = self.q_proj(hidden_states).view(batch_size, q_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(batch_size, q_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(batch_size, q_len, self.num_kv_heads, self.head_dim)

        # Flatten pre-rotary q/k/v for memory sharing
        q_flat = q.flatten(-2, -1)  # [b, l, hidden_size]
        k_flat = k.flatten(-2, -1)  # [b, l, kv_dim]
        v_flat = v.flatten(-2, -1)  # [b, l, kv_dim]

        k_cache, v_cache = kv_cache
        seqlen_offset = k_cache.size(1)
        q, k_rotated = self.rotary(q, k, seqlen_offset=seqlen_offset)

        full_k = cache_append(self.k_storage, k_cache, k_rotated)
        full_v = cache_append(self.v_storage, v_cache, v)

        q = q.to(torch.bfloat16)
        if hidden_states.device.type == "cpu":
            # flash-attn's Triton path has no CPU implementation.  This is a
            # numerically equivalent tiny-check fallback with the same causal
            # and left-window mask as the GPU call above.
            q_attn = q.transpose(1, 2)
            k_attn = full_k.transpose(1, 2)
            v_attn = full_v.transpose(1, 2)
            if self.num_heads != self.num_kv_heads:
                repeat = self.num_heads // self.num_kv_heads
                k_attn = k_attn.repeat_interleave(repeat, dim=1)
                v_attn = v_attn.repeat_interleave(repeat, dim=1)
            q_positions = torch.arange(q_len, device=q.device) + full_k.size(1) - q_len
            k_positions = torch.arange(full_k.size(1), device=q.device)
            attn_mask = k_positions.unsqueeze(0) <= q_positions.unsqueeze(1)
            if self.window_size is not None:
                attn_mask = attn_mask & (
                    k_positions.unsqueeze(0)
                    >= q_positions.unsqueeze(1) - (self.window_size - 1)
                )
            attn_output = F.scaled_dot_product_attention(
                q_attn,
                k_attn,
                v_attn,
                attn_mask=attn_mask,
                is_causal=False,
            ).transpose(1, 2)
        else:
            attn_output = flash_attn_func(
                q, full_k, full_v,
                causal=True,
                window_size=(-1, -1) if self.window_size is None else (self.window_size - 1, 0)
            )

        attn_output = attn_output.view(batch_size, q_len, self.hidden_size)

        if self.enable_attention_output_proj:
            attn_output = self.o_proj(attn_output)
        return attn_output, k_flat, v_flat, q_flat

    def attention_tokv(self, feature: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project feature using attention's k/v projections for memory use."""
        k = self.k_proj(feature).flatten(-2, -1)  # [b, l, kv_dim]
        v = self.v_proj(feature).flatten(-2, -1)  # [b, l, kv_dim]
        return k, v

    def update_kv(
        self,
        last_hidden_states: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        seqlen_offset: int,
        apply_layer_kv: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if apply_layer_kv:
            k = self.k_proj(last_hidden_states).unflatten(-1, (self.num_kv_heads, self.head_dim))
            v = self.v_proj(last_hidden_states).unflatten(-1, (self.num_kv_heads, self.head_dim))
        else:
            k, v = last_hidden_states.unflatten(-1, (2, self.num_kv_heads, self.head_dim)).unbind(2)

        _, k = self.rotary(None, k, seqlen_offset=seqlen_offset)

        k_cache, v_cache = kv_cache
        new_k_cache = cache_append(self.k_storage, k_cache, k)
        new_v_cache = cache_append(self.v_storage, v_cache, v)

        return (new_k_cache, new_v_cache)


# ============================================================================
# Transformer Block
# ============================================================================

class RecurrentLactBlock(GradientCheckpointingLayer):
    def __init__(self, config: RecurrentLactRefUtttDenseV3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.attn_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)
        self.attn = RecurrentAttention(config, layer_idx)

        # Memory block (Fast Weight MLP)
        if config.use_memory_block:
            # `memory_norm` is only consumed by the sequential residual path.
            # Keeping it trainable in parallel mode creates a dead optimizer
            # param whose state is never materialized in checkpoints.
            if config.residual_style == "sequential":
                self.memory_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)
            else:
                self.memory_norm = None
            self.memory = FastWeightMLPSubNN(config, layer_idx)

            
            fw_num_heads = config.fw_num_heads if config.fw_num_heads is not None else (config.num_kv_heads or config.num_heads)
            lr_out_dim = 3 * fw_num_heads if config.per_head_lr else 3
            self.to_lr = nn.Linear(config.hidden_size, lr_out_dim)
            self.per_head_lr = config.per_head_lr
            self.base_lr_inv = inv_softplus(config.base_lr)
        else:
            self.memory_norm = None
            self.memory = None

        self.enable_attention_memory_output_proj = config.enable_attention_memory_output_proj
        if self.enable_attention_memory_output_proj:
            self.memory_output_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        self.mlp_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)
        self.mlp = TransformerMLP(
            hidden_size=config.hidden_size,
            hidden_ratio=config.hidden_ratio,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            fuse_swiglu=config.fuse_swiglu,
        )

    def memory_toq(self, feature: torch.Tensor) -> torch.Tensor:
        """Project feature to query using memory's to_q projection."""
        return self.memory.to_q(feature)

    def attention_toq(self, feature: torch.Tensor) -> torch.Tensor:
        """Project feature to query using attention's q_proj."""
        return self.attn.q_proj(feature)

    def memory_tokv(self, feature: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project feature to key/value using memory's to_k/to_v projections."""
        k = self.memory.to_k(feature)
        v = self.memory.to_v(feature)
        return k, v

    def _apply_mlp(self, hidden_states: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        if hidden_states.device.type == "cpu":
            # fla's fused SwiGLU is Triton-only.  Use the same three learned
            # projections and activation for the CPU bank check.
            gate = self.mlp.gate_proj(hidden_states)
            value = self.mlp.up_proj(hidden_states)
            return self.mlp.down_proj(F.silu(gate) * value)
        return self.mlp(hidden_states, **kwargs)

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        fastw: Tuple[torch.Tensor, ...] | None = None,
        seqlen_offset: int = 0,
        **kwargs: Any,
    ):
        if self.memory is not None and fastw is None:
            raise ValueError("uTTT-Dense v3 blocks must receive the shared fast state")

        if self.config.residual_style == "parallel":

            
            residual = hidden_states

            # Attention
            x_attn_in = self.attn_norm(hidden_states)
            attn_out, attn_k, attn_v, attn_q = self.attn(hidden_states=x_attn_in, kv_cache=kv_cache)


            if self.memory is not None and fastw is not None:
                x_memory_in = x_attn_in
                
                if self.config.memory_kv_mode == "reuse_kv":
                    q = attn_q
                elif self.config.memory_kv_proj == "reuse_attn_kv_proj":
                    q = self.attention_toq(x_memory_in)
                else:
                    q = self.memory_toq(x_memory_in)

                memory_out = self.memory(q, fastw, seqlen_offset=seqlen_offset)
                if self.memory.learnable_ttt_scale:
                    memory_out = self.memory.apply_ttt_scale(memory_out, x_attn_in)

                hidden_states = memory_out + attn_out
                if self.enable_attention_memory_output_proj:
                    hidden_states = residual + self.memory_output_proj(hidden_states)
                else:
                    hidden_states = residual + hidden_states
            else:
                raise ValueError(f"Memory block is not enabled")
        
        elif self.config.residual_style == "sequential":

            # Attention
            residual = hidden_states

            x_attn_in = self.attn_norm(hidden_states)
            attn_out, attn_k, attn_v, attn_q = self.attn(hidden_states=x_attn_in, kv_cache=kv_cache)
            hidden_states = residual + attn_out

            # Memory
            x_memory_in = None
            if self.memory is not None and fastw is not None:
                residual = hidden_states
                x_memory_in = self.memory_norm(hidden_states)

                if self.config.memory_kv_mode == "reuse_kv":
                    q = attn_q
                elif self.config.memory_kv_proj == "reuse_attn_kv_proj":
                    q = self.attention_toq(x_memory_in)
                else:
                    q = self.memory_toq(x_memory_in)

                memory_out = self.memory(q, fastw, seqlen_offset=seqlen_offset)
                if self.memory.learnable_ttt_scale:
                    memory_out = self.memory.apply_ttt_scale(memory_out, x_memory_in)

                hidden_states = memory_out
                if self.enable_attention_memory_output_proj:
                    hidden_states = residual + self.memory_output_proj(hidden_states)
                else:
                    hidden_states = residual + hidden_states
            else:
                raise ValueError(f"Memory block is not enabled")
        else:
            raise ValueError(f"Invalid residual_style: {self.config.residual_style}")    
        
        # MLP
        residual = hidden_states
        hidden_states = self.mlp_norm(hidden_states)
        hidden_states = residual + self._apply_mlp(hidden_states, **kwargs)

        # The shared state is read-only during layer traversal.  Model-level
        # aggregate writeback happens after every layer has produced features.
        return hidden_states, x_attn_in, x_memory_in, attn_k, attn_v, fastw, None, None

    def compute_fast_weight_contribution(
        self,
        pre_vi: torch.Tensor,
        fastw: Tuple[torch.Tensor, ...],
        masterw: Tuple[torch.Tensor, ...] | None = None,
        momentum_buf: Tuple[torch.Tensor, ...] | None = None,
        ki: torch.Tensor = None,
        vi: torch.Tensor = None,
        seqlen_offset: int = 0,
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor | None]:
        """Return this layer's raw contribution to the shared bank update."""
        if self.memory is None:
            return (torch.zeros_like(fastw[0]), torch.zeros_like(fastw[1]), torch.zeros_like(fastw[2])), None

        # LR prediction
        lri = self.to_lr(pre_vi)
        lri = F.softplus(lri.float() + self.base_lr_inv)

        # ki and vi are already projected by Model (via memory_tokv or
        # attention_tokv).  No state is changed here.
        return self.memory.compute_update(
            ki,
            vi,
            lri,
            fastw,
            pre_vi=pre_vi,
            seqlen_offset=seqlen_offset,
        )

    def compute_grad_contribution(
        self,
        pre_vi: torch.Tensor,
        fastw: Tuple[torch.Tensor, ...],
        ki: torch.Tensor,
        vi: torch.Tensor,
        seqlen_offset: int = 0,
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor | None]:
        """Return one layer's raw contribution against a shared snapshot."""
        return self.compute_fast_weight_contribution(
            pre_vi,
            fastw,
            masterw=fastw,
            ki=ki,
            vi=vi,
            seqlen_offset=seqlen_offset,
        )



    def update_kv(
        self,
        last_hidden_states: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        apply_layer_kv: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.attn.update_kv(last_hidden_states, kv_cache, kv_cache[0].size(1), apply_layer_kv)


# ============================================================================
# Pre-trained Model Base
# ============================================================================

class RecurrentLactRefUtttDenseV3PreTrainedModel(PreTrainedModel):
    config_class = RecurrentLactRefUtttDenseV3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _no_split_modules = ["RecurrentLactBlock"]
    _supports_cache_class = False

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(
        self,
        module: nn.Module,
        rescale_prenorm_residual: bool = False,
        num_residuals_per_layer: int = 2,
    ):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, (RMSNorm, nn.RMSNorm)):
            if hasattr(module, "weight") and module.weight is not None:
                nn.init.ones_(module.weight)
            if hasattr(module, "bias") and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, FastWeightMLPSubNN):
            if module.qk_rescale:
                nn.init.ones_(module.qk_scale)
                nn.init.zeros_(module.qk_offset)
        elif hasattr(module, "reset_parameters"):
            module.reset_parameters()

        # Initialize memory qk_scale and qk_offset (Q/K Rescale)
        if hasattr(module, "qk_scale"):
            nn.init.ones_(module.qk_scale)
        if hasattr(module, "qk_offset"):
            nn.init.zeros_(module.qk_offset)

        if rescale_prenorm_residual:
            p = None
            if hasattr(module, "o_proj"):
                p = module.o_proj.weight
            elif hasattr(module, "down_proj"):
                p = module.down_proj.weight
            if p is not None:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(num_residuals_per_layer * self.config.num_hidden_layers)


# ============================================================================
# Main Model
# ============================================================================

class RecurrentLactRefUtttDenseV3Model(RecurrentLactRefUtttDenseV3PreTrainedModel):
    def __init__(self, config: RecurrentLactRefUtttDenseV3Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.chunk_size = config.chunk_size

        self.memory_v_gap = config.memory_v_gap
        self.memory_k_gap = config.memory_k_gap
        self.memory_v_layer_idxs = []
        self.memory_k_layer_idxs = []
        for i in range(config.num_hidden_layers):
            if self.memory_v_gap is not None:
                self.memory_v_layer_idxs.append(max(min(i + self.memory_v_gap, config.num_hidden_layers - 1), 0))
            if self.memory_k_gap is not None:
                self.memory_k_layer_idxs.append(max(min(i + self.memory_k_gap, config.num_hidden_layers - 1), 0))

        self.attention_k_gap = config.attention_k_gap
        self.attention_v_gap = config.attention_v_gap
        self.attention_k_layer_idxs = []
        self.attention_v_layer_idxs = []
        for i in range(config.num_hidden_layers):
            if self.attention_v_gap is not None:
                self.attention_v_layer_idxs.append(max(min(i + self.attention_v_gap, config.num_hidden_layers - 1), 0))
            if self.attention_k_gap is not None:
                self.attention_k_layer_idxs.append(max(min(i + self.attention_k_gap, config.num_hidden_layers - 1), 0))

        self.shared_kv_cache = config.shared_kv_cache

        self.use_memory_block = config.use_memory_block
        self.memory_kv_mode = config.memory_kv_mode
        self.memory_kv_feature = config.memory_kv_feature
        self.memory_kv_proj = config.memory_kv_proj

        num_kv_heads = config.num_kv_heads if config.num_kv_heads is not None else config.num_heads
        self._shared_fw_num_heads = config.fw_num_heads if config.fw_num_heads is not None else num_kv_heads
        if self._shared_fw_num_heads <= 0 or config.hidden_size % self._shared_fw_num_heads != 0:
            raise ValueError(
                "uTTT-Dense v3 requires hidden_size divisible by a positive "
                f"fw_num_heads; got hidden_size={config.hidden_size}, "
                f"fw_num_heads={self._shared_fw_num_heads}."
            )
        self._shared_head_dim = config.hidden_size // self._shared_fw_num_heads
        self._shared_inter_dim = self._shared_head_dim * config.uttt_dense_r
        self.w0_w2_low_rank = config.w0_w2_low_rank
        self.fw_init_gain = config.fw_init_gain
        self._use_momentum = bool(config.use_momentum)
        self._use_moun = bool(config.use_moun)

        h = self._shared_fw_num_heads
        r = config.uttt_dense_r
        d = self._shared_head_dim
        rank = self.w0_w2_low_rank
        if rank > 0:
            self.shared_w0_left = nn.Parameter(torch.empty(h, r, d, rank))
            self.shared_w0_right = nn.Parameter(torch.empty(h, r, rank, d))
            self.shared_w2_left = nn.Parameter(torch.empty(h, r, d, rank))
            self.shared_w2_right = nn.Parameter(torch.empty(h, r, rank, d))
        else:
            self.shared_w0 = nn.Parameter(torch.empty(h, r * d, d))
            self.shared_w2 = nn.Parameter(torch.empty(h, r * d, d))
        # w1 is always a complete [H, D, rD] slow parameter.
        self.shared_w1 = nn.Parameter(torch.empty(h, d, r * d))

        if self.attention_v_gap is not None:
            self.pre_attention_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)

        if self.memory_kv_proj == "model_kv_proj":
            num_kv_heads = config.num_kv_heads if config.num_kv_heads is not None else config.num_heads
            kv_head_dim = config.hidden_size // config.num_heads
            kv_dim = 2 * num_kv_heads * kv_head_dim
            self.memory_tokv = nn.Linear(config.hidden_size, kv_dim, bias=False)

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [RecurrentLactBlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        if config.use_memory_block and any(layer.memory is None for layer in self.layers):
            raise ValueError(
                "uTTT-Dense v3 requires every layer to own a memory block: "
                "the aggregate mean divides by len(layers), so a memory-less "
                "layer would silently dilute the shared update."
            )
        self.norm = (RMSNorm if config.last_layer_fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)

        if self.shared_kv_cache:
            kv_dim = 2 * self.layers[0].attn.num_kv_heads * self.layers[0].attn.head_dim
            self.to_kv_cache = nn.Linear(config.hidden_size, kv_dim, bias=config.qkv_bias)

        self.gradient_checkpointing = False
        self.post_init()
        self._reset_shared_bank()

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings = value

    def _apply_output_norm(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply the final norm, with a CPU fallback for FLA's Triton norm."""
        if hidden_states.device.type == "cpu" and isinstance(self.norm, RMSNorm):
            # ``fla.modules.RMSNorm`` dispatches to Triton even on CPU.  Keep
            # the same RMSNorm computation available for the tiny bank check
            # without changing the CUDA/fused path used by training.
            normalized = hidden_states.float()
            normalized = normalized * torch.rsqrt(
                normalized.square().mean(dim=-1, keepdim=True) + self.config.norm_eps,
            )
            weight = getattr(self.norm, "weight", None)
            if weight is not None:
                normalized = normalized * weight.float()
            return normalized.to(dtype=hidden_states.dtype)
        return self.norm(hidden_states)

    def reset_parameters(self) -> None:
        """Reinitialize the model-owned bank during HF ``post_init``.

        This is also needed for the meta -> ``to_empty`` construction path:
        Transformers invokes ``post_init`` again after materializing storage.
        """
        if hasattr(self, "shared_w1"):
            self._reset_shared_bank()

    def _reset_shared_bank(self) -> None:
        """Initialize the single model-owned slow bank.

        ``w1`` uses the v2/Lane-B convention ``1/sqrt(D)``.  For a low-rank
        ``w0``/``w2`` bank the factors use the reference factor scales; the
        per-slice identity is added only by materialization.
        """
        d = self._shared_head_dim
        rank = self.w0_w2_low_rank
        if rank > 0:
            nn.init.normal_(self.shared_w0_left, std=self.fw_init_gain / math.sqrt(rank))
            nn.init.normal_(self.shared_w0_right, std=self.fw_init_gain / math.sqrt(d))
            nn.init.normal_(self.shared_w2_left, std=self.fw_init_gain / math.sqrt(rank))
            nn.init.normal_(self.shared_w2_right, std=self.fw_init_gain / math.sqrt(d))
        else:
            nn.init.normal_(self.shared_w0, mean=0.0, std=1.0 / math.sqrt(d))
            nn.init.normal_(self.shared_w2, mean=0.0, std=1.0 / math.sqrt(d))
        # w1 contracts over the full inter dim (r*d): ref_v4's 1/sqrt(fan_in)
        # convention. A per-slice 1/sqrt(d) init inflates the summed read by
        # ~sqrt(r), and the norm-preserving writeback then locks that operating
        # point in for the whole sequence (reviewer finding, 2026-08-08).
        nn.init.normal_(
            self.shared_w1, mean=0.0, std=1.0 / math.sqrt(self.shared_w1.shape[-1])
        )

    def _materialize_shared_w0_w2(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Materialize full runtime ``w0``/``w2`` with slice-local identity."""
        if self.w0_w2_low_rank <= 0:
            return self.shared_w0, self.shared_w2

        h = self._shared_fw_num_heads
        r = self.config.uttt_dense_r
        d = self._shared_head_dim
        rank = self.w0_w2_low_rank
        w0_slices = torch.matmul(self.shared_w0_left, self.shared_w0_right)
        w2_slices = torch.matmul(self.shared_w2_left, self.shared_w2_right)
        eye = torch.eye(
            d,
            device=w0_slices.device,
            dtype=w0_slices.dtype,
        ).view(1, 1, d, d)
        w0_slices = w0_slices + 0.5 * eye
        w2_slices = w2_slices + 0.5 * eye
        # Keep the explicit slice axis until after identity addition so the
        # identity cannot accidentally cover only the first D rows.
        return (
            w0_slices.reshape(h, r * d, d),
            w2_slices.reshape(h, r * d, d),
        )

    def _init_shared_fast_weights(
        self,
        batch_size: int = 1,
    ) -> tuple[
        tuple[torch.Tensor, ...],
        tuple[torch.Tensor, ...],
        tuple[torch.Tensor, ...] | None,
    ]:
        """Create one batch-expanded fast/master/momentum state triplet."""
        base_w0, base_w2 = self._materialize_shared_w0_w2()
        base_w1 = self.shared_w1
        masterw = (
            base_w0.float().unsqueeze(0).repeat(batch_size, 1, 1, 1),
            base_w1.float().unsqueeze(0).repeat(batch_size, 1, 1, 1),
            base_w2.float().unsqueeze(0).repeat(batch_size, 1, 1, 1),
        )
        fastw = tuple(weight.to(dtype=torch.bfloat16) for weight in masterw)
        if self._use_momentum:
            momentum_buf = tuple(torch.zeros_like(weight) for weight in masterw)
        else:
            momentum_buf = None
        return fastw, masterw, momentum_buf

    @staticmethod
    def _wide_to_slices(
        weight: torch.Tensor,
        matrix_name: str,
        r: int,
    ) -> torch.Tensor:
        if matrix_name in {"w0", "w2"}:
            b, h, inter_dim, d = weight.shape
            if inter_dim != r * d:
                raise ValueError(f"Invalid {matrix_name} state shape: {tuple(weight.shape)}")
            return weight.reshape(b, h, r, d, d).reshape(b * h * r, d, d)
        b, h, d, inter_dim = weight.shape
        if inter_dim != r * d:
            raise ValueError(f"Invalid {matrix_name} state shape: {tuple(weight.shape)}")
        return (
            weight.reshape(b, h, d, r, d)
            .permute(0, 1, 3, 2, 4)
            .reshape(b * h * r, d, d)
        )

    @staticmethod
    def _slices_to_wide(
        slices: torch.Tensor,
        matrix_name: str,
        batch_size: int,
        num_heads: int,
        r: int,
    ) -> torch.Tensor:
        _, d, e = slices.shape
        if d != e or slices.shape[0] != batch_size * num_heads * r:
            raise ValueError(
                f"Invalid {matrix_name} slice shape: {tuple(slices.shape)}"
            )
        if matrix_name in {"w0", "w2"}:
            return slices.reshape(batch_size, num_heads, r * d, d)
        return (
            slices.reshape(batch_size, num_heads, r, d, d)
            .permute(0, 1, 3, 2, 4)
            .reshape(batch_size, num_heads, d, r * d)
        )


    def _apply_shared_update(
        self,
        agg_grads: tuple[torch.Tensor, ...],
        fastw: tuple[torch.Tensor, ...],
        masterw: tuple[torch.Tensor, ...],
        momentum_buf: tuple[torch.Tensor, ...] | None,
        momentum_coeffs: list[torch.Tensor] | torch.Tensor | None,
    ) -> tuple[
        tuple[torch.Tensor, ...],
        tuple[torch.Tensor, ...],
        tuple[torch.Tensor, ...] | None,
    ]:
        """Apply the one Lane-B chunk-boundary writeback."""
        if len(agg_grads) != 3 or len(fastw) != 3 or len(masterw) != 3:
            raise ValueError("uTTT-Dense v3 expects three FW matrices")
        b, h, inter_dim, d = fastw[0].shape
        r = self.config.uttt_dense_r
        if (h, inter_dim, d) != (self._shared_fw_num_heads, r * self._shared_head_dim, self._shared_head_dim):
            raise ValueError(
                "Invalid shared fast state shape: "
                f"w0={tuple(fastw[0].shape)}, expected [B,{h},{r * self._shared_head_dim},{self._shared_head_dim}]"
            )
        if tuple(fastw[1].shape) != (b, h, d, r * d):
            raise ValueError(f"Invalid shared w1 state shape: {tuple(fastw[1].shape)}")
        if tuple(fastw[2].shape) != tuple(fastw[0].shape):
            raise ValueError(f"Invalid shared w2 state shape: {tuple(fastw[2].shape)}")

        if torch.is_tensor(momentum_coeffs):
            coeff = momentum_coeffs
        elif momentum_coeffs:
            coeff = torch.stack(momentum_coeffs, dim=0).mean(dim=0)
        else:
            coeff = None
        if coeff is not None:
            coeff = coeff.to(device=fastw[0].device, dtype=masterw[0].dtype)

        # r == 1 deliberately has no separate writeback: at r=1 the slice view
        # is the whole matrix, and one implementation is what the parity gate
        # certifies for every r (reviewer finding, 2026-08-08).
        matrix_names = ("w0", "w1", "w2")
        fast_slices = [
            self._wide_to_slices(weight, name, r)
            for name, weight in zip(matrix_names, fastw, strict=True)
        ]
        master_slices = [
            self._wide_to_slices(weight, name, r)
            for name, weight in zip(matrix_names, masterw, strict=True)
        ]

        # Keep the complete recurrent write in one numerical region, matching
        # ref_v4's single-layer update.  In particular, the old fast-state
        # norms are captured before momentum/Muon and the master remains fp32;
        # only the projected view makes the bf16 round trip for the next read.
        with torch.autocast(
            device_type=fastw[0].device.type,
            enabled=fastw[0].device.type == "cuda",
            dtype=torch.bfloat16,
        ):
            norm_source = master_slices if self.config.uttt_master_renorm else fast_slices
            # Row norms accumulate in fp32 like ref_v4's compiled reduction;
            # a bf16-accumulated norm costs ~1e-2 CPU parity by itself.
            old_norms = [
                source.float().norm(dim=2, keepdim=True)
                for source in norm_source
            ]

            deltas = list(agg_grads)
            if self._use_momentum and momentum_buf is not None and coeff is not None:
                # Momentum is added to the raw aggregate gradient.  The
                # buffer stores that pre-Muon value; Muon never consumes the
                # previous buffer directly and never overwrites the buffer
                # with its orthogonalized result.
                deltas = [
                    delta + old * coeff
                    for delta, old in zip(deltas, momentum_buf, strict=True)
                ]
                # Freshly allocated by the out-of-place add above; storing by
                # reference is byte- and gradient-identical to ref_v4.
                momentum_buf = tuple(deltas)

            delta_slices = [
                self._wide_to_slices(delta, name, r)
                for name, delta in zip(matrix_names, deltas, strict=True)
            ]
            if self._use_moun:
                # Match ref_v4's w1 -> w0 -> w2 Muon order while retaining
                # one independent square update per dense-r slice.
                delta_w1 = zeropower_via_newtonschulz5(delta_slices[1], 5)
                delta_w0 = zeropower_via_newtonschulz5(delta_slices[0], 5)
                delta_w2 = zeropower_via_newtonschulz5(delta_slices[2], 5)
                delta_slices = [
                    delta_w0,
                    delta_w1,
                    delta_w2,
                ]

            # This scale controls the geometric step and therefore stays
            # post-Muon, immediately before the fp32 master update.
            delta_slices = [
                delta * self.config.uttt_agg_lr_scale
                for delta in delta_slices
            ]
            updated_master_slices = [
                master + delta.to(master.dtype)
                for master, delta in zip(master_slices, delta_slices, strict=True)
            ]
            projected_slices = [
                updated
                / (updated.norm(dim=2, keepdim=True) + 1e-5)
                * old_norm
                for updated, old_norm in zip(
                    updated_master_slices,
                    old_norms,
                    strict=True,
                )
            ]
            if self.config.uttt_master_renorm:
                updated_master_slices = projected_slices

        new_masterw = tuple(
            self._slices_to_wide(slices, name, b, h, r)
            for name, slices in zip(
                matrix_names,
                updated_master_slices,
                strict=True,
            )
        )
        new_fastw = tuple(
            self._slices_to_wide(slices, name, b, h, r).to(dtype=torch.bfloat16)
            for name, slices in zip(matrix_names, projected_slices, strict=True)
        )
        return new_fastw, new_masterw, momentum_buf

    def _memory_update_inputs(
        self,
        layer_idx: int,
        caches: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Resolve one layer's k/v/pre-v features without touching state."""
        layer = self.layers[layer_idx]
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

        pre_vi = cache_for_ki[self.memory_kv_feature]
        return pre_vi, ki, vi

    def _compose_memory_update_inputs(
        self,
        layer: RecurrentLactBlock,
        layer_idx: int,
        caches: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compatibility spelling used by the v2/global aggregate path."""
        del layer
        return self._memory_update_inputs(layer_idx, caches)

    def _aggregate_memory_update(
        self,
        caches: list[dict[str, Any]],
        fastw: tuple[torch.Tensor, ...],
        masterw: tuple[torch.Tensor, ...],
        momentum_buf: tuple[torch.Tensor, ...] | None,
        chunk_start: int,
    ) -> tuple[
        tuple[torch.Tensor, ...],
        tuple[torch.Tensor, ...],
        tuple[torch.Tensor, ...] | None,
    ]:
        """Read one shared snapshot, pool all layer contributions, write once."""
        if not self.use_memory_block:
            return fastw, masterw, momentum_buf

        # Every shape — including L=1, r=1 — goes through the one aggregate
        # path below. There deliberately is no single-layer bypass: the parity
        # gate certifies this exact code, and a bypass would exempt the
        # production path from that certificate (reviewer finding, 2026-08-08).
        aggregate: list[torch.Tensor] | None = None
        momentum_coeffs: list[torch.Tensor] = []
        for layer_idx, layer in enumerate(self.layers):
            pre_vi, ki, vi = self._compose_memory_update_inputs(layer, layer_idx, caches)

            def _layer_contribution(pv, w0f, w1f, w2f, k_in, v_in, _layer=layer):
                return _layer.compute_grad_contribution(
                    pv,
                    (w0f, w1f, w2f),
                    ki=k_in,
                    vi=v_in,
                    seqlen_offset=chunk_start,
                )

            # ref_v4 always recomputes its update path (update_fast_weight is
            # torch.compile + checkpoint regardless of the AC mode). The
            # shared-bank aggregate must do the same: the per-layer
            # dhidden/dgate/dw chains scale with r and otherwise stay resident
            # for the whole depth traversal, dominating activation memory.
            if self.config.uttt_update_checkpoint:
                contribution, momentum_coeff = checkpoint(
                    _layer_contribution,
                    pre_vi,
                    fastw[0],
                    fastw[1],
                    fastw[2],
                    ki,
                    vi,
                    preserve_rng_state=False,
                    use_reentrant=False,
                )
            else:
                contribution, momentum_coeff = _layer_contribution(
                    pre_vi, fastw[0], fastw[1], fastw[2], ki, vi
                )
            if aggregate is None:
                aggregate = list(contribution)
            else:
                aggregate = [
                    previous + current
                    for previous, current in zip(aggregate, contribution, strict=True)
                ]
            if momentum_coeff is not None:
                momentum_coeffs.append(momentum_coeff)

        if aggregate is None:
            return fastw, masterw, momentum_buf
        if self.config.uttt_agg_reduce == "mean":
            divisor = float(len(self.layers))
            aggregate = [delta / divisor for delta in aggregate]
        avg_m_coeff = (
            torch.stack(momentum_coeffs, dim=0).mean(dim=0)
            if momentum_coeffs
            else None
        )
        # The owner of the bank writes the bank: no hop through layer modules,
        # no weakref back-pointer (unpicklable, deepcopy-unsafe — reviewer
        # finding, 2026-08-08).
        return self._apply_shared_update(
            tuple(aggregate),
            fastw,
            masterw,
            momentum_buf,
            [] if avg_m_coeff is None else [avg_m_coeff],
        )

    def extra_repr(self) -> str:
        lines = []
        for i, v_layer_idx in enumerate(self.attention_v_layer_idxs):
            lines.append(f"vmap: {i} <- {v_layer_idx}")
        for i, k_layer_idx in enumerate(self.attention_k_layer_idxs):
            lines.append(f"kmap: {i} <- {k_layer_idx}")
        for i, v_layer_idx in enumerate(self.memory_v_layer_idxs):
            lines.append(f"vmap: {i} <- {v_layer_idx}")
        for i, k_layer_idx in enumerate(self.memory_k_layer_idxs):
            lines.append(f"kmap: {i} <- {k_layer_idx}")
        return "\n".join(lines)

    def _init_or_load_generation_caches(
        self,
        past_key_values: Any | None,
        batch_size: int,
        max_len: int,
        device: torch.device,
    ) -> tuple[
        list[dict[str, Any]],
        int,
        int,
        tuple[torch.Tensor, ...],
        tuple[torch.Tensor, ...],
        tuple[torch.Tensor, ...] | None,
    ]:
        cache_state = past_key_values if is_recurrent_lact_cache(past_key_values) else None
        seen_tokens = recurrent_lact_cache_seen_tokens(cache_state)
        pending_start = int(cache_state.get("pending_start", seen_tokens)) if cache_state is not None else seen_tokens
        max_len = max(max_len, 1)

        if cache_state is None:
            fastw, masterw, momentum_buf = self._init_shared_fast_weights(batch_size)
            caches = []
            for layer in self.layers:
                k_cache, v_cache = layer.attn.cache_init(batch_size, max_len, device=device, dtype=torch.bfloat16)
                if layer.memory is not None:
                    layer.memory.init_rotary_cache(max_len, device=device, dtype=torch.bfloat16)
                caches.append({
                    "kv_cache": (k_cache, v_cache),
                    "xi_out": None,
                    "x_attn_in": None,
                    "x_memory_in": None,
                    "attn_k": None,
                    "attn_v": None,
                })
            return caches, seen_tokens, pending_start, fastw, masterw, momentum_buf

        caches = [dict(layer_cache) for layer_cache in cache_state["layers"]]
        for cache in caches:
            # v3 serializes one recurrent FW state at cache top level.  Drop
            # stale per-layer fields if a cache was produced by an older
            # development snapshot.
            if any(key in cache for key in ("fastw", "masterw", "momentum_buf")):
                raise ValueError(
                    "uTTT-Dense v3 cannot resume a per-layer recurrent cache "
                    "(ref_v4/v2 format): the shared bank state would be "
                    "silently reset. Re-encode the context instead."
                )
        for layer, cache in zip(self.layers, caches, strict=True):
            k_cache, v_cache = cache["kv_cache"]
            cache_len = k_cache.size(1)
            layer.attn._ensure_storage(batch_size, max_len, device=device, dtype=torch.bfloat16)
            if cache_len > 0:
                layer.attn.k_storage.data[:batch_size, :cache_len] = k_cache.to(device=device, dtype=torch.bfloat16)
                layer.attn.v_storage.data[:batch_size, :cache_len] = v_cache.to(device=device, dtype=torch.bfloat16)
            cache["kv_cache"] = (
                layer.attn.k_storage[:batch_size, :cache_len],
                layer.attn.v_storage[:batch_size, :cache_len],
            )
            layer.attn.rotary._update_cos_sin_cache(max_len * 2, device=device, dtype=torch.bfloat16)
            if layer.memory is not None:
                layer.memory.init_rotary_cache(max_len, device=device, dtype=torch.bfloat16)
        fastw = cache_state.get("fastw")
        masterw = cache_state.get("masterw")
        momentum_buf = cache_state.get("momentum_buf")
        if fastw is None or masterw is None:
            fastw, masterw, momentum_buf = self._init_shared_fast_weights(batch_size)
        else:
            fastw = tuple(state.to(device=device) for state in fastw)
            masterw = tuple(state.to(device=device, dtype=torch.float32) for state in masterw)
            if momentum_buf is not None:
                momentum_buf = tuple(state.to(device=device, dtype=torch.float32) for state in momentum_buf)
        if self._use_momentum and momentum_buf is None:
            momentum_buf = tuple(torch.zeros_like(state) for state in masterw)
        return caches, seen_tokens, pending_start, fastw, masterw, momentum_buf

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

    def _update_generation_memory_blocks(
        self,
        caches: list[dict[str, Any]],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Aggregate one completed generation chunk and write the bank once.

        The production v3 path passes ``(fastw, masterw, momentum_buf,
        chunk_start)`` and receives the updated state tuple.  The v2-style
        compatibility path passes ``(chunk_start, shared_state=...)`` (or the
        same two values positionally) and updates that dictionary in place.
        """
        shared_state = kwargs.pop("shared_state", None)
        chunk_start_kw = kwargs.pop("chunk_start", None)
        fastw_kw = kwargs.pop("fastw", None)
        masterw_kw = kwargs.pop("masterw", None)
        momentum_buf_kw = kwargs.pop("momentum_buf", None)
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected generation-update keyword(s): {unexpected}")

        if shared_state is not None or (
            len(args) == 2 and isinstance(args[1], dict)
        ):
            if shared_state is None:
                chunk_start, shared_state = args
            else:
                if len(args) == 1:
                    (chunk_start,) = args
                elif len(args) == 0 and chunk_start_kw is not None:
                    chunk_start = chunk_start_kw
                else:
                    raise TypeError(
                        "shared-state generation update expects chunk_start and shared_state"
                    )
            fastw = shared_state.get("shared_fastw", shared_state.get("fastw"))
            masterw = shared_state.get("shared_masterw", shared_state.get("masterw"))
            momentum_buf = shared_state.get(
                "shared_momentum_buf",
                shared_state.get("momentum_buf"),
            )
            if fastw is None or masterw is None:
                raise ValueError("shared_state must contain fastw and masterw")
            updated = self._aggregate_memory_update(
                caches,
                fastw,
                masterw,
                momentum_buf,
                int(chunk_start),
            )
            state_names = (
                ("shared_fastw", "shared_masterw", "shared_momentum_buf")
                if "shared_fastw" in shared_state or "shared_masterw" in shared_state
                else ("fastw", "masterw", "momentum_buf")
            )
            shared_state[state_names[0]], shared_state[state_names[1]], shared_state[state_names[2]] = updated
            return None

        if fastw_kw is not None or masterw_kw is not None or momentum_buf_kw is not None:
            if args:
                raise TypeError("do not mix positional and keyword generation state arguments")
            if fastw_kw is None or masterw_kw is None or chunk_start_kw is None:
                raise TypeError(
                    "keyword generation update expects fastw, masterw, momentum_buf, chunk_start"
                )
            fastw, masterw, momentum_buf, chunk_start = (
                fastw_kw,
                masterw_kw,
                momentum_buf_kw,
                chunk_start_kw,
            )
        elif len(args) != 4:
            raise TypeError(
                "generation update expects fastw, masterw, momentum_buf, chunk_start"
            )
        else:
            fastw, masterw, momentum_buf, chunk_start = args
        if not self.use_memory_block:
            return fastw, masterw, momentum_buf
        return self._aggregate_memory_update(
            caches,
            fastw,
            masterw,
            momentum_buf,
            chunk_start,
        )

    def _forward_with_generation_cache(
        self,
        *,
        inputs_embeds: torch.FloatTensor,
        past_key_values: Any | None,
        output_hidden_states: bool,
        return_dict: bool,
    ) -> tuple | BaseModelOutputWithPast:
        batch_size, seq_len, _ = inputs_embeds.shape
        max_len = recurrent_lact_cache_seen_tokens(past_key_values) + seq_len
        (
            caches,
            seen_tokens,
            pending_start,
            fastw,
            masterw,
            momentum_buf,
        ) = self._init_or_load_generation_caches(
            past_key_values,
            batch_size,
            max_len,
            inputs_embeds.device,
        )
        pending_len = self._pending_len(caches, seen_tokens, pending_start)

        # HF contract: `hidden_states` is one FULL-sequence tensor per layer
        # (plus the final normed state) -- not one entry per (chunk, layer).
        # Collect per-chunk slices and concatenate along seq at the end, so
        # `hidden_states[-1]` is the full-length final state that downstream
        # per-token-loss evaluation expects.
        layer_inputs_per_chunk = [] if output_hidden_states else None
        all_hidden_states = None
        chunk_outputs = []
        local_start = 0
        current_seen = seen_tokens

        while local_start < seq_len:
            if pending_len >= self.chunk_size:
                fastw, masterw, momentum_buf = self._update_generation_memory_blocks(
                    caches,
                    fastw,
                    masterw,
                    momentum_buf,
                    pending_start,
                )
                self._clear_pending_generation_features(caches)
                pending_start = current_seen
                pending_len = 0

            take = min(seq_len - local_start, self.chunk_size - pending_len)
            chunk_inputs = inputs_embeds[:, local_start:local_start + take]
            append_pending = pending_len > 0
            hidden_states = chunk_inputs
            chunk_layer_inputs = [] if output_hidden_states else None

            for layer_idx, layer in enumerate(self.layers):
                if output_hidden_states:
                    chunk_layer_inputs.append(hidden_states)
                (
                    hidden_states,
                    x_attn_in,
                    x_memory_in,
                    attn_k,
                    attn_v,
                    fastw_out,
                    masterw_out,
                    momentum_buf_out,
                ) = layer(
                    hidden_states,
                    caches[layer_idx]["kv_cache"],
                    fastw,
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

            chunk_outputs.append(hidden_states)
            if output_hidden_states:
                layer_inputs_per_chunk.append(chunk_layer_inputs)

            for layer_idx, layer in enumerate(self.layers):
                if self.attention_v_gap is not None:
                    pre_vi = self.pre_attention_norm(caches[self.attention_v_layer_idxs[layer_idx]]["xi_out"])
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
        hidden_states = self._apply_output_norm(hidden_states)
        if output_hidden_states:
            all_hidden_states = tuple(
                torch.cat([chunk[i] for chunk in layer_inputs_per_chunk], dim=1)
                for i in range(len(self.layers))
            ) + (hidden_states,)
        next_cache = make_recurrent_lact_cache(
            layers=caches,
            seen_tokens=current_seen,
            pending_start=pending_start,
            batch_size=batch_size,
            fastw=fastw,
            masterw=masterw,
            momentum_buf=momentum_buf,
        )

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, None] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=None,
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
    ) -> tuple | BaseModelOutputWithPast:
        if output_attentions:
            warnings.warn(
                "`RecurrentLactRefUtttDenseV3Model` does not support output attention weights, "
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

        batch_size, seq_len, _ = inputs_embeds.shape
        if use_cache:
            return self._forward_with_generation_cache(
                inputs_embeds=inputs_embeds,
                past_key_values=past_key_values,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # One shared recurrent state is read by every layer in this chunk.
        fastw, masterw, momentum_buf = self._init_shared_fast_weights(batch_size)

        # Initialize attention/features caches for all layers.
        caches = []
        for layer in self.layers:
            k_cache, v_cache = layer.attn.cache_init(batch_size, seq_len, device=inputs_embeds.device, dtype=torch.bfloat16)
            if layer.memory is not None:
                layer.memory.init_rotary_cache(seq_len, device=inputs_embeds.device, dtype=torch.bfloat16)
            caches.append({
                "kv_cache": (k_cache, v_cache),
                "xi_out": None,
                "x_attn_in": None,
                "x_memory_in": None,
                "attn_k": None,
                "attn_v": None,
            })

        # See the generation-cache path: `hidden_states` must be one
        # FULL-sequence tensor per layer, so accumulate per chunk and
        # concatenate along the sequence dim once all chunks are done.
        layer_inputs_per_chunk = [] if output_hidden_states else None
        all_hidden_states = None
        chunk_outputs = []

        # Process sequence chunk by chunk
        for chunk_start in range(0, seq_len, self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, seq_len)
            chunk_inputs = inputs_embeds[:, chunk_start:chunk_end]

            # Forward through all layers (always use gradient checkpointing)
            hidden_states = chunk_inputs
            chunk_layer_inputs = [] if output_hidden_states else None
            for layer_idx, layer in enumerate(self.layers):
                if output_hidden_states:
                    chunk_layer_inputs.append(hidden_states)

                def layer_forward_with_gradient_checkpointing(layer, hidden_states, kv_cache, fastw, seqlen_offset):
                    return checkpoint(
                        layer,
                        hidden_states,
                        kv_cache,
                        fastw,
                        seqlen_offset=seqlen_offset,
                        preserve_rng_state=False,
                        use_reentrant=False,
                    )

                hidden_states, x_attn_in, x_memory_in, attn_k, attn_v, fastw_out, masterw_out, momentum_buf_out = layer_forward_with_gradient_checkpointing(
                    layer if chunk_start == 0 else layer.forward,
                    hidden_states,
                    caches[layer_idx]["kv_cache"],
                    fastw,
                    chunk_start,
                )
                caches[layer_idx]["xi_out"] = hidden_states
                caches[layer_idx]["x_attn_in"] = x_attn_in
                caches[layer_idx]["x_memory_in"] = x_memory_in
                caches[layer_idx]["attn_k"] = attn_k
                caches[layer_idx]["attn_v"] = attn_v

            chunk_outputs.append(hidden_states)

            if output_hidden_states:
                layer_inputs_per_chunk.append(chunk_layer_inputs)

            # Attention Update
            # Update KV cache for all layers
            for layer_idx, layer in enumerate(self.layers):
                if self.attention_v_gap is not None:
                    pre_vi = self.pre_attention_norm(caches[self.attention_v_layer_idxs[layer_idx]]["xi_out"])
                    if self.shared_kv_cache:
                        pre_vi = self.to_kv_cache(pre_vi)
                else:
                    pre_vi = caches[layer_idx]["x_attn_in"]

                caches[layer_idx]["kv_cache"] = layer.update_kv(pre_vi, caches[layer_idx]["kv_cache"], apply_layer_kv=not self.shared_kv_cache)

            # Memory Update.  Keep ref_v4's shifted causal order: read this
            # chunk with its start snapshot, then write once (except at EOF).
            is_last_chunk = (chunk_start + self.chunk_size >= seq_len)
            if self.use_memory_block and not is_last_chunk:
                fastw, masterw, momentum_buf = self._aggregate_memory_update(
                    caches,
                    fastw,
                    masterw,
                    momentum_buf,
                    chunk_start,
                )

        # Concatenate all chunk outputs and apply final norm
        hidden_states = torch.cat(chunk_outputs, dim=1)
        hidden_states = self._apply_output_norm(hidden_states)
        if output_hidden_states:
            all_hidden_states = tuple(
                torch.cat([chunk[i] for chunk in layer_inputs_per_chunk], dim=1)
                for i in range(len(self.layers))
            ) + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, None, all_hidden_states, None] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=None,
            hidden_states=all_hidden_states,
            attentions=None,
        )


# ============================================================================
# Causal LM
# ============================================================================

class RecurrentLactRefUtttDenseV3ForCausalLM(
    RecurrentLactRefUtttDenseV3PreTrainedModel,
    GenerationMixin,
):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = RecurrentLactRefUtttDenseV3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.criterion = None

        self.post_init()

    def get_input_embeddings(self):
        return self.model.embeddings

    def set_input_embeddings(self, value):
        self.model.embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

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

    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
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
    ) -> tuple | CausalLMOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)

        outputs = self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

        hidden_states = outputs[0]
        logits = None if self.config.fuse_linear_cross_entropy else self.lm_head(hidden_states[:, -logits_to_keep:])

        loss = None
        if labels is not None:
            if getattr(self, "criterion", None) is None:
                if self.config.fuse_linear_cross_entropy:
                    criterion = FusedLinearCrossEntropyLoss(use_l2warp=self.config.use_l2warp)
                elif self.config.fuse_cross_entropy:
                    criterion = FusedCrossEntropyLoss(inplace_backward=True)
                else:
                    criterion = nn.CrossEntropyLoss()
            else:
                criterion = self.criterion
            labels = labels.to(hidden_states.device)
            labels = torch.cat((labels[..., 1:], torch.full_like(labels[:, :1], criterion.ignore_index)), 1)
            if self.config.fuse_linear_cross_entropy:
                loss = criterion(hidden_states, labels, self.lm_head.weight, self.lm_head.bias)
            else:
                loss = criterion(logits.view(labels.numel(), -1), labels.view(-1))
                loss = l2_warp(loss, logits) if self.config.use_l2warp else loss

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# Explicit v3 names mirror the v2 test/import surface while keeping the
# underlying ref_v4 block implementation local to this directory.
FastWeightMLPSubNNUtttDenseV3 = FastWeightMLPSubNN
RecurrentLactRefUtttDenseV3Block = RecurrentLactBlock
RecurrentLactRefModel = RecurrentLactRefUtttDenseV3Model
RecurrentLactRefForCausalLM = RecurrentLactRefUtttDenseV3ForCausalLM


__all__ = [
    "FastWeightMLPSubNN",
    "FastWeightMLPSubNNUtttDenseV3",
    "RecurrentLactRefForCausalLM",
    "RecurrentLactRefModel",
    "RecurrentLactRefUtttDenseV3Block",
    "RecurrentLactRefUtttDenseV3ForCausalLM",
    "RecurrentLactRefUtttDenseV3Model",
]
