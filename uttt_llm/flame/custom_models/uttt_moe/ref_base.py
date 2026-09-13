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

from .configuration_recurrent_lact import RecurrentLactRefConfig
from .rotary import RotaryEmbedding

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


def resolve_memory_compose_modes(
    config: RecurrentLactRefConfig,
    memory_v_gap: int | None,
) -> Tuple[str, str]:
    explicit_k_mode = getattr(config, "memory_k_compose_mode", None)
    explicit_v_mode = getattr(config, "memory_v_compose_mode", None)
    explicit_modes = (
        getattr(config, "memory_shared_kv_cache", False),
        getattr(config, "memory_reuse_kv_cache", False),
        getattr(config, "memory_perlayer_k_share_v", False),
        getattr(config, "memory_perlayer_k_reuse_v", False),
    )
    if explicit_k_mode is not None or explicit_v_mode is not None:
        if any(explicit_modes):
            raise ValueError(
                "`memory_*_compose_mode` cannot be mixed with legacy explicit memory compose flags.",
            )
        if config.memory_kv_mode == "reuse_kv":
            k_mode = "reuse"
            v_mode = "reuse"
        elif config.memory_kv_mode == "to_kv":
            if config.memory_kv_proj == "reuse_attn_kv_proj":
                k_mode = "attn-proj"
                v_mode = "attn-proj"
            elif config.memory_kv_proj == "model_kv_proj":
                k_mode = "shared"
                v_mode = "shared"
            elif config.memory_kv_proj == "ttt_kv_proj":
                k_mode = "ttt-proj"
                v_mode = "ttt-proj"
            else:
                raise ValueError(f"Invalid memory_kv_proj: {config.memory_kv_proj}")
        else:
            raise ValueError(f"Invalid memory_kv_mode: {config.memory_kv_mode}")
        if memory_v_gap is not None and explicit_v_mode is None:
            v_mode = "attn-proj"
        return explicit_k_mode or k_mode, explicit_v_mode or v_mode

    if any(explicit_modes):
        if getattr(config, "memory_perlayer_k_share_v", False):
            return "attn-proj", "shared"
        if getattr(config, "memory_perlayer_k_reuse_v", False):
            return "attn-proj", "reuse"
        if getattr(config, "memory_shared_kv_cache", False):
            return "shared", "shared"
        return "reuse", "reuse"

    if config.memory_kv_mode == "reuse_kv":
        k_mode = "reuse"
        v_mode = "reuse"
    elif config.memory_kv_mode == "to_kv":
        if config.memory_kv_proj == "reuse_attn_kv_proj":
            k_mode = "attn-proj"
            v_mode = "attn-proj"
        elif config.memory_kv_proj == "model_kv_proj":
            k_mode = "shared"
            v_mode = "shared"
        elif config.memory_kv_proj == "ttt_kv_proj":
            k_mode = "ttt-proj"
            v_mode = "ttt-proj"
        else:
            raise ValueError(f"Invalid memory_kv_proj: {config.memory_kv_proj}")
    else:
        raise ValueError(f"Invalid memory_kv_mode: {config.memory_kv_mode}")

    # Keep legacy v-gap behavior: source hidden is normalized, then current
    # layer attention v_proj produces the memory V.
    if memory_v_gap is not None:
        v_mode = "attn-proj"
    return k_mode, v_mode


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


class LowRankFastWeight(nn.Module):
    def __init__(
        self,
        num_heads: int,
        out_features: int,
        in_features: int,
        rank: int = 32,
        init_gain: float = 0.5,
        add_identity: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.out_features = out_features
        self.in_features = in_features
        self.rank = rank
        self.init_gain = init_gain
        self.add_identity = add_identity

        self.w_left = nn.Parameter(torch.randn(num_heads, out_features, rank))
        self.w_right = nn.Parameter(torch.randn(num_heads, rank, in_features))

    def _init_weights(self):
        nn.init.normal_(self.w_left, std=1.0 / math.sqrt(self.rank) * self.init_gain)
        nn.init.normal_(
            self.w_right,
            std=1.0 / math.sqrt(self.in_features) * self.init_gain,
        )

    def reset_parameters(self):
        self._init_weights()

    def forward(self) -> torch.Tensor:
        weight = self.w_left @ self.w_right
        if self.add_identity:
            weight = weight + (
                torch.eye(
                    self.out_features,
                    self.in_features,
                    device=weight.device,
                    dtype=weight.dtype,
                ).unsqueeze(0)
                * 0.5
            )
        return weight


class FastWeightMLPSubNN(nn.Module):
    """
    Fast Weight MLP that updates online during forward pass.
    Closely follows the reference LaCT implementation.
    """

    def __init__(
        self,
        config: RecurrentLactRefConfig,
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

        # QKV projections (only created for the TTT paths that are actually used)
        use_ttt_qkv_proj = config.memory_kv_proj == "ttt_kv_proj" and config.memory_kv_mode != "reuse_kv"
        use_reuse_kv_v_only_ttt_proj = (
            config.memory_kv_proj == "ttt_kv_proj"
            and config.memory_kv_mode == "reuse_kv"
            and getattr(config, "memory_v_compose_mode", None) == "ttt-proj"
            and getattr(config, "memory_k_compose_mode", None) in {None, "reuse"}
            and getattr(config, "memory_k_gap", None) == 0
            and getattr(config, "memory_v_gap", None) is not None
        )
        self.to_q = None
        self.to_k = None
        self.to_v = None
        if use_ttt_qkv_proj:
            self.to_q = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
            self.to_k = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
            self.to_v = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        elif use_reuse_kv_v_only_ttt_proj:
            # In reuse-kv mode, K still comes from attention. Only V needs a TTT projection.
            self.to_v = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        # Fast weight matrices (per head)
        if self.w0_w2_low_rank > 0:
            self.w0 = LowRankFastWeight(
                self.fw_num_heads,
                self.inter_dim,
                self.head_dim,
                rank=self.w0_w2_low_rank,
                init_gain=self.fw_init_gain,
                add_identity=True,
            )
            self.w2 = LowRankFastWeight(
                self.fw_num_heads,
                self.inter_dim,
                self.head_dim,
                rank=self.w0_w2_low_rank,
                init_gain=self.fw_init_gain,
                add_identity=True,
            )
        else:
            self.w0 = nn.Parameter(
                torch.randn(self.fw_num_heads, self.inter_dim, self.head_dim) / math.sqrt(self.head_dim)
            )
            self.w2 = nn.Parameter(
                torch.randn(self.fw_num_heads, self.inter_dim, self.head_dim) / math.sqrt(self.head_dim)
            )
        self.w1 = nn.Parameter(
            torch.randn(self.fw_num_heads, self.head_dim, self.inter_dim) / math.sqrt(self.inter_dim)
        )

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
        if self.w0_w2_low_rank > 0:
            self.w0.reset_parameters()
            self.w2.reset_parameters()
        else:
            nn.init.normal_(self.w0, mean=0.0, std=1.0 / math.sqrt(self.head_dim))
            nn.init.normal_(self.w2, mean=0.0, std=1.0 / math.sqrt(self.head_dim))
        nn.init.normal_(self.w1, mean=0.0, std=1.0 / math.sqrt(self.inter_dim))

    def init_rotary_cache(self, max_len: int, device: torch.device, dtype: torch.dtype):
        """Pre-compute cos/sin cache for TTT RoPE."""
        if self.ttt_rotary is not None:
            self.ttt_rotary._update_cos_sin_cache(max_len, device=device, dtype=dtype)

    def init_fast_weights(self, batch_size: int = 1):
        """Initialize fast weights from base weights."""
        base_w0 = self.w0() if self.w0_w2_low_rank > 0 else self.w0
        base_w2 = self.w2() if self.w0_w2_low_rank > 0 else self.w2
        master_weight = (
            base_w0.unsqueeze(0).repeat(batch_size, 1, 1, 1),
            self.w1.unsqueeze(0).repeat(batch_size, 1, 1, 1),
            base_w2.unsqueeze(0).repeat(batch_size, 1, 1, 1),
        )
        weight = tuple(w.to(torch.bfloat16) for w in master_weight)
        if self.use_momentum:
            momentum_buf = (
                torch.zeros_like(master_weight[0]),
                torch.zeros_like(master_weight[1]),
                torch.zeros_like(master_weight[2]),
            )
        else:
            momentum_buf = None
        return weight, master_weight, momentum_buf

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
            out = self.ttt_norm(out)
            out = rearrange(out, '(b n_h) s d -> b s (n_h d)', n_h=self.fw_num_heads)
        if self.enable_memory_output_proj:
            out = self.output_proj(out)

        return out

    def apply_ttt_scale(self, memory_out: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply per-head learnable scale to memory output."""
        memory_out = rearrange(memory_out, 'b s (n_h d) -> (b n_h) s d', n_h=self.fw_num_heads)
        ttt_x_normed = self.ttt_norm(memory_out)
        # scalar scale per head
        ttt_scale = F.silu(self.ttt_scale_proj(hidden_states), inplace=False)
        ttt_scale = rearrange(ttt_scale, 'b s (n_h d) -> (b n_h) s d', n_h=self.fw_num_heads)
        ttt_x_normed = ttt_x_normed * ttt_scale
        ttt_x_normed = rearrange(ttt_x_normed, '(b n_h) s d -> b s (n_h d)', n_h=self.fw_num_heads)
        return ttt_x_normed

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
    ) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...] | None]:
        """
        Update fast weights using gradient descent.
        Args:
            k: [b, l, d] - pre-projected key values (from to_k or attention)
            v: [b, l, d] - pre-projected target values (from to_v or attention)
            lr: [b, l, 3] - per-element learning rates
            fastw: (w0, w1, w2) BF16 fast weights
            masterw: (w0, w1, w2) FP32 master weights
            momentum_buf: (dw0_mom, dw1_mom, dw2_mom) momentum buffers, or None
            pre_vi: [b, l, d] - pre-projected key values (from to_k or attention)
            seqlen_offset: position offset for TTT RoPE
        Returns:
            (new_fastw, new_masterw, new_momentum_buf)
        """
        w0_master, w1_master, w2_master = masterw
        w0, w1, w2 = fastw
        b, l, d = k.shape

        # Preprocess k and v on original positions (before repeat),
        # so that RoPE can be applied with correct position indices.
        # All preprocessing ops are element-wise per-token, so
        # preprocess(repeat(x)) == repeat(preprocess(x)).
        if self.qk_rescale:
            _, k = self._rescale_qk(k, k)  # only apply k channel for update
        if self.qkv_silu:
            k = F.silu(k)
            v = F.silu(v)
        if self.v_norm:
            v = self.norm_fn(v)
        if self.qk_norm:
            k = rearrange(k, 'b l (h d) -> (b h) l d', h=self.fw_num_heads)
            k = self.qk_norm_fn(k)
            k = rearrange(k, '(b h) l d -> b l (h d)', h=self.fw_num_heads)

        # Apply TTT RoPE on original positions (before repeat)
        if self.ttt_rotary is not None:
            k = rearrange(k, 'b l (nh d) -> b l nh d', nh=self.n_rope_heads)
            k = k.to(self.ttt_rotary._cos_cached.dtype)
            _, k = self.ttt_rotary(None, k, seqlen_offset=seqlen_offset)
            k = rearrange(k, 'b l nh d -> b l (nh d)')

        k_mh = rearrange(k, 'b l (h hd) -> (b h) l hd', h=self.fw_num_heads)
        v_mh = rearrange(v, 'b l (h hd) -> (b h) l hd', h=self.fw_num_heads)
        w0_flat = rearrange(w0, 'b h k d -> (b h) k d')
        w1_flat = rearrange(w1, 'b h d k -> (b h) d k')
        w2_flat = rearrange(w2, 'b h k d -> (b h) k d')

        if lr.shape[2] == 3:
            # shared mode: lr [b, l, 3] → broadcast across heads
            lr0, lr1, lr2 = lr.split(1, dim=2)  # [b, l, 1] each
            lr0_mh = rearrange(
                lr0.expand(-1, -1, self.fw_num_heads),
                'b l h -> (b h) l 1',
            )
            lr1_mh = rearrange(
                lr1.expand(-1, -1, self.fw_num_heads),
                'b l h -> (b h) l 1',
            )
            lr2_mh = rearrange(
                lr2.expand(-1, -1, self.fw_num_heads),
                'b l h -> (b h) l 1',
            )
        else:
            # per-head mode: lr [b, l, fw_num_heads*3] → per-head
            lr = lr.view(lr.shape[0], lr.shape[1], self.fw_num_heads, 3)  # [b, l, h, 3]
            lr0_mh = rearrange(lr[:, :, :, 0:1], 'b l h d -> (b h) l d')
            lr1_mh = rearrange(lr[:, :, :, 1:2], 'b l h d -> (b h) l d')
            lr2_mh = rearrange(lr[:, :, :, 2:3], 'b l h d -> (b h) l d')
        w0_master_flat = rearrange(w0_master, 'b h k d -> (b h) k d')
        w1_master_flat = rearrange(w1_master, 'b h d k -> (b h) d k')
        w2_master_flat = rearrange(w2_master, 'b h k d -> (b h) k d')

        if self.use_momentum and momentum_buf is not None and pre_vi is not None:
            momentum = self.momentum_proj(pre_vi)
            momentum = rearrange(momentum, 'b l h -> (b h) l 1')
            dw0_mom, dw1_mom, dw2_mom = momentum_buf
            dw0_mom = rearrange(dw0_mom, 'b h k d -> (b h) k d')
            dw1_mom = rearrange(dw1_mom, 'b h d k -> (b h) d k')
            dw2_mom = rearrange(dw2_mom, 'b h k d -> (b h) k d')
        else:
            momentum = None
            dw0_mom = None
            dw1_mom = None
            dw2_mom = None

        # Keep the lact single-step update path inline so recurrent state updates
        # follow the same numerics without routing through a separate helper.
        with torch.autocast(
            device_type=k_mh.device.type,
            enabled=k_mh.device.type == "cuda",
            dtype=torch.bfloat16,
        ):
            w0_norm = w0_flat.norm(dim=2, keepdim=True)
            w1_norm = w1_flat.norm(dim=2, keepdim=True)
            w2_norm = w2_flat.norm(dim=2, keepdim=True)

            vi = v_mh.transpose(1, 2)
            gate_before_act = torch.bmm(w0_flat, k_mh.transpose(1, 2))
            hidden_before_mul = torch.bmm(w2_flat, k_mh.transpose(1, 2))
            hidden = F.silu(gate_before_act, inplace=False) * hidden_before_mul

            dhidden = torch.bmm(w1_flat.transpose(1, 2), vi)
            dhidden_before_mul = dhidden * F.silu(gate_before_act, inplace=False)
            dgate = dhidden * hidden_before_mul
            dgate_before_act = silu_backprop(dgate, gate_before_act)

            dw1 = torch.bmm(vi, (hidden.transpose(1, 2) * lr1_mh).type_as(vi))
            dw0 = torch.bmm(dgate_before_act, (k_mh * lr0_mh).type_as(dgate_before_act))
            dw2 = torch.bmm(dhidden_before_mul, (k_mh * lr2_mh).type_as(dhidden_before_mul))

            if momentum is not None:
                m_i = momentum.mean(dim=1, keepdim=True)
                dw0 = dw0 + dw0_mom * m_i
                dw1 = dw1 + dw1_mom * m_i
                dw2 = dw2 + dw2_mom * m_i
                dw0_mom = dw0
                dw1_mom = dw1
                dw2_mom = dw2

            if self.use_moun:
                dw1 = zeropower_via_newtonschulz5(dw1, 5)
                dw0 = zeropower_via_newtonschulz5(dw0, 5)
                dw2 = zeropower_via_newtonschulz5(dw2, 5)

            w1_master_flat = w1_master_flat + dw1
            w0_master_flat = w0_master_flat + dw0
            w2_master_flat = w2_master_flat + dw2

            w0_flat = w0_master_flat / (w0_master_flat.norm(dim=2, keepdim=True) + 1e-5) * w0_norm
            w1_flat = w1_master_flat / (w1_master_flat.norm(dim=2, keepdim=True) + 1e-5) * w1_norm
            w2_flat = w2_master_flat / (w2_master_flat.norm(dim=2, keepdim=True) + 1e-5) * w2_norm

        masterw = (
            rearrange(w0_master_flat, '(b h) k d -> b h k d', b=b, h=self.fw_num_heads),
            rearrange(w1_master_flat, '(b h) d k -> b h d k', b=b, h=self.fw_num_heads),
            rearrange(w2_master_flat, '(b h) k d -> b h k d', b=b, h=self.fw_num_heads),
        )
        weight = (
            rearrange(w0_flat, '(b h) k d -> b h k d', b=b, h=self.fw_num_heads).to(torch.bfloat16),
            rearrange(w1_flat, '(b h) d k -> b h d k', b=b, h=self.fw_num_heads).to(torch.bfloat16),
            rearrange(w2_flat, '(b h) k d -> b h k d', b=b, h=self.fw_num_heads).to(torch.bfloat16),
        )
        if momentum is not None:
            momentum_buf = (
                rearrange(dw0_mom, '(b h) k d -> b h k d', b=b, h=self.fw_num_heads),
                rearrange(dw1_mom, '(b h) d k -> b h d k', b=b, h=self.fw_num_heads),
                rearrange(dw2_mom, '(b h) k d -> b h k d', b=b, h=self.fw_num_heads),
            )

        return weight, masterw, momentum_buf

    def extra_repr(self) -> str:
        return (
            f"head_dim={self.head_dim}, inter_dim={self.inter_dim}, "
            f"fw_num_heads={self.fw_num_heads}, low_rank={self.w0_w2_low_rank}"
        )


# ============================================================================
# Recurrent Attention
# ============================================================================

class RecurrentAttention(nn.Module):
    def __init__(self, config: RecurrentLactRefConfig, layer_idx: int):
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
        k = self.k_proj(feature)  # [b, l, kv_dim]
        v = self.v_proj(feature)  # [b, l, kv_dim]
        return k, v

    def _compose_memory_kv(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.attention_tokv(hidden_states)

    @torch.compile
    def compose_memory_kv(self, *args):
        return checkpoint(
            self._compose_memory_kv,
            *args,
            preserve_rng_state=False,
            use_reentrant=False,
        )

    def _compose_memory_k(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.k_proj(hidden_states)

    @torch.compile
    def compose_memory_k(self, *args):
        return checkpoint(
            self._compose_memory_k,
            *args,
            preserve_rng_state=False,
            use_reentrant=False,
        )

    def _compose_memory_v(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.v_proj(hidden_states)

    @torch.compile
    def compose_memory_v(self, *args):
        return checkpoint(
            self._compose_memory_v,
            *args,
            preserve_rng_state=False,
            use_reentrant=False,
        )

    def project_memory_vgap_hidden(self, *args):
        return self.compose_memory_v(*args)

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
    def __init__(self, config: RecurrentLactRefConfig, layer_idx: int):
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

    def init_fast_weight(self, batch_size: int = 1):
        """Initialize fast weights for memory block."""
        if self.memory is not None:
            return self.memory.init_fast_weights(batch_size=batch_size)
        return None, None, None

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

    def memory_tov(self, feature: torch.Tensor) -> torch.Tensor:
        """Project feature to value using memory's V projection only."""
        return self.memory.to_v(feature)

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        fastw: Tuple[torch.Tensor, ...] | None = None,
        seqlen_offset: int = 0,
        **kwargs: Any,
    ):
        # Initialize fast weights inside FSDP forward scope (first chunk only)
        masterw = None
        momentum_buf = None
        if self.memory is not None and fastw is None:
            fastw, masterw, momentum_buf = self.memory.init_fast_weights(
                batch_size=hidden_states.size(0)
            )

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
        hidden_states = residual + self.mlp(hidden_states, **kwargs)

        return hidden_states, x_attn_in, x_memory_in, attn_k, attn_v, fastw, masterw, momentum_buf

    def _update_fast_weight(
        self,
        pre_vi: torch.Tensor,
        fastw: Tuple[torch.Tensor, ...],
        masterw: Tuple[torch.Tensor, ...],
        momentum_buf: Tuple[torch.Tensor, ...] | None = None,
        ki: torch.Tensor = None,
        vi: torch.Tensor = None,
        seqlen_offset: int = 0,
    ) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...] | None]:
        """Update fast weights following reference implementation."""
        if self.memory is None:
            return fastw, masterw, momentum_buf

        # LR prediction
        lri = self.to_lr(pre_vi)
        lri = F.softplus(lri.float() + self.base_lr_inv)

        # ki and vi are projected by Model before entering this update path.
        fastw, masterw, momentum_buf = self.memory.update(
            ki, vi, lri, fastw, masterw,
            momentum_buf=momentum_buf,
            pre_vi=pre_vi,
            seqlen_offset=seqlen_offset,
        )
        return fastw, masterw, momentum_buf

    @torch.compile
    def update_fast_weight(self, *args, **kwargs):
        return checkpoint(self._update_fast_weight, *args, **kwargs, preserve_rng_state=False, use_reentrant=False)

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

class RecurrentLactRefPreTrainedModel(PreTrainedModel):
    config_class = RecurrentLactRefConfig
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
            if module.w0_w2_low_rank > 0:
                module.w0._init_weights()
                module.w2._init_weights()
            else:
                nn.init.normal_(module.w0, mean=0.0, std=1.0 / math.sqrt(module.head_dim))
                nn.init.normal_(module.w2, mean=0.0, std=1.0 / math.sqrt(module.head_dim))
            nn.init.normal_(module.w1, mean=0.0, std=1.0 / math.sqrt(module.inter_dim))
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

class RecurrentLactRefModel(RecurrentLactRefPreTrainedModel):
    def __init__(self, config: RecurrentLactRefConfig):
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
        self.memory_k_compose_mode, self.memory_v_compose_mode = resolve_memory_compose_modes(
            config,
            self.memory_v_gap,
        )

        if self.attention_v_gap is not None:
            self.pre_attention_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)
        if self.memory_v_gap is not None and self.memory_v_compose_mode in {"attn-proj", "shared"}:
            self.pre_memory_v_rms_norm = (RMSNorm if config.fuse_norm else nn.RMSNorm)(
                config.hidden_size,
                eps=config.norm_eps,
            )
        else:
            self.pre_memory_v_rms_norm = None

        num_kv_heads = config.num_kv_heads if config.num_kv_heads is not None else config.num_heads
        kv_head_dim = config.hidden_size // config.num_heads
        kv_dim = num_kv_heads * kv_head_dim
        if self.memory_k_compose_mode == "shared":
            self.memory_shared_k_proj = nn.Linear(config.hidden_size, kv_dim, bias=config.qkv_bias)
        else:
            self.memory_shared_k_proj = None
        if self.memory_v_compose_mode == "shared":
            self.memory_shared_v_proj = nn.Linear(config.hidden_size, kv_dim, bias=config.qkv_bias)
        else:
            self.memory_shared_v_proj = None

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [RecurrentLactBlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = (RMSNorm if config.last_layer_fuse_norm else nn.RMSNorm)(config.hidden_size, eps=config.norm_eps)

        if self.shared_kv_cache:
            kv_dim = 2 * self.layers[0].attn.num_kv_heads * self.layers[0].attn.head_dim
            self.to_kv_cache = nn.Linear(config.hidden_size, kv_dim, bias=config.qkv_bias)

        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings = value

    def extra_repr(self) -> str:
        lines = [f"memory_compose: k={self.memory_k_compose_mode}, v={self.memory_v_compose_mode}"]
        for i, v_layer_idx in enumerate(self.attention_v_layer_idxs):
            lines.append(f"vmap: {i} <- {v_layer_idx}")
        for i, k_layer_idx in enumerate(self.attention_k_layer_idxs):
            lines.append(f"kmap: {i} <- {k_layer_idx}")
        for i, v_layer_idx in enumerate(self.memory_v_layer_idxs):
            lines.append(f"vmap: {i} <- {v_layer_idx}")
        for i, k_layer_idx in enumerate(self.memory_k_layer_idxs):
            lines.append(f"kmap: {i} <- {k_layer_idx}")
        return "\n".join(lines)

    def _get_memory_k_source_cache(self, caches: list[dict[str, Any]], layer_idx: int) -> dict[str, Any]:
        if self.memory_k_gap is not None:
            return caches[self.memory_k_layer_idxs[layer_idx]]
        return caches[layer_idx]

    def _get_memory_v_source_cache(self, caches: list[dict[str, Any]], layer_idx: int) -> dict[str, Any]:
        if self.memory_v_gap is not None:
            return caches[self.memory_v_layer_idxs[layer_idx]]
        return caches[layer_idx]

    def _prepare_memory_v_source_feature(self, cache: dict[str, Any]) -> torch.Tensor:
        if self.memory_v_gap is not None:
            feature = cache["xi_out"]
            if self.pre_memory_v_rms_norm is not None:
                feature = self.pre_memory_v_rms_norm(feature)
            return feature
        return cache[self.memory_kv_feature]

    def _project_shared_memory_k(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.memory_shared_k_proj is None:
            raise ValueError("`memory_shared_k_proj` is not initialized.")
        return self.memory_shared_k_proj(hidden_states)

    @torch.compile
    def project_shared_memory_k(self, *args):
        return checkpoint(
            self._project_shared_memory_k,
            *args,
            preserve_rng_state=False,
            use_reentrant=False,
        )

    def _project_shared_memory_v(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.memory_shared_v_proj is None:
            raise ValueError("`memory_shared_v_proj` is not initialized.")
        return self.memory_shared_v_proj(hidden_states)

    @torch.compile
    def project_shared_memory_v(self, *args):
        return checkpoint(
            self._project_shared_memory_v,
            *args,
            preserve_rng_state=False,
            use_reentrant=False,
        )

    def _compose_memory_update_inputs(
        self,
        layer: RecurrentLactBlock,
        layer_idx: int,
        caches: list[dict[str, Any]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cache_for_ki = self._get_memory_k_source_cache(caches, layer_idx)
        cache_for_vi = self._get_memory_v_source_cache(caches, layer_idx)

        pre_update = cache_for_ki[self.memory_kv_feature]
        k_feature = cache_for_ki[self.memory_kv_feature]
        v_feature = self._prepare_memory_v_source_feature(cache_for_vi)

        if self.memory_k_compose_mode == "reuse":
            ki = cache_for_ki["attn_k"]
        elif self.memory_k_compose_mode == "attn-proj":
            if self.memory_v_compose_mode == "attn-proj" and k_feature is v_feature:
                ki, vi = layer.attn.compose_memory_kv(k_feature)
                return pre_update, ki, vi
            ki = layer.attn.compose_memory_k(k_feature)
        elif self.memory_k_compose_mode == "shared":
            ki = self.project_shared_memory_k(k_feature)
        elif self.memory_k_compose_mode == "ttt-proj":
            if self.memory_v_compose_mode == "ttt-proj" and k_feature is v_feature:
                ki, vi = layer.memory_tokv(k_feature)
                return pre_update, ki, vi
            ki, _ = layer.memory_tokv(k_feature)
        else:
            raise ValueError(f"Invalid memory K compose mode: {self.memory_k_compose_mode}")

        if self.memory_v_compose_mode == "reuse":
            vi = cache_for_vi["attn_v"]
        elif self.memory_v_compose_mode == "attn-proj":
            vi = layer.attn.compose_memory_v(v_feature)
        elif self.memory_v_compose_mode == "shared":
            vi = self.project_shared_memory_v(v_feature)
        elif self.memory_v_compose_mode == "ttt-proj":
            vi = layer.memory_tov(v_feature)
        else:
            raise ValueError(f"Invalid memory V compose mode: {self.memory_v_compose_mode}")

        return pre_update, ki, vi

    def _update_memory_blocks_for_chunk(
        self,
        caches: list[dict[str, Any]],
        chunk_start: int,
        seq_len: int,
    ) -> None:
        is_last_chunk = (chunk_start + self.chunk_size >= seq_len)
        if not self.use_memory_block or is_last_chunk:
            return

        for layer_idx in range(len(self.layers)):
            layer = self.layers[layer_idx]
            cache = caches[layer_idx]
            pre_update, ki, vi = self._compose_memory_update_inputs(layer, layer_idx, caches)

            fastw, masterw, momentum_buf = layer.update_fast_weight(
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
                "`RecurrentLactRefModel` does not support output attention weights, "
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

        # Initialize caches for all layers
        # Note: fastw/masterw/momentum_buf are initialized lazily inside
        # each layer's first forward() call, within the FSDP all-gather scope.
        caches = []
        for layer in self.layers:
            k_cache, v_cache = layer.attn.cache_init(batch_size, seq_len, device=inputs_embeds.device, dtype=torch.bfloat16)
            if layer.memory is not None:
                layer.memory.init_rotary_cache(seq_len, device=inputs_embeds.device, dtype=torch.bfloat16)
            caches.append({
                "kv_cache": (k_cache, v_cache),
                "fastw": None,
                "masterw": None,
                "momentum_buf": None,
                "xi_out": None,
                "x_attn_in": None,
                "x_memory_in": None,
                "attn_k": None,
                "attn_v": None,
            })

        all_hidden_states = () if output_hidden_states else None
        chunk_outputs = []

        # Process sequence chunk by chunk
        for chunk_start in range(0, seq_len, self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, seq_len)
            chunk_inputs = inputs_embeds[:, chunk_start:chunk_end]

            # Forward through all layers (always use gradient checkpointing)
            hidden_states = chunk_inputs
            for layer_idx, layer in enumerate(self.layers):
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

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
                    caches[layer_idx]["fastw"],
                    chunk_start,
                )
                caches[layer_idx]["xi_out"] = hidden_states
                caches[layer_idx]["x_attn_in"] = x_attn_in
                caches[layer_idx]["x_memory_in"] = x_memory_in
                caches[layer_idx]["attn_k"] = attn_k
                caches[layer_idx]["attn_v"] = attn_v
                # First chunk: store fast weights initialized inside FSDP scope
                if masterw_out is not None:
                    caches[layer_idx]["fastw"] = fastw_out
                    caches[layer_idx]["masterw"] = masterw_out
                    caches[layer_idx]["momentum_buf"] = momentum_buf_out

            chunk_outputs.append(hidden_states)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

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

            # Memory Update
            # Update memory blocks for all layers (skip last chunk to match lact_swiglu reference)
            self._update_memory_blocks_for_chunk(caches, chunk_start, seq_len)

        # Concatenate all chunk outputs and apply final norm
        hidden_states = torch.cat(chunk_outputs, dim=1)
        hidden_states = self.norm(hidden_states)

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

class RecurrentLactRefForCausalLM(RecurrentLactRefPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = RecurrentLactRefModel(config)
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
