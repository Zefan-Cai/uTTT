# -*- coding: utf-8 -*-

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint
from transformers.utils import logging

from fla.models.utils import Cache
from fla.modules import RMSNorm, RotaryEmbedding
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange, repeat

from .ttt_operation import (
    block_causal_lact_swiglu,
    prenorm_block_causal_lact_swiglu,
    l2_norm,
    silu_backprop,
    zeropower_via_newtonschulz5,
)

from .ttt_operation_fused_kernel import (
    postnorm_block_causal_lact_swiglu_fused_kernel_triton,
    prenorm_block_causal_lact_swiglu_fused_kernel_triton,
)
from ...lact_cache_utils import is_recurrent_lact_cache

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input
except ImportError:
    warnings.warn(
        "Flash Attention is not installed. Please install it via `pip install flash-attn --no-build-isolation`",
        category=ImportWarning,
    )
    flash_attn_func = None

logger = logging.get_logger(__name__)


def inv_softplus(x):
    if isinstance(x, torch.Tensor):
        y = x + torch.log(-torch.expm1(-x))
    else:
        y = x + math.log(-math.expm1(-x))
    return y


class LowRankFastWeight(nn.Module):
    """
    Low rank fast weight. This is a compromise to keep the number of parameters low when comparing against baselines.
    Idealy, low-rank parameterization always hurts the performance.
    Args:
        num_heads: number of heads
        out_features: output features
        in_features: input features
        rank: rank of the low rank fast weight
        init_gain: initialization gain
        add_identity: whether to add identity matrix to the fast weight
    Returns:
        W: [num_heads, out_features, in_features]
    W = W_left @ W_right + I * 0.5
        where I is the identity matrix if add_identity is True.
    """

    def __init__(
        self,
        num_heads,
        out_features,
        in_features,
        rank=32,
        init_gain=0.5,
        add_identity=False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.out_features = out_features
        self.in_features = in_features
        self.rank = rank
        self.add_identity = add_identity

        self.w_left = nn.Parameter(torch.randn(num_heads, out_features, rank))
        self.w_right = nn.Parameter(torch.randn(num_heads, rank, in_features))
        self.init_gain = init_gain

        print("init low rank fast weight", num_heads, out_features, in_features, rank)

    def _init_weights(self):

        nn.init.normal_(self.w_left, std=1.0 / math.sqrt(self.rank) * self.init_gain)
        nn.init.normal_(
            self.w_right, std=1.0 / math.sqrt(self.in_features) * self.init_gain
        )

    def reset_parameters(self):
        self._init_weights()

    def forward(
        self,
    ):
        """
        Returns:
            W: [num_heads, out_features, in_features]
            W = W_left @ W_right + I * 0.5
            where I is the identity matrix if add_identity is True.
        """

        W = self.w_left @ self.w_right

        if self.add_identity:
            W += (
                torch.eye(
                    self.out_features, self.in_features, device=W.device, dtype=W.dtype
                ).unsqueeze(0)
                * 0.5
            )

        return W


class LaCTSWIGLULayer(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_attn_heads: int,
        num_lact_heads: int,
        inter_multi: float,
        window_size: int,
        lact_chunk_size: int,
        qkv_bias: bool = False,
        attn_qk_norm: bool = True,
        qkv_silu: bool = True,
        no_v_silu: bool = False,
        lr_dim: int = 1,
        use_muon: bool = False,
        lr_parameterization: str = "mamba",
        learnable_ttt_scale: bool = False,
        ttt_prenorm: bool = False,
        ttt_nope: bool = False,
        rope_theta: float = 500000.0,
        layer_idx: int = None,
        max_position_embeddings: int = 2048,
        w0_w2_low_rank: int = -1,
        use_momentum: bool = False,
        ttt_loss_type: str = "dot_product",
        fw_init_gain: float = 0.5,  # init the fast weights
        use_fused_kernel: bool = False,
        fp32_states: bool = False,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_attn_heads  # num of heads for attention
        self.inter_multi = inter_multi
        self.window_size = window_size
        # head dim for attention
        self.head_dim = hidden_size // num_attn_heads

        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias)

        self.attn_qk_norm = attn_qk_norm
        if self.attn_qk_norm:
            self.q_norm = RMSNorm(self.hidden_size)
            self.k_norm = RMSNorm(self.hidden_size)

        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.rope_theta = rope_theta
        self.rotary = RotaryEmbedding(dim=self.head_dim, base=self.rope_theta)
        self.layer_idx = layer_idx
        self.max_position_embeddings = max_position_embeddings

        ### Fast Weight init
        self.use_muon = use_muon
        self.lact_chunk_size = lact_chunk_size
        self.num_fw_heads = num_lact_heads
        self.fw_head_dim = self.hidden_size // self.num_fw_heads
        self.qkv_silu = qkv_silu
        self.no_v_silu = no_v_silu
        self.ttt_prenorm = ttt_prenorm
        self.ttt_nope = ttt_nope

        d_in, d_out = self.fw_head_dim, self.fw_head_dim
        d_h = int(d_in * inter_multi)

        self.d_h = d_h
        self.d_in = d_in
        self.d_out = d_out
        self.w0_w2_low_rank = w0_w2_low_rank
        self.fw_init_gain = fw_init_gain

        # Low Rank parameterization of the fast weights.
        # This is a compromise to keep the number of parameters low when comparing against baselines.
        # Idealy, low-rank parameterization always hurts the performance.
        if self.w0_w2_low_rank > 0:
            self.w0 = LowRankFastWeight(
                self.num_fw_heads,
                d_h,
                d_in,
                self.w0_w2_low_rank,
                init_gain=self.fw_init_gain,
                add_identity=True,
            )
            self.w2 = LowRankFastWeight(
                self.num_fw_heads,
                d_h,
                d_in,
                self.w0_w2_low_rank,
                init_gain=self.fw_init_gain,
                add_identity=True,
            )
        else:
            self.w0 = nn.Parameter(
                torch.randn(self.num_fw_heads, int(d_h), d_in) / math.sqrt(d_in)
            )  # [num_fw_heads, d_h, d_in]
            self.w2 = nn.Parameter(
                torch.randn(self.num_fw_heads, int(d_h), d_in) / math.sqrt(d_in)
            )  # [num_fw_heads, d_h, d_in]
        self.w1 = nn.Parameter(
            torch.randn(self.num_fw_heads, int(d_out), d_h) / math.sqrt(d_h)
        )  # [num_fw_heads, d_out, d_h]

        #### Per-Token LR parameterization.
        self.lr_dim = int(lr_dim * 3 * self.num_fw_heads)
        self.lr_proj = nn.Linear(self.hidden_size, self.lr_dim)
        base_lr = 0.001
        # Lr parameterization and initialization
        if lr_parameterization.lower() == "mamba":
            self.base_lr_inv = inv_softplus(base_lr)
        self.lr_parameterization = lr_parameterization

        #### per-channel scaling and offset for Q, and K.
        self.qk_scale = nn.Parameter(torch.ones(hidden_size, 2))
        self.qk_offset = nn.Parameter(torch.zeros(hidden_size, 2))
        self.learnable_ttt_scale = learnable_ttt_scale
        if self.learnable_ttt_scale:
            # per-head scaling.
            self.ttt_scale_proj = nn.Linear(hidden_size, self.num_fw_heads)

        # ttt output norm per head.
        self.ttt_norm = RMSNorm(self.fw_head_dim, elementwise_affine=True)

        self.use_momentum = use_momentum
        if self.use_momentum:
            self.momentum_proj = nn.Sequential(
                nn.Linear(hidden_size, self.num_fw_heads),
                nn.Sigmoid(),
            )

        self.ttt_loss_type = ttt_loss_type
        self.use_fused_kernel = use_fused_kernel
        self.fp32_states = fp32_states

        assert self.ttt_loss_type in [
            "dot_product"
        ], f"Loss type {self.ttt_loss_type} not supported"

    def reset_parameters(self):
        if self.w0_w2_low_rank > 0:
            self.w0.reset_parameters()
            self.w2.reset_parameters()
        else:
            nn.init.normal_(self.w0, mean=0.0, std=1.0 / math.sqrt(self.fw_head_dim))
            nn.init.normal_(self.w2, mean=0.0, std=1.0 / math.sqrt(self.fw_head_dim))
        nn.init.normal_(self.w1, mean=0.0, std=1.0 / math.sqrt(self.d_h))
        nn.init.ones_(self.qk_scale)
        nn.init.zeros_(self.qk_offset)

    def _rescale_qk(self, q, k):
        """
        Args:
            q: [b, s, d]
            k: [b, s, d]
        Returns:
            q: [b, s, d]
            k: [b, s, d]
        """
        qk_scale = self.qk_scale.view(1, 1, -1, 2)
        qk_offset = self.qk_offset.view(1, 1, -1, 2)
        q = q * qk_scale[:, :, :, 0] + qk_offset[:, :, :, 0]
        k = k * qk_scale[:, :, :, 1] + qk_offset[:, :, :, 1]
        return q, k

    def _init_generation_fast_weight_state(
        self,
        layer_state: dict[str, Any],
        batch_size: int,
    ) -> None:
        if layer_state.get("fastw") is not None:
            return

        if self.w0_w2_low_rank > 0:
            fw_w0 = self.w0().repeat(batch_size, 1, 1)
            fw_w2 = self.w2().repeat(batch_size, 1, 1)
        else:
            fw_w0 = self.w0.repeat(batch_size, 1, 1)
            fw_w2 = self.w2.repeat(batch_size, 1, 1)
        fw_w1 = self.w1.repeat(batch_size, 1, 1)

        if self.fp32_states:
            fw_w0 = fw_w0.to(torch.float32)
            fw_w1 = fw_w1.to(torch.float32)
            fw_w2 = fw_w2.to(torch.float32)

        layer_state["state_dtype"] = fw_w0.dtype
        layer_state["fastw"] = (fw_w0, fw_w1, fw_w2)
        layer_state["masterw"] = (fw_w0, fw_w1, fw_w2)
        layer_state["w_norms"] = (
            fw_w0.norm(dim=2, keepdim=True),
            fw_w1.norm(dim=2, keepdim=True),
            fw_w2.norm(dim=2, keepdim=True),
        )
        if self.use_momentum and layer_state.get("momentum_buf") is None:
            layer_state["momentum_buf"] = (
                torch.zeros_like(fw_w0),
                torch.zeros_like(fw_w1),
                torch.zeros_like(fw_w2),
            )

    @staticmethod
    def _append_generation_tensor(
        layer_state: dict[str, Any],
        key: str,
        value: torch.Tensor | None,
    ) -> None:
        if value is None:
            layer_state[key] = None
        elif layer_state.get(key) is None:
            layer_state[key] = value
        else:
            layer_state[key] = torch.cat((layer_state[key], value), dim=1)

    @staticmethod
    def _pending_generation_len(layer_state: dict[str, Any]) -> int:
        pending_k = layer_state.get("pending_k")
        return 0 if pending_k is None else int(pending_k.size(1))

    def _apply_generation_update(self, layer_state: dict[str, Any]) -> None:
        k = layer_state.get("pending_k")
        if k is None or k.size(1) == 0:
            return

        v = layer_state["pending_v"]
        lr0 = layer_state["pending_lr0"]
        lr1 = layer_state["pending_lr1"]
        lr2 = layer_state["pending_lr2"]
        momentum = layer_state.get("pending_momentum")
        w0, w1, w2 = layer_state["fastw"]
        w0_master, w1_master, w2_master = layer_state.get("masterw", layer_state["fastw"])
        w0_norm, w1_norm, w2_norm = layer_state["w_norms"]

        state_dtype = w0.dtype
        k = k.to(state_dtype)
        v = v.to(state_dtype)
        lr0 = lr0.to(state_dtype)
        lr1 = lr1.to(state_dtype)
        lr2 = lr2.to(state_dtype)

        vi = v.transpose(1, 2)
        gate_before_act = torch.bmm(w0, k.transpose(1, 2))
        hidden_before_mul = torch.bmm(w2, k.transpose(1, 2))
        hidden = F.silu(gate_before_act, inplace=False) * hidden_before_mul
        dhidden = torch.bmm(w1.transpose(1, 2), vi)
        dhidden_before_mul = dhidden * F.silu(gate_before_act, inplace=False)
        dgate = dhidden * hidden_before_mul
        dgate_before_act = silu_backprop(dgate, gate_before_act)

        dw1 = torch.bmm(vi, (hidden.transpose(1, 2) * lr1).type_as(vi))
        dw0 = torch.bmm(dgate_before_act, (k * lr0).type_as(dgate_before_act))
        dw2 = torch.bmm(dhidden_before_mul, (k * lr2).type_as(dhidden_before_mul))

        momentum_buf = layer_state.get("momentum_buf")
        if momentum is not None and momentum_buf is not None:
            m_i = momentum.mean(dim=1, keepdim=True).to(dw0.dtype)
            dw0 = dw0 + momentum_buf[0] * m_i
            dw1 = dw1 + momentum_buf[1] * m_i
            dw2 = dw2 + momentum_buf[2] * m_i
            momentum_buf = (dw0.clone(), dw1.clone(), dw2.clone())

        if self.use_muon:
            dw1 = zeropower_via_newtonschulz5(dw1)
            dw0 = zeropower_via_newtonschulz5(dw0)
            dw2 = zeropower_via_newtonschulz5(dw2)

        if self.ttt_prenorm:
            w0_master = w0_master + dw0
            w1_master = w1_master + dw1
            w2_master = w2_master + dw2
            w0 = w0_master / (w0_master.norm(dim=2, keepdim=True) + 1e-5) * w0_norm
            w1 = w1_master / (w1_master.norm(dim=2, keepdim=True) + 1e-5) * w1_norm
            w2 = w2_master / (w2_master.norm(dim=2, keepdim=True) + 1e-5) * w2_norm
        else:
            w0 = w0 + dw0
            w1 = w1 + dw1
            w2 = w2 + dw2
            w0 = w0 / (w0.norm(dim=2, keepdim=True) + 1e-5) * w0_norm
            w1 = w1 / (w1.norm(dim=2, keepdim=True) + 1e-5) * w1_norm
            w2 = w2 / (w2.norm(dim=2, keepdim=True) + 1e-5) * w2_norm
            w0_master, w1_master, w2_master = w0, w1, w2

        if not self.fp32_states:
            target_dtype = layer_state.get("state_dtype", state_dtype)
            w0 = w0.to(target_dtype)
            w1 = w1.to(target_dtype)
            w2 = w2.to(target_dtype)
            w0_master = w0_master.to(target_dtype)
            w1_master = w1_master.to(target_dtype)
            w2_master = w2_master.to(target_dtype)
            if momentum_buf is not None:
                momentum_buf = tuple(buf.to(target_dtype) for buf in momentum_buf)

        layer_state["fastw"] = (w0, w1, w2)
        layer_state["masterw"] = (w0_master, w1_master, w2_master)
        layer_state["momentum_buf"] = momentum_buf
        layer_state["pending_k"] = None
        layer_state["pending_v"] = None
        layer_state["pending_lr0"] = None
        layer_state["pending_lr1"] = None
        layer_state["pending_lr2"] = None
        layer_state["pending_momentum"] = None

    def _apply_generation_fast_weights(
        self,
        layer_state: dict[str, Any],
        fast_q: torch.Tensor,
        fast_k: torch.Tensor,
        fast_v: torch.Tensor,
        fw_lr1: torch.Tensor,
        fw_lr2: torch.Tensor,
        fw_lr3: torch.Tensor,
        momentum: torch.Tensor | None,
        batch_size: int,
    ) -> torch.Tensor:
        self._init_generation_fast_weight_state(layer_state, batch_size)
        outputs = []
        local_start = 0
        seq_len = fast_q.size(1)

        while local_start < seq_len:
            if self._pending_generation_len(layer_state) >= self.lact_chunk_size:
                self._apply_generation_update(layer_state)

            pending_len = self._pending_generation_len(layer_state)
            take = min(seq_len - local_start, self.lact_chunk_size - pending_len)
            q_slice = fast_q[:, local_start:local_start + take]
            w0, w1, w2 = layer_state["fastw"]
            output_dtype = q_slice.dtype
            qi = q_slice.transpose(1, 2).to(w0.dtype)
            h = torch.bmm(w2, qi)
            gate = F.silu(torch.bmm(w0, qi), inplace=False)
            outputs.append(torch.bmm(w1, gate * h).transpose(1, 2).to(output_dtype))

            self._append_generation_tensor(layer_state, "pending_k", fast_k[:, local_start:local_start + take])
            self._append_generation_tensor(layer_state, "pending_v", fast_v[:, local_start:local_start + take])
            self._append_generation_tensor(layer_state, "pending_lr0", fw_lr1[:, local_start:local_start + take])
            self._append_generation_tensor(layer_state, "pending_lr1", fw_lr2[:, local_start:local_start + take])
            self._append_generation_tensor(layer_state, "pending_lr2", fw_lr3[:, local_start:local_start + take])
            if momentum is not None:
                self._append_generation_tensor(
                    layer_state,
                    "pending_momentum",
                    momentum[:, local_start:local_start + take],
                )

            local_start += take

        return torch.cat(outputs, dim=1)

    def forward(
        self,
        hidden_states: torch.Tensor,  # [b, s, d]
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        generation_cache = past_key_values if is_recurrent_lact_cache(past_key_values) else None
        attention_cache = (
            generation_cache.get("attention_cache") if generation_cache is not None else past_key_values
        )
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        batch_size, q_len, _ = hidden_states.size()

        q, k, v = self.qkv(hidden_states).chunk(3, dim=-1)
        #### compute window attention first, then do ttt. ####

        if self.attn_qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)

        # rescale and reshift the q, k for test-time training layer.
        fast_q, fast_k = self._rescale_qk(q, k)
        fast_v = v

        q = rearrange(q, "... (h d) -> ... h d", d=self.head_dim)
        k = rearrange(k, "... (h d) -> ... h d", d=self.head_dim)
        v = rearrange(v, "... (h d) -> ... h d", d=self.head_dim)

        # WARNING: current implementation ignores cu_seqlens for ttt-layer.
        cu_seqlens = kwargs.get("cu_seqlens", None)

        seqlen_offset, max_seqlen = 0, q_len
        if attention_cache is not None:
            seqlen_offset = attention_cache.get_seq_length(self.layer_idx)
            max_seqlen = q.shape[1] + seqlen_offset

            if attention_mask is not None:
                # to deliminate the offsets of padding tokens
                seqlen_offset = (
                    seqlen_offset + attention_mask.sum(-1) - attention_mask.shape[-1]
                )
                max_seqlen = q.shape[1] + max(seqlen_offset)

        if self.max_position_embeddings is not None:
            max_seqlen = max(max_seqlen, self.max_position_embeddings)
        # [b, s, n_h, d]
        q, k = self.rotary(
            q,
            k,
            seqlen_offset=seqlen_offset,
            max_seqlen=max_seqlen,
            cu_seqlens=cu_seqlens,
        )

        if attention_cache is not None:
            cache_has_content = attention_cache.get_seq_length(self.layer_idx) > 0
            k_cached, v_cached = attention_cache.update(
                attn_state=(k.flatten(-2, -1), v.flatten(-2, -1)),
                layer_idx=self.layer_idx,
                offset=q_len,
                cache_kwargs=dict(window_size=self.window_size),
            )["attn_state"]
            if cache_has_content:
                k, v = k_cached, v_cached
                k = rearrange(k, "... (h d) -> ... h d", d=self.head_dim)
                v = rearrange(v, "... (h d) -> ... h d", d=self.head_dim)

        if flash_attn_func is None:
            raise ImportError(
                "Please install Flash Attention via `pip install flash-attn --no-build-isolation` first"
            )

        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            q, k, v, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                q, k, v, attention_mask, q_len
            )
            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_q, max_seqlen_k = max_seq_lens
            o = flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                causal=True,
                window_size=(
                    (-1, -1) if self.window_size is None else (self.window_size - 1, 0)
                ),
            )
            o = pad_input(o, indices_q, batch_size, q_len)
        elif cu_seqlens is not None:
            o = flash_attn_varlen_func(
                q.squeeze(0),
                k.squeeze(0),
                v.squeeze(0),
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                causal=True,
                window_size=(
                    (-1, -1) if self.window_size is None else (self.window_size - 1, 0)
                ),
            ).unsqueeze(0)
        else:
            o = flash_attn_func(
                q,
                k,
                v,
                causal=True,
                window_size=(
                    (-1, -1) if self.window_size is None else (self.window_size - 1, 0)
                ),
            )
        o = o.reshape(batch_size, q_len, -1)

        ##### TTT starts here.
        # Split heads then merge it to batch dimension
        fast_q = rearrange(fast_q, "b s (n_h d) -> (b n_h) s d", n_h=self.num_fw_heads)
        fast_k = rearrange(fast_k, "b s (n_h d) -> (b n_h) s d", n_h=self.num_fw_heads)
        fast_v = rearrange(fast_v, "b s (n_h d) -> (b n_h) s d", n_h=self.num_fw_heads)

        if self.qkv_silu:
            if self.no_v_silu:
                fast_q = F.silu(fast_q)
                fast_k = F.silu(fast_k)
            else:
                fast_q = F.silu(fast_q)
                fast_k = F.silu(fast_k)
                fast_v = F.silu(fast_v)

        # per head l2 norm for fast_q, fast_k.
        fast_q = l2_norm(fast_q)
        fast_k = l2_norm(fast_k)

        if not self.ttt_nope:
            #### Apply rotary embedding.  Here we use the same rope as the attention layer.
            # I observed that using NoPE for ttt (No positional encoding) here also works.
            fast_q = rearrange(
                fast_q, "(b n_h) s d -> b s (n_h d)", n_h=self.num_fw_heads
            )
            fast_k = rearrange(
                fast_k, "(b n_h) s d -> b s (n_h d)", n_h=self.num_fw_heads
            )

            fast_q = rearrange(fast_q, "b s (n_h d) -> b s n_h d", n_h=self.num_heads)
            fast_k = rearrange(fast_k, "b s (n_h d) -> b s n_h d", n_h=self.num_heads)

            fast_q, fast_k = self.rotary(
                fast_q,
                fast_k,
                seqlen_offset=seqlen_offset,
                max_seqlen=max_seqlen,
                cu_seqlens=cu_seqlens,
            )

            fast_q = rearrange(fast_q, "b s n_h d -> b s (n_h d)", n_h=self.num_heads)
            fast_k = rearrange(fast_k, "b s n_h d -> b s (n_h d)", n_h=self.num_heads)

            fast_q = rearrange(
                fast_q, "b s (n_h d) -> (b n_h) s d", n_h=self.num_fw_heads
            )
            fast_k = rearrange(
                fast_k, "b s (n_h d) -> (b n_h) s d", n_h=self.num_fw_heads
            )
            #### RoPE done. ####

        if self.w0_w2_low_rank > 0:
            fw_w0 = self.w0().repeat(batch_size, 1, 1)
            fw_w2 = self.w2().repeat(batch_size, 1, 1)
        else:
            fw_w0 = self.w0.repeat(
                batch_size, 1, 1
            )  # [nh, d_h, d_in] -> [b*nh, d_h, d_in]
            fw_w2 = self.w2.repeat(
                batch_size, 1, 1
            )  # [nh, d_h, d_in] -> [b*nh, d_h, d_in]

        fw_w1 = self.w1.repeat(
            batch_size, 1, 1
        )  # [nh, d_out, d_h] -> [b*nh, d_out, d_h]

        lr = self.lr_proj(hidden_states)  # [b, s, num_heads * lr_dim_per_head]
        if self.lr_parameterization == "mamba":
            lr = torch.nn.functional.softplus(lr.float() + self.base_lr_inv)
        else:
            raise NotImplementedError(
                f"LR parameterization {self.lr_parameterization} not implemented"
            )
        fw_lr = rearrange(
            lr, "b s (n_h lr_dim) -> (b n_h) s lr_dim", n_h=self.num_fw_heads
        )
        fw_lr1, fw_lr2, fw_lr3 = fw_lr.chunk(3, dim=-1)

        if self.use_momentum:
            momentum = self.momentum_proj(hidden_states).float()  # [b, s, nh]
            momentum = rearrange(
                momentum, "b s (n_h d) -> (b n_h) s d", n_h=self.num_fw_heads
            )
        else:
            momentum = None

        if self.fp32_states:
            # here we cast the fast weights to fp32, but all matmuls are still in bf16
            # only fast weight updates are in fp32.  This is similar to bf16 training of slow weights.
            fw_w0 = fw_w0.to(torch.float32)
            fw_w1 = fw_w1.to(torch.float32)
            fw_w2 = fw_w2.to(torch.float32)

        # [b * nh, s, d_ttt_head]
        if generation_cache is not None:
            fw_x = self._apply_generation_fast_weights(
                generation_cache["layers"][self.layer_idx],
                fast_q,
                fast_k,
                fast_v,
                fw_lr1,
                fw_lr2,
                fw_lr3,
                momentum,
                batch_size,
            )
        elif self.ttt_prenorm:
            # pre-norm version of ttt.   state = state + f(norm(state))
            if self.use_fused_kernel:
                fw_x = prenorm_block_causal_lact_swiglu_fused_kernel_triton(
                    fw_w0,
                    fw_w1,
                    fw_w2,
                    fast_q,
                    fast_k,
                    fast_v,
                    fw_lr1,
                    fw_lr2,
                    fw_lr3,
                    chunk_size=self.lact_chunk_size,
                    use_muon=self.use_muon,
                    momentum=momentum,
                )
            else:
                fw_x = prenorm_block_causal_lact_swiglu(
                    fw_w0,
                    fw_w1,
                    fw_w2,
                    fast_q,
                    fast_k,
                    fast_v,
                    fw_lr1,
                    fw_lr2,
                    fw_lr3,
                    chunk_size=self.lact_chunk_size,
                    use_muon=self.use_muon,
                    momentum=momentum,
                )
        else:
            # post-norm version of ttt.   state = norm(state + f(state))
            if self.use_fused_kernel:
                fw_x = postnorm_block_causal_lact_swiglu_fused_kernel_triton(
                    fw_w0,
                    fw_w1,
                    fw_w2,
                    fast_q,
                    fast_k,
                    fast_v,
                    fw_lr1,
                    fw_lr2,
                    fw_lr3,
                    chunk_size=self.lact_chunk_size,
                    use_muon=self.use_muon,
                    momentum=momentum,
                )
            else:
                fw_x = block_causal_lact_swiglu(
                    fw_w0,
                    fw_w1,
                    fw_w2,
                    fast_q,
                    fast_k,
                    fast_v,
                    fw_lr1,
                    fw_lr2,
                    fw_lr3,
                    chunk_size=self.lact_chunk_size,
                    use_muon=self.use_muon,
                    momentum=momentum,
                )

        # per-head output norm for ttt layer.
        ttt_x_normed = self.ttt_norm(fw_x)
        if self.learnable_ttt_scale:
            ttt_scale = F.silu(self.ttt_scale_proj(hidden_states), inplace=False)
            ttt_scale = rearrange(
                ttt_scale, "b s (n_h d) -> (b n_h) s d", n_h=self.num_fw_heads
            )
            ttt_x_normed = ttt_x_normed * ttt_scale

        ttt_x_normed = rearrange(
            ttt_x_normed, "(b n_h) s d -> b s (n_h d)", n_h=self.num_fw_heads
        )

        o = o + ttt_x_normed
        o = self.o_proj(o)

        if not output_attentions:
            attentions = None

        return o, attentions, generation_cache if generation_cache is not None else attention_cache

    def _upad_input(self, q, k, v, attention_mask, q_len):
        batch_size, seq_len, num_key_value_heads, head_dim = k.shape
        cache_mask = attention_mask[:, -seq_len:]
        seqlens = cache_mask.sum(-1, dtype=torch.int32)
        indices_k = torch.nonzero(cache_mask.flatten(), as_tuple=False).flatten()
        max_seqlen_k = seqlens.max().item()
        cu_seqlens_k = F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))

        k = index_first_axis(
            k.reshape(batch_size * seq_len, num_key_value_heads, head_dim), indices_k
        )
        v = index_first_axis(
            v.reshape(batch_size * seq_len, num_key_value_heads, head_dim), indices_k
        )
        if q_len == seq_len:
            q = index_first_axis(
                q.reshape(batch_size * seq_len, self.num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_q = max_seqlen_k
            indices_q = indices_k
        elif q_len == 1:
            max_seqlen_q = 1
            # There is a memcpy here, that is very bad.
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=q.device
            )
            indices_q = cu_seqlens_q[:-1]
            q = q.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -q_len:]
            q, indices_q, cu_seqlens_q, max_seqlen_q = unpad_input(q, attention_mask)

        return (
            q,
            k,
            v,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_q, max_seqlen_k),
        )
