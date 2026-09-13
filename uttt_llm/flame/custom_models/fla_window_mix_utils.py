from __future__ import annotations

import warnings

import torch
import torch.nn as nn
from einops import rearrange

try:
    from torch.distributed._tensor.placement_types import Replicate
    from torch.distributed.tensor import DTensor
except ImportError:
    DTensor = None
    Replicate = None

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
except ImportError:
    warnings.warn(
        "Flash Attention is not installed. Please install it via `pip install flash-attn --no-build-isolation`",
        category=ImportWarning,
    )
    flash_attn_func = None
    flash_attn_varlen_func = None


def normalize_cu_seqlens(cu_seqlens: torch.Tensor | None) -> torch.Tensor | None:
    if cu_seqlens is not None and cu_seqlens.ndim == 2:
        if cu_seqlens.shape[0] != 1:
            raise ValueError(f"Expected cu_seqlens shape [1, n] or [n], got {tuple(cu_seqlens.shape)}")
        cu_seqlens = cu_seqlens.squeeze(0)
    return cu_seqlens


def max_seqlen_from_cu_seqlens(cu_seqlens: torch.Tensor | None, fallback: int) -> int:
    if cu_seqlens is None:
        return fallback
    return int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())


def cast_floating_point_parameters(module: nn.Module, dtype: torch.dtype) -> None:
    for child in module.modules():
        for name, parameter in list(child.named_parameters(recurse=False)):
            if parameter is None or not parameter.is_floating_point() or parameter.dtype == dtype:
                continue
            replacement = nn.Parameter(parameter.to(dtype=dtype), requires_grad=parameter.requires_grad)
            replacement.__dict__.update(getattr(parameter, "__dict__", {}))
            child._parameters[name] = replacement


def tensor_to_parameter_layout(tensor: torch.Tensor, parameter: torch.Tensor, *, run_check: bool = False) -> torch.Tensor:
    tensor = tensor.to(dtype=parameter.dtype)
    if DTensor is None or not isinstance(parameter, DTensor):
        return tensor.to(device=parameter.device)
    if isinstance(tensor, DTensor):
        return tensor.redistribute(device_mesh=parameter.device_mesh, placements=parameter.placements, async_op=True)
    placements = [Replicate()] * len(parameter.placements)
    dtensor = DTensor.from_local(tensor, device_mesh=parameter.device_mesh, placements=placements, run_check=run_check)
    return dtensor.redistribute(device_mesh=parameter.device_mesh, placements=parameter.placements, async_op=True)


def copy_tensor_to_parameter(parameter: torch.Tensor, tensor: torch.Tensor, *, run_check: bool = False) -> None:
    parameter.copy_(tensor_to_parameter_layout(tensor, parameter, run_check=run_check))


def sliding_window_attention_from_projections(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    qk_scale: torch.Tensor,
    qk_offset: torch.Tensor,
    attn_head_dim: int,
    rotary,
    window_size: int | None,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: int | None = None,
) -> torch.Tensor:
    if flash_attn_func is None or flash_attn_varlen_func is None:
        raise ImportError("Please install Flash Attention via `pip install flash-attn --no-build-isolation` first")

    batch_size, q_len, key_dim = q.shape
    value_dim = v.shape[-1]
    if k.shape != q.shape:
        raise ValueError(f"q and k must have the same shape, got {tuple(q.shape)} and {tuple(k.shape)}")
    if key_dim % attn_head_dim != 0:
        raise ValueError(f"key_dim={key_dim} must be divisible by attn_head_dim={attn_head_dim}")
    if value_dim % attn_head_dim != 0:
        raise ValueError(f"value_dim={value_dim} must be divisible by attn_head_dim={attn_head_dim}")

    q_heads = key_dim // attn_head_dim
    v_heads = value_dim // attn_head_dim
    if v_heads % q_heads != 0:
        raise ValueError(f"value attention heads ({v_heads}) must be a multiple of q/k heads ({q_heads})")

    scale = qk_scale.to(dtype=q.dtype).view(1, 1, key_dim, 2)
    offset = qk_offset.to(dtype=q.dtype).view(1, 1, key_dim, 2)
    q = q * scale[..., 0] + offset[..., 0]
    k = k * scale[..., 1] + offset[..., 1]

    q = rearrange(q, "... (h d) -> ... h d", d=attn_head_dim)
    k = rearrange(k, "... (h d) -> ... h d", d=attn_head_dim)
    head_repeat = v_heads // q_heads
    if head_repeat != 1:
        q = torch.repeat_interleave(q, head_repeat, dim=-2)
        k = torch.repeat_interleave(k, head_repeat, dim=-2)
    v = rearrange(v, "... (h d) -> ... h d", d=attn_head_dim)
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()

    cu_seqlens = normalize_cu_seqlens(cu_seqlens)
    max_seqlen = max_seqlen if max_seqlen is not None else max_seqlen_from_cu_seqlens(cu_seqlens, q_len)
    rotary_max_seqlen = q_len if cu_seqlens is not None else max_seqlen
    q, k = rotary(q, k, seqlen_offset=0, max_seqlen=rotary_max_seqlen, cu_seqlens=cu_seqlens)
    q, k = q.contiguous(), k.contiguous()

    window = (-1, -1) if window_size is None else (window_size - 1, 0)
    if cu_seqlens is not None:
        if batch_size != 1:
            raise ValueError("Packed varlen flash attention expects batch size 1")
        out = flash_attn_varlen_func(
            q.squeeze(0),
            k.squeeze(0),
            v.squeeze(0),
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            causal=True,
            window_size=window,
        ).unsqueeze(0)
    else:
        out = flash_attn_func(q, k, v, causal=True, window_size=window)
    return out.contiguous().reshape(batch_size, q_len, value_dim)
