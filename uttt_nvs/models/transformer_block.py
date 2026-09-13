import math
from typing import Tuple, Optional

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from flash_attn import flash_attn_func

from . import debug_utils
from .debug_utils import sketch_tensor


class CacheAppend(torch.autograd.Function):
    """
    storage: [b_storage, L_storage, heads, head_dim]
    active_cache: [b, L_active, heads, head_dim]
    x: [b, L_new, heads, head_dim]
    """

    @staticmethod
    def forward(ctx, storage, active_cache, x, start):
        bs = active_cache.size(0)
        end = start + x.size(1)
        assert end <= storage.size(1), "End index exceeds storage size"
        storage.data[:bs, start:end] = x
        ctx.bs = bs
        ctx.start = start
        ctx.end = end
        return storage[:bs, :end]

    @staticmethod
    def backward(ctx, grad_output):
        bs = ctx.bs
        start = ctx.start
        end = ctx.end
        return None, grad_output[:bs, :start], grad_output[:bs, start:end], None, None


cache_append = CacheAppend.apply


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim,
        head_dim,
        bias=False,
        transformer_version=None,
        qkv_transform=None,
        kv_transform=None,
    ):
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim

        self.transformer_version = transformer_version
        if self.transformer_version == "v0":
            pass
        elif self.transformer_version == "v1":
            self.qkv_transform = qkv_transform
        elif self.transformer_version == "v2":
            self.qkv_transform = nn.Linear(dim, dim, bias=False)
        elif self.transformer_version == "v3":
            self.q_transform = nn.Linear(dim, dim, bias=False)
            self.kv_transform = kv_transform
        elif self.transformer_version == "v4":
            self.q_transform = nn.Linear(dim, dim, bias=False)
            self.kv_transform = nn.Linear(dim, dim, bias=False)
        else:
            raise ValueError(f"Invalid transformer version: {self.transformer_version}")

        self.to_q = nn.Linear(dim, dim, bias=bias)
        self.to_kv = nn.Linear(dim, 2 * dim, bias=bias)
        self.to_out = nn.Linear(dim, dim, bias=bias)

        heads = dim // head_dim
        self.k_storage = torch.empty(0, 0, heads, head_dim)
        self.v_storage = torch.empty(0, 0, heads, head_dim)

    def _ensure_storage(self, batch_size, needed_len, device, dtype):
        b, l, h, d = self.k_storage.shape
        batch_size = max(batch_size, b)
        needed_len = max(needed_len, l)
        if b < batch_size or l < needed_len or self.k_storage.device != device or self.k_storage.dtype != dtype:
            self.k_storage = self.k_storage.new_empty(batch_size, needed_len, h, d, device=device, dtype=dtype)
            self.v_storage = self.v_storage.new_empty(batch_size, needed_len, h, d, device=device, dtype=dtype)

    def cache_init(self, batch_size, max_len, device, dtype):
        self._ensure_storage(batch_size, max_len, device=device, dtype=dtype)
        k_active = self.k_storage[:batch_size, :0]      # Return empty KV
        v_active = self.v_storage[:batch_size, :0]
        return k_active, v_active

    def forward(self, x, kv_cache=None):

        if self.transformer_version == "v0":
            x_q = x
            x_kv = x
        elif self.transformer_version in ["v1", "v2"]:
            x = self.qkv_transform(x)
            x_q = x
            x_kv = x
        elif self.transformer_version in ["v3", "v4"]:
            x_q = self.q_transform(x)
            x_kv = self.kv_transform(x)
        else:
            raise ValueError(f"Invalid transformer version: {self.transformer_version}")

        # [b, l, dim] -> [b, l, heads, head_dim]
        q = self.to_q(x_q)
        heads = self.dim // self.head_dim
        q = q.unflatten(-1, (heads, self.head_dim))

        # [b, l, 2 * dim] -> [b, l, 2, heads, head_dim] -> [2, b, l, heads, head_dim]
        kv = self.to_kv(x_kv)
        k, v = kv.unflatten(-1, (2, heads, self.head_dim)).movedim(2, 0).contiguous()

        k_cache, v_cache = kv_cache
        start = k_cache.size(1)
        full_k = cache_append(self.k_storage, k_cache, k, start)
        full_v = cache_append(self.v_storage, v_cache, v, start)

        x = flash_attn_func(q, full_k, full_v)

        # [b, l, heads, head_dim] -> [b, l, heads * head_dim]
        x = x.flatten(-2)
        x = self.to_out(x)

        return x
    
    def extra_repr(self) -> str:
        return f"transformer_version: {self.transformer_version}"


class MLP(nn.Module):
    def __init__(
        self,
        dim,
        inter_multi,
        bias=False
    ):
        super().__init__()
        inter_dim = int(dim * inter_multi)

        self.gate = nn.Linear(dim, inter_dim, bias=bias)
        self.up = nn.Linear(dim, inter_dim, bias=bias)
        self.down= nn.Linear(inter_dim, dim, bias=bias)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))
    


class MemoryBlock(nn.Module):
    def __init__(
        self,
        dim,
        attn_head_dim=64,
        bias=False,
        vi_rms_norm=False,
        inter_multi=2,
        transformer_version=None,
    ):
        super().__init__()
        self.dim = dim
        self.head_dim = attn_head_dim
        self.vi_rms_norm = vi_rms_norm
        if transformer_version is not None:
            self.transformer_version = transformer_version
        else: raise ValueError(f"Invalid transformer version: {transformer_version}")

        # Attention
        self.ln_attn = nn.RMSNorm(dim, eps=1e-5)

        if self.transformer_version == "v0":
            self.self_attention = SelfAttention(dim, attn_head_dim, bias=bias, transformer_version=transformer_version)
        elif self.transformer_version == "v1":
            self.qkv_transform = nn.Linear(dim, dim, bias=False)
            self.self_attention = SelfAttention(dim, attn_head_dim, bias=bias, transformer_version=transformer_version, qkv_transform=self.qkv_transform)
        elif self.transformer_version == "v2":
            self.qkv_transform = nn.Linear(dim, dim, bias=False)
            self.self_attention = SelfAttention(dim, attn_head_dim, bias=bias, transformer_version=transformer_version)
        elif self.transformer_version == "v3":
            self.kv_transform = nn.Linear(dim, dim, bias=False)
            self.self_attention = SelfAttention(dim, attn_head_dim, bias=bias, transformer_version=transformer_version, kv_transform=self.kv_transform)
        elif self.transformer_version == "v4":
            self.kv_transform = nn.Linear(dim, dim, bias=False)
            self.self_attention = SelfAttention(dim, attn_head_dim, bias=bias, transformer_version=transformer_version)
        else:
            raise ValueError(f"Invalid transformer version: {transformer_version}")

        # Slow MLP (Knowledge)
        self.ln_mlp = nn.RMSNorm(dim, eps=1e-5)
        self.mlp = MLP(dim, inter_multi, bias=bias)

    def _forward(self, x, kv_cache):

        # Window Attention
        skip = x
        x = self.ln_attn(x)
        xi_memory_in = x
        x = self.self_attention(x, kv_cache)
        x = skip + x

        # Slow weight apply
        skip = x
        x = self.ln_mlp(x)
        x = skip + self.mlp(x)

        return x, xi_memory_in

    # @torch.compile
    def forward(self, *args):
        return checkpoint(self._forward, *args, preserve_rng_state=False, use_reentrant=False)
    
    def _update_kv_cache(self, pre_vi, kv_cache, to_kv_fn=None):
        if self.vi_rms_norm:  # only enabled when skipping layers
            pre_vi = F.rms_norm(pre_vi, normalized_shape=(self.dim,), eps=1e-5)

        if self.transformer_version == "v0":
            vi = pre_vi
        elif self.transformer_version == "v1":
            vi = self.qkv_transform(pre_vi)
        elif self.transformer_version == "v2":
            vi = self.qkv_transform(pre_vi)
        elif self.transformer_version == "v3":
            vi = self.kv_transform(pre_vi)
        elif self.transformer_version == "v4":
            vi = self.kv_transform(pre_vi)
        else:
            raise ValueError(f"Invalid transformer version: {self.transformer_version}")

        # [b, l, 2 * heads * head_dim] -> [b, l, 2, heads, head_dim] -> [2, b, l, heads, head_dim]
        if to_kv_fn is not None:
            kv = to_kv_fn(vi)
        else:
            kv = self.self_attention.to_kv(vi)
        k, v = kv.unflatten(-1, (2, -1, self.head_dim)).movedim(2, 0).contiguous()

        k_cache, v_cache = kv_cache
        start = k_cache.size(1)
        new_key_cache = cache_append(self.self_attention.k_storage, k_cache, k, start)
        new_value_cache = cache_append(self.self_attention.v_storage, v_cache, v, start)

        return (new_key_cache, new_value_cache)

    # @torch.compile
    def update_kv_cache(self, pre_vi, kv_cache, to_kv_fn=None):
        return self._update_kv_cache(pre_vi, kv_cache, to_kv_fn)


class MemoryModel(nn.Module):
    def __init__(self, layers, dim, v_gap=None, pre_vi_source="xi_memory_in",
                 use_separate_last_to_kv=None, to_kv_mode=None, **kwargs):
        super().__init__()
        self.v_gap = v_gap
        self.pre_vi_source = pre_vi_source

        # Backwards compatibility: convert the old bool argument to the new str argument
        if to_kv_mode is None:
            if use_separate_last_to_kv is None or use_separate_last_to_kv == False:
                to_kv_mode = "shared_last"
            elif use_separate_last_to_kv == True:
                to_kv_mode = "separate"
        self.to_kv_mode = to_kv_mode

        if to_kv_mode == "separate":
            self.last_to_kv = nn.Linear(dim, 2 * dim, bias=False)

        self.blocks = []
        self.v_layer_idxs = []
        for i in range(layers):
            if v_gap is not None:
                self.v_layer_idxs.append(max(min(i + v_gap, layers - 1), 0))
            self.blocks.append(MemoryBlock(dim=dim, **kwargs))
        self.blocks = nn.ModuleList(self.blocks)

    def forward(self, x, info_dict):
        bs = x.size(0)
        max_len = x.size(1)
        caches = []
        for block in self.blocks:
            k_cache, v_cache = block.self_attention.cache_init(bs, max_len, device=x.device, dtype=torch.bfloat16)
            caches.append({"kv_cache": (k_cache, v_cache)})

        outputs = []
        infos = []
        for opidx, (start, end, update, _, _) in enumerate(info_dict["ttt_config"]):
            op_info = []

            xi = x[:, start:end, :]

            # Apply all blocks to xi
            for idx, (block, cache) in enumerate(zip(self.blocks, caches)):
                xi, xi_memory_in = block.forward(xi, cache["kv_cache"])
                cache.update({"xi_out": xi, "xi_memory_in": xi_memory_in})

            if update:
                for i, cache in enumerate(caches):
                    if self.v_gap is not None:
                        cache["pre_vi"] = caches[self.v_layer_idxs[i]][self.pre_vi_source]
                    else:
                        cache["pre_vi"] = cache["xi_memory_in"]

                # Determine to_kv function based on mode
                if self.to_kv_mode == "separate":
                    to_kv_fn = self.last_to_kv
                elif self.to_kv_mode == "shared_last":
                    to_kv_fn = self.blocks[-1].self_attention.to_kv
                elif self.to_kv_mode == "per_layer":
                    to_kv_fn = None  # pass None so each layer uses its own to_kv
                else:
                    raise ValueError(f"Invalid to_kv_mode: {self.to_kv_mode}")

                # Update fast weights for all blocks
                for i, (block, cache) in enumerate(zip(self.blocks, caches)):
                    kv_cache = block.update_kv_cache(cache["pre_vi"], cache["kv_cache"], to_kv_fn=to_kv_fn)
                    cache.update({"kv_cache": kv_cache})
            
            outputs.append(caches[-1]["xi_out"])
            infos.append(op_info)

        outputs = torch.cat(outputs, dim=1)
        return outputs, infos
    
    def extra_repr(self) -> str:
        lines = [f"pre_vi_source: {self.pre_vi_source}", f"to_kv_mode: {self.to_kv_mode}"]
        lines.extend(f"vmap: {i} <- {v_layer_idx}" for i, v_layer_idx in enumerate(self.v_layer_idxs))
        return "\n".join(lines)
        
