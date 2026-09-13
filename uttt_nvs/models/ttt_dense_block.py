import math

import torch 
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from einops import rearrange

from . import debug_utils
from .debug_utils import sketch_tensor


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim,
        head_dim,
    ):
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim

        self.to_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        """
        x: (b, l, D)
        """
        # token split, multi-head attention, token cat
        q, k, v = rearrange(
            self.to_qkv(x), 
            "b l (qkv h d) -> qkv b h l d", 
            qkv=3, d=self.head_dim
        )

        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b h l d -> b l (h d)")
        x = self.to_out(x)

        return x


class MLP(nn.Module):
    def __init__(
        self,
        dim,
        inter_multi,
    ):
        super().__init__()
        inter_dim = int(dim * inter_multi)

        self.gate = nn.Linear(dim, inter_dim, bias=False)
        self.up = nn.Linear(dim, inter_dim, bias=False)
        self.down= nn.Linear(inter_dim, dim, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))
    

def silu_backprop(dy: torch.Tensor, x: torch.Tensor):
    sigma = torch.sigmoid(x)
    dx = dy * sigma * (1 + x * (1 - sigma))
    return dx


def zeropower_via_newtonschulz5(G, steps):
    """
    Args:
        G: [b, d, d]
        steps: int
    Returns:
        X: [b, d, d]
    """
    if steps == 0:
        return G
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(1) > G.size(2):
        X = X.transpose(1, 2)
    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(1, 2), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.transpose(1, 2)
        B = (
            b * A + c * A @ A
        )  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(1) > G.size(2):
        X = X.transpose(1, 2)
    return X


class FastWeightMLPSubNN(nn.Module):
    """
    Self-attention layer
    """
    def __init__(
        self,
        dim,
        fw_head_dim,
        inter_multi,
        use_muon=True,
        l2_norm=False,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_muon = use_muon
        self.l2_norm = l2_norm
        if fw_head_dim is None:
            fw_head_dim = dim
        self.head_dim = fw_head_dim
        self.num_heads = dim // fw_head_dim

        # Input projection
        self.input_proj = nn.Sequential(nn.Linear(dim, dim, bias=False), nn.SiLU())
        if not self.l2_norm:
            self.input_rms_norm = nn.RMSNorm(
                self.head_dim, eps=1e-5, elementwise_affine=False
            )

        # Separate K input projection (share_qk=False by default)
        self.k_input_proj = nn.Sequential(nn.Linear(dim, dim, bias=False), nn.SiLU())
        
        # Fast weights
        inter_dim = int(self.head_dim * inter_multi)
        self.inter_dim = inter_dim
        self.w0 = nn.Parameter(torch.randn(self.num_heads, self.head_dim, inter_dim) / math.sqrt(self.head_dim))
        self.w1 = nn.Parameter(torch.randn(self.num_heads, inter_dim, self.head_dim) / math.sqrt(inter_dim))
        self.w2 = nn.Parameter(torch.randn(self.num_heads, self.head_dim, inter_dim) / math.sqrt(self.head_dim))

        # Output projection
        self.output_rms_norm = nn.RMSNorm(
            self.head_dim, eps=1e-5, elementwise_affine=False
        )
        self.output_proj = nn.Linear(dim, dim, bias=False)


    def init_fast_weights(self, batch_size=1):
        master_weight = (
            self.w0.unsqueeze(0).repeat(batch_size, 1, 1, 1),
            self.w1.unsqueeze(0).repeat(batch_size, 1, 1, 1),
            self.w2.unsqueeze(0).repeat(batch_size, 1, 1, 1),
        )
        weight = tuple(
            F.normalize(w, dim=2, eps=1e-5).to(torch.bfloat16)
            for w in master_weight
        )
        return weight, master_weight

    def forward(self, input, fastw):
        """
        x: [b, l, d]
        fastw: BF16([b, h, hd, dh], [b, h, dh, hd], [b, h, hd, dh])
        """
        w0, w1, w2 = fastw

        x = self.input_proj(input)
        x = rearrange(x, "b l (h d) -> b h l d", h=self.num_heads)
        if self.l2_norm:
            x = F.normalize(x, dim=-1, eps=1e-5).to(x.dtype)
        else:
            x = self.input_rms_norm(x)
        gate = F.silu(x @ w0)
        up = x @ w2
        hidden = gate * up
        x = out = hidden @ w1
        x = self.output_rms_norm(x)
        x = rearrange(x, "b h l d -> b l (h d)")

        x = self.output_proj(x)

        if debug_utils.log_status:
            info = {
                "w0": sketch_tensor(w0),
                "w1": sketch_tensor(w1),
                "w2": sketch_tensor(w2),
                "gate": sketch_tensor(gate),
                "up": sketch_tensor(up),
                "hidden": sketch_tensor(hidden),
                "out": sketch_tensor(out),
                "residual": sketch_tensor(x),
            }
        else:
            info = {}

        return x, info
    
    def update(self, input, grad, lr, fastw, masterw):
        """
        input: [b, l, d]
        grad: [b, l, d]
        lr: [b, l, 3 * h]
        fastw: BF16([b, h, hd, dh], [b, h, dh, hd], [b, h, hd, dh])
        masterw: FP32([b, h, hd, dh], [b, h, dh, hd], [b, h, hd, dh])
        """
        w0_master, w1_master, w2_master = masterw
        w0, w1, w2 = fastw
        lr = rearrange(lr, "b l (h d) -> b h l d", h=self.num_heads, d=3)
        lr0, lr1, lr2 = lr.chunk(3, dim=3)

        L = input.shape[1]

        # Input projection
        x = self.k_input_proj(input)
        x = rearrange(x, "b l (h d) -> b h l d", h=self.num_heads)
        if self.l2_norm:
            x = F.normalize(x, dim=-1, eps=1e-5).to(x.dtype)
        else:
            x = self.input_rms_norm(x)

        # Forward
        gate_before_act = x @ w0
        hidden_before_mul = x @ w2
        hidden = F.silu(gate_before_act) * hidden_before_mul

        # Grad div by L
        grad = grad.float() / L
        grad = rearrange(grad, "b l (h d) -> b h l d", h=self.num_heads)

        # Backward grad
        dhidden = grad @ w1.transpose(-1, -2)
        dhidden_before_mul = dhidden * F.silu(gate_before_act)
        dgate = dhidden * hidden_before_mul
        dgate_before_act = silu_backprop(dgate, gate_before_act)

        # Weight grad
        w1_grad_raw = (hidden * lr1).transpose(-1, -2) @ grad
        w0_grad_raw = (x * lr0).transpose(-1, -2) @ dgate_before_act
        w2_grad_raw = (x * lr2).transpose(-1, -2) @ dhidden_before_mul

        # Muon
        if self.use_muon:
            b, h, _, _ = w0_grad_raw.shape
            w0_grad = zeropower_via_newtonschulz5(w0_grad_raw.flatten(0, 1), 5).view(b, h, self.head_dim, self.inter_dim)
            w1_grad = zeropower_via_newtonschulz5(w1_grad_raw.flatten(0, 1), 5).view(b, h, self.inter_dim, self.head_dim)
            w2_grad = zeropower_via_newtonschulz5(w2_grad_raw.flatten(0, 1), 5).view(b, h, self.head_dim, self.inter_dim)
        else:
            w0_grad, w1_grad, w2_grad = w0_grad_raw, w1_grad_raw, w2_grad_raw

        if debug_utils.log_status:
            info = {
                "update_gate": sketch_tensor(F.silu(gate_before_act)),
                "update_up": sketch_tensor(hidden_before_mul),
                "update_hidden": sketch_tensor(hidden),
                "update_dhidden": sketch_tensor(dhidden),
                "update_w0_grad": sketch_tensor(w0_grad_raw),
                "update_w1_grad": sketch_tensor(w1_grad_raw),
                "update_w2_grad": sketch_tensor(w2_grad_raw),
                "update_w0_master": sketch_tensor(w0_master),
                "update_w1_master": sketch_tensor(w1_master),
                "update_w2_master": sketch_tensor(w2_master),
            }
        else:
            info = {}

        # Master weight update
        w0_master = w0_master + w0_grad
        w1_master = w1_master + w1_grad
        w2_master = w2_master + w2_grad
        masterw = (w0_master, w1_master, w2_master)

        # Normalize master weight
        w0 = F.normalize(w0_master, dim=2, eps=1e-5).to(torch.bfloat16)
        w1 = F.normalize(w1_master, dim=2, eps=1e-5).to(torch.bfloat16)
        w2 = F.normalize(w2_master, dim=2, eps=1e-5).to(torch.bfloat16)
        weight = (w0, w1, w2)

        return weight, masterw, info
    
    def extra_repr(self) -> str:
        return (
            f"fw_head_dim: {self.head_dim}, num_heads: {self.num_heads}, inter_dim: {self.inter_dim}, w0: {self.w0.shape}, w1: {self.w1.shape}, w2: {self.w2.shape}"
            f"use_muon: {self.use_muon}, l2_norm: {self.l2_norm}"
        )


def inv_softplus(x):
    y = x + math.log(-math.expm1(-x))
    return y


class MemoryBlock(nn.Module):
    def __init__(
        self,
        dim,
        fw_head_dim=None,
        attn_head_dim=64,
        inter_multi=2,
        fw_inter_multi=2,
        base_lr=0.01,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim

        # Attention
        self.ln_attn = nn.RMSNorm(dim, eps=1e-5)
        self.self_attention = SelfAttention(dim, attn_head_dim)

        # Fast MLP (Memory)
        self.ln_memory = nn.RMSNorm(dim, eps=1e-5)
        self.memory = FastWeightMLPSubNN(dim, fw_head_dim, fw_inter_multi, **kwargs)

        # FW supervisions
        self.to_v = nn.Sequential(nn.Linear(dim, dim, bias=False), nn.SiLU())
        self.to_lr = nn.Linear(dim, 3 * self.memory.num_heads, bias=False)
        self.base_lr_inv = inv_softplus(base_lr)

        # Slow MLP (Knowledge)
        self.ln_mlp = nn.RMSNorm(dim, eps=1e-5)
        self.mlp = MLP(dim, inter_multi)
    
    def init_fast_weight(self, batch_size=1):
        fastw, masterw = self.memory.init_fast_weights(batch_size=batch_size)
        return fastw, masterw

    def _forward(self, x, fastw):

        # Window Attention
        skip = x
        x = self.ln_attn(x)
        x = skip + self.self_attention(x)

        # Memory
        skip = x
        x_memory_in = x = self.ln_memory(x)
        x, info = self.memory(x, fastw)
        x = skip + x

        # Slow weight apply
        skip = x
        x = self.ln_mlp(x)
        x = skip + self.mlp(x)

        return x, x_memory_in, info
    
    @torch.compile
    def forward(self, *args):
        # if debug_utils.log_status: pass
        return checkpoint(self._forward, *args, preserve_rng_state=False, use_reentrant=False)
    
    def _update_fast_weight(
        self, memory_in, pre_vi, fastw, masterw, update_keep_mask=None
    ):
        # rms_norm
        pre_vi = F.rms_norm(pre_vi, normalized_shape=(self.dim,), eps=1e-5)

        # V and LR prediction
        vi = self.to_v(pre_vi)
        with torch.autocast(device_type="cuda", enabled=False):
            lri = self.to_lr(pre_vi.float())  # [b, l, 3 * h]
            lri = torch.nn.functional.softplus(lri + self.base_lr_inv)
            if update_keep_mask is not None:
                # Broadcast one patch decision to all heads and lr0/lr1/lr2.
                lri = lri * update_keep_mask[..., None].to(dtype=lri.dtype)

        fastw, masterw, info = self.memory.update(memory_in, vi, lri, fastw, masterw)
        return fastw, masterw, info
    
    @torch.compile
    def update_fast_weight(self, *args):
        # if debug_utils.log_status: pass
        return checkpoint(
            self._update_fast_weight, 
            *args, 
            preserve_rng_state=False, use_reentrant=False
        )


class MemoryModel(nn.Module):
    def __init__(self, layers, dim, v_gap=None, **kwargs):
        super().__init__()
        self.v_gap = v_gap
        # Keep runtime-only instrumentation outside the module/state_dict tree.
        object.__setattr__(self, "_update_mask_callback", None)

        self.blocks = []
        self.v_layer_idxs = []
        for i in range(layers):
            if v_gap is not None:
                self.v_layer_idxs.append(max(min(i + v_gap, layers - 1), 0))
            self.blocks.append(MemoryBlock(dim=dim, **kwargs))
        self.blocks = nn.ModuleList(self.blocks)

    def set_update_mask_callback(self, callback):
        """Set an optional per-source-view callback returning a [B, L] keep mask."""
        object.__setattr__(self, "_update_mask_callback", callback)

    def _resolve_update_keep_mask(self, source_view, start, end, x):
        callback = self._update_mask_callback
        if callback is None:
            return None
        mask = callback(
            source_view=source_view,
            start=start,
            end=end,
            batch_size=x.shape[0],
            patch_count=end - start,
            device=x.device,
            stage="update",
        )
        if not isinstance(mask, torch.Tensor):
            mask = torch.as_tensor(mask, device=x.device)
        elif mask.device != x.device:
            mask = mask.to(device=x.device)
        expected = (x.shape[0], end - start)
        if tuple(mask.shape) != expected:
            raise ValueError(
                f"update mask for source view {source_view} has shape "
                f"{tuple(mask.shape)}, expected {expected}"
            )
        return mask
    
    def forward(self, x, info_dict):
        caches = []
        for block in self.blocks:
            fastw, masterw = block.init_fast_weight(batch_size=x.shape[0])
            caches.append({"fastw": fastw, "masterw": masterw})

        outputs = []
        infos = []
        source_view = 0
        for opidx, (start, end, update, _, _) in enumerate(info_dict["ttt_config"]):
            op_info = []

            xi = x[:, start:end, :]

            # Apply all blocks to xi
            for block, cache in zip(self.blocks, caches):
                xi, xi_memory_in, info = block.forward(xi, cache["fastw"])
                cache.update({"xi_out": xi, "xi_memory_in": xi_memory_in})
                if info:
                    op_info.append(info)
            
            if update:
                # Resolve once per source view, before entering any layer update.
                update_keep_mask = self._resolve_update_keep_mask(
                    source_view, start, end, x
                )
                # Get pre_v from corresponding block output
                for i, cache in enumerate(caches):
                    if self.v_gap is not None:
                        cache["pre_vi"] = caches[self.v_layer_idxs[i]]["xi_out"]
                    else:
                        cache["pre_vi"] = cache["xi_memory_in"]
                
                # Update fast weights for all blocks
                for i, (block, cache) in enumerate(zip(self.blocks, caches)):
                    if update_keep_mask is None:
                        fastw, masterw, info = block.update_fast_weight(
                            cache["xi_memory_in"], cache["pre_vi"], cache["fastw"], cache["masterw"]
                        )
                    else:
                        fastw, masterw, info = block.update_fast_weight(
                            cache["xi_memory_in"], cache["pre_vi"], cache["fastw"],
                            cache["masterw"], update_keep_mask
                        )
                    cache.update({"fastw": fastw, "masterw": masterw})
                    if info:
                        op_info[i].update(info)
                source_view += 1
            
            outputs.append(caches[-1]["xi_out"])
            infos.append(op_info)

        outputs = torch.cat(outputs, dim=1)
        return outputs, infos
    
    def extra_repr(self) -> str:
        return "\n".join(f"vmap: {i} <- {v_layer_idx}" for i, v_layer_idx in enumerate(self.v_layer_idxs))