import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from einops import rearrange

from . import debug_utils
from .debug_utils import sketch_tensor


# PyTorch replacements for fla modules
class ShortConvolution(nn.Module):
    """Causal 1D depthwise convolution with optional SiLU activation."""
    def __init__(self, hidden_size, kernel_size, bias=False, activation=None):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            groups=hidden_size,
            padding=kernel_size - 1,
            bias=bias
        )
        self.activation = activation

    def forward(self, x, cache=None, output_final_state=False):
        x = self.conv(x.transpose(1, 2))[..., :x.shape[1]].transpose(1, 2)
        if self.activation == 'silu':
            x = F.silu(x)
        return x, cache


class FusedRMSNormGated(nn.Module):
    """RMS normalization with swish gating."""
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x, g):
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * rms * self.weight) * F.silu(g)


class RMSNorm(nn.Module):
    """Standard RMS normalization."""
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


def elu_p1(x):
    """ELU + 1 activation for positive outputs."""
    return (F.elu(x, 1., False) + 1.).to(x)


def sum_norm(x):
    """Sum normalization."""
    return (x / x.sum(-1, keepdim=True)).to(x)


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim,
        head_dim,
        bias=False,
    ):
        super().__init__()
        self.dim = dim
        self.head_dim = head_dim

        self.to_qkv = nn.Linear(dim, 3 * dim, bias=bias)
        self.to_out = nn.Linear(dim, dim, bias=bias)

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
        bias=False
    ):
        super().__init__()
        inter_dim = int(dim * inter_multi)

        self.gate = nn.Linear(dim, inter_dim, bias=bias)
        self.up = nn.Linear(dim, inter_dim, bias=bias)
        self.down= nn.Linear(inter_dim, dim, bias=bias)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


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
    assert len(G.shape) == 3
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


class FastWeightDeltaSubNN(nn.Module):
    """
    DeltaNet-aligned Fast Weight module for TTT.

    Reference: "Parallelizing Linear Transformers with the Delta Rule"
    https://arxiv.org/abs/2406.06484

    Args:
        dim: Hidden dimension
        fw_num_heads: Number of fast weight heads
        expand_k: Expansion ratio for key dimension (default: 1.0)
        expand_v: Expansion ratio for value dimension (default: 1.0)
        use_beta: Whether to use learnable beta (default: True)
        noise_repeat: Repetition factor for noise
        repeat_function: "repeat" or "linear"
        use_muon: Whether to use Newton-Schulz orthogonalization
        conv_size: Short convolution kernel size
        conv_bias: Whether to use bias in short convolution
        qk_activation: Activation for Q/K ("silu", "relu", "elu", "identity")
        qk_norm: Normalization for Q/K ("l2" or "sum")
        use_output_gate: Whether to use gated output
        allow_neg_eigval: Allow negative eigenvalues (2x beta)
        norm_eps: Epsilon for normalization layers
    """
    def __init__(
        self,
        dim,
        fw_num_heads=1,
        expand_k=1.0,
        expand_v=1.0,
        use_beta=True,
        noise_repeat=1,
        repeat_function=None,
        use_muon=False,
        conv_size=4,
        conv_bias=False,
        qk_activation='silu',
        qk_norm='l2',
        use_output_gate=False,
        allow_neg_eigval=False,
        use_short_conv=True,
        norm_eps=1e-5,
    ):
        super().__init__()
        self.dim = dim
        self.use_muon = use_muon
        self.qk_activation = qk_activation
        self.qk_norm = qk_norm
        self.use_output_gate = use_output_gate
        self.use_beta = use_beta
        self.allow_neg_eigval = allow_neg_eigval
        self.norm_eps = norm_eps

        assert qk_activation in ['silu', 'relu', 'elu', 'identity'], f"Invalid qk_activation: {qk_activation}"
        assert qk_norm in ['l2', 'sum'], f"Invalid qk_norm: {qk_norm}"

        # Multi-head configuration with expand_k/expand_v
        self.fw_num_heads = fw_num_heads
        self.expand_k = expand_k
        self.expand_v = expand_v

        self.key_dim = int(dim * expand_k)
        self.value_dim = int(dim * expand_v)
        self.head_k_dim = self.key_dim // fw_num_heads
        self.head_v_dim = self.value_dim // fw_num_heads

        assert self.key_dim % fw_num_heads == 0, f"key_dim ({self.key_dim}) must be divisible by fw_num_heads ({fw_num_heads})"
        assert self.value_dim % fw_num_heads == 0, f"value_dim ({self.value_dim}) must be divisible by fw_num_heads ({fw_num_heads})"

        self.noise_repeat = noise_repeat
        self.repeat_function = repeat_function

        # Input projections with separate key/value dimensions
        self.q_proj = nn.Linear(dim, self.key_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.key_dim, bias=False)
        # NOTE: no v_proj here. The value target of the delta update is produced by MemoryBlock.to_v
        # (LaCT-style, needed for v_gap); the former v_proj/v_conv1d never received gradient.

        # Beta projection (optional)
        if self.use_beta:
            self.b_proj = nn.Linear(dim, fw_num_heads, bias=False)

        # Short convolutions (from reference DeltaNet)
        self.conv_size = conv_size
        self.use_short_conv = use_short_conv
        # Short causal depthwise conv on q/k (SiLU fused when qk_activation == 'silu'), as in fla.
        # With use_short_conv=False the projections get SiLU only when qk_activation == 'silu' (fla parity).
        if use_short_conv:
            act = 'silu' if qk_activation == 'silu' else None
            self.q_conv1d = ShortConvolution(hidden_size=self.key_dim, kernel_size=conv_size, bias=conv_bias, activation=act)
            self.k_conv1d = ShortConvolution(hidden_size=self.key_dim, kernel_size=conv_size, bias=conv_bias, activation=act)
        else:
            self.q_conv1d = self.k_conv1d = None

        # Fast Weight: W matrix [h, head_v_dim, head_k_dim]
        self.w = nn.Parameter(torch.randn(fw_num_heads, self.head_v_dim, self.head_k_dim) / math.sqrt(self.head_k_dim))

        # Output gate and norm (from reference DeltaNet)
        if use_output_gate:
            self.g_proj = nn.Linear(dim, self.value_dim, bias=False)
            self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)

        # Output projection
        self.o_proj = nn.Linear(self.value_dim, dim, bias=False)

    def init_fast_weights(self, batch_size=1):
        # Initialize batch-wise master weights
        master_w = self.w.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        # Normalize along last dimension
        fast_w = F.normalize(master_w, dim=-1, eps=1e-5).to(torch.bfloat16)
        return (fast_w,), (master_w,)

    def _qk_features(self, input_tensor, need_q=True):
        """q/k projections followed by short conv (use_short_conv) or, for qk_activation == 'silu', SiLU only.
        relu/elu/identity activations are applied later by _apply_qk_activation, as before."""
        q = self.q_proj(input_tensor) if need_q else None
        k = self.k_proj(input_tensor)
        if self.use_short_conv:
            if need_q:
                q, _ = self.q_conv1d(x=q, cache=None, output_final_state=False)
            k, _ = self.k_conv1d(x=k, cache=None, output_final_state=False)
        elif self.qk_activation == 'silu':
            if need_q:
                q = F.silu(q)
            k = F.silu(k)
        return q, k

    def _apply_qk_norm(self, q, k):
        """Apply Q/K normalization based on qk_norm setting."""
        if self.qk_norm == 'l2':
            q = F.normalize(q, dim=-1, eps=self.norm_eps)
            k = F.normalize(k, dim=-1, eps=self.norm_eps)
        elif self.qk_norm == 'sum':
            q = sum_norm(q)
            k = sum_norm(k)
        return q, k

    def _apply_qk_activation(self, q, k):
        """Apply Q/K activation if not handled by short conv."""
        if self.qk_activation == 'relu':
            q, k = q.relu(), k.relu()
        elif self.qk_activation == 'elu':
            q, k = elu_p1(q), elu_p1(k)
        # silu is handled by short conv, identity needs no activation
        return q, k

    def _apply_fast_weights(self, q: torch.Tensor, w: torch.Tensor):
        """
        q: [b, l, h, head_k_dim] (already multi-head split and normalized)
        w: [b, h, head_v_dim, head_k_dim]
        Returns: output [b, l, h, head_v_dim]
        """
        # Linear application: y = q @ W.T
        # [b, l, h, head_k_dim] @ [b, h, head_k_dim, head_v_dim].T -> [b, l, h, head_v_dim]
        out_mh = torch.einsum('blhk,bhvk->blhv', q, w)
        return out_mh

    def forward(self, input_tensor, fastw):
        """
        input_tensor: [b, l, d]
        fastw: (BF16[b, h, head_v_dim, head_k_dim],) single weight tuple
        """
        (w,) = fastw

        # Project and apply short convolution
        q, k = self._qk_features(input_tensor)

        # Multi-head split with separate key/value dimensions
        q = rearrange(q, 'b l (h d) -> b l h d', d=self.head_k_dim)
        k = rearrange(k, 'b l (h d) -> b l h d', d=self.head_k_dim)

        # Apply activation if not silu (silu is handled by conv)
        if self.qk_activation != 'silu':
            q, k = self._apply_qk_activation(q, k)

        # Apply Q/K normalization (L2 or sum)
        q, k = self._apply_qk_norm(q, k)

        # Apply Fast Weights
        out = self._apply_fast_weights(q, w)

        # Output gating and normalization
        if self.use_output_gate:
            g = rearrange(self.g_proj(input_tensor), 'b l (h d) -> b l h d', d=self.head_v_dim)
            out = self.o_norm(out, g)
        else:
            out = self.o_norm(out)

        out = rearrange(out, 'b l h d -> b l (h d)')
        out = self.o_proj(out)

        if debug_utils.log_status:
            info = {
                "w": sketch_tensor(w),
                "q": sketch_tensor(q),
                "k": sketch_tensor(k),
                "out": sketch_tensor(out),
            }
        else:
            info = {}

        return out, info

    def update(self, input_tensor, v_target, fastw, masterw):
        """
        Delta Rule Update: W = W + beta * (v - Wk) @ k.T

        input_tensor: [b, l, d]
        v_target: [b, l, value_dim] (target values)
        fastw: (BF16[b, h, head_v_dim, head_k_dim],)
        masterw: (FP32[b, h, head_v_dim, head_k_dim],)
        """
        (w_master,) = masterw
        (w,) = fastw
        b, l, _ = input_tensor.shape

        # 1. Compute beta (bounded learning rate) or use ones
        if self.use_beta:
            with torch.autocast(device_type="cuda", enabled=False):
                beta = self.b_proj(input_tensor.float()).sigmoid()  # [b, l, h]
                if self.allow_neg_eigval:
                    beta = beta * 2.0
        else:
            beta = torch.ones(b, l, self.fw_num_heads, device=input_tensor.device, dtype=input_tensor.dtype)

        # 2. Project and apply short convolution for K
        _, k = self._qk_features(input_tensor, need_q=False)

        # Handle repeat function for noise
        if self.repeat_function == "repeat":
            k = k.repeat(1, self.noise_repeat, 1)
            v_target = v_target.repeat(1, self.noise_repeat, 1)
            beta = beta.repeat(1, self.noise_repeat, 1)
        elif self.repeat_function == "linear":
            # For linear, we need to expand dimensions appropriately
            k = k.reshape(b, l, self.noise_repeat, self.key_dim).reshape(b, l * self.noise_repeat, self.key_dim)
            v_target = v_target.reshape(b, l, self.noise_repeat, self.value_dim).reshape(b, l * self.noise_repeat, self.value_dim)
            beta = beta.unsqueeze(2).expand(-1, -1, self.noise_repeat, -1).reshape(b, l * self.noise_repeat, self.fw_num_heads)
        else:
            raise ValueError(f"Invalid repeat function: {self.repeat_function}")

        seq_len = l * self.noise_repeat

        # Multi-head split for k (using head_k_dim)
        k_mh = rearrange(k, 'b l (h d) -> b l h d', h=self.fw_num_heads, d=self.head_k_dim)

        # Apply activation and normalization to k
        if self.qk_activation != 'silu':
            k_mh, _ = self._apply_qk_activation(k_mh, k_mh)
        if self.qk_norm == 'l2':
            k_mh = F.normalize(k_mh, dim=-1, eps=self.norm_eps)
        elif self.qk_norm == 'sum':
            k_mh = sum_norm(k_mh)

        # 3. Compute current prediction: v_hat = k @ W.T
        # k_mh: [b, l, h, head_k_dim], w: [b, h, head_v_dim, head_k_dim]
        v_hat_mh = torch.einsum('blhk,bhvk->blhv', k_mh, w)

        # 4. Compute prediction error: e = v - v_hat
        # v_target: [b, l, value_dim] -> [b, l, h, head_v_dim]
        v_mh = rearrange(v_target, 'b l (h d) -> b l h d', h=self.fw_num_heads, d=self.head_v_dim)
        error_mh = v_mh.float() - v_hat_mh.float()

        # 5. Compute gradient with beta: grad = beta * error.T @ k
        # beta shape: [b, l, h] -> expand for element-wise multiply
        beta_expanded = beta.unsqueeze(-1)  # [b, l, h, 1]

        # [b, l, h, head_v_dim] @ [b, l, h, head_k_dim] -> [b, h, head_v_dim, head_k_dim]
        delta_update = torch.einsum('blhv,blhk->bhvk', beta_expanded * error_mh, k_mh)

        # Average over sequence
        grad_raw = delta_update / seq_len

        # 6. Optionally apply Newton-Schulz orthogonalization (MUON)
        if self.use_muon:
            b_size, h = grad_raw.shape[:2]
            grad_flat = rearrange(grad_raw, 'b h d1 d2 -> (b h) d1 d2')
            grad_ns = zeropower_via_newtonschulz5(grad_flat, 5)
            grad = rearrange(grad_ns, '(b h) d1 d2 -> b h d1 d2', b=b_size, h=h)
        else:
            grad = grad_raw

        # 7. Update Master Weight
        w_master = w_master + grad

        # 8. Normalize and convert back to BF16
        w_new = F.normalize(w_master, dim=-1, eps=1e-5).to(torch.bfloat16)

        if debug_utils.log_status:
            info = {
                "update_k": sketch_tensor(k_mh),
                "update_v_hat": sketch_tensor(v_hat_mh),
                "update_error": sketch_tensor(error_mh),
                "update_beta": sketch_tensor(beta),
                "update_grad": sketch_tensor(grad_raw),
                "update_w_master": sketch_tensor(w_master),
            }
        else:
            info = {}

        return (w_new,), (w_master,), info

    def extra_repr(self) -> str:
        return (f"w shape: {self.w.shape}, fw_num_heads: {self.fw_num_heads}, "
                f"expand_k: {self.expand_k}, expand_v: {self.expand_v}, use_beta: {self.use_beta}, "
                f"qk_activation: {self.qk_activation}, qk_norm: {self.qk_norm}")


class MemoryBlock(nn.Module):
    def __init__(
        self,
        dim,
        attn_head_dim=64,
        inter_multi=2,
        fw_num_heads=1,
        expand_k=1.0,
        expand_v=1.0,
        use_beta=True,
        bias=False,
        repeat_function="repeat",
        noise_repeat=1,
        use_muon=False,
        conv_size=4,
        conv_bias=False,
        qk_activation='silu',
        qk_norm='l2',
        use_output_gate=True,
        allow_neg_eigval=False,
        use_short_conv=True,
        norm_eps=1e-5,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.expand_v = expand_v

        self.repeat_function = repeat_function
        self.noise_repeat = noise_repeat

        # Compute value dimension for to_v projection
        self.value_dim = int(dim * expand_v)

        # Attention
        self.ln_attn = nn.RMSNorm(dim, eps=norm_eps)
        self.self_attention = SelfAttention(dim, attn_head_dim, bias=bias)

        # Fast Weight (Memory) - DeltaNet aligned
        self.ln_memory = nn.RMSNorm(dim, eps=norm_eps)
        self.memory = FastWeightDeltaSubNN(
            dim,
            fw_num_heads=fw_num_heads,
            expand_k=expand_k,
            expand_v=expand_v,
            use_beta=use_beta,
            noise_repeat=noise_repeat,
            repeat_function=repeat_function,
            use_muon=use_muon,
            conv_size=conv_size,
            conv_bias=conv_bias,
            qk_activation=qk_activation,
            qk_norm=qk_norm,
            use_output_gate=use_output_gate,
            allow_neg_eigval=allow_neg_eigval,
            use_short_conv=use_short_conv,
            norm_eps=norm_eps,
            **kwargs
        )

        # V projection (uses value_dim for expand_v support)
        if self.repeat_function == "repeat":
            self.to_v = nn.Linear(dim, self.value_dim, bias=False)
        elif self.repeat_function == "linear":
            self.to_v = nn.Linear(dim, noise_repeat * self.value_dim, bias=False)
        else:
            raise ValueError(f"Invalid repeat function: {self.repeat_function}")

        # Slow MLP (Knowledge)
        self.ln_mlp = nn.RMSNorm(dim, eps=norm_eps)
        self.mlp = MLP(dim, inter_multi, bias=bias)
    
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
        o, info = self.memory(x, fastw)
        x = skip + o

        # Slow weight apply
        skip = x
        x = self.ln_mlp(x)
        x = skip + self.mlp(x)

        return x, x_memory_in, info
    
    @torch.compile
    def forward(self, *args):
        # if debug_utils.log_status: pass
        return checkpoint(self._forward, *args, preserve_rng_state=False, use_reentrant=False)
    
    def _update_fast_weight(self, memory_in, pre_vi, fastw, masterw):
        # rms_norm
        pre_vi = F.rms_norm(pre_vi, normalized_shape=(self.dim,), eps=1e-5)

        # V prediction (beta is now computed inside memory.update)
        vi = self.to_v(pre_vi)

        fastw, masterw, info = self.memory.update(memory_in, vi, fastw, masterw)
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

        self.blocks = []
        self.v_layer_idxs = []
        for i in range(layers):
            if v_gap is not None:
                self.v_layer_idxs.append(max(min(i + v_gap, layers - 1), 0))
            self.blocks.append(MemoryBlock(dim=dim, **kwargs))
        self.blocks = nn.ModuleList(self.blocks)

    def forward(self, x, info_dict):
        caches = []
        for block in self.blocks:
            fastw, masterw = block.init_fast_weight(batch_size=x.shape[0])
            caches.append({"fastw": fastw, "masterw": masterw})

        outputs = []
        infos = []
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
                # Get pre_v from corresponding block output
                for i, cache in enumerate(caches):
                    if self.v_gap is not None:
                        cache["pre_vi"] = caches[self.v_layer_idxs[i]]["xi_out"]
                    else:
                        cache["pre_vi"] = cache["xi_memory_in"]
                
                # Update fast weights for all blocks
                for i, (block, cache) in enumerate(zip(self.blocks, caches)):
                    fastw, masterw, info = block.update_fast_weight(
                        cache["xi_memory_in"], cache["pre_vi"], cache["fastw"], cache["masterw"]
                    )
                    cache.update({"fastw": fastw, "masterw": masterw})
                    if info:
                        op_info[i].update(info)
            
            outputs.append(caches[-1]["xi_out"])
            infos.append(op_info)

        outputs = torch.cat(outputs, dim=1)
        return outputs, infos
    
    def extra_repr(self) -> str:
        return "\n".join(f"vmap: {i} <- {v_layer_idx}" for i, v_layer_idx in enumerate(self.v_layer_idxs))
        

