"""MHA fast-weight with cross-layer shared weights (mha_global_v2).

Same as memory_block_v2_mha_global but adds aggregate-write semantics:

  aggregate_write=False (default):  sequential update — each layer writes back
      to the shared pool before the next layer reads it (old behaviour).

  aggregate_write=True:  all layers compute raw gradients from the SAME W_N
      snapshot, gradients are summed, Muon/NS5 is applied ONCE on the sum,
      then W_{N+1} is written exactly once.

New methods compared with memory_block_v2_mha_global:
  FastWeightMLPSubNN.compute_raw_grad()   — raw grad, no Muon, no write
  FastWeightMLPSubNN.apply_raw_update()   — Muon + write + normalize once
  MemoryBlock._compute_raw_grad_fast_weight() / compute_raw_grad_fast_weight()
  MemoryModel.__init__: aggregate_write=False
"""

import math

import torch
from einops import rearrange
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from . import debug_utils
from .debug_utils import sketch_tensor

from .ttt_dense_block import (
    SelfAttention,
    MLP,
    silu_backprop,
    zeropower_via_newtonschulz5,
    inv_softplus,
)


class FastWeightMLPSubNN(nn.Module):
    """Stateless per-head MHA FW.

    Same forward/update math as memory_block_v2_mha_l2norm.FastWeightMLPSubNN
    but w0/w1/w2 are not owned here — they are passed in from the shared pool
    owned by MemoryModel.
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

        self.input_proj = nn.Sequential(nn.Linear(dim, dim, bias=False), nn.SiLU())
        if not self.l2_norm:
            self.input_rms_norm = nn.RMSNorm(
                self.head_dim, eps=1e-5, elementwise_affine=False
            )

        self.k_input_proj = nn.Sequential(nn.Linear(dim, dim, bias=False), nn.SiLU())

        inter_dim = int(self.head_dim * inter_multi)
        self.inter_dim = inter_dim

        # No w0/w1/w2 owned here — provided by MemoryModel's shared_w*

        self.output_rms_norm = nn.RMSNorm(
            self.head_dim, eps=1e-5, elementwise_affine=False
        )
        self.output_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, input, fastw):
        """
        input: [b, l, d]
        fastw: BF16 tuple ([b, h, hd, dh], [b, h, dh, hd], [b, h, hd, dh])
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
        grad:  [b, l, d]
        lr:    [b, l, 3 * h]
        fastw:   BF16 tuple ([b, h, hd, dh], [b, h, dh, hd], [b, h, hd, dh])
        masterw: FP32 tuple ([b, h, hd, dh], [b, h, dh, hd], [b, h, hd, dh])
        """
        w0_master, w1_master, w2_master = masterw
        w0, w1, w2 = fastw
        lr = rearrange(lr, "b l (h d) -> b h l d", h=self.num_heads, d=3)
        lr0, lr1, lr2 = lr.chunk(3, dim=3)

        L = input.shape[1]

        x = self.k_input_proj(input)
        x = rearrange(x, "b l (h d) -> b h l d", h=self.num_heads)
        if self.l2_norm:
            x = F.normalize(x, dim=-1, eps=1e-5).to(x.dtype)
        else:
            x = self.input_rms_norm(x)

        gate_before_act = x @ w0
        hidden_before_mul = x @ w2
        hidden = F.silu(gate_before_act) * hidden_before_mul

        grad = grad.float() / L
        grad = rearrange(grad, "b l (h d) -> b h l d", h=self.num_heads)

        dhidden = grad @ w1.transpose(-1, -2)
        dhidden_before_mul = dhidden * F.silu(gate_before_act)
        dgate = dhidden * hidden_before_mul
        dgate_before_act = silu_backprop(dgate, gate_before_act)

        w1_grad_raw = (hidden * lr1).transpose(-1, -2) @ grad
        w0_grad_raw = (x * lr0).transpose(-1, -2) @ dgate_before_act
        w2_grad_raw = (x * lr2).transpose(-1, -2) @ dhidden_before_mul

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

        w0_master = w0_master + w0_grad
        w1_master = w1_master + w1_grad
        w2_master = w2_master + w2_grad
        masterw = (w0_master, w1_master, w2_master)

        w0 = F.normalize(w0_master, dim=2, eps=1e-5).to(torch.bfloat16)
        w1 = F.normalize(w1_master, dim=2, eps=1e-5).to(torch.bfloat16)
        w2 = F.normalize(w2_master, dim=2, eps=1e-5).to(torch.bfloat16)
        weight = (w0, w1, w2)

        return weight, masterw, info

    def compute_raw_grad(self, input, grad, lr, fastw):
        """Compute raw dw0/dw1/dw2 from the given fastw snapshot.

        Identical to the first half of update() up to and including the
        gradient outer products, but:
          - does NOT apply Muon/NS5
          - does NOT write masterw
          - does NOT normalize

        Returns:
            (w0_grad_raw, w1_grad_raw, w2_grad_raw) all float32, info dict
        """
        w0, w1, w2 = fastw
        lr = rearrange(lr, "b l (h d) -> b h l d", h=self.num_heads, d=3)
        lr0, lr1, lr2 = lr.chunk(3, dim=3)

        L = input.shape[1]

        x = self.k_input_proj(input)
        x = rearrange(x, "b l (h d) -> b h l d", h=self.num_heads)
        if self.l2_norm:
            x = F.normalize(x, dim=-1, eps=1e-5).to(x.dtype)
        else:
            x = self.input_rms_norm(x)

        gate_before_act = x @ w0
        hidden_before_mul = x @ w2
        hidden = F.silu(gate_before_act) * hidden_before_mul

        grad = grad.float() / L
        grad = rearrange(grad, "b l (h d) -> b h l d", h=self.num_heads)

        dhidden = grad @ w1.transpose(-1, -2)
        dhidden_before_mul = dhidden * F.silu(gate_before_act)
        dgate = dhidden * hidden_before_mul
        dgate_before_act = silu_backprop(dgate, gate_before_act)

        w1_grad_raw = ((hidden * lr1).transpose(-1, -2) @ grad).float()
        w0_grad_raw = ((x * lr0).transpose(-1, -2) @ dgate_before_act).float()
        w2_grad_raw = ((x * lr2).transpose(-1, -2) @ dhidden_before_mul).float()

        if debug_utils.log_status:
            info = {
                "update_w0_grad": sketch_tensor(w0_grad_raw),
                "update_w1_grad": sketch_tensor(w1_grad_raw),
                "update_w2_grad": sketch_tensor(w2_grad_raw),
            }
        else:
            info = {}

        return (w0_grad_raw, w1_grad_raw, w2_grad_raw), info

    def apply_raw_update(self, raw_grads, masterw):
        """Apply Muon once on the aggregated raw grads, write masterw, normalize.

        Called once after summing raw grads across all layers.
        Returns new_fastw (bf16), new_masterw (fp32).
        """
        w0_grad_raw, w1_grad_raw, w2_grad_raw = raw_grads

        if self.use_muon:
            b, h, _, _ = w0_grad_raw.shape
            w0_grad = zeropower_via_newtonschulz5(w0_grad_raw.flatten(0, 1), 5).view(b, h, self.head_dim, self.inter_dim)
            w1_grad = zeropower_via_newtonschulz5(w1_grad_raw.flatten(0, 1), 5).view(b, h, self.inter_dim, self.head_dim)
            w2_grad = zeropower_via_newtonschulz5(w2_grad_raw.flatten(0, 1), 5).view(b, h, self.head_dim, self.inter_dim)
        else:
            w0_grad, w1_grad, w2_grad = w0_grad_raw, w1_grad_raw, w2_grad_raw

        w0_master, w1_master, w2_master = masterw
        w0_master = w0_master + w0_grad
        w1_master = w1_master + w1_grad
        w2_master = w2_master + w2_grad
        new_masterw = (w0_master, w1_master, w2_master)

        new_fastw = (
            F.normalize(w0_master, dim=2, eps=1e-5).to(torch.bfloat16),
            F.normalize(w1_master, dim=2, eps=1e-5).to(torch.bfloat16),
            F.normalize(w2_master, dim=2, eps=1e-5).to(torch.bfloat16),
        )
        return new_fastw, new_masterw

    def extra_repr(self) -> str:
        return (
            f"fw_head_dim: {self.head_dim}, num_heads: {self.num_heads}, inter_dim: {self.inter_dim}, "
            f"(shared weights owned by MemoryModel), use_muon: {self.use_muon}, l2_norm: {self.l2_norm}"
        )


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

        # Fast MLP (Memory) — stateless
        self.ln_memory = nn.RMSNorm(dim, eps=1e-5)
        self.memory = FastWeightMLPSubNN(dim, fw_head_dim, fw_inter_multi, **kwargs)

        # FW supervisions (per-layer, stay independent across layers)
        self.to_v = nn.Sequential(nn.Linear(dim, dim, bias=False), nn.SiLU())
        self.to_lr = nn.Linear(dim, 3 * self.memory.num_heads, bias=False)
        self.base_lr_inv = inv_softplus(base_lr)

        # Slow MLP (Knowledge) — per-layer
        self.ln_mlp = nn.RMSNorm(dim, eps=1e-5)
        self.mlp = MLP(dim, inter_multi)

    def _forward(self, x, fastw):
        skip = x
        x = self.ln_attn(x)
        x = skip + self.self_attention(x)

        skip = x
        x_memory_in = x = self.ln_memory(x)
        x, info = self.memory(x, fastw)
        x = skip + x

        skip = x
        x = self.ln_mlp(x)
        x = skip + self.mlp(x)

        return x, x_memory_in, info

    @torch.compile
    def forward(self, *args):
        return checkpoint(self._forward, *args, preserve_rng_state=False, use_reentrant=False)

    def _update_fast_weight(self, memory_in, pre_vi, fastw, masterw):
        pre_vi = F.rms_norm(pre_vi, normalized_shape=(self.dim,), eps=1e-5)

        vi = self.to_v(pre_vi)
        with torch.autocast(device_type="cuda", enabled=False):
            lri = self.to_lr(pre_vi.float())
            lri = torch.nn.functional.softplus(lri + self.base_lr_inv)

        fastw, masterw, info = self.memory.update(memory_in, vi, lri, fastw, masterw)
        return fastw, masterw, info

    @torch.compile
    def update_fast_weight(self, *args):
        return checkpoint(
            self._update_fast_weight,
            *args,
            preserve_rng_state=False,
            use_reentrant=False,
        )

    def _compute_raw_grad_fast_weight(self, memory_in, pre_vi, fastw):
        pre_vi = F.rms_norm(pre_vi, normalized_shape=(self.dim,), eps=1e-5)

        vi = self.to_v(pre_vi)
        with torch.autocast(device_type="cuda", enabled=False):
            lri = self.to_lr(pre_vi.float())
            lri = torch.nn.functional.softplus(lri + self.base_lr_inv)

        raw_grads, info = self.memory.compute_raw_grad(memory_in, vi, lri, fastw)
        return raw_grads, info

    def compute_raw_grad_fast_weight(self, *args):
        return checkpoint(
            self._compute_raw_grad_fast_weight,
            *args,
            preserve_rng_state=False,
            use_reentrant=False,
        )


class MemoryModel(nn.Module):
    """MHA fast-weight with cross-layer shared weights (v2).

    aggregate_write=False (default): sequential update — same as the original
        memory_block_v2_mha_global behaviour.

    aggregate_write=True: all layers share W_N for forward; each layer
        computes a raw grad on W_N; grads are summed; Muon+normalize applied
        once; W_{N+1} written exactly once per chunk.
    """
    def __init__(
        self,
        layers,
        dim,
        v_gap=None,
        fw_head_dim=None,
        fw_inter_multi=1,
        aggregate_write=False,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.v_gap = v_gap
        self.aggregate_write = aggregate_write

        if fw_head_dim is None:
            fw_head_dim = dim
        self.head_dim = fw_head_dim
        self.num_heads = dim // fw_head_dim
        self.inter_dim = int(self.head_dim * fw_inter_multi)
        self.fw_inter_multi = fw_inter_multi

        # Shared FW parameters (ONE set across all layers)
        self.shared_w0 = nn.Parameter(
            torch.randn(self.num_heads, self.head_dim, self.inter_dim) / math.sqrt(self.head_dim)
        )
        self.shared_w1 = nn.Parameter(
            torch.randn(self.num_heads, self.inter_dim, self.head_dim) / math.sqrt(self.inter_dim)
        )
        self.shared_w2 = nn.Parameter(
            torch.randn(self.num_heads, self.head_dim, self.inter_dim) / math.sqrt(self.head_dim)
        )

        # Blocks (each has its own attn/mlp/to_v/to_lr, but their memory is stateless)
        self.blocks = []
        self.v_layer_idxs = []
        for i in range(layers):
            if v_gap is not None:
                self.v_layer_idxs.append(max(min(i + v_gap, layers - 1), 0))
            self.blocks.append(
                MemoryBlock(
                    dim=dim,
                    fw_head_dim=fw_head_dim,
                    fw_inter_multi=fw_inter_multi,
                    **kwargs,
                )
            )
        self.blocks = nn.ModuleList(self.blocks)

    def init_shared_fast_weight(self, batch_size):
        """Expand shared weights to per-batch fastw/masterw tuples."""
        master_weight = (
            self.shared_w0.unsqueeze(0).repeat(batch_size, 1, 1, 1),
            self.shared_w1.unsqueeze(0).repeat(batch_size, 1, 1, 1),
            self.shared_w2.unsqueeze(0).repeat(batch_size, 1, 1, 1),
        )
        weight = tuple(
            F.normalize(w, dim=2, eps=1e-5).to(torch.bfloat16)
            for w in master_weight
        )
        return weight, master_weight

    def forward(self, x, info_dict):
        # ONE shared fastw/masterw shared across all layers
        fastw, masterw = self.init_shared_fast_weight(batch_size=x.shape[0])

        caches = [{} for _ in self.blocks]

        outputs = []
        infos = []
        for start, end, update, _, _ in info_dict["ttt_config"]:
            op_info = []

            xi = x[:, start:end, :]

            # Forward: all layers consume the SAME shared fastw (W_N snapshot)
            for block, cache in zip(self.blocks, caches):
                xi, xi_memory_in, info = block.forward(xi, fastw)
                cache.update({"xi_out": xi, "xi_memory_in": xi_memory_in})
                if info:
                    op_info.append(info)

            if update:
                # Resolve pre_vi per layer (respecting v_gap)
                for i, cache in enumerate(caches):
                    if self.v_gap is not None:
                        cache["pre_vi"] = caches[self.v_layer_idxs[i]]["xi_out"]
                    else:
                        cache["pre_vi"] = cache["xi_memory_in"]

                if self.aggregate_write:
                    # All layers compute raw grad from the same W_N snapshot,
                    # sum across layers, apply Muon+normalize exactly once.
                    agg_dw0 = torch.zeros_like(masterw[0])
                    agg_dw1 = torch.zeros_like(masterw[1])
                    agg_dw2 = torch.zeros_like(masterw[2])

                    for i, (block, cache) in enumerate(zip(self.blocks, caches)):
                        raw_grads, info = block.compute_raw_grad_fast_weight(
                            cache["xi_memory_in"],
                            cache["pre_vi"],
                            fastw,  # same W_N snapshot for every layer
                        )
                        agg_dw0 = agg_dw0 + raw_grads[0]
                        agg_dw1 = agg_dw1 + raw_grads[1]
                        agg_dw2 = agg_dw2 + raw_grads[2]
                        if info and i < len(op_info):
                            op_info[i].update(info)

                    # One Muon + write + normalize for the aggregated gradient
                    fastw, masterw = self.blocks[0].memory.apply_raw_update(
                        (agg_dw0, agg_dw1, agg_dw2), masterw
                    )
                else:
                    # Sequential update (original behaviour): each layer writes
                    # back to the shared pool; next layer reads updated state.
                    for i, (block, cache) in enumerate(zip(self.blocks, caches)):
                        fastw, masterw, info = block.update_fast_weight(
                            cache["xi_memory_in"], cache["pre_vi"], fastw, masterw
                        )
                        if info and i < len(op_info):
                            op_info[i].update(info)

            outputs.append(caches[-1]["xi_out"])
            infos.append(op_info)

        outputs = torch.cat(outputs, dim=1)
        return outputs, infos

    def extra_repr(self) -> str:
        lines = [
            f"num_heads: {self.num_heads}, head_dim: {self.head_dim}, inter_dim: {self.inter_dim} (fw_inter_multi={self.fw_inter_multi})",
            f"aggregate_write: {self.aggregate_write}",
            f"shared_w0: {self.shared_w0.shape}, shared_w1: {self.shared_w1.shape}, shared_w2: {self.shared_w2.shape}",
        ]
        lines.extend(
            f"vmap: {i} <- {v_layer_idx}"
            for i, v_layer_idx in enumerate(self.v_layer_idxs)
        )
        return "\n".join(lines)


def _unit_test_mha_global_v2():
    # python -m uttt_nvs.models.uttt_dense_block
    torch.manual_seed(0)
    if not torch.cuda.is_available():
        print("CUDA not available; skipping mha_global_v2 unit test.")
        return {"skipped": True}

    b, L, dim = 2, 512, 512
    layers = 4
    num_img_tokens = 128

    ttt_config = []
    for i in range(L // num_img_tokens):
        start = i * num_img_tokens
        end = start + num_img_tokens
        do_update = (i % 2 == 0)
        ttt_config.append((start, end, do_update, None, None))

    info_dict = {"num_img_tokens": num_img_tokens, "ttt_config": ttt_config}

    common_kwargs = dict(
        layers=layers,
        dim=dim,
        attn_head_dim=64,
        fw_head_dim=64,
        inter_multi=3,
        base_lr=0.01,
        use_muon=False,
        l2_norm=True,
    )

    # --- aggregate_write=False (original path) ---
    print("\n" + "="*60)
    print("Testing aggregate_write=False (all fw_inter_multi)")
    print("="*60)
    for fw_inter_multi in [1, 2, 4, 8]:
        print(f"\n  fw_inter_multi={fw_inter_multi}")
        torch.manual_seed(42)
        x = torch.randn(b, L, dim, device="cuda", dtype=torch.bfloat16)
        model = MemoryModel(fw_inter_multi=fw_inter_multi, aggregate_write=False,
                            **common_kwargs).to("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs, _ = model(x, info_dict)
        outputs.sum().backward()
        print(f"  outputs {tuple(outputs.shape)} {outputs.dtype}  backward OK")

    # --- aggregate_write=True ---
    print("\n" + "="*60)
    print("Testing aggregate_write=True (all fw_inter_multi)")
    print("="*60)
    for fw_inter_multi in [1, 2, 4, 8]:
        print(f"\n  fw_inter_multi={fw_inter_multi}")
        torch.manual_seed(42)
        x = torch.randn(b, L, dim, device="cuda", dtype=torch.bfloat16)
        model = MemoryModel(fw_inter_multi=fw_inter_multi, aggregate_write=True,
                            **common_kwargs).to("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs, _ = model(x, info_dict)
        outputs.sum().backward()
        print(f"  outputs {tuple(outputs.shape)} {outputs.dtype}  backward OK")

    # --- layers=1: aggregate_write=True must match False exactly ---
    print("\n" + "="*60)
    print("Testing layers=1 equivalence (True ~= False)")
    print("="*60)
    torch.manual_seed(42)
    x1 = torch.randn(b, L, dim, device="cuda", dtype=torch.bfloat16)
    kw1 = dict(**common_kwargs, fw_inter_multi=2, layers=1)
    m_agg = MemoryModel(aggregate_write=True,  **kw1).to("cuda")
    m_seq = MemoryModel(aggregate_write=False, **kw1).to("cuda")
    m_seq.load_state_dict(m_agg.state_dict())
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out_agg, _ = m_agg(x1, info_dict)
        out_seq, _ = m_seq(x1, info_dict)
    close = torch.allclose(out_agg.float(), out_seq.float(), atol=1e-2)
    print(f"  layers=1 outputs close: {close}  (expected True)")
    assert close, "layers=1: aggregate_write True/False must produce identical outputs"

    # --- layers>1: outputs must differ ---
    print("\n" + "="*60)
    print("Testing layers>1 divergence (True != False)")
    print("="*60)
    torch.manual_seed(42)
    x4 = torch.randn(b, L, dim, device="cuda", dtype=torch.bfloat16)
    kw4 = dict(**common_kwargs, fw_inter_multi=2, layers=4)
    m_agg4 = MemoryModel(aggregate_write=True,  **kw4).to("cuda")
    m_seq4 = MemoryModel(aggregate_write=False, **kw4).to("cuda")
    m_seq4.load_state_dict(m_agg4.state_dict())
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out_agg4, _ = m_agg4(x4, info_dict)
        out_seq4, _ = m_seq4(x4, info_dict)
    differ = not torch.allclose(out_agg4.float(), out_seq4.float(), atol=1e-2)
    print(f"  layers=4 outputs differ: {differ}  (expected True)")
    assert differ, "layers>1: aggregate_write True/False should produce different outputs"

    return {"success": True}


if __name__ == "__main__":
    print(_unit_test_mha_global_v2())
