import math

import torch
from einops import rearrange
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from . import debug_utils
from .debug_utils import sketch_tensor

from .routing import create_router_mask_sizes_probs
from .kernels.lact_swiglu_ffn import grouped_swiglu_ffn_fwd
from .kernels.lact_fw_grad import grouped_lact_swiglu_ffn_fast_weight_grads
from .kernels.triton_permute import (
    permute_kv_and_lrs,
    permute_with_expert_mask,
    unpermute_and_merge_with_probs,
)


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
        q, k, v = rearrange(
            self.to_qkv(x), "b l (qkv h d) -> qkv b h l d", qkv=3, d=self.head_dim
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
        self.down = nn.Linear(inter_dim, dim, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


def inv_softplus(x):
    y = x + math.log(-math.expm1(-x))
    return y


def zeropower_via_newtonschulz5(G, steps=5):
    if steps == 0:
        return G
    a, b, c = (3.4445, -4.775, 2.0315)
    X = G.bfloat16()
    if G.size(1) > G.size(2):
        X = X.transpose(1, 2)
    X = X / (X.norm(dim=(1, 2), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.transpose(1, 2)
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(1) > G.size(2):
        X = X.transpose(1, 2)
    return X


class FastWeightMLPSubNN(nn.Module):
    """
    Fast weight MLP with MoE routing.
    Unlike moe_global, this module does NOT own w0/w1/w2 or router weights.
    They are passed in from the shared pool owned by MemoryModel.
    """
    def __init__(
        self,
        dim,
        fw_inter_multi,
        fw_head_dim=None,
        base_lr=0.01,
        use_muon=True,
        num_experts=2,
        num_active_experts=1,
        sigmoid_router=False,
        router_alpha=1.0,
        router_from_v=False,
        l2_norm=False,
        use_shared_expert=False,
    ):
        super().__init__()
        self.dim = dim
        self.use_muon = use_muon
        self.l2_norm = l2_norm
        if fw_head_dim is None:
            fw_head_dim = dim
        self.head_dim = fw_head_dim
        self.num_heads = dim // fw_head_dim

        self.num_experts = num_experts
        self.num_active_experts = num_active_experts
        self.sigmoid_router = sigmoid_router
        self.router_alpha = router_alpha
        self.router_from_v = router_from_v
        self.use_shared_expert = use_shared_expert

        self.input_proj = nn.Sequential(nn.Linear(dim, dim, bias=False), nn.SiLU())
        if not self.l2_norm:
            self.input_rms_norm = nn.RMSNorm(
                self.head_dim, eps=1e-5, elementwise_affine=False
            )

        self.k_input_proj = nn.Sequential(nn.Linear(dim, dim, bias=False), nn.SiLU())

        inter_dim = int(self.head_dim * fw_inter_multi)
        self.inter_dim = inter_dim

        # No w0/w1/w2 parameters — provided by shared pool
        # No router_proj_weights — provided by MemoryModel

        self.to_v = nn.Sequential(nn.Linear(dim, dim, bias=False), nn.SiLU())
        self.to_lr = nn.Linear(dim, 3 * self.num_heads, bias=False)
        self.base_lr_inv = inv_softplus(base_lr)

        # Disabled by default. The callback stays outside model configuration
        # and state_dict, preserving checkpoint compatibility.
        self._router_trace_callback = None
        self._router_trace_layer_idx = None
        self._router_trace_metadata_cache = None

        self.output_rms_norm = nn.RMSNorm(self.head_dim, eps=1e-5, elementwise_affine=False)
        self.output_proj = nn.Linear(dim, dim, bias=False)

    def set_router_trace_callback(self, callback, layer_idx=None):
        """Attach an optional token/head-level update-router trace callback.

        The callback receives token metadata shaped ``[batch, patch * head]``
        plus detached key/value tensors and flags that state whether they are
        already unit normalized. No trace tensors are constructed while the
        callback is ``None``.
        """
        if callback is not None and not callable(callback):
            raise TypeError("router trace callback must be callable or None")
        self._router_trace_callback = callback
        # Do not retain metadata tensors from a previous trace session.
        self._router_trace_metadata_cache = None
        if layer_idx is not None:
            self._router_trace_layer_idx = int(layer_idx)

    def _emit_router_trace(
        self, expert_mask, pre_topk_router_probs, key, value, stage
    ):
        callback = self._router_trace_callback
        if callback is None:
            return
        if self._router_trace_layer_idx is None:
            raise RuntimeError("router trace layer index has not been set")

        accepts_stage = getattr(callback, "accepts_stage", None)
        if accepts_stage is not None and not accepts_stage(stage):
            return

        # Use the actual mask for the selected expert and the complete router
        # distribution for the true top-1/top-2 probability margin.
        with torch.no_grad():
            probs = pre_topk_router_probs.float()
            batch_size, token_heads = probs.shape[0], probs.shape[2]
            trace_index = None
            select_indices = getattr(callback, "select_indices", None)
            if select_indices is not None:
                trace_index = select_indices(
                    stage=stage,
                    layer_idx=self._router_trace_layer_idx,
                    token_heads=token_heads,
                    device=probs.device,
                    num_heads=self.num_heads,
                )
                probs = probs.index_select(2, trace_index)
                expert_mask = expert_mask.index_select(2, trace_index)
                key = key.index_select(1, trace_index)
                value = value.index_select(1, trace_index)

            selected_probs = probs.masked_fill(
                ~expert_mask, torch.finfo(probs.dtype).min
            )
            expert = selected_probs.argmax(dim=1)
            top1_prob = torch.gather(
                probs, 1, expert.unsqueeze(1)
            ).squeeze(1)
            top_count = min(2, probs.shape[1])
            top_probs = torch.topk(probs, k=top_count, dim=1).values
            if top_count == 2:
                margin = top_probs[:, 0, :] - top_probs[:, 1, :]
            else:
                margin = top_probs[:, 0, :].clone()

            traced_token_heads = expert.shape[1]
            cache_key = (
                batch_size,
                token_heads,
                traced_token_heads,
                expert.device,
                trace_index.data_ptr() if trace_index is not None else None,
            )
            cached = self._router_trace_metadata_cache
            if cached is None or cached[0] != cache_key:
                flat_index = (
                    trace_index
                    if trace_index is not None
                    else torch.arange(token_heads, device=expert.device)
                )
                patch = (
                    (flat_index // self.num_heads)
                    .unsqueeze(0)
                    .expand(batch_size, -1)
                )
                head = (
                    (flat_index % self.num_heads)
                    .unsqueeze(0)
                    .expand(batch_size, -1)
                )
                batch = (
                    torch.arange(batch_size, device=expert.device)
                    .unsqueeze(1)
                    .expand(-1, traced_token_heads)
                )
                layer = torch.full_like(patch, self._router_trace_layer_idx)
                cached = (cache_key, batch, layer, patch, head)
                self._router_trace_metadata_cache = cached
            _, batch, layer, patch, head = cached

            callback({
                "stage": stage,
                "layer_idx": self._router_trace_layer_idx,
                "batch": batch,
                "layer": layer,
                "patch": patch,
                "head": head,
                "expert": expert,
                "top1_prob": top1_prob,
                "margin": margin,
                "key": key.detach(),
                "value": value.detach(),
                "key_is_unit_normalized": bool(self.l2_norm),
                "value_is_unit_normalized": False,
            })

    def _shared_expert_ffn(self, x, sw0, sw1, sw2):
        """SwiGLU FFN for the shared expert (no routing).
        x: [B, T, D], sw*: [B, 1, ...]  -> [B, T, D]
        """
        sw0 = sw0.squeeze(1)  # [B, inter_dim, D]
        sw1 = sw1.squeeze(1)  # [B, D, inter_dim]
        sw2 = sw2.squeeze(1)  # [B, inter_dim, D]
        gate = torch.bmm(x, sw0.transpose(1, 2))   # [B, T, inter_dim]
        up = torch.bmm(x, sw2.transpose(1, 2))     # [B, T, inter_dim]
        hidden = F.silu(gate) * up
        return torch.bmm(hidden, sw1.transpose(1, 2))  # [B, T, D]

    def forward(self, input, fastw, router_proj_weights, se_fastw=None):
        w0, w1, w2 = fastw

        x = self.input_proj(input)
        # [B, L, H*D] -> [B, L*H, D]
        x = rearrange(x, "b l (h d) -> b (l h) d", h=self.num_heads)
        if self.l2_norm:
            x = F.normalize(x, dim=-1, eps=1e-5).to(x.dtype)
        else:
            x = self.input_rms_norm(x)

        with torch.autocast(device_type="cuda", enabled=False):
            # float[E, D] @ float[B, D, L*H] -> float[B, E, L*H]
            logits = torch.matmul(router_proj_weights.float(), x.transpose(1, 2).float())

        # [B, E, L*H] --> [B, E], [B, E]
        expert_mask, group_sizes, router_probs, lb_loss = create_router_mask_sizes_probs(
            logits, topk=self.num_active_experts, alpha=self.router_alpha, use_sigmoid=self.sigmoid_router
        )
        router_probs_sum_b_e = router_probs.sum(dim=-1)

        # [B, L*H, D], [B, E, L*H], [B, E, L*H]
        # -> [B, num_activated, D], [B, T, 2*num_experts+1]
        q_permuted, _, q_row_id_map = permute_with_expert_mask(
            x, expert_mask, router_probs, self.num_active_experts, None
        )

        w0_w2 = torch.cat([w0, w2], dim=2)
        o_permuted = grouped_swiglu_ffn_fwd(w0_w2, w1, q_permuted, group_sizes)
        o = unpermute_and_merge_with_probs(o_permuted, q_row_id_map, router_probs)

        if self.use_shared_expert and se_fastw is not None:
            sw0, sw1, sw2 = se_fastw
            o = o + self._shared_expert_ffn(x, sw0, sw1, sw2)

        # [B, L*H, D] -> [B, L, H*D]
        o = self.output_rms_norm(o)
        o = rearrange(o, "b (l h) d -> b l (h d)", h=self.num_heads, d=self.head_dim)
        o = self.output_proj(o)

        if debug_utils.log_status:
            info = {
                "apply_load_balancing_loss": sketch_tensor(lb_loss),
            }

            # [B, E]
            # For each batch, the percentage of tokens routed to each expert
            expert_tokens_percentage = group_sizes.float() / group_sizes.float().sum(dim=1, keepdim=True)

            max_violation_rate = expert_tokens_percentage.max(dim=1)[0] * self.num_experts - 1.
            min_violation_rate = torch.abs(expert_tokens_percentage.min(dim=1)[0] * self.num_experts - 1.)

            info["apply_max_violation_rate"] = sketch_tensor(max_violation_rate)
            info["apply_min_violation_rate"] = sketch_tensor(min_violation_rate)

            for expert_id in range(self.num_experts):
                info[f"apply_expert_{expert_id}_tokens_percentage"] = (
                    sketch_tensor(expert_tokens_percentage[:, expert_id])
                )
        else:
            info = {}

        return o, (router_probs_sum_b_e, group_sizes), info

    def _shared_expert_grad(self, k, v, lr0, lr1, lr2, sw0, sw1, sw2):
        """Compute fast weight gradients for the shared expert (no routing).
        k, v: [B, T, D], lr*: [B, T, 1], sw*: [B, 1, ...]
        Returns dsw0, dsw1, dsw2 with shape [B, 1, ...]
        """
        sw0_s = sw0.squeeze(1)  # [B, inter_dim, D]
        sw1_s = sw1.squeeze(1)  # [B, D, inter_dim]
        sw2_s = sw2.squeeze(1)  # [B, inter_dim, D]

        # Forward: o = sw1 @ (silu(sw0 @ k) * (sw2 @ k))
        gate = torch.bmm(k, sw0_s.transpose(1, 2))   # [B, T, inter_dim]
        up = torch.bmm(k, sw2_s.transpose(1, 2))     # [B, T, inter_dim]
        gate_act = F.silu(gate)
        hidden = gate_act * up
        o = torch.bmm(hidden, sw1_s.transpose(1, 2))  # [B, T, D]

        err = o - v  # [B, T, D]

        d_sw1 = torch.bmm(err.transpose(1, 2), hidden)  # [B, D, inter_dim]
        d_hidden = torch.bmm(err, sw1_s)  # [B, T, inter_dim]
        d_gate_act = d_hidden * up
        d_up = d_hidden * gate_act

        sig = torch.sigmoid(gate)
        d_gate = d_gate_act * (sig * (1.0 + gate * (1.0 - sig)))

        d_sw0 = torch.bmm(d_gate.transpose(1, 2), k)  # [B, inter_dim, D]
        d_sw2 = torch.bmm(d_up.transpose(1, 2), k)    # [B, inter_dim, D]

        lr0_mean = lr0.mean(dim=1, keepdim=True).squeeze(1).unsqueeze(1)  # [B, 1, 1]
        lr1_mean = lr1.mean(dim=1, keepdim=True).squeeze(1).unsqueeze(1)
        lr2_mean = lr2.mean(dim=1, keepdim=True).squeeze(1).unsqueeze(1)

        d_sw0 = (d_sw0 * lr0_mean).unsqueeze(1)  # [B, 1, inter_dim, D]
        d_sw1 = (d_sw1 * lr1_mean).unsqueeze(1)  # [B, 1, D, inter_dim]
        d_sw2 = (d_sw2 * lr2_mean).unsqueeze(1)  # [B, 1, inter_dim, D]

        return d_sw0, d_sw1, d_sw2

    def update(self, pre_k, pre_v, fastw, masterw, router_proj_weights,
               se_fastw=None, se_masterw=None, update_keep_mask=None):
        if update_keep_mask is not None and self.use_shared_expert:
            raise ValueError(
                "per-token update masking is unsupported when use_shared_expert=True"
            )
        w0_master, w1_master, w2_master = masterw
        w0, w1, w2 = fastw

        k = self.k_input_proj(pre_k)
        # [B, L, H*D] -> [B, L*H, D]
        k = rearrange(k, "b l (h d) -> b (l h) d", h=self.num_heads, d=self.head_dim)
        if self.l2_norm:
            k = F.normalize(k, dim=-1, eps=1e-5).to(k.dtype)
        else:
            k = self.input_rms_norm(k)

        # [B, L, H*D] -> [B, L*H, D]
        v = self.to_v(pre_v)
        v = rearrange(v, "b l (h d) -> b (l h) d", h=self.num_heads, d=self.head_dim)

        # [B, L, D] -> [B, L, 3 H]
        with torch.autocast(device_type=v.device.type, enabled=False):
            lr = self.to_lr(pre_v.float())
            lr = torch.nn.functional.softplus(lr + self.base_lr_inv)
            if update_keep_mask is not None:
                lr = lr * update_keep_mask[..., None].to(dtype=lr.dtype)
        # [B, L, H * 3] --> [B, L*H, 3]
        lr = rearrange(lr, "b l (h d) -> b (l h) d", h=self.num_heads, d=3)
        lr0, lr1, lr2 = lr.chunk(3, dim=2)  # [B, L*H, 1] each

        with torch.autocast(device_type="cuda", enabled=False):
            # float[E, D] @ float[B, D, L*H] -> float[B, E, L*H]
            router_input = v if self.router_from_v else k
            logits = torch.matmul(
                router_proj_weights.float(), router_input.transpose(1, 2).float()
            )

        expert_mask, group_sizes, router_probs, lb_loss = create_router_mask_sizes_probs(
            logits, topk=self.num_active_experts, alpha=self.router_alpha, use_sigmoid=self.sigmoid_router
        )
        self._emit_router_trace(
            expert_mask, router_probs, k, v, stage="update"
        )
        router_probs_sum_b_e = router_probs.sum(dim=-1)

        k_permuted, v_permuted, lr0_p, lr1_p, lr2_p = permute_kv_and_lrs(
            k, v, lr0, lr1, lr2, expert_mask, router_probs, self.num_active_experts
        )

        # [B, E, d_inter, d_head] --> [B, E, d_inter + d_inter, d_head]
        w0_w2 = torch.cat([w0, w2], dim=2)
        dw0_w2, dw1 = grouped_lact_swiglu_ffn_fast_weight_grads(
            w0_w2, w1, k_permuted, v_permuted, lr0_p, lr1_p, lr2_p, group_sizes
        )
        dw0 = dw0_w2[:, :, :self.inter_dim, :]
        dw2 = dw0_w2[:, :, self.inter_dim:, :]

        if self.use_muon:
            b, e, dh, din = dw0.shape
            dw0 = zeropower_via_newtonschulz5(dw0.flatten(0, 1)).view(b, e, dh, din)
            dw1 = zeropower_via_newtonschulz5(dw1.flatten(0, 1)).view(b, e, din, dh)
            dw2 = zeropower_via_newtonschulz5(dw2.flatten(0, 1)).view(b, e, dh, din)

        w0_master = w0_master + dw0
        w1_master = w1_master + dw1
        w2_master = w2_master + dw2

        masterw = (w0_master, w1_master, w2_master)
        weight = (
            F.normalize(w0_master, dim=3, eps=1e-5).to(torch.bfloat16),
            F.normalize(w1_master, dim=3, eps=1e-5).to(torch.bfloat16),
            F.normalize(w2_master, dim=3, eps=1e-5).to(torch.bfloat16),
        )

        # Shared expert update (independent per-layer weights)
        new_se_fastw, new_se_masterw = None, None
        if self.use_shared_expert and se_fastw is not None:
            sw0, sw1, sw2 = se_fastw
            sw0_master, sw1_master, sw2_master = se_masterw
            dsw0, dsw1, dsw2 = self._shared_expert_grad(k, v, lr0, lr1, lr2, sw0, sw1, sw2)
            if self.use_muon:
                dsw0 = zeropower_via_newtonschulz5(dsw0.flatten(0, 1)).view(dsw0.shape)
                dsw1 = zeropower_via_newtonschulz5(dsw1.flatten(0, 1)).view(dsw1.shape)
                dsw2 = zeropower_via_newtonschulz5(dsw2.flatten(0, 1)).view(dsw2.shape)
            sw0_master = sw0_master + dsw0
            sw1_master = sw1_master + dsw1
            sw2_master = sw2_master + dsw2
            new_se_masterw = (sw0_master, sw1_master, sw2_master)
            new_se_fastw = (
                F.normalize(sw0_master, dim=3, eps=1e-5).to(torch.bfloat16),
                F.normalize(sw1_master, dim=3, eps=1e-5).to(torch.bfloat16),
                F.normalize(sw2_master, dim=3, eps=1e-5).to(torch.bfloat16),
            )

        if debug_utils.log_status:
            info = {
                "update_load_balancing_loss": sketch_tensor(lb_loss),
            }

            # [B, E]
            # For each batch, the percentage of tokens routed to each expert
            expert_tokens_percentage = group_sizes.float() / group_sizes.float().sum(dim=1, keepdim=True)

            max_violation_rate = expert_tokens_percentage.max(dim=1)[0] * self.num_experts - 1.
            min_violation_rate = torch.abs(expert_tokens_percentage.min(dim=1)[0] * self.num_experts - 1.)

            info["update_max_violation_rate"] = sketch_tensor(max_violation_rate)
            info["update_min_violation_rate"] = sketch_tensor(min_violation_rate)

            for expert_id in range(self.num_experts):
                info[f"update_expert_{expert_id}_tokens_percentage"] = (
                    sketch_tensor(expert_tokens_percentage[:, expert_id])
                )

        else:
            info = {}

        return weight, masterw, new_se_fastw, new_se_masterw, (router_probs_sum_b_e, group_sizes), info

    def compute_raw_grad(self, pre_k, pre_v, fastw, router_proj_weights,
                         se_fastw=None, update_keep_mask=None):
        """Compute raw dw0/dw1/dw2 from the given fastw snapshot.
        Does NOT apply Muon, does NOT write masterw, does NOT normalize.
        Returns (raw_grads, lb_info, info, se_raw_grads).
        """
        if update_keep_mask is not None and self.use_shared_expert:
            raise ValueError(
                "per-token update masking is unsupported when use_shared_expert=True"
            )
        w0, w1, w2 = fastw

        k = self.k_input_proj(pre_k)
        k = rearrange(k, "b l (h d) -> b (l h) d", h=self.num_heads, d=self.head_dim)
        if self.l2_norm:
            k = F.normalize(k, dim=-1, eps=1e-5).to(k.dtype)
        else:
            k = self.input_rms_norm(k)

        v = self.to_v(pre_v)
        v = rearrange(v, "b l (h d) -> b (l h) d", h=self.num_heads, d=self.head_dim)

        with torch.autocast(device_type=v.device.type, enabled=False):
            lr = self.to_lr(pre_v.float())
            lr = torch.nn.functional.softplus(lr + self.base_lr_inv)
            if update_keep_mask is not None:
                lr = lr * update_keep_mask[..., None].to(dtype=lr.dtype)
        lr = rearrange(lr, "b l (h d) -> b (l h) d", h=self.num_heads, d=3)
        lr0, lr1, lr2 = lr.chunk(3, dim=2)

        with torch.autocast(device_type="cuda", enabled=False):
            router_input = v if self.router_from_v else k
            logits = torch.matmul(
                router_proj_weights.float(), router_input.transpose(1, 2).float()
            )

        expert_mask, group_sizes, router_probs, lb_loss = create_router_mask_sizes_probs(
            logits, topk=self.num_active_experts, alpha=self.router_alpha, use_sigmoid=self.sigmoid_router
        )
        self._emit_router_trace(
            expert_mask, router_probs, k, v, stage="compute_raw_grad"
        )
        router_probs_sum_b_e = router_probs.sum(dim=-1)

        k_permuted, v_permuted, lr0_p, lr1_p, lr2_p = permute_kv_and_lrs(
            k, v, lr0, lr1, lr2, expert_mask, router_probs, self.num_active_experts
        )

        w0_w2 = torch.cat([w0, w2], dim=2)
        dw0_w2, dw1 = grouped_lact_swiglu_ffn_fast_weight_grads(
            w0_w2, w1, k_permuted, v_permuted, lr0_p, lr1_p, lr2_p, group_sizes
        )
        dw0 = dw0_w2[:, :, :self.inter_dim, :]
        dw2 = dw0_w2[:, :, self.inter_dim:, :]

        se_raw_grads = None
        if self.use_shared_expert and se_fastw is not None:
            sw0, sw1, sw2 = se_fastw
            dsw0, dsw1, dsw2 = self._shared_expert_grad(k, v, lr0, lr1, lr2, sw0, sw1, sw2)
            se_raw_grads = (dsw0.float(), dsw1.float(), dsw2.float())

        if debug_utils.log_status:
            info = {
                "update_load_balancing_loss": sketch_tensor(lb_loss),
            }
            expert_tokens_percentage = group_sizes.float() / group_sizes.float().sum(dim=1, keepdim=True)
            max_violation_rate = expert_tokens_percentage.max(dim=1)[0] * self.num_experts - 1.
            min_violation_rate = torch.abs(expert_tokens_percentage.min(dim=1)[0] * self.num_experts - 1.)
            info["update_max_violation_rate"] = sketch_tensor(max_violation_rate)
            info["update_min_violation_rate"] = sketch_tensor(min_violation_rate)
            for expert_id in range(self.num_experts):
                info[f"update_expert_{expert_id}_tokens_percentage"] = (
                    sketch_tensor(expert_tokens_percentage[:, expert_id])
                )
        else:
            info = {}

        return (dw0.float(), dw1.float(), dw2.float()), (router_probs_sum_b_e, group_sizes), info, se_raw_grads

    def apply_raw_update(self, raw_grads, masterw):
        """Apply Muon once on aggregated raw grads, write to masterw, normalize.
        Works for both routed weights (E=num_experts) and shared expert (E=1).
        """
        dw0, dw1, dw2 = raw_grads

        if self.use_muon:
            b, e, dh, din = dw0.shape
            dw0 = zeropower_via_newtonschulz5(dw0.flatten(0, 1)).view(b, e, dh, din)
            dw1 = zeropower_via_newtonschulz5(dw1.flatten(0, 1)).view(b, e, din, dh)
            dw2 = zeropower_via_newtonschulz5(dw2.flatten(0, 1)).view(b, e, dh, din)

        w0_master, w1_master, w2_master = masterw
        w0_master = w0_master + dw0
        w1_master = w1_master + dw1
        w2_master = w2_master + dw2

        new_masterw = (w0_master, w1_master, w2_master)
        new_fastw = (
            F.normalize(w0_master, dim=3, eps=1e-5).to(torch.bfloat16),
            F.normalize(w1_master, dim=3, eps=1e-5).to(torch.bfloat16),
            F.normalize(w2_master, dim=3, eps=1e-5).to(torch.bfloat16),
        )
        return new_fastw, new_masterw

    def extra_repr(self) -> str:
        return (
            f"fw_head_dim: {self.head_dim}, num_heads: {self.num_heads}, inter_dim: {self.inter_dim}, "
            f"num_experts: {self.num_experts}, num_active_experts: {self.num_active_experts}, use_muon: {self.use_muon}, l2_norm: {self.l2_norm}, \n"
            f"router_from_v: {self.router_from_v}, router_alpha: {self.router_alpha}, sigmoid_router: {self.sigmoid_router}, \n"
            f"use_shared_expert: {self.use_shared_expert}, (pool weights and router provided by MemoryModel)"
        )


class MemoryBlock(nn.Module):
    def __init__(
        self,
        dim,
        attn_head_dim=64,
        inter_multi=2,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim

        self.ln_attn = nn.RMSNorm(dim, eps=1e-5)
        self.self_attention = SelfAttention(dim, attn_head_dim)

        self.ln_memory = nn.RMSNorm(dim, eps=1e-5)
        self.memory = FastWeightMLPSubNN(
            dim,
            **kwargs
        )

        self.ln_mlp = nn.RMSNorm(dim, eps=1e-5)
        self.mlp = MLP(dim, inter_multi)

    def _forward(self, x, fastw, router_proj_weights, se_fastw=None):
        skip = x
        x = self.ln_attn(x)
        x = skip + self.self_attention(x)

        skip = x
        x_memory_in = x = self.ln_memory(x)
        x, lb_info, info = self.memory(x, fastw, router_proj_weights, se_fastw=se_fastw)
        x = skip + x

        skip = x
        x = self.ln_mlp(x)
        x = skip + self.mlp(x)

        return x, x_memory_in, lb_info, info

    # @torch.compile
    def forward(self, *args, **kwargs):
        return checkpoint(self._forward, *args, **kwargs, preserve_rng_state=False, use_reentrant=False)

    def _update_fast_weight(self, memory_in, pre_vi, fastw, masterw, router_proj_weights,
                            se_fastw=None, se_masterw=None, update_keep_mask=None):
        pre_vi = F.rms_norm(pre_vi, normalized_shape=(self.dim,), eps=1e-5)
        if update_keep_mask is None:
            fastw, masterw, se_fastw, se_masterw, lb_info, info = self.memory.update(
                memory_in, pre_vi, fastw, masterw, router_proj_weights,
                se_fastw=se_fastw, se_masterw=se_masterw,
            )
        else:
            fastw, masterw, se_fastw, se_masterw, lb_info, info = self.memory.update(
                memory_in, pre_vi, fastw, masterw, router_proj_weights,
                se_fastw=se_fastw, se_masterw=se_masterw,
                update_keep_mask=update_keep_mask,
            )
        return fastw, masterw, se_fastw, se_masterw, lb_info, info

    # @torch.compile
    def update_fast_weight(self, *args, **kwargs):
        return checkpoint(
            self._update_fast_weight,
            *args,
            **kwargs,
            preserve_rng_state=False,
            use_reentrant=False,
        )

    def _compute_raw_grad_fast_weight(self, memory_in, pre_vi, fastw, router_proj_weights,
                                      se_fastw=None, update_keep_mask=None):
        pre_vi = F.rms_norm(pre_vi, normalized_shape=(self.dim,), eps=1e-5)
        if update_keep_mask is None:
            raw_grads, lb_info, info, se_raw_grads = self.memory.compute_raw_grad(
                memory_in, pre_vi, fastw, router_proj_weights, se_fastw=se_fastw,
            )
        else:
            raw_grads, lb_info, info, se_raw_grads = self.memory.compute_raw_grad(
                memory_in, pre_vi, fastw, router_proj_weights, se_fastw=se_fastw,
                update_keep_mask=update_keep_mask,
            )
        return raw_grads, lb_info, info, se_raw_grads

    def compute_raw_grad_fast_weight(self, *args, **kwargs):
        return checkpoint(
            self._compute_raw_grad_fast_weight,
            *args,
            **kwargs,
            preserve_rng_state=False,
            use_reentrant=False,
        )


class MemoryModel(nn.Module):
    def __init__(
        self,
        layers,
        dim,
        v_gap=None,
        load_balancing_loss_alpha=0.01,
        num_experts=2,
        fw_head_dim=None,
        fw_inter_multi=1,
        pool_mode="shared",
        router_share_mode="per_layer",
        use_shared_expert=False,
        shared_expert_per_layer=False,
        aggregate_write=False,
        **kwargs
    ):
        super().__init__()
        object.__setattr__(self, "_update_mask_callback", None)
        self.v_gap = v_gap
        self.load_balancing_loss_alpha = load_balancing_loss_alpha
        self.pool_mode = pool_mode
        self.router_share_mode = router_share_mode
        self.use_shared_expert = use_shared_expert
        # shared_expert_per_layer=True: each layer has its own shared expert (avoids gradient conflict)
        # shared_expert_per_layer=False: shared expert follows pool_mode (shared pool → 1 set, per_block → per-layer)
        self.shared_expert_per_layer = shared_expert_per_layer
        # aggregate_write: all layers compute raw grad from same W_N, sum, apply Muon+write once
        self.aggregate_write = aggregate_write

        head_dim = fw_head_dim if fw_head_dim is not None else dim
        inter_dim = int(head_dim * fw_inter_multi)

        # Expert weights
        if pool_mode == "shared":
            # All layers share one pool
            self.pool_w0 = nn.Parameter(
                torch.randn(1, num_experts, inter_dim, head_dim) / math.sqrt(head_dim)
            )
            self.pool_w1 = nn.Parameter(
                torch.randn(1, num_experts, head_dim, inter_dim) / math.sqrt(inter_dim)
            )
            self.pool_w2 = nn.Parameter(
                torch.randn(1, num_experts, inter_dim, head_dim) / math.sqrt(head_dim)
            )
        elif pool_mode == "per_block":
            # Each block owns its own experts
            self.block_w0_list = nn.ParameterList([
                nn.Parameter(torch.randn(1, num_experts, inter_dim, head_dim) / math.sqrt(head_dim))
                for _ in range(layers)
            ])
            self.block_w1_list = nn.ParameterList([
                nn.Parameter(torch.randn(1, num_experts, head_dim, inter_dim) / math.sqrt(inter_dim))
                for _ in range(layers)
            ])
            self.block_w2_list = nn.ParameterList([
                nn.Parameter(torch.randn(1, num_experts, inter_dim, head_dim) / math.sqrt(head_dim))
                for _ in range(layers)
            ])
        else:
            raise ValueError(f"Invalid pool_mode: {pool_mode}")

        # Shared expert weights
        # shared_expert_per_layer=True → always per-layer (recommended for shared pool)
        # shared_expert_per_layer=False → follows pool_mode (shared pool → 1 set, per_block → per-layer)
        self._se_is_per_layer = False
        if use_shared_expert:
            if shared_expert_per_layer or pool_mode == "per_block":
                self._se_is_per_layer = True
                self.shared_expert_w0_list = nn.ParameterList([
                    nn.Parameter(torch.randn(1, 1, inter_dim, head_dim) / math.sqrt(head_dim))
                    for _ in range(layers)
                ])
                self.shared_expert_w1_list = nn.ParameterList([
                    nn.Parameter(torch.randn(1, 1, head_dim, inter_dim) / math.sqrt(inter_dim))
                    for _ in range(layers)
                ])
                self.shared_expert_w2_list = nn.ParameterList([
                    nn.Parameter(torch.randn(1, 1, inter_dim, head_dim) / math.sqrt(head_dim))
                    for _ in range(layers)
                ])
            else:
                # Single shared expert across all layers (old behavior)
                self.shared_expert_w0 = nn.Parameter(
                    torch.randn(1, 1, inter_dim, head_dim) / math.sqrt(head_dim)
                )
                self.shared_expert_w1 = nn.Parameter(
                    torch.randn(1, 1, head_dim, inter_dim) / math.sqrt(inter_dim)
                )
                self.shared_expert_w2 = nn.Parameter(
                    torch.randn(1, 1, inter_dim, head_dim) / math.sqrt(head_dim)
                )

        # Modification 1: fail-fast checks moved from forward() to __init__
        if aggregate_write and pool_mode != "shared":
            raise ValueError("aggregate_write=True is only supported with pool_mode='shared'.")
        if aggregate_write and use_shared_expert and not self._se_is_per_layer:
            raise ValueError(
                "aggregate_write=True requires shared_expert_per_layer=True "
                "when use_shared_expert=True."
            )

        # Router(s)
        if router_share_mode == "per_layer":
            self.router_proj_weights_list = nn.ParameterList([
                nn.Parameter(
                    torch.randn(1, num_experts, head_dim) / math.sqrt(head_dim)
                )
                for _ in range(layers)
            ])
        elif router_share_mode == "share_all":
            self.shared_router_proj_weights = nn.Parameter(
                torch.randn(1, num_experts, head_dim) / math.sqrt(head_dim)
            )
        else:
            raise ValueError(f"Invalid router_share_mode: {router_share_mode}")

        # Blocks (weights are managed by MemoryModel, not by blocks)
        block_kwargs = dict(
            num_experts=num_experts,
            fw_head_dim=fw_head_dim,
            fw_inter_multi=fw_inter_multi,
            use_shared_expert=use_shared_expert,
            **kwargs,
        )
        self.blocks = []
        self.v_layer_idxs = []
        for i in range(layers):
            if v_gap is not None:
                self.v_layer_idxs.append(max(min(i + v_gap, layers - 1), 0))
            block = MemoryBlock(dim=dim, **block_kwargs)
            block.memory.set_router_trace_callback(None, layer_idx=i)
            self.blocks.append(block)
        self.blocks = nn.ModuleList(self.blocks)

    def set_router_trace_callback(self, callback):
        """Enable or disable update-router tracing for every memory layer."""
        for layer_idx, block in enumerate(self.blocks):
            block.memory.set_router_trace_callback(callback, layer_idx=layer_idx)

    def set_update_mask_callback(self, callback):
        """Set an optional per-source-view callback returning a [B, L] keep mask."""
        if callback is not None and self.use_shared_expert:
            raise ValueError(
                "per-token update masking is unsupported when use_shared_expert=True"
            )
        object.__setattr__(self, "_update_mask_callback", callback)

    def _resolve_update_keep_mask(self, source_view, start, end, x, stage):
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
            stage=stage,
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

    def _get_router(self, layer_idx):
        if self.router_share_mode == "per_layer":
            return self.router_proj_weights_list[layer_idx]
        return self.shared_router_proj_weights

    @staticmethod
    def _norm_bf16(w):
        return F.normalize(w, dim=3, eps=1e-5).to(torch.bfloat16)

    @staticmethod
    def _make_routed_fastw(masterw):
        """Normalize routed expert weights (always a 3-tuple)."""
        return (
            F.normalize(masterw[0], dim=3, eps=1e-5).to(torch.bfloat16),
            F.normalize(masterw[1], dim=3, eps=1e-5).to(torch.bfloat16),
            F.normalize(masterw[2], dim=3, eps=1e-5).to(torch.bfloat16),
        )

    @staticmethod
    def _make_se_fastw(se_masterw):
        """Normalize shared expert weights (always a 3-tuple)."""
        return (
            F.normalize(se_masterw[0], dim=3, eps=1e-5).to(torch.bfloat16),
            F.normalize(se_masterw[1], dim=3, eps=1e-5).to(torch.bfloat16),
            F.normalize(se_masterw[2], dim=3, eps=1e-5).to(torch.bfloat16),
        )

    def _init_weights(self, batch_size):
        """Returns:
            routed: (fastw, masterw) — shared across layers if pool_mode="shared", per-layer list otherwise
            shared_expert: list of (se_fastw_i, se_masterw_i) per layer, or None
        """
        # Routed experts
        if self.pool_mode == "shared":
            routed_masterw = (
                self.pool_w0.repeat(batch_size, 1, 1, 1),
                self.pool_w1.repeat(batch_size, 1, 1, 1),
                self.pool_w2.repeat(batch_size, 1, 1, 1),
            )
            routed_fastw = self._make_routed_fastw(routed_masterw)
            routed = (routed_fastw, routed_masterw)
        else:
            routed_fastw_list, routed_masterw_list = [], []
            for i in range(len(self.blocks)):
                mw = (
                    self.block_w0_list[i].repeat(batch_size, 1, 1, 1),
                    self.block_w1_list[i].repeat(batch_size, 1, 1, 1),
                    self.block_w2_list[i].repeat(batch_size, 1, 1, 1),
                )
                routed_fastw_list.append(self._make_routed_fastw(mw))
                routed_masterw_list.append(mw)
            routed = (routed_fastw_list, routed_masterw_list)

        # Shared expert
        se_weights = None
        if self.use_shared_expert:
            if self._se_is_per_layer:
                se_weights = []
                for i in range(len(self.blocks)):
                    se_mw = (
                        self.shared_expert_w0_list[i].repeat(batch_size, 1, 1, 1),
                        self.shared_expert_w1_list[i].repeat(batch_size, 1, 1, 1),
                        self.shared_expert_w2_list[i].repeat(batch_size, 1, 1, 1),
                    )
                    se_weights.append((self._make_se_fastw(se_mw), se_mw))
            else:
                # Single shared expert for all layers (old behavior)
                se_mw = (
                    self.shared_expert_w0.repeat(batch_size, 1, 1, 1),
                    self.shared_expert_w1.repeat(batch_size, 1, 1, 1),
                    self.shared_expert_w2.repeat(batch_size, 1, 1, 1),
                )
                se_fw = self._make_se_fastw(se_mw)
                # Wrap as list with same reference for all layers
                se_weights = [(se_fw, se_mw)] * len(self.blocks)

        return routed, se_weights

    def forward(self, x, info_dict):
        B = x.shape[0]
        device = x.device
        dtype = torch.float32
        is_shared = self.pool_mode == "shared"

        # Init weights: routed (shared or per-layer) + shared expert (always per-layer)
        routed, se_weights = self._init_weights(B)
        if is_shared:
            routed_fastw, routed_masterw = routed
        else:
            routed_fastw_list, routed_masterw_list = routed

        # Per-layer caches (for xi_out, xi_memory_in, pre_vi)
        caches = [{} for _ in self.blocks]

        outputs = []
        infos = []
        source_view = 0

        num_experts = self.blocks[0].memory.num_experts

        # [Layers, B, E]
        apply_accum_probs = torch.zeros(len(self.blocks), B, num_experts, device=device, dtype=dtype)
        apply_accum_freqs = torch.zeros(len(self.blocks), B, num_experts, device=device, dtype=dtype)
        update_accum_probs = torch.zeros(len(self.blocks), B, num_experts, device=device, dtype=dtype)
        update_accum_freqs = torch.zeros(len(self.blocks), B, num_experts, device=device, dtype=dtype)

        for start, end, update, _, _ in info_dict["ttt_config"]:
            op_info = []

            xi = x[:, start:end, :]

            # Forward
            for i, block in enumerate(self.blocks):
                router_w = self._get_router(i)
                fastw_i = routed_fastw if is_shared else routed_fastw_list[i]
                se_fastw_i = se_weights[i][0] if se_weights is not None else None

                xi, xi_memory_in, apply_lb_info, info = block.forward(
                    xi, fastw_i, router_w, se_fastw=se_fastw_i,
                )

                apply_probs_chunk, apply_freqs_chunk = apply_lb_info
                apply_accum_probs[i] += apply_probs_chunk.float()
                apply_accum_freqs[i] += apply_freqs_chunk.float()

                caches[i].update({"xi_out": xi, "xi_memory_in": xi_memory_in})
                if info:
                    op_info.append(info)

            if update:
                update_stage = "compute_raw_grad" if is_shared and self.aggregate_write else "update"
                update_keep_mask = self._resolve_update_keep_mask(
                    source_view, start, end, x, update_stage
                )
                for i, cache in enumerate(caches):
                    if self.v_gap is not None:
                        cache["pre_vi"] = caches[self.v_layer_idxs[i]]["xi_out"]
                    else:
                        cache["pre_vi"] = cache["xi_memory_in"]

                if is_shared and self.aggregate_write:
                    agg_dw0 = torch.zeros_like(routed_masterw[0])
                    agg_dw1 = torch.zeros_like(routed_masterw[1])
                    agg_dw2 = torch.zeros_like(routed_masterw[2])

                    for i, block in enumerate(self.blocks):
                        cache = caches[i]
                        router_w = self._get_router(i)
                        se_fw_i = se_weights[i][0] if se_weights is not None else None

                        # All layers compute raw grad from the same W_N snapshot (routed_fastw).
                        if update_keep_mask is None:
                            raw_grads, update_lb_info, info, se_raw = block.compute_raw_grad_fast_weight(
                                cache["xi_memory_in"],
                                cache["pre_vi"],
                                routed_fastw,
                                router_w,
                                se_fastw=se_fw_i,
                            )
                        else:
                            raw_grads, update_lb_info, info, se_raw = block.compute_raw_grad_fast_weight(
                                cache["xi_memory_in"],
                                cache["pre_vi"],
                                routed_fastw,
                                router_w,
                                se_fastw=se_fw_i,
                                update_keep_mask=update_keep_mask,
                            )

                        agg_dw0 = agg_dw0 + raw_grads[0]
                        agg_dw1 = agg_dw1 + raw_grads[1]
                        agg_dw2 = agg_dw2 + raw_grads[2]

                        update_probs_chunk, update_freqs_chunk = update_lb_info
                        update_accum_probs[i] += update_probs_chunk.float()
                        update_accum_freqs[i] += update_freqs_chunk.float()

                        if info and i < len(op_info):
                            op_info[i].update(info)

                        # Per-layer shared expert update (independent weights, update immediately).
                        if se_weights is not None and se_raw is not None:
                            se_mw_i = se_weights[i][1]
                            new_se_fw, new_se_mw = self.blocks[0].memory.apply_raw_update(
                                se_raw, se_mw_i
                            )
                            se_weights[i] = (new_se_fw, new_se_mw)

                    # Muon + write + normalize once on the aggregated grad.
                    routed_fastw, routed_masterw = self.blocks[0].memory.apply_raw_update(
                        (agg_dw0, agg_dw1, agg_dw2),
                        routed_masterw,
                    )
                else:
                    for i, block in enumerate(self.blocks):
                        cache = caches[i]
                        router_w = self._get_router(i)

                        if is_shared:
                            fastw_i, masterw_i = routed_fastw, routed_masterw
                        else:
                            fastw_i, masterw_i = routed_fastw_list[i], routed_masterw_list[i]

                        se_fw_i = se_weights[i][0] if se_weights is not None else None
                        se_mw_i = se_weights[i][1] if se_weights is not None else None

                        if update_keep_mask is None:
                            new_fastw, new_masterw, new_se_fw, new_se_mw, update_lb_info, info = block.update_fast_weight(
                                cache["xi_memory_in"],
                                cache["pre_vi"],
                                fastw_i,
                                masterw_i,
                                router_w,
                                se_fastw=se_fw_i,
                                se_masterw=se_mw_i,
                            )
                        else:
                            new_fastw, new_masterw, new_se_fw, new_se_mw, update_lb_info, info = block.update_fast_weight(
                                cache["xi_memory_in"],
                                cache["pre_vi"],
                                fastw_i,
                                masterw_i,
                                router_w,
                                se_fastw=se_fw_i,
                                se_masterw=se_mw_i,
                                update_keep_mask=update_keep_mask,
                            )

                        update_probs_chunk, update_freqs_chunk = update_lb_info
                        update_accum_probs[i] += update_probs_chunk.float()
                        update_accum_freqs[i] += update_freqs_chunk.float()

                        if is_shared:
                            routed_fastw, routed_masterw = new_fastw, new_masterw
                        else:
                            routed_fastw_list[i], routed_masterw_list[i] = new_fastw, new_masterw

                        if se_weights is not None and new_se_fw is not None:
                            if self._se_is_per_layer:
                                se_weights[i] = (new_se_fw, new_se_mw)
                            else:
                                # Old behavior: shared expert is shared across layers
                                # Propagate update to all layers
                                for j in range(len(se_weights)):
                                    se_weights[j] = (new_se_fw, new_se_mw)

                        if info and i < len(op_info):
                            op_info[i].update(info)
                source_view += 1

            outputs.append(caches[-1]["xi_out"])
            infos.append(op_info)

        outputs = torch.cat(outputs, dim=1)

        # Calculate Global LB Loss
        # Sum over batches to get [layer, E]
        apply_total_probs = apply_accum_probs.sum(dim=1)
        apply_total_freqs = apply_accum_freqs.sum(dim=1)

        # Average over E to get distributions [layer, E]
        apply_probs_mean = apply_total_probs / (apply_total_probs.sum(dim=1, keepdim=True))
        apply_freqs_mean = apply_total_freqs / (apply_total_freqs.sum(dim=1, keepdim=True))

        # [layer, E] * [layer, E] -> sum(E) -> [layer] -> sum(layer) -> scalar
        apply_lb_loss = ((apply_probs_mean * apply_freqs_mean).sum(dim=1) * num_experts).sum()

        # Calculate Global LB Loss for update
        # Sum over batches to get [layer, E]
        update_total_probs = update_accum_probs.sum(dim=1)
        update_total_freqs = update_accum_freqs.sum(dim=1)

        # Average over E to get distributions [layer, E]
        update_probs_mean = update_total_probs / (update_total_probs.sum(dim=1, keepdim=True))
        update_freqs_mean = update_total_freqs / (update_total_freqs.sum(dim=1, keepdim=True))
        update_lb_loss = ((update_probs_mean * update_freqs_mean).sum(dim=1) * num_experts).sum()

        # Calculate Global Violation
        def compute_violation(accum_freqs):
            # accum_freqs: [Layers, B, E]
            accum_freqs = accum_freqs.float()
            percentages = accum_freqs / accum_freqs.sum(dim=-1, keepdim=True)

            max_v = percentages.max(dim=-1)[0] * num_experts - 1.0
            min_v = torch.abs(percentages.min(dim=-1)[0] * num_experts - 1.0)

            return max_v.mean(), min_v.mean()

        apply_max_v, apply_min_v = compute_violation(apply_accum_freqs)
        update_max_v, update_min_v = compute_violation(update_accum_freqs)

        def compute_freq_extrema(accum_freqs):
            freq = accum_freqs.float().sum(dim=1)
            if self.pool_mode == "shared":
                freq = freq.sum(dim=0)
            else:
                freq = freq.flatten()
            freq = freq / freq.sum().clamp_min(1e-12)
            return freq.max(), freq.min()

        moe_freq_max, moe_freq_min = compute_freq_extrema(
            apply_accum_freqs + update_accum_freqs
        )

        lb_loss = (apply_lb_loss + update_lb_loss) * self.load_balancing_loss_alpha
        infos.append(
            {
                "apply_max_violation_rate": apply_max_v,
                "apply_min_violation_rate": apply_min_v,
                "update_max_violation_rate": update_max_v,
                "update_min_violation_rate": update_min_v,
                "moe_freq_max": moe_freq_max,
                "moe_freq_min": moe_freq_min,
            }
        )
        return outputs, lb_loss, infos

    def extra_repr(self) -> str:
        lines = [
            f"pool_mode: {self.pool_mode}, aggregate_write: {self.aggregate_write}",
            f"router_share_mode: {self.router_share_mode}",
            f"use_shared_expert: {self.use_shared_expert}, shared_expert_per_layer: {self.shared_expert_per_layer}",
        ]
        if self.pool_mode == "shared":
            lines.append(f"pool_w0: {self.pool_w0.shape}, pool_w1: {self.pool_w1.shape}, pool_w2: {self.pool_w2.shape}")
        else:
            lines.append(f"block_w0_list: {len(self.block_w0_list)} x {self.block_w0_list[0].shape}")
        lines.append(f"load_balancing_loss_alpha: {self.load_balancing_loss_alpha} + Use global lb loss")
        lines.extend(
            f"vmap: {i} <- {v_layer_idx}"
            for i, v_layer_idx in enumerate(self.v_layer_idxs)
        )
        return "\n".join(lines)


def _unit_test_global_moe_v2():
    # python -m uttt_nvs.models.uttt_moe_block
    torch.manual_seed(0)
    if not torch.cuda.is_available():
        print("CUDA not available; skipping MoFW+MoE triton unit test.")
        return {"skipped": True}

    b, l, dim = 16, 1024, 512
    num_experts = 4
    layers = 4
    num_img_tokens = 256

    ttt_config = []
    for i in range(l // num_img_tokens):
        start = i * num_img_tokens
        end = start + num_img_tokens
        do_update = (i % 2 == 0)
        ttt_config.append((start, end, do_update, None, None))

    info_dict = {
        "num_img_tokens": num_img_tokens,
        "ttt_config": ttt_config,
    }

    common_kwargs = dict(
        layers=layers,
        dim=dim,
        fw_head_dim=128,
        attn_head_dim=64,
        inter_multi=3,
        fw_inter_multi=1,
        base_lr=0.01,
        num_experts=num_experts,
        num_active_experts=1,
        use_muon=True,
        sigmoid_router=False,
        router_alpha=1.0,
        load_balancing_loss_alpha=0.01,
        l2_norm=True,
    )

    configs = [
        {"pool_mode": "shared", "router_share_mode": "per_layer"},
        {"pool_mode": "shared", "router_share_mode": "share_all"},
        {"pool_mode": "per_block", "router_share_mode": "per_layer"},
        {"pool_mode": "per_block", "router_share_mode": "share_all"},
    ]

    for cfg in configs:
        label = f"pool_mode={cfg['pool_mode']}, router_share_mode={cfg['router_share_mode']}"
        print(f"\n{'='*60}")
        print(f"Testing: {label}")
        print(f"{'='*60}")

        torch.manual_seed(42)
        x = torch.randn(b, l, dim, device="cuda", dtype=torch.bfloat16)

        model = MemoryModel(**common_kwargs, **cfg).to("cuda")
        print(model)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs, lb_loss, infos = model(x, info_dict)
        print("x:", tuple(x.shape), x.dtype)
        print("outputs:", tuple(outputs.shape), outputs.dtype)
        print("lb_loss:", lb_loss.item())

        (outputs.sum() + lb_loss).backward()
        print(f"backward OK  [{label}]")

    # Test aggregate_write=True
    print(f"\n{'='*60}")
    print("Testing: aggregate_write=True (shared pool, per_layer router)")
    print(f"{'='*60}")

    torch.manual_seed(42)
    x_aw = torch.randn(b, l, dim, device="cuda", dtype=torch.bfloat16)
    model_aw = MemoryModel(
        **common_kwargs, pool_mode="shared", router_share_mode="per_layer", aggregate_write=True
    ).to("cuda")
    print(model_aw)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        outputs_aw, lb_loss_aw, _ = model_aw(x_aw, info_dict)
    print("outputs_aw:", tuple(outputs_aw.shape), outputs_aw.dtype)
    print("lb_loss_aw:", lb_loss_aw.item())
    (outputs_aw.sum() + lb_loss_aw).backward()
    print("backward OK [aggregate_write=True]")

    # Verify outputs differ from sequential shared (different update semantics)
    torch.manual_seed(42)
    model_seq = MemoryModel(
        **common_kwargs, pool_mode="shared", router_share_mode="per_layer", aggregate_write=False
    ).to("cuda")
    model_seq.load_state_dict(model_aw.state_dict())
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        outputs_seq, _, _ = model_seq(x_aw.detach(), info_dict)
    differ = not torch.allclose(outputs_aw.float().detach(), outputs_seq.float(), atol=1e-2)
    print(f"aggregate_write=True vs False outputs differ: {differ} (expected True)")
    assert differ, "aggregate_write=True/False should produce different outputs"

    # Verify per_block path is unaffected (aggregate_write=False is default)
    torch.manual_seed(42)
    x_pb = torch.randn(b, l, dim, device="cuda", dtype=torch.bfloat16)
    model_pb_new = MemoryModel(**common_kwargs, pool_mode="per_block", router_share_mode="per_layer").to("cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        outputs_pb_new, lb_loss_pb_new, _ = model_pb_new(x_pb, info_dict)
    (outputs_pb_new.sum() + lb_loss_pb_new).backward()
    print("backward OK [per_block unchanged]")

    # layers=1: aggregate_write=True must be identical to sequential (same single-layer operations)
    print(f"\n{'='*60}")
    print("Testing: layers=1 aggregate_write equivalence")
    print(f"{'='*60}")
    one_layer_kw = {**common_kwargs, "layers": 1}
    one_layer_info = {
        "num_img_tokens": num_img_tokens,
        "ttt_config": [
            (i * num_img_tokens, (i + 1) * num_img_tokens, i % 2 == 0, None, None)
            for i in range(l // num_img_tokens)
        ],
    }
    torch.manual_seed(77)
    x_1L = torch.randn(b, l, dim, device="cuda", dtype=torch.bfloat16)
    model_agg_1L = MemoryModel(**one_layer_kw, pool_mode="shared", aggregate_write=True).to("cuda")
    model_seq_1L = MemoryModel(**one_layer_kw, pool_mode="shared", aggregate_write=False).to("cuda")
    model_seq_1L.load_state_dict(model_agg_1L.state_dict())
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out_agg_1L, lb_agg_1L, _ = model_agg_1L(x_1L, one_layer_info)
        out_seq_1L, lb_seq_1L, _ = model_seq_1L(x_1L.detach(), one_layer_info)
    max_diff = (out_agg_1L.float() - out_seq_1L.float()).abs().max().item()
    assert max_diff < 1e-2, f"layers=1 aggregate_write should match sequential. Max diff: {max_diff}"
    print(f"layers=1: aggregate_write=True matches sequential (max_diff={max_diff:.2e})")

    return {"success": True}


if __name__ == "__main__":
    print(_unit_test_global_moe_v2())
