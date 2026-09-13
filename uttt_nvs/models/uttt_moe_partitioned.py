import math

import torch
import torch.nn.functional as F
from torch import nn

from . import uttt_moe_block as base


class MemoryModel(base.MemoryModel):
    """Global MoE with contiguous multi-pool cross-layer sharing.

    Compared with `memory_block_v2_global_moe.MemoryModel`, routed experts are
    shared within contiguous layer groups instead of across all layers.

    Router semantics:
    - `share_pool`: each pool owns one router shared by the layers in that pool
    - `share_all`: all pools reuse the same local router weights

    `num_experts` is interpreted as the total expert budget across pools.
    Each pool receives `num_experts // num_shared_pools` experts.
    """

    def __init__(
        self,
        layers,
        dim,
        v_gap=None,
        load_balancing_loss_alpha=0.01,
        num_experts=2,
        fw_head_dim=None,
        fw_inter_multi=1,
        num_shared_pools=1,
        pool_mode="shared",
        router_share_mode="share_pool",
        use_shared_expert=False,
        shared_expert_per_layer=False,
        aggregate_write=False,
        **kwargs,
    ):
        nn.Module.__init__(self)
        self.v_gap = v_gap
        self.load_balancing_loss_alpha = load_balancing_loss_alpha
        self.num_shared_pools = int(num_shared_pools)
        self.router_share_mode = router_share_mode
        self.use_shared_expert = use_shared_expert
        self.shared_expert_per_layer = shared_expert_per_layer
        self.aggregate_write = aggregate_write

        if self.num_shared_pools <= 0:
            raise ValueError("num_shared_pools must be positive.")
        if pool_mode != "shared":
            raise ValueError("multipool global_moe only supports pool_mode='shared'.")
        if layers % self.num_shared_pools != 0:
            raise ValueError(
                f"layers ({layers}) must be divisible by num_shared_pools ({self.num_shared_pools})."
            )
        if num_experts % self.num_shared_pools != 0:
            raise ValueError(
                f"num_experts ({num_experts}) must be divisible by num_shared_pools ({self.num_shared_pools})."
            )
        if router_share_mode not in {"share_pool", "share_all"}:
            raise ValueError(
                "router_share_mode must be either 'share_pool' or 'share_all'."
            )

        # _se_is_per_layer follows the same logic as base: True when shared_expert_per_layer=True
        # (pool_mode is always "shared" here, so the per_block branch never fires)
        _se_is_per_layer = use_shared_expert and shared_expert_per_layer
        if aggregate_write and use_shared_expert and not _se_is_per_layer:
            raise ValueError(
                "aggregate_write=True requires shared_expert_per_layer=True "
                "when use_shared_expert=True."
            )

        self.layers = layers
        self.num_experts = num_experts
        self.pool_size = layers // self.num_shared_pools
        self.experts_per_pool = num_experts // self.num_shared_pools
        self.layer_pool_idxs = [i // self.pool_size for i in range(layers)]

        head_dim = fw_head_dim if fw_head_dim is not None else dim
        inter_dim = int(head_dim * fw_inter_multi)

        self.pool_w0_list = nn.ParameterList(
            [
                nn.Parameter(
                    torch.randn(1, self.experts_per_pool, inter_dim, head_dim)
                    / math.sqrt(head_dim)
                )
                for _ in range(self.num_shared_pools)
            ]
        )
        self.pool_w1_list = nn.ParameterList(
            [
                nn.Parameter(
                    torch.randn(1, self.experts_per_pool, head_dim, inter_dim)
                    / math.sqrt(inter_dim)
                )
                for _ in range(self.num_shared_pools)
            ]
        )
        self.pool_w2_list = nn.ParameterList(
            [
                nn.Parameter(
                    torch.randn(1, self.experts_per_pool, inter_dim, head_dim)
                    / math.sqrt(head_dim)
                )
                for _ in range(self.num_shared_pools)
            ]
        )

        self._se_is_per_layer = False
        if use_shared_expert:
            if shared_expert_per_layer:
                self._se_is_per_layer = True
                self.shared_expert_w0_list = nn.ParameterList(
                    [
                        nn.Parameter(
                            torch.randn(1, 1, inter_dim, head_dim) / math.sqrt(head_dim)
                        )
                        for _ in range(layers)
                    ]
                )
                self.shared_expert_w1_list = nn.ParameterList(
                    [
                        nn.Parameter(
                            torch.randn(1, 1, head_dim, inter_dim)
                            / math.sqrt(inter_dim)
                        )
                        for _ in range(layers)
                    ]
                )
                self.shared_expert_w2_list = nn.ParameterList(
                    [
                        nn.Parameter(
                            torch.randn(1, 1, inter_dim, head_dim) / math.sqrt(head_dim)
                        )
                        for _ in range(layers)
                    ]
                )
            else:
                self.shared_expert_w0 = nn.Parameter(
                    torch.randn(1, 1, inter_dim, head_dim) / math.sqrt(head_dim)
                )
                self.shared_expert_w1 = nn.Parameter(
                    torch.randn(1, 1, head_dim, inter_dim) / math.sqrt(inter_dim)
                )
                self.shared_expert_w2 = nn.Parameter(
                    torch.randn(1, 1, inter_dim, head_dim) / math.sqrt(head_dim)
                )

        if router_share_mode == "share_pool":
            self.pool_router_proj_weights_list = nn.ParameterList(
                [
                    nn.Parameter(
                        torch.randn(1, self.experts_per_pool, head_dim)
                        / math.sqrt(head_dim)
                    )
                    for _ in range(self.num_shared_pools)
                ]
            )
        else:
            self.shared_router_proj_weights = nn.Parameter(
                torch.randn(1, self.experts_per_pool, head_dim) / math.sqrt(head_dim)
            )

        block_kwargs = dict(
            num_experts=self.experts_per_pool,
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
            self.blocks.append(base.MemoryBlock(dim=dim, **block_kwargs))
        self.blocks = nn.ModuleList(self.blocks)

    def _get_router(self, layer_idx):
        if self.router_share_mode == "share_pool":
            return self.pool_router_proj_weights_list[self.layer_pool_idxs[layer_idx]]
        return self.shared_router_proj_weights

    def _init_weights(self, batch_size):
        routed_fastw_list, routed_masterw_list = [], []
        for pool_idx in range(self.num_shared_pools):
            mw = (
                self.pool_w0_list[pool_idx].repeat(batch_size, 1, 1, 1),
                self.pool_w1_list[pool_idx].repeat(batch_size, 1, 1, 1),
                self.pool_w2_list[pool_idx].repeat(batch_size, 1, 1, 1),
            )
            routed_fastw_list.append(self._make_routed_fastw(mw))
            routed_masterw_list.append(mw)
        routed = (routed_fastw_list, routed_masterw_list)

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
                se_mw = (
                    self.shared_expert_w0.repeat(batch_size, 1, 1, 1),
                    self.shared_expert_w1.repeat(batch_size, 1, 1, 1),
                    self.shared_expert_w2.repeat(batch_size, 1, 1, 1),
                )
                se_fw = self._make_se_fastw(se_mw)
                se_weights = [(se_fw, se_mw)] * len(self.blocks)

        return routed, se_weights

    def forward(self, x, info_dict):
        B = x.shape[0]
        device = x.device
        dtype = torch.float32

        (routed_fastw_list, routed_masterw_list), se_weights = self._init_weights(B)

        caches = [{} for _ in self.blocks]
        outputs = []
        infos = []

        num_experts = self.experts_per_pool
        apply_accum_probs = torch.zeros(
            len(self.blocks), B, num_experts, device=device, dtype=dtype
        )
        apply_accum_freqs = torch.zeros(
            len(self.blocks), B, num_experts, device=device, dtype=dtype
        )
        update_accum_probs = torch.zeros(
            len(self.blocks), B, num_experts, device=device, dtype=dtype
        )
        update_accum_freqs = torch.zeros(
            len(self.blocks), B, num_experts, device=device, dtype=dtype
        )

        for start, end, update, _, _ in info_dict["ttt_config"]:
            op_info = []
            xi = x[:, start:end, :]

            for i, block in enumerate(self.blocks):
                pool_idx = self.layer_pool_idxs[i]
                router_w = self._get_router(i)
                fastw_i = routed_fastw_list[pool_idx]
                se_fastw_i = se_weights[i][0] if se_weights is not None else None

                xi, xi_memory_in, apply_lb_info, info = block.forward(
                    xi, fastw_i, router_w, se_fastw=se_fastw_i
                )

                apply_probs_chunk, apply_freqs_chunk = apply_lb_info
                apply_accum_probs[i] += apply_probs_chunk.float()
                apply_accum_freqs[i] += apply_freqs_chunk.float()

                caches[i].update({"xi_out": xi, "xi_memory_in": xi_memory_in})
                if info:
                    op_info.append(info)

            if update:
                for i, cache in enumerate(caches):
                    if self.v_gap is not None:
                        cache["pre_vi"] = caches[self.v_layer_idxs[i]]["xi_out"]
                    else:
                        cache["pre_vi"] = cache["xi_memory_in"]

                if self.aggregate_write:
                    # Per-pool aggregate: sum raw grads from all layers in each pool,
                    # then apply Muon + write once per pool.
                    agg_by_pool = [
                        [
                            torch.zeros_like(routed_masterw_list[p][j])
                            for j in range(3)
                        ]
                        for p in range(self.num_shared_pools)
                    ]

                    for i, block in enumerate(self.blocks):
                        pool_idx = self.layer_pool_idxs[i]
                        cache = caches[i]
                        router_w = self._get_router(i)
                        se_fw_i = se_weights[i][0] if se_weights is not None else None

                        # All layers in a pool use the same W_N snapshot for that pool.
                        raw_grads, update_lb_info, info, se_raw = block.compute_raw_grad_fast_weight(
                            cache["xi_memory_in"],
                            cache["pre_vi"],
                            routed_fastw_list[pool_idx],
                            router_w,
                            se_fastw=se_fw_i,
                        )

                        agg_by_pool[pool_idx][0] += raw_grads[0]
                        agg_by_pool[pool_idx][1] += raw_grads[1]
                        agg_by_pool[pool_idx][2] += raw_grads[2]

                        update_probs_chunk, update_freqs_chunk = update_lb_info
                        update_accum_probs[i] += update_probs_chunk.float()
                        update_accum_freqs[i] += update_freqs_chunk.float()

                        if info and i < len(op_info):
                            op_info[i].update(info)

                        if se_weights is not None and se_raw is not None:
                            se_mw_i = se_weights[i][1]
                            new_se_fw, new_se_mw = self.blocks[0].memory.apply_raw_update(
                                se_raw, se_mw_i
                            )
                            se_weights[i] = (new_se_fw, new_se_mw)

                    for pool_idx in range(self.num_shared_pools):
                        agg_grads = tuple(agg_by_pool[pool_idx])
                        routed_fastw_list[pool_idx], routed_masterw_list[pool_idx] = \
                            self.blocks[0].memory.apply_raw_update(
                                agg_grads, routed_masterw_list[pool_idx]
                            )
                else:
                    for i, block in enumerate(self.blocks):
                        pool_idx = self.layer_pool_idxs[i]
                        cache = caches[i]
                        router_w = self._get_router(i)

                        fastw_i = routed_fastw_list[pool_idx]
                        masterw_i = routed_masterw_list[pool_idx]
                        se_fw_i = se_weights[i][0] if se_weights is not None else None
                        se_mw_i = se_weights[i][1] if se_weights is not None else None

                        (
                            new_fastw,
                            new_masterw,
                            new_se_fw,
                            new_se_mw,
                            update_lb_info,
                            info,
                        ) = block.update_fast_weight(
                            cache["xi_memory_in"],
                            cache["pre_vi"],
                            fastw_i,
                            masterw_i,
                            router_w,
                            se_fastw=se_fw_i,
                            se_masterw=se_mw_i,
                        )

                        update_probs_chunk, update_freqs_chunk = update_lb_info
                        update_accum_probs[i] += update_probs_chunk.float()
                        update_accum_freqs[i] += update_freqs_chunk.float()

                        routed_fastw_list[pool_idx] = new_fastw
                        routed_masterw_list[pool_idx] = new_masterw

                        if se_weights is not None and new_se_fw is not None:
                            if self._se_is_per_layer:
                                se_weights[i] = (new_se_fw, new_se_mw)
                            else:
                                for j in range(len(se_weights)):
                                    se_weights[j] = (new_se_fw, new_se_mw)

                        if info and i < len(op_info):
                            op_info[i].update(info)

            outputs.append(caches[-1]["xi_out"])
            infos.append(op_info)

        outputs = torch.cat(outputs, dim=1)

        apply_total_probs = apply_accum_probs.sum(dim=1)
        apply_total_freqs = apply_accum_freqs.sum(dim=1)
        apply_probs_mean = apply_total_probs / (
            apply_total_probs.sum(dim=1, keepdim=True)
        )
        apply_freqs_mean = apply_total_freqs / (
            apply_total_freqs.sum(dim=1, keepdim=True)
        )
        apply_lb_loss = (
            (apply_probs_mean * apply_freqs_mean).sum(dim=1) * num_experts
        ).sum()

        update_total_probs = update_accum_probs.sum(dim=1)
        update_total_freqs = update_accum_freqs.sum(dim=1)
        update_probs_mean = update_total_probs / (
            update_total_probs.sum(dim=1, keepdim=True)
        )
        update_freqs_mean = update_total_freqs / (
            update_total_freqs.sum(dim=1, keepdim=True)
        )
        update_lb_loss = (
            (update_probs_mean * update_freqs_mean).sum(dim=1) * num_experts
        ).sum()

        def compute_violation(accum_freqs):
            accum_freqs = accum_freqs.float()
            percentages = accum_freqs / accum_freqs.sum(dim=-1, keepdim=True)

            max_v = percentages.max(dim=-1)[0] * num_experts - 1.0
            min_v = torch.abs(percentages.min(dim=-1)[0] * num_experts - 1.0)
            return max_v.mean(), min_v.mean()

        apply_max_v, apply_min_v = compute_violation(apply_accum_freqs)
        update_max_v, update_min_v = compute_violation(update_accum_freqs)

        def compute_freq_extrema(accum_freqs):
            freq = accum_freqs.float().sum(dim=1)
            per_pool_freq = freq.new_zeros(self.num_shared_pools, num_experts)
            for layer_idx, pool_idx in enumerate(self.layer_pool_idxs):
                per_pool_freq[pool_idx] += freq[layer_idx]
            freq = per_pool_freq.flatten()
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
            f"num_shared_pools: {self.num_shared_pools}, pool_size: {self.pool_size}, aggregate_write: {self.aggregate_write}",
            f"num_experts(total): {self.num_experts}, experts_per_pool: {self.experts_per_pool}",
            f"router_share_mode: {self.router_share_mode}",
            f"use_shared_expert: {self.use_shared_expert}, shared_expert_per_layer: {self.shared_expert_per_layer}",
            f"pool_w0_list: {len(self.pool_w0_list)} x {self.pool_w0_list[0].shape}",
            f"load_balancing_loss_alpha: {self.load_balancing_loss_alpha} + Use global lb loss",
        ]
        if self.router_share_mode == "share_pool":
            lines.append(
                f"pool_router_proj_weights_list: {len(self.pool_router_proj_weights_list)} x {self.pool_router_proj_weights_list[0].shape}"
            )
        else:
            lines.append(
                f"shared_router_proj_weights: {self.shared_router_proj_weights.shape}"
            )
        lines.extend(
            f"layer {i} -> pool {pool_idx}"
            for i, pool_idx in enumerate(self.layer_pool_idxs)
        )
        lines.extend(
            f"vmap: {i} <- {v_layer_idx}"
            for i, v_layer_idx in enumerate(self.v_layer_idxs)
        )
        return "\n".join(lines)


def _unit_test_global_moe_multipool_v2():
    # python -m uttt_nvs.models.uttt_moe_partitioned
    torch.manual_seed(0)
    if not torch.cuda.is_available():
        print("CUDA not available; skipping multipool global_moe unit test.")
        return {"skipped": True}

    b, l, dim = 16, 1024, 512
    layers = 8
    num_img_tokens = 256

    ttt_config = []
    for i in range(l // num_img_tokens):
        start = i * num_img_tokens
        end = start + num_img_tokens
        do_update = i % 2 == 0
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
        num_experts=64,
        num_active_experts=1,
        use_muon=True,
        sigmoid_router=False,
        router_alpha=1.0,
        load_balancing_loss_alpha=0.01,
        l2_norm=True,
    )

    configs = [
        {"num_shared_pools": 1, "router_share_mode": "share_all"},
        {"num_shared_pools": 2, "router_share_mode": "share_pool"},
        {"num_shared_pools": 2, "router_share_mode": "share_all"},
        {"num_shared_pools": 4, "router_share_mode": "share_pool"},
        {"num_shared_pools": 4, "router_share_mode": "share_all"},
    ]

    for cfg in configs:
        label = (
            f"num_shared_pools={cfg['num_shared_pools']}, "
            f"router_share_mode={cfg['router_share_mode']}"
        )
        print(f"\n{'=' * 60}")
        print(f"Testing: {label}")
        print(f"{'=' * 60}")

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

    # Test aggregate_write=True (num_shared_pools=2)
    print(f"\n{'='*60}")
    print("Testing: aggregate_write=True (num_shared_pools=2)")
    print(f"{'='*60}")

    torch.manual_seed(42)
    x_aw = torch.randn(b, l, dim, device="cuda", dtype=torch.bfloat16)
    model_aw = MemoryModel(**common_kwargs, num_shared_pools=2, router_share_mode="share_pool",
                           aggregate_write=True).to("cuda")
    print(model_aw)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        outputs_aw, lb_loss_aw, _ = model_aw(x_aw, info_dict)
    print("outputs_aw:", tuple(outputs_aw.shape))
    print("lb_loss_aw:", lb_loss_aw.item())
    (outputs_aw.sum() + lb_loss_aw).backward()
    print("backward OK [aggregate_write=True, num_shared_pools=2]")

    # Outputs should differ from sequential
    torch.manual_seed(42)
    model_seq = MemoryModel(**common_kwargs, num_shared_pools=2, router_share_mode="share_pool",
                            aggregate_write=False).to("cuda")
    model_seq.load_state_dict(model_aw.state_dict())
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        outputs_seq, _, _ = model_seq(x_aw.detach(), info_dict)
    differ = not torch.allclose(outputs_aw.float().detach(), outputs_seq.float(), atol=1e-2)
    print(f"aggregate_write=True vs False outputs differ: {differ} (expected True)")
    assert differ, "aggregate_write=True/False should produce different outputs"

    return {"success": True}


if __name__ == "__main__":
    print(_unit_test_global_moe_multipool_v2())
