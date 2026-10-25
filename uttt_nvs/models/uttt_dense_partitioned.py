"""Dense global MHA fast weights shared within contiguous layer pools.

This is the multipool analogue of memory_block_v2_mha_global:
- num_shared_pools=1 matches the original global sharing layout.
- num_shared_pools=2 shares one fast-weight state across layers 0-3 and one
  across layers 4-7 when layers=8.
- num_shared_pools=4 shares one fast-weight state per two-layer group.

There is no router here; "pool" only means a contiguous layer group owns a
separate dense global-MHA fast-weight state.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from .uttt_dense_block import MemoryBlock


class MemoryModel(nn.Module):
    """MHA fast-weight with contiguous multi-pool cross-layer sharing."""

    def __init__(
        self,
        layers,
        dim,
        v_gap=None,
        fw_head_dim=None,
        fw_inter_multi=1,
        num_shared_pools=1,
        pool_mode="shared",
        aggregate_write=False,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.v_gap = v_gap
        self.num_shared_pools = int(num_shared_pools)
        self.fw_inter_multi = fw_inter_multi
        self.aggregate_write = aggregate_write

        if self.num_shared_pools <= 0:
            raise ValueError("num_shared_pools must be positive.")
        if pool_mode != "shared":
            raise ValueError("multipool global_mha only supports pool_mode='shared'.")
        if layers % self.num_shared_pools != 0:
            raise ValueError(
                f"layers ({layers}) must be divisible by num_shared_pools ({self.num_shared_pools})."
            )

        if fw_head_dim is None:
            fw_head_dim = dim
        self.head_dim = fw_head_dim
        self.num_heads = dim // fw_head_dim
        self.inter_dim = int(self.head_dim * fw_inter_multi)
        self.pool_size = layers // self.num_shared_pools
        self.layer_pool_idxs = [i // self.pool_size for i in range(layers)]

        self.pool_w0_list = nn.ParameterList(
            [
                nn.Parameter(
                    torch.randn(self.num_heads, self.head_dim, self.inter_dim)
                    / math.sqrt(self.head_dim)
                )
                for _ in range(self.num_shared_pools)
            ]
        )
        self.pool_w1_list = nn.ParameterList(
            [
                nn.Parameter(
                    torch.randn(self.num_heads, self.inter_dim, self.head_dim)
                    / math.sqrt(self.inter_dim)
                )
                for _ in range(self.num_shared_pools)
            ]
        )
        self.pool_w2_list = nn.ParameterList(
            [
                nn.Parameter(
                    torch.randn(self.num_heads, self.head_dim, self.inter_dim)
                    / math.sqrt(self.head_dim)
                )
                for _ in range(self.num_shared_pools)
            ]
        )

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

    def _init_pool_fast_weight(self, pool_idx, batch_size):
        master_weight = (
            self.pool_w0_list[pool_idx].unsqueeze(0).repeat(batch_size, 1, 1, 1),
            self.pool_w1_list[pool_idx].unsqueeze(0).repeat(batch_size, 1, 1, 1),
            self.pool_w2_list[pool_idx].unsqueeze(0).repeat(batch_size, 1, 1, 1),
        )
        fast_weight = tuple(
            F.normalize(w, dim=2, eps=1e-5).to(torch.bfloat16)
            for w in master_weight
        )
        return fast_weight, master_weight

    def init_shared_fast_weights(self, batch_size):
        fastw_list = []
        masterw_list = []
        for pool_idx in range(self.num_shared_pools):
            fastw, masterw = self._init_pool_fast_weight(pool_idx, batch_size)
            fastw_list.append(fastw)
            masterw_list.append(masterw)
        return fastw_list, masterw_list

    def forward(self, x, info_dict):
        fastw_list, masterw_list = self.init_shared_fast_weights(batch_size=x.shape[0])
        caches = [{} for _ in self.blocks]

        outputs = []
        infos = []
        for start, end, update, _, _ in info_dict["ttt_config"]:
            op_info = []
            xi = x[:, start:end, :]

            for i, (block, cache) in enumerate(zip(self.blocks, caches)):
                pool_idx = self.layer_pool_idxs[i]
                xi, xi_memory_in, info = block.forward(xi, fastw_list[pool_idx])
                cache.update({"xi_out": xi, "xi_memory_in": xi_memory_in})
                if info:
                    op_info.append(info)

            if update:
                for i, cache in enumerate(caches):
                    if self.v_gap is not None:
                        cache["pre_vi"] = caches[self.v_layer_idxs[i]]["xi_out"]
                    else:
                        cache["pre_vi"] = cache["xi_memory_in"]

                if self.aggregate_write:
                    agg_by_pool = [
                        [
                            torch.zeros_like(masterw_list[pool_idx][weight_idx])
                            for weight_idx in range(3)
                        ]
                        for pool_idx in range(self.num_shared_pools)
                    ]

                    for i, (block, cache) in enumerate(zip(self.blocks, caches)):
                        pool_idx = self.layer_pool_idxs[i]
                        raw_grads, info = block.compute_raw_grad_fast_weight(
                            cache["xi_memory_in"],
                            cache["pre_vi"],
                            fastw_list[pool_idx],
                        )
                        agg_by_pool[pool_idx][0] = agg_by_pool[pool_idx][0] + raw_grads[0]
                        agg_by_pool[pool_idx][1] = agg_by_pool[pool_idx][1] + raw_grads[1]
                        agg_by_pool[pool_idx][2] = agg_by_pool[pool_idx][2] + raw_grads[2]
                        if info and i < len(op_info):
                            op_info[i].update(info)

                    for pool_idx in range(self.num_shared_pools):
                        fastw_list[pool_idx], masterw_list[pool_idx] = self.blocks[0].memory.apply_raw_update(
                            tuple(agg_by_pool[pool_idx]),
                            masterw_list[pool_idx],
                        )
                else:
                    for i, (block, cache) in enumerate(zip(self.blocks, caches)):
                        pool_idx = self.layer_pool_idxs[i]
                        fastw, masterw, info = block.update_fast_weight(
                            cache["xi_memory_in"],
                            cache["pre_vi"],
                            fastw_list[pool_idx],
                            masterw_list[pool_idx],
                        )
                        fastw_list[pool_idx] = fastw
                        masterw_list[pool_idx] = masterw
                        if info and i < len(op_info):
                            op_info[i].update(info)

            outputs.append(caches[-1]["xi_out"])
            infos.append(op_info)

        outputs = torch.cat(outputs, dim=1)
        return outputs, infos

    def extra_repr(self) -> str:
        lines = [
            f"num_shared_pools: {self.num_shared_pools}, pool_size: {self.pool_size}",
            f"aggregate_write: {self.aggregate_write}",
            f"num_heads: {self.num_heads}, head_dim: {self.head_dim}, inter_dim: {self.inter_dim} (fw_inter_multi={self.fw_inter_multi})",
            f"pool_w0_list: {len(self.pool_w0_list)} x {self.pool_w0_list[0].shape}",
        ]
        lines.extend(
            f"layer {i} -> pool {pool_idx}"
            for i, pool_idx in enumerate(self.layer_pool_idxs)
        )
        lines.extend(
            f"vmap: {i} <- {v_layer_idx}"
            for i, v_layer_idx in enumerate(self.v_layer_idxs)
        )
        return "\n".join(lines)


def _unit_test_mha_global_multipool():
    # python -m uttt_nvs.models.uttt_dense_partitioned
    torch.manual_seed(0)
    if not torch.cuda.is_available():
        print("CUDA not available; skipping mha_global_multipool unit test.")
        return {"skipped": True}

    b, L, dim = 2, 512, 512
    layers = 8
    num_img_tokens = 128
    info_dict = {
        "num_img_tokens": num_img_tokens,
        "ttt_config": [
            (i * num_img_tokens, (i + 1) * num_img_tokens, i % 2 == 0, None, None)
            for i in range(L // num_img_tokens)
        ],
    }

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

    for cfg in [
        {"num_shared_pools": 1, "fw_inter_multi": 8},
        {"num_shared_pools": 2, "fw_inter_multi": 4, "aggregate_write": False},
        {"num_shared_pools": 2, "fw_inter_multi": 4, "aggregate_write": True},
        {"num_shared_pools": 4, "fw_inter_multi": 2, "aggregate_write": True},
    ]:
        print(f"\nTesting {cfg}")
        x = torch.randn(b, L, dim, device="cuda", dtype=torch.bfloat16)
        model = MemoryModel(**common_kwargs, **cfg).to("cuda")
        print(model)
        outputs, infos = model(x, info_dict)
        print("outputs:", tuple(outputs.shape), outputs.dtype)
        outputs.sum().backward()
        print("backward OK")

    return {"success": True}


if __name__ == "__main__":
    print(_unit_test_mha_global_multipool())
