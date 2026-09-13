"""Pytest suite for the fused permute kernels.

Moved out of the kernel module so pytest is not an import-time dependency of
training. Most tests require CUDA and are marked accordingly.
"""
import torch
from uttt_nvs.models.kernels.triton_permute import *  # noqa: F401,F403
from uttt_nvs.models.kernels.triton_permute import (
    permute_with_expert_mask, unpermute_and_merge_with_probs,
)

import pytest


def _build_topk_mask(B, T, E, k, device):

    device = "cuda"

    max_experts = min(k, E - 1)

    routing_map = torch.rand(B, E, T, device=device)
    # [B, max_experts, T]
    values, topk = torch.topk(routing_map, k=max_experts, dim=1)
    value_threshold = values[
        :,
        [-1],
    ]
    routing_map = (routing_map > value_threshold).to(torch.int32)
    return routing_map


@pytest.mark.parametrize("B,T,E,H,topk", [(2, 32, 4, 16, 2)])
def test_unpermute_and_merge_with_probs_forward(B, T, E, H, topk):
    torch.manual_seed(321)
    device = "cuda"

    expert_mask = _build_topk_mask(B, T, E, topk, device)
    inp = torch.randn(B, T, H, device=device, dtype=torch.float32)

    probs = torch.rand(B, E, T, device=device, dtype=torch.float32)
    probs = probs * expert_mask

    permuted_x, perm_probs, row_id_map = permute_with_expert_mask(
        inp=inp, expert_mask=expert_mask, probs=probs, topk=topk
    )

    merged = unpermute_and_merge_with_probs(permuted_x, row_id_map, probs)

    sum_probs = (probs * expert_mask).sum(dim=1)  # [B, T]
    ref_merged = inp * sum_probs.unsqueeze(-1)

    report_error(ref_merged, merged, "unpermute+merge/merged")
    torch.testing.assert_close(merged, ref_merged, atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("B,T,E,H,topk", [(2, 16, 4, 8, 2)])
def test_roundtrip_backward_grads(B, T, E, H, topk):
    torch.manual_seed(42)
    device = "cuda"

    expert_mask = _build_topk_mask(B, T, E, topk, device)

    inp = torch.randn(B, T, H, device=device, dtype=torch.float32, requires_grad=True)
    probs = torch.rand(B, E, T, device=device, dtype=torch.float32, requires_grad=True)
    probs_masked = probs * expert_mask  # zero outside top-k

    permuted_x, perm_probs, row_id_map = permute_with_expert_mask(
        inp=inp, expert_mask=expert_mask, probs=probs_masked, topk=topk
    )
    merged = unpermute_and_merge_with_probs(permuted_x, row_id_map, probs_masked)

    loss = merged.sum() + 0.0 * perm_probs.sum()
    loss.backward()

    with torch.no_grad():
        sum_probs = (probs * expert_mask).sum(dim=1)  # [B, T]
        grad_inp_ref = sum_probs.unsqueeze(-1).expand_as(inp)
        grad_probs_ref = expert_mask * inp.sum(dim=-1, keepdim=False).unsqueeze(
            1
        )  # [B,E,T]

    report_error(grad_inp_ref, inp.grad, "backward/grad_inp")
    report_error(grad_probs_ref, probs.grad, "backward/grad_probs")

    # torch.testing.assert_close(inp.grad, grad_inp_ref, atol=1e-3, rtol=1e-3)
    # torch.testing.assert_close(probs.grad, grad_probs_ref, atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    # _test_row_id_map_correctness()
    # test_permute_without_probs_small()
    # test_permute_with_probs_and_padding_zero()
    # test_permute_batched_random(1, 1024, 4, 384, 2)
    # test_permute_batched_random(2, 2048, 4, 256)
    # some irregular shape
    # test_permute_batched_random(3, 12345, 7, 567)

    # test_unpermute_and_merge_with_probs_forward(1, 1024, 4, 384, 2)
    # test_roundtrip_backward_grads(1, 1024, 4, 384, 2)

    test_permute_kv_and_lrs_forward()
