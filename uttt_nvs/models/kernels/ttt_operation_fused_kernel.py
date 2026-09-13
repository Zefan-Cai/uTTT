# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F

try:
    # from l2norm_triton_kernels import l2_norm_add_fused
    # from triton_prenorm_update_with_momentum import (
    #     fused_prenorm_update_with_momentum_and_l2_norm,
    # )
    from lact_swiglu_ffn import (
        grouped_swiglu_ffn_fwd,
    )
    from lact_fw_grad import grouped_lact_swiglu_ffn_fast_weight_grads
    from triton_permute import (
        permute_with_expert_mask,
        unpermute_and_merge_with_probs,
        permute_kv_and_lrs,
    )
except ImportError:
    # from .l2norm_triton_kernels import l2_norm_add_fused
    # from .lact_swiglu_ffn import (
    #     fused_swiglu_ffn_fwd,
    # )
    from .lact_fw_grad import grouped_lact_swiglu_ffn_fast_weight_grads
    from .lact_swiglu_ffn import (
        grouped_swiglu_ffn_fwd,
    )
    from .triton_permute import (
        permute_with_expert_mask,
        unpermute_and_merge_with_probs,
        permute_kv_and_lrs,
    )
from einops import rearrange


@torch.compile()
def silu_backprop(dy: torch.Tensor, x: torch.Tensor):
    """
    Args:
        dy: [b, d, l], gradient of the outer loss wrt the y
        x: [b, d, l], input of the silu activation
    outs:
        dx: [b, d, l], gradient of the outer loss wrt the x
        dx = dy * sigma * (1 + x * (1 - sigma))
    """
    sigma = torch.sigmoid(x)
    dx = dy * sigma * (1 + x * (1 - sigma))
    return dx


@torch.compile()
def zeropower_via_newtonschulz5(G):
    """
    This is an updated version of the zeropower_via_newtonschulz5 function in here:
    https://github.com/KellerJordan/modded-nanogpt/blob/master/train_gpt_medium.py#L26
    The code is modified from https://github.com/MoonshotAI/Moonlight/blob/master/examples/toy_train.py#L49, which contains the original muon implementation.
    Major change: G is [b, d, d] rather than [d, d]
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
    Args:
        G: [b, d, d']
    Returns:
        X: [b, d, d']
    FLOPS:  When d=d', Total FLOPS=30 * b * d^3
    """
    assert len(G.shape) == 3
    X = G.bfloat16()
    if G.size(1) > G.size(2):
        X = X.transpose(1, 2)
    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(1, 2), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for a, b, c in [
        (4.0848, -6.8946, 2.9270),
        (3.9505, -6.3029, 2.6377),
        (3.7418, -5.5913, 2.3037),
        (2.8769, -3.1427, 1.2046),
        (2.8366, -3.0525, 1.2012),
    ]:
        A = X @ X.transpose(1, 2)
        B = (
            b * A + c * A @ A
        )  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(1) > G.size(2):
        X = X.transpose(1, 2)
    return X


def create_router_mask_sizes_probs(
    router_logits: torch.Tensor, topk: int, alpha: float = 1.0
):
    """
    router_logits: [num_chunks, B, E, chunk_size]

    Returns:
    router_mask: [num_chunks, B, E, chunk_size]
    group_sizes: [num_chunks, B, E]
    router_probs: [num_chunks, B, E, chunk_size]
    """
    # don't use threshold value to create the router mask
    # because in bf16, it will create tiles,
    # which cause the total number of actived experts >= topk
    _, topk_idx = torch.topk(router_logits, k=topk, dim=-2)
    router_mask = torch.zeros_like(router_logits, dtype=torch.bool)
    router_mask.scatter_(-2, topk_idx, True)

    # [num_chunks, B, E]
    group_sizes = router_mask.sum(dim=-1).to(torch.int32)

    router_probs = (
        torch.softmax(router_logits.to(torch.float32), dim=-2).to(router_logits.dtype)
        * alpha
    )

    # below we compute the load balancing loss:
    # There are two different load balancing losses:
    # 1. compute lb loss per sequence, then average over all sequences.
    # 2. compute lb loss over all batches.
    # here, we do the second one, maybe the first one is more correct.
    # [num_chunks, B, E, chunk_size] -> [E]
    router_probs_mean_over_batch_and_seqlen = (
        router_probs.mean(dim=0).mean(dim=0).mean(dim=-1)
    )
    # [num_chunks, B, E] -> [E]
    group_sizes_per_expert = (group_sizes * 1.0).mean(dim=0).mean(dim=0)
    # [E]
    frequency_per_expert = group_sizes_per_expert / group_sizes_per_expert.sum()

    lb_loss = router_probs_mean_over_batch_and_seqlen * frequency_per_expert
    lb_loss = lb_loss.sum() * topk

    return router_mask, group_sizes, router_probs, lb_loss


def moe_auxiliary_loss(
    router_mask: torch.Tensor, group_sizes: torch.Tensor, router_probs: torch.Tensor
):
    """
    Args:
        router_mask: [b, E, num_tokens]
        group_sizes: [b, E]
        router_probs: [b, E, num_tokens]
    Returns:
        loss: [b, E]
    """
    group_sizes = group_sizes.sum(dim=0)  # [E]
    # [E]
    average_prob_per_experts = router_probs.mean(dim=-1).mean(dim=0)

    # [E]
    expert_frequency = group_sizes.float() / group_sizes.float().sum()

    # [E]
    loss = average_prob_per_experts * expert_frequency

    return loss


@torch.compile()
def prenorm_block_causal_lact_swiglu_moe_triton(
    w0: torch.Tensor,  # [B, E, H, D]
    w1: torch.Tensor,  # [B, E, D, H]
    w2: torch.Tensor,  # [B, E, H, D]
    q: torch.Tensor,  # [B, L, D]
    k: torch.Tensor,  # [B, L * num_heads, D]
    v: torch.Tensor,  # [B, L * num_heads, D]
    lr0: torch.Tensor,
    lr1: torch.Tensor,
    lr2: torch.Tensor,
    router_proj_weights: torch.Tensor,  # [B, E, D]
    momentum: torch.Tensor,  # [b, s]
    chunk_size: int = 2048,  # test-time training chunk size
    topk: int = 2,
    kv_repeat: int = 1,
):

    w0_w2 = torch.cat([w0, w2], dim=2)
    w0_w2_norm = w0_w2.norm(dim=3, keepdim=True)  # [b, 2*Hidden]
    w1_norm = w1.norm(dim=3, keepdim=True)  # [b, D]

    w0_w2_main = w0_w2
    w1_main = w1

    w0_w2 = w0_w2.to(torch.bfloat16)
    w1 = w1.to(torch.bfloat16)
    if momentum is not None:
        dw1_momentum = torch.zeros_like(w1_main)
        dw0_w2_momentum = torch.zeros_like(w0_w2_main)

    q_original_length = q.shape[1]

    kv_chunk_size = int(chunk_size * kv_repeat)
    k = F.pad(k, (0, 0, 0, -k.shape[1] % kv_chunk_size))
    v = F.pad(v, (0, 0, 0, -v.shape[1] % kv_chunk_size))
    q = F.pad(q, (0, 0, 0, -q.shape[1] % chunk_size))
    lr0 = F.pad(lr0, (0, 0, 0, -lr0.shape[1] % kv_chunk_size))
    lr1 = F.pad(lr1, (0, 0, 0, -lr1.shape[1] % kv_chunk_size))
    lr2 = F.pad(lr2, (0, 0, 0, -lr2.shape[1] % kv_chunk_size))
    if momentum is not None:
        momentum = F.pad(momentum, (0, 0, 0, -momentum.shape[1] % kv_chunk_size))

    num_chunks = q.shape[1] // chunk_size

    # [B, E, num_chunk * chunk_size]
    with torch.cuda.amp.autocast(enabled=False):
        k_logits = torch.bmm(
            router_proj_weights.to(torch.float32), k.transpose(1, 2).to(torch.float32)
        ).to(torch.float32)
        q_logits = torch.bmm(
            router_proj_weights.to(torch.float32), q.transpose(1, 2).to(torch.float32)
        ).to(torch.float32)

    k = rearrange(k, "b (n c) d -> n b c d", n=num_chunks).contiguous()
    v = rearrange(v, "b (n c) d -> n b c d", n=num_chunks).contiguous()
    q = rearrange(q, "b (n c) d -> n b c d", n=num_chunks).contiguous()
    lr0 = rearrange(lr0, "b (n c) d -> n b c d", n=num_chunks, d=1).contiguous()
    lr1 = rearrange(lr1, "b (n c) d -> n b c d", n=num_chunks, d=1).contiguous()
    lr2 = rearrange(lr2, "b (n c) d -> n b c d", n=num_chunks, d=1).contiguous()
    momentum = rearrange(
        momentum, "b (n c) d -> n b c d", n=num_chunks, d=1
    ).contiguous()

    # [num_chunks, B, E, chunk_size]
    k_logits = rearrange(k_logits, "b e (n c) -> n b e c", n=num_chunks).contiguous()
    q_logits = rearrange(q_logits, "b e (n c) -> n b e c", n=num_chunks).contiguous()
    output = torch.zeros_like(q)

    ## compute the moe logits

    k_router_mask, k_router_sizes, k_router_probs = create_router_mask_sizes_probs(
        k_logits, topk
    )
    q_router_mask, q_router_sizes, q_router_probs = create_router_mask_sizes_probs(
        q_logits, topk
    )

    k_router_sizes_avg = (k_router_sizes * 1.0).mean(dim=0)
    q_router_sizes_avg = (q_router_sizes * 1.0).mean(dim=0)
    e_index = 0
    seq_len = k.shape[1]
    # if torch.rand(1).item() < 0.01:
    #     print(
    #         "q_router_sizes: ",
    #         q_router_sizes.sum(dim=0).sum(dim=0),
    #         "k_router_sizes: ",
    #         k_router_sizes.sum(dim=0).sum(dim=0),
    #     )

    # TODO, maybe multiple the k_router_probs onto the vi, to indicates adpative gradients.
    for chunk_idx in range(num_chunks - 1):

        # [b, l, dk]
        ki = k[chunk_idx]  # .contiguous()  # bf16
        # [b, l, dv]
        vi = v[chunk_idx]  # .contiguous()  # bf16
        # [b, l, dq]
        qi = q[chunk_idx]  # .contiguous()
        # [b, l, d/1] fp32
        lr1i = lr1[chunk_idx]  # .contiguous()  # [b, l, d/1] fp32
        lr2i = lr2[chunk_idx]  # .contiguous()  # [b, l, d/1] fp32
        lr0i = lr0[chunk_idx]  # .contiguous()  # [b, l, d/1] fp32

        k_router_mask_i = k_router_mask[chunk_idx]
        q_router_mask_i = q_router_mask[chunk_idx]

        k_group_sizes_i = k_router_sizes[chunk_idx]
        q_group_sizes_i = q_router_sizes[chunk_idx]

        k_router_probs_i = k_router_probs[chunk_idx]
        q_router_probs_i = q_router_probs[chunk_idx]

        ki_permuted, ki_permuted_router_probs, k_row_id_map = permute_with_expert_mask(
            ki, k_router_mask_i, k_router_probs_i, topk, None
        )

        vi_permuted, vi_permuted_router_probs, _ = permute_with_expert_mask(
            vi, k_router_mask_i, k_router_probs_i, topk, k_row_id_map
        )
        # for dot-product loss, we need to multiply the router probs onto the vi, to indicates adpative gradients.
        vi_permuted = (vi_permuted * ki_permuted_router_probs.unsqueeze(dim=-1)).to(
            ki_permuted.dtype
        )

        qi_permuted, qi_permuted_router_probs, q_row_id_map = permute_with_expert_mask(
            qi, q_router_mask_i, q_router_probs_i, topk, None
        )
        lr0i_permuted, _, _ = permute_with_expert_mask(
            lr0i, k_router_mask_i, None, topk, k_row_id_map
        )
        lr1i_permuted, _, _ = permute_with_expert_mask(
            lr1i, k_router_mask_i, None, topk, k_row_id_map
        )
        lr2i_permuted, _, _ = permute_with_expert_mask(
            lr2i, k_router_mask_i, None, topk, k_row_id_map
        )
        lr0i_permuted = lr0i_permuted.squeeze(dim=-1)
        lr1i_permuted = lr1i_permuted.squeeze(dim=-1)
        lr2i_permuted = lr2i_permuted.squeeze(dim=-1)

        o_permuted = grouped_swiglu_ffn_fwd(w0_w2, w1, qi_permuted, q_group_sizes_i)

        output[chunk_idx] = unpermute_and_merge_with_probs(
            o_permuted, q_row_id_map, q_router_probs_i
        )

        # use previous w0 and w1 to get the final output
        # [b, dh, dk] @ [b, dk, l] -> [b, dh, l]
        dw0_w2, dw1 = grouped_lact_swiglu_ffn_fast_weight_grads(
            w0_w2,
            w1,
            ki_permuted,
            vi_permuted,
            lr0i_permuted,
            lr1i_permuted,
            lr2i_permuted,
            k_group_sizes_i,
        )

        m_i = momentum[chunk_idx].contiguous()
        m_i = m_i.mean(dim=1, keepdim=True).unsqueeze(dim=-1)

        dw0_w2 = dw0_w2 + dw0_w2_momentum * m_i
        dw1 = dw1 + dw1_momentum * m_i
        dw0_w2_momentum = dw0_w2
        dw1_momentum = dw1

        w0_w2_main = w0_w2_main + dw0_w2
        w1_main = w1_main + dw1

        w0_w2 = w0_w2_main / (w0_w2_main.norm(dim=3, keepdim=True) + 1e-5) * w0_w2_norm
        w1 = w1_main / (w1_main.norm(dim=3, keepdim=True) + 1e-5) * w1_norm
        w0_w2 = w0_w2.to(torch.bfloat16)
        w1 = w1.to(torch.bfloat16)

        # w0_w2_main, dw0_w2_momentum, w0_w2 = (
        #     fused_prenorm_update_with_momentum_and_l2_norm(
        #         w0_w2_main, dw0_w2, dw0_w2_momentum, m_i, w0_w2_norm, 1e-5
        #     )
        # )
        # w1_main, dw1_momentum, w1 = fused_prenorm_update_with_momentum_and_l2_norm(
        #     w1_main, dw1, dw1_momentum, m_i, w1_norm, 1e-5
        # )

    # for the last chunk, don't update the fast weights, directly apply the fast weights to the query.
    s_index = e_index
    e_index = seq_len

    qi = q[-1].contiguous()
    qi_router_mask_i = q_router_mask[-1]
    qi_router_probs_i = q_router_probs[-1]
    q_router_sizes_i = q_router_sizes[-1]
    qi_permuted, qi_permuted_router_probs, q_row_id_map = permute_with_expert_mask(
        qi, qi_router_mask_i, qi_router_probs_i, topk, None
    )

    o_permuted = grouped_swiglu_ffn_fwd(w0_w2, w1, qi_permuted, q_router_sizes_i)

    output[-1] = unpermute_and_merge_with_probs(
        o_permuted, q_row_id_map, qi_router_probs_i
    )

    output = rearrange(output, "n b c d -> b (n c) d")
    output = output[:, :q_original_length, :]

    # [B, total_n, d], [n_chunks, B, E, total_n], [n_chunks, B, E, total_n]
    return output, k_logits, q_logits, k_router_sizes_avg, q_router_sizes_avg


@torch.compile()
def block_causal_lact_swiglu_triton(
    w0: torch.Tensor,  # [B, E, H, D]. E means number of experts.
    w1: torch.Tensor,  # [B, E, D, H]
    w2: torch.Tensor,  # [B, E, H, D]
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lr0: torch.Tensor,
    lr1: torch.Tensor,
    lr2: torch.Tensor,
    router_proj_weights: torch.Tensor,  # [B, E, D]
    momentum: torch.Tensor,  # [b, s]
    chunk_size: int = 2048,  # test-time training chunk size
    topk: int = 2,  # means number of active experts for each token.
):

    w0_w2 = torch.cat([w0, w2], dim=2)
    w0_w2_norm = w0_w2.norm(dim=3, keepdim=True)  # [b, 2*Hidden]
    w1_norm = w1.norm(dim=3, keepdim=True)  # [b, D]

    w0_w2_main = w0_w2
    w1_main = w1

    w0_w2 = w0_w2.to(torch.bfloat16)
    w1 = w1.to(torch.bfloat16)

    dw1_momentum = torch.zeros_like(w1_main)
    dw0_w2_momentum = torch.zeros_like(w0_w2_main)

    num_chunks = k.shape[1] // chunk_size

    # [B, E, num_chunk * chunk_size]
    with torch.cuda.amp.autocast(enabled=False):
        k_logits = torch.bmm(
            router_proj_weights.to(torch.float32), k.transpose(1, 2).to(torch.float32)
        ).to(torch.float32)
        q_logits = torch.bmm(
            router_proj_weights.to(torch.float32), q.transpose(1, 2).to(torch.float32)
        ).to(torch.float32)

    k = rearrange(k, "b (n c) d -> n b c d", n=num_chunks).contiguous()
    v = rearrange(v, "b (n c) d -> n b c d", n=num_chunks).contiguous()
    q = rearrange(q, "b (n c) d -> n b c d", n=num_chunks).contiguous()
    lr0 = rearrange(lr0, "b (n c) d -> n b (c d)", n=num_chunks).contiguous()
    lr1 = rearrange(lr1, "b (n c) d -> n b (c d)", n=num_chunks).contiguous()
    lr2 = rearrange(lr2, "b (n c) d -> n b (c d)", n=num_chunks).contiguous()
    momentum = rearrange(momentum, "b (n c) d -> n b (c d)", n=num_chunks).contiguous()

    # [num_chunks, B, E, chunk_size]
    k_logits = rearrange(k_logits, "b e (n c) -> n b e c", n=num_chunks).contiguous()
    q_logits = rearrange(q_logits, "b e (n c) -> n b e c", n=num_chunks).contiguous()
    output = torch.zeros_like(q)

    ## compute the moe logits
    ## Maybe it's a good idea to move this function outside of the ttt function
    # [num_chunks, B, E, T], [num_chunks, B, E], [num_chunks, B, E, T]

    k_router_mask, k_router_sizes, k_router_probs, k_lb_loss = (
        create_router_mask_sizes_probs(k_logits, topk)
    )
    q_router_mask, q_router_sizes, q_router_probs, q_lb_loss = (
        create_router_mask_sizes_probs(q_logits, topk)
    )

    k_router_sizes_avg = (k_router_sizes * 1.0).mean(dim=0).mean(dim=0)
    q_router_sizes_avg = (q_router_sizes * 1.0).mean(dim=0).mean(dim=0)
    e_index = 0
    seq_len = k.shape[1]

    # TODO, maybe multiple the k_router_probs onto the vi, to indicates adpative gradients.
    for chunk_idx in range(num_chunks - 1):

        # [b, l, dk]
        ki = k[chunk_idx]  # .contiguous()  # bf16
        # [b, l, dv]
        vi = v[chunk_idx]  # .contiguous()  # bf16
        # [b, l, dq]
        qi = q[chunk_idx]  # .contiguous()
        # [b, l, d/1] fp32
        lr1i = lr1[chunk_idx]  # .contiguous()  # [b, l] fp32
        lr2i = lr2[chunk_idx]  # .contiguous()  # [b, l] fp32
        lr0i = lr0[chunk_idx]  # .contiguous()  # [b, l] fp32

        # [b, E, L]
        k_router_mask_i = k_router_mask[chunk_idx]
        q_router_mask_i = q_router_mask[chunk_idx]

        # [b, E]
        k_group_sizes_i = k_router_sizes[chunk_idx]
        q_group_sizes_i = q_router_sizes[chunk_idx]

        # [b, E, L]
        k_router_probs_i = k_router_probs[chunk_idx]
        q_router_probs_i = q_router_probs[chunk_idx]

        # shape for ki, vi: [b, num_tokens, d], mostly bf16
        # shape for lr0i, lr1i, lr2i: [b, num_tokens], fp32
        # shape for k_router_mask_i: [b, num_experts, num_tokens], bool
        # shape for k_router_probs_i: [b, num_experts, num_tokens], fp32
        # shape for lr0i_permuted, lr1i_permuted, lr2i_permuted: [b, num_tokens], fp32
        # topk: int, number of activedexperts to be routed to the tokens

        # note, vi_permuted is multiplied by k_router_probs_i.unsqueeze(-1)
        ki_permuted, vi_permuted, lr0i_permuted, lr1i_permuted, lr2i_permuted = (
            permute_kv_and_lrs(
                ki, vi, lr0i, lr1i, lr2i, k_router_mask_i, k_router_probs_i, topk
            )
        )

        qi_permuted, qi_permuted_router_probs, q_row_id_map = permute_with_expert_mask(
            qi, q_router_mask_i, q_router_probs_i, topk, None
        )

        o_permuted = grouped_swiglu_ffn_fwd(w0_w2, w1, qi_permuted, q_group_sizes_i)

        output[chunk_idx] = unpermute_and_merge_with_probs(
            o_permuted, q_row_id_map, q_router_probs_i
        )

        # use previous w0 and w1 to get the final output
        # [b, dh, dk] @ [b, dk, l] -> [b, dh, l]
        dw0_w2, dw1 = grouped_lact_swiglu_ffn_fast_weight_grads(
            w0_w2,
            w1,
            ki_permuted,
            vi_permuted,
            lr0i_permuted,
            lr1i_permuted,
            lr2i_permuted,
            k_group_sizes_i,
        )

        m_i = momentum[chunk_idx].contiguous()
        m_i = m_i.mean(dim=1, keepdim=True).unsqueeze(dim=-1).unsqueeze(dim=-1)

        dw0_w2 = dw0_w2 + dw0_w2_momentum * m_i
        dw1 = dw1 + dw1_momentum * m_i
        dw0_w2_momentum = dw0_w2
        dw1_momentum = dw1

        w0_w2_main = w0_w2_main + dw0_w2
        w1_main = w1_main + dw1

        w0_w2 = w0_w2_main / (w0_w2_main.norm(dim=3, keepdim=True) + 1e-5) * w0_w2_norm
        w1 = w1_main / (w1_main.norm(dim=3, keepdim=True) + 1e-5) * w1_norm
        w0_w2 = w0_w2.to(torch.bfloat16)
        w1 = w1.to(torch.bfloat16)

        # w0_w2_main, dw0_w2_momentum, w0_w2 = (
        #     fused_prenorm_update_with_momentum_and_l2_norm(
        #         w0_w2_main, dw0_w2, dw0_w2_momentum, m_i, w0_w2_norm, 1e-5
        #     )
        # )
        # w1_main, dw1_momentum, w1 = fused_prenorm_update_with_momentum_and_l2_norm(
        #     w1_main, dw1, dw1_momentum, m_i, w1_norm, 1e-5
        # )

    # for the last chunk, don't update the fast weights, directly apply the fast weights to the query.
    s_index = e_index
    e_index = seq_len

    qi = q[-1].contiguous()
    qi_router_mask_i = q_router_mask[-1]
    qi_router_probs_i = q_router_probs[-1]
    q_router_sizes_i = q_router_sizes[-1]
    qi_permuted, qi_permuted_router_probs, q_row_id_map = permute_with_expert_mask(
        qi, qi_router_mask_i, qi_router_probs_i, topk, None
    )

    o_permuted = grouped_swiglu_ffn_fwd(w0_w2, w1, qi_permuted, q_router_sizes_i)

    output[-1] = unpermute_and_merge_with_probs(
        o_permuted, q_row_id_map, qi_router_probs_i
    )

    output = rearrange(output, "n b c d -> b (n c) d")

    return output, k_lb_loss, q_lb_loss, k_router_sizes_avg, q_router_sizes_avg


@torch.compile()
def block_causal_lact_swiglu_triton_with_permuted_input(
    w0: torch.Tensor,  # [B, E, H, D]. E means number of experts.
    w1: torch.Tensor,  # [B, E, D, H]
    w2: torch.Tensor,  # [B, E, H, D]
    q_permuted: torch.Tensor,  # [num_chunks * b, num_actived_tokens, d]
    k_permuted: torch.Tensor,  # [num_chunks * b, num_actived_tokens, d]
    v_permuted: torch.Tensor,  # [num_chunks * b, num_actived_tokens, d]
    lr0_permuted: torch.Tensor,  # [num_chunks * b, num_actived_tokens]
    lr1_permuted: torch.Tensor,  # [num_chunks * b, num_actived_tokens]
    lr2_permuted: torch.Tensor,  # [num_chunks * b, num_actived_tokens]
    momentum: torch.Tensor,  # [b, s, 1], fp32
    k_group_sizes: torch.Tensor,  # [num_chunks * B, E] int
    q_group_sizes: torch.Tensor,  # [num_chunks * B, E] int
):

    w0_w2 = torch.cat([w0, w2], dim=2)
    w0_w2_norm = w0_w2.norm(dim=3, keepdim=True)  # [b, 2*Hidden]
    w1_norm = w1.norm(dim=3, keepdim=True)  # [b, D]

    w0_w2_main = w0_w2
    w1_main = w1

    w0_w2 = w0_w2.to(torch.bfloat16)
    w1 = w1.to(torch.bfloat16)

    dw1_momentum = torch.zeros_like(w1_main)
    dw0_w2_momentum = torch.zeros_like(w0_w2_main)

    batch_size = w0.shape[0]
    num_chunks = q_permuted.shape[0] // batch_size

    # [B, E, num_chunk * chunk_size]

    momentum = rearrange(momentum, "b (n c) d -> n b (c d)", n=num_chunks).contiguous()

    output_permuted = torch.zeros_like(q_permuted)

    ## compute the moe logits
    ## Maybe it's a good idea to move this function outside of the ttt function
    # [num_chunks, B, E, T], [num_chunks, B, E], [num_chunks, B, E, T]

    e_index = 0

    # TODO, maybe multiple the k_router_probs onto the vi, to indicates adpative gradients.
    for chunk_idx in range(num_chunks - 1):

        s_index = chunk_idx * batch_size
        e_index = s_index + batch_size

        # [b, num_actived_tokens, d]
        ki = k_permuted[s_index:e_index]
        # [b, num_actived_tokens, d]
        vi = v_permuted[s_index:e_index]
        # [b, num_actived_tokens, d]
        qi = q_permuted[s_index:e_index]
        # [b, num_actived_tokens]
        lr1i = lr1_permuted[s_index:e_index]
        lr2i = lr2_permuted[s_index:e_index]
        lr0i = lr0_permuted[s_index:e_index]
        ki_group_sizes = k_group_sizes[s_index:e_index]
        qi_group_sizes = q_group_sizes[s_index:e_index]

        o_permuted = grouped_swiglu_ffn_fwd(w0_w2, w1, qi, qi_group_sizes)

        output_permuted[s_index:e_index] = o_permuted

        # use previous w0 and w1 to get the final output
        # [b, dh, dk] @ [b, dk, l] -> [b, dh, l]
        dw0_w2, dw1 = grouped_lact_swiglu_ffn_fast_weight_grads(
            w0_w2,
            w1,
            ki,
            vi,
            lr0i,
            lr1i,
            lr2i,
            ki_group_sizes,
        )

        m_i = momentum[chunk_idx].contiguous()
        m_i = m_i.mean(dim=1, keepdim=True).unsqueeze(dim=-1).unsqueeze(dim=-1)

        dw0_w2 = dw0_w2 + dw0_w2_momentum * m_i
        dw1 = dw1 + dw1_momentum * m_i
        dw0_w2_momentum = dw0_w2
        dw1_momentum = dw1

        w0_w2_main = w0_w2_main + dw0_w2
        w1_main = w1_main + dw1

        w0_w2 = w0_w2_main / (w0_w2_main.norm(dim=3, keepdim=True) + 1e-5) * w0_w2_norm
        w1 = w1_main / (w1_main.norm(dim=3, keepdim=True) + 1e-5) * w1_norm
        w0_w2 = w0_w2.to(torch.bfloat16)
        w1 = w1.to(torch.bfloat16)

    # for the last chunk, don't update the fast weights, directly apply the fast weights to the query.
    s_index = e_index

    qi = q_permuted[s_index:].contiguous()
    qi_group_sizes = q_group_sizes[s_index:]

    o_permuted = grouped_swiglu_ffn_fwd(w0_w2, w1, qi, qi_group_sizes)

    output_permuted[s_index:] = o_permuted

    return output_permuted


@torch.compile()
def block_causal_lact_swiglu_shared_expert(
    w0: torch.Tensor,  # [B, H, D]. E means number of experts.
    w1: torch.Tensor,  # [B, D, H]
    w2: torch.Tensor,  # [B, H, D]
    q: torch.Tensor,  # [num_chunks * b, chunk_size, d]
    k: torch.Tensor,  # [num_chunks * b, chunk_size, d]
    v: torch.Tensor,  # [num_chunks * b, chunk_size, d]
    lr0: torch.Tensor,  # [num_chunks * b, chunk_size]
    lr1: torch.Tensor,  # [num_chunks * b, chunk_size]
    lr2: torch.Tensor,  # [num_chunks * b, chunk_size]
    momentum: torch.Tensor,  # [b, s, 1], fp32
):

    # adding detach here sometimes improves stability.
    w0_norm = w0.norm(dim=2, keepdim=True)
    w1_norm = w1.norm(dim=2, keepdim=True)
    w2_norm = w2.norm(dim=2, keepdim=True)

    w0_main = w0
    w2_main = w2
    w1_main = w1

    w0 = w0.to(torch.bfloat16)
    w2 = w2.to(torch.bfloat16)
    w1 = w1.to(torch.bfloat16)

    if momentum is not None:
        dw1_momentum = torch.zeros_like(w1_main)
        dw0_momentum = torch.zeros_like(w0_main)
        dw2_momentum = torch.zeros_like(w2_main)

    batch_size = w0.shape[0]
    num_chunks = q.shape[0] // batch_size

    # [B, E, num_chunk * chunk_size]

    momentum = rearrange(momentum, "b (n c) d -> n b (c d)", n=num_chunks).contiguous()

    q = q.transpose(1, 2)  # [nc*b, dk, l]
    v = v.transpose(1, 2)
    output = torch.zeros_like(q)

    ## compute the moe logits
    ## Maybe it's a good idea to move this function outside of the ttt function
    # [num_chunks, B, E, T], [num_chunks, B, E], [num_chunks, B, E, T]

    e_index = 0

    # TODO, maybe multiple the k_router_probs onto the vi, to indicates adpative gradients.
    for chunk_idx in range(num_chunks - 1):

        s_index = chunk_idx * batch_size
        e_index = s_index + batch_size

        # [b, num_actived_tokens, d]
        ki = k[s_index:e_index]
        # [b, num_actived_tokens, d]
        vi = v[s_index:e_index]
        # [b, num_actived_tokens, d]
        qi = q[s_index:e_index]
        # [b, num_actived_tokens]
        lr1i = lr1[s_index:e_index]
        lr2i = lr2[s_index:e_index]
        lr0i = lr0[s_index:e_index]

        # [b, dh, dk] @ [b, dk, l] -> [b, dh, l]
        h = torch.bmm(w2, qi)
        gate = F.silu(torch.bmm(w0, qi), inplace=True)
        output[s_index:e_index, :, :] = torch.bmm(w1, gate * h)
        # [b, dv, dh] @ [b, dh, l] -> [b, dv, l] -> [b, l, dv]

        # use previous w0 and w1 to get the final output
        # [b, dh, dk] @ [b, dk, l] -> [b, dh, l]
        ########## compute the gradient and update the fast weights
        # [b, dh, dk] @ [b, dk, l] -> [b, dh, l]
        gate_before_act = torch.bmm(w0, ki.transpose(1, 2))
        hidden_before_mul = torch.bmm(w2, ki.transpose(1, 2))

        hidden = F.silu(gate_before_act, inplace=False) * hidden_before_mul

        # [b, dh, dv] @ [b, dv, l] -> [b, dh, l]
        dhidden = torch.bmm(w1.transpose(1, 2), vi)

        dhidden_before_mul = dhidden * F.silu(gate_before_act, inplace=False)

        dgate = dhidden * hidden_before_mul
        dgate_before_act = silu_backprop(dgate, gate_before_act)

        # [b, dv, l] @ [b, l, dh] -> [b, dv, dh]
        # it's better to cast the mat to bf16 before bmm.
        dw1 = torch.bmm(vi, (hidden.transpose(1, 2) * lr1i).type_as(vi))  # [b, d, d]
        # [b, dh, l] @ [b, l, dk] -> [b, dh, dk]
        dw0 = torch.bmm(dgate_before_act, (ki * lr0i).type_as(dgate_before_act))
        dw2 = torch.bmm(dhidden_before_mul, (ki * lr2i).type_as(dhidden_before_mul))

        m_i = momentum[chunk_idx].contiguous()
        m_i = m_i.mean(dim=1, keepdim=False).unsqueeze(dim=-1).unsqueeze(dim=-1)

        dw0 = dw0 + dw0_momentum * m_i
        dw1 = dw1 + dw1_momentum * m_i
        dw2 = dw2 + dw2_momentum * m_i
        dw0_momentum = dw0
        dw1_momentum = dw1
        dw2_momentum = dw2

        w0_main = w0_main + dw0
        w2_main = w2_main + dw2
        w1_main = w1_main + dw1

        w0 = w0_main / (w0_main.norm(dim=2, keepdim=True) + 1e-5) * w0_norm
        w1 = w1_main / (w1_main.norm(dim=2, keepdim=True) + 1e-5) * w1_norm
        w2 = w2_main / (w2_main.norm(dim=2, keepdim=True) + 1e-5) * w2_norm

        w0 = w0.to(torch.bfloat16)
        w1 = w1.to(torch.bfloat16)
        w2 = w2.to(torch.bfloat16)

    # for the last chunk, don't update the fast weights, directly apply the fast weights to the query.
    s_index = e_index

    qi = q[s_index:]
    # use the last w0 and w1 to get the final output
    # [b, dh, dk] @ [b, dk, l] -> [b, dh, l]
    h = torch.bmm(w2, qi)
    gate = F.silu(torch.bmm(w0, qi), inplace=True)
    # [b, dv, dh] @ [b, dh, l] -> [b, dv, l]
    output[s_index:] = torch.bmm(w1, gate * h)
    # -> [b, l, dv]
    return output.transpose(1, 2)


def _test_fused_kernel():
    B, L, D, H = 4, 8192, 512, 1024
    chunk_size = 2048
    use_muon = True
    fp32_states = True
    dtype = torch.bfloat16

    w0 = torch.randn(B, H, D).to("cuda").to(dtype)
    w1 = torch.randn(B, D, H).to("cuda").to(dtype)
    w2 = torch.randn(B, H, D).to("cuda").to(dtype)
    q = torch.randn(B, L, D).to("cuda").to(dtype)
    k = torch.randn(B, L, D).to("cuda").to(dtype)
    v = torch.randn(B, L, D).to("cuda").to(dtype)
    lr0 = torch.randn(B, L, 1).to("cuda").to(dtype)
    lr1 = torch.randn(B, L, 1).to("cuda").to(dtype)
    lr2 = torch.randn(B, L, 1).to("cuda").to(dtype)
    momentum = torch.randn(B, L, 1).to("cuda").to(dtype)

    with torch.autocast(device_type="cuda", enabled=True, dtype=dtype):
        output = block_causal_lact_swiglu_fused_kernel(
            w0,
            w1,
            w2,
            q,
            k,
            v,
            lr0,
            lr1,
            lr2,
            chunk_size,
            use_muon,
            momentum,
            fp32_states,
        )

    print(output.shape, output.dtype)


if __name__ == "__main__":
    _test_fused_kernel()
