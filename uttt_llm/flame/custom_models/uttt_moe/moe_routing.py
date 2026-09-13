import math

import torch
import torch.nn.functional as F


def create_router_mask_sizes_probs(
    router_logits: torch.Tensor,
    topk: int,
    alpha: float = 1.0,
    use_sigmoid: bool = False,
    router_type: str = "softmax",
    combine_mode: str = "selected_prob",
    router_scale: float = 0.0,
    norm_eps: float = 1e-6,
    routing_bias: torch.Tensor | None = None,
):
    num_experts = router_logits.shape[-2]
    if use_sigmoid:
        raise NotImplementedError(
            "Sigmoid routing is not implemented for ttt_dense_base_moe.",
        )

    logits = router_logits.to(torch.float32)
    if router_type == "softmax":
        selection_scores = logits
        balance_probs = torch.softmax(logits, dim=-2)
    elif router_type == "norm_softmax":
        scale = router_scale if router_scale > 0 else math.sqrt(float(num_experts))
        z = logits / logits.norm(dim=-2, keepdim=True).clamp_min(norm_eps)
        selection_scores = z * scale
        balance_probs = torch.softmax(selection_scores, dim=-2)
    elif router_type == "norm_relu":
        scale = router_scale if router_scale > 0 else math.sqrt(float(num_experts))
        z = logits / logits.norm(dim=-2, keepdim=True).clamp_min(norm_eps)
        selection_scores = F.relu(z) * scale
        score_sum = selection_scores.sum(dim=-2, keepdim=True)
        uniform_probs = torch.full_like(selection_scores, 1.0 / float(num_experts))
        balance_probs = torch.where(
            score_sum > 0,
            selection_scores / score_sum.clamp_min(1e-9),
            uniform_probs,
        )
    else:
        raise ValueError(
            "`router_type` must be 'softmax', 'norm_softmax', or 'norm_relu', "
            f"got {router_type!r}.",
        )

    routing_scores = selection_scores if routing_bias is None else selection_scores + routing_bias
    _, topk_idx = torch.topk(routing_scores, k=topk, dim=-2)
    router_mask = torch.zeros_like(router_logits, dtype=torch.bool)
    router_mask.scatter_(-2, topk_idx, True)

    group_sizes = router_mask.sum(dim=-1).to(torch.int32)

    if combine_mode == "selected_prob":
        combine_weights = balance_probs.to(router_logits.dtype) * alpha
    elif combine_mode == "topk_renorm":
        selected = balance_probs * router_mask.to(balance_probs.dtype)
        combine_weights = selected / selected.sum(dim=-2, keepdim=True).clamp_min(1e-9)
        combine_weights = combine_weights.to(router_logits.dtype) * alpha
    else:
        raise ValueError(
            "`combine_mode` must be 'selected_prob' or 'topk_renorm', "
            f"got {combine_mode!r}.",
        )

    balance_probs = balance_probs.to(router_logits.dtype)
    router_probs_mean = balance_probs.mean(dim=0).mean(dim=-1)
    group_sizes_mean = group_sizes.to(torch.float32).mean(dim=0)
    frequency = group_sizes_mean / group_sizes_mean.sum()
    lb_loss = (router_probs_mean * frequency).sum() * num_experts

    return router_mask, group_sizes, combine_weights, balance_probs, lb_loss
