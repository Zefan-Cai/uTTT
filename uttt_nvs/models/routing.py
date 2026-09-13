import torch


def create_router_mask_sizes_probs(
    router_logits: torch.Tensor,
    topk: int,
    alpha: float = 1.0,
    use_sigmoid: bool = False,
):
    """
    router_logits: [B, E, L]
    topk: int
    alpha: float
    use_sigmoid: False -> Softmax, True -> Sigmoid
    """
    num_experts = router_logits.shape[-2]
    _, topk_idx = torch.topk(router_logits, k=topk, dim=-2)
    router_mask = torch.zeros_like(router_logits, dtype=torch.bool)
    router_mask.scatter_(-2, topk_idx, True)

    # [B, E, L] --> [B, E], how many tokens are routed to each expert
    group_sizes = router_mask.sum(dim=-1).to(torch.int32)

    # Compute prob; normalized over the E dimension.
    if use_sigmoid:
        assert False, "Please do not use sigmoid now; the average is not correct."
        # Sigmoid should return router_probs without normalization.
        # However, 

        # router_probs = torch.sigmoid(router_logits.to(torch.float32)) * alpha
        # router_probs = router_probs / router_probs.sum(dim=-2, keepdim=True)
    else:
        router_probs = torch.softmax(router_logits.to(torch.float32), dim=-2).to(
            router_logits.dtype
        ) * alpha

    # Compute load balancing loss.
    # [B, E, L] --> [E], mean reduction over B and L
    router_probs_mean = router_probs.mean(dim=0).mean(dim=-1)
    group_sizes_mean = group_sizes.to(torch.float32).mean(dim=0)
    frequency = group_sizes_mean / group_sizes_mean.sum()
    lb_loss = (router_probs_mean * frequency).sum() * num_experts

    return router_mask, group_sizes, router_probs, lb_loss


