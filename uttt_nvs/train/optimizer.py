# Doc: https://huggingface.co/docs/transformers/v4.30.0/en/main_classes/optimizer_schedules#transformers.get_constant_schedule_with_warmup

import inspect
import math

import torch
from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)
from uttt_nvs.train.distributed import print_rank0


def configure_optimizer(model, weight_decay, learning_rate, betas):
    # start with all of the candidate parameters
    all_param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in all_param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params, nodecay_params = [], []
    decay_names, nodecay_names = [], []
    for n, p in param_dict.items():
        # if nerf_mlp uses tinycudann, then its mlp parameters are 1D due to flattening
        if p.dim() >= 2:
            decay_params.append(p)
            decay_names.append(n)
        else:
            nodecay_params.append(p)
            nodecay_names.append(n)

    print_rank0(
        f"Decay params ({len(decay_params)}): {decay_names}\n"
        f"No Decay params ({len(nodecay_params)}): {nodecay_names}"
    )

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]

    # Create AdamW optimizer and use the fused version if it is available
    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and next(model.parameters()).is_cuda
    # print(f"Using fused AdamW? {use_fused}")
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(
        optim_groups, lr=learning_rate, betas=betas, **extra_args
    )

    return optimizer, param_dict, all_param_dict


def configure_lr_scheduler(
    optimizer,
    total_train_steps,
    warm_up_steps,
    scheduler_type="cosine",
    warmup_init_ratio=0.0,
):
    if warmup_init_ratio == 0.0:
        if scheduler_type == "linear":
            lr_scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warm_up_steps,
                num_training_steps=total_train_steps,
            )
        elif scheduler_type == "cosine":
            lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warm_up_steps,
                num_training_steps=total_train_steps,
            )
        elif scheduler_type == "constant":
            lr_scheduler = get_constant_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warm_up_steps,
            )
        else:
            raise ValueError(f"Not support LR scheduler type {scheduler_type}.")
        return lr_scheduler

    if not (0.0 <= warmup_init_ratio <= 1.0):
        raise ValueError("warmup_init_ratio must be in [0, 1].")

    def lr_lambda(current_step: int):
        if warm_up_steps > 0 and current_step < warm_up_steps:
            progress = float(current_step) / float(max(1, warm_up_steps))
            return warmup_init_ratio + (1.0 - warmup_init_ratio) * progress

        if scheduler_type == "constant":
            return 1.0
        if scheduler_type == "linear":
            if total_train_steps <= warm_up_steps:
                return 1.0
            remaining_steps = total_train_steps - current_step
            decay_steps = max(1, total_train_steps - warm_up_steps)
            return max(0.0, float(remaining_steps) / float(decay_steps))
        if scheduler_type == "cosine":
            if total_train_steps <= warm_up_steps:
                return 1.0
            progress = float(current_step - warm_up_steps) / float(
                max(1, total_train_steps - warm_up_steps)
            )
            progress = min(max(progress, 0.0), 1.0)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        raise ValueError(f"Not support LR scheduler type {scheduler_type}.")

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    return lr_scheduler
