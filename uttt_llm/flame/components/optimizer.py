from __future__ import annotations

from typing import Any, TypeVar

import torch
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict
from torch.optim import Optimizer
from torchtitan.components.ft import FTManager
from torchtitan.components.optimizer import (
    FTOptimizersContainer,
    OptimizersContainer,
    OptimizersInBackwardContainer,
    build_optimizers as build_torchtitan_optimizers,
)

T = TypeVar("T", bound=Optimizer)


def _no_weight_decay_name(name: str) -> bool:
    return (
        name.endswith("prev_chunk_alpha")
        or name.endswith("prev_chunk_alphas")
        or name.endswith("first_chunk_alpha")
        or name.endswith("first_chunk_alphas")
    )


def _has_alpha_params(model_parts: list[nn.Module]) -> bool:
    return any(_no_weight_decay_name(name) for model in model_parts for name, _ in model.named_parameters())


def _use_alpha_no_weight_decay(model_parts: list[nn.Module], job_config) -> bool:
    return not getattr(job_config.optimizer, "alpha_weight_decay", False) and _has_alpha_params(model_parts)


def _param_groups(model: nn.Module, weight_decay: float) -> tuple[list[dict[str, Any]], list[nn.Parameter]]:
    decay, no_decay, all_params = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        all_params.append(param)
        (no_decay if _no_weight_decay_name(name) else decay).append(param)

    groups: list[dict[str, Any]] = []
    if decay:
        groups.append({"params": decay})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return groups, all_params


class NoDecayAlphaOptimizersContainer(OptimizersContainer[T]):
    def __init__(
        self,
        model_parts: list[nn.Module],
        optimizer_cls: type[T],
        optimizer_kwargs: dict[str, Any],
    ) -> None:
        all_params = []
        self.optimizers = []
        self.model_parts = model_parts
        weight_decay = optimizer_kwargs["weight_decay"]
        for model in self.model_parts:
            groups, params = _param_groups(model, weight_decay)
            self.optimizers.append(optimizer_cls(groups, **optimizer_kwargs))
            all_params.extend(params)
        self._validate_length(len(self.model_parts))
        self._post_init(all_params, optimizer_kwargs)


class NoDecayAlphaOptimizersInBackwardContainer(OptimizersInBackwardContainer):
    def __init__(
        self,
        model_parts: list[nn.Module],
        optimizer_cls: type[T],
        optimizer_kwargs: dict[str, Any],
    ) -> None:
        all_params = []
        self.model_parts = model_parts

        optim_dict = {}
        for model in self.model_parts:
            for name, param in model.named_parameters():
                all_params.append(param)
                if not param.requires_grad:
                    continue
                kwargs = dict(optimizer_kwargs)
                if _no_weight_decay_name(name):
                    kwargs["weight_decay"] = 0.0
                optim_dict[param] = optimizer_cls([param], **kwargs)

        def optim_hook(param):
            optim_dict[param].step()
            optim_dict[param].zero_grad()

        for model in self.model_parts:
            for param in model.parameters():
                if param.requires_grad:
                    param.register_post_accumulate_grad_hook(optim_hook)

        self.optimizers = list(optim_dict.values())
        self._validate_length(sum(len(list(model.parameters())) for model in self.model_parts))
        self._post_init(all_params, optimizer_kwargs)


class NoDecayAlphaFTOptimizersContainer(FTOptimizersContainer, NoDecayAlphaOptimizersContainer[T]):
    def __init__(
        self,
        model_parts: list[nn.Module],
        optimizer_cls: type[T],
        optimizer_kwargs: dict[str, Any],
        ft_manager,
        use_ft_optimizer: bool = True,
    ) -> None:
        NoDecayAlphaOptimizersContainer.__init__(self, model_parts, optimizer_cls, optimizer_kwargs)
        _ = {
            k: v
            for sd in map(get_optimizer_state_dict, model_parts, self.optimizers)
            for k, v in sd.items()
        }
        self.cache_state_dict: dict[str, Any] = {}
        self._ft_optimizer = __import__("torchft").Optimizer(ft_manager, self)
        self._use_ft_optimizer: bool = use_ft_optimizer


def build_optimizers(
    model_parts: list[nn.Module],
    job_config,
    ft_manager: FTManager,
):
    if not _use_alpha_no_weight_decay(model_parts, job_config):
        return build_torchtitan_optimizers(model_parts, job_config, ft_manager)

    optim_in_bwd = job_config.optimizer.early_step_in_backward
    if optim_in_bwd and job_config.parallelism.pipeline_parallel_degree > 1:
        raise NotImplementedError("Optimizers in backward is not supported with pipeline parallelism.")

    optimizer_kwargs = {
        "lr": job_config.optimizer.lr,
        "betas": (job_config.optimizer.beta1, job_config.optimizer.beta2),
        "eps": job_config.optimizer.eps,
        "weight_decay": job_config.optimizer.weight_decay,
        "fused": job_config.optimizer.implementation == "fused",
        "foreach": job_config.optimizer.implementation == "foreach",
    }

    optimizer_classes = {
        "Adam": torch.optim.Adam,
        "AdamW": torch.optim.AdamW,
    }
    if job_config.optimizer.name not in optimizer_classes:
        raise NotImplementedError(f"Optimizer {job_config.optimizer.name} not added.")
    optimizer_cls = optimizer_classes[job_config.optimizer.name]

    if optim_in_bwd and ft_manager.enabled:
        raise ValueError("TorchFT is not supported with optimizers in backward.")
    if optim_in_bwd:
        return NoDecayAlphaOptimizersInBackwardContainer(model_parts, optimizer_cls, optimizer_kwargs)
    if ft_manager.enabled:
        return NoDecayAlphaFTOptimizersContainer(
            model_parts,
            optimizer_cls,
            optimizer_kwargs,
            ft_manager.manager,
            use_ft_optimizer=job_config.fault_tolerance.semi_sync_method is None,
        )
    return NoDecayAlphaOptimizersContainer(model_parts, optimizer_cls, optimizer_kwargs)
