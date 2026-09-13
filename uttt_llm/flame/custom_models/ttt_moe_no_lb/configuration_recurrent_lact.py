from __future__ import annotations

from .configuration_recurrent_lact_base import (
    RecurrentLactRefConfig as BaseRecurrentLactRefConfig,
)


class RecurrentLactRefMoeV3Config(BaseRecurrentLactRefConfig):
    model_type = "ttt_moe_no_lb"

    def __init__(
        self,
        *args,
        memory_num_experts: int = 1,
        memory_num_active_experts: int = 1,
        memory_router_alpha: float = 1.0,
        memory_router_use_sigmoid: bool = False,
        memory_router_from_v: bool = False,
        memory_lb_loss_alpha: float = 0.0,
        **kwargs,
    ):
        kwargs.pop("native_reference_impl", None)
        super().__init__(*args, **kwargs)

        self.memory_num_experts = memory_num_experts
        self.memory_num_active_experts = memory_num_active_experts
        self.memory_router_alpha = memory_router_alpha
        self.memory_router_use_sigmoid = memory_router_use_sigmoid
        self.memory_router_from_v = memory_router_from_v
        self.memory_lb_loss_alpha = memory_lb_loss_alpha

        if self.memory_num_experts < 1:
            raise ValueError("`memory_num_experts` must be >= 1.")
        if self.memory_num_active_experts < 1:
            raise ValueError("`memory_num_active_experts` must be >= 1.")
        if self.memory_num_active_experts > self.memory_num_experts:
            raise ValueError(
                "`memory_num_active_experts` cannot be greater than `memory_num_experts`.",
            )


RecurrentLactRefConfig = RecurrentLactRefMoeV3Config
