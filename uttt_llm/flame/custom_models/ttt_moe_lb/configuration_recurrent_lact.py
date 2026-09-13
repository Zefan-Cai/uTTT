from __future__ import annotations

from .configuration_recurrent_lact_base import (
    RecurrentLactRefConfig as BaseRecurrentLactRefConfig,
)


class RecurrentLactRefMoeV5Config(BaseRecurrentLactRefConfig):
    model_type = "ttt_moe_lb"

    def __init__(
        self,
        *args,
        memory_num_experts: int = 1,
        memory_num_active_experts: int = 1,
        memory_router_alpha: float = 1.0,
        memory_router_use_sigmoid: bool = False,
        memory_router_from_v: bool = False,
        memory_lb_loss_alpha: float = 0.0,
        memory_lb_mode: str = "loss",
        memory_lb_scope: str = "layer",
        memory_loss_free_update_rate: float = 1e-4,
        memory_loss_free_bias_max: float = 2.0,
        memory_loss_free_warmup_steps: int = 0,
        memory_loss_free_update_interval: int = 1,
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
        self.memory_lb_mode = memory_lb_mode
        self.memory_lb_scope = memory_lb_scope
        self.memory_loss_free_update_rate = memory_loss_free_update_rate
        self.memory_loss_free_bias_max = memory_loss_free_bias_max
        self.memory_loss_free_warmup_steps = memory_loss_free_warmup_steps
        self.memory_loss_free_update_interval = memory_loss_free_update_interval

        if self.memory_num_experts < 1:
            raise ValueError("`memory_num_experts` must be >= 1.")
        if self.memory_num_active_experts < 1:
            raise ValueError("`memory_num_active_experts` must be >= 1.")
        if self.memory_num_active_experts > self.memory_num_experts:
            raise ValueError(
                "`memory_num_active_experts` cannot be greater than `memory_num_experts`.",
            )
        if self.memory_lb_mode not in {"loss", "loss_free", "none"}:
            raise ValueError("`memory_lb_mode` must be 'loss', 'loss_free', or 'none'.")
        if self.memory_lb_scope not in {"layer", "pool"}:
            raise ValueError("`memory_lb_scope` must be 'layer' or 'pool'.")
        if self.memory_loss_free_update_rate < 0:
            raise ValueError("`memory_loss_free_update_rate` must be non-negative.")
        if self.memory_loss_free_bias_max < 0:
            raise ValueError("`memory_loss_free_bias_max` must be non-negative.")
        if self.memory_loss_free_warmup_steps < 0:
            raise ValueError("`memory_loss_free_warmup_steps` must be non-negative.")
        if self.memory_loss_free_update_interval < 1:
            raise ValueError("`memory_loss_free_update_interval` must be >= 1.")


RecurrentLactRefConfig = RecurrentLactRefMoeV5Config
