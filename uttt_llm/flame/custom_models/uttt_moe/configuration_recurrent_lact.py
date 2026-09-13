from __future__ import annotations

from .configuration_recurrent_lact_base import RecurrentLactRefConfig as BaseRecurrentLactRefConfig


class RecurrentLactRefMoeGlobalV45Config(BaseRecurrentLactRefConfig):
    model_type = "uttt_moe"

    def __init__(
        self,
        *args,
        memory_num_experts: int = 1,
        memory_num_active_experts: int = 1,
        memory_router_alpha: float = 1.0,
        memory_router_use_sigmoid: bool = False,
        memory_router_from_v: bool = False,
        memory_router_share_mode: str = "share_all",
        memory_router_type: str = "softmax",
        memory_router_combine_mode: str = "selected_prob",
        memory_router_scale: float = 0.0,
        memory_router_norm_eps: float = 1e-6,
        memory_pool_mode: str = "shared",
        memory_lb_loss_alpha: float = 0.0,
        memory_lb_scope: str = "layer",
        memory_lb_mode: str = "loss",
        memory_loss_free_update_rate: float = 1e-4,
        memory_loss_free_bias_max: float = 2.0,
        memory_loss_free_warmup_steps: int = 0,
        memory_loss_free_update_interval: int = 1,
        memory_update_backward_mode: str = "stop_state",
        memory_detach_m_coeff: bool = False,
        memory_update_aggregate_write: bool = False,
        memory_aggregate_momentum_mode: str = "none",
        memory_aggregate_lr_scale: float = 1.0,
        **kwargs,
    ):
        kwargs.pop("native_reference_impl", None)
        super().__init__(*args, **kwargs)

        self.memory_num_experts = memory_num_experts
        self.memory_num_active_experts = memory_num_active_experts
        self.memory_router_alpha = memory_router_alpha
        self.memory_router_use_sigmoid = memory_router_use_sigmoid
        self.memory_router_from_v = memory_router_from_v
        self.memory_router_share_mode = memory_router_share_mode
        self.memory_router_type = memory_router_type
        self.memory_router_combine_mode = memory_router_combine_mode
        self.memory_router_scale = memory_router_scale
        self.memory_router_norm_eps = memory_router_norm_eps
        self.memory_pool_mode = memory_pool_mode
        self.memory_lb_loss_alpha = memory_lb_loss_alpha
        self.memory_lb_scope = memory_lb_scope
        self.memory_lb_mode = memory_lb_mode
        self.memory_loss_free_update_rate = memory_loss_free_update_rate
        self.memory_loss_free_bias_max = memory_loss_free_bias_max
        self.memory_loss_free_warmup_steps = memory_loss_free_warmup_steps
        self.memory_loss_free_update_interval = memory_loss_free_update_interval
        if memory_update_backward_mode == "first_order":
            memory_update_backward_mode = "stop_state"
        self.memory_update_backward_mode = memory_update_backward_mode
        self.memory_detach_m_coeff = memory_detach_m_coeff
        self.memory_update_aggregate_write = memory_update_aggregate_write
        self.memory_aggregate_momentum_mode = memory_aggregate_momentum_mode
        # This is a post-Muon aggregate step-strength knob, not a linear LR:
        # master weights are renormalized to their old norm after each update.
        self.memory_aggregate_lr_scale = memory_aggregate_lr_scale

        if self.memory_num_experts < 1:
            raise ValueError("`memory_num_experts` must be >= 1.")
        if self.memory_num_active_experts < 1:
            raise ValueError("`memory_num_active_experts` must be >= 1.")
        if self.memory_num_active_experts > self.memory_num_experts:
            raise ValueError(
                "`memory_num_active_experts` cannot be greater than `memory_num_experts`.",
            )
        if self.memory_pool_mode != "shared":
            raise ValueError("global-v4.5 currently supports only `memory_pool_mode='shared'`.")
        if self.memory_router_share_mode not in {"share_all", "per_layer"}:
            raise ValueError("`memory_router_share_mode` must be 'share_all' or 'per_layer'.")
        if self.memory_router_type not in {"softmax", "norm_softmax", "norm_relu"}:
            raise ValueError(
                "`memory_router_type` must be 'softmax', 'norm_softmax', or 'norm_relu', "
                f"got {self.memory_router_type!r}.",
            )
        if self.memory_router_combine_mode not in {"selected_prob", "topk_renorm"}:
            raise ValueError(
                "`memory_router_combine_mode` must be 'selected_prob' or 'topk_renorm', "
                f"got {self.memory_router_combine_mode!r}.",
            )
        if self.memory_router_scale < 0:
            raise ValueError("`memory_router_scale` must be non-negative.")
        if self.memory_router_norm_eps <= 0:
            raise ValueError("`memory_router_norm_eps` must be positive.")
        if self.memory_lb_scope not in {"layer", "pool"}:
            raise ValueError("`memory_lb_scope` must be 'layer' or 'pool'.")
        if self.memory_lb_mode not in {"loss", "loss_free", "none"}:
            raise ValueError("`memory_lb_mode` must be 'loss', 'loss_free', or 'none'.")
        if self.memory_loss_free_update_rate < 0:
            raise ValueError("`memory_loss_free_update_rate` must be non-negative.")
        if self.memory_loss_free_bias_max < 0:
            raise ValueError("`memory_loss_free_bias_max` must be non-negative.")
        if self.memory_loss_free_warmup_steps < 0:
            raise ValueError("`memory_loss_free_warmup_steps` must be non-negative.")
        if self.memory_loss_free_update_interval < 1:
            raise ValueError("`memory_loss_free_update_interval` must be >= 1.")
        if self.memory_update_backward_mode not in {"full", "first_order_state", "stop_state"}:
            raise ValueError(
                "`memory_update_backward_mode` must be 'full', 'first_order_state', or "
                "'stop_state', "
                f"got {self.memory_update_backward_mode!r}.",
            )
        if self.memory_aggregate_momentum_mode not in {"none", "pool_mean"}:
            raise ValueError(
                "`memory_aggregate_momentum_mode` must be 'none' or 'pool_mean', "
                f"got {self.memory_aggregate_momentum_mode!r}.",
            )
        if self.memory_aggregate_lr_scale <= 0:
            raise ValueError("`memory_aggregate_lr_scale` must be positive.")
        if self.memory_update_aggregate_write:
            if self.memory_aggregate_momentum_mode == "pool_mean" and not self.use_momentum:
                raise ValueError(
                    "`memory_aggregate_momentum_mode='pool_mean'` requires "
                    "`use_momentum=True`.",
                )
            if self.memory_aggregate_momentum_mode == "none" and self.use_momentum:
                raise ValueError(
                    "`memory_aggregate_momentum_mode='none'` with "
                    "`use_momentum=True` would silently drop momentum. Set "
                    "`use_momentum=False` or use `memory_aggregate_momentum_mode='pool_mean'`.",
                )
        else:
            if self.memory_aggregate_momentum_mode != "none":
                raise ValueError(
                    "`memory_aggregate_momentum_mode` requires "
                    "`memory_update_aggregate_write=True`.",
                )
            if self.memory_aggregate_lr_scale != 1.0:
                raise ValueError(
                    "`memory_aggregate_lr_scale` requires "
                    "`memory_update_aggregate_write=True`.",
                )


RecurrentLactRefConfig = RecurrentLactRefMoeGlobalV45Config
RecurrentLactRefMoeConfig = RecurrentLactRefMoeGlobalV45Config
