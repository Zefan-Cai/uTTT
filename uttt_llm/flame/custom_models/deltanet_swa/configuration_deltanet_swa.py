from __future__ import annotations

from fla.models.delta_net.configuration_delta_net import DeltaNetConfig


class DeltaNetWindowMixConfig(DeltaNetConfig):
    model_type = "deltanet_swa"

    def __init__(
        self,
        attn_head_dim: int = 64,
        window_size: int | None = 4096,
        rope_theta: float = 1000000.0,
        use_window_attention: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.attn_head_dim = attn_head_dim
        self.window_size = window_size
        self.rope_theta = rope_theta
        # False drops the sliding-window-attention branch: the mixer runs alone (no-SWA ablation).
        self.use_window_attention = use_window_attention
