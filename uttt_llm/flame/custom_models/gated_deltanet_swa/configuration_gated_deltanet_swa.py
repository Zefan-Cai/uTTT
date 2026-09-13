from __future__ import annotations

from fla.models.gated_deltanet.configuration_gated_deltanet import GatedDeltaNetConfig


class GatedDeltaNetWindowMixConfig(GatedDeltaNetConfig):
    model_type = "gated_deltanet_swa"

    def __init__(
        self,
        window_size: int | None = 4096,
        rope_theta: float = 1000000.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.attn_head_dim = self.head_dim
        self.window_size = window_size
        self.rope_theta = rope_theta
