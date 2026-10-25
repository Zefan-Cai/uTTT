from __future__ import annotations

from fla.models.gated_deltanet.configuration_gated_deltanet import GatedDeltaNetConfig


class GatedDeltaNetWindowMixConfig(GatedDeltaNetConfig):
    model_type = "gated_deltanet_swa"

    def __init__(
        self,
        attn_head_dim: int | None = None,
        window_size: int | None = 4096,
        rope_theta: float = 1000000.0,
        use_window_attention: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # SWA head size. None keeps the legacy behaviour (SWA head dim == GDN head_dim,
        # e.g. 3 heads x 256 for the 124M fwalign config); set 64 for standard 12 x 64 SWA.
        self.attn_head_dim = attn_head_dim if attn_head_dim is not None else self.head_dim
        self.window_size = window_size
        self.rope_theta = rope_theta
        # False drops the sliding-window-attention branch: the mixer runs alone (no-SWA ablation).
        self.use_window_attention = use_window_attention
