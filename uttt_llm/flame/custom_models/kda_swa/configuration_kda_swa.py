from __future__ import annotations

from fla.models.kda.configuration_kda import KDAConfig


class KDAWindowMixConfig(KDAConfig):
    model_type = "kda_swa"

    def __init__(
        self,
        attn_head_dim: int | None = None,
        window_size: int | None = 4096,
        rope_theta: float = 1000000.0,
        use_window_attention: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # SWA head size. None ties it to the KDA head_dim; set 64 for the standard
        # 12 x 64 SWA split of the shared 768-wide q/k projections.
        self.attn_head_dim = attn_head_dim if attn_head_dim is not None else self.head_dim
        self.window_size = window_size
        self.rope_theta = rope_theta
        # False drops the sliding-window-attention branch: the mixer runs alone (no-SWA ablation).
        self.use_window_attention = use_window_attention
