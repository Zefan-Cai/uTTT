from __future__ import annotations

from fla.models.mom.configuration_mom import MomConfig


class MomWindowMixConfig(MomConfig):
    model_type = "mom_swa"

    def __init__(
        self,
        attn_head_dim: int | None = None,
        window_size: int | None = 4096,
        rope_theta: float = 1000000.0,
        use_window_attention: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.attn_head_dim = attn_head_dim if attn_head_dim is not None else self.head_dim
        self.window_size = window_size
        self.rope_theta = rope_theta
        self.use_window_attention = use_window_attention
