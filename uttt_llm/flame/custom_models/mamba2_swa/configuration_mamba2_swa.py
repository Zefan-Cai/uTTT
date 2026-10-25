from __future__ import annotations

from typing import Dict, Optional

from fla.models.mamba2.configuration_mamba2 import Mamba2Config


class Mamba2WindowMixConfig(Mamba2Config):
    model_type = "mamba2_swa"

    def __init__(
        self,
        attn_head_dim: int = 64,
        attn_num_heads: int | None = None,
        window_size: int | None = 4096,
        rope_theta: float = 1000000.0,
        use_window_attention: bool = True,
        attn: Optional[Dict] = None,
        hidden_ratio: float | None = None,
        intermediate_size: int | None = None,
        fuse_swiglu: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.attn_head_dim = attn_head_dim
        # SWA branch width = attn_num_heads * attn_head_dim. None keeps the legacy behaviour
        # (width tied to the Mamba-2 `expand`, i.e. intermediate_size / attn_head_dim heads).
        self.attn_num_heads = attn_num_heads
        self.window_size = window_size
        self.rope_theta = rope_theta
        # False drops the sliding-window-attention branch: the mixer runs alone (no-SWA ablation).
        self.use_window_attention = use_window_attention
        self.attn = attn
        # Optional SwiGLU MLP after the mixer (as in the DeltaNet/GatedDeltaNet window-mix blocks).
        # None keeps the legacy Mamba-2 block (mixer only).
        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size
        self.fuse_swiglu = fuse_swiglu

        if attn is not None:
            if not isinstance(attn, dict):
                raise ValueError("attn must be a dictionary")
            if "layers" not in attn:
                raise ValueError("Layer indices must be provided to initialize hybrid attention layers")
            if "num_heads" not in attn:
                raise ValueError("Number of heads must be provided to initialize hybrid attention layers")
            attn["num_kv_heads"] = attn.get("num_kv_heads", attn["num_heads"])
            attn["qkv_bias"] = attn.get("qkv_bias", False)
            attn["window_size"] = attn.get("window_size", None)
            attn["rope_theta"] = attn.get("rope_theta", rope_theta)
