import math
import warnings
from numbers import Integral, Real
from typing import Any

from transformers.configuration_utils import PretrainedConfig


def _normalize_dense_r(value: Any) -> int:
    """Return the positive integer uTTT-Dense inner-width multiplier."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(
            f"`uttt_dense_r` must be a positive integer, got {value!r}.",
        )
    numeric_value = float(value)
    if not math.isfinite(numeric_value) or not numeric_value.is_integer():
        raise ValueError(
            f"`uttt_dense_r` must be a positive integer, got {value!r}.",
        )
    dense_r = int(numeric_value)
    if dense_r < 1:
        raise ValueError(
            f"`uttt_dense_r` must be a positive integer, got {value!r}.",
        )
    return dense_r


class RecurrentLactRefUtttDenseV3Config(PretrainedConfig):

    model_type = "uttt_dense"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        hidden_size: int = 2048,
        num_hidden_layers: int = 24,
        num_heads: int = 32,
        num_kv_heads: int | None = None,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        window_size: int | None = None,
        rope_theta: float | None = 10000.0,
        max_position_embeddings: int = 2048,
        hidden_ratio: int | None = 4,
        intermediate_size: int | None = None,
        hidden_act: str = "swish",
        initializer_range: float = 0.02,
        elementwise_affine: bool | None = True,
        norm_eps: float = 1e-6,
        use_cache: bool = True,
        pad_token_id: int | None = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = False,
        fuse_norm: bool = True,
        fuse_swiglu: bool = True,
        fuse_cross_entropy: bool = True,
        fuse_linear_cross_entropy: bool = False,
        use_l2warp: bool = False,
        vocab_size: int = 32000,
        chunk_size: int = 1024,
        memory_v_gap: int | None = None,
        memory_k_gap: int | None = None,
        attention_v_gap: int | None = None,
        attention_k_gap: int | None = None,
        shared_kv_cache: bool = False,
        # Fast Weight Memory Block Configuration
        use_memory_block: bool = True,
        fw_num_heads: int | None = None,
        fw_inter_multi: float | None = None,
        base_lr: float = 0.01,
        per_head_lr: bool = False,  # True: 3*fw_num_heads 个独立LR; False: 3个共享LR
        repeat_function: str = "repeat",  # "repeat" or "linear"
        w0_w2_low_rank: int = -1,
        fw_init_gain: float = 0.5,
        # Memory KV 配置
        memory_kv_mode: str = "to_kv",  # "reuse_kv", "to_kv"
        memory_kv_feature: str = "x_memory_in",  # "x_attn_in", "x_memory_in", "xi_out"
        memory_kv_proj: str = "ttt_kv_proj",  # "reuse_attn_kv_proj", "ttt_kv_proj"
        memory_qkv_silu: bool = False,  # 是否对 memory Q/K/V 施加 SiLU
        memory_qk_norm: bool = False,  # 是否对 memory Q/K 施加 L2 norm
        memory_qk_norm_type: str = "l2",  # "l2" or "rms"
        memory_norm_type: str = "rms",  # "l2" or "rms", controls all parameter-free norms in memory
        memory_v_norm: bool = False,  # 是否对 memory V 施加 norm (类型由 memory_norm_type 控制)
        memory_qk_rescale: bool = False,  # 对 memory 的 q/k 施加 learnable scale+offset
        learnable_ttt_scale: bool = False,  # 对 memory 输出施加 learnable per-dim scale
        ttt_prenorm: bool = False,  # True: keep raw fast-weight state and apply normalized view each step
        same_rope: bool = False,  # TTT RoPE 使用与 attention 相同的 theta 和 head_dim
        last_layer_fuse_norm: bool = True,  # 最终 norm 层是否用 fused 版本
        ttt_rope_theta: float = 0.0,  # TTT RoPE theta (0 = disabled, > 0 enables RoPE on TTT Q/K)
        use_momentum: bool = False,  # 对 fast weight 梯度施加 momentum
        use_moun: bool = False,  # 对 fast weight 梯度施加 Newton-Schulz (muon) 正规化
        enable_memory_output_proj: bool = False,  # 是否启用 memory 输出投影
        enable_attention_output_proj: bool = False,  # 是否启用 attention 输出投影
        enable_attention_memory_output_proj: bool = False,  # 是否启用 attention+memory 输出投影
        residual_style: str = "parallel",  # "parallel" or "sequential"
        memory_adaLN_modulation_enabled: bool = False,
        uttt_dense_r: int = 1,
        uttt_agg_reduce: str = "mean",
        uttt_agg_lr_scale: float = 1.0,
        uttt_master_renorm: bool = False,
        uttt_update_mode: str = "aggregate",
        uttt_update_checkpoint: bool = True,
        **kwargs,
    ):
        if "memory_aggregate_lr_scale" in kwargs:
            raise ValueError(
                "Use `uttt_agg_lr_scale`; the MoE-only "
                "`memory_aggregate_lr_scale` alias is not accepted.",
            )
        dense_r = _normalize_dense_r(uttt_dense_r)
        if uttt_agg_reduce not in ("mean", "sum"):
            raise ValueError(
                "`uttt_agg_reduce` must be 'mean' or 'sum', "
                f"got {uttt_agg_reduce!r}.",
            )
        if isinstance(uttt_agg_lr_scale, bool) or not isinstance(
            uttt_agg_lr_scale,
            Real,
        ):
            raise ValueError(
                "`uttt_agg_lr_scale` must be a finite positive number, "
                f"got {uttt_agg_lr_scale!r}.",
            )
        agg_lr_scale = float(uttt_agg_lr_scale)
        if not math.isfinite(agg_lr_scale) or agg_lr_scale <= 0.0:
            raise ValueError(
                "`uttt_agg_lr_scale` must be a finite positive number, "
                f"got {uttt_agg_lr_scale!r}.",
            )
        if not isinstance(uttt_master_renorm, bool):
            raise ValueError(
                "`uttt_master_renorm` must be a bool, "
                f"got {uttt_master_renorm!r}.",
            )
        if uttt_update_mode != "aggregate":
            raise ValueError(
                "`uttt_update_mode` must be 'aggregate' for uTTT-Dense, "
                f"got {uttt_update_mode!r}.",
            )
        if use_memory_block is not True:
            raise ValueError("uTTT-Dense requires `use_memory_block=True`.")

        if fw_inter_multi is not None:
            try:
                serialized_r = _normalize_dense_r(fw_inter_multi)
            except ValueError as exc:
                raise ValueError(
                    "`fw_inter_multi`, when present, must exactly match "
                    "`uttt_dense_r`."
                ) from exc
            if serialized_r != dense_r:
                raise ValueError(
                    "`fw_inter_multi` must exactly match `uttt_dense_r`: "
                    f"got fw_inter_multi={fw_inter_multi!r} and "
                    f"uttt_dense_r={dense_r!r}.",
                )
        fw_inter_multi = dense_r

        if isinstance(w0_w2_low_rank, bool) or not isinstance(w0_w2_low_rank, Integral):
            raise ValueError(
                "`w0_w2_low_rank` must be an integer (use <= 0 for full-rank), "
                f"got {w0_w2_low_rank!r}."
            )
        effective_fw_num_heads = (
            fw_num_heads
            if fw_num_heads is not None
            else (num_kv_heads if num_kv_heads is not None else num_heads)
        )
        if effective_fw_num_heads <= 0:
            raise ValueError(
                "`fw_num_heads` must be positive for uTTT-Dense low-rank "
                "validation."
            )
        if hidden_size % effective_fw_num_heads != 0:
            raise ValueError(
                "`hidden_size` must be divisible by `fw_num_heads` for "
                f"uTTT-Dense, got hidden_size={hidden_size}, "
                f"fw_num_heads={effective_fw_num_heads}."
            )
        head_dim = hidden_size // effective_fw_num_heads
        if w0_w2_low_rank > head_dim / 2:
            raise ValueError(
                "`w0_w2_low_rank` must be <= D/2 for uTTT-Dense, where "
                f"D=hidden_size/fw_num_heads={hidden_size}/{effective_fw_num_heads}="
                f"{head_dim:g}; got R={w0_w2_low_rank}. "
                "Use `w0_w2_low_rank <= 0` for the full-parameter path."
            )

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.qkv_bias = qkv_bias
        self.qk_norm = qk_norm
        self.window_size = window_size
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act

        self.initializer_range = initializer_range
        self.elementwise_affine = elementwise_affine
        self.norm_eps = norm_eps
        self.use_cache = use_cache

        self.fuse_norm = fuse_norm
        self.fuse_swiglu = fuse_swiglu
        self.fuse_cross_entropy = fuse_cross_entropy
        self.fuse_linear_cross_entropy = fuse_linear_cross_entropy
        self.use_l2warp = use_l2warp
        self.vocab_size = vocab_size
        self.chunk_size = chunk_size
        self.memory_v_gap = memory_v_gap
        self.memory_k_gap = memory_k_gap
        self.attention_v_gap = attention_v_gap
        self.attention_k_gap = attention_k_gap
        self.shared_kv_cache = shared_kv_cache

        # Fast Weight Memory Block
        self.use_memory_block = use_memory_block
        self.fw_num_heads = fw_num_heads
        self.fw_inter_multi = fw_inter_multi
        self.base_lr = base_lr
        self.per_head_lr = per_head_lr
        self.repeat_function = repeat_function
        self.w0_w2_low_rank = w0_w2_low_rank
        self.fw_init_gain = fw_init_gain
        self.memory_kv_mode = memory_kv_mode
        self.memory_kv_feature = memory_kv_feature
        self.memory_kv_proj = memory_kv_proj
        self.memory_qkv_silu = memory_qkv_silu
        self.memory_qk_norm = memory_qk_norm
        self.memory_qk_norm_type = memory_qk_norm_type
        self.memory_norm_type = memory_norm_type
        self.memory_v_norm = memory_v_norm
        self.memory_qk_rescale = memory_qk_rescale
        self.learnable_ttt_scale = learnable_ttt_scale
        self.ttt_prenorm = ttt_prenorm
        self.same_rope = same_rope
        self.last_layer_fuse_norm = last_layer_fuse_norm
        self.ttt_rope_theta = ttt_rope_theta
        self.use_momentum = use_momentum
        self.use_moun = use_moun
        self.enable_memory_output_proj = enable_memory_output_proj
        self.enable_attention_output_proj = enable_attention_output_proj
        self.enable_attention_memory_output_proj = enable_attention_memory_output_proj
        self.residual_style = residual_style
        self.memory_adaLN_modulation_enabled = memory_adaLN_modulation_enabled
        self.uttt_dense_r = dense_r
        self.uttt_agg_reduce = uttt_agg_reduce
        self.uttt_agg_lr_scale = agg_lr_scale
        self.uttt_master_renorm = uttt_master_renorm
        self.uttt_update_mode = uttt_update_mode
        # Recompute the per-layer update chains in backward (ref_v4's always-on
        # behavior). Turn off only when memory headroom makes the extra update
        # forward not worth it; the trainer-level AC mode cannot see this knob.
        self.uttt_update_checkpoint = bool(uttt_update_checkpoint)

        removed_impl_keys = [
            key for key in kwargs if key.endswith("_reference_impl")
        ]
        if removed_impl_keys:
            raise ValueError(
                f"Removed config key(s) for uttt_dense: "
                f"{', '.join(removed_impl_keys)}"
            )

        # Optimizer-state experiments live in recurrent_lact_optimizer_tokens.
        for legacy_key in (
            "use_memory_adam",
            "memory_adam_beta1",
            "memory_adam_beta2",
            "memory_adam_eps",
            "optimizer_state_tokens",
            "optimizer_state_token_dim",
            "optimizer_state_max_history",
            "optimizer_state_token_detach",
        ):
            kwargs.pop(legacy_key, None)

        if fuse_cross_entropy and fuse_linear_cross_entropy:
            raise ValueError(
                "`fuse_cross_entropy` and `fuse_linear_cross_entropy` cannot be True at the same time.",
            )
        if fuse_linear_cross_entropy:
            warnings.warn(
                "`fuse_linear_cross_entropy` is enabled, which can improves memory efficiency "
                "at the potential cost of reduced precision. "
                "If you observe issues like loss divergence, consider disabling this setting.",
            )

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

