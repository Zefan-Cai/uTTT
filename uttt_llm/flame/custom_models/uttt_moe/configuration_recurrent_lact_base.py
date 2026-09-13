import warnings

from transformers.configuration_utils import PretrainedConfig


class RecurrentLactRefConfig(PretrainedConfig):

    model_type = "uttt_moe_base"
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
        memory_shared_kv_cache: bool = False,
        memory_reuse_kv_cache: bool = False,
        memory_perlayer_k_share_v: bool = False,
        memory_perlayer_k_reuse_v: bool = False,
        memory_k_compose_mode: str | None = None,
        memory_v_compose_mode: str | None = None,
        # Fast Weight Memory Block Configuration
        use_memory_block: bool = False,
        fw_num_heads: int | None = None,
        fw_inter_multi: float = 2.0,
        base_lr: float = 0.01,
        per_head_lr: bool = False,  # True: 3*fw_num_heads independent LRs; False: 3 shared LRs
        repeat_function: str = "repeat",  # "repeat" or "linear"
        w0_w2_low_rank: int = -1,
        fw_init_gain: float = 0.5,
        # Memory KV settings
        memory_kv_mode: str = "to_kv",  # "reuse_kv", "to_kv"
        memory_kv_feature: str = "x_memory_in",  # "x_attn_in", "x_memory_in", "xi_out"
        memory_kv_proj: str = "ttt_kv_proj",  # "reuse_attn_kv_proj", "ttt_kv_proj"
        memory_qkv_silu: bool = False,  # whether to apply SiLU to memory Q/K/V
        memory_qk_norm: bool = False,  # whether to apply L2 norm to memory Q/K
        memory_qk_norm_type: str = "l2",  # "l2" or "rms"
        memory_norm_type: str = "rms",  # "l2" or "rms", controls all parameter-free norms in memory
        memory_v_norm: bool = False,  # whether to normalise memory V (kind set by memory_norm_type)
        memory_qk_rescale: bool = False,  # apply a learnable scale+offset to memory q/k
        learnable_ttt_scale: bool = False,  # apply a learnable per-dim scale to the memory output
        ttt_prenorm: bool = False,  # True: keep raw fast-weight state and apply normalized view each step
        same_rope: bool = False,  # TTT RoPE uses the same theta and head_dim as attention
        last_layer_fuse_norm: bool = True,  # whether the final norm uses the fused variant
        ttt_rope_theta: float = 0.0,  # TTT RoPE theta (0 = disabled, > 0 enables RoPE on TTT Q/K)
        use_momentum: bool = False,  # apply momentum to fast-weight gradients
        use_moun: bool = False,  # apply Newton-Schulz (Muon) orthogonalisation to fast-weight gradients
        enable_memory_output_proj: bool = False,  # whether to enable the memory output projection
        enable_attention_output_proj: bool = False,  # whether to enable the attention output projection
        enable_attention_memory_output_proj: bool = False,  # whether to enable the joint attention+memory output projection
        residual_style: str = "parallel",  # "parallel" or "sequential"
        memory_adaLN_modulation_enabled: bool = False,
        **kwargs,
    ):
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
        self.memory_shared_kv_cache = memory_shared_kv_cache
        self.memory_reuse_kv_cache = memory_reuse_kv_cache
        self.memory_perlayer_k_share_v = memory_perlayer_k_share_v
        self.memory_perlayer_k_reuse_v = memory_perlayer_k_reuse_v
        self.memory_k_compose_mode = memory_k_compose_mode
        self.memory_v_compose_mode = memory_v_compose_mode

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

        removed_impl_keys = [
            key for key in kwargs if key.endswith("_reference_impl")
        ]
        if removed_impl_keys:
            raise ValueError(
                f"Removed config key(s) for ttt_dense_base: {', '.join(removed_impl_keys)}"
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
        if memory_shared_kv_cache and memory_reuse_kv_cache:
            raise ValueError(
                "`memory_shared_kv_cache` and `memory_reuse_kv_cache` cannot both be True.",
            )
        if memory_perlayer_k_share_v and memory_perlayer_k_reuse_v:
            raise ValueError(
                "`memory_perlayer_k_share_v` and `memory_perlayer_k_reuse_v` cannot both be True.",
            )
        if (memory_perlayer_k_share_v or memory_perlayer_k_reuse_v) and memory_reuse_kv_cache:
            raise ValueError(
                "`memory_perlayer_k_*` modes are incompatible with `memory_reuse_kv_cache`.",
            )
        if (memory_perlayer_k_share_v or memory_perlayer_k_reuse_v) and memory_shared_kv_cache:
            raise ValueError(
                "`memory_perlayer_k_*` modes are incompatible with `memory_shared_kv_cache`.",
            )
        compose_modes = {"reuse", "attn-proj", "shared", "ttt-proj"}
        if memory_k_compose_mode is not None and memory_k_compose_mode not in compose_modes:
            raise ValueError(
                f"`memory_k_compose_mode` must be one of {sorted(compose_modes)}, got {memory_k_compose_mode!r}.",
            )
        if memory_v_compose_mode is not None and memory_v_compose_mode not in compose_modes:
            raise ValueError(
                f"`memory_v_compose_mode` must be one of {sorted(compose_modes)}, got {memory_v_compose_mode!r}.",
            )
        if (memory_k_compose_mode is not None or memory_v_compose_mode is not None) and (
            memory_shared_kv_cache or memory_reuse_kv_cache or memory_perlayer_k_share_v or memory_perlayer_k_reuse_v
        ):
            raise ValueError(
                "`memory_*_compose_mode` cannot be mixed with legacy explicit memory compose flags.",
            )
        if (
            memory_k_compose_mode == "ttt-proj" or memory_v_compose_mode == "ttt-proj"
        ) and memory_kv_proj != "ttt_kv_proj":
            raise ValueError(
                "`ttt-proj` memory compose modes require `memory_kv_proj=\"ttt_kv_proj\"`.",
            )
        if memory_kv_mode == "reuse_kv" and memory_k_compose_mode == "ttt-proj":
            raise ValueError(
                "`memory_k_compose_mode=\"ttt-proj\"` is incompatible with `memory_kv_mode=\"reuse_kv\"`.",
            )
        if memory_kv_mode == "reuse_kv" and memory_v_compose_mode == "ttt-proj":
            if memory_k_compose_mode not in {None, "reuse"}:
                raise ValueError(
                    "`memory_v_compose_mode=\"ttt-proj\"` with `memory_kv_mode=\"reuse_kv\"` "
                    "requires `memory_k_compose_mode` to be omitted or set to \"reuse\".",
                )
            if memory_k_gap != 0:
                raise ValueError(
                    "`memory_v_compose_mode=\"ttt-proj\"` with `memory_kv_mode=\"reuse_kv\"` "
                    "requires `memory_k_gap` to be 0.",
                )
            if memory_v_gap is None:
                raise ValueError(
                    "`memory_v_compose_mode=\"ttt-proj\"` with `memory_kv_mode=\"reuse_kv\"` "
                    "requires `memory_v_gap` to be set.",
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
