import torch
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS


def _default_rope_parameters(config, device):
    rope_parameters = getattr(config, "rope_parameters", None)
    base = getattr(config, "rope_theta", 10000.0)
    if isinstance(rope_parameters, dict):
        base = rope_parameters.get("rope_theta", base)

    dim = getattr(config, "head_dim", None)
    if dim is None:
        dim = config.hidden_size // config.num_attention_heads

    inv_freq = 1.0 / (
        base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
    )
    return inv_freq, 1.0


def _get_rope_type(rotary_embedding):
    rope_type = getattr(rotary_embedding, "rope_type", None)
    if rope_type is not None:
        return rope_type

    config = rotary_embedding.config
    rope_parameters = getattr(config, "rope_parameters", None)
    if isinstance(rope_parameters, dict):
        return rope_parameters.get("rope_type", "default")

    rope_scaling = getattr(config, "rope_scaling", None)
    if isinstance(rope_scaling, dict):
        return rope_scaling.get("rope_type") or rope_scaling.get("type") or "default"

    return "default"


def reset_llama_rotary_embedding(rotary_embedding):
    device = rotary_embedding.inv_freq.device
    if device.type == "meta":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rope_type = _get_rope_type(rotary_embedding)
    if rope_type == "default":
        rope_init_fn = getattr(rotary_embedding, "compute_default_rope_parameters", None)
        rope_init_fn = rope_init_fn or ROPE_INIT_FUNCTIONS.get("default") or _default_rope_parameters
    else:
        rope_init_fn = ROPE_INIT_FUNCTIONS[rope_type]

    try:
        inv_freq, rotary_embedding.attention_scaling = rope_init_fn(rotary_embedding.config, device=device)
    except TypeError:
        inv_freq, rotary_embedding.attention_scaling = rope_init_fn(rotary_embedding.config, device)

    rotary_embedding.register_buffer("inv_freq", inv_freq, persistent=False)
    if hasattr(rotary_embedding, "original_inv_freq") and "original_inv_freq" not in rotary_embedding._buffers:
        delattr(rotary_embedding, "original_inv_freq")
    rotary_embedding.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)
