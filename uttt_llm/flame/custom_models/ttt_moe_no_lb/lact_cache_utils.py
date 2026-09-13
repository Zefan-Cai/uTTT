from __future__ import annotations

from typing import Any

import torch


RECURRENT_LACT_CACHE_TYPE = "recurrent_lact_generation_cache"


def is_recurrent_lact_cache(past_key_values: Any) -> bool:
    return (
        isinstance(past_key_values, dict)
        and past_key_values.get("cache_type") == RECURRENT_LACT_CACHE_TYPE
    )


def recurrent_lact_cache_seen_tokens(past_key_values: Any) -> int:
    if past_key_values is None:
        return 0
    if is_recurrent_lact_cache(past_key_values):
        return int(past_key_values.get("seen_tokens", 0))
    if hasattr(past_key_values, "get_seq_length"):
        return int(past_key_values.get_seq_length())
    if isinstance(past_key_values, (list, tuple)) and past_key_values:
        layer_cache = past_key_values[0]
        if isinstance(layer_cache, (list, tuple)) and layer_cache:
            return int(layer_cache[0].shape[-2])
    return 0


def make_recurrent_lact_cache(
    *,
    layers: list[dict[str, Any]],
    seen_tokens: int,
    pending_start: int,
    batch_size: int,
    **extra: Any,
) -> dict[str, Any]:
    cache = {
        "cache_type": RECURRENT_LACT_CACHE_TYPE,
        "layers": layers,
        "seen_tokens": int(seen_tokens),
        "pending_start": int(pending_start),
        "batch_size": int(batch_size),
    }
    cache.update(extra)
    return cache


def prepare_recurrent_lact_generation_inputs(
    *,
    input_ids: torch.LongTensor,
    past_key_values: Any | None = None,
    attention_mask: torch.Tensor | None = None,
    inputs_embeds: torch.FloatTensor | None = None,
    use_cache: bool | None = None,
    logits_to_keep: int | None = None,
    kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    kwargs = kwargs or {}
    use_cache = True if use_cache is None else use_cache
    past_length = recurrent_lact_cache_seen_tokens(past_key_values) if use_cache else 0

    if inputs_embeds is not None and past_length == 0:
        model_inputs = {"inputs_embeds": inputs_embeds}
    else:
        if past_length > 0:
            if input_ids.shape[1] > past_length:
                input_ids = input_ids[:, past_length:]
            else:
                input_ids = input_ids[:, -1:]
        model_inputs = {"input_ids": input_ids.contiguous()}

    model_inputs["past_key_values"] = past_key_values
    model_inputs["use_cache"] = use_cache

    if attention_mask is not None:
        model_inputs["attention_mask"] = attention_mask
    if logits_to_keep is not None:
        model_inputs["logits_to_keep"] = logits_to_keep

    for key, value in kwargs.items():
        if key not in {"cache_position", "position_ids"}:
            model_inputs[key] = value

    return model_inputs


def reorder_recurrent_lact_cache(past_key_values: Any, beam_idx: torch.LongTensor) -> Any:
    if not is_recurrent_lact_cache(past_key_values):
        if hasattr(past_key_values, "reorder_cache"):
            past_key_values.reorder_cache(beam_idx)
        return past_key_values

    batch_size = int(past_key_values.get("batch_size", beam_idx.numel()))

    def reorder(value: Any) -> Any:
        if torch.is_tensor(value):
            if value.dim() > 0 and value.shape[0] == batch_size:
                return value.index_select(0, beam_idx.to(value.device))
            if value.dim() > 0 and batch_size > 0 and value.shape[0] % batch_size == 0:
                group = value.shape[0] // batch_size
                if group > 1:
                    reshaped = value.reshape(batch_size, group, *value.shape[1:])
                    return reshaped.index_select(0, beam_idx.to(value.device)).reshape(
                        beam_idx.numel() * group,
                        *value.shape[1:],
                    )
            return value
        if isinstance(value, tuple):
            return tuple(reorder(item) for item in value)
        if isinstance(value, list):
            return [reorder(item) for item in value]
        if isinstance(value, dict):
            return {key: reorder(item) for key, item in value.items()}
        if hasattr(value, "reorder_cache"):
            value.reorder_cache(beam_idx)
            return value
        return value

    reordered = reorder(past_key_values)
    reordered["batch_size"] = int(beam_idx.numel())
    return reordered
