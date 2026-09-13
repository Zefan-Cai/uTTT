#!/usr/bin/env python3
"""Run RULER with lm-evaluation-harness for a local HF checkpoint."""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import os
import random
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM
from transformers.generation import GenerationMixin


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import flame.custom_models as custom_models  # noqa: F401,E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument("--tasks", default="ruler")
    parser.add_argument("--batch-size", default="1")
    parser.add_argument("--max-batch-size", type=int, default=None)
    parser.add_argument("--limit", type=float, default=None)
    parser.add_argument("--custom-module", default="")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--metadata-max-lengths",
        default="4096,8192,16384,32768",
        help="Comma-separated RULER max_seq_lengths metadata override.",
    )
    parser.add_argument("--log-samples", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--use-cache",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Force generation KV cache on or off when the lm-eval HF backend calls generate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed Python, NumPy, and torch before RULER evaluation.",
    )
    parser.add_argument("--sample-shard-rank", type=int, default=0)
    parser.add_argument("--sample-shard-world-size", type=int, default=1)
    return parser.parse_args()


def set_eval_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def patch_generation_mixin(module_name: str) -> None:
    custom_module = importlib.import_module(module_name)
    for attr_name in dir(custom_module):
        if not attr_name.endswith("ForCausalLM"):
            continue
        model_cls = getattr(custom_module, attr_name)
        if not isinstance(model_cls, type) or issubclass(model_cls, GenerationMixin):
            continue

        patched_cls = type(model_cls.__name__, (model_cls, GenerationMixin), {})
        setattr(custom_module, attr_name, patched_cls)
        config_cls = getattr(model_cls, "config_class", None)
        if isinstance(config_cls, type):
            AutoModelForCausalLM.register(config_cls, patched_cls, exist_ok=True)


def patch_lm_eval_hf_use_cache(use_cache: bool) -> None:
    from lm_eval.models.huggingface import HFLM, stop_sequences_criteria

    original_create_model = HFLM._create_model

    def _create_model_with_cache_flag(self, *args, **kwargs):
        result = original_create_model(self, *args, **kwargs)
        if hasattr(self.model, "config"):
            self.model.config.use_cache = use_cache
        if getattr(self.model, "generation_config", None) is not None:
            self.model.generation_config.use_cache = use_cache
        return result

    def _model_generate_with_cache_flag(
        self,
        context,
        max_length: int,
        stop: list[str],
        **generation_kwargs,
    ) -> torch.Tensor:
        generation_kwargs["temperature"] = generation_kwargs.get("temperature", 0.0)
        do_sample = generation_kwargs.get("do_sample")
        if generation_kwargs.get("temperature") == 0.0 and do_sample is None:
            generation_kwargs["do_sample"] = do_sample = False
        if do_sample is False and generation_kwargs.get("temperature") == 0.0:
            generation_kwargs.pop("temperature", None)

        generation_kwargs.pop("use_cache", None)
        if hasattr(self.model, "config"):
            self.model.config.use_cache = use_cache
        if getattr(self.model, "generation_config", None) is not None:
            self.model.generation_config.use_cache = use_cache

        stopping_criteria = stop_sequences_criteria(
            self.tokenizer, stop, context.shape[1], context.shape[0]
        )
        with torch.autocast(
            device_type=self.device.type,
            dtype=self.mixed_precision_dtype,
            enabled=self.mixed_precision_dtype is not None,
        ):
            return self.model.generate(
                input_ids=context,
                max_length=max_length,
                stopping_criteria=stopping_criteria,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=use_cache,
                **generation_kwargs,
            )

    HFLM._create_model = _create_model_with_cache_flag
    HFLM._model_generate = _model_generate_with_cache_flag


def parse_metadata_max_lengths(value: str, fallback: int) -> list[int]:
    lengths = [int(item.strip()) for item in value.split(",") if item.strip()]
    return lengths or [fallback]


def is_padding_attention_mask_error(error: ValueError) -> bool:
    return "does not support padded attention_mask" in str(error)


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "samples"


def write_samples(samples: object, output_dir: Path) -> list[str]:
    if not samples:
        return []

    sample_dir = output_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    if isinstance(samples, dict):
        sample_items = samples.items()
    else:
        sample_items = [("samples", samples)]

    for task_name, task_samples in sample_items:
        path = sample_dir / f"{safe_filename(str(task_name))}.jsonl"
        records = task_samples if isinstance(task_samples, list) else [task_samples]
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True, default=str))
                handle.write("\n")
        written.append(str(path.relative_to(output_dir)))

    return written


def build_sample_shards(
    task_names: list[str],
    model_args: dict[str, object],
    metadata: dict[str, object],
    shard_rank: int,
    shard_world_size: int,
) -> tuple[dict[str, list[int]] | None, dict[str, object]]:
    if shard_world_size <= 1:
        return None, {
            "enabled": False,
            "rank": shard_rank,
            "world_size": shard_world_size,
            "tasks": {},
        }
    if shard_rank < 0 or shard_rank >= shard_world_size:
        raise ValueError(
            f"sample shard rank must be in [0, {shard_world_size}), got {shard_rank}"
        )

    from lm_eval.tasks import TaskManager

    task_manager = TaskManager(metadata=model_args | metadata)
    loaded = task_manager.load(task_names)
    samples: dict[str, list[int]] = {}
    task_manifest: dict[str, object] = {}

    for task_name, task_obj in loaded["tasks"].items():
        total = len(task_obj.eval_docs)
        indices = list(range(shard_rank, total, shard_world_size))
        if not indices:
            continue
        samples[task_name] = indices
        task_manifest[task_name] = {
            "total_samples": total,
            "shard_samples": len(indices),
            "first_index": indices[0],
            "last_index": indices[-1],
        }

    if not samples:
        raise ValueError(
            f"sample shard rank {shard_rank}/{shard_world_size} received no samples"
        )

    return samples, {
        "enabled": True,
        "rank": shard_rank,
        "world_size": shard_world_size,
        "tasks": task_manifest,
    }


def main() -> int:
    args = parse_args()
    if args.seed is not None:
        set_eval_seed(args.seed)
    model_dir = Path(args.model_dir).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    if args.custom_module:
        patch_generation_mixin(args.custom_module)
    if args.use_cache is not None:
        patch_lm_eval_hf_use_cache(args.use_cache)

    model_args = {
        "pretrained": str(model_dir),
        "tokenizer": str(model_dir),
        "dtype": args.dtype,
        "max_length": args.max_length,
        "trust_remote_code": args.trust_remote_code,
    }
    metadata = {
        "max_seq_lengths": parse_metadata_max_lengths(
            args.metadata_max_lengths,
            args.max_length,
        )
    }
    task_names = [task.strip() for task in args.tasks.split(",") if task.strip()]
    sample_shards, sample_shard_manifest = build_sample_shards(
        task_names=task_names,
        model_args=model_args,
        metadata=metadata,
        shard_rank=args.sample_shard_rank,
        shard_world_size=args.sample_shard_world_size,
    )

    from lm_eval import evaluator

    eval_kwargs = {}
    if args.seed is not None:
        try:
            import inspect

            params = inspect.signature(evaluator.simple_evaluate).parameters
            for key in ("random_seed", "numpy_random_seed", "torch_random_seed", "fewshot_random_seed"):
                if key in params:
                    eval_kwargs[key] = args.seed
        except Exception:
            eval_kwargs = {}

    try:
        results = evaluator.simple_evaluate(
            model="hf",
            model_args=model_args,
            tasks=task_names,
            batch_size=args.batch_size,
            max_batch_size=args.max_batch_size,
            device=args.device,
            limit=args.limit,
            samples=sample_shards,
            bootstrap_iters=0,
            log_samples=args.log_samples,
            metadata=metadata,
            **eval_kwargs,
        )
    except ValueError as exc:
        if not is_padding_attention_mask_error(exc):
            raise
        if args.batch_size == "1" and args.max_batch_size is None:
            raise
        print(
            "Model rejected a padded attention_mask; retrying this RULER shard "
            "with batch_size=1.",
            file=sys.stderr,
        )
        exc.__traceback__ = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        results = evaluator.simple_evaluate(
            model="hf",
            model_args=model_args,
            tasks=task_names,
            batch_size="1",
            max_batch_size=None,
            device=args.device,
            limit=args.limit,
            samples=sample_shards,
            bootstrap_iters=0,
            log_samples=args.log_samples,
            metadata=metadata,
            **eval_kwargs,
        )
    if results is None:
        return 0
    if args.use_cache is not None:
        results.setdefault("config", {}).setdefault("model_args", {})["use_cache"] = args.use_cache
    if args.seed is not None:
        results.setdefault("config", {}).setdefault("model_args", {})["seed"] = args.seed

    sample_files = write_samples(results.pop("samples", None), output_path.parent)
    if sample_files:
        results["sample_files"] = sample_files
    results["sample_shard"] = sample_shard_manifest

    shard_manifest_path = output_path.parent / "sample_shard_manifest.json"
    shard_manifest_path.write_text(
        json.dumps(sample_shard_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")

    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
