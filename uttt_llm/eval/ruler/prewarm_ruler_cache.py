#!/usr/bin/env python3
"""Materialize RULER/lm-eval task datasets into the configured HF cache."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import flame.custom_models as custom_models  # noqa: F401,E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument("--tasks", default="ruler")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--metadata-max-lengths",
        default="4096,8192,16384,32768",
        help="Comma-separated RULER max_seq_lengths metadata override.",
    )
    return parser.parse_args()


def parse_metadata_max_lengths(value: str, fallback: int) -> list[int]:
    lengths = [int(item.strip()) for item in value.split(",") if item.strip()]
    return lengths or [fallback]


def main() -> int:
    args = parse_args()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    model_dir = Path(args.model_dir).resolve()
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

    from lm_eval.tasks import TaskManager

    task_manager = TaskManager(metadata=model_args | metadata)
    loaded = task_manager.load(task_names)

    manifest: dict[str, object] = {
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "tasks_requested": task_names,
        "ruler_max_length": args.max_length,
        "ruler_max_seq_lengths": metadata["max_seq_lengths"],
        "cache_env": {
            "HF_HOME": os.environ.get("HF_HOME"),
            "HF_DATASETS_CACHE": os.environ.get("HF_DATASETS_CACHE"),
            "TRANSFORMERS_CACHE": os.environ.get("TRANSFORMERS_CACHE"),
            "HUGGINGFACE_HUB_CACHE": os.environ.get("HUGGINGFACE_HUB_CACHE"),
        },
        "tasks": {},
    }

    for task_name, task_obj in loaded["tasks"].items():
        docs = task_obj.eval_docs
        total = len(docs)
        if total:
            _ = docs[0]
            _ = docs[total - 1]
        manifest["tasks"][task_name] = {
            "total_samples": total,
            "status": "cached",
        }

    output_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
