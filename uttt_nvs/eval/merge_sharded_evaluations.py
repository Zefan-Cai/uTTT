#!/usr/bin/env python3
"""Validate and merge deterministic NVS evaluation shards into one full test."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import shutil
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from uttt_nvs.eval.common import resolve_experiment
from uttt_nvs.eval.evaluate import _log_wandb_outputs, _validate_complete_records, _wandb_run
from uttt_nvs.eval.statistics import RAW_FIELDS, write_evaluation_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--shards-root", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--wandb-mode", choices=("online", "offline", "disabled"), default="online"
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _read_records(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != tuple(RAW_FIELDS):
            raise ValueError(f"{path}: unexpected raw-metric fields {reader.fieldnames}")
        return [
            {
                "dataset": row["dataset"],
                "model": row["model"],
                "scene_id": row["scene_id"],
                "view_index": int(row["view_index"]),
                "psnr": float(row["psnr"]),
                "ssim": float(row["ssim"]),
                "lpips": float(row["lpips"]),
            }
            for row in reader
        ]


def main() -> int:
    args = parse_args()
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    experiment = resolve_experiment(
        args.registry, args.experiment, output_root=args.output_root
    )
    output_dir = experiment.output_dir
    completion_path = output_dir / "completion.json"
    if completion_path.exists() and not args.overwrite:
        raise FileExistsError(f"{completion_path} already marks a complete merge")
    if args.overwrite:
        shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    shard_metadata: list[dict[str, Any]] = []
    for shard_index in range(args.num_shards):
        shard_dir = (
            args.shards_root
            / f"shard_{shard_index}"
            / experiment.dataset_id
            / experiment.id
        )
        marker = shard_dir / "completion.json"
        raw_path = shard_dir / "raw_metrics.csv"
        metadata_path = shard_dir / "run_metadata.json"
        if not marker.is_file() or not raw_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(f"incomplete shard {shard_index}: {shard_dir}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("shard_index") != shard_index:
            raise ValueError(f"{shard_dir}: wrong shard_index metadata")
        if metadata.get("num_shards") != args.num_shards:
            raise ValueError(f"{shard_dir}: wrong num_shards metadata")
        shard_metadata.append(metadata)
        records.extend(_read_records(raw_path))

    reference = shard_metadata[0]
    invariant_keys = (
        "experiment_id",
        "dataset",
        "checkpoint_sha256",
        "config_sha256",
        "view_manifest_sha256",
        "evaluation_code_sha256",
        "v_gap",
    )
    for metadata in shard_metadata[1:]:
        mismatches = [
            key for key in invariant_keys if metadata.get(key) != reference.get(key)
        ]
        if mismatches:
            raise ValueError(f"shards disagree on metadata: {mismatches}")

    keys = [(row["scene_id"], row["view_index"]) for row in records]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate scene/view records across shards")
    _validate_complete_records(
        records,
        expected_scenes=int(experiment.dataset["expected_scenes"]),
        expected_rows=int(experiment.dataset["expected_rows"]),
    )
    records.sort(key=lambda row: (row["scene_id"], row["view_index"]))
    started_at = time.time()
    overall, per_view = write_evaluation_outputs(
        records,
        output_dir,
        bootstrap_samples=10_000,
        bootstrap_seed=int(experiment.defaults.get("seed", 9595)),
    )
    metadata = dict(reference)
    metadata.update(
        {
            "max_scenes": None,
            "shard_index": None,
            "num_shards": args.num_shards,
            "merge_protocol": "deterministic strided scene shards",
            "merged_at_unix": time.time(),
        }
    )
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    overall["runtime_seconds"] = time.time() - started_at
    overall["expected_scenes"] = int(experiment.dataset["expected_scenes"])
    overall["expected_rows"] = int(experiment.dataset["expected_rows"])
    (output_dir / "overall_summary.json").write_text(
        json.dumps(overall, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    run = _wandb_run(
        SimpleNamespace(max_scenes=None, wandb_mode=args.wandb_mode),
        experiment,
        metadata,
    )
    try:
        _log_wandb_outputs(run, overall, per_view, output_dir, experiment.view_manifest_path)
        run.summary["state"] = "complete"
    finally:
        run.finish()
    completion_path.write_text(
        json.dumps(
            {
                "state": "complete",
                "experiment_id": experiment.id,
                "dataset": experiment.dataset_id,
                "expected_scenes": int(experiment.dataset["expected_scenes"]),
                "expected_rows": int(experiment.dataset["expected_rows"]),
                "checkpoint_sha256": metadata["checkpoint_sha256"],
                "config_sha256": metadata["config_sha256"],
                "view_manifest_sha256": metadata["view_manifest_sha256"],
                "evaluation_code_sha256": metadata["evaluation_code_sha256"],
                "wandb_run_id": run.id,
                "wandb_mode": args.wandb_mode,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"experiment": experiment.id, "overall": overall}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
