#!/usr/bin/env python3
"""Validate and combine completed NVS evaluation outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from uttt_nvs.eval.common import REPO_ROOT, load_registry
from uttt_nvs.eval.statistics import RAW_FIELDS


DEFAULT_REGISTRY = REPO_ROOT / "eval" / "experiments.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--dataset", action="append")
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    registry = load_registry(args.registry)
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root
        else Path(registry["defaults"]["output_root"]).expanduser().resolve()
    )
    aggregate_dir = output_root / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    selected_datasets = set(args.dataset or [])

    import pandas as pd

    raw_frames = []
    per_view_frames = []
    overall_rows = []
    missing = []
    for experiment in registry["experiments"]:
        dataset_id = str(experiment["dataset"])
        if selected_datasets and dataset_id not in selected_datasets:
            continue
        run_dir = output_root / dataset_id / str(experiment["id"])
        required = {
            "raw": run_dir / "raw_metrics.csv",
            "per_view": run_dir / "per_view_summary.csv",
            "overall": run_dir / "overall_summary.json",
        }
        absent = [str(path) for path in required.values() if not path.is_file()]
        if absent:
            missing.append({"experiment": experiment["id"], "files": absent})
            continue

        raw = pd.read_csv(required["raw"])
        expected_rows = int(registry["datasets"][dataset_id]["expected_rows"])
        expected_scenes = int(
            registry["datasets"][dataset_id]["expected_scenes"]
        )
        if len(raw) != expected_rows:
            raise RuntimeError(
                f"{experiment['id']}: expected {expected_rows} raw rows, "
                f"got {len(raw)}"
            )
        if raw["scene_id"].nunique() != expected_scenes:
            raise RuntimeError(
                f"{experiment['id']}: expected {expected_scenes} scenes, "
                f"got {raw['scene_id'].nunique()}"
            )
        expected_views = set(range(1, 24))
        if set(raw["view_index"].unique()) != expected_views:
            raise RuntimeError(
                f"{experiment['id']}: raw records do not cover View 1..23"
            )
        if list(raw.columns) != list(RAW_FIELDS):
            raise RuntimeError(
                f"{experiment['id']}: raw schema is {list(raw.columns)}, "
                f"expected {list(RAW_FIELDS)}"
            )
        raw.insert(0, "experiment_id", experiment["id"])
        raw_frames.append(raw)

        per_view = pd.read_csv(required["per_view"])
        per_view.insert(0, "experiment_id", experiment["id"])
        per_view_frames.append(per_view)

        overall = json.loads(required["overall"].read_text())
        overall_rows.append(
            {
                "experiment_id": experiment["id"],
                "dataset": dataset_id,
                "model": experiment["display_name"],
                "paper_model": experiment["paper_model"],
                "checkpoint_impl": experiment["checkpoint_impl"],
                **overall,
            }
        )

    if missing and not args.allow_incomplete:
        preview = json.dumps(missing[:5], indent=2)
        raise FileNotFoundError(
            f"{len(missing)} registered evaluations are incomplete:\n{preview}"
        )
    if not raw_frames:
        raise RuntimeError("No complete evaluation outputs were found")

    all_raw = pd.concat(raw_frames, ignore_index=True)
    all_per_view = pd.concat(per_view_frames, ignore_index=True)
    all_overall = pd.DataFrame.from_records(overall_rows)
    all_raw.to_parquet(aggregate_dir / "all_raw_metrics.parquet", index=False)
    all_per_view.to_csv(
        aggregate_dir / "all_per_view_metrics.csv",
        index=False,
    )
    all_overall.to_csv(
        aggregate_dir / "all_overall_metrics.csv",
        index=False,
    )
    (aggregate_dir / "missing_runs.json").write_text(
        json.dumps(missing, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Aggregated {len(overall_rows)} runs, {len(all_raw)} raw rows, "
        f"and {len(all_per_view)} per-view summary rows into {aggregate_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
