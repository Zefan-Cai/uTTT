#!/usr/bin/env python3
"""Evaluate one registered NVS checkpoint on its complete local test set."""

from __future__ import annotations

import argparse
import contextlib
import csv
import dataclasses
import json
import os
import time
from pathlib import Path
from typing import Any

from uttt_nvs.eval.common import (
    REPO_ROOT,
    canonical_json_sha256,
    evaluation_code_sha256,
    git_commit,
    import_model,
    load_checkpoint_strict,
    load_model_config,
    model_batch,
    resolve_experiment,
    sha256_file,
    validate_resolved_experiment,
    validate_v_gap,
)
from uttt_nvs.eval.fixed_view_dataset import FixedViewNVSDataset
from uttt_nvs.eval.metrics import ImageMetricComputer
from uttt_nvs.eval.runtime_fingerprint import (
    deterministic_runtime_fingerprint,
    runtime_fingerprint_sha256,
)
from uttt_nvs.eval.statistics import RAW_FIELDS, write_evaluation_outputs


DEFAULT_REGISTRY = REPO_ROOT / "eval" / "experiments.yaml"


def evaluation_run_signature(
    *, experiment_id: str, assets: dict[str, Any],
    runtime: dict[str, Any], amp_dtype: str,
) -> str:
    """Return the standard evaluator identity, including the ML runtime."""
    return canonical_json_sha256({
        "protocol_version": "nvs-eval-v2",
        "experiment_id": experiment_id,
        "assets": dict(assets),
        "runtime": dict(runtime),
        "amp_dtype": amp_dtype,
        "conditioning_protocol": "teacher_forced_sequential",
    })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--lpips-batch-size", type=int, default=32)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _scene_ids(batch: dict[str, Any]) -> list[str]:
    values = batch["scene_id"]
    if isinstance(values, str):
        return [values]
    return [str(value) for value in values]


def _autocast_context(device: Any, amp_dtype: str) -> Any:
    import torch

    if device.type != "cuda" or amp_dtype == "fp32":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _validate_complete_records(
    records: list[dict[str, Any]],
    *,
    expected_scenes: int,
    expected_rows: int,
) -> None:
    from collections import defaultdict

    scene_views: dict[str, set[int]] = defaultdict(set)
    for record in records:
        scene_views[str(record["scene_id"])].add(int(record["view_index"]))
    expected_views = set(range(1, 24))
    incomplete = {
        scene_id: sorted(expected_views.difference(views))
        for scene_id, views in scene_views.items()
        if views != expected_views
    }
    if incomplete:
        preview = dict(list(incomplete.items())[:5])
        raise RuntimeError(f"incomplete per-scene view coverage: {preview}")
    if len(scene_views) != expected_scenes:
        raise RuntimeError(
            f"expected {expected_scenes} scenes, got {len(scene_views)}"
        )
    if len(records) != expected_rows:
        raise RuntimeError(f"expected {expected_rows} rows, got {len(records)}")


def _wandb_run(
    args: argparse.Namespace,
    experiment: Any,
    metadata: dict[str, Any],
) -> Any:
    import wandb

    wandb_config = experiment.defaults.get("wandb", {})
    stable_id = canonical_json_sha256(
        {
            "protocol": "nvs-eval-v2",
            "experiment_id": experiment.id,
            "checkpoint_sha256": metadata["checkpoint_sha256"],
            "config_sha256": metadata["config_sha256"],
            "view_manifest_sha256": metadata["view_manifest_sha256"],
            "evaluation_code_sha256": metadata["evaluation_code_sha256"],
            "max_scenes": args.max_scenes,
            "shard_index": metadata["shard_index"],
            "num_shards": metadata["num_shards"],
        }
    )[:12]
    run_name = str(experiment.raw.get("display_name", experiment.id))
    if args.max_scenes is not None:
        run_name = f"{run_name}-smoke-{args.max_scenes}"
    num_shards = int(getattr(args, "num_shards", 1))
    shard_index = int(getattr(args, "shard_index", 0))
    if num_shards > 1:
        run_name = (
            f"{run_name}-shard-{shard_index + 1:02d}-of-{num_shards:02d}"
        )
    return wandb.init(
        entity=wandb_config.get("entity"),
        project=wandb_config.get("project", "nvs-eval"),
        name=run_name,
        group=experiment.dataset_id,
        job_type="full-test-metrics",
        id=stable_id,
        resume="allow",
        mode=args.wandb_mode,
        dir=str(experiment.output_dir / "wandb"),
        config=metadata,
        tags=[
            experiment.dataset_id,
            str(experiment.raw.get("group", "")),
            f"v_gap-{'null' if metadata['v_gap'] is None else metadata['v_gap']}",
            str(experiment.raw.get("evaluation_class", "formal")),
        ] + (["legacy"] if experiment.raw.get("legacy", False) else []),
    )


def _log_wandb_outputs(
    run: Any,
    overall: dict[str, Any],
    per_view: Any,
    output_dir: Path,
    view_manifest_path: Path,
) -> None:
    import wandb

    for key, value in overall.items():
        run.summary[key] = value

    run.log(
        {
            "per_view/summary": wandb.Table(dataframe=per_view),
        }
    )
    try:
        import plotly.graph_objects as go

        for metric in ("psnr", "ssim", "lpips"):
            frame = per_view[per_view["metric"] == metric].sort_values(
                "view_index"
            )
            figure = go.Figure()
            figure.add_trace(
                go.Scatter(
                    x=frame["view_index"],
                    y=frame["ci95_high"],
                    mode="lines",
                    line={"width": 0},
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            figure.add_trace(
                go.Scatter(
                    x=frame["view_index"],
                    y=frame["ci95_low"],
                    mode="lines",
                    line={"width": 0},
                    fill="tonexty",
                    fillcolor="rgba(31,119,180,0.20)",
                    name="95% CI",
                    hoverinfo="skip",
                )
            )
            figure.add_trace(
                go.Scatter(
                    x=frame["view_index"],
                    y=frame["mean"],
                    mode="lines+markers",
                    name="mean",
                )
            )
            figure.update_layout(
                title=f"{metric.upper()} by target view",
                xaxis_title="View index (View 0 is input)",
                yaxis_title=metric.upper(),
            )
            run.log({f"per_view/{metric}": figure})
    except ImportError:
        pass

    artifact = wandb.Artifact(
        name=f"{run.id}-metrics",
        type="nvs-evaluation",
    )
    for filename in (
        "raw_metrics.csv",
        "raw_metrics.parquet",
        "overall_summary.json",
        "per_view_summary.csv",
        "run_metadata.json",
    ):
        artifact.add_file(str(output_dir / filename), name=filename)
    artifact.add_file(str(view_manifest_path), name="view_manifest.json")
    run.log_artifact(artifact)


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.max_scenes is not None and args.max_scenes <= 0:
        raise ValueError("--max-scenes must be positive")
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= index < --num-shards")

    experiment = resolve_experiment(
        args.registry,
        args.experiment,
        output_root=args.output_root,
    )
    if args.max_scenes is not None and args.output_root is None:
        smoke_root = (
            experiment.output_dir.parents[2]
            / "eval_outputs_smoke"
            / f"n{args.max_scenes}"
        )
        experiment = dataclasses.replace(
            experiment,
            output_dir=smoke_root / experiment.dataset_id / experiment.id,
        )
    validate_resolved_experiment(experiment)
    experiment.output_dir.mkdir(parents=True, exist_ok=True)
    final_raw_path = experiment.output_dir / "raw_metrics.csv"
    partial_raw_path = experiment.output_dir / "raw_metrics.partial.csv"
    completion_path = experiment.output_dir / "completion.json"
    if completion_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{completion_path} marks this evaluation complete; pass "
            "--overwrite to rerun"
        )
    if args.overwrite:
        completion_path.unlink(missing_ok=True)
    elif final_raw_path.exists():
        print(
            f"{final_raw_path} exists without a completion marker; "
            "recomputing the interrupted run"
        )

    import torch
    from torch.utils.data import DataLoader, Subset

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    config = load_model_config(experiment)
    v_gap = validate_v_gap(experiment, config)
    dataset = FixedViewNVSDataset(
        dataset_manifest=experiment.local_manifest_path,
        view_manifest=experiment.view_manifest_path,
        image_size=int(config.model.image_size),
        max_workers=args.image_workers,
    )
    full_scene_count = len(dataset)
    selected_indices = list(
        range(args.shard_index, full_scene_count, args.num_shards)
    )
    if args.max_scenes is not None:
        selected_indices = selected_indices[: args.max_scenes]
    dataset_for_loader: Any
    if args.num_shards > 1 or args.max_scenes is not None:
        dataset_for_loader = Subset(dataset, selected_indices)
    else:
        dataset_for_loader = dataset
    loader = DataLoader(
        dataset_for_loader,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )

    model = import_model(config).to(device)
    checkpoint_metadata = load_checkpoint_strict(
        model,
        experiment.checkpoint_path,
    )
    model.eval()
    metric_computer = ImageMetricComputer(
        device=device,
        lpips_batch_size=args.lpips_batch_size,
    )

    checkpoint_sha256 = sha256_file(experiment.checkpoint_path)
    config_sha256 = sha256_file(experiment.config_path)
    dataset_manifest_sha256 = sha256_file(experiment.local_manifest_path)
    view_manifest_sha256 = sha256_file(experiment.view_manifest_path)
    registry_sha256 = sha256_file(args.registry.resolve())
    code_sha256 = evaluation_code_sha256()
    runtime = deterministic_runtime_fingerprint(device)
    runtime_sha256 = runtime_fingerprint_sha256(runtime)
    assets = {
        "registry_sha256": registry_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "config_sha256": config_sha256,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "view_manifest_sha256": view_manifest_sha256,
        "code_commit": git_commit(),
        "evaluation_code_sha256": code_sha256,
    }
    run_signature = evaluation_run_signature(
        experiment_id=experiment.id,
        assets=assets,
        runtime=runtime,
        amp_dtype=args.amp_dtype,
    )
    metadata = {
        "protocol_version": "nvs-eval-v2",
        "run_signature": run_signature,
        "experiment_id": experiment.id,
        "dataset": experiment.dataset_id,
        "model": experiment.raw.get("display_name", experiment.model_name),
        "paper_model": experiment.model_name,
        "checkpoint_impl": experiment.raw.get("checkpoint_impl"),
        "checkpoint": str(experiment.checkpoint_path),
        "checkpoint_step": experiment.raw.get("checkpoint_step"),
        "checkpoint_sha256": checkpoint_sha256,
        "config": str(experiment.config_path),
        "config_sha256": config_sha256,
        "dataset_manifest": str(experiment.local_manifest_path),
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "view_manifest": str(experiment.view_manifest_path),
        "view_manifest_sha256": view_manifest_sha256,
        "registry": str(args.registry.resolve()),
        "registry_sha256": registry_sha256,
        "code_commit": assets["code_commit"],
        "evaluation_code_sha256": code_sha256,
        "runtime": runtime,
        "runtime_sha256": runtime_sha256,
        "assets": assets,
        "v_gap": v_gap,
        "evaluation_class": experiment.raw.get("evaluation_class", "formal"),
        "source_wandb_run": experiment.raw.get("source_wandb_run"),
        "source_commit": experiment.raw.get("source_commit"),
        "num_views": 24,
        "scored_views": list(range(1, 24)),
        "conditioning_protocol": "teacher_forced_sequential",
        "conditioning_definition": (
            "prediction for View k is conditioned on ground-truth Views 0..k-1"
        ),
        "amp_dtype": args.amp_dtype,
        "max_scenes": args.max_scenes,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        **checkpoint_metadata,
    }
    (experiment.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    run = _wandb_run(args, experiment, metadata)

    records: list[dict[str, Any]] = []
    display_model = str(experiment.raw.get("display_name", experiment.model_name))
    start_time = time.time()
    completion_payload: dict[str, Any] | None = None
    try:
        with partial_raw_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=RAW_FIELDS)
            writer.writeheader()
            with torch.inference_mode():
                for batch_index, batch in enumerate(loader):
                    scene_ids = _scene_ids(batch)
                    inputs = model_batch(batch, device)
                    with _autocast_context(device, args.amp_dtype):
                        result = model(inputs)
                    target = result["target"]["image"].float()
                    prediction = result["rendering"].float()
                    if target.shape[1] != 23 or prediction.shape[1] != 23:
                        raise RuntimeError(
                            "model must produce exactly View 1..23; got target "
                            f"{tuple(target.shape)} and prediction "
                            f"{tuple(prediction.shape)}"
                        )
                    batch_size, target_views = target.shape[:2]
                    flattened_target = target.reshape(
                        batch_size * target_views,
                        *target.shape[2:],
                    )
                    flattened_prediction = prediction.reshape(
                        batch_size * target_views,
                        *prediction.shape[2:],
                    )
                    metric_values = metric_computer(
                        flattened_target,
                        flattened_prediction,
                    )
                    cpu_values = {
                        name: values.detach().cpu().tolist()
                        for name, values in metric_values.items()
                    }
                    for scene_offset, scene_id in enumerate(scene_ids):
                        for view_index in range(1, 24):
                            flat_index = scene_offset * 23 + view_index - 1
                            record = {
                                "dataset": experiment.dataset_id,
                                "model": display_model,
                                "scene_id": scene_id,
                                "view_index": view_index,
                                "psnr": float(cpu_values["psnr"][flat_index]),
                                "ssim": float(cpu_values["ssim"][flat_index]),
                                "lpips": float(cpu_values["lpips"][flat_index]),
                            }
                            writer.writerow(record)
                            records.append(record)
                    handle.flush()
                    if (
                        batch_index % max(args.progress_every, 1) == 0
                        or batch_index + 1 == len(loader)
                    ):
                        run.log(
                            {
                                "progress/scenes": len(records) // 23,
                                "progress/batches": batch_index + 1,
                            }
                        )

        expected_scenes = len(dataset_for_loader)
        expected_rows = expected_scenes * 23
        _validate_complete_records(
            records,
            expected_scenes=expected_scenes,
            expected_rows=expected_rows,
        )
        if args.max_scenes is None and args.num_shards == 1:
            registry_expected_scenes = int(experiment.dataset["expected_scenes"])
            registry_expected_rows = int(experiment.dataset["expected_rows"])
            if expected_scenes != registry_expected_scenes:
                raise RuntimeError(
                    f"registry expects {registry_expected_scenes} scenes, "
                    f"dataset contains {expected_scenes}"
                )
            if expected_rows != registry_expected_rows:
                raise RuntimeError(
                    f"registry expects {registry_expected_rows} rows, "
                    f"evaluation produced {expected_rows}"
                )

        overall, per_view = write_evaluation_outputs(
            records,
            experiment.output_dir,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=int(experiment.defaults.get("seed", 9595)),
        )
        runtime_seconds = time.time() - start_time
        overall["runtime_seconds"] = runtime_seconds
        overall["expected_scenes"] = expected_scenes
        overall["expected_rows"] = expected_rows
        (experiment.output_dir / "overall_summary.json").write_text(
            json.dumps(overall, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _log_wandb_outputs(
            run,
            overall,
            per_view,
            experiment.output_dir,
            experiment.view_manifest_path,
        )
        partial_raw_path.unlink(missing_ok=True)
        run.summary["state"] = "complete"
        artifact_sha256 = {
            filename: sha256_file(experiment.output_dir / filename)
            for filename in (
                "raw_metrics.csv", "raw_metrics.parquet", "overall_summary.json",
                "per_view_summary.csv", "run_metadata.json",
            )
            if (experiment.output_dir / filename).is_file()
        }
        completion_payload = {
            "state": "complete",
            "experiment_id": experiment.id,
            "dataset": experiment.dataset_id,
            "expected_scenes": expected_scenes,
            "expected_rows": expected_rows,
            "run_signature": run_signature,
            "runtime_sha256": runtime_sha256,
            "registry_sha256": registry_sha256,
            "dataset_manifest_sha256": dataset_manifest_sha256,
            "checkpoint_sha256": checkpoint_sha256,
            "config_sha256": config_sha256,
            "view_manifest_sha256": view_manifest_sha256,
            "evaluation_code_sha256": code_sha256,
            "raw_metrics_sha256": artifact_sha256["raw_metrics.csv"],
            "artifact_sha256": artifact_sha256,
            "wandb_run_id": run.id,
            "wandb_mode": args.wandb_mode,
        }
        print(
            json.dumps(
                {
                    "experiment": experiment.id,
                    "output_dir": str(experiment.output_dir),
                    "overall": overall,
                },
                indent=2,
                sort_keys=True,
            )
        )
    except Exception:
        run.summary["state"] = "failed"
        run.summary["partial_rows"] = len(records)
        raise
    finally:
        run.finish()
    if completion_payload is not None:
        temporary_completion_path = completion_path.with_suffix(".json.tmp")
        temporary_completion_path.write_text(
            json.dumps(completion_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary_completion_path.replace(completion_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
