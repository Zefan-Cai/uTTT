#!/usr/bin/env python3
"""Render deterministic TTT-Dense / TTT-MoE / uTTT-MoE comparison grids."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import re
from pathlib import Path
from typing import Any

from uttt_nvs.eval.common import (
    REPO_ROOT,
    import_model,
    load_checkpoint_strict,
    load_model_config,
    model_batch,
    resolve_experiment,
    validate_resolved_experiment,
)
from uttt_nvs.eval.fixed_view_dataset import FixedViewNVSDataset
from uttt_nvs.eval.metrics import ImageMetricComputer


DEFAULT_REGISTRY = REPO_ROOT / "eval" / "experiments.yaml"
MAINLINE_SUFFIXES = ("ttt_dense", "ttt_moe_e8a1", "uttt_moe_e64a1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--dataset", choices=("gso", "dl3dv"), required=True)
    parser.add_argument("--scene-id", action="append")
    parser.add_argument("--scene-count", type=int, default=8)
    parser.add_argument("--view-indices", default="1,12,23")
    parser.add_argument("--seed", type=int, default=9595)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("renders"),
    )
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="disabled",
    )
    return parser.parse_args()


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    suffix = hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]
    return f"{cleaned[:80] or 'scene'}-{suffix}"


def _select_indices(
    dataset: FixedViewNVSDataset,
    requested_scene_ids: list[str] | None,
    scene_count: int,
    seed: int,
) -> list[int]:
    scene_to_index = {
        str(dataset.metadata(index)["scene_id"]): index
        for index in range(len(dataset))
    }
    if requested_scene_ids:
        missing = sorted(set(requested_scene_ids).difference(scene_to_index))
        if missing:
            raise KeyError(f"Unknown scene ids: {missing}")
        return [scene_to_index[scene_id] for scene_id in requested_scene_ids]
    ranked = sorted(
        scene_to_index,
        key=lambda scene_id: hashlib.sha256(
            f"{seed}\0{scene_id}".encode("utf-8")
        ).digest(),
    )
    return [scene_to_index[scene_id] for scene_id in ranked[:scene_count]]


def _to_pil(tensor: Any) -> Any:
    import numpy as np
    from PIL import Image

    array = (
        tensor.detach()
        .float()
        .clamp(0.0, 1.0)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    return Image.fromarray((array * 255.0).round().astype(np.uint8), mode="RGB")


def _autocast(device: Any, amp_dtype: str) -> Any:
    import torch

    if device.type != "cuda" or amp_dtype == "fp32":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
    return torch.autocast("cuda", dtype=dtype)


def _collate_one(sample: dict[str, Any]) -> dict[str, Any]:
    from torch.utils.data._utils.collate import default_collate

    return default_collate([sample])


def _load_font(size: int, *, bold: bool = False) -> Any:
    from PIL import ImageFont

    candidates = (
        (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            if bold
            else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        ),
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _build_grid(
    scene_id: str,
    view_indices: list[int],
    latest_context: dict[int, Any],
    ground_truth: dict[int, Any],
    predictions: dict[str, dict[int, Any]],
    metrics: dict[str, dict[int, dict[str, float]]],
) -> Any:
    from PIL import Image, ImageDraw

    labels = ["Latest GT context", "Ground truth", *predictions.keys()]
    tile_width, image_height = latest_context[view_indices[0]].size
    header_height = 72
    metric_height = 64
    tile_height = image_height + metric_height
    canvas = Image.new(
        "RGB",
        (
            tile_width * len(labels),
            header_height + tile_height * len(view_indices),
        ),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    title_font = _load_font(16, bold=True)
    header_font = _load_font(15, bold=True)
    body_font = _load_font(14)
    draw.text((10, 7), f"Scene: {scene_id}", fill="black", font=title_font)
    draw.text(
        (10, 29),
        (
            "Teacher-forced sequential NVS: target View k uses "
            "ground-truth Views 0..k-1"
        ),
        fill="black",
        font=body_font,
    )
    for column, label in enumerate(labels):
        left = column * tile_width
        box = draw.textbbox((0, 0), label, font=header_font)
        text_width = box[2] - box[0]
        x = left + max((tile_width - text_width) // 2, 8)
        draw.text((x, 51), label, fill="black", font=header_font)

    for row, view_index in enumerate(view_indices):
        y = header_height + row * tile_height
        canvas.paste(latest_context[view_index], (0, y))
        canvas.paste(ground_truth[view_index], (tile_width, y))
        draw.text(
            (8, y + image_height + 4),
            (
                f"Latest: GT View {view_index - 1}\n"
                f"Full context: GT Views 0..{view_index - 1}"
            ),
            fill="black",
            font=body_font,
            spacing=3,
        )
        draw.text(
            (tile_width + 8, y + image_height + 4),
            f"Target: GT View {view_index}",
            fill="black",
            font=body_font,
        )
        for prediction_offset, (label, images) in enumerate(predictions.items()):
            column = prediction_offset + 2
            x = column * tile_width
            canvas.paste(images[view_index], (x, y))
            values = metrics[label][view_index]
            draw.text(
                (x + 6, y + image_height + 4),
                (
                    f"PSNR {values['psnr']:.2f}   "
                    f"SSIM {values['ssim']:.4f}\n"
                    f"LPIPS {values['lpips']:.4f}"
                ),
                fill="black",
                font=body_font,
                spacing=3,
            )
    return canvas


def main() -> int:
    args = parse_args()
    view_indices = [int(value) for value in args.view_indices.split(",")]
    if not view_indices or any(value < 1 or value > 23 for value in view_indices):
        raise ValueError("--view-indices must be a comma-separated subset of 1..23")
    if len(set(view_indices)) != len(view_indices):
        raise ValueError("--view-indices contains duplicates")

    experiment_ids = [
        f"{args.dataset}_{suffix}" for suffix in MAINLINE_SUFFIXES
    ]
    experiments = [
        resolve_experiment(args.registry, experiment_id)
        for experiment_id in experiment_ids
    ]
    for experiment in experiments:
        validate_resolved_experiment(experiment)

    import torch

    device = torch.device(args.device)
    first_config = load_model_config(experiments[0])
    dataset = FixedViewNVSDataset(
        dataset_manifest=experiments[0].local_manifest_path,
        view_manifest=experiments[0].view_manifest_path,
        image_size=int(first_config.model.image_size),
    )
    selected_indices = _select_indices(
        dataset,
        args.scene_id,
        args.scene_count,
        args.seed,
    )
    samples = [_collate_one(dataset[index]) for index in selected_indices]
    scene_ids = [str(sample["scene_id"][0]) for sample in samples]

    output_root = args.output_root.expanduser().resolve() / args.dataset
    output_root.mkdir(parents=True, exist_ok=True)
    inputs: dict[str, Any] = {}
    latest_context: dict[str, dict[int, Any]] = {}
    ground_truth: dict[str, dict[int, Any]] = {}
    predictions: dict[str, dict[str, dict[int, Any]]] = {}
    metric_rows: dict[str, dict[str, dict[int, dict[str, float]]]] = {}
    for scene_id, sample in zip(scene_ids, samples):
        inputs[scene_id] = _to_pil(sample["image"][0, 0])
        latest_context[scene_id] = {
            view_index: _to_pil(sample["image"][0, view_index - 1])
            for view_index in view_indices
        }
        ground_truth[scene_id] = {
            view_index: _to_pil(sample["image"][0, view_index])
            for view_index in view_indices
        }

    metric_computer = ImageMetricComputer(device=device)
    for experiment in experiments:
        label = str(experiment.raw["display_name"])
        config = load_model_config(experiment)
        model = import_model(config).to(device)
        load_checkpoint_strict(model, experiment.checkpoint_path)
        model.eval()
        predictions[label] = {}
        metric_rows[label] = {}
        with torch.inference_mode():
            for scene_id, sample in zip(scene_ids, samples):
                batch = model_batch(sample, device)
                with _autocast(device, args.amp_dtype):
                    result = model(batch)
                target = result["target"]["image"][0].float()
                prediction = result["rendering"][0].float()
                selected_offsets = [view_index - 1 for view_index in view_indices]
                selected_target = target[selected_offsets]
                selected_prediction = prediction[selected_offsets]
                values = metric_computer(selected_target, selected_prediction)
                predictions[label][scene_id] = {
                    view_index: _to_pil(prediction[view_index - 1])
                    for view_index in view_indices
                }
                metric_rows[label][scene_id] = {
                    view_index: {
                        metric: float(metric_values[offset].item())
                        for metric, metric_values in values.items()
                    }
                    for offset, view_index in enumerate(view_indices)
                }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    grids = []
    for scene_id in scene_ids:
        scene_dir = output_root / _safe_name(scene_id)
        scene_dir.mkdir(parents=True, exist_ok=True)
        inputs[scene_id].save(scene_dir / "input_view_00.png")
        for view_index in view_indices:
            ground_truth[scene_id][view_index].save(
                scene_dir / f"ground_truth_view_{view_index:02d}.png"
            )
            for label, by_scene in predictions.items():
                by_scene[scene_id][view_index].save(
                    scene_dir
                    / f"{_safe_name(label)}_view_{view_index:02d}.png"
                )
        grid = _build_grid(
            scene_id,
            view_indices,
            latest_context[scene_id],
            ground_truth[scene_id],
            {
                label: by_scene[scene_id]
                for label, by_scene in predictions.items()
            },
            {
                label: by_scene[scene_id]
                for label, by_scene in metric_rows.items()
            },
        )
        grid_path = scene_dir / "comparison.png"
        grid.save(grid_path)
        grids.append((scene_id, grid_path))

    if args.wandb_mode != "disabled":
        import wandb

        defaults = experiments[0].defaults
        wandb_config = defaults.get("wandb", {})
        run = wandb.init(
            entity=wandb_config.get("entity"),
            project=wandb_config.get("project", "nvs-eval"),
            name=f"{args.dataset}-mainline-render-comparison",
            job_type="render-comparison",
            mode=args.wandb_mode,
            config={
                "dataset": args.dataset,
                "scene_ids": scene_ids,
                "view_indices": view_indices,
                "seed": args.seed,
                "conditioning_protocol": "teacher_forced_sequential",
                "conditioning_definition": (
                    "prediction for View k is conditioned on "
                    "ground-truth Views 0..k-1"
                ),
                "models": [
                    experiment.raw["display_name"]
                    for experiment in experiments
                ],
            },
        )
        table = wandb.Table(columns=["scene_id", "comparison"])
        for scene_id, grid_path in grids:
            table.add_data(scene_id, wandb.Image(str(grid_path)))
        run.log({"render_comparisons": table})
        run.finish()

    print(f"Wrote {len(grids)} comparison grids to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
