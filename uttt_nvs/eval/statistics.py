"""Raw-record validation and scene-bootstrap summaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping


RAW_FIELDS = (
    "dataset",
    "model",
    "scene_id",
    "view_index",
    "psnr",
    "ssim",
    "lpips",
)
METRIC_NAMES = ("psnr", "ssim", "lpips")


def records_frame(records: Iterable[Mapping[str, Any]] | Any) -> Any:
    import pandas as pd

    if isinstance(records, pd.DataFrame):
        frame = records.copy()
    else:
        frame = pd.DataFrame.from_records(records)
    missing = [field for field in RAW_FIELDS if field not in frame.columns]
    if missing:
        raise ValueError(f"raw metric records are missing columns: {missing}")
    frame = frame.loc[:, RAW_FIELDS].copy()
    frame["view_index"] = pd.to_numeric(
        frame["view_index"], errors="raise"
    ).astype(int)
    invalid_views = frame.loc[
        ~frame["view_index"].between(1, 23), "view_index"
    ].unique()
    if len(invalid_views):
        raise ValueError(
            f"view_index must be in 1..23, got {sorted(invalid_views.tolist())}"
        )
    for metric in METRIC_NAMES:
        frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
    return frame


def _bootstrap_mean_ci(
    values: Any,
    *,
    samples: int,
    seed: int,
    chunk_size: int = 512,
) -> tuple[float | None, float | None]:
    import numpy as np

    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None, None
    if finite.size == 1 or samples <= 0:
        value = float(finite[0])
        return value, value
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, chunk_size):
        stop = min(start + chunk_size, samples)
        indices = rng.integers(
            0,
            finite.size,
            size=(stop - start, finite.size),
        )
        means[start:stop] = finite[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def summarize_records(
    records: Iterable[Mapping[str, Any]] | Any,
    *,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 9595,
) -> tuple[dict[str, Any], Any]:
    """Return overall scene-macro metrics and long-form per-view statistics."""

    import numpy as np
    import pandas as pd

    frame = records_frame(records)
    datasets = frame["dataset"].dropna().astype(str).unique()
    models = frame["model"].dropna().astype(str).unique()
    if len(datasets) != 1 or len(models) != 1:
        raise ValueError(
            "summarize_records expects exactly one dataset and one model"
        )
    dataset = datasets[0]
    model = models[0]

    overall: dict[str, Any] = {
        "dataset": dataset,
        "model": model,
        "num_valid_scenes": int(frame["scene_id"].nunique()),
        "num_valid_samples": int(len(frame)),
        "bootstrap_samples": int(bootstrap_samples),
        "bootstrap_seed": int(bootstrap_seed),
    }
    per_view_rows: list[dict[str, Any]] = []

    for metric_offset, metric in enumerate(METRIC_NAMES):
        scene_values = (
            frame.groupby("scene_id", sort=True)[metric]
            .mean()
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        overall[f"overall_{metric}"] = (
            float(scene_values.mean()) if len(scene_values) else None
        )
        overall[f"overall_{metric}_n_valid"] = int(len(scene_values))
        ci_low, ci_high = _bootstrap_mean_ci(
            scene_values.to_numpy(),
            samples=bootstrap_samples,
            seed=bootstrap_seed + metric_offset * 100_000,
        )
        overall[f"overall_{metric}_ci95_low"] = ci_low
        overall[f"overall_{metric}_ci95_high"] = ci_high

        for view_index in range(1, 24):
            values = (
                frame.loc[frame["view_index"] == view_index, metric]
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
                .to_numpy(dtype=float)
            )
            mean = float(np.mean(values)) if len(values) else None
            ci_low, ci_high = _bootstrap_mean_ci(
                values,
                samples=bootstrap_samples,
                # Reusing this seed schedule across models gives paired
                # bootstrap indices whenever their valid scene sets match.
                seed=bootstrap_seed + metric_offset * 100_000 + view_index,
            )
            per_view_rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "view_index": view_index,
                    "metric": metric,
                    "mean": mean,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "n_valid": int(len(values)),
                }
            )

    return overall, pd.DataFrame.from_records(per_view_rows)


def write_evaluation_outputs(
    records: Iterable[Mapping[str, Any]] | Any,
    output_dir: str | Path,
    *,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 9595,
) -> tuple[dict[str, Any], Any]:
    """Write raw CSV/Parquet plus overall and per-view summaries."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame = records_frame(records)
    overall, per_view = summarize_records(
        frame,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    frame.to_csv(output / "raw_metrics.csv", index=False)
    frame.to_parquet(output / "raw_metrics.parquet", index=False)
    per_view.to_csv(output / "per_view_summary.csv", index=False)
    (output / "overall_summary.json").write_text(
        json.dumps(overall, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return overall, per_view
