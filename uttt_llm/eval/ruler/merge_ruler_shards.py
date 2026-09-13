#!/usr/bin/env python3
"""Merge RULER sample-sharded lm-eval outputs."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-shards", type=int, required=True)
    return parser.parse_args()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "samples"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def numeric_metric_items(record: dict[str, Any]):
    metric_names = record.get("metrics")
    if not isinstance(metric_names, list):
        metric_names = [
            key
            for key, value in record.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ]

    for metric_name in metric_names:
        value = record.get(metric_name)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value != -1:
            yield str(metric_name), float(value)


def normalize_metric_name(metric_name: str) -> str:
    metric_name = str(metric_name)
    if "," in metric_name:
        metric_name = metric_name.split(",", 1)[0]
    return metric_name


def result_metric_items(task_result: dict[str, Any]):
    for metric_name, value in task_result.items():
        metric_name = normalize_metric_name(metric_name)
        if metric_name in {"alias", "name", "sample_len"} or metric_name.endswith("_stderr"):
            continue
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value != -1:
            yield metric_name, float(value)


def sample_len_from_result(task_result: dict[str, Any], fallback: int = 1) -> int:
    sample_len = task_result.get("sample_len", fallback)
    try:
        sample_len = int(sample_len)
    except (TypeError, ValueError):
        sample_len = fallback
    return max(sample_len, 1)


def main() -> int:
    args = parse_args()
    shards_dir = args.shards_dir.resolve()
    output_dir = args.output_dir.resolve()
    sample_out_dir = output_dir / "samples"
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_out_dir.mkdir(parents=True, exist_ok=True)

    rank_dirs = sorted(
        path
        for path in shards_dir.iterdir()
        if path.is_dir() and (path / "shard_summary.json").exists()
    )
    shard_summaries = []
    shard_results = []
    metrics: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    weighted_metrics: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(lambda: [0.0, 0.0])
    )
    result_sample_counts: dict[str, int] = defaultdict(int)
    sample_log_counts: dict[str, int] = defaultdict(int)
    sample_output_handles = {}
    sample_files: set[str] = set()
    used_result_metrics = False

    try:
        for rank_dir in rank_dirs:
            summary_path = rank_dir / "shard_summary.json"
            if summary_path.exists():
                shard_summaries.append(load_json(summary_path))

            result_path = rank_dir / "results" / "ruler" / "results.json"
            if result_path.exists():
                shard_result = load_json(result_path)
                shard_results.append(shard_result)
                for task_name, task_result in sorted(
                    (shard_result.get("results") or {}).items()
                ):
                    if not isinstance(task_result, dict):
                        continue
                    sample_len = sample_len_from_result(task_result)
                    found_task_metric = False
                    for metric_name, metric_value in result_metric_items(task_result):
                        weighted = weighted_metrics[task_name][metric_name]
                        weighted[0] += metric_value * sample_len
                        weighted[1] += sample_len
                        found_task_metric = True
                    if found_task_metric:
                        used_result_metrics = True
                        result_sample_counts[task_name] += sample_len

            samples_dir = rank_dir / "results" / "ruler" / "samples"
            for sample_path in sorted(samples_dir.glob("*.jsonl")):
                task_name = sample_path.stem
                merged_sample_path = sample_out_dir / f"{safe_filename(task_name)}.jsonl"
                sample_files.add(str(merged_sample_path.relative_to(output_dir)))
                handle = sample_output_handles.get(task_name)
                if handle is None:
                    handle = merged_sample_path.open("w", encoding="utf-8")
                    sample_output_handles[task_name] = handle

                for record in iter_jsonl(sample_path):
                    record["_shard"] = rank_dir.name
                    handle.write(json.dumps(record, sort_keys=True, default=str))
                    handle.write("\n")
                    sample_log_counts[task_name] += 1
                    for metric_name, metric_value in numeric_metric_items(record):
                        metrics[task_name][metric_name].append(metric_value)
    finally:
        for handle in sample_output_handles.values():
            handle.close()

    task_results: dict[str, dict[str, float]] = {}
    merge_source = "shard_results"
    if used_result_metrics:
        for task_name, metric_values in sorted(weighted_metrics.items()):
            task_results[task_name] = {
                metric_name: weighted_sum / weight
                for metric_name, (weighted_sum, weight) in sorted(metric_values.items())
                if weight
            }
    else:
        merge_source = "sample_logs"
        for task_name, metric_values in sorted(metrics.items()):
            task_results[task_name] = {
                metric_name: sum(values) / len(values)
                for metric_name, values in sorted(metric_values.items())
                if values
            }
    sample_counts = result_sample_counts if used_result_metrics else sample_log_counts

    group_metric_values: dict[str, list[float]] = defaultdict(list)
    for task_metric_values in task_results.values():
        for metric_name, metric_value in task_metric_values.items():
            group_metric_values[metric_name].append(metric_value)

    group_results = {
        metric_name: sum(values) / len(values)
        for metric_name, values in sorted(group_metric_values.items())
        if values
    }

    all_shards_present = len(shard_summaries) >= args.expected_shards
    all_shards_ok = all(
        summary.get("exit_codes", {}).get("ruler") == 0 for summary in shard_summaries
    )
    status = "completed" if all_shards_present and all_shards_ok else "partial"

    first_result = shard_results[0] if shard_results else {}
    final_results = {
        "results": task_results,
        "groups": {"ruler": group_results},
        "sample_counts": dict(sorted(sample_counts.items())),
        "sample_files": sorted(sample_files),
        "shards": shard_summaries,
        "merge": {
            "generated_at_utc": utc_now_iso(),
            "expected_shards": args.expected_shards,
            "found_shards": len(shard_summaries),
            "status": status,
            "source": merge_source,
        },
        "config": first_result.get("config", {}),
        "versions": first_result.get("versions", {}),
        "git_hash": first_result.get("git_hash"),
    }
    (output_dir / "results.json").write_text(
        json.dumps(final_results, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

    summary = {
        "generated_at_utc": utc_now_iso(),
        "overall_status": status,
        "expected_shards": args.expected_shards,
        "found_shards": len(shard_summaries),
        "exit_codes": {
            summary.get("shard", {}).get("label", f"rank-{i}"): summary.get(
                "exit_codes", {}
            ).get("ruler")
            for i, summary in enumerate(shard_summaries)
        },
        "final_result_file": "final/results.json",
        "sample_files": [f"final/{path}" for path in sorted(sample_files)],
        "sample_counts": dict(sorted(sample_counts.items())),
        "shards": shard_summaries,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return 0 if status == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
