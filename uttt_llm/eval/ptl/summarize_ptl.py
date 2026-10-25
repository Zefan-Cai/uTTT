#!/usr/bin/env python3
"""Reduce a per-position loss JSON to the paper's PTL number.

`flame.eval_loss` writes `<save_prefix>_loss_per_token_position.json`: a flat
list with the mean next-token loss at every position of the evaluation
sequences (32,768 entries at 32K). The last position has no next token and is
NaN by design. The paper's per-token loss (PTL) is the NaN-aware mean over
positions >= 30,000 (0-based), i.e. the final 2,768 positions of a 32K sequence.

    python eval/ptl/summarize_ptl.py eval_out/default_loss_per_token_position.json
    # (<job.dump_folder>/<training.save_prefix>_loss_per_token_position.json)
    python eval/ptl/summarize_ptl.py --chunks 8 a.json b.json   # also 8 equal blocks

No torch dependency, so it runs on the login node.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def nan_mean(values) -> float:
    kept = [v for v in values if v is not None and not math.isnan(v)]
    return sum(kept) / len(kept) if kept else math.nan


def load_positions(path: Path) -> list[float]:
    data = json.loads(path.read_text())
    if not isinstance(data, list) or not data:
        raise ValueError(f"{path}: expected a non-empty JSON list of per-position losses")
    out = []
    for v in data:
        if v is None:
            out.append(math.nan)
        elif isinstance(v, (int, float)):
            out.append(float(v))
        else:
            raise ValueError(f"{path}: non-numeric entry {v!r}")
    return out


def summarize(losses: list[float], min_position: int, chunks: int = 0) -> dict:
    if min_position < 0 or min_position >= len(losses):
        raise ValueError(f"min_position {min_position} outside 0..{len(losses) - 1}")
    tail = losses[min_position:]
    result = {
        "positions": len(losses),
        "min_position": min_position,
        "tail_positions": len(tail),
        "tail_nan": sum(1 for v in tail if math.isnan(v)),
        "ptl": nan_mean(tail),
        "mean_all": nan_mean(losses),
    }
    if chunks:
        size = math.ceil(len(losses) / chunks)
        result["chunk_means"] = [
            nan_mean(losses[i:i + size]) for i in range(0, len(losses), size)
        ][:chunks]
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("json", nargs="+", type=Path, help="per-position loss JSON file(s)")
    parser.add_argument("--min-position", type=int, default=30000,
                        help="first 0-based position included in PTL (paper: 30000)")
    parser.add_argument("--expect-positions", type=int, default=32768,
                        help="warn when a file does not have this many positions (0 disables)")
    parser.add_argument("--chunks", type=int, default=0,
                        help="also report NaN-aware means over this many equal position blocks")
    parser.add_argument("--output", type=Path, default=None, help="write the summaries as JSON")
    args = parser.parse_args(argv)

    summaries = {}
    for path in args.json:
        losses = load_positions(path)
        if args.expect_positions and len(losses) != args.expect_positions:
            print(f"warning: {path} has {len(losses)} positions, expected {args.expect_positions}",
                  file=sys.stderr)
        summary = summarize(losses, args.min_position, args.chunks)
        summaries[str(path)] = summary
        line = f"{path}: PTL(>= {args.min_position}) = {summary['ptl']:.4f}  (mean over all positions {summary['mean_all']:.4f}"
        if summary["tail_nan"]:
            line += f", {summary['tail_nan']} NaN position(s) skipped"
        print(line + ")")
        if args.chunks:
            print("  blocks: " + ", ".join(f"{m:.4f}" for m in summary["chunk_means"]))
    if args.output:
        args.output.write_text(json.dumps(summaries, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
