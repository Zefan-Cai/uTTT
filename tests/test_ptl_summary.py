"""The PTL reducer reproduces the paper's definition on synthetic data."""

import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "uttt_llm" / "eval" / "ptl"))
import summarize_ptl  # noqa: E402


def test_ptl_is_nan_aware_mean_over_tail(tmp_path):
    losses = [float(i) for i in range(32768)]
    losses[-1] = math.nan                      # no next token at the last position
    path = tmp_path / "x.json"
    path.write_text(json.dumps([None if math.isnan(v) else v for v in losses]))

    summary = summarize_ptl.summarize(summarize_ptl.load_positions(path), 30000, chunks=4)
    assert summary["tail_positions"] == 2768
    assert summary["tail_nan"] == 1
    assert summary["ptl"] == pytest.approx(sum(range(30000, 32767)) / 2767)
    assert len(summary["chunk_means"]) == 4
    assert not math.isnan(summary["chunk_means"][-1])   # last block is NaN-safe


def test_cli_writes_json_and_warns_on_short_files(tmp_path, capsys):
    path = tmp_path / "short.json"
    path.write_text(json.dumps([1.0, 2.0, 3.0, None]))
    out = tmp_path / "summary.json"
    assert summarize_ptl.main([str(path), "--min-position", "1", "--output", str(out)]) == 0
    captured = capsys.readouterr()
    assert "expected 32768" in captured.err
    assert json.loads(out.read_text())[str(path)]["ptl"] == pytest.approx(2.5)


def test_min_position_out_of_range():
    with pytest.raises(ValueError):
        summarize_ptl.summarize([1.0, 2.0], 2)
