# Changelog

## 2026-09-12 — Repository split and reproducibility upgrade

### Preserved

- Renamed the original GitHub repository to `uTTT-DEV`, preserving its history.
  Published the enhanced files in the independent `uTTT` repository as a fresh
  initial commit, without importing the development repository's commits.
- Retained all original model implementations, 64 NVS YAMLs, 57 LLM model JSONs,
  three LLM job TOMLs, result figures and license.
- No new accuracy, latency, memory or convergence claims; no released weights
  or datasets were added.

### Improved

- CPU-only configuration validation, including static class/model-type checks,
  dimension/routing/batch invariants, and a generated complete config catalog.
- Shared training-launch validation, `--help` / `--dry-run`, portable config
  paths, explicit node ranks, standalone rendezvous and safe argument forwarding.
- LLM extra options work even when an optional job name is omitted; the launcher
  also works with Bash 3.2 when no local dataset registry exists.
- NVS checkpoint writes are atomic; numeric checkpoint discovery excludes
  temporary/malformed filenames; corrupt latest files can fall back to older ones.
- NVS selects one restore source even when a fine-tune resets counters to zero.
- Toy scenes default to 24 views, use proper right-handed OpenCV camera frames,
  reject invalid dimensions and emit absolute manifest paths.
- Local manifest preflight checks validate calibration, view counts, image paths
  and optionally image contents before distributed training.
- CPU regression tests and GitHub Actions for Python 3.10/3.11.
- Detailed architecture, installation, configuration, data, training, evaluation,
  troubleshooting, testing, contribution, migration and Chinese guides.

### Intentional behavior changes

- NVS `save_last_n_ckpts` now counts checkpoint files, not step distance; the
  default is three files. Review explicit legacy values such as 1001.
- Explicit missing/empty/unreadable NVS checkpoint sources raise errors instead
  of silently initializing from scratch.
- Invalid or partial distributed environments fail before launching workers.
  Multi-node jobs must specify `NODE_RANK` or the legacy `RANK` node index.
- Synthetic camera generation changes geometry and view count; regenerate toy
  fixtures instead of comparing their image/pose files to the old generator.
