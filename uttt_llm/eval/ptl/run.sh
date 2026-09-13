#!/usr/bin/env bash
# Per-token-loss evaluation over a trained checkpoint.
#
#   NPROC_PER_NODE=8 bash eval/ptl/run.sh \
#       configs/exp/eval_ptl.toml /path/to/hf_model_dir [extra flame args]
#
# The checkpoint is an HF-format model directory, produced from a training run
# with `python -m flame.utils.convert_dcp_to_hf`. The toml carries the model
# config, Book-3 data files, and the chunked-lm_head settings under [eval] that
# keep 32K x vocab logits from being materialised at once. Any further
# arguments are forwarded to flame.eval_loss verbatim (for example
# `--job.dump_folder out/`).
set -euo pipefail
EVAL_CONFIG="${1:?usage: run.sh <eval_ptl.toml> <hf_checkpoint_dir> [extra args]}"
CKPT_DIR="${2:?usage: run.sh <eval_ptl.toml> <hf_checkpoint_dir> [extra args]}"
NGPU="${NPROC_PER_NODE:-8}"
cd "$(dirname "$0")/../.."
torchrun --standalone --nproc_per_node="${NGPU}" \
  -m flame.eval_loss \
  --job.config_file "${EVAL_CONFIG}" \
  --checkpoint.load_path "${CKPT_DIR}" \
  "${@:3}"
