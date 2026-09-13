#!/usr/bin/env bash
# Train one paper configuration.
#
#   Single node:
#     NPROC_PER_NODE=8 bash launch/train.sh \
#         configs/exp/train_124M_32k.toml configs/main/124M/uttt_moe.json my_run
#
#   Multiple nodes (the published 760M runs used two): set the standard
#   PyTorch distributed variables and run the same command on every node.
#   WORLD_SIZE is the TOTAL number of processes (nodes x GPUs-per-node); the
#   node count is derived from it. MASTER_PORT being set is what selects
#   multi-node mode.
#
# Data paths come from uttt_llm/datasets.yaml (copy datasets.example.yaml and
# fill it in once); an explicit data_files in the exp toml is overridden by it.
# The exp toml declares expected_global_batch_size = 32, and the trainer
# refuses to start if per-device batch x data-parallel degree does not match.
set -euo pipefail
DOMAIN_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$DOMAIN_ROOT/../scripts/launch_common.sh"
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  printf 'Usage: train.sh [--dry-run] <exp.toml> <model.json> [job_name] [extra flame args]\n'
  exit 0
fi
DRY_RUN=false
if [[ "${1:-}" == --dry-run ]]; then
  DRY_RUN=true
  shift
fi
(( $# >= 2 )) || uttt_fail 'Expected <exp.toml> <model.json>; use --help for usage.'
EXP_CONFIG="$(uttt_resolve_file "$1" "$DOMAIN_ROOT/$1")"
MODEL_CONFIG="$(uttt_resolve_file "$2" "$DOMAIN_ROOT/$2")"
shift 2
JOB_NAME="$(basename "${MODEL_CONFIG%.json}")"
if [[ $# -gt 0 && "$1" != --* ]]; then
  JOB_NAME="$1"
  shift
fi
NGPU="${NPROC_PER_NODE:-8}"
RDZV_ID="${JOB_UUID:-uttt-llm}"
uttt_rendezvous "$NGPU" "$RDZV_ID"
cd "$DOMAIN_ROOT"

COMMAND=(torchrun "${RDZV_ARGS[@]}" --nproc_per_node="${NGPU}"
  -m flame.train --job.config_file "${EXP_CONFIG}"
  --job.dump_folder "runs/${JOB_NAME}" --model.config "${MODEL_CONFIG}")
if [ -f datasets.yaml ]; then
  TRAIN_FILES="$("${PYTHON:-python3}" -c 'import yaml
with open("datasets.yaml", encoding="utf-8") as handle:
    config = yaml.safe_load(handle)
training = config.get("train") if isinstance(config, dict) else None
files = training.get("files") if isinstance(training, dict) else None
if not isinstance(files, str) or not files.strip():
    raise SystemExit("datasets.yaml must contain a non-empty train.files string")
print(files)')"
  COMMAND+=(--training.data_files "${TRAIN_FILES}")
fi

COMMAND+=("$@")
uttt_print_command "${COMMAND[@]}"
if "$DRY_RUN"; then
  exit 0
fi
exec "${COMMAND[@]}"
