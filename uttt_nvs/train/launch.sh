#!/usr/bin/env bash
# Launch uTTT NVS training with torchrun.
#
#   Single GPU or single node:
#     bash train/launch.sh configs/ownership/obj/uttt_moe_e64a1.yaml -s exp_name my_run
#
#   Multiple nodes: set the standard PyTorch distributed variables and run the
#   same command on every node. WORLD_SIZE is the standard one -- the TOTAL
#   number of processes, i.e. nodes x GPUs-per-node -- and the node count is
#   derived from it. Setting it to a node count instead would make torchrun wait
#   for nodes that never arrive, and the job would hang rather than fail.
#
# Environment variables:
#   NPROC_PER_NODE  GPUs per node (default: 4, matching total_batch_size 128)
#   WORLD_SIZE      total processes across all nodes (nodes x NPROC_PER_NODE)
#   RANK            this node's index, 0-based
#   MASTER_ADDR     hostname of the rank-0 node
#   MASTER_PORT     rendezvous port; its presence is what selects multi-node mode
#   JOB_UUID        rendezvous id, must match on all nodes (default: uttt-nvs)
#   NODE_SYNC       optional script run before torchrun on multi-node jobs

set -euo pipefail
umask 007

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
source "$REPO_ROOT/scripts/launch_common.sh"
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  printf 'Usage: launch.sh [--dry-run] <config.yaml> [trainer args]\n'
  exit 0
fi
DRY_RUN=false
if [[ "${1:-}" == --dry-run ]]; then
  DRY_RUN=true
  shift
fi
(( $# >= 1 )) || uttt_fail 'Expected <config.yaml>; use --help for usage.'
CONFIG="$(uttt_resolve_file "$1" "$REPO_ROOT/$1" "$REPO_ROOT/uttt_nvs/$1")"
shift
NUM_GPU="${NPROC_PER_NODE:-4}"
RDZV_ID="${JOB_UUID:-uttt-nvs}"
uttt_rendezvous "$NUM_GPU" "$RDZV_ID"

export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

if ! "$DRY_RUN" && (( NNODES > 1 )) && [[ -n "${NODE_SYNC:-}" ]]; then
  [[ -f "$NODE_SYNC" ]] || uttt_fail "NODE_SYNC script not found: $NODE_SYNC"
  "${PYTHON:-python3}" "$NODE_SYNC"
fi

# Run from the repository root so that `uttt_nvs` is importable.
cd "$REPO_ROOT"

COMMAND=(torchrun "${RDZV_ARGS[@]}" \
  --nproc_per_node="${NUM_GPU}" \
  -m uttt_nvs.train.trainer "$CONFIG" "$@")
uttt_print_command "${COMMAND[@]}"
if "$DRY_RUN"; then
  exit 0
fi
exec "${COMMAND[@]}"
