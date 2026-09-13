#!/usr/bin/env bash

uttt_fail() {
  printf 'Launch error: %s\n' "$*" >&2
  exit 2
}

uttt_positive_integer() {
  [[ "$2" =~ ^[1-9][0-9]{0,8}$ ]] || uttt_fail "$1 must be a positive integer (at most nine digits), got '$2'."
}

uttt_resolve_file() {
  local requested="$1"
  shift
  local candidate
  for candidate in "$requested" "$@"; do
    if [[ -f "$candidate" && -r "$candidate" ]]; then
      printf '%s/%s\n' "$(cd "$(dirname "$candidate")" && pwd)" "$(basename "$candidate")"
      return 0
    fi
  done
  uttt_fail "Cannot read configuration: $requested"
}

uttt_rendezvous() {
  local processes="$1"
  local rendezvous_id="$2"
  uttt_positive_integer NPROC_PER_NODE "$processes"
  RDZV_ARGS=(--standalone)
  NNODES=1
  if [[ -n "${MASTER_PORT:-}" ]]; then
    [[ -n "${MASTER_ADDR:-}" && -n "${WORLD_SIZE:-}" ]] || uttt_fail 'MASTER_PORT requires MASTER_ADDR and WORLD_SIZE (total processes).'
    uttt_positive_integer MASTER_PORT "$MASTER_PORT"
    (( MASTER_PORT <= 65535 )) || uttt_fail 'MASTER_PORT must be in 1..65535.'
    uttt_positive_integer WORLD_SIZE "$WORLD_SIZE"
    (( WORLD_SIZE % processes == 0 )) || uttt_fail 'WORLD_SIZE must be a multiple of NPROC_PER_NODE; it counts processes, not nodes.'
    NNODES=$((WORLD_SIZE / processes))
    local node_rank="${NODE_RANK:-${RANK:-}}"
    if [[ -z "$node_rank" ]]; then
      (( NNODES == 1 )) || uttt_fail 'Set NODE_RANK (or RANK) to this node index for a multi-node run.'
      node_rank=0
    fi
    [[ "$node_rank" =~ ^(0|[1-9][0-9]{0,8})$ ]] || uttt_fail 'NODE_RANK/RANK must be a non-negative integer.'
    (( node_rank < NNODES )) || uttt_fail "NODE_RANK/RANK must be less than NNODES=$NNODES."
    RDZV_ARGS=(--nnodes="$NNODES" --node-rank="$node_rank" --rdzv-id="$rendezvous_id"
               --rdzv-backend=c10d --rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}")
  elif [[ -n "${MASTER_ADDR:-}${WORLD_SIZE:-}${NODE_RANK:-}${RANK:-}" ]]; then
    uttt_fail 'Partial distributed environment: set MASTER_PORT, MASTER_ADDR, WORLD_SIZE and NODE_RANK together, or unset them for a standalone run.'
  fi
}

uttt_print_command() {
  printf 'Working directory: %s\nCommand: ' "$PWD"
  printf '%q ' "$@"
  printf '\n'
}
