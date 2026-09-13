#!/bin/bash
# Training on 16 GPUs. Only the total GPU count matters for the batch size --
# how your site groups them into nodes does not.
#
#     sbatch uttt_nvs/examples/slurm_multigpu.sbatch \
#         uttt_nvs/configs/capacity/obj/uttt_moe_e64a1.yaml \
#         -s training.batch_size_per_gpu 8
#
# 16 GPUs at the published total of 128 needs 8 per GPU, hence the override.
# Change the GPU count and `batch_size_per_gpu` together, or the trainer stops
# with a batch-size mismatch.
#
# This is an example, not a supported entry point. Edit the header for your site.

#SBATCH --job-name=uttt-nvs-16gpu
#SBATCH --nodes=2
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

CONFIG="${1:?usage: sbatch slurm_multigpu.sbatch <config.yaml> [extra -s overrides]}"
shift || true

# 16 GPUs total. The two SBATCH lines above and this one must agree.
export NPROC_PER_NODE=8
export JOB_UUID="uttt-nvs-${SLURM_JOB_ID:-$$}"

# launch.sh reads the standard PyTorch distributed variables. Slurm does not set
# them itself, so derive them from the allocation.
export MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)"
export MASTER_PORT="${MASTER_PORT:-29500}"
# WORLD_SIZE is the standard one: total processes across all nodes.
export WORLD_SIZE=$(( SLURM_JOB_NUM_NODES * NPROC_PER_NODE ))
export RANK="${SLURM_NODEID:-0}"

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# One task per node; srun fans the same command out to every node, and torchrun
# starts NPROC_PER_NODE workers within each.
srun --ntasks-per-node=1 bash -c '
    export RANK=${SLURM_NODEID}
    bash uttt_nvs/train/launch.sh "$@"
' _ "${CONFIG}" "$@"
