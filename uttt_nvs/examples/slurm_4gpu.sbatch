#!/bin/bash
# Train one uTTT NVS config on 4 GPUs — the published setup
# (32 per GPU x 4 GPUs = total batch 128).
#
#     sbatch uttt_nvs/examples/slurm_4gpu.sbatch uttt_nvs/configs/ownership/obj/uttt_moe_e64a1.yaml
#
# This is an example, not a supported entry point: partition names, account
# strings and module systems differ everywhere. Copy it and edit the header.

#SBATCH --job-name=uttt-nvs
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

CONFIG="${1:?usage: sbatch slurm_4gpu.sbatch <config.yaml> [extra -s overrides]}"
shift || true

# 4 GPUs keeps the per-GPU batch at the published 32 and the total at 128.
# Changing the GPU count means changing batch_size_per_gpu to match, or the
# trainer will stop with a batch-size mismatch.
export NPROC_PER_NODE=4

# Distinct rendezvous id per job, so concurrent jobs do not collide.
export JOB_UUID="uttt-nvs-${SLURM_JOB_ID:-$$}"

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

bash uttt_nvs/train/launch.sh "${CONFIG}" "$@"
