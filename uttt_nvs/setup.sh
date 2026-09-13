#!/usr/bin/env bash
# Install uTTT NVS dependencies.
#
# PyTorch is not installed here: install the verified reference build first.
# Reference environment: Python 3.10, CUDA 12.8, torch 2.8.0, torchvision 0.23.0.
#
#   pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
#   bash uttt_nvs/setup.sh
set -euo pipefail
pip install -r "$(dirname "$0")/requirements.txt"

# flash-attn cannot go in requirements.txt: it needs torch at build time, so
# pip's build isolation has to be off. It backs models/transformer_block.py.
pip install flash-attn --no-build-isolation
