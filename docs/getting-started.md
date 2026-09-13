# Getting started

Run commands from the repository root unless a block explicitly changes directory.
The first two levels below need no dataset download, W&B account or CUDA runtime.
They deliberately do not construct the research models.

## 1. Inspect configurations on any CPU machine

Use Python 3.10 or 3.11 for the development checks. Python 3.10 is the original
CUDA reference environment; Python 3.11 is also covered by the CPU CI matrix.

```bash
git clone https://github.com/Zefan-Cai/uTTT.git
cd uTTT
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install pyyaml
python -m tools.check_configs --check-catalog
```

On Python 3.10, also install `tomli`. The checker validates all 124 released
configuration files and verifies that the generated catalog is current. It
checks source class names using Python's AST, not by importing CUDA extensions.
It permits dataset/checkpoint placeholders: those are deployment inputs, not
static architecture errors.

Inspect the launch command without starting workers:

```bash
bash uttt_nvs/train/launch.sh --dry-run \
  uttt_nvs/configs/ownership/obj/uttt_moe_e64a1.yaml

bash uttt_llm/launch/train.sh --dry-run \
  uttt_llm/configs/exp/train_124M_32k.toml \
  uttt_llm/configs/main/124M/uttt_moe.json first_run
```

Dry runs validate configuration file paths and distributed environment values,
print the working directory and shell-escaped command, and exit without
launching torchrun or executing `NODE_SYNC`. They do not verify GPU capacity or
parse every trainer override. If an LLM `datasets.yaml` exists, it is read so
that the printed command reflects the real data override.

## 2. Validate the CPU pipeline

Install CPU PyTorch separately from the CUDA training environment:

```bash
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

On macOS, use `python -m pip install torch==2.8.0` instead of the Linux CPU
wheel index. These dependencies exercise checkpointing, launchers, configuration
invariants and synthetic data, but do not include flash-attn or Triton.

Generate a tiny, valid camera dataset:

```bash
python -m uttt_nvs.data.make_toy_dataset \
  --out /tmp/uttt_toy --scenes 16 --views 24 --size 256 --repeat 16
python -m tools.check_manifest /tmp/uttt_toy/manifest.txt \
  --min-views 24 --check-images
```

This yields 16 unique scenes, 24 images per scene, and 256 manifest entries.
Repeated entries make complete training batches possible; they do not increase
dataset diversity. The images are procedural pipeline fixtures, not meaningful
multi-view training data. Camera paths in the manifest are absolute.

## 3. Install the CUDA training environment

Use a separate environment on Linux with NVIDIA GPUs. The reference stack is
Python 3.10, CUDA 12.8, torch 2.8.0+cu128 and Triton 3.4.0. NVS also uses
torchvision 0.23.0. A different CUDA stack requires a deliberate compatibility
check, not just changing one dependency version.

```bash
python -m pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r uttt_nvs/requirements.txt
python -m pip install flash-attn --no-build-isolation
python -m uttt_nvs.tests.smoke_configs --construct --strict
```

For LLM work, install `uttt_llm/requirements.txt` instead, after PyTorch, then
build flash-attn. Run the model-construction check from the domain directory:

```bash
(cd uttt_llm && python -m tests.smoke_configs --construct)
```

Construction can require substantial host RAM for the 760M configurations.
Passing construction proves imports and initialization, not forward/backward
kernel correctness. See [testing](testing.md) for the remaining gates.

## 4. Run a bounded NVS smoke experiment

Create the ignored file `uttt_nvs/datasets.yaml` with the absolute path printed
by the generator:

```yaml
obj:
  train: /tmp/uttt_toy/manifest.txt
  eval: /tmp/uttt_toy/manifest.txt
```

Using the same manifest twice is acceptable only for this synthetic plumbing
check. Real train/test splits must be disjoint.

```bash
NPROC_PER_NODE=4 bash uttt_nvs/train/launch.sh \
  uttt_nvs/configs/ownership/obj/uttt_moe_e64a1.yaml \
  -s exp_name toy_smoke \
  -s training.max_fwdbwd_passes 5 \
  -s training.round_max_fwdbwd_passes_to_epoch False \
  -s training.wandb_offline True
```

Expected artifacts live under `experiments/toy_smoke/`, including a completed
`ckpt_*.pt`. The first step includes Triton compilation/autotuning. Check finite
loss, successful backward/optimizer updates and checkpoint creation; do not
interpret toy-set quality as model performance. Reusing `toy_smoke` resumes its
existing checkpoint, so use a new experiment name for a fresh smoke test.

For a smaller GPU topology, change the batch and accumulation deliberately
using [training arithmetic](training.md). No universal VRAM-fit guarantee is
made for these research configurations.

## 5. Move to real data

Follow [data contracts](data-contracts.md), choose a config from the
[catalog](configuration-catalog.md), and save the exact launch command and Git
SHA. Start with a short run, test resume, then extend the schedule. For LLM
training, follow the [domain quick start](../uttt_llm/README.md): the release
does not synthesize a long-context language dataset for you.
