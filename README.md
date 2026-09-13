# uTTT — Universal Test-Time Training

[Documentation](docs/README.md) · [中文指南](docs/README.zh-CN.md) ·
[Configuration catalog](docs/configuration-catalog.md) · [Changelog](CHANGELOG.md)

> **Repository split:** this is the upgraded `uTTT` repository. The original
> repository is preserved as [uTTT-DEV](https://github.com/Zefan-Cai/uTTT-DEV),
> while this release starts with a fresh initial commit containing all code and
> documentation, without importing the old commit history. Existing `/uTTT.git` remotes
> now point to this release, not the development repository. See
> [migration notes](docs/migration.md) before pushing from an older checkout.

Test-Time Training compresses context into fast weights, and existing designs
make those fast weights **layer-private**: every layer owns a state no other
layer touches. This repository studies the alternative — **universal fast
weights**, where the whole depth stack shares one state or one pool of states —
instantiated as **uTTT-Dense** (a single shared dense operator) and
**uTTT-MoE** (a globally shared expert pool), with **TTT-MoE** (a routed pool
that stays layer-local) as the controlled midpoint. Both task domains are
covered: LVSM-style novel view synthesis and 32K-context language modeling.

## The ownership ladder

| Regime | Who owns the fast weights | Models |
|---|---|---|
| Layer-private | each layer, exclusively | LaCT, TTT-Dense |
| Layer-local pool | each layer routes within its own pool | TTT-MoE |
| Universal (shared) | the whole stack | **uTTT-Dense**, **uTTT-MoE** |

The two domains implement the ladder in opposite ways, and each half's README
explains its own: in `uttt_nvs/` the ladder is a configuration switch on a
single implementation, while `uttt_llm/` ships one model package per paper row,
exactly as the experiments ran.

## Repository structure

```
uTTT/
├── uttt_nvs/    novel view synthesis: training + PSNR/SSIM/LPIPS evaluation
│                64 configs — Objaverse/GSO and DL3DV, including scale ladders
├── uttt_llm/    language modeling at 32K: training + PTL/RULER evaluation
│                57 configs — 124M and 760M, global batch 32
├── tools/       CPU-only config and local-manifest validation
├── tests/       CPU regression tests for launchers, checkpoints and data
├── scripts/     shared launcher validation
└── docs/        project page, architecture and reproducibility guides
```

Coverage of the two uTTT variants:

| | Novel view synthesis | Language modeling |
|---|---|---|
| **uTTT-Dense** | released (r1/r2/r4/r8) | not yet released |
| **uTTT-MoE** | released | released (124M, 760M) |

Each domain also ships its baselines and the TTT-MoE midpoint; see the
per-domain READMEs for the full model tables.

## What is not included

This repository ships code and configurations. It does not ship:

- **Trained checkpoints.** Every number in the paper comes from a run started
  from scratch: 20,000 steps at a global batch of 128 for novel view synthesis,
  10,240 (124M) and 40,960 (760M) steps at a global batch of 32 for language
  modeling. Reproducing a row means paying that cost.
- **Datasets or renderings.** Each half's README names the official sources and
  the format its loader expects; nothing is redistributed here.
- **The evaluation registry's contents.** `uttt_nvs/eval/experiments.yaml`
  describes the metric rerun in terms of your own checkpoints, manifests and
  Weights & Biases account, and its placeholders have to be filled in before it
  will run.

## Install and verify

**No GPU yet? Start with the static checks.** They require only PyYAML (and
`tomli` on Python 3.10), not torch, flash-attn, Triton or a dataset:

```bash
python -m pip install pyyaml 'tomli; python_version < "3.11"'
python -m tools.check_configs --check-catalog
bash uttt_nvs/train/launch.sh --dry-run \
    uttt_nvs/configs/ownership/obj/uttt_moe_e64a1.yaml
```

The checker covers all 64 NVS YAMLs, 57 LLM architecture JSONs and 3 LLM job
TOMLs. See [getting started](docs/getting-started.md) for the CPU test environment,
toy-data validation and a bounded first run. Static validation is not GPU
execution or evidence that a published result was reproduced.

The two halves have separate `requirements.txt` and share one reference
environment: Python 3.10 · CUDA 12.8 · torch 2.8.0+cu128.

```bash
# novel view synthesis
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r uttt_nvs/requirements.txt && pip install flash-attn --no-build-isolation
python -m uttt_nvs.tests.smoke_configs --construct --strict

# language modeling
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r uttt_llm/requirements.txt && pip install flash-attn --no-build-isolation
(cd uttt_llm && python -m tests.smoke_configs --construct)
```

The last command of each block builds every shipped model on CPU without
touching data. It is the quickest way to catch a flash-attn wheel built against
the wrong torch, which is the usual first failure, before spending GPU time.

## Your first run

Both commands below train **uTTT-MoE**, the paper's globally shared pool. Stop
them whenever you like: they checkpoint as they go.

**Novel view synthesis**, four GPUs, no data required to start:

```bash
python -m uttt_nvs.data.make_toy_dataset --out /tmp/uttt_toy   # 16 scenes, 24 views each
python -m tools.check_manifest /tmp/uttt_toy/manifest.txt --check-images
# paste the two paths it prints into uttt_nvs/datasets.yaml
NPROC_PER_NODE=4 bash uttt_nvs/train/launch.sh \
    uttt_nvs/configs/ownership/obj/uttt_moe_e64a1.yaml -s exp_name first_run
```

The run writes to `experiments/first_run/`. The synthetic scenes carry no real
3D structure — they exist so the loader, model, optimizer and checkpointing can
be exercised before Objaverse or DL3DV are in place. Swap
`uttt_nvs/datasets.yaml` for real manifests when you have them; see
[`uttt_nvs/data/README.md`](uttt_nvs/data/README.md).

**Language modeling**, 124M at 32K context:

```bash
cd uttt_llm
python -m flame.utils.preprocess \
    --dataset togethercomputer/Long-Data-Collections \
    --path /path/to/arrow-shards          # any HF text dataset works for a trial
# point uttt_llm/datasets.yaml at those shards
NPROC_PER_NODE=8 bash launch/train.sh \
    configs/exp/train_124M_32k.toml configs/main/124M/uttt_moe.json first_run
cd ..
```

On fewer than eight GPUs, add `--training.gradient_accumulation_steps` to keep
the global batch at 32; the trainer refuses to start otherwise.

## Then pick a domain

| | Data, training and evaluation |
|---|---|
| Novel view synthesis | [`uttt_nvs/README.md`](uttt_nvs/README.md) — manifests, the 64 configs, PSNR/SSIM/LPIPS with bootstrap intervals |
| Language modeling | [`uttt_llm/README.md`](uttt_llm/README.md) — arrow shards, the 57 configs, per-token loss and RULER |

## Project page

[`docs/index.html`](docs/index.html) — open locally, or serve with GitHub
Pages.

## Acknowledgements

See the [detailed documentation](docs/README.md) for architecture, configuration
semantics, data contracts, training/resume, evaluation and troubleshooting.


This code builds on [LaCT](https://tianyuanzhang.com/projects/ttt-done-right/),
[flame](https://github.com/fla-org/flame) and
[flash-linear-attention](https://github.com/fla-org/flash-linear-attention)
from fla-org, and [torchtitan](https://github.com/pytorch/torchtitan).

## License

MIT. See [LICENSE](LICENSE).
