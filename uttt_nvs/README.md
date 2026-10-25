# uTTT — Novel View Synthesis

New to the release? See the [bounded first run](../docs/getting-started.md),
[complete configuration catalog](../docs/configuration-catalog.md),
[architecture map](../docs/architecture.md) and [checkpoint guide](../docs/training.md).

Fast-weight memory for novel view synthesis, comparing three **ownership regimes**
for the fast-weight bank: layer-private (TTT-Dense), layer-local pools (TTT-MoE), and a
single globally shared pool (uTTT-MoE).

Novel view synthesis makes the read/write structure explicit: posed views are
written into the fast-weight bank and pose-only cameras query it, so rendering
quality depends directly on what the shared memory retained. It also allows a
control the language-modeling side does not — the sharing domain can be
partitioned across depth at fixed capacity and fixed active width, restricting
*which* layers reach *which* experts while changing nothing else.

---

## The ownership ladder is a few parameters, not several implementations

This is the most useful thing to know about this codebase. The method axes the
paper studies are **parameters on one class**, not separate models:

| Paper model | File | Parameters |
|---|---|---|
| TTT-Dense (layer-private, uTTT-Dense-w1-p8) | `models/ttt_dense_block.py` | — |
| Transformer (full-attention reference) | `models/transformer_block.py` | — |
| DeltaNet / Gated DeltaNet (linear-attention references) | `models/deltanet_block.py`, `models/gated_deltanet_block.py` | — |
| uTTT-Dense w1/w2/w4/w8 (shared, p=1; `r` in the config names) | `models/uttt_dense_block.py` | `fw_inter_multi: 1,2,4,8` |
| uTTT-Dense, depth-partitioned p=2/4 | `models/uttt_dense_partitioned.py` | — |
| uTTT-MoE, depth-partitioned p=2/4 | `models/uttt_moe_partitioned.py` | — |
| **TTT-MoE (layer-local pool)** | **`models/uttt_moe_block.py`** | `pool_mode: per_block`, `router_share_mode: per_layer` |
| **uTTT-MoE (global pool)** | **same file** | `pool_mode: shared`, `router_share_mode: share_all`, `aggregate_write: true` |
| **Write schedule (aggregated / chained)** | **same file** | `aggregate_write: true / false` |

The references live in `configs/reference/{obj,dl3dv}/`, the partitioned dense
rows in `configs/topology/{obj,dl3dv}/uttt_dense_w*_p*.yaml`. The references and
the `w4_p2` / `w2_p4` files are the paper's run configs; `w1_p2` / `w1_p4` are
those files with `fw_inter_multi: 1`, as their headers say.

Moving from the paper's TTT-MoE to its uTTT-MoE — the central comparison —
changes **three** keys of the same block, all set explicitly in the shipped
configs:

| Key | TTT-MoE (`ownership/*/ttt_moe_e8a1.yaml`) | uTTT-MoE (`ownership/*/uttt_moe_e64a1.yaml`) | Meaning |
|---|---|---|---|
| `pool_mode` | `per_block` | `shared` | who owns the expert pool: each layer its own, or one pool for the whole stack |
| `router_share_mode` | `per_layer` | `share_all` | one router per layer, or a single router whose parameters every layer reuses |
| `aggregate_write` | absent (default `false`) | `true` | each layer writes its own pool in turn, or every layer's raw gradient is accumulated against the same pre-write state and applied once per chunk boundary |

`router_share_mode` is independent of ownership: a shared pool with
`per_layer` routers is a valid model (the LLM uTTT-MoE uses exactly that), and
the module's self-test constructs all four combinations. The paper's NVS
uTTT-MoE shares the router across layers; the paper's LLM uTTT-MoE keeps a
per-layer top-1 router over the shared pool.

`aggregate_write` defaults to **`false`** in the code. It is only defined for a
shared bank, and the code enforces the pairing:

```python
if aggregate_write and pool_mode != "shared":
    raise ValueError("aggregate_write=True is only supported with pool_mode='shared'.")
```

Every shipped `pool_mode: shared` config outside `configs/scale/` sets
`aggregate_write: true` explicitly (that is the paper's aggregated schedule;
`configs/schedule/*/uttt_moe_e64a1_chained.yaml` sets it to `false` for the
chained comparison). The `configs/scale/` files leave it at the default, i.e.
chained writes. Aggregated writes exist so every layer's contribution
accumulates on the same tensors and is applied once at the chunk boundary,
which is what makes grouped execution possible.

---

## Quick start

### 1. Install

```bash
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r uttt_nvs/requirements.txt
pip install flash-attn --no-build-isolation
```

Reference environment, verified working:
`Python 3.10 · CUDA 12.8 · torch 2.8.0+cu128 · torchvision 0.23.0 · triton 3.4.0`

torch is pinned to that verified build on purpose: an unpinned `pip install
torch` pulls the newest release, which silently breaks the flash-attn ABI. If
your CUDA toolkit needs a different build, change the pin deliberately and
expect to rebuild flash-attn against it.

`triton` ships with the official PyTorch CUDA wheels and is required by the fused
MoE kernels in `models/kernels/`.

**flash-attn installs on its own line** because it needs torch present at build
time, which pip's build isolation prevents — hence `--no-build-isolation`, and
hence torch first. The wheel must match your torch build; a mismatched one fails
at import with `undefined symbol: _ZN3c104cuda...` rather than a version
message, which means rebuild flash-attn rather than upgrade torch.

It backs `models/transformer_block.py`, which is the Transformer row of the
paper's tables. If you only need the fast-weight models you can skip it: every
config except two constructs and runs a forward pass with `flash_attn`
unimportable — the two that do not are `configs/transformer/{obj,dl3dv}/transformer.yaml`.

**What the Transformer reference actually caches.** `transformer_block.py` is
not a standard per-layer KV cache. With the shipped
`use_separate_last_to_kv: false` (code: `to_kv_mode="shared_last"`), each
layer's cache holds keys and values computed from *that layer's* own
normalized input of every earlier posed view, but projected with the **last
block's** `to_kv` weights at every write; only the current view's keys and
values inside a layer's attention come from that layer's own projection. The
alternative modes (`separate`: one extra `to_kv` shared by all caches;
`per_layer`: each layer's own `to_kv`) are in the code but not in any shipped
config. Keep this in mind before treating the row as a generic LVSM baseline.

### 2. Fill in two files

This repository ships code, not data or credentials. Fill each in **once** — not
per experiment. Both copies are git-ignored.

```bash
cp uttt_nvs/api_keys.example.yaml uttt_nvs/api_keys.yaml
cp uttt_nvs/datasets.example.yaml uttt_nvs/datasets.yaml
```

`api_keys.yaml` — your Weights & Biases key. Exporting `WANDB_API_KEY` works too
and takes precedence.

```yaml
wandb: "your-key-from-https://wandb.ai/authorize"
```

`datasets.yaml` — where your data lives. Every config carries
`training.dataset: obj` or `dl3dv`, which selects the block to read, so this one
file wires up every config.

```yaml
obj:
  train: /path/to/objaverse_train_manifest.txt
  eval:  /path/to/gso_test_manifest.txt
dl3dv:
  train: /path/to/dl3dv_train_manifest.txt
  eval:  /path/to/dl3dv_benchmark_manifest.txt
```

A *manifest* is a text file with one camera-JSON path per line; entries may be
local paths, `s3://` objects or `gs://` blobs, and may be mixed within one
manifest (the loader opens each entry by its own prefix and builds S3/GCS
clients lazily in whichever worker reads it).
The manifest file itself may also be local or remote. The CPU manifest checker
validates local entries only. See [`data/README.md`](data/README.md) for the
format, the `build_manifest.py` generator, and where to obtain Objaverse,
Google Scanned Objects and DL3DV.

### 3. Train

```bash
NPROC_PER_NODE=4 bash uttt_nvs/train/launch.sh \
    uttt_nvs/configs/ownership/obj/uttt_moe_e64a1.yaml \
    -s exp_name my_run
```

Each run writes its checkpoints, sample renders and W&B files to
`experiments/<exp_name>/` inside the repository. Point that elsewhere with
`-s training.out_dir /path/to/runs` or by exporting `UTTT_NVS_OUT_DIR`.
Weights & Biases is optional: with no `WANDB_API_KEY` and no `api_keys.yaml`
the run trains with logging disabled.

> **Running inside a cluster job?** Schedulers often export `MASTER_PORT`,
> `MASTER_ADDR`, `WORLD_SIZE` and `RANK` into every shell. The launcher treats
> a set `MASTER_PORT` as "this is a multi-node launch", so leftover values
> from the scheduler make a single-node run fail its divisibility check. Clear
> them first:
> `env -u MASTER_PORT -u MASTER_ADDR -u WORLD_SIZE -u RANK -u NODE_RANK NPROC_PER_NODE=4 bash uttt_nvs/train/launch.sh ...`

**The published experiments use a total batch of 128 — 32 per GPU across 4
GPUs.** The GPU count comes from your launcher and is invisible to the config, so
each config records the total in `total_batch_size` and the trainer refuses to
start on a mismatch. To run on a different number of GPUs, see
[Training](#training).

---

## Repository layout

```
uttt_nvs/
├── models/          model definitions
│   └── kernels/     fused Triton grouped-GEMM / SwiGLU kernels
├── data/            manifest-based loader (local / S3 / GCS)  → data/README.md
├── configs/         experiment configs, grouped by paper section
├── train/           trainer, optimizer, checkpointing, launcher
├── eval/            full-test evaluation with bootstrap CIs   → eval/README.md
├── examples/        Slurm launch templates                    → examples/README.md
└── tests/           config resolution and construction check
```

---

## Experiments

`configs/` is organised by the paper's experiment axes. Axes overlap — a run that
serves as both a capacity point and a width point appears in both directories —
so the first release's **64 config files covered 52 distinct runs**, across
`obj` = Objaverse/GSO and `dl3dv` = DL3DV; this release adds the two
linear-attention references and the depth-partitioned uTTT-Dense runs of the
paper's full tables.

| Directory | Question | Files per dataset |
|---|---|---:|
| `configs/ownership/` | private vs layer-local vs globally shared | 4 |
| `configs/capacity/` | stored experts 8→64 at fixed top-1 access | 4 |
| `configs/width/` | active experts per head, a ∈ {1,2,4,8} | 8 |
| `configs/topology/` | sharing domain partitioned into p ∈ {1,2,4} groups: MoE, plus uTTT-Dense `w{4,2,1}_p{2,4}` | 10 |
| `configs/dense/` | shared dense state, width w ∈ {1,2,4,8} (`r1`–`r8` in the file names), no router | 4 |
| `configs/schedule/` | aggregated vs chained write | 2 |
| `configs/transformer/` | full-attention reference | 1 |
| `configs/reference/` | DeltaNet and Gated DeltaNet references | 2 |
| `configs/scale/` | 242M v3 model, three-stage resolution ladder (Objaverse only) | 3 |

Every config is self-contained: nothing is inherited or merged at load time, so
what you read in a file is what runs. All runs use `v_gap: null`.

### Run naming

Filename and directory together give the run identifier:
`{axis}/{dataset}/{name}.yaml` sets `exp_name: {name}_{dataset}`.

```
ownership/obj/ttt_dense.yaml              → ttt_dense_obj
capacity/dl3dv/uttt_moe_e64a1.yaml        → uttt_moe_e64a1_dl3dv
dense/obj/uttt_dense_r8.yaml              → uttt_dense_r8_obj
topology/obj/uttt_moe_p2_e64.yaml         → uttt_moe_p2_e64_obj
schedule/obj/uttt_moe_e64a1_chained.yaml  → uttt_moe_e64a1_chained_obj
```

Configs describing the **same run** under different axes carry the same name, so
the same file appears in several directories. `uttt_moe_e64a1.yaml` is in
`ownership/`, `capacity/`, `width/`, `topology/` and `schedule/` — it is the
global pool at top-1, which is simultaneously the ownership ladder's top rung, a
capacity point, a width point, the `P=1` topology, and the aggregated-write
baseline. That is why the original 64 files map onto 52 runs.

Two axes therefore have no separate `p1` or `aggregated` file: `p=1` *is* the
plain global pool, and every shared-pool config already sets
`aggregate_write: true` (the code default is `false`; only the `_chained`
schedule files and `configs/scale/` run without it).

### Are the two datasets' configs identical?

Almost. An `obj`/`dl3dv` pair differs only in the dataset it reads — same
`image_size`, `patch_size`, `dim`, `layers`, `batch_size_per_gpu`, `num_views`,
`lr` and step count. Since the dataset path comes from `datasets.yaml`, the two
files are byte-identical apart from `exp_name` and the `dataset:` key. This is
expected: the paper holds the architecture fixed across domains on purpose.

### Verify every config

Configs name their model classes as fully qualified import strings resolved at
runtime, so a wrong path surfaces only when the model is built. This check
catches that up front, and needs no data, no GPU and no credentials:

```bash
python -m uttt_nvs.tests.smoke_configs              # resolve class paths
python -m uttt_nvs.tests.smoke_configs --construct  # also build each model on CPU
```

---

## Training

### Match the GPU count to the batch size

The effective batch size is `batch_size_per_gpu × world_size`, and the world size
comes from your launcher. Running the same config on a different number of GPUs
would silently change the effective batch size while the learning rate stays put,
so each config records what the experiment is defined at:

```yaml
total_batch_size: 128
batch_size_per_gpu: 32
```

Pick the line that matches your machine:

```bash
# 4 GPUs, as published
NPROC_PER_NODE=4 bash uttt_nvs/train/launch.sh <config> -s exp_name my_run

# 8 GPUs: halve the per-GPU batch
NPROC_PER_NODE=8 bash uttt_nvs/train/launch.sh <config> -s training.batch_size_per_gpu 16

# 2 GPUs: double it
NPROC_PER_NODE=2 bash uttt_nvs/train/launch.sh <config> -s training.batch_size_per_gpu 64

# override any config value at runtime with -s key value
NPROC_PER_NODE=4 bash uttt_nvs/train/launch.sh <config> -s model.layers 8 -s training.lr 4e-4
```

At the published total of 128:

| GPUs | `batch_size_per_gpu` |
|---:|---:|
| 2 | 64 |
| **4** | **32 (as shipped)** |
| 8 | 16 |
| 16 | 8 |
| 32 | 4 |

Getting it wrong is not silent:

```
ValueError: Effective total batch size is 256 (32 per GPU x 8 GPUs x 1 grad-accum
steps), but this experiment is defined at 128.
Launch on 4 GPUs, or scale `batch_size_per_gpu` so the product is 128, or change
`total_batch_size` to run a deliberately different experiment.
```

[`examples/`](examples/) has Slurm templates. For multi-node, export the standard
PyTorch distributed variables (`MASTER_ADDR`, `MASTER_PORT`, `WORLD_SIZE`,
`RANK`) — most schedulers do this for you — and use a shared `JOB_UUID`.

### Reported training setup

| | Object-level | Scene-level |
|---|---|---|
| GPUs | 4 | 4 |
| Batch per GPU | 32 | 32 |
| Total batch | 128 | 128 |
| `max_fwdbwd_passes` (config) | 20,000 | 20,000 |
| Actual steps (rounded up to whole epochs) | 24,476 (4 × 6,119) | 20,046 (6 × 3,341) |
| Resolution | 256×256 | 256×256 |
| Precision | bf16 | bf16 |
| Peak LR | 4e-4, cosine, 2500 warmup | same |

The published runs stopped at **24,476** (Objaverse/GSO) and **20,046**
(DL3DV) forward/backward passes, not 20,000: `round_max_fwdbwd_passes_to_epoch`
defaults to **true**, and the trainer computes
`epochs = floor(max_fwdbwd_passes / passes_per_epoch) + 1`, so a 20,000-pass
budget becomes four epochs of 6,119 passes on Objaverse and six epochs of 3,341
on DL3DV. With `grad_accum_steps: 1` a pass is an optimizer step; the paper's
tables and the evaluation registry use those final steps (the Transformer
reference was evaluated at 24,000 / 20,000).

### Running only a few steps

`round_max_fwdbwd_passes_to_epoch` defaults to **true**, which rounds the step
budget *up to a whole epoch*. Setting `max_fwdbwd_passes: 5` on its own therefore
trains for one full epoch, not five steps. Disable the rounding too:

```bash
NPROC_PER_NODE=4 bash uttt_nvs/train/launch.sh <config> \
    -s training.max_fwdbwd_passes 5 \
    -s training.round_max_fwdbwd_passes_to_epoch False
```

### The first step is much slower than the rest

The Triton kernels in `models/kernels/` autotune on first use, so step 1 is not
steady-state throughput.

---

## Larger models

`configs/scale/obj/` is the paper's large uTTT-MoE: 242M parameters (width 768,
24 layers), 96 shared experts with top-1 routing, fast-weight head dim 192 and
inner width ×2, aggregated writes, trained as a **three-stage
progressive-resolution ladder** on Objaverse. Each stage fine-tunes the previous
one at a higher resolution, so the stages are a chain, not independent runs. The
files are the paper's run configs with only paths rewritten; the DL3DV ladder
is not released.

| Stage | Config | Resolution | Total batch | Steps | LR | Starts from |
|---|---|---:|---:|---:|---|---|
| 1 | `scale/obj/uttt_moe_large_256.yaml` | 256² | 512 | 81,038 | 4e-4 | scratch |
| 2 | `scale/obj/uttt_moe_large_512.yaml` | 512² | 128 | 70,000 | 1e-5 | stage 1 |
| 3 | `scale/obj/uttt_moe_large_1024.yaml` | 1024² | 64 | 4,500 | 1e-5 | stage 2 |

Same model code as the small runs — `models/uttt_moe_block.py` with
`pool_mode: shared` — plus three things the small configs never set:

- `apply_ckpt_group: 6` and `offload_master_states: true` are memory switches of
  the block (coarser activation-checkpoint regions over the aggregated update,
  fp32 master states parked in pinned host RAM). Both default off and leave the
  computation unchanged.
- `perceptual_loss_weight: 0.5` with `l2_warmup_steps: 4000`: the loss is pure
  L2 (+ load-balancing) for the first 4,000 global steps, then adds a VGG-19
  perceptual term. `models/perceptual_loss.py` loads the MatConvNet weights;
  download `imagenet-vgg-verydeep-19.mat` from
  <https://www.vlfeat.org/matconvnet/models/> (md5
  `106118b7cf60435e6d8e04f6a6dc3657`) and point `VGG19_MAT_PATH` at it or place
  it next to the module. Missing weights fail at start-up, not silently.
- `batch_size_per_gpu` assumes 64 ranks. The invariant is the total batch in the
  table; the launcher's batch-size check stops any other shape.

### Stage 1: train from scratch

```bash
# Published topology: 64 GPUs (total batch 512 at 8 per GPU). Run multi-node
# with the distributed variables set as in examples/; the batch-size check
# stops any other shape.
VGG19_MAT_PATH=/path/to/imagenet-vgg-verydeep-19.mat \
NPROC_PER_NODE=8 bash uttt_nvs/train/launch.sh uttt_nvs/configs/scale/obj/uttt_moe_large_256.yaml
```

### Stages 2 and 3: fine-tune from the previous stage

```bash
# Published topology: 64 GPUs (total batch 128 at 2 per GPU), multi-node as above.
NPROC_PER_NODE=8 bash uttt_nvs/train/launch.sh uttt_nvs/configs/scale/obj/uttt_moe_large_512.yaml \
    --load /path/to/stage1/checkpoint_dir \
    -s training.force_reset_training_state True
```

`--load` accepts a `.pt` file or a directory. Directory loads select the newest
readable checkpoint by numeric step and can fall back past corrupt files; an
explicit missing or entirely unreadable source now raises instead of silently
starting from scratch. Use an exact file for an unambiguous stage boundary.
`force_reset_training_state` keeps weights while resetting counters and optimizer
state. Existing output-directory checkpoints take precedence over `--load`, so
use a fresh `exp_name` when starting a new fine-tuning stage. See the
[checkpoint lifecycle guide](../docs/training.md) for atomic writes and the
file-count retention setting (three checkpoints by default).

**Stages 2 and 3 need stage 1's weights, which this repository does not ship.**
Train stage 1 yourself, or substitute your own checkpoint. The paper's runs of
this ladder did not finish either stage chain, so no reference numbers or demo
checkpoints exist for it.

### Turn off in-training evaluation at these scales

The evaluation split is sharded across ranks, and past a certain rank count some
ranks get no batch at all — those ranks never reach the collective and the job
hangs. GSO has 1,019 test scenes: at 64 ranks × 8 per rank the split no longer
divides evenly and some ranks come up empty.

The trainer detects this and stops with an explanation rather than hanging. Pass:

```bash
-s training.eval_every 0
```

and evaluate separately with `python -m uttt_nvs.eval.evaluate`, which shards
correctly for the full-test protocol.

---

## Evaluation

`eval/` implements the full-test protocol: every target view 1–23 scored under
teacher-forced sequential conditioning, averaged within a scene and then over
scenes, with 95% percentile bootstrap confidence intervals over
scenes (10,000 resamples, seed 9595).

```bash
python -m uttt_nvs.eval.evaluate --registry uttt_nvs/eval/experiments.yaml --experiment <id>
python -m uttt_nvs.eval.aggregate --registry uttt_nvs/eval/experiments.yaml
```

`eval/experiments.yaml` maps each experiment id to its paper model name, config
and checkpoint. Paths in it are placeholders (`/PATH/TO/...`); fill them in for
your setup. See [`eval/README.md`](eval/README.md) for the protocol details and
the paper-name-to-loader mapping.
