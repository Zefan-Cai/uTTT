# uTTT — Language Modeling

For shared tooling, see the [getting-started guide](../docs/getting-started.md),
[complete configuration catalog](../docs/configuration-catalog.md),
[architecture map](../docs/architecture.md) and [training/resume guide](../docs/training.md).

Long-context language modeling at 32K tokens, at two scales (124M and 760M),
comparing three **ownership regimes** for the fast-weight bank — layer-private
(LaCT / TTT-Dense), layer-local pools (TTT-MoE), and a single globally shared
pool (uTTT-MoE) — against full-attention and linear-attention baselines. Every
experiment is defined at a global batch of 32 sequences.

---

## The ownership ladder, as shipped

In the NVS half of this repository the ownership ladder is a configuration
switch on a single implementation. Here it is the opposite: **each rung is its
own model package, shipped exactly as the paper's runs used it.** The rows were
trained on successive implementations, and each package's `model_type` string
keeps its original value so released checkpoints stay loadable.

Each row of the paper's language-modeling table maps to one model package and
one config file.

| Paper row | Model package (`flame/custom_models/`) | Config (`configs/main/{124M,760M}/`) |
|---|---|---|
| Transformer (full attention) | fla built-in (external dependency) | `transformer.json` |
| Transformer-SWA | fla built-in (external dependency) | `transformer_swa.json` |
| DeltaNet-SWA | `deltanet_swa/` | `deltanet_swa.json` |
| Gated DeltaNet-SWA | `gated_deltanet_swa/` | `gated_deltanet_swa.json` |
| LaCT | `lact/` | `lact.json` |
| TTT-Dense | `ttt_dense/` | `ttt_dense.json` |
| TTT-MoE w/o LB | `ttt_moe_no_lb/` | `ttt_moe_no_lb.json` |
| TTT-MoE w/ LB | `ttt_moe_lb/` | `ttt_moe_lb.json` |
| uTTT-MoE | `uttt_moe/` | `uttt_moe.json` |

At runtime each config selects its implementation through a `model_type`
identifier registered in `flame/custom_models/__init__.py`; the identifier
matches the package name above. Two rows are exceptions, because their
identifiers belong to the upstream projects rather than to this release:
`lact_swiglu` from the LaCT release, and `transformer` from fla.

Notes the table cannot show:

- **The two TTT-MoE rows ran on successive implementations** of the layer-local
  pool, one package per row.
- **uTTT-MoE's loss-free balancing update rate differs per scale** — 3e-5 at
  124M, 1e-3 at 760M — while TTT-MoE w/ LB uses 1e-5 at both. This matches the
  paper's appendix.
- `ttt_dense/` and `ttt_moe_no_lb/` return the full-sequence hidden state from
  `forward()` rather than per-chunk fragments, so the chunked per-token-loss
  evaluation below works on 40–80GB GPUs; logits are unchanged.

---

## Quick start

### 1. Install

```bash
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r uttt_llm/requirements.txt
pip install flash-attn --no-build-isolation
```

torch is pinned to the verified reference build on purpose: an unpinned
`pip install torch` pulls the newest release, which silently breaks the
flash-attn ABI and exceeds the torchtitan pin in `requirements.txt`. If your
CUDA toolkit needs a different build, change the pin deliberately and expect to
rebuild flash-attn against it.

flash-attn is **required**: every fast-weight model imports it at module scope.
The wheel must match your torch build; `undefined symbol: _ZN3c104cuda...` at
import means rebuild flash-attn rather than upgrade torch.

### 2. Data

```bash
cp uttt_llm/datasets.example.yaml uttt_llm/datasets.yaml   # git-ignored
```

Both datasets are consumed as pre-processed arrow shards — this repository
redistributes no data:

| Key | Dataset | Source |
|---|---|---|
| `train.files` | Long-Data-Collections | `togethercomputer/Long-Data-Collections` on HF |
| `ptl.files` | Book-3 from The Pile | the Pile's Books3 subset, 32K-token rows |
| `ruler.cache` | RULER synthetic tasks | built by `eval/ruler/prewarm_ruler_cache.py` |

After installation, run the remaining LLM commands from `uttt_llm/`:

```bash
cd uttt_llm
```

Shards are produced with `flame/utils/preprocess.py`, for example:

```bash
python -m flame.utils.preprocess \
    --dataset togethercomputer/Long-Data-Collections \
    --path /path/to/long-data-collections-arrow
```

The tokenizer is the public `fla-hub/transformer-1.3B-100B` (32K vocab), pulled
from the HF hub by name; `preprocess.py` uses it by default.

### 3. Train

```bash
NPROC_PER_NODE=8 bash launch/train.sh \
    configs/exp/train_124M_32k.toml configs/main/124M/uttt_moe.json my_run
```

> **Running inside a cluster job?** Schedulers (Slurm, Kubernetes, and similar)
> often export `MASTER_PORT`, `MASTER_ADDR`, `WORLD_SIZE` and `RANK` into every
> shell. The launcher treats a set `MASTER_PORT` as "this is a multi-node
> launch" and reads `WORLD_SIZE` as the **total process count**, so leftover
> single-node values make it exit with a divisibility error. For a single-node
> run inside such an environment, clear them first:
> `env -u MASTER_PORT -u MASTER_ADDR -u WORLD_SIZE -u RANK -u NODE_RANK NPROC_PER_NODE=8 bash launch/train.sh ...`

**The global batch of 32 is enforced.** The exp configs pin the per-device
batch (4 at 124M, 2 at 760M) and declare `expected_global_batch_size = 32`; the
trainer multiplies per-device batch × data-parallel degree × grad-accum and
refuses to start on a mismatch, because the GPU count comes from the launcher
and is invisible to the config. On fewer GPUs, make up the difference with
gradient accumulation — for example 4 GPUs × batch 4 × `--training.gradient_accumulation_steps 2`.
The published runs used 8 GPUs at 124M and 16 (two nodes) at 760M; multi-node
launches set the standard torchrun rendezvous variables as described in
`launch/train.sh`. Re-running the same command resumes from the latest
checkpoint (`checkpoint.load_step = -1`).

### 4. Verify the install

```bash
python -m tests.smoke_configs               # resolve every model_type
python -m tests.smoke_configs --construct   # also build each model on CPU
```

Every shipped config names its implementation through a `model_type` string
that is only resolved at construction time, so this is the check that catches a
broken environment before any GPU time is spent.

Use `bash launch/train.sh --dry-run <exp.toml> <model.json> [job_name]` to check
resolved paths and rendezvous arguments without starting workers. Extra trainer
options can follow the model JSON even when the optional job name is omitted.
`NODE_RANK` is the preferred multi-node index; legacy `RANK` is still accepted.
The [CPU validation suite](../docs/testing.md) is separate from model construction
and makes no claim about fused-kernel numerical correctness.

---

## Repository layout

```
uttt_llm/
├── flame/                training/eval framework (fork of fla-org's flame, on torchtitan)
│   └── custom_models/    seven model packages; two rows use fla built-ins
├── configs/
│   ├── main/             9 configs per scale, filename = paper row
│   ├── balancing/        the paper's balancing sweep (24 at 124M, 15 at 760M)
│   └── exp/              job TOMLs: training per scale + PTL evaluation
├── eval/
│   ├── ptl/              per-token loss on Book-3 (chunked lm_head path)
│   └── ruler/            RULER retrieval evaluation
├── launch/train.sh       generic torchrun launcher
├── datasets.example.yaml
└── tests/smoke_configs.py
```

---

## Experiments

**Table 1.** One config per row and scale, named after the row (see the map
above). Training runs 10,240 steps at 124M and 40,960 at 760M, both at the
global batch of 32.

**Balancing study.** `configs/balancing/` holds the paper's balancing sweep —
mechanism (auxiliary loss vs loss-free), scope (per-layer vs pool), and
bias-update rate, at both scales. The 39 configs reuse the Table-1 model
packages; there is no extra code.

Not part of this release: the throughput implementation comparison, the
sequential-schedule modules, uTTT-Dense, and the length-extrapolation study.

---

## Evaluation

### Convert a checkpoint to HF format

Both evaluations run on an HF-format model directory, produced from a training
run's DCP checkpoint:

```bash
python -m flame.utils.convert_dcp_to_hf \
    --path runs/my_run \
    --step 10240 \
    --config configs/main/124M/uttt_moe.json \
    --tokenizer fla-hub/transformer-1.3B-100B
```

The HF model is written into the `--path` directory itself, which can then be
passed wherever a model directory is expected.

### Per-token loss (PTL)

```bash
NPROC_PER_NODE=8 bash eval/ptl/run.sh configs/exp/eval_ptl.toml runs/my_run
```

Before running, fill in four values in `configs/exp/eval_ptl.toml`:

- `model.config` — the model config of the checkpoint being evaluated;
- `training.data_files` — the Book-3 arrow shards;
- `training.steps` — number of rows ÷ (batch × GPUs);
- `training.data_parallel_replicate_degree` — **must equal the GPU count you
  launch with**, or the evaluator stops at startup.

The toml also carries the two settings that make 32K evaluation fit on 40–80GB
GPUs:

```toml
[eval]
per_token_loss_from_hidden = true
lm_head_chunk_size = 2048
```

Without them the evaluator materialises the full `[batch, 32768, vocab]`
logits tensor in a single lm_head call and goes out of memory. The output is
mean loss by token position; the final position of each row has no next token
and is reported as NaN by design — aggregate with NaN-aware means.

### RULER

RULER runs on the HF-format directory on a single GPU. Cache the synthetic-task
assets once, then evaluate:

```bash
python eval/ruler/prewarm_ruler_cache.py --model-dir runs/my_run \
    --output ruler_cache/ --max-length 32768 --trust-remote-code

python eval/ruler/run_ruler_eval.py --model-dir runs/my_run \
    --output results.json --max-length 32768 --dtype bfloat16 --trust-remote-code
```

`merge_ruler_shards.py --shards-dir <dir>` combines sharded runs.
