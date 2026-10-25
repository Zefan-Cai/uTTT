# uTTT — Language Modeling

For shared tooling, see the [getting-started guide](../docs/getting-started.md),
[complete configuration catalog](../docs/configuration-catalog.md),
[architecture map](../docs/architecture.md) and [training/resume guide](../docs/training.md).

Long-context language modeling at 32K tokens, at two scales (124M and 760M),
comparing three **ownership regimes** for the fast-weight bank — layer-private
(LaCT / TTT-Dense), layer-local pools (TTT-MoE), and a single globally shared
pool (uTTT-MoE) — against full-attention, sliding-window, linear-attention and
state-space baselines. Every experiment is defined at a global batch of 32
sequences.

---

## The ownership ladder, as shipped

In the NVS half of this repository the ownership ladder is a configuration
switch on a single implementation. Here it is the opposite: **each rung is its
own model package, shipped exactly as the paper's runs used it.** The rows were
trained on successive implementations, one package per row. Each package's
`model_type` string is the stable identifier of that implementation in this
release; no checkpoints are released, so the strings carry no compatibility
promise beyond the code in this repository.

The paper's main language-modeling table (`tab:lm-results`, eleven rows) maps
row by row onto one model package and one config file:

| Paper row (main table) | Model package (`flame/custom_models/`) | Config (`configs/main/{124M,760M}/`) |
|---|---|---|
| Transformer† (full attention) | fla built-in (external dependency) | `transformer.json` |
| Transformer-SWA | fla built-in (external dependency) | `transformer_swa.json` |
| Gated DeltaNet-SWA | `gated_deltanet_swa/` | `gated_deltanet_swa.json` |
| DeltaNet-SWA | `deltanet_swa/` | `deltanet_swa.json` |
| KDA-SWA | `kda_swa/` | `kda_swa.json` |
| Mamba-2-SWA | `mamba2_swa/` | `mamba2_swa.json` |
| MoM-SWA | `mom_swa/` | `mom_swa.json` |
| LaCT (published) | `lact/` | `lact.json` |
| TTT-Dense | `ttt_dense/` | `ttt_dense.json` |
| TTT-MoE (loss-free balancing, update rate 1e-5) | `ttt_moe_lb/` | `ttt_moe_lb.json` |
| uTTT-MoE | `uttt_moe/` | `uttt_moe.json` |

† The full-attention Transformer keeps a key–value cache that grows with
context and is excluded from the paper's bounded-state comparison.

Appendix rows (not in the main table):

| Paper row (appendix) | Model package | Config |
|---|---|---|
| TTT-MoE, no balancing (`tab:lm-ttt-moe`) | `ttt_moe_no_lb/` | `configs/main/{124M,760M}/ttt_moe_no_lb.json` |
| uTTT-Dense, r ∈ {1, 2, 4, 6} (`tab:lm-uttt-dense`, 124M only) | `uttt_dense/` | `configs/main/124M/uttt_dense_r{1,2,4,6}.json` |
| Balancing sweep, both routed families (`tab:lm-balancing`, `tab:lm-ttt-moe`, `tab:lm-alpha`) | `uttt_moe/`, `ttt_moe_lb/`, `ttt_moe_no_lb/` | `configs/balancing/` |

At runtime each config selects its implementation through a `model_type`
identifier registered in `flame/custom_models/__init__.py`; the identifier
matches the package name above. Two rows are exceptions, because their
identifiers belong to the upstream projects rather than to this release:
`lact_swiglu` from the LaCT release, and `transformer` from fla.

Notes the tables cannot show:

- **The paper's TTT-MoE row is the balanced one.** `ttt_moe_lb/` (loss-free
  balancing, update rate 1e-5 at both scales) is the TTT-MoE of the main table;
  `ttt_moe_no_lb/` is an earlier implementation of the same layer-local pool
  without balancing and appears only in the appendix balancing table.
- **uTTT-MoE's loss-free balancing update rate differs per scale** — 3e-5 at
  124M, 1e-3 at 760M — while TTT-MoE uses 1e-5 at both. This matches the
  paper's appendix.
- **uTTT-Dense** shares one dense fast-weight state across all layers, `r` times
  as wide as one TTT-Dense layer's (`uttt_dense_r`), with a full-rank learned
  initial state (`w0_w2_low_rank = 0`) and the aggregated chunk schedule
  (`uttt_update_mode = "aggregate"`). The paper reports it at 124M only.
- Every fast-weight package (`ttt_dense/`, `ttt_moe_no_lb/`, `ttt_moe_lb/`,
  `uttt_moe/`, `uttt_dense/`) returns the full-sequence hidden state from
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

| Key | Dataset | Source | Read by |
|---|---|---|---|
| `train.files` | Long-Data-Collections | `togethercomputer/Long-Data-Collections` on HF | `launch/train.sh` (`--training.data_files`) |
| `ptl.files` | Books3 from The Pile, 32K-token rows | not redistributed; withdrawn from its original host in 2023 | nothing — a note for yourself; PTL reads `training.data_files` in `configs/exp/eval_ptl.toml` |
| `ruler.cache` | RULER synthetic tasks | built by `eval/ruler/prewarm_ruler_cache.py` | nothing — RULER assets live in the HF datasets cache (`HF_HOME` / `HF_DATASETS_CACHE`) |

Only `train.files` is consumed by code. The `ptl` and `ruler` blocks are
bookkeeping for the evaluation paths you still have to type into the PTL TOML
and the RULER commands below.

After installation, run the remaining LLM commands from `uttt_llm/`:

```bash
cd uttt_llm
```

Shards are produced with `flame/utils/preprocess.py`, for example:

```bash
huggingface-cli login   # Long-Data-Collections is a gated dataset on the HF hub
python -m flame.utils.preprocess \
    --dataset togethercomputer/Long-Data-Collections \
    --data_dir pretrain \
    --path /path/to/long-data-collections-arrow
```

The paper used the `pretrain/` files of that dataset; `--data_dir` selects them
(the repository also ships a `fine-tune/` split that must not be mixed in).
Pass `--dataset_name <config>` only for datasets that define several
`BuilderConfig`s; the default is the dataset's single default config.

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
Loss-free balancing then still updates its bias once per optimizer step: the
trainer scales `memory_loss_free_update_interval` by the accumulation factor
and the model accumulates routing counts across the micro-batches.
Both training TOMLs enable Weights & Biases (`metrics.enable_wandb = true`); log in
with `wandb login` or export `WANDB_API_KEY` first, or set `WANDB_MODE=offline`.
Without credentials rank 0 blocks on an interactive login prompt in a terminal.
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
that is only resolved at construction time, so this is the first check to run
on a new environment: it catches a flash-attn wheel built against the wrong
torch and a registry that fails to import. It does not import torchtitan,
torchdata, `lm_eval` or `matplotlib`, so a broken training or evaluation stack
only shows up when `launch/train.sh`, `eval/ptl/run.sh` or the RULER scripts
start.

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
│   └── custom_models/    eleven model packages; two rows use fla built-ins
├── configs/
│   ├── main/             one config per paper row and scale (plus uttt_dense_r* at 124M)
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

**Main table.** One config per row and scale, named after the row (see the map
above). Training runs 10,240 steps at 124M and 40,960 at 760M, both at the
global batch of 32 sequences of 32,768 tokens (10.7B and 42.9B tokens).

**Balancing study.** `configs/balancing/` holds the paper's balancing sweep —
mechanism (auxiliary loss vs loss-free), scope (per-layer vs pool), and
bias-update rate, at both scales. The 39 configs reuse the main-table model
packages; there is no extra code.

### Not part of this release

- **Throughput and efficiency benchmarks** (the paper's measured-cost section
  and kernel ablations): no benchmark scripts ship.
- **Mechanism experiments.** The Cut backward mode ships as a model option
  (`memory_update_backward_mode = "cross_layer_cut"` in `uttt_moe/`), but the
  Cut-training analysis, the write-freezing / own-write knockouts, the
  Jacobian read-output probes and the reuse statistics are not included.
- **The length-extrapolation study** (64K / 128K / 256K evaluation).
- **The sequential-schedule modules** behind the write-schedule comparison.
- **Checkpoints or weights**, and **datasets** (see "Data" above).

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

Unless the step directory already holds a rank-0 `model.pt`, the converter
first rewrites the *entire* DCP step — model weights **and** optimizer state —
into one `checkpoint.pt` inside a `tempfile.TemporaryDirectory()`
(`dcp_to_torch_save`) and then `torch.load`s it on the CPU to pick out the
`model` entry. Budget scratch space on `TMPDIR` for the whole
`checkpoint/step-<N>/` directory and host RAM of the same order; a small `/tmp`
is the usual cause of a conversion that dies with "No space left on device".
Point it elsewhere with `TMPDIR=/scratch python -m flame.utils.convert_dcp_to_hf ...`.

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

The paper's PTL is the NaN-aware mean over positions ≥ 30,000 (0-based) of
the 32,768-position curve. `eval/ptl/summarize_ptl.py` computes it from the
JSON the evaluator writes (no torch needed):

```bash
python eval/ptl/summarize_ptl.py eval_out/default_loss_per_token_position.json
```

The evaluator writes `<job.dump_folder>/<training.save_prefix>_loss_per_token_position.json`;
with the shipped `eval_ptl.toml` that is `uttt_llm/eval_out/default_loss_per_token_position.json`.
Pass `--job.dump_folder runs/my_run/ptl --training.save_prefix step10240` after the
checkpoint directory to put it next to the run instead.

### RULER

RULER runs on the HF-format directory on a single GPU. Cache the synthetic-task
assets once, then evaluate:

```bash
# Task assets land in the HF datasets cache (HF_HOME / HF_DATASETS_CACHE);
# --output is the path of the manifest JSON describing what was cached.
python eval/ruler/prewarm_ruler_cache.py --model-dir runs/my_run \
    --output ruler_cache_manifest.json --max-length 32768 --trust-remote-code

python eval/ruler/run_ruler_eval.py --model-dir runs/my_run \
    --output results.json --max-length 32768 --dtype bfloat16 --trust-remote-code
```

The paper's RULER score is the mean over 13 tasks at each of 4K, 8K, 16K and
32K, then the mean of the four lengths: run the command once per
`--max-length` in {4096, 8192, 16384, 32768} and average the four `results.json`
files yourself (see [docs/evaluation.md](../docs/evaluation.md)).

`--use-cache` / `--no-use-cache` forces the generation KV/fast-weight cache on
or off inside lm-eval's `generate` calls (default: the model config's
`use_cache`). The paper's 124M RULER numbers were produced with the cache
disabled (`--no-use-cache`); the cached path replays fragmented chunks so both
modes agree, but state the mode you used when reporting.

Sharded runs are combined with
`merge_ruler_shards.py --shards-dir <dir> --output-dir <dir> --expected-shards <n>`;
it refuses to merge when fewer than `--expected-shards` result files are present.

RULER needs `lm_eval` (lm-evaluation-harness >= 0.4.9), listed in `requirements.txt`.
