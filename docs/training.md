# Training, distribution and checkpoints

## Launch contracts

Both training launchers accept `--help` and a leading `--dry-run`. Configuration
paths can be absolute, relative to the caller's directory, or relative to the
domain (LLM) / repository and domain (NVS). An existing caller-relative file wins.
Paths are resolved before the script changes directory. Shell arguments are
passed as arrays, so spaces are preserved and extra options are not evaluated
as shell code.

NVS runs from the repository root. LLM runs from `uttt_llm/`. This still matters
for dataset and checkpoint paths embedded inside configs or forwarded trainer
arguments: only the launcher's config-file arguments are normalized.

### Single node

```bash
NPROC_PER_NODE=4 bash uttt_nvs/train/launch.sh --dry-run \
  uttt_nvs/configs/ownership/obj/uttt_moe_e64a1.yaml

NPROC_PER_NODE=8 bash uttt_llm/launch/train.sh --dry-run \
  uttt_llm/configs/exp/train_124M_32k.toml \
  uttt_llm/configs/main/124M/uttt_moe.json my_run
```

Standalone mode uses torchrun's `--standalone` rendezvous rather than NVS's old
fixed port, avoiding collisions between independent local jobs. If a scheduler
left distributed variables in your shell, remove all of them for standalone:

```bash
env -u MASTER_ADDR -u MASTER_PORT -u WORLD_SIZE -u RANK -u NODE_RANK \
  NPROC_PER_NODE=4 bash uttt_nvs/train/launch.sh --dry-run \
  uttt_nvs/configs/ownership/obj/uttt_moe_e64a1.yaml
```

### Multiple nodes

| Variable | Contract |
|---|---|
| `NPROC_PER_NODE` | Positive number of local worker processes / GPUs |
| `WORLD_SIZE` | Total worker processes across all nodes, not node count |
| `MASTER_ADDR` | Reachable rendezvous host, identical on every node |
| `MASTER_PORT` | Port 1–65535; setting it selects distributed rendezvous |
| `NODE_RANK` | Zero-based node index; preferred over `RANK` when both exist |
| `RANK` | Legacy node-index fallback in these launchers, not a worker rank |
| `JOB_UUID` | Shared rendezvous identifier; choose a unique value per job |
| `NODE_SYNC` | Optional NVS-only prelaunch Python script; not run in dry mode |

For two nodes with eight GPUs each, set `WORLD_SIZE=16`, not 2. Run this on
both nodes with `NODE_RANK=0` or `1` respectively, replacing the master hostname:

```bash
MASTER_ADDR=training-node-0 MASTER_PORT=29500 \
WORLD_SIZE=16 NPROC_PER_NODE=8 NODE_RANK=0 JOB_UUID=uttt-760m-run-001 \
bash uttt_llm/launch/train.sh --dry-run \
  uttt_llm/configs/exp/train_760M_32k.toml \
  uttt_llm/configs/main/760M/uttt_moe.json run_760m
```

Remove `--dry-run` only after every node has matching code/configs, compatible
CUDA extensions, accessible data and the same rendezvous identity. Invalid
counts, partial environments and out-of-range node indices fail before torchrun.
Torchrun sets worker-level `RANK`/`LOCAL_RANK` for the training process; do not
confuse those with the launcher's legacy node-index input.

## Batch arithmetic

| Recipe | Per-device batch | Data-parallel workers | Accumulation | Global batch |
|---|---:|---:|---:|---:|
| Standard NVS | 32 | 4 | 1 | 128 |
| NVS, deliberately smaller per-device batch | 8 | 4 | 4 | 128 |
| LLM 124M published topology | 4 | 8 | 1 | 32 |
| LLM 124M on four workers | 4 | 4 | 2 | 32 |
| LLM 760M published topology | 2 | 16 | 1 | 32 |

Maintaining global batch does not automatically preserve a schedule: NVS's
limit is in forward/backward passes, so changing accumulation changes the
number of optimizer updates for the same pass count. Adjust deliberately and
report the changed recipe. Memory use also depends on context length,
resolution, model size, active experts and activation checkpointing.

## Exact smoke-run stopping

NVS rounds its pass limit to a full epoch by default. Requesting five passes
alone does not guarantee five passes:

```bash
NPROC_PER_NODE=4 bash uttt_nvs/train/launch.sh \
  uttt_nvs/configs/ownership/obj/uttt_moe_e64a1.yaml \
  -s exp_name bounded_smoke \
  -s training.max_fwdbwd_passes 5 \
  -s training.round_max_fwdbwd_passes_to_epoch False \
  -s training.wandb_offline True
```

For large distributed NVS runs with too few eval samples per rank, disable
in-training evaluation using `-s training.eval_every 0` and use the separate
full-test evaluator. An evaluation manifest is still loaded during trainer
initialization; disabling evaluation does not remove that startup requirement.

## NVS checkpoint lifecycle

1. Rank zero collects model, optimizer, scheduler and both step counters.
2. It saves to a temporary file on the output filesystem, flushes/fsyncs it,
   then atomically replaces the final `ckpt_<step>.pt` filename.
3. Discovery ignores temporary files, directories and malformed checkpoint
   names and sorts numeric steps, including unpadded legacy filenames.
4. Retention keeps the newest `training.save_last_n_ckpts` **files** at or
   before the current step. The default is three. Future-step files are not
   removed by a run resumed at an earlier step.

The retention parameter previously behaved like a step-distance threshold.
If you explicitly used a value such as 1001, review it: it now means 1001 files,
not roughly two checkpoint intervals. This change is listed in the changelog.
Atomic publication prevents ordinary interrupted writes from appearing as
complete checkpoints; filesystem/hardware durability still depends on storage.

### Restore precedence and failure behavior

The trainer selects exactly one source in this order:

1. Existing checkpoints in `experiments/<exp_name>/` (automatic resume).
2. Explicit `--load /absolute/path/to/checkpoint.pt` or a checkpoint directory.
3. `training.load` from the configuration.
4. No source: initialize a fresh run.

Within a directory, the newest deserializable checkpoint is used; corrupt newer
files are reported and older files are tried. An explicit missing path, empty
checkpoint directory, or directory with no readable checkpoint raises instead
of silently training from scratch. A readable but incompatible state dict is
not a license to ignore model-load warnings: use the matching architecture.
Only load checkpoints you trust.

### Fine-tuning a resolution stage

Use a fresh experiment name and the preceding stage's checkpoint. For example:

```bash
NPROC_PER_NODE=8 bash uttt_nvs/train/launch.sh \
  uttt_nvs/configs/scale/obj/uttt_moe_large_512.yaml \
  --load /data/stage1/ckpt_0000000000081038.pt \
  -s exp_name obj_stage2 \
  -s training.force_reset_training_state True \
  -s training.eval_every 0
```

This stage's released batch topology requires 64 workers total; use the
multi-node environment on eight 8-GPU nodes. The path is illustrative: use the
actual completed checkpoint step, which can differ under epoch rounding.

Reset loads model weights but resets optimizer/scheduler/counters unless
`load_optimizer_state_when_reset_training_state` is explicitly enabled.
Zeroed counters no longer cause the trainer to fall through and load a second
source. Reusing an output directory still gives automatic resume precedence
over `--load`; use a fresh name when switching source checkpoints.

## LLM checkpoints and experiment records

LLM training uses the flame/torchtitan distributed-checkpoint path, not the NVS
`ckpt_*.pt` format. The released training TOMLs use `load_step=-1` to resume
the latest checkpoint. Convert a completed DCP step to HF format for PTL/RULER
using the [LLM evaluation instructions](../uttt_llm/README.md).

Record the Git SHA, architecture config, job config, all overrides, data/tokenizer
revision, node/GPU topology, environment versions, random seed, checkpoint step
and completion status. Never put API keys into configs, command lines, commits
or bug reports. W&B offline mode is suitable for smoke tests; it is not proof
that remote experiment tracking is configured.
