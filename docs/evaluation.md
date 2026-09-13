# Evaluation and reproducibility

The release contains evaluation code, not ready-to-run checkpoints or completed
result registries. A model constructed on CPU or trained briefly on synthetic
data has not reproduced a paper result. Preserve the original figures and do
not replace them with extrapolated or invented scores.

## NVS full-test protocol

The detailed executable protocol lives in the [NVS evaluation guide](../uttt_nvs/eval/README.md)
and [experiment registry](../uttt_nvs/eval/experiments.yaml).

1. Prepare the correct held-out dataset and a frozen 24-view manifest.
2. Use the same sampled scene/view ordering for every compared model. The
   sampled indices are intentionally not sorted.
3. Evaluate targets 1–23 with **teacher-forced sequential conditioning**:
   ground-truth conditioning views accumulate across the sequence.
4. Save per-scene, per-view PSNR, SSIM and LPIPS records before aggregation.
5. Average within each scene, then across scenes for the overall metric.
6. Report scene-bootstrap percentile 95% intervals using 10,000 resamples,
   seed 9595, as implemented by the statistics module.

This is not the protocol “condition on View 0 once and independently render all
23 remaining views.” Mixing those protocols makes the resulting numbers
incomparable even if the metric names and image sizes match.

### Registry preparation

The registry's `/PATH/TO/...` placeholders must point to your actual trained
checkpoints, configs, manifests and output directories. A code-only static
config pass does not resolve these placeholders. Keep registry edits local or
publish a sanitized, portable version without credentials or private paths.

```bash
python -m uttt_nvs.eval.evaluate \
  --registry uttt_nvs/eval/experiments.yaml \
  --experiment gso_ttt_moe_e8a1 --max-scenes 2 --wandb-mode disabled
```

Start with the two-scene smoke check; inspect renders, finite metrics and
provenance before removing `--max-scenes`. Then aggregate completed experiments:

```bash
python -m uttt_nvs.eval.aggregate --registry uttt_nvs/eval/experiments.yaml
```

### Result artifacts and their meaning

| Artifact | Purpose |
|---|---|
| `raw_metrics.csv`, `raw_metrics.parquet` | Dataset, model, scene_id, view_index, PSNR, SSIM, LPIPS |
| `overall_summary.json` | Scene-macro metrics, valid-scene counts and intervals |
| `per_view_summary.csv` | View 1–23 statistics, one row per view and metric |
| `run_metadata.json` | Checkpoint/config/code/view-manifest provenance |
| `completion.json` | Completion marker, not merely “process started” |

Average by scene, not by whichever scene happened to yield the most valid
views. Missing/non-finite values affect valid counts and should be disclosed.
Compare like-for-like scene sets; a change in evaluation coverage can change the
mean without any model improvement. Completed runs are skipped by default;
use `--overwrite` only for an intentional rerun.

## Language modeling: convert first

The evaluation scripts expect an HF-format model directory. From `uttt_llm/`:

```bash
python -m flame.utils.convert_dcp_to_hf \
  --path runs/my_run --step 10240 \
  --config configs/main/124M/uttt_moe.json \
  --tokenizer fla-hub/transformer-1.3B-100B
```

The exported HF files are written into the `--path` directory itself. Preserve
the training run, config and tokenizer identities. A 760M checkpoint needs its
matching architecture file and actual completed step; do not copy the 124M
example's numbers blindly.

### Per-token loss (PTL)

Fill the four deployment fields in `configs/exp/eval_ptl.toml`: model config,
Book-3 Arrow glob, evaluation step count and data-parallel replicate degree.
The released PTL launcher is standalone; replicate degree must match the
launched GPU count. Compute the evaluated row count explicitly and disclose
any remainder rather than silently treating a partial evaluation as complete.

```bash
NPROC_PER_NODE=8 bash eval/ptl/run.sh configs/exp/eval_ptl.toml runs/my_run
```

Keep `per_token_loss_from_hidden=true` and `lm_head_chunk_size=2048` to avoid
materializing a full batch × 32768 × vocabulary logits tensor at once. These
settings chunk the output projection, not the definition of the loss. The
last position has no next-token target and is NaN by design; aggregate with
NaN-aware means and retain per-position valid counts where available.

### RULER

```bash
python eval/ruler/prewarm_ruler_cache.py --model-dir runs/my_run \
  --output ruler_cache/ --max-length 32768 --trust-remote-code
python eval/ruler/run_ruler_eval.py --model-dir runs/my_run \
  --output results.json --max-length 32768 --dtype bfloat16 --trust-remote-code
```

RULER runs on one GPU in this entry point. Keep task definitions, cache assets,
maximum length, generation settings, dtype and seed consistent. The
`--trust-remote-code` option permits code from the selected model source; use
it only for sources you have reviewed and trust. Sharded runs can be combined
with `merge_ruler_shards.py`; check task/sample coverage before reporting totals.

## What to report with a new result

- Repository SHA, exact architecture/job config and all command-line overrides.
- Checkpoint source, training step and whether the run completed or resumed.
- Dataset/split revision, tokenizer and fixed-view manifest identity.
- Evaluation protocol, metric aggregation unit, sample/scene counts and seeds.
- Hardware/software stack and precision; warmup/autotuning policy for timing.
- Failed, skipped or missing examples and whether compared methods share coverage.

Engineering changes in this release add checks and improve lifecycle handling.
They do not supply new accuracy, throughput, VRAM or convergence measurements.
