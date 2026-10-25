# NVS full-test evaluation inventory

`experiments.yaml` is the source of truth for the metric rerun. Its first 34
entries pair the 17 configurations of the paper's metric rerun with both test
domains; 22 further entries cover the remaining rows of
the paper's full tables. All configurations have `v_gap: null`.

## What the registry covers, and what it does not

The paper's full tables (`tab:nvs-gso-full`, `tab:nvs-dl3dv-full`) report
28 rows per dataset. The registry now lists every one of them except the
Transformer rows, which live in the separate `experiments_transformer.yaml`
(two entries):

- the metric rerun's 17 configurations per dataset (ownership, capacity,
  width and dense ladders), whose checkpoints were the curated paper set;
- the depth-partitioned runs `uTTT-MoE-e{32,16}-p2`, `uTTT-MoE-e{16,8}-p4`
  and `uTTT-Dense-w{4,2,1}-p{2,4}`, the chained-write schedule and the DeltaNet /
  Gated DeltaNet references. These 22 entries follow the same layout, but no
  checkpoint inventory backs them: train the configuration, place the
  checkpoint under `checkpoint_root` at the path the entry names (or edit the
  path), then run the entry like any other.

## Fixed protocol

- Use 24 ordered views per scene with seed `9595`.
- View 0 is the initial conditioning view and is not scored.
- The repository's existing forward pass is teacher-forced sequential NVS:
  prediction for View `k` is conditioned on ground-truth Views `0..k-1`.
  Views 1–23 are scored for PSNR, SSIM, and LPIPS.
- GSO has 1,019 scenes and must yield 23,437 raw metric rows per model.
- DL3DV has 140 scenes and must yield 3,220 raw metric rows per model.
- Upload runs to your W&B entity/project (see `experiments.yaml`).
- Resolve `config` relative to the repository root and `checkpoint` relative
  to `defaults.checkpoint_root` in the registry (`/PATH/TO/CHECKPOINTS` in the
  shipped file).

The checkpoint filenames encode the actual saved steps: Object/GSO uses step
24,476 and DL3DV uses step 20,046. These are deliberately recorded instead of
the rounded 24,450/20,000 W&B summary values.

## Local test data

Choose two local data roots on the evaluation node, for example:

- GSO: `/path/to/gso_renderings`
- DL3DV: `/path/to/dl3dv_benchmark`

The existing source manifests may contain S3 paths. Before evaluation, convert
them into the two local manifests named in `experiments.yaml`:

- `/PATH/TO/MANIFESTS/gso_v0.1_instant3d_test_localssd.txt`
- `/PATH/TO/MANIFESTS/dl3dv10k_benchmark_local.txt`

Then generate and freeze each dataset's `view_manifest`. A run must fail
preflight if either its local manifest, its frozen view manifest, or any
referenced local scene is missing; it must not silently fall back to S3 or
randomly replace a scene.

## Paper names versus loaders

`paper_model` and `display_name` are the public identities used in tables and
W&B. `checkpoint_impl` records which implementation family produced each
curated checkpoint; it is provenance metadata written into
`run_metadata.json`, while the implementation actually used for loading comes
from the `config` field.

The two DL3DV TTT-MoE rows record `checkpoint_impl: ttt_moe_alt`: those
curated checkpoints were saved by an alternate implementation of the same
architecture, so a state dict trained with the standard implementation is not
interchangeable with them. The alternate implementation is mathematically the
same model as `uttt_moe_block.py` with `pool_mode: per_block` and differs only
in parameter layout, so re-training the released
`configs/ownership/dl3dv/ttt_moe_e{4,8}a1.yaml` reproduces the row. W&B and
paper-facing output use `TTT-MoE-e4a1` and `TTT-MoE-e8a1` regardless.

## Configuration count

Each dataset contains the same 17 paper configurations:

| Family | Configurations | Count |
|---|---|---:|
| TTT-Dense | baseline | 1 |
| TTT-MoE | e4a1, e8a1 | 2 |
| uTTT-Dense | r1, r2, r4, r8 | 4 |
| uTTT-MoE top-1 | e8a1, e16a1, e32a1, e64a1 | 4 |
| uTTT-MoE active sweep | e32/e64 × a2/a4/a8 | 6 |
| **Total per dataset** |  | **17** |

## Prepare local manifests

Run these commands from the repository root after both local data directories
are present:

```bash
python -m uttt_nvs.eval.build_local_dataset_manifest \
  --source-manifest /path/to/gso_test_manifest.txt \
  --source-prefix s3://YOUR_BUCKET/gso_renderings/ \
  --local-root /path/to/gso_renderings \
  --output /PATH/TO/MANIFESTS/gso_v0.1_instant3d_test_localssd.txt

python -m uttt_nvs.eval.build_local_dataset_manifest \
  --source-manifest /path/to/dl3dv_benchmark_manifest.txt \
  --source-prefix s3://YOUR_BUCKET/dl3dv_benchmark/ \
  --local-root /path/to/dl3dv_benchmark \
  --output /PATH/TO/MANIFESTS/dl3dv10k_benchmark_local.txt
```

Freeze the exact `random.sample` order used by the existing dataset protocol:

```bash
python -m uttt_nvs.eval.build_view_manifest \
  --dataset-manifest /PATH/TO/MANIFESTS/gso_v0.1_instant3d_test_localssd.txt \
  --dataset gso \
  --seed 9595 \
  --output /PATH/TO/MANIFESTS/gso_views_seed9595_v24.json

python -m uttt_nvs.eval.build_view_manifest \
  --dataset-manifest /PATH/TO/MANIFESTS/dl3dv10k_benchmark_local.txt \
  --dataset dl3dv \
  --seed 9595 \
  --output /PATH/TO/MANIFESTS/dl3dv_views_seed9595_v24.json
```

The sampled indices are intentionally not sorted. View index measures how
many randomly ordered ground-truth conditioning views have accumulated,
matching the teacher-forced forward in
`uttt_nvs/models/lvsm.py` and `uttt_nvs/models/lvsm_moe.py`. This is not a single-input protocol
that predicts all 23 targets from View 0 alone.

## Run metrics

Activate the environment described in the top-level README. If the machine has
a user-site PyTorch that could shadow it, exclude the user site:

```bash
export PYTHONNOUSERSITE=1
```

Two-scene smoke test:

```bash
python -m uttt_nvs.eval.evaluate \
  --experiment gso_ttt_moe_e8a1 \
  --max-scenes 2 \
  --wandb-mode disabled
```

Run the whole matrix by iterating over the registry (and
`experiments_transformer.yaml` for the Transformer rows); the batch launcher
used for the paper is not part of this repository.

```bash
python - <<'EOF'
import subprocess, yaml
reg = "uttt_nvs/eval/experiments.yaml"
for exp in yaml.safe_load(open(reg))["experiments"]:
    subprocess.run(["python", "-m", "uttt_nvs.eval.evaluate",
                    "--registry", reg, "--experiment", exp["id"]], check=False)
EOF
```

`evaluate.py` builds its LPIPS-VGG metric on the evaluation device (the `lpips`
package downloads the weights on first use). It skips runs with `completion.json`
by default; use `--overwrite` for an intentional rerun. A failed run without a
completion marker is recomputed safely on the next launch.

Each completed run writes:

- `raw_metrics.csv` and `raw_metrics.parquet` with the exact required raw
  schema;
- `overall_summary.json` with scene-macro PSNR/SSIM/LPIPS;
- `per_view_summary.csv` with View 1–23 mean, 95% scene-bootstrap interval,
  and valid-scene count;
- `run_metadata.json` with checkpoint/config/code/view-manifest provenance.
- `completion.json`, written only after local summaries and W&B logging
  complete.

Combine and strictly validate completed full runs with:

```bash
python -m uttt_nvs.eval.aggregate
```

## Render comparisons

The paper-facing rendering script uses TTT-Dense, TTT-MoE-e8a1, and
uTTT-MoE-e64a1 on the same frozen scenes and views. It reads each column's
checkpoint and config from the registry entry of the chosen dataset; there is
no implementation switching inside the script, so the DL3DV TTT-MoE column
uses whatever `config:`/`checkpoint:` pair its registry entry names.

```bash
python -m uttt_nvs.eval.render_comparison \
  --dataset gso \
  --scene-count 8 \
  --view-indices 1,12,23 \
  --wandb-mode online

python -m uttt_nvs.eval.render_comparison \
  --dataset dl3dv \
  --scene-count 8 \
  --view-indices 1,12,23 \
  --wandb-mode online
```

The deterministic hash-based scene selection avoids cherry-picking. Raw PNGs
and a grid with latest-GT-context/GT/TTT-Dense/TTT-MoE/uTTT-MoE columns are written
under `/PATH/TO/MANIFESTS`. Every row labels the full
teacher-forced context (`GT 0..k-1`) for target View `k`.
