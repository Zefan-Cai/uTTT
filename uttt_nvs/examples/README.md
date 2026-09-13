# Launch examples

These are **examples, not supported entry points**. Partition names, account
strings, module systems and container setups differ at every site; copy one and
edit the header.

| File | GPUs |
|---|---:|
| `slurm_4gpu.sbatch` | 4 — 32 per GPU, the total batch the configs declare |
| `slurm_multigpu.sbatch` | 16 — spread over whatever nodes your site gives you |

Without a scheduler, `train/launch.sh` runs directly:

```bash
NPROC_PER_NODE=4 bash uttt_nvs/train/launch.sh uttt_nvs/configs/ownership/obj/uttt_moe_e64a1.yaml
```

## Match the GPU count to the batch size

Each config declares `total_batch_size` — the batch size the experiment is
defined at. The GPU count comes from your launcher and is invisible to the
config, so the trainer multiplies out
`batch_size_per_gpu x world_size` and refuses to start on a mismatch.

At the published total of 128:

| GPUs | `batch_size_per_gpu` |
|---:|---:|
| 2 | 64 |
| **4** | **32 (as shipped)** |
| 8 | 16 |
| 16 | 8 |
| 32 | 4 |

How those GPUs are grouped into nodes does not matter to the batch size — only
the total count does.

## Choosing a GPU count

Pick a count that divides the total batch evenly at a per-GPU batch your cards
can hold. The published 4 x 32 needs about 2 GB per GPU at 256 x 256; the larger
`configs/scale/` models need considerably more.
