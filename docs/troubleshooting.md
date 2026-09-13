# Troubleshooting

Use the earliest failing layer of the [validation ladder](testing.md). Do not
start a multi-node training run to diagnose a configuration typo or a missing
image. Save the exact command and the first error, not only the final torchrun
summary.

## Environment and imports

| Symptom | Likely cause | Next check |
|---|---|---|
| `No module named flame` | Running an LLM module from the repository root | `cd uttt_llm` or use the training launcher |
| `No module named uttt_nvs` | Running NVS modules from inside the domain directory | Return to the repository root |
| flash-attn `undefined symbol` | Extension built against a different torch/CUDA ABI | Confirm active environment and rebuild flash-attn after installing the intended torch |
| torch/torchtitan varlen API import failure | Incompatible torchtitan version | Use the released `>=0.1,<0.2` requirement with the reference torch build |
| Missing `wheel` or `ninja` while building flash-attn | Build tools absent in the active environment | Install domain requirements before `flash-attn --no-build-isolation` |
| CPU checks pass, model import fails | CPU checks do not load the model stack | Run the domain construction check in the CUDA environment |

Collect non-secret environment evidence:

```bash
python --version
python -m pip show torch torchvision triton flash-attn torchtitan flash-linear-attention
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
nvidia-smi
```

Do not “fix” ABI errors by repeatedly upgrading everything. Establish one
consistent environment and rebuild extensions against it. An attention stub
can help isolate import/construction problems but is not numerical or
performance validation of the real attention implementation.

## Launcher exits before training

`--dry-run` exposes the resolved config paths, execution directory and complete
torchrun arguments. The launchers intentionally reject these states:

- `NPROC_PER_NODE=0`, a negative/non-integer value, or leading-zero integers.
- A partial distributed environment, such as `MASTER_ADDR` without a port.
- `WORLD_SIZE` that is not divisible by the local process count.
- A missing multi-node rank or a rank outside `[0, node_count)`.
- A configuration file that cannot be read.
- An existing LLM dataset registry without a non-empty `train.files` string.

For a standalone run, unset `MASTER_ADDR`, `MASTER_PORT`, `WORLD_SIZE`, `RANK`
and `NODE_RANK`. For a multi-node run, set them consistently on every node and
use a unique shared `JOB_UUID`. A dry run does not contact the rendezvous host;
firewalls, DNS and reachability still need a real distributed smoke run.

## Training data fails or the loader keeps retrying

Run `python -m tools.check_manifest ... --check-images` for local NVS data.
Common causes are fewer views than `num_views` / `eval_num_views`, relative
paths interpreted from the wrong directory, missing images, invalid camera
matrices and cloud objects inaccessible from workers.

A manifest entry is a camera JSON, not an image or a scene directory. A camera
JSON must contain `frames`, and each frame must contain intrinsics, `w2c` and
`file_path`. The toy generator's default 24 views cover both released training
and evaluation sampling. Its repeated entries are for smoke-test batch sizing,
not representative training data.

If the trainer reports zero complete batches, calculate samples per rank and
per-device batch size. Lowering only one config's batch field can trigger the
global-batch guard; update the full arithmetic intentionally.

## Resume and checkpoint problems

| Symptom | Explanation / action |
|---|---|
| `--load` seems ignored | Existing output-directory checkpoints take precedence; use a fresh `exp_name` |
| Missing checkpoint now raises | Intended: explicitly requesting a checkpoint must not silently start from random weights |
| Latest checkpoint is corrupt | Directory restore tries older checkpoint files; inspect the reported fallback step |
| Every checkpoint is corrupt | Restore fails; recover a valid copy instead of rerunning under the same name |
| Fine-tune counters return to zero | Expected with reset flags; model weights are still loaded |
| Retention uses too much space | `save_last_n_ckpts` now counts files; review legacy values like 1001 |
| Model-load missing/unexpected keys | Compare architecture, package version and config before continuing |

Never disable checkpoint-deserialization safety just to load an untrusted file.
Atomic writes help prevent partial new files, but they do not repair historical
corruption, full disks or a mismatched architecture.

## Smoke run takes too long or evaluation hangs

- For NVS, set both the pass limit and
  `training.round_max_fwdbwd_passes_to_epoch=False`.
- The first CUDA step can compile and autotune Triton kernels; it is not
  representative steady-state timing.
- If evaluation has too few complete batches per rank, set
  `training.eval_every=0` and use the separately sharded full-test evaluator.
  A valid evaluation manifest is still required at initialization.
- For PTL out-of-memory failures, retain hidden-state projection and a bounded
  `lm_head_chunk_size`; verify that the model implements that path.
- A NaN in the last PTL position is expected because there is no next-token
  label. Widespread non-finite losses are not explained by that convention.

## Filing a useful issue

Include the repository SHA, domain, exact config files and overrides, launch
command, GPU topology, environment versions, first traceback, expected versus
actual behavior and the smallest reproducer. State whether the failure occurs
in static checks, CPU tests, model construction, single-GPU execution or
multi-node execution. Remove access tokens, W&B keys, private dataset contents
and sensitive paths before posting.
