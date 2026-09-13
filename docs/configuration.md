# Configuration semantics

The [generated catalog](configuration-catalog.md) enumerates the complete release:
64 NVS model/training YAMLs, 57 LLM architecture JSONs and 3 LLM job TOMLs.
Configuration counts do not count trained checkpoints or successful GPU runs.

## NVS experiment families

| Directory | Count | Question |
|---|---:|---|
| `ownership/` | 8 | Layer-private, layer-local pool or universal pool? |
| `dense/` | 8 | How does shared dense fast-weight width change capacity? |
| `width/` | 16 | How do pool size and active expert count interact? |
| `capacity/` | 8 | What changes under the released capacity variants? |
| `schedule/` | 4 | Aggregated versus chained shared-state writes? |
| `topology/` | 12 | One universal pool or several depth-partitioned pools? |
| `transformer/` | 2 | The non-TTT attention baseline |
| `scale/` | 6 | Three-stage progressive-resolution training in each domain |

`obj` means the Objaverse/GSO setup; `dl3dv` means the DL3DV setup. Do not infer
that the two families share data, identical scale schedules or the same number
of evaluation scenes.

### Architecture fields

| YAML field | Meaning / constraint |
|---|---|
| `model.class_name` | Fully qualified image-model class |
| `model.image_size`, `patch_size` | Square image resolution and patch width; divisible |
| `model.dim`, `layers` | Outer feature width and stack depth |
| `block_config.type` | Fully qualified memory/attention implementation |
| `params.pool_mode` | `shared` for a universal pool, `per_block` for layer-local pools |
| `params.num_experts` | Routed experts in each pool |
| `params.num_active_experts` | Active routed experts; must lie in `1..num_experts` |
| `params.num_shared_pools` | Number of depth-partitioned pools; depth must divide into groups |
| `params.router_share_mode` | Router parameter ownership, independent of fast-state ownership |
| `params.aggregate_write` | Shared-state aggregated update schedule, not a cosmetic flag |
| `params.fw_inter_multi` | Fast-weight intermediate width multiplier |
| `params.attn_head_dim`, `fw_head_dim` | Attention / fast-weight head dimensions |

Not every block supports every field. A parameter absent from one family cannot
be copied blindly from another: the target constructor remains authoritative.

### Training fields

`batch_size_per_gpu × world_size × grad_accum_steps = total_batch_size`.
The trainer refuses a mismatch. `max_fwdbwd_passes` counts forward/backward
passes; it is not necessarily the number of optimizer updates. For exact smoke
limits also set `round_max_fwdbwd_passes_to_epoch: false`.

`training.dataset` selects a key in `uttt_nvs/datasets.yaml`; the registry gives
train/eval manifest paths. Explicit dataset-path fields take precedence in the
trainer. `-s dotted.key value` overrides a field from the command line. Record
every override alongside the config: an unchanged filename does not mean an
unchanged experiment.

The scale configs are a chain of fine-tunes, not six independent scratch runs.
Use the preceding stage's checkpoint and the reset settings described in
[training](training.md). The catalog reports literal per-GPU/global batches and
the implied GPU count, including the 64-GPU DL3DV 128² stage.

## LLM architecture versus job configuration

The architecture JSON supplies widths, depths, `model_type`, attention/memory
settings and routing/balancing choices. The TOML supplies sequence lengths,
dataset, optimizer, batch, distributed degrees, checkpoint cadence and logging.
The training launcher combines these two files; neither is a complete run alone.

| JSON field | Interpretation |
|---|---|
| `model_type` | Selects a Transformers-registered implementation; keep it checkpoint-compatible |
| `hidden_size`, `num_hidden_layers` | Model width and depth |
| `num_heads`, `fw_num_heads` | Attention and fast-weight head counts when applicable |
| `max_position_embeddings` | Configured context extent, not proof of length extrapolation |
| `chunk_size` | Context-processing chunk size for implementations that expose it |
| `memory_num_experts`, `memory_num_active_experts` | Memory pool size / active experts |
| `memory_pool_mode` | Shared-pool ownership in uTTT-MoE |
| `memory_router_share_mode` | Router sharing across layers |
| `memory_update_aggregate_write` | Aggregate writes to the shared memory state |
| `memory_lb_mode`, `memory_lb_scope` | Balancing mechanism and scope |
| `memory_loss_free_update_rate` | Routing-bias update rate, not the outer optimizer LR |

The two scales have different uTTT-MoE loss-free update rates: `3e-5` at 124M
and `1e-3` at 760M in the main configs. The TTT-MoE-with-LB main configs use
`1e-5`. Preserve these distinctions when reproducing the main table.

The LLM training batch formula uses **data-parallel degree**, not blindly total
world size. They agree for the released training jobs because tensor/context/
pipeline parallel degrees are one. Changing those degrees requires revisiting
the arithmetic and the implementation's supported parallelism.

## Editing and checking

```bash
python -m tools.check_configs
python -m tools.check_configs --write-catalog
python -m tools.check_configs --check-catalog
```

The checker covers positive integer dimensions, head/patch divisibility,
active-expert bounds, selected shared-pool invariants, batch arithmetic, source
class existence, model-type declarations and TOML structure. It does not
promise to validate every constructor-specific option. It also does not check
dataset accessibility, checkpoint compatibility, available GPU memory, fused
kernel correctness or model quality. Run the domain construction checks after
changing a model or adding a config.
