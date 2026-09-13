# Architecture and source map

## What is universal?

Test-time training (TTT) turns context into a fast-weight state. A model reads
that state to produce features and updates it using information from the
context. Outer-loop training learns the model's ordinary parameters; inner-loop
updates adapt the fast weights while processing a sample.

The central experimental axis here is **ownership across depth**, not whether
the model has attention or whether the router itself is shared.

| Regime | Fast-state owner | Cross-layer sharing | Released examples |
|---|---|---|---|
| Layer-private | One layer | None | LaCT, TTT-Dense |
| Layer-local pool | One layer, containing routed experts | None between layers | TTT-MoE |
| Universal dense | The depth stack | One dense operator | uTTT-Dense in NVS |
| Universal expert pool | The depth stack | One expert pool | uTTT-MoE in NVS and LLM |
| Partitioned universal pools | A group of layers | Within each group | NVS topology sweep |

The release does not contain LLM uTTT-Dense. Do not create that row by renaming
the LLM TTT-Dense package: layer-private and shared state are different methods.

## Three distinct kinds of state

| State | Lifetime | Where to look |
|---|---|---|
| Learned parameters / fast-weight initialization | Across training steps | Model parameters and saved state dict |
| Adapted fast weights and inner-loop update state | During context processing; generation can carry caches forward | NVS `_init_weights` and LLM fast-weight/cache initialization |
| Outer optimizer, scheduler and step counters | Across training steps and resume | NVS checkpoint payload; LLM DCP checkpoint |

NVS initializes its adapted weights inside `MemoryModel.forward` for the batch.
Those per-example adapted tensors are not a global service cache. LLM generation
has explicit cache paths; changing prompt boundaries or reusing caches across
unrelated examples changes the experiment. The NVS checkpoint utility saves
model/optimizer/scheduler state, not an in-progress context's ephemeral tensors.

## Reading, routing and writing are separate decisions

1. **Ownership** chooses which layers can access a fast-weight bank.
2. **Routing** selects active experts for a feature/token and combines outputs.
3. **Router sharing** chooses whether layers reuse the same router parameters.
4. **Write scheduling** decides when the bank changes and which version each
   layer observes when computing its update.
5. **Load balancing** regulates expert use through an auxiliary objective or
   routing-bias updates, depending on the implementation.

Sharing a router does not make layer-private fast weights universal. Similarly,
using a shared pool does not imply that every expert is active for every token.
The `E/K` notation in the catalog means pool size / active experts, not number
of layers or GPUs.

## Aggregated versus chained writes

In the NVS shared-pool aggregated path, layers compute raw fast-weight gradients
against the same pre-write state. The implementation accumulates those gradients
and applies the write afterward. In a chained schedule, later updates observe
state changed by earlier updates. These schedules are not generally equivalent.

Conceptually, for a shared pre-write state `F` and layer contributions `g_l`:

```text
aggregated: compute g_1(F), ..., g_L(F); then apply the combined update
chained:    F_1 = update(F, g_1(F)); F_2 = update(F_1, g_2(F_1)); ...
```

This is a schedule diagram, not the exact optimizer equation: learning-rate
gates, normalization, momentum, Muon and implementation-specific aggregation
rules live in the actual block code. In particular, the LLM
`memory_aggregate_momentum_mode` and `memory_aggregate_lr_scale` must not be
silently replaced by a simple arithmetic mean.

## NVS path through the code

| Stage | Source | Responsibility |
|---|---|---|
| Data | [loader.py](../uttt_nvs/data/loader.py) | Sample views, load RGB/calibration, resize/crop, form tensors |
| Image model | [lvsm.py](../uttt_nvs/models/lvsm.py), [lvsm_moe.py](../uttt_nvs/models/lvsm_moe.py) | Image/ray features and view-conditioned prediction |
| Layer-private dense | [ttt_dense_block.py](../uttt_nvs/models/ttt_dense_block.py) | TTT-Dense baseline |
| Shared dense | [uttt_dense_block.py](../uttt_nvs/models/uttt_dense_block.py) | uTTT-Dense width variants |
| Routed ownership ladder | [uttt_moe_block.py](../uttt_nvs/models/uttt_moe_block.py) | Shared or per-block pools; router and write schedules |
| Partitioned pools | [uttt_moe_partitioned.py](../uttt_nvs/models/uttt_moe_partitioned.py) | Several depth-shared pools |
| GPU primitives | [kernels](../uttt_nvs/models/kernels) | Grouped GEMMs, SwiGLU, routing/permutation and backward operations |
| Outer loop | [trainer.py](../uttt_nvs/train/trainer.py) | DDP, batch checks, updates, evaluation and checkpoints |

The YAML's `model.class_name` resolves the image model; `block_config.type`
selects the memory block. The CPU checker verifies that the named classes exist
but does not instantiate them.

## LLM path through the code

The LLM release intentionally preserves separate packages for the experimental
rows. The [registry](../uttt_llm/flame/custom_models/__init__.py) imports their
Transformers registrations. `model_type` is a checkpoint compatibility field,
not just a display label.

- [TTT-Dense](../uttt_llm/flame/custom_models/ttt_dense) and the two
  [unbalanced](../uttt_llm/flame/custom_models/ttt_moe_no_lb) /
  [balanced](../uttt_llm/flame/custom_models/ttt_moe_lb) TTT-MoE implementations
  correspond to the released training runs, rather than aliases of uTTT-MoE.
- [uTTT-MoE modeling](../uttt_llm/flame/custom_models/uttt_moe/modeling_recurrent_lact.py)
  owns the shared-pool initialization, aggregate update path, routing statistics,
  forward execution and generation cache handling.
- [uTTT-MoE configuration](../uttt_llm/flame/custom_models/uttt_moe/configuration_recurrent_lact.py)
  exposes model-specific defaults. The shipped JSON overrides are the experiment
  definitions; defaults alone are not a reproduction recipe.
- [flame.train](../uttt_llm/flame/train.py) handles the distributed outer loop;
  [flame.eval_loss](../uttt_llm/flame/eval_loss.py) implements PTL evaluation.

## Safe extension boundaries

Keep ownership, routing, schedule and capacity changes in separate ablations.
Preserve legacy `model_type` identifiers and state-dict keys unless migration is
intentional and tested. A kernel optimization needs forward and backward
comparison against a reference on CUDA, including uneven expert loads and
multiple sequence/view lengths. CPU construction and shell tests cannot justify
a speedup, memory reduction or accuracy claim.
