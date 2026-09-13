# Configuration catalog

Generated from the checked-in configurations by `python -m tools.check_configs --write-catalog`.
Do not edit tables by hand. Counts describe configuration files, not validated training runs.

See [configuration semantics](configuration.md) before changing ownership, routing or update schedules.

## Novel view synthesis

GPU count is total_batch_size / (batch_size_per_gpu × grad_accum_steps); it is not a memory-fit guarantee.
Steps are the configured forward/backward pass limit, before optional epoch rounding.

| Config | Block | Pool | Experts / active | Resolution | Layers × width | Global batch | GPUs | Steps |
|---|---|---|---|---:|---:|---:|---:|---:|
| [capacity/dl3dv/uttt_moe_e16a1.yaml](../uttt_nvs/configs/capacity/dl3dv/uttt_moe_e16a1.yaml) | `uttt_moe_block` | shared | 16 / 1 | 256² | 8 × 512 | 128 | 4 | 20,000 |
| [capacity/dl3dv/uttt_moe_e32a1.yaml](../uttt_nvs/configs/capacity/dl3dv/uttt_moe_e32a1.yaml) | `uttt_moe_block` | shared | 32 / 1 | 256² | 8 × 512 | 128 | 4 | 20,000 |
| [capacity/dl3dv/uttt_moe_e64a1.yaml](../uttt_nvs/configs/capacity/dl3dv/uttt_moe_e64a1.yaml) | `uttt_moe_block` | shared | 64 / 1 | 256² | 8 × 512 | 128 | 4 | 20,000 |
| [capacity/dl3dv/uttt_moe_e8a1.yaml](../uttt_nvs/configs/capacity/dl3dv/uttt_moe_e8a1.yaml) | `uttt_moe_block` | shared | 8 / 1 | 256² | 8 × 512 | 128 | 4 | 20,000 |
| [capacity/obj/uttt_moe_e16a1.yaml](../uttt_nvs/configs/capacity/obj/uttt_moe_e16a1.yaml) | `uttt_moe_block` | shared | 16 / 1 | 256² | 8 × 512 | 128 | 4 | 20,000 |
| [capacity/obj/uttt_moe_e32a1.yaml](../uttt_nvs/configs/capacity/obj/uttt_moe_e32a1.yaml) | `uttt_moe_block` | shared | 32 / 1 | 256² | 8 × 512 | 128 | 4 | 20,000 |
| [capacity/obj/uttt_moe_e64a1.yaml](../uttt_nvs/configs/capacity/obj/uttt_moe_e64a1.yaml) | `uttt_moe_block` | shared | 64 / 1 | 256² | 8 × 512 | 128 | 4 | 20,000 |
| [capacity/obj/uttt_moe_e8a1.yaml](../uttt_nvs/configs/capacity/obj/uttt_moe_e8a1.yaml) | `uttt_moe_block` | shared | 8 / 1 | 256² | 8 × 512 | 128 | 4 | 20,000 |
| [dense/dl3dv/uttt_dense_r1.yaml](../uttt_nvs/configs/dense/dl3dv/uttt_dense_r1.yaml) | `uttt_dense_block` | see block | — | 256² | 8 × 512 | 128 | 4 | 20,000 |
| [dense/dl3dv/uttt_dense_r2.yaml](../uttt_nvs/configs/dense/dl3dv/uttt_dense_r2.yaml) | `uttt_dense_block` | see block | — | 256² | 8 × 512 | 128 | 4 | 20,000 |
| [dense/dl3dv/uttt_dense_r4.yaml](../uttt_nvs/configs/dense/dl3dv/uttt_dense_r4.yaml) | `uttt_dense_block` | see block | — | 256² | 8 × 512 | 128 | 4 | 20,000 |
| [dense/dl3dv/uttt_dense_r8.yaml](../uttt_nvs/configs/dense/dl3dv/uttt_dense_r8.yaml) | `uttt_dense_block` | see block | — | 256² | 8 × 512 | 128 | 4 | 20,000 |
| [dense/obj/uttt_dense_r1.yaml](../uttt_nvs/configs/dense/obj/uttt_dense_r1.yaml) | `uttt_dense_block` | see block | — | 256² | 8 × 512 | 128 | 4 | 20,000 |
| [dense/obj/uttt_dense_r2.yaml](../uttt_nvs/configs/dense/obj/uttt_dense_r2.yaml) | `uttt_dense_block` | see block | — | 256² | 8 × 512 | 128 | 4 | 20,000 |
| [dense/obj/uttt_dense_r4.yaml](../uttt_nvs/configs/dense/obj/uttt_dense_r4.yaml) | `uttt_dense_block` | see block | — | 256² | 8 × 512 | 128 | 4 | 20,000 |
| [dense/obj/uttt_dense_r8.yaml](../uttt_nvs/configs/dense/obj/uttt_dense_r8.yaml) | `uttt_dense_block` | see block | — | 256² | 8 × 512 | 128 | 4 | 20,000 |
| [ownership/dl3dv/ttt_dense.yaml](../uttt_nvs/configs/ownership/dl3dv/ttt_dense.yaml) | `ttt_dense_block` | see block | — | 256² | 8 × 512 | 128 | 4 | 20,000 |
| [ownership/dl3dv/ttt_moe_e4a1.yaml](../uttt_nvs/configs/ownership/dl3dv/ttt_moe_e4a1.yaml) | `uttt_moe_block` | per_block | 4 / 1 | 256² | 8 × 512 | 128 | 4 | 20,000 |
| [ownership/dl3dv/ttt_moe_e8a1.yaml](../uttt_nvs/configs/ownership/dl3dv/ttt_moe_e8a1.yaml) | `uttt_moe_block` | per_block | 8 / 1 | 256² | 8 × 512 | 128 | 4 | 20,000 |
| [ownership/dl3dv/uttt_moe_e64a1.yaml](../uttt_nvs/configs/ownership/dl3dv/uttt_moe_e64a1.yaml) | `uttt_moe_block` | shared | 64 / 1 | 256² | 8 × 512 | 128 | 4 | 20,000 |
| [ownership/obj/ttt_dense.yaml](../uttt_nvs/configs/ownership/obj/ttt_dense.yaml) | `ttt_dense_block` | see block | — | 256² | 8 × 512 | 128 | 4 | 20,000 |
| [ownership/obj/ttt_moe_e4a1.yaml](../uttt_nvs/configs/ownership/obj/ttt_moe_e4a1.yaml) | `uttt_moe_block` | per_block | 4 / 1 | 256² | 8 × 512 | 128 | 4 | 20,000 |
| [ownership/obj/ttt_moe_e8a1.yaml](../uttt_nvs/configs/ownership/obj/ttt_moe_e8a1.yaml) | `uttt_moe_block` | per_block | 8 / 1 | 256² | 8 × 512 | 128 | 4 | 20,000 |
| [ownership/obj/uttt_moe_e64a1.yaml](../uttt_nvs/configs/ownership/obj/uttt_moe_e64a1.yaml) | `uttt_moe_block` | shared | 64 / 1 | 256² | 8 × 512 | 128 | 4 | 20,000 |
| [scale/dl3dv/uttt_moe_large_128.yaml](../uttt_nvs/configs/scale/dl3dv/uttt_moe_large_128.yaml) | `uttt_moe_block` | shared | 64 / 1 | 128² | 24 × 768 | 768 | 64 | 120,000 |
| [scale/dl3dv/uttt_moe_large_256.yaml](../uttt_nvs/configs/scale/dl3dv/uttt_moe_large_256.yaml) | `uttt_moe_block` | shared | 64 / 1 | 256² | 24 × 768 | 256 | 64 | 12,000 |
| [scale/dl3dv/uttt_moe_large_512.yaml](../uttt_nvs/configs/scale/dl3dv/uttt_moe_large_512.yaml) | `uttt_moe_block` | shared | 64 / 1 | 512² | 24 × 768 | 64 | 64 | 7,000 |
| [scale/obj/uttt_moe_large_1024.yaml](../uttt_nvs/configs/scale/obj/uttt_moe_large_1024.yaml) | `uttt_moe_block` | shared | 64 / 1 | 1024² | 24 × 768 | 64 | 64 | 4,500 |
| [scale/obj/uttt_moe_large_256.yaml](../uttt_nvs/configs/scale/obj/uttt_moe_large_256.yaml) | `uttt_moe_block` | shared | 64 / 1 | 256² | 24 × 768 | 512 | 64 | 81,038 |
| [scale/obj/uttt_moe_large_512.yaml](../uttt_nvs/configs/scale/obj/uttt_moe_large_512.yaml) | `uttt_moe_block` | shared | 64 / 1 | 512² | 24 × 768 | 128 | 64 | 70,000 |
| [schedule/dl3dv/uttt_moe_e64a1.yaml](../uttt_nvs/configs/schedule/dl3dv/uttt_moe_e64a1.yaml) | `uttt_moe_block` | shared | 64 / 1 | 256² | 8 × 512 | 128 | 4 | 20,000 |
| [schedule/dl3dv/uttt_moe_e64a1_chained.yaml](../uttt_nvs/configs/schedule/dl3dv/uttt_moe_e64a1_chained.yaml) | `uttt_moe_block` | shared | 64 / 1 | 256² | 8 × 512 | 128 | 4 | 20,000 |
| [schedule/obj/uttt_moe_e64a1.yaml](../uttt_nvs/configs/schedule/obj/uttt_moe_e64a1.yaml) | `uttt_moe_block` | shared | 64 / 1 | 256² | 8 × 512 | 128 | 4 | 20,000 |
| [schedule/obj/uttt_moe_e64a1_chained.yaml](../uttt_nvs/configs/schedule/obj/uttt_moe_e64a1_chained.yaml) | `uttt_moe_block` | shared | 64 / 1 | 256² | 8 × 512 | 128 | 4 | 20,000 |
| [topology/dl3dv/uttt_moe_e32a1.yaml](../uttt_nvs/configs/topology/dl3dv/uttt_moe_e32a1.yaml) | `uttt_moe_block` | shared | 32 / 1 | 256² | 8 × 512 | 128 | 4 | 20,000 |
| [topology/dl3dv/uttt_moe_e64a1.yaml](../uttt_nvs/configs/topology/dl3dv/uttt_moe_e64a1.yaml) | `uttt_moe_block` | shared | 64 / 1 | 256² | 8 × 512 | 128 | 4 | 20,000 |
| [topology/dl3dv/uttt_moe_p2_e32.yaml](../uttt_nvs/configs/topology/dl3dv/uttt_moe_p2_e32.yaml) | `uttt_moe_partitioned` | 2 shared pools | 32 / 1 | 256² | 8 × 512 | 128 | 4 | 20,000 |
| [topology/dl3dv/uttt_moe_p2_e64.yaml](../uttt_nvs/configs/topology/dl3dv/uttt_moe_p2_e64.yaml) | `uttt_moe_partitioned` | 2 shared pools | 64 / 1 | 256² | 8 × 512 | 128 | 4 | 20,000 |
| [topology/dl3dv/uttt_moe_p4_e32.yaml](../uttt_nvs/configs/topology/dl3dv/uttt_moe_p4_e32.yaml) | `uttt_moe_partitioned` | 4 shared pools | 32 / 1 | 256² | 8 × 512 | 128 | 4 | 20,000 |
| [topology/dl3dv/uttt_moe_p4_e64.yaml](../uttt_nvs/configs/topology/dl3dv/uttt_moe_p4_e64.yaml) | `uttt_moe_partitioned` | 4 shared pools | 64 / 1 | 256² | 8 × 512 | 128 | 4 | 20,000 |
| [topology/obj/uttt_moe_e32a1.yaml](../uttt_nvs/configs/topology/obj/uttt_moe_e32a1.yaml) | `uttt_moe_block` | shared | 32 / 1 | 256² | 8 × 512 | 128 | 4 | 20,000 |
| [topology/obj/uttt_moe_e64a1.yaml](../uttt_nvs/configs/topology/obj/uttt_moe_e64a1.yaml) | `uttt_moe_block` | shared | 64 / 1 | 256² | 8 × 512 | 128 | 4 | 20,000 |
| [topology/obj/uttt_moe_p2_e32.yaml](../uttt_nvs/configs/topology/obj/uttt_moe_p2_e32.yaml) | `uttt_moe_partitioned` | 2 shared pools | 32 / 1 | 256² | 8 × 512 | 128 | 4 | 20,000 |
| [topology/obj/uttt_moe_p2_e64.yaml](../uttt_nvs/configs/topology/obj/uttt_moe_p2_e64.yaml) | `uttt_moe_partitioned` | 2 shared pools | 64 / 1 | 256² | 8 × 512 | 128 | 4 | 20,000 |
| [topology/obj/uttt_moe_p4_e32.yaml](../uttt_nvs/configs/topology/obj/uttt_moe_p4_e32.yaml) | `uttt_moe_partitioned` | 4 shared pools | 32 / 1 | 256² | 8 × 512 | 128 | 4 | 20,000 |
| [topology/obj/uttt_moe_p4_e64.yaml](../uttt_nvs/configs/topology/obj/uttt_moe_p4_e64.yaml) | `uttt_moe_partitioned` | 4 shared pools | 64 / 1 | 256² | 8 × 512 | 128 | 4 | 20,000 |
| [transformer/dl3dv/transformer.yaml](../uttt_nvs/configs/transformer/dl3dv/transformer.yaml) | `transformer_block` | see block | — | 256² | 8 × 512 | 128 | 4 | 20,000 |
| [transformer/obj/transformer.yaml](../uttt_nvs/configs/transformer/obj/transformer.yaml) | `transformer_block` | see block | — | 256² | 8 × 512 | 128 | 4 | 20,000 |
| [width/dl3dv/uttt_moe_e32a1.yaml](../uttt_nvs/configs/width/dl3dv/uttt_moe_e32a1.yaml) | `uttt_moe_block` | shared | 32 / 1 | 256² | 8 × 512 | 128 | 4 | 20,000 |
| [width/dl3dv/uttt_moe_e32a2.yaml](../uttt_nvs/configs/width/dl3dv/uttt_moe_e32a2.yaml) | `uttt_moe_block` | shared | 32 / 2 | 256² | 8 × 512 | 128 | 4 | 20,000 |
| [width/dl3dv/uttt_moe_e32a4.yaml](../uttt_nvs/configs/width/dl3dv/uttt_moe_e32a4.yaml) | `uttt_moe_block` | shared | 32 / 4 | 256² | 8 × 512 | 128 | 4 | 20,000 |
| [width/dl3dv/uttt_moe_e32a8.yaml](../uttt_nvs/configs/width/dl3dv/uttt_moe_e32a8.yaml) | `uttt_moe_block` | shared | 32 / 8 | 256² | 8 × 512 | 128 | 4 | 20,000 |
| [width/dl3dv/uttt_moe_e64a1.yaml](../uttt_nvs/configs/width/dl3dv/uttt_moe_e64a1.yaml) | `uttt_moe_block` | shared | 64 / 1 | 256² | 8 × 512 | 128 | 4 | 20,000 |
| [width/dl3dv/uttt_moe_e64a2.yaml](../uttt_nvs/configs/width/dl3dv/uttt_moe_e64a2.yaml) | `uttt_moe_block` | shared | 64 / 2 | 256² | 8 × 512 | 128 | 4 | 20,000 |
| [width/dl3dv/uttt_moe_e64a4.yaml](../uttt_nvs/configs/width/dl3dv/uttt_moe_e64a4.yaml) | `uttt_moe_block` | shared | 64 / 4 | 256² | 8 × 512 | 128 | 4 | 20,000 |
| [width/dl3dv/uttt_moe_e64a8.yaml](../uttt_nvs/configs/width/dl3dv/uttt_moe_e64a8.yaml) | `uttt_moe_block` | shared | 64 / 8 | 256² | 8 × 512 | 128 | 4 | 20,000 |
| [width/obj/uttt_moe_e32a1.yaml](../uttt_nvs/configs/width/obj/uttt_moe_e32a1.yaml) | `uttt_moe_block` | shared | 32 / 1 | 256² | 8 × 512 | 128 | 4 | 20,000 |
| [width/obj/uttt_moe_e32a2.yaml](../uttt_nvs/configs/width/obj/uttt_moe_e32a2.yaml) | `uttt_moe_block` | shared | 32 / 2 | 256² | 8 × 512 | 128 | 4 | 20,000 |
| [width/obj/uttt_moe_e32a4.yaml](../uttt_nvs/configs/width/obj/uttt_moe_e32a4.yaml) | `uttt_moe_block` | shared | 32 / 4 | 256² | 8 × 512 | 128 | 4 | 20,000 |
| [width/obj/uttt_moe_e32a8.yaml](../uttt_nvs/configs/width/obj/uttt_moe_e32a8.yaml) | `uttt_moe_block` | shared | 32 / 8 | 256² | 8 × 512 | 128 | 4 | 20,000 |
| [width/obj/uttt_moe_e64a1.yaml](../uttt_nvs/configs/width/obj/uttt_moe_e64a1.yaml) | `uttt_moe_block` | shared | 64 / 1 | 256² | 8 × 512 | 128 | 4 | 20,000 |
| [width/obj/uttt_moe_e64a2.yaml](../uttt_nvs/configs/width/obj/uttt_moe_e64a2.yaml) | `uttt_moe_block` | shared | 64 / 2 | 256² | 8 × 512 | 128 | 4 | 20,000 |
| [width/obj/uttt_moe_e64a4.yaml](../uttt_nvs/configs/width/obj/uttt_moe_e64a4.yaml) | `uttt_moe_block` | shared | 64 / 4 | 256² | 8 × 512 | 128 | 4 | 20,000 |
| [width/obj/uttt_moe_e64a8.yaml](../uttt_nvs/configs/width/obj/uttt_moe_e64a8.yaml) | `uttt_moe_block` | shared | 64 / 8 | 256² | 8 × 512 | 128 | 4 | 20,000 |

## Language-model architectures

Architecture JSONs must be paired with a training/evaluation TOML. E/K denotes routed pool size and active experts.

| Config | model_type | Layers × width | Context | Chunk | E / K | LB mode / update rate |
|---|---|---:|---:|---:|---:|---|
| [balancing/124M/ttt_moe_lb_e4a1_lossfree_u1e-4.json](../uttt_llm/configs/balancing/124M/ttt_moe_lb_e4a1_lossfree_u1e-4.json) | `ttt_moe_lb` | 12 × 768 | 32,768 | 4096 | 4 / 1 | loss_free / 0.0001 |
| [balancing/124M/ttt_moe_lb_e4a1_lossfree_u3e-5.json](../uttt_llm/configs/balancing/124M/ttt_moe_lb_e4a1_lossfree_u3e-5.json) | `ttt_moe_lb` | 12 × 768 | 32,768 | 4096 | 4 / 1 | loss_free / 3e-05 |
| [balancing/124M/ttt_moe_lb_e4a1_lossfree_u3e-6.json](../uttt_llm/configs/balancing/124M/ttt_moe_lb_e4a1_lossfree_u3e-6.json) | `ttt_moe_lb` | 12 × 768 | 32,768 | 4096 | 4 / 1 | loss_free / 3e-06 |
| [balancing/124M/ttt_moe_lb_e4a2_lossfree_u1e-4.json](../uttt_llm/configs/balancing/124M/ttt_moe_lb_e4a2_lossfree_u1e-4.json) | `ttt_moe_lb` | 12 × 768 | 32,768 | 4096 | 4 / 2 | loss_free / 0.0001 |
| [balancing/124M/ttt_moe_lb_e4a2_lossfree_u1e-5.json](../uttt_llm/configs/balancing/124M/ttt_moe_lb_e4a2_lossfree_u1e-5.json) | `ttt_moe_lb` | 12 × 768 | 32,768 | 4096 | 4 / 2 | loss_free / 1e-05 |
| [balancing/124M/ttt_moe_lb_e4a2_lossfree_u3e-5.json](../uttt_llm/configs/balancing/124M/ttt_moe_lb_e4a2_lossfree_u3e-5.json) | `ttt_moe_lb` | 12 × 768 | 32,768 | 4096 | 4 / 2 | loss_free / 3e-05 |
| [balancing/124M/ttt_moe_lb_e4a2_lossfree_u3e-6.json](../uttt_llm/configs/balancing/124M/ttt_moe_lb_e4a2_lossfree_u3e-6.json) | `ttt_moe_lb` | 12 × 768 | 32,768 | 4096 | 4 / 2 | loss_free / 3e-06 |
| [balancing/124M/ttt_moe_lb_e8a1_lossfree_u1e-4.json](../uttt_llm/configs/balancing/124M/ttt_moe_lb_e8a1_lossfree_u1e-4.json) | `ttt_moe_lb` | 12 × 768 | 32,768 | 4096 | 8 / 1 | loss_free / 0.0001 |
| [balancing/124M/ttt_moe_lb_e8a1_lossfree_u1e-5.json](../uttt_llm/configs/balancing/124M/ttt_moe_lb_e8a1_lossfree_u1e-5.json) | `ttt_moe_lb` | 12 × 768 | 32,768 | 4096 | 8 / 1 | loss_free / 1e-05 |
| [balancing/124M/ttt_moe_lb_e8a1_lossfree_u3e-5.json](../uttt_llm/configs/balancing/124M/ttt_moe_lb_e8a1_lossfree_u3e-5.json) | `ttt_moe_lb` | 12 × 768 | 32,768 | 4096 | 8 / 1 | loss_free / 3e-05 |
| [balancing/124M/ttt_moe_lb_e8a1_lossfree_u3e-6.json](../uttt_llm/configs/balancing/124M/ttt_moe_lb_e8a1_lossfree_u3e-6.json) | `ttt_moe_lb` | 12 × 768 | 32,768 | 4096 | 8 / 1 | loss_free / 3e-06 |
| [balancing/124M/ttt_moe_no_lb_e4a1_lb1e-5.json](../uttt_llm/configs/balancing/124M/ttt_moe_no_lb_e4a1_lb1e-5.json) | `ttt_moe_no_lb` | 12 × 768 | 32,768 | 4096 | 4 / 1 | default / — |
| [balancing/124M/ttt_moe_no_lb_e4a1_lb1e-6.json](../uttt_llm/configs/balancing/124M/ttt_moe_no_lb_e4a1_lb1e-6.json) | `ttt_moe_no_lb` | 12 × 768 | 32,768 | 4096 | 4 / 1 | default / — |
| [balancing/124M/ttt_moe_no_lb_e4a2_lb0.json](../uttt_llm/configs/balancing/124M/ttt_moe_no_lb_e4a2_lb0.json) | `ttt_moe_no_lb` | 12 × 768 | 32,768 | 4096 | 4 / 2 | default / — |
| [balancing/124M/ttt_moe_no_lb_e8a1_lb0.json](../uttt_llm/configs/balancing/124M/ttt_moe_no_lb_e8a1_lb0.json) | `ttt_moe_no_lb` | 12 × 768 | 32,768 | 4096 | 8 / 1 | default / — |
| [balancing/124M/uttt_moe_e48a1_lossfree_layer_u1e-3_agg_poolmean_lr1_full_router_per_layer.json](../uttt_llm/configs/balancing/124M/uttt_moe_e48a1_lossfree_layer_u1e-3_agg_poolmean_lr1_full_router_per_layer.json) | `uttt_moe` | 12 × 768 | 32,768 | 4096 | 48 / 1 | loss_free / 0.001 |
| [balancing/124M/uttt_moe_e48a1_lossfree_layer_u1e-4_agg_poolmean_lr1_full_router_per_layer.json](../uttt_llm/configs/balancing/124M/uttt_moe_e48a1_lossfree_layer_u1e-4_agg_poolmean_lr1_full_router_per_layer.json) | `uttt_moe` | 12 × 768 | 32,768 | 4096 | 48 / 1 | loss_free / 0.0001 |
| [balancing/124M/uttt_moe_e48a1_lossfree_layer_u1e-5_agg_poolmean_lr1_full_router_per_layer.json](../uttt_llm/configs/balancing/124M/uttt_moe_e48a1_lossfree_layer_u1e-5_agg_poolmean_lr1_full_router_per_layer.json) | `uttt_moe` | 12 × 768 | 32,768 | 4096 | 48 / 1 | loss_free / 1e-05 |
| [balancing/124M/uttt_moe_e48a1_lossfree_layer_u3e-4_agg_poolmean_lr1_full_router_per_layer.json](../uttt_llm/configs/balancing/124M/uttt_moe_e48a1_lossfree_layer_u3e-4_agg_poolmean_lr1_full_router_per_layer.json) | `uttt_moe` | 12 × 768 | 32,768 | 4096 | 48 / 1 | loss_free / 0.0003 |
| [balancing/124M/uttt_moe_e48a1_lossfree_layer_u3e-5_agg_poolmean_lr1_full_router_per_layer.json](../uttt_llm/configs/balancing/124M/uttt_moe_e48a1_lossfree_layer_u3e-5_agg_poolmean_lr1_full_router_per_layer.json) | `uttt_moe` | 12 × 768 | 32,768 | 4096 | 48 / 1 | loss_free / 3e-05 |
| [balancing/124M/uttt_moe_e48a1_lossfree_pool_u1e-3_agg_poolmean_lr1_full_router_per_layer.json](../uttt_llm/configs/balancing/124M/uttt_moe_e48a1_lossfree_pool_u1e-3_agg_poolmean_lr1_full_router_per_layer.json) | `uttt_moe` | 12 × 768 | 32,768 | 4096 | 48 / 1 | loss_free / 0.001 |
| [balancing/124M/uttt_moe_e48a1_lossfree_pool_u1e-4_agg_poolmean_lr1_full_router_per_layer.json](../uttt_llm/configs/balancing/124M/uttt_moe_e48a1_lossfree_pool_u1e-4_agg_poolmean_lr1_full_router_per_layer.json) | `uttt_moe` | 12 × 768 | 32,768 | 4096 | 48 / 1 | loss_free / 0.0001 |
| [balancing/124M/uttt_moe_e48a1_lossfree_pool_u1e-5_agg_poolmean_lr1_full_router_per_layer.json](../uttt_llm/configs/balancing/124M/uttt_moe_e48a1_lossfree_pool_u1e-5_agg_poolmean_lr1_full_router_per_layer.json) | `uttt_moe` | 12 × 768 | 32,768 | 4096 | 48 / 1 | loss_free / 1e-05 |
| [balancing/124M/uttt_moe_e48a1_lossfree_pool_u3e-4_agg_poolmean_lr1_full_router_per_layer.json](../uttt_llm/configs/balancing/124M/uttt_moe_e48a1_lossfree_pool_u3e-4_agg_poolmean_lr1_full_router_per_layer.json) | `uttt_moe` | 12 × 768 | 32,768 | 4096 | 48 / 1 | loss_free / 0.0003 |
| [balancing/760M/ttt_moe_lb_e4a1_lossfree_u1e-4.json](../uttt_llm/configs/balancing/760M/ttt_moe_lb_e4a1_lossfree_u1e-4.json) | `ttt_moe_lb` | 24 × 1536 | 32,768 | 4096 | 4 / 1 | loss_free / 0.0001 |
| [balancing/760M/ttt_moe_lb_e4a1_lossfree_u3e-5.json](../uttt_llm/configs/balancing/760M/ttt_moe_lb_e4a1_lossfree_u3e-5.json) | `ttt_moe_lb` | 24 × 1536 | 32,768 | 4096 | 4 / 1 | loss_free / 3e-05 |
| [balancing/760M/ttt_moe_no_lb_e4a1_lb1e-4.json](../uttt_llm/configs/balancing/760M/ttt_moe_no_lb_e4a1_lb1e-4.json) | `ttt_moe_no_lb` | 24 × 1536 | 32,768 | 4096 | 4 / 1 | default / — |
| [balancing/760M/ttt_moe_no_lb_e4a1_lb1e-5.json](../uttt_llm/configs/balancing/760M/ttt_moe_no_lb_e4a1_lb1e-5.json) | `ttt_moe_no_lb` | 24 × 1536 | 32,768 | 4096 | 4 / 1 | default / — |
| [balancing/760M/ttt_moe_no_lb_e4a1_lb3e-5.json](../uttt_llm/configs/balancing/760M/ttt_moe_no_lb_e4a1_lb3e-5.json) | `ttt_moe_no_lb` | 24 × 1536 | 32,768 | 4096 | 4 / 1 | default / — |
| [balancing/760M/ttt_moe_no_lb_e4a2_lb0.json](../uttt_llm/configs/balancing/760M/ttt_moe_no_lb_e4a2_lb0.json) | `ttt_moe_no_lb` | 24 × 1536 | 32,768 | 4096 | 4 / 2 | default / — |
| [balancing/760M/ttt_moe_no_lb_e8a1_lb0.json](../uttt_llm/configs/balancing/760M/ttt_moe_no_lb_e8a1_lb0.json) | `ttt_moe_no_lb` | 24 × 1536 | 32,768 | 4096 | 8 / 1 | default / — |
| [balancing/760M/uttt_moe_e96a1_lossfree_pool_u1e-4_agg_poolmean_lr1_full_router_per_layer.json](../uttt_llm/configs/balancing/760M/uttt_moe_e96a1_lossfree_pool_u1e-4_agg_poolmean_lr1_full_router_per_layer.json) | `uttt_moe` | 24 × 1536 | 32,768 | 4096 | 96 / 1 | loss_free / 0.0001 |
| [balancing/760M/uttt_moe_e96a1_lossfree_pool_u1e-4_alpha2_agg_poolmean_lr1_full_router_per_layer.json](../uttt_llm/configs/balancing/760M/uttt_moe_e96a1_lossfree_pool_u1e-4_alpha2_agg_poolmean_lr1_full_router_per_layer.json) | `uttt_moe` | 24 × 1536 | 32,768 | 4096 | 96 / 1 | loss_free / 0.0001 |
| [balancing/760M/uttt_moe_e96a1_lossfree_pool_u1e-4_alpha4_agg_poolmean_lr1_full_router_per_layer.json](../uttt_llm/configs/balancing/760M/uttt_moe_e96a1_lossfree_pool_u1e-4_alpha4_agg_poolmean_lr1_full_router_per_layer.json) | `uttt_moe` | 24 × 1536 | 32,768 | 4096 | 96 / 1 | loss_free / 0.0001 |
| [balancing/760M/uttt_moe_e96a1_lossfree_pool_u1e-5_agg_poolmean_lr1_full_router_per_layer.json](../uttt_llm/configs/balancing/760M/uttt_moe_e96a1_lossfree_pool_u1e-5_agg_poolmean_lr1_full_router_per_layer.json) | `uttt_moe` | 24 × 1536 | 32,768 | 4096 | 96 / 1 | loss_free / 1e-05 |
| [balancing/760M/uttt_moe_e96a1_lossfree_pool_u3e-4_agg_poolmean_lr1_full_router_per_layer.json](../uttt_llm/configs/balancing/760M/uttt_moe_e96a1_lossfree_pool_u3e-4_agg_poolmean_lr1_full_router_per_layer.json) | `uttt_moe` | 24 × 1536 | 32,768 | 4096 | 96 / 1 | loss_free / 0.0003 |
| [balancing/760M/uttt_moe_e96a1_lossfree_pool_u3e-4_alpha2_agg_poolmean_lr1_full_router_per_layer.json](../uttt_llm/configs/balancing/760M/uttt_moe_e96a1_lossfree_pool_u3e-4_alpha2_agg_poolmean_lr1_full_router_per_layer.json) | `uttt_moe` | 24 × 1536 | 32,768 | 4096 | 96 / 1 | loss_free / 0.0003 |
| [balancing/760M/uttt_moe_e96a1_lossfree_pool_u3e-5_agg_poolmean_lr1_full_router_per_layer.json](../uttt_llm/configs/balancing/760M/uttt_moe_e96a1_lossfree_pool_u3e-5_agg_poolmean_lr1_full_router_per_layer.json) | `uttt_moe` | 24 × 1536 | 32,768 | 4096 | 96 / 1 | loss_free / 3e-05 |
| [balancing/760M/uttt_moe_e96a1_lossfree_pool_u3e-5_alpha2_agg_poolmean_lr1_full_router_per_layer.json](../uttt_llm/configs/balancing/760M/uttt_moe_e96a1_lossfree_pool_u3e-5_alpha2_agg_poolmean_lr1_full_router_per_layer.json) | `uttt_moe` | 24 × 1536 | 32,768 | 4096 | 96 / 1 | loss_free / 3e-05 |
| [main/124M/deltanet_swa.json](../uttt_llm/configs/main/124M/deltanet_swa.json) | `deltanet_swa` | 12 × 768 | 32,768 | — | — | default / — |
| [main/124M/gated_deltanet_swa.json](../uttt_llm/configs/main/124M/gated_deltanet_swa.json) | `gated_deltanet_swa` | 12 × 768 | 32,768 | — | — | default / — |
| [main/124M/lact.json](../uttt_llm/configs/main/124M/lact.json) | `lact_swiglu` | 12 × 768 | 32,768 | — | — | default / — |
| [main/124M/transformer.json](../uttt_llm/configs/main/124M/transformer.json) | `transformer` | 12 × 768 | 32,768 | — | — | default / — |
| [main/124M/transformer_swa.json](../uttt_llm/configs/main/124M/transformer_swa.json) | `transformer` | 12 × 768 | 32,768 | — | — | default / — |
| [main/124M/ttt_dense.json](../uttt_llm/configs/main/124M/ttt_dense.json) | `ttt_dense` | 12 × 768 | 32,768 | 4096 | — | default / — |
| [main/124M/ttt_moe_lb.json](../uttt_llm/configs/main/124M/ttt_moe_lb.json) | `ttt_moe_lb` | 12 × 768 | 32,768 | 4096 | 4 / 1 | loss_free / 1e-05 |
| [main/124M/ttt_moe_no_lb.json](../uttt_llm/configs/main/124M/ttt_moe_no_lb.json) | `ttt_moe_no_lb` | 12 × 768 | 32,768 | 4096 | 4 / 1 | default / — |
| [main/124M/uttt_moe.json](../uttt_llm/configs/main/124M/uttt_moe.json) | `uttt_moe` | 12 × 768 | 32,768 | 4096 | 48 / 1 | loss_free / 3e-05 |
| [main/760M/deltanet_swa.json](../uttt_llm/configs/main/760M/deltanet_swa.json) | `deltanet_swa` | 24 × 1536 | 32,768 | — | — | default / — |
| [main/760M/gated_deltanet_swa.json](../uttt_llm/configs/main/760M/gated_deltanet_swa.json) | `gated_deltanet_swa` | 24 × 1536 | 32,768 | — | — | default / — |
| [main/760M/lact.json](../uttt_llm/configs/main/760M/lact.json) | `lact_swiglu` | 24 × 1536 | 32,768 | — | — | default / — |
| [main/760M/transformer.json](../uttt_llm/configs/main/760M/transformer.json) | `transformer` | 24 × 1536 | 32,768 | — | — | default / — |
| [main/760M/transformer_swa.json](../uttt_llm/configs/main/760M/transformer_swa.json) | `transformer` | 24 × 1536 | 32,768 | — | — | default / — |
| [main/760M/ttt_dense.json](../uttt_llm/configs/main/760M/ttt_dense.json) | `ttt_dense` | 24 × 1536 | 32,768 | 4096 | — | default / — |
| [main/760M/ttt_moe_lb.json](../uttt_llm/configs/main/760M/ttt_moe_lb.json) | `ttt_moe_lb` | 24 × 1536 | 32,768 | 4096 | 4 / 1 | loss_free / 1e-05 |
| [main/760M/ttt_moe_no_lb.json](../uttt_llm/configs/main/760M/ttt_moe_no_lb.json) | `ttt_moe_no_lb` | 24 × 1536 | 32,768 | 4096 | 4 / 1 | default / — |
| [main/760M/uttt_moe.json](../uttt_llm/configs/main/760M/uttt_moe.json) | `uttt_moe` | 24 × 1536 | 32,768 | 4096 | 96 / 1 | loss_free / 0.001 |

## Language-model jobs

| Config | Per-device batch | Accumulation | Expected global batch | Sequence length | Steps |
|---|---:|---:|---:|---:|---:|
| [eval_ptl.toml](../uttt_llm/configs/exp/eval_ptl.toml) | 1 | 1 | not enforced | 32,768 | 1 |
| [train_124M_32k.toml](../uttt_llm/configs/exp/train_124M_32k.toml) | 4 | 1 | 32 | 32,768 | 10,240 |
| [train_760M_32k.toml](../uttt_llm/configs/exp/train_760M_32k.toml) | 2 | 1 | 32 | 32,768 | 40,960 |
