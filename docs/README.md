# uTTT documentation

This guide explains how to read, run and extend the release. The original
experimental implementations and configurations remain in their domain
directories; the tools here make the release easier to inspect and validate.

## Choose your path

| Your goal | Start here | What you can verify |
|---|---|---|
| Understand the method | [Architecture](architecture.md) | State ownership, routing, update schedules and source entry points |
| Run something without a GPU | [Getting started](getting-started.md) | Configuration checks, synthetic data and launch dry runs |
| Pick an experiment | [Configuration semantics](configuration.md), [complete catalog](configuration-catalog.md) | All 64 NVS YAMLs, 57 LLM JSONs and 3 job TOMLs |
| Prepare data | [Data contracts](data-contracts.md) | Camera geometry, manifests, Arrow shards and split separation |
| Train or resume | [Training and checkpoints](training.md) | Batch arithmetic, rendezvous, exact stopping and restore precedence |
| Reproduce metrics | [Evaluation](evaluation.md) | Conditioning protocol, aggregation units and evidence to retain |
| Debug a failure | [Troubleshooting](troubleshooting.md) | Environment, startup, data, checkpoint and evaluation failures |
| Contribute a change | [Contributing](../CONTRIBUTING.md), [testing](testing.md) | CPU vs GPU coverage and reproducibility requirements |
| Understand the repository split | [Migration](migration.md) | Repository identities, history and remote URLs |
| Read in Chinese | [中文使用指南](README.zh-CN.md) | 安装、验证、数据、训练、恢复及评估要点 |

## Source of truth

1. A checked-in config defines an experiment's architecture and hyperparameters.
2. The domain implementation determines how those fields are interpreted.
3. The generated catalog summarizes literal config values; it is not a second
   configuration system and does not override the files.
4. A passing CPU check is not evidence that a model fits in GPU memory, that a
   fused kernel is numerically correct, or that a published metric was reproduced.

The release contains code, configuration files and figures, not trained
checkpoints, licensed datasets or a completed metric registry. Fill in your own
paths and run the GPU validation ladder before scheduling a large experiment.

## Domain references

- [Novel view synthesis](../uttt_nvs/README.md): model families and scale ladders.
- [NVS data preparation](../uttt_nvs/data/README.md): local, S3 and GCS manifests.
- [NVS full-test protocol](../uttt_nvs/eval/README.md): fixed views and result schema.
- [Language modeling](../uttt_llm/README.md): training, DCP conversion, PTL and RULER.
- [Project page](index.html): the research overview and existing result figures.
