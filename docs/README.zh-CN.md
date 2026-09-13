# uTTT 中文使用指南

本仓库研究的是 **fast weights 在网络深度上的所有权**：从每层独占的
TTT-Dense / LaCT，到每层独立专家池 TTT-MoE，再到跨层共享的
uTTT-Dense / uTTT-MoE。语言建模与新视角合成（NVS）使用不同实现，
不能仅靠更改模型名字把一个实验变成另一个实验。

## 仓库关系

- [uTTT-DEV](https://github.com/Zefan-Cai/uTTT-DEV)：原仓库改名，保留原有身份和历史。
- [uTTT](https://github.com/Zefan-Cai/uTTT)：全新的升级仓库，全部代码和文档重新提交为
  一个初始提交，不继承旧仓库的提交历史。
- 旧 clone 的 `/uTTT.git` 地址现在指向新仓库；继续开发原版时需要主动改成
  `/uTTT-DEV.git`。详见[迁移说明](migration.md)。

升级重点是运行可靠性、测试、可复现性和文档，不是未经验证地更改算法。
原模型实现、实验配置、论文结果图片保持不变；本次没有新增性能或精度结论。

## 先选入口

| 目标 | 入口 |
|---|---|
| 读懂状态共享、路由及更新顺序 | [架构与源码地图](architecture.md) |
| 查全部实验配置 | [配置语义](configuration.md)、[自动生成的完整索引](configuration-catalog.md) |
| 不使用 GPU 验证仓库 | [安装与第一步](getting-started.md) |
| 准备数据 | [数据契约](data-contracts.md) |
| 单机、多机训练及恢复 | [训练指南](training.md) |
| 复现 NVS / PTL / RULER 指标 | [评估协议](evaluation.md) |
| 排错或提交修改 | [排错](troubleshooting.md)、[测试](testing.md)、[贡献指南](../CONTRIBUTING.md) |

## 无 GPU 时也能完成的检查

从仓库根目录运行。建议使用 Python 3.10 或 3.11；3.10 还需要 `tomli`。

```bash
python -m pip install pyyaml 'tomli; python_version < "3.11"'
python -m tools.check_configs --check-catalog
bash uttt_nvs/train/launch.sh --dry-run \
  uttt_nvs/configs/ownership/obj/uttt_moe_e64a1.yaml
bash uttt_llm/launch/train.sh --dry-run \
  uttt_llm/configs/exp/train_124M_32k.toml \
  uttt_llm/configs/main/124M/uttt_moe.json first_run
```

静态检查覆盖 **64 份 NVS YAML、57 份 LLM JSON、3 份任务 TOML**。
检查维度整除、专家数量、共享池约束、batch 算术及源码类名，不导入 CUDA 模型。
`--dry-run` 只校验启动参数并打印命令，不启动 GPU 进程，也不运行 `NODE_SYNC`。

安装 CPU 版 torch 2.8.0 和 `requirements-dev.txt` 后，可运行 `python -m pytest -q`。
Linux CPU wheel 的安装方式见英文安装指南；macOS 使用普通 PyTorch wheel。
CPU 测试通过不代表 CUDA kernel、多机通信或论文指标已经验证。

## NVS 数据与首次运行

生成一个只用于打通流程的合成数据集：

```bash
python -m uttt_nvs.data.make_toy_dataset --out /tmp/uttt_toy
python -m tools.check_manifest /tmp/uttt_toy/manifest.txt --check-images
```

默认 16 个场景，每场景 24 张图片，manifest 重复 16 次得到 256 个样本条目。
生成器使用绝对路径和右手 OpenCV 相机坐标系；这些图片没有可用于报告质量的
真实多视角结构。将输出路径写入本地、Git 忽略的 `uttt_nvs/datasets.yaml`：

```yaml
obj:
  train: /tmp/uttt_toy/manifest.txt
  eval: /tmp/uttt_toy/manifest.txt
```

训练集与测试集共用 manifest **仅限合成流程测试**。真实实验必须使用正确、
分离的训练和评估数据。相机 JSON 需包含 `frames`，每个 frame 都要有像素
内参、4×4 `w2c` 及相对相机 JSON 目录的图片路径。

在参考 CUDA 环境安装好 NVS 依赖后，可运行明确限制步数的检查：

```bash
NPROC_PER_NODE=4 bash uttt_nvs/train/launch.sh \
  uttt_nvs/configs/ownership/obj/uttt_moe_e64a1.yaml \
  -s exp_name toy_smoke \
  -s training.max_fwdbwd_passes 5 \
  -s training.round_max_fwdbwd_passes_to_epoch False \
  -s training.wandb_offline True
```

只设置 `max_fwdbwd_passes=5` 不够：默认会按 epoch 调整长度，所以还需关闭
取整。首次执行可能编译和 autotune Triton，不能拿第一步估算稳定吞吐。

## 多机与 batch 计算

NVS：`每卡 batch × 总进程数 × 梯度累积 = total_batch_size`。
LLM：`每设备 batch × 数据并行度 × 梯度累积 = expected_global_batch_size`。

- NVS 标准配置：32 × 4 × 1 = 128。
- LLM 124M：4 × 8 × 1 = 32；四卡时可用累积 2 保持全局 batch。
- LLM 760M：2 × 16 × 1 = 32。
- 多机的 `WORLD_SIZE` 是**总进程数**，不是节点数；两台八卡机应填 16。
- `NODE_RANK` 是从 0 开始的节点编号，优先于兼容字段 `RANK`。
- 各节点使用相同 `MASTER_ADDR`、`MASTER_PORT` 和 `JOB_UUID`。

保持 batch 不等于保持完整实验：改变 NVS 累积步数后，同样的前后向次数会
对应不同的 optimizer update 次数。训练日程也需要核对，不能静默更改。

## Checkpoint 恢复的关键变化

NVS checkpoint 先写临时文件，再原子替换最终文件；恢复按数值 step 排序。
最新文件损坏时可尝试更早文件，全部不可读或显式路径不存在时会报错，
不会悄悄从随机权重开始。

恢复优先级是：已有输出目录 → 显式 `--load` → 配置 `training.load` → 从头训练。
换训练阶段请使用新的 `exp_name`，否则自动恢复会优先于 `--load`。
重置训练状态时保留模型权重，但把 optimizer/scheduler/计数器重置；
计数器为零不再导致继续加载第二个来源。

`save_last_n_ckpts` 现在表示**保留文件数量**，默认 3，不再表示 step 差值。
旧配置若填了 1001，需要重新检查磁盘占用含义。LLM 使用 DCP checkpoint，
不是 NVS 的 `.pt` 格式，评估前需按领域文档转为 HF 格式。

## 评估不能省略的细节

- NVS 固定 24 个 view，评估 1–23，采用逐步累积真实条件视图的 teacher forcing；
  不是只给第 0 张图就一次预测所有后续视图。
- 总体指标先在场景内平均，再在场景间平均；95% 区间按场景 bootstrap。
- PTL 使用匹配 tokenizer 的数据和正确 replica degree，保留分块 `lm_head`
  设置以避免完整 logits 占满显存。每行最后一个位置没有 next-token 标签，NaN 正常。
- RULER 要保持任务缓存、长度、dtype、生成参数与覆盖范围一致。
- 仓库不提供训练好的 checkpoint、原始数据或已填好的评估 registry。

记录 Git SHA、全部配置与 override、数据/分词器版本、GPU 拓扑、随机种子、
checkpoint step 和有效评估样本数量。不要把 token、W&B key 或私有数据提交到 Git。
