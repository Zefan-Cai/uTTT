# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import time
from datetime import timedelta

import torch
import torch.nn.functional as F
from datasets import interleave_datasets, load_dataset
from safetensors.torch import load_file as load_safetensors_file
from torch.distributed.elastic.multiprocessing.errors import record
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import fla  # noqa
from fla.modules.fused_linear_cross_entropy import FusedLinearCrossEntropyLoss
from fla.ops.utils import prepare_position_ids
from flame.components.checkpoint import TrainState
from flame.config_manager import JobConfig
from flame.data import build_dataloader, shuffle
from flame.models.parallelize_fla import parallelize_fla
from flame.models.pipeline_fla import pipeline_fla
from flame.tools.utils import get_nparams_and_flops
from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.ft import FTParallelDims, init_ft_manager
from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.metrics import build_device_memory_monitor, build_metrics_processor, ensure_pp_loss_visible
from flame.components.optimizer import build_optimizers
from torchtitan.distributed import ParallelDims
from torchtitan.distributed import utils as dist_utils
from torchtitan.protocols.model_converter import build_model_converters
from torchtitan.protocols.train_spec import TrainSpec, get_train_spec, register_train_spec
from torchtitan.tools import utils
from torchtitan.tools.logging import init_logger, logger
from torchtitan.tools.profiling import maybe_enable_memory_snapshot, maybe_enable_profiling
import flame.custom_models as custom_models
import torch.distributed as dist
import wandb
from fla.modules import FusedCrossEntropyLoss, FusedLinearCrossEntropyLoss
from typing import Optional
from flame.utils.per_position_loss_tracker import AverageTracker
def build_tokenizer(job_config: JobConfig) -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(job_config.model.tokenizer_path)


register_train_spec(
    TrainSpec(
        name="fla",
        cls=AutoModelForCausalLM,
        config=AutoConfig,
        parallelize_fn=parallelize_fla,
        pipelining_fn=pipeline_fla,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_dataloader,
        build_tokenizer_fn=build_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
    )
)


def mesh_mean(t, group):
    """
    Element‑wise average of `t` across `group`.
    Keeps gradient flow if `t` still requires_grad.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return t

    t = t.clone()                        # keep autograd graph intact
    dist.all_reduce(t, op=dist.ReduceOp.SUM, group=group)
    t /= dist.get_world_size(group)
    return t


def _is_llama_family_config(model_config) -> bool:
    model_type = str(getattr(model_config, "model_type", ""))
    return model_type.startswith(("custom_llama", "custom_recurrent_llama"))


def _is_recurrent_llama_config(model_config) -> bool:
    model_type = str(getattr(model_config, "model_type", ""))
    return model_type.startswith("custom_recurrent_llama")


def _local_weight_files(model_load_path: str) -> list[str]:
    index_path = os.path.join(model_load_path, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {})
        return sorted({os.path.join(model_load_path, name) for name in weight_map.values()})

    safetensors_path = os.path.join(model_load_path, "model.safetensors")
    if os.path.exists(safetensors_path):
        return [safetensors_path]

    pytorch_path = os.path.join(model_load_path, "pytorch_model.bin")
    if os.path.exists(pytorch_path):
        return [pytorch_path]

    raise FileNotFoundError(f"No local model weights found in {model_load_path}")


def _load_model_from_config_and_local_weights(model_load_path: str, model_config):
    logger.warning(
        "Loading recurrent llama weights with from_config + load_state_dict; "
        "Transformers from_pretrained can report success while leaving these "
        "custom model weights randomly initialized."
    )
    model = AutoModelForCausalLM.from_config(model_config, trust_remote_code=True)
    model.to(dtype=torch.bfloat16)

    loaded_keys: set[str] = set()
    unexpected_keys: set[str] = set()
    for weight_file in _local_weight_files(model_load_path):
        if weight_file.endswith(".safetensors"):
            state_dict = load_safetensors_file(weight_file, device="cpu")
        else:
            state_dict = torch.load(weight_file, map_location="cpu", weights_only=False)
        loaded_keys.update(state_dict.keys())
        incompatible = model.load_state_dict(state_dict, strict=False)
        unexpected_keys.update(incompatible.unexpected_keys)
        del state_dict

    model.tie_weights()
    model_keys = set(model.state_dict().keys())
    missing_keys = model_keys - loaded_keys
    tied_keys = getattr(model, "_tied_weights_keys", {})
    if getattr(model.config, "tie_word_embeddings", False) and isinstance(tied_keys, dict):
        missing_keys = {
            key
            for key in missing_keys
            if key not in tied_keys or tied_keys[key] not in loaded_keys
        }
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            "Manual recurrent llama weight load mismatch: "
            f"missing={sorted(missing_keys)[:20]} "
            f"unexpected={sorted(unexpected_keys)[:20]}"
        )
    return model


def _per_token_loss_from_hidden_states(
    model,
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-position CE without materializing the full [B, S, V] logits tensor."""
    if chunk_size < 1:
        raise ValueError("eval.lm_head_chunk_size must be a positive integer")

    output_embeddings = model.get_output_embeddings()
    if output_embeddings is None:
        raise ValueError("Model does not expose output embeddings for chunked per-token loss")

    labels = torch.cat(
        (labels[..., 1:], torch.full_like(labels[:, :1], ignore_index)),
        dim=1,
    )
    bs, seq_len = labels.shape
    per_token_loss = torch.empty((bs, seq_len), device=hidden_states.device, dtype=torch.float32)

    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        logits = output_embeddings(hidden_states[:, start:end, :]).float()
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels[:, start:end].reshape(-1),
            ignore_index=ignore_index,
            reduction="none",
        )
        per_token_loss[:, start:end] = loss.view(bs, end - start)

    valid_labels = labels.ne(ignore_index)
    loss = per_token_loss.masked_select(valid_labels).mean()
    # cross_entropy returns 0 for ignore_index positions, so the per-position
    # curve must average only over valid labels. Otherwise the final position
    # (whose label is the shift-in ignore_index) contributes a spurious 0 and
    # drags down every tail mean that includes it by roughly loss/window.
    valid_counts = valid_labels.sum(dim=0)
    per_position = torch.where(
        valid_counts > 0,
        per_token_loss.masked_fill(~valid_labels, 0.0).sum(dim=0)
        / valid_counts.clamp(min=1),
        torch.full_like(per_token_loss[0], float("nan")),
    )
    return loss, per_position


# Enable debug tracing on failure: https://pytorch.org/docs/stable/elastic/errors.html
@record
def main(job_config: JobConfig):
    logger.info(f"Starting job: {job_config.job.description}")

    if job_config.experimental.custom_model_path:
        utils.import_module_from_path(job_config.experimental.custom_model_path)

    # used for colorful printing
    color = utils.NoColor if job_config.metrics.disable_color_printing else utils.Color

    if job_config.job.print_args:
        logger.info(
            f"{color.green}{json.dumps(job_config.to_dict(), indent=2, sort_keys=True)}{color.reset}"
        )

    # take control of garbage collection to avoid stragglers
    gc_handler = utils.GarbageCollection(gc_freq=job_config.training.gc_freq)

    device_module, device_type = utils.device_module, utils.device_type
    device = torch.device(f"{device_type}:{int(os.environ['LOCAL_RANK'])}")
    # Device has to be set before creating TorchFT manager.
    device_module.set_device(device)
    ft_manager = init_ft_manager(job_config)

    # init distributed
    world_size = int(os.environ["WORLD_SIZE"])
    if not ft_manager.enabled:
        parallel_dims = ParallelDims(
            dp_shard=job_config.training.data_parallel_shard_degree,
            dp_replicate=job_config.training.data_parallel_replicate_degree,
            cp=job_config.experimental.context_parallel_degree,
            tp=job_config.training.tensor_parallel_degree,
            pp=job_config.experimental.pipeline_parallel_degree,
            world_size=world_size,
            enable_loss_parallel=not job_config.training.disable_loss_parallel,
        )
    else:
        parallel_dims = FTParallelDims(
            dp_shard=job_config.training.data_parallel_shard_degree,
            dp_replicate=job_config.training.data_parallel_replicate_degree,
            cp=job_config.experimental.context_parallel_degree,
            tp=job_config.training.tensor_parallel_degree,
            pp=job_config.experimental.pipeline_parallel_degree,
            world_size=world_size,
            enable_loss_parallel=not job_config.training.disable_loss_parallel,
            ft_manager=ft_manager,
        )
    dist_utils.init_distributed(job_config)
    # initialize device memory monitor and get peak flops for MFU calculation
    device_memory_monitor = build_device_memory_monitor()
    gpu_peak_flops = utils.get_peak_flops(device_memory_monitor.device_name)
    logger.info(f"Peak FLOPS used for computing MFU: {gpu_peak_flops:.3e}")

    # build meshes
    world_mesh = parallel_dims.build_mesh(device_type=device_type)
    if parallel_dims.dp_enabled:
        dp_mesh = world_mesh["dp"]
        dp_degree, dp_rank = dp_mesh.size(), dp_mesh.get_local_rank()
    else:
        dp_degree, dp_rank = 1, 0


    # Set random seed, and maybe enable deterministic mode (mainly for debugging, expect perf loss)
    dist_utils.set_determinism(
        world_mesh, device, job_config.training.seed, job_config.training.deterministic
    )
    train_spec = get_train_spec(job_config.model.name)

    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        job_config.model.tokenizer_path,
        trust_remote_code=True,
        model_max_length=int(1e10),
    )
    logger.info(f"{tokenizer}")
    logger.info(
        f"Loading dataset {job_config.training.dataset}"
        f":{job_config.training.dataset_name}"
        if job_config.training.dataset_name is not None
        else ""
    )

    min_num_shards = dp_degree * job_config.training.num_workers
    if len(job_config.training.dataset.split(",")) == 1:
        dataset = load_dataset(
            path=job_config.training.dataset,
            name=getattr(job_config.training, "dataset_name", None),
            data_dir=getattr(job_config.training, "data_dir", None),
            data_files=getattr(job_config.training, "data_files", None),
            split=job_config.training.dataset_split or "train",
            trust_remote_code=True,
            streaming=job_config.training.streaming,
            num_proc=(
                job_config.training.num_workers
                if not job_config.training.streaming
                else None
            ),
        )
        logger.info(f"{dataset}")

        logger.info(f"Shuffling the dataset with seed {job_config.training.seed}")
        if not job_config.training.streaming:
            # the states of map-style dataset is recoverable after shuffling
            dataset = dataset.shuffle(
                seed=job_config.training.seed
            ).to_iterable_dataset(num_shards=min_num_shards)
        else:
            if dataset.num_shards < min_num_shards:
                logger.warning(
                    f"{color.red}"
                    f"Dataset {job_config.training.dataset} has insufficient shards ({dataset.num_shards}). "
                    f"Need {min_num_shards} shards minimum for {dp_degree} data parallel workers × "
                    f"{job_config.training.num_workers} dataloader workers. "
                    f"Disabling the streaming mode and resharding dataset to {min_num_shards} shards."
                    f"{color.reset}"
                )
                dataset = (
                    load_dataset(
                        path=job_config.training.dataset,
                        name=getattr(job_config.training, "dataset_name", None),
                        data_dir=getattr(job_config.training, "data_dir", None),
                        data_files=getattr(job_config.training, "data_files", None),
                        split=job_config.training.dataset_split or "train",
                        trust_remote_code=True,
                        streaming=False,
                        num_proc=job_config.training.num_workers,
                    )
                    .shuffle(seed=job_config.training.seed)
                    .to_iterable_dataset(num_shards=min_num_shards)
                )
            else:
                dataset = shuffle(dataset, seed=job_config.training.seed)
    else:
        datasets = job_config.training.dataset.split(",")
        if job_config.training.dataset_name is not None:
            dataset_names = [
                name or None for name in job_config.training.dataset_name.split(",")
            ]
            assert len(dataset_names) == len(datasets), (
                "The number of dataset names must match the number of datasets"
            )
        else:
            dataset_names = [None] * len(datasets)
        if job_config.training.dataset_split is not None:
            dataset_splits = [
                split or "train"
                for split in job_config.training.dataset_split.split(",")
            ]
            assert len(dataset_splits) == len(datasets), (
                "The number of dataset splits must match the number of datasets"
            )
        else:
            dataset_splits = ["train"] * len(datasets)
        if job_config.training.data_dir is not None:
            data_dirs = [
                data_dir or None for data_dir in job_config.training.data_dir.split(",")
            ]
            assert len(data_dirs) == len(datasets), (
                "The number of data dirs must match the number of datasets"
            )
        else:
            data_dirs = [None] * len(datasets)
        if job_config.training.data_files is not None:
            data_files = job_config.training.data_files.split(",")
            assert len(data_files) == len(datasets), (
                "The number of data files must match the number of datasets"
            )
        else:
            data_files = [None] * len(datasets)
        if job_config.training.data_probs is not None:
            data_probs = [float(p) for p in job_config.training.data_probs.split(",")]
            assert len(data_probs) == len(datasets), (
                "The number of data probabilities must match the number of datasets"
            )
        else:
            raise ValueError(
                "Data sampling probabilities are required if using multiple datasets"
            )

        subsets = []
        for i, prob in enumerate(data_probs):
            subset = load_dataset(
                path=datasets[i],
                name=dataset_names[i],
                data_dir=data_dirs[i],
                data_files=data_files[i],
                split=dataset_splits[i],
                trust_remote_code=True,
                streaming=job_config.training.streaming,
                num_proc=(
                    job_config.training.num_workers
                    if not job_config.training.streaming
                    else None
                ),
            )
            logger.info(
                f"Subset {color.cyan}{datasets[i]}"
                + (f":{dataset_names[i]} " if dataset_names[i] else " ")
                + f"(p = {prob:.3f}){color.reset}:\n"
                + f"{subset}"
            )

            logger.info(f"Shuffling the dataset with seed {job_config.training.seed}")
            if not job_config.training.streaming:
                # the states of map-style dataset is recoverable after shuffling
                subset = subset.shuffle(
                    seed=job_config.training.seed
                ).to_iterable_dataset(num_shards=min_num_shards)
            else:
                if subset.num_shards < min_num_shards:
                    logger.warning(
                        f"{color.red}"
                        f"Dataset {datasets[i]} has insufficient shards ({subset.num_shards}). "
                        f"Need {min_num_shards} shards minimum for {dp_degree} data parallel workers × "
                        f"{job_config.training.num_workers} dataloader workers. "
                        f"Resharding dataset to {min_num_shards} shards and disabling streaming mode."
                        f"{color.reset}"
                    )
                    # again, it's ok to directly shuffle the map-style dataset
                    # we expect an error raised if the map-style dataset still has not enough data shards
                    subset = (
                        load_dataset(
                            path=datasets[i],
                            name=dataset_names[i],
                            data_dir=data_dirs[i],
                            data_files=data_files[i],
                            split=dataset_splits[i],
                            trust_remote_code=True,
                            streaming=False,
                            num_proc=job_config.training.num_workers,
                        )
                        .shuffle(seed=job_config.training.seed)
                        .to_iterable_dataset(min_num_shards)
                    )
                else:
                    # we set relatively small buffer size here as interleaving could provide some randomness
                    subset = shuffle(
                        subset,
                        seed=job_config.training.seed,
                        buffer_size=max(128, 1024 // len(datasets)),
                    )

            if "text" in subset.column_names:
                subset = subset.select_columns("text")
            elif "content" in subset.column_names:
                subset = subset.select_columns("content")
            else:
                raise ValueError(
                    f"Subset {datasets[i]} has no 'text' or 'content' column"
                )
            subsets.append(subset)

        logger.info(
            f"Interleaving {len(subsets)} datasets with probabilities {data_probs}"
        )
        dataset = interleave_datasets(
            datasets=subsets,
            probabilities=data_probs,
            stopping_strategy="all_exhausted",
            seed=job_config.training.seed,
        )
        logger.info(f"{dataset}")

    logger.info(f"Building dataloader...for rank: {dp_rank} of dp_degree: {dp_degree} or world_size: {world_size}")
    dataloader = build_dataloader(
        dataset=dataset,
        tokenizer=tokenizer,
        rank=dp_rank,
        world_size=dp_degree,
        batch_size=job_config.training.batch_size,
        seq_len=job_config.training.seq_len,
        context_len=job_config.training.context_len,
        varlen=job_config.training.varlen,
        num_workers=job_config.training.num_workers,
        pin_memory=job_config.training.pin_memory,
        persistent_workers=job_config.training.persistent_workers,
        snapshot_every_n_steps=job_config.checkpoint.interval,
    )

    logger.info(f"Loading model config from {job_config.model.config}")
    model_config = AutoConfig.from_pretrained(job_config.model.config)
    # set the model configs from training inputs:
    # 1. norm type to decide which norm layer to use
    # 2. disable fused norm if TP is enabled
    # 3. vocab size from tokenizer
    # 4. context_len base on inputs
    if parallel_dims.tp_enabled:
        if model_config.fuse_norm:
            logger.warning(
                f"{color.red}"
                f"Fused norm is not compatible with tensor parallelism. "
                f"Disabling it for now."
                f"{color.reset}"
            )
            model_config.fuse_norm = False
    if parallel_dims.loss_parallel_enabled:
        if model_config.fuse_cross_entropy:
            logger.warning(
                f"{color.red}"
                f"Loss parallel enabled. Disabling fused cross entropy for now."
                f"{color.reset}"
            )
            model_config.fuse_cross_entropy = False
    model_config.vocab_size = max(tokenizer.vocab_size, model_config.vocab_size)

    config_dir = os.path.dirname(job_config.model.config)

    # Use checkpoint path if provided, otherwise use config directory
    if hasattr(job_config.checkpoint, 'load_path') and job_config.checkpoint.load_path is not None:
        model_load_path = job_config.checkpoint.load_path
        logger.info(f"Loading model from checkpoint path: {model_load_path}")
    else:
        model_load_path = config_dir
        logger.info(f"Loading model from config directory: {model_load_path}")

    converted_config_path = os.path.join(model_load_path, "config.json")
    if os.path.exists(converted_config_path) and os.path.abspath(model_load_path) != os.path.abspath(config_dir):
        logger.info(f"Loading converted model config from checkpoint path: {model_load_path}")
        model_config = AutoConfig.from_pretrained(model_load_path)
        model_config.vocab_size = max(tokenizer.vocab_size, model_config.vocab_size)

    save_prefix = job_config.training.save_prefix
    report_save_path = os.path.join(job_config.job.dump_folder, f"{save_prefix}_loss_per_token_position.json")
    use_hidden_state_per_token_loss = (
        getattr(job_config.eval, "per_token_loss_from_hidden", False)
        or _is_llama_family_config(model_config)
    )
    lm_head_chunk_size = getattr(job_config.eval, "lm_head_chunk_size", 512)
    if use_hidden_state_per_token_loss:
        logger.info(
            "Using hidden-state chunked lm_head path for per-token loss "
            f"(chunk_size={lm_head_chunk_size}, model_type={model_config.model_type})"
        )

    # assert that there are model.safetensors files in the model_load_path
    # assert os.path.exists(os.path.join(model_load_path, "model.safetensors")), "model.safetensors file not found in the model load path"

    logger.info(
        f"Building model from the config\n{color.green}{model_config}{color.reset}"
    )

    # Time the model loading process
    model_load_start = time.perf_counter()
    if _is_recurrent_llama_config(model_config):
        model = _load_model_from_config_and_local_weights(model_load_path, model_config)
    else:
        # AutoModelForCausalLM.from_pretrained will automatically load the safetensors if present.
        model = AutoModelForCausalLM.from_pretrained(
            model_load_path,
            config=model_config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )
    model_load_time = time.perf_counter() - model_load_start
    logger.info(f"Model loading took {model_load_time:.2f} seconds")
    
    model.config.use_cache = False
    model.fuse_cross_entropy = False
    model.criterion = FusedCrossEntropyLoss(reduction="mean")
    logger.info(f"{color.blue}\n{model}{color.reset}\n")

    # Build the collection of model converters. No-op if `model.converters` empty
    # model_converters = build_model_converters(job_config, parallel_dims)
    # model_converters.convert(model)

    # calculate model size and flops per token
    model_param_count, num_flops_per_token = get_nparams_and_flops(
        model, model_config, job_config.training.context_len
    )

    # move sharded model to CPU/GPU and initialize weights via DTensor
    if job_config.checkpoint.create_seed_checkpoint:
        init_device = "cpu"
    elif job_config.training.enable_cpu_offload:
        init_device = "cpu"
    else:
        init_device = device_type

    # apply parallelisms and initialization

    # apply PT-D Tensor Parallel, activation checkpointing, torch.compile, Data Parallel
    train_spec.parallelize_fn(model, world_mesh, parallel_dims, job_config)
    
    # move model to device
    model.to(device_type)
    model.eval()


    device_mem_stats = device_memory_monitor.get_peak_stats()
    logger.info(
        f"{device_type.upper()} memory usage for model: "
        f"{device_mem_stats.max_reserved_gib:.2f}GiB"
        f"({device_mem_stats.max_reserved_pct:.2f}%)"
    )

 
    train_state = TrainState()

    

    metric_logger = build_metrics_processor(job_config, parallel_dims)
    # Set dependent attributes for metric_logger
    metric_logger.num_flops_per_token = num_flops_per_token

    # plot losses loaded from checkpoint (if any) to TensorBoard
    # NOTE: Loss info after the last log step before checkpoint saving will not be ploted.
    #       This can be avoided by setting checkpoint.interval to be a multiple of metrics.log_freq
    if train_state.step > 0 and len(metric_logger.data_loading_times) > 0:
        for idx, step in enumerate(train_state.log_steps):
            metric_logger.log(
                step,
                global_avg_loss=train_state.global_avg_losses[idx],
                global_max_loss=train_state.global_max_losses[idx],
            )

    data_iterator = iter(dataloader)

    train_context = dist_utils.get_train_context(
        parallel_dims.loss_parallel_enabled,
        job_config.experimental.enable_compiled_autograd,
    )

    # variables used to keep info for metrics logging
    device_memory_monitor.reset_peak_stats()

    global_batch_size = (
        job_config.training.batch_size
        * dp_degree
        * job_config.training.gradient_accumulation_steps
    )
    num_tokens_per_step = global_batch_size * job_config.training.seq_len
    # train loop
    logger.info(f"{color.red}***** Running training *****{color.reset}")
    logger.info(f"{color.green}  Training starts at step {train_state.step + 1}")
    logger.info(
        f"{color.green}  Number of tokens per sequence = {job_config.training.seq_len:,}"
    )
    logger.info(
        f"{color.green}  Gradient Accumulation steps = {job_config.training.gradient_accumulation_steps}"
    )
    logger.info(
        f"{color.green}  Instantaneous batch size (per device) = {job_config.training.batch_size:,}"
    )
    logger.info(
        f"{color.green}  Global batch size (w. parallel, distributed & accumulation) = {global_batch_size:,}"
        f" ({num_tokens_per_step:,} tokens)"
    )
    logger.info(
        f"{color.green}  Total optimization steps = {job_config.training.steps:,} "
        f"({job_config.training.steps * num_tokens_per_step:,} tokens)"
    )
    logger.info(
        f"{color.green}  Warmup steps = {job_config.lr_scheduler.warmup_steps:,}"
        f" ({job_config.lr_scheduler.warmup_steps * num_tokens_per_step:,} tokens)"
    )
    logger.info(
        f"{color.green}  Number of parameters = {model_param_count:,} {color.reset}"
    )

    with (
        maybe_enable_profiling(
            job_config, global_step=train_state.step
        ) as torch_profiler,
        maybe_enable_memory_snapshot(
            job_config, global_step=train_state.step
        ) as memory_profiler,
        torch.no_grad(),
    ):
        loss_tracker = AverageTracker()
        loss_at_per_token_tracker = AverageTracker()
        while train_state.step < job_config.training.steps:
            train_state.step += 1
            gc_handler.run(train_state.step)
            
            # do gradient accumulation if enabled
            # get batch
            data_load_start = time.perf_counter()
            batch = next(data_iterator)
            input_ids, labels = batch["input_ids"], batch["labels"]
            # logger.info(f"debug input_ids.shape: {input_ids.shape}, labels.shape: {labels.shape}")

            # Update metrics processor state before forward/backward
            metric_logger.ntokens_since_last_log += labels.numel()
            metric_logger.data_loading_times.append(
                time.perf_counter() - data_load_start
            )

            input_ids = input_ids.to(device_type)

            """
            TODO[flame]: We need to carefully handle the position_ids for TP/CP
            Depending on the Models'PE, the position_ids might be different.

            e.g. for TP
                For RoPE, all ranks have the same position_ids. [FOR HF model]
                For sinusoidal, each rank has the coresponding chunked  position_ids. [FOR HF model]

            e.g. for CP, [optional_context_parallel_ctx shoudl automatically distbute the position_ids]
                Each rank has the coresponding chunked position_ids. [FOR All model]

            """
            labels = labels.to(device_type)
            cu_seqlens = (
                batch["cu_seqlens"].to(device_type)
                if "cu_seqlens" in batch
                else None
            )
            if cu_seqlens is not None:
                position_ids = prepare_position_ids(cu_seqlens).to(torch.int32)
            else:
                position_ids = (
                    torch.arange(0, input_ids.shape[1], device=device_type)
                    .repeat(input_ids.shape[0], 1)
                    .to(torch.int32)
                )
            # apply context parallelism if cp is enabled
            # ensure CP handles the separate freqs_cis buffer for each pp stage
            optional_context_parallel_ctx = (
                dist_utils.create_context_parallel_ctx(
                    cp_mesh=world_mesh["cp"],
                    cp_buffers=[input_ids, labels, position_ids],
                    cp_seq_dims=[1, 1, 1],
                    cp_no_restore_buffers={input_ids, labels, position_ids},
                    cp_rotate_method=job_config.experimental.context_parallel_rotate_method,
                )
                if parallel_dims.cp_enabled
                else None
            )

        
            # Non-PP forward / backward
            with train_context(optional_context_parallel_ctx):
                if use_hidden_state_per_token_loss:
                    output = model(
                        input_ids=input_ids,
                        labels=None,
                        position_ids=position_ids,
                        cu_seqlens=cu_seqlens,
                        output_hidden_states=True,
                        logits_to_keep=1,
                    )
                    if output.hidden_states is None:
                        raise RuntimeError("output_hidden_states=True did not return hidden states")
                    labels = labels.to(output.hidden_states[-1].device)
                    loss, per_token_loss = _per_token_loss_from_hidden_states(
                        model=model,
                        hidden_states=output.hidden_states[-1],
                        labels=labels,
                        ignore_index=model.criterion.ignore_index,
                        chunk_size=lm_head_chunk_size,
                    )
                    loss = loss / job_config.training.gradient_accumulation_steps
                else:
                    output = model(
                        input_ids=input_ids,
                        labels=labels,
                        position_ids=position_ids,
                        cu_seqlens=cu_seqlens,
                    )
                    loss = (
                        output.loss
                        / job_config.training.gradient_accumulation_steps
                    )

                    # [bs, seq_len, vocab_size]
                    logits = output.logits
                    # [bs, seq_len]
                    labels = labels.to(logits.device)
                    bs, seq_len = labels.shape
                    labels = torch.cat((labels[..., 1:], torch.full_like(labels[:, :1], model.criterion.ignore_index)), 1)

                    criterion = FusedCrossEntropyLoss(reduction="none")
                    flatten_logits = logits.view(labels.numel(), -1)
                    flatten_labels = labels.view(-1)
                    per_token_loss = criterion(flatten_logits, flatten_labels) # [bs * seq_len]
                    per_token_loss = per_token_loss.view(bs, seq_len)
                    # Sum across batch dimension and divide by batch size to get per-token loss
                    per_token_loss = per_token_loss.mean(dim=0) # [seq_len]
            
            loss_tracker.update(loss)
            loss_at_per_token_tracker.update(per_token_loss)

            # logger.info(f"rank: {torch.distributed.get_rank()} loss: {loss} labels: {flatten_labels[10:15]}")
            
            

            # log metrics - Use MetricsProcessor
            if metric_logger.should_log(train_state.step):
                if (
                    parallel_dims.dp_replicate_enabled
                    or parallel_dims.dp_shard_enabled
                    or parallel_dims.cp_enabled
                ):
                    loss = loss.detach()
                    # Use dist_mean/max on the accumulated loss for the step
                    global_avg_loss, global_max_loss = (
                        dist_utils.dist_mean(
                            loss,
                            world_mesh["dp_cp"],
                        ),
                        dist_utils.dist_max(
                            loss,
                            world_mesh["dp_cp"],
                        ),
                    )
                    logger.info(f"rank: {torch.distributed.get_rank()} global_avg_loss: {global_avg_loss} local_avg_loss: {loss}")
                else:
                    # Scale back the loss before logging
                    global_avg_loss = global_max_loss = loss.item()

                # Update train state tokens and elapsed time
                time_now = time.perf_counter()
                time_delta = (
                    time_now - metric_logger.time_last_log
                )  # Use metric_logger's time
                train_state.token += (
                    metric_logger.ntokens_since_last_log  # Use tokens tracked by metric_logger
                    * parallel_dims.world_size
                    / parallel_dims.non_data_parallel_size
                )
                train_state.elapsed += timedelta(seconds=time_delta)
                train_state.log_steps.append(train_state.step)
                train_state.global_avg_losses.append(global_avg_loss)
                train_state.global_max_losses.append(global_max_loss)

                # Log using the metric processor
                metric_logger.log(train_state.step, global_avg_loss, global_max_loss)

                loss_at_chunks = loss_at_per_token_tracker.per_chunk_average(num_chunks=16).detach().cpu().numpy()
                new_log_dict = {}
                for i, chunk_avg_loss in enumerate(loss_at_chunks):
                    new_log_dict[f"eval_loss/loss_at_chunk_{i}"] = chunk_avg_loss
                metric_logger.logger.log(new_log_dict, train_state.step)

            # reduce timeout after first train step for faster signal
            # (assuming lazy init and compilation are finished)
            if train_state.step == 1:
                dist_utils.set_pg_timeouts(
                    timeout=timedelta(seconds=job_config.comm.train_timeout_seconds),
                    world_mesh=world_mesh,
                )

        loss_tracker.sync_ddp()
        loss_at_per_token_tracker.sync_ddp()
        loss = loss_tracker.average()
        loss_at_per_token = loss_at_per_token_tracker.average()

        print("debug loss: ", loss)
        print("debug loss_at_per_token: ", loss_at_per_token, loss_at_per_token.shape)

        per_chunk_loss = loss_at_per_token_tracker.per_chunk_average(num_chunks=16)
        print("debug per_chunk_loss: ", per_chunk_loss, per_chunk_loss.shape)


    if torch.distributed.get_rank() == 0:
        loss_at_per_token_tracker.save_to_json(report_save_path)
        plot_save_path = os.path.join(job_config.job.dump_folder, f"{save_prefix}_loss_at_per_token_position.png")
        loss_at_per_token_tracker.plot_curve(save_path=plot_save_path)
        print(f"Saved loss at per token to {report_save_path}")
        print(f"Saved loss at per token plot to {plot_save_path}")
        num_of_tokens_tracked = loss_at_per_token_tracker.num_of_tokens_tracked()
        print(f"Number of tokens tracked: {num_of_tokens_tracked / 1e6}M")
        # Load the saved plot image and log it to wandb
        if job_config.metrics.enable_wandb and wandb.run is not None:

            per_token_loss_list = loss_at_per_token_tracker.average().detach().cpu().tolist()

            wandb.define_metric("token_position")
            wandb.define_metric("per_token_loss", step_metric="token_position")
            for i, loss_val in enumerate(per_token_loss_list):
                if loss_val == 0 and i == len(per_token_loss_list) - 1:
                    continue
                wandb.log({"per_token_loss": loss_val, "token_position": i})

            # Log per-token loss as individual W&B metrics for native chart support
            wandb.define_metric("token_position")
            wandb.define_metric("loss_metrics/per_token_loss", step_metric="token_position")
            for i, loss_val in enumerate(per_token_loss_list):
                if loss_val == 0 and i == len(per_token_loss_list) - 1:
                    continue
                wandb.log({"loss_metrics/per_token_loss": loss_val, "token_position": i})

            from PIL import Image
            pil_image = Image.open(plot_save_path)
            wandb.log({"loss_curve": wandb.Image(pil_image, caption="Per-token loss curve")})
        elif job_config.metrics.enable_wandb:
            logger.warning("Skipping final W&B eval_loss artifacts because wandb.init did not create a run")
            

        # fianlly log the loss at each chunk
        new_log_dict = {}
        for i, chunk_avg_loss in enumerate(loss_at_chunks):
            new_log_dict[f"final_loss/loss_at_chunk_{i}"] = chunk_avg_loss
        metric_logger.logger.log(new_log_dict, train_state.step)

        logger.info("Sleeping 2 seconds for other ranks to complete")
        time.sleep(2)

    metric_logger.close()
    logger.info("Training completed")


if __name__ == "__main__":
    init_logger()
    config = JobConfig()

    config.parser.add_argument("--training.save_prefix", type=str, default="default")
    config.parser.add_argument("--checkpoint.load_path", type=str, default=None,
                              help="Path to checkpoint directory to load model from. If not specified, uses model.config directory.")
    config.parser.add_argument("--eval.per_token_loss_from_hidden", action="store_true",
                              help="Compute per-token loss by chunking the final hidden states through lm_head.")
    config.parser.add_argument("--eval.lm_head_chunk_size", type=int, default=512,
                              help="Token chunk size for hidden-state per-token loss.")
    config.parse_args()
    main(config)
    torch.distributed.destroy_process_group()
