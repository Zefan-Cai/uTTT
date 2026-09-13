import math
from ast import literal_eval
import argparse
import copy
import importlib
import os
import datetime
import functools
import random

import time
from collections import defaultdict
from contextlib import nullcontext
import numpy as np

import torch
import torch.nn as nn
import wandb
import yaml
from easydict import EasyDict as edict
import omegaconf
from torch.distributed import destroy_process_group, init_process_group
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from rich import print

from uttt_nvs.train.checkpoint import (
    checkpoint_job,
    get_job_overview,
    resume_job,
    select_resume_path,
    delete_previous_job,
)
from uttt_nvs.train.optimizer import configure_lr_scheduler, configure_optimizer
from uttt_nvs.train.distributed import print_rank0, dist_avg_loss_dict, unwrap_model
from uttt_nvs.models import debug_utils

##############################################################################################################
# DDP setup
##############################################################################################################
init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=3600))
ddp_rank = int(os.environ["RANK"])
ddp_local_rank = int(os.environ["LOCAL_RANK"])
ddp_local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
ddp_world_size = int(os.environ["WORLD_SIZE"])
ddp_node_rank = int(os.environ["GROUP_RANK"])
device = f"cuda:{ddp_local_rank}"
torch.cuda.set_device(device)

# Set seed
parser = argparse.ArgumentParser(description="Override YAML values")
parser.add_argument("config", type=str, help="Path to YAML configuration file")
parser.add_argument("--seed", type=int, default=9595, help="Seed for the random number generator")
parser.add_argument(
    "--load", type=str, default="", help="Force to load the weight from somewhere else"
)
parser.add_argument("--debug", action="store_true", help="Debug mode")
parser.add_argument(
    "--set",
    "-s",
    type=str,
    action="append",
    nargs=2,
    metavar=("KEY", "VALUE"),
    help="New value for the key",
)
parser.add_argument("--wandb_project", type=str, default="uttt_nvs", help="Wandb project name")
args = parser.parse_args()



random_seed = args.seed + ddp_rank
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)  # each process gets a different seed

print(
    f"Process {ddp_rank}/{ddp_world_size} is using device {ddp_local_rank}/{ddp_local_world_size} on node {ddp_node_rank}"
)


##############################################################################################################
# Config setup
##############################################################################################################
def set_nested_key(data, keys, value):
    """Sets value in nested dictionary"""
    key = keys.pop(0)
    try:
        # Converting key to int to support setting list
        key = int(key)
    except ValueError:
        key = key
    if len(keys) > 0:
        if not isinstance(key, int) and key not in data:
            data[key] = {}
        set_nested_key(data[key], keys, value)
    else:
        try:
            # attempt to eval it it (e.g. if bool, number, or etc)
            attempt = literal_eval(value)
        except (SyntaxError, ValueError):
            # if that goes wrong, just use the string
            attempt = value
        data[key] = attempt



config = omegaconf.OmegaConf.load(args.config)

# Override the YAML values
if args.set is not None:
    for key_value in args.set:
        key, value = key_value
        short2long = {
            "block.": "model.block_config.params.",
            "block0.": "model.block_config.0.params.",
            "block1.": "model.block_config.1.params.",
            "block2.": "model.block_config.2.params.",
        }
        for short, long in short2long.items():
            key = key.replace(short, long)
        if dist.get_rank() == 0:
            print(f"Overriding {key} with {value}")
        key_parts = key.split(".")
        set_nested_key(config, key_parts, value)

config = edict(config)
print_rank0(config)

# Where this run writes checkpoints, sample renders and W&B files. The default
# sits inside the repository rather than beside it; override it per run with
# `-s training.out_dir /path/to/somewhere` or the UTTT_NVS_OUT_DIR variable.
_out_root = os.environ.get("UTTT_NVS_OUT_DIR") or config.training.get("out_dir", "experiments")
out_dir = os.path.join(_out_root, config.exp_name)

##############################################################################################################
# Check if API config exists
##############################################################################################################
# Weights & Biases is optional: a run with no key still trains, with logging
# switched off, which is what the language-modelling half of this repository
# does as well. Export WANDB_API_KEY, or fill in api_keys.yaml, to enable it.
_wandb_key = os.environ.get("WANDB_API_KEY")
if not _wandb_key:
    _key_path = config.training.get("api_key_path")
    if _key_path and os.path.exists(_key_path):
        _wandb_key = (yaml.safe_load(open(_key_path, "r")) or {}).get("wandb")
api_keys = edict({"wandb": _wandb_key or ""})

##############################################################################################################
# Resolve dataset manifests
##############################################################################################################
# Paths may be written straight into a config, but the usual route is to fill in
# uttt_nvs/datasets.yaml once and let every config read from it. A config opts in
# with `training.dataset: obj` (or `dl3dv`); an explicit path in the config still
# wins, so a one-off experiment can override the registry.
_ds_key = config.training.get("dataset")
if _ds_key:
    _registry_path = config.training.get("dataset_registry", "uttt_nvs/datasets.yaml")
    if not os.path.exists(_registry_path):
        raise FileNotFoundError(
            f"Dataset registry '{_registry_path}' not found. Copy "
            "uttt_nvs/datasets.example.yaml to uttt_nvs/datasets.yaml and fill in "
            "your manifest paths, or set training.dataset_path explicitly."
        )
    _registry = yaml.safe_load(open(_registry_path, "r")) or {}
    if _ds_key not in _registry:
        raise KeyError(
            f"Dataset '{_ds_key}' is not defined in {_registry_path}. "
            f"Available: {sorted(_registry)}"
        )
    _entry = _registry[_ds_key]
    for _field, _cfg_key in (("train", "dataset_path"), ("eval", "eval_dataset_path")):
        _value = _entry.get(_field)
        if not _value or str(_value).startswith("PATH/TO/"):
            raise ValueError(
                f"`{_ds_key}.{_field}` in {_registry_path} still holds its placeholder "
                "value. Point it at a manifest; see uttt_nvs/data/README.md."
            )
        # An explicit path in the config wins, as the config comments promise.
        # Only fill in what the config left unset or left as a placeholder.
        if _cfg_key == "dataset_path":
            _current = config.training.get("dataset_path")
            if not _current or str(_current).startswith("PATH/TO/"):
                config.training.dataset_path = _value
        else:
            _current = config.get("eval_dataset_path")
            if not _current or str(_current).startswith("PATH/TO/"):
                config.eval_dataset_path = _value

##############################################################################################################
# Load data
##############################################################################################################
dataset_name = config.training.get("dataset_name", "uttt_nvs.data.loader.NVSDataset")
module, class_name = dataset_name.rsplit(".", 1)
Dataset = importlib.import_module(module).__dict__[class_name]
dataset = Dataset(config)

eval_config = copy.deepcopy(config)
_eval_dataset_path = eval_config.get("eval_dataset_path")
if not _eval_dataset_path or str(_eval_dataset_path).startswith("PATH/TO/"):
    raise ValueError(
        "`eval_dataset_path` still holds its placeholder value. Point it at an "
        "evaluation manifest; "
        "see uttt_nvs/data/README.md for the manifest format and "
        "uttt_nvs/data/build_manifest.py for a generator."
    )
eval_config.training.dataset_path = _eval_dataset_path
eval_config.training.num_views = config.training.get(
    "eval_num_views", 
    config.training.num_views
)
eval_dataset = Dataset(eval_config)
print(f"Eval dataset loaded! Length: {len(eval_dataset)}")


if ddp_rank == 0:
    print("Dataset loaded! Example data:")
    for k, v in dataset[0].items():
        try:
            print(f"{k}: {v.shape}")
        except:
            print(f"{k}: {type(v)}")

    from einops import rearrange

    # import numpy as np
    from PIL import Image

    os.makedirs(
        os.path.join(out_dir, "data_examples"), exist_ok=True
    )
    im = dataset[0]["image"]
    im = rearrange(im, "v c h w -> h (v w) c").detach().cpu().numpy()
    im = (im[..., :4] * 255).astype(np.uint8)
    Image.fromarray(im).save(
        os.path.join(out_dir, "data_examples", "image.png")
    )

dataloader_seed_generator = torch.Generator()
dataloader_seed_generator.manual_seed(95 + ddp_rank)
datasampler = DistributedSampler(dataset, shuffle=True, drop_last=True)
# With drop_last=True a rank that holds fewer samples than one per-GPU batch
# yields nothing at all, and the training loop dies on a bare StopIteration
# several hundred lines later. Say so here instead.
_per_rank = len(dataset) // ddp_world_size
if _per_rank < config.training.batch_size_per_gpu:
    raise ValueError(
        f"The training set has {len(dataset)} samples, which is {_per_rank} per rank "
        f"across {ddp_world_size} ranks, but each rank needs at least "
        f"{config.training.batch_size_per_gpu} (training.batch_size_per_gpu) to fill "
        f"one batch. Use a larger dataset, or lower batch_size_per_gpu and "
        f"total_batch_size together."
    )

dataloader = DataLoader(
    dataset,
    batch_size=config.training.batch_size_per_gpu,
    shuffle=False,
    num_workers=config.training.num_workers,
    persistent_workers=True,
    pin_memory=True,
    drop_last=True,
    prefetch_factor=config.training.prefetch_factor,
    sampler=datasampler,
    generator=dataloader_seed_generator,
    # num_threads=config.training.num_threads,
)

eval_datasampler = DistributedSampler(eval_dataset, shuffle=False, drop_last=True)
eval_dataloader = DataLoader(
    eval_dataset,
    batch_size=config.training.batch_size_per_gpu,
    shuffle=False,
    # num_workers=config.training.num_workers,
    num_workers=0,
    persistent_workers=False,
    pin_memory=True,
    drop_last=True,
    generator=dataloader_seed_generator,
    sampler=eval_datasampler,
)

##############################################################################################################
# Overview job
##############################################################################################################
effective_max_fwdbwd_passes = (
    config.training.get("max_fwdbwd_passes", int(1e10))
    * max(int(config.training.get("grad_accum_steps", 1)), 1)
)
round_max_fwdbwd_passes_to_epoch = config.training.get(
    "round_max_fwdbwd_passes_to_epoch", True
)
job_overview = get_job_overview(
    num_gpus=ddp_world_size,
    num_train_samples=len(dataset),
    batch_size_per_gpu=config.training.batch_size_per_gpu,
    gradient_accumulation_steps=config.training.grad_accum_steps,
    max_fwdbwd_passes=effective_max_fwdbwd_passes,
    round_max_fwdbwd_passes_to_epoch=round_max_fwdbwd_passes_to_epoch,
)
print_rank0(job_overview)

# The total batch size is a product of the config and the launcher: it is
# `batch_size_per_gpu * world_size * grad_accum_steps`, and the world size comes
# from torchrun, not from this file. Running a config on a different number of
# GPUs therefore silently changes the effective batch size while the learning
# rate stays put. If a config states what it expects, hold it to that.
# The GPU count comes from the launcher, not from any file in this repository,
# so the effective total batch size is invisible to the config. Record it there
# and hold the run to it, otherwise the same config silently trains at a
# different batch size on a different machine while the learning rate stays put.
_total_batch = config.training.get("total_batch_size")
if _total_batch is not None:
    _actual = job_overview.batch_size_per_param_update
    if _actual != int(_total_batch):
        _per_gpu = config.training.batch_size_per_gpu
        _accum = config.training.grad_accum_steps
        raise ValueError(
            f"Effective total batch size is {_actual} "
            f"({_per_gpu} per GPU x {ddp_world_size} GPUs x {_accum} grad-accum steps), "
            f"but this experiment is defined at {int(_total_batch)}.\n"
            f"Launch on {int(_total_batch) // (_per_gpu * max(_accum, 1))} GPUs, or scale "
            f"`batch_size_per_gpu` / `grad_accum_steps` so the product is {int(_total_batch)}, "
            f"or change `total_batch_size` to run a deliberately different experiment."
        )
    print_rank0(f"Total batch size {_actual} matches the experiment definition.")

# Set torch magics before the actual model
torch.set_float32_matmul_precision("high")
torch._dynamo.config.optimize_ddp = False
torch._dynamo.config.specialize_int = True
torch._dynamo.config.specialize_float = True

##############################################################################################################
# model setup; dynamic import
##############################################################################################################
module, class_name = config.model.class_name.rsplit(".", 1)
Images2NeRF = importlib.import_module(module).__dict__[class_name]
model = Images2NeRF(config).to(device)
model_overview = model.get_overview()

##############################################################################################################
# Optimizer, scheduler setup, and chekcpoint loading
##############################################################################################################
optimizer, optim_param_dict, all_param_dict = configure_optimizer(
    model,
    config.training.weight_decay,
    config.training.lr,
    (config.training.beta1, config.training.beta2),
)
optim_param_list = list(optim_param_dict.values())

scheduler_type = config.training.get("scheduler_type", "cosine")
lr_scheduler = configure_lr_scheduler(
    optimizer,
    job_overview.num_param_updates,
    config.training.warmup,
    scheduler_type=scheduler_type,
    warmup_init_ratio=config.training.get("warmup_init_ratio", 0.0),
)

fwdbwd_pass_step, param_update_step = 0, 0

try_load_path = select_resume_path(out_dir, args.load, config.training.get("load", ""))
reset_training_state = config.training.get("reset_training_state", False) and (
    try_load_path != out_dir
)
reset_training_state = reset_training_state or config.training.get(
    "force_reset_training_state", False
)
load_optimizer_state_when_reset_training_state = config.training.get(
    "load_optimizer_state_when_reset_training_state", False
) and reset_training_state
optimizer, lr_scheduler, fwdbwd_pass_step, param_update_step = resume_job(
    try_load_path,
    model,
    optimizer,
    lr_scheduler,
    reset_training_state,
    load_optimizer_state_when_reset_training_state,
)

optimizer_overview = edict(
    num_optim_params=sum(p.numel() for n, p in optim_param_dict.items()),
    num_all_params=sum(p.numel() for n, p in all_param_dict.items()),
    optim_param_names=list(optim_param_dict.keys()),
    freeze_param_names=list(set(all_param_dict.keys()) - set(optim_param_dict.keys())),
)

if ddp_rank == 0:
    print(model)
    print(optimizer_overview)

model = DDP(model, device_ids=[ddp_local_rank])

if config.model.act_ckpt:
    # The wrapper this flag used to install targets an attribute these models
    # do not have, and the blocks already apply gradient checkpointing
    # unconditionally -- so the flag could only ever fail at runtime. Refuse it
    # up front instead of crashing mid-setup.
    raise NotImplementedError(
        "model.act_ckpt is not supported: gradient checkpointing is already "
        "applied inside the blocks. Remove `act_ckpt: true` from the config."
    )

do_torch_compile = config.training.get("torch_compile", False)
if do_torch_compile:
    torch._dynamo.config.cache_size_limit = 256

    print("Compiling model")
    model = torch.compile(model)  # pytorch 2.0 feature

if ddp_rank == 0:
    print("The model after DDP wrapper and compilation.")
    print(model)

print_rank0(
    f"Before training loop: fwdbwd_pass_step: {fwdbwd_pass_step}, param_update_step: {param_update_step}, optimizer.param_groups[0]['lr']: {optimizer.param_groups[0]['lr']}"
)
print_rank0(
    f"Before training loop: {[(x, y) for x, y in lr_scheduler.state_dict().items()]}"
)

##############################################################################################################
# wandb setup
##############################################################################################################
if ddp_rank == 0:
    if api_keys.wandb:
        os.environ["WANDB_API_KEY"] = api_keys.wandb
    elif not os.environ.get("WANDB_MODE"):
        os.environ["WANDB_MODE"] = "disabled"
        print(
            "No Weights & Biases API key found: training continues with W&B "
            "logging disabled. Export WANDB_API_KEY or fill in api_keys.yaml "
            "to record this run."
        )
    if config.training.wandb_offline:
        os.environ["WANDB_MODE"] = "offline"

    # try getting the wandb id from the file
    wandb_id = None
    if os.path.exists(os.path.join(out_dir, "wandb_id.txt")):
        with open(os.path.join(out_dir, "wandb_id.txt"), "r") as f:
            wandb_id = f.read().strip()
        print(f"Resuming wandb run with id {wandb_id}")

    if api_keys.wandb:
        wandb.login()
    config_copy = copy.deepcopy(config)
    config_copy["job_overview"] = job_overview
    config_copy["optimizer_overview"] = optimizer_overview
    config_copy["model_overview"] = model_overview
    save_dir = os.path.join(out_dir, "wandb")
    os.makedirs(save_dir, exist_ok=True)
    wandb_resume = config.training.get("wandb_resume", "allow")
    wandb.init(
        # entity="research-3gi",
        project=args.wandb_project,
        name=config.exp_name,
        dir=save_dir,
        id=wandb_id,
        resume=wandb_resume,
        config=config_copy,
    )
    # wandb.run.log_code(".")

    # save wandb id if a new run for later resuming wandb
    if wandb_id is None:
        wandb_id = wandb.run.id
        with open(os.path.join(out_dir, "wandb_id.txt"), "w") as f:
            f.write(wandb_id)

    print("Wandb setup done")


##############################################################################################################
# Training loop
##############################################################################################################
torch.distributed.barrier()

print(f"ddp_rank={ddp_rank}, Starting training loop")

dataloader_iter = iter(dataloader)
start_fwdbwd_pass_step = fwdbwd_pass_step
grad_accum_steps = max(int(config.training.get("grad_accum_steps", 1)), 1)
model.train()
optimizer.zero_grad(set_to_none=True)
_train_loop_start_time = time.time()
save_checkpoint = False

while fwdbwd_pass_step < job_overview.num_fwdbwd_passes:
    tic = time.time()

    cur_epoch = fwdbwd_pass_step // job_overview.num_fwdbwd_passes_per_epoch

    loss_computer = unwrap_model(model).loss_computer
    if hasattr(loss_computer, "l2_warmup"):
        loss_computer.l2_warmup(set=fwdbwd_pass_step < config.training.get("l2_warmup_steps", 0))

    if fwdbwd_pass_step % job_overview.num_fwdbwd_passes_per_epoch == 0:
        print(
            f"ddp_rank={ddp_rank}, Resetting dataloader epoch to {cur_epoch}; might take a while..."
        )

        datasampler.set_epoch(cur_epoch)
        dataloader_iter = iter(dataloader)

    batch = next(dataloader_iter)
    batch = {k: v.to(device) for k, v in batch.items()}

    next_fwdbwd_pass_step = fwdbwd_pass_step + 1
    should_step_optimizer = (
        next_fwdbwd_pass_step % grad_accum_steps == 0
        or next_fwdbwd_pass_step == job_overview.num_fwdbwd_passes
    )
    sync_context = (
        model.no_sync()
        if grad_accum_steps > 1 and not should_step_optimizer and hasattr(model, "no_sync")
        else nullcontext()
    )

    with sync_context:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            result = model(batch)

        # Convert to easydict 
        result = edict(result)
        result.loss_metrics = edict(result.loss_metrics)

        # All ranks must agree on whether to run backward(): in DDP, backward()
        # is what joins the gradient all-reduce, so a rank that skips it alone
        # leaves every other rank blocked in NCCL until the watchdog aborts
        # them. The all_reduce sits OUTSIDE the branch so every rank reaches
        # it; MIN makes one non-finite rank turn the step into a collective
        # skip instead of a deadlock.
        _loss_finite = torch.isfinite(result.loss_metrics.loss).float()
        if dist.is_initialized():
            dist.all_reduce(_loss_finite, op=dist.ReduceOp.MIN)
        loss_is_finite = bool(_loss_finite.item())
        if loss_is_finite:
            (result.loss_metrics.loss / grad_accum_steps).backward()

    fwdbwd_pass_step = next_fwdbwd_pass_step

    skip_optimizer_step = not loss_is_finite
    should_reset_grads = skip_optimizer_step
    if skip_optimizer_step:
        print("WARNING: NaN or inf loss encountered, skipping optimizer step")

    total_grad_norm = None
    if should_step_optimizer and not should_reset_grads:
        if config.training.grad_clip_norm > 0:
            total_grad_norm = torch.nn.utils.clip_grad_norm_(
                optim_param_list,
                max_norm=config.training.grad_clip_norm,
            ).item()

            if not math.isfinite(total_grad_norm):
                print(f"WARNING: step {fwdbwd_pass_step} grad norm is not finite {total_grad_norm}, skipping")
                skip_optimizer_step = True
            if total_grad_norm > config.training.grad_clip_norm * 2.0:
                if ddp_rank == 0:
                    print(f"WARNING: step {fwdbwd_pass_step} grad norm too large {total_grad_norm} > {config.training.grad_clip_norm * 2.}")
                    wandb.log({"grad_norm": total_grad_norm}, step=fwdbwd_pass_step)

            allowed_gradnorm = config.training.grad_clip_norm * config.training.get("allowed_gradnorm_factor", 5.0)
            if fwdbwd_pass_step > config.training.get("may_skip_gradnorm_after", 0):
                if total_grad_norm > allowed_gradnorm:
                    skip_optimizer_step = True
                    if ddp_rank == 0:
                        print(f"WARNING: step {fwdbwd_pass_step} grad norm too large {total_grad_norm} > {allowed_gradnorm}, skipping optimizer step")
                        wandb.log({"grad_norm": total_grad_norm}, step=fwdbwd_pass_step)

        if not skip_optimizer_step:
            optimizer.step()
            param_update_step += 1
            lr_scheduler.step()

        should_reset_grads = True

    if should_reset_grads:
        optimizer.zero_grad(set_to_none=True)

    # logging and checkpointing
    if (fwdbwd_pass_step % config.training.print_every == 0) or (
        fwdbwd_pass_step < 100 + start_fwdbwd_pass_step
    ):
        loss_name2value = {
            k: v.item() for k, v in result.loss_metrics.items()
            if isinstance(v, torch.Tensor) and v.numel() == 1
        }
        loss_values_str = [f"{k}: {v:06f}" for k, v in loss_name2value.items()]
        print_rank0(
            f"epoch: {cur_epoch}, fwdbwd_pass_step: {fwdbwd_pass_step}/{job_overview.num_fwdbwd_passes_per_epoch}, time: {time.time() - tic:06f}, param_update_step: {param_update_step}, lr: {optimizer.param_groups[0]['lr']:06f}"
        )
        print_rank0(f"\t{', '.join(loss_values_str)}")

    # Wandb value logging
    if (fwdbwd_pass_step % config.training.wandb_log_every == 0) or (
        fwdbwd_pass_step < 100 + start_fwdbwd_pass_step
    ):
        # Expand the location-based metrics.
        for key in list(result.loss_metrics):
            if key.endswith("_loc"):
                for loc, v in enumerate(result.loss_metrics[key]):
                    result.loss_metrics[f"{key}_{loc:03}"] = v
        name2loss = {
            k: v for k, v in result.loss_metrics.items()
            if isinstance(v, torch.Tensor) and v.numel() == 1
        }
        name2loss = dist_avg_loss_dict(name2loss)
        name2loss = {k: v.item() for k, v in name2loss.items()}
        log_dict = {
            "iter": fwdbwd_pass_step,  # wandb needs this
            "fwdbwd_pass_step": fwdbwd_pass_step,
            "param_update_step": param_update_step,
            "lr": optimizer.param_groups[0]["lr"],
            "iter_time": time.time() - tic,
            "epoch": cur_epoch,
        }
        if total_grad_norm is not None:
            log_dict["grad_norm"] = total_grad_norm
        log_dict.update({"train/" + k: v for k, v in name2loss.items() if "_loc_" not in k})
        log_dict.update({"train_loc/" + k: v for k, v in name2loss.items() if "_loc_" in k})

        if ddp_rank == 0:
            # log_dict.update(debug_utils.log_weight_and_grad(model))
            wandb.log(
                log_dict,
                step=fwdbwd_pass_step,
            )
        torch.distributed.barrier()

    save_checkpoint = (fwdbwd_pass_step % config.training.checkpoint_every == 0
                       or fwdbwd_pass_step == job_overview.num_fwdbwd_passes)
    if save_checkpoint:
        if ddp_rank == 0:
            checkpoint_job(
                out_dir,
                model,
                optimizer,
                lr_scheduler,
                fwdbwd_pass_step,
                param_update_step,
            )

            save_last_n_ckpts = config.training.get("save_last_n_ckpts", 3)
            delete_previous_job(
                out_dir, fwdbwd_pass_step, save_last_n_ckpts
            )
        torch.cuda.empty_cache()
        torch.distributed.barrier()

    # Visual logging
    vis_first_step = config.training.get("vis_first_step", True)
    create_visual = (
        vis_first_step and fwdbwd_pass_step == start_fwdbwd_pass_step
    ) or (
        fwdbwd_pass_step % config.training.vis_every == 0
    )
    if create_visual:
        if ddp_rank == 0:
            visual_dict = unwrap_model(model).save_visuals(
                os.path.join(
                    out_dir, f"iter_{fwdbwd_pass_step:08d}"
                ),
                result,
                batch
            )
            if visual_dict is not None and len(visual_dict) > 0:
                visual_dict = {"train_" + k: wandb.Image(v) for k, v in visual_dict.items()}
                wandb.log(visual_dict, step=fwdbwd_pass_step)
        torch.distributed.barrier()

    # do evaluation
    eval_every = config.training.get("eval_every", 2000)
    # `eval_every <= 0` disables in-training evaluation entirely, which is the
    # escape hatch the sharding check below points at.
    eval_disabled = int(eval_every) <= 0
    if eval_disabled:
        eval_steps = 0
    elif args.debug:
        eval_steps = 1
    else:
        eval_steps = min(511 // config.training.batch_size_per_gpu // ddp_world_size + 1, 8)
        max_available_batches = len(eval_dataset) // ddp_world_size // config.training.batch_size_per_gpu
        # With enough ranks the evaluation set stops dividing: every rank needs at
        # least one batch, or the ones that come up empty never reach the collective
        # and the job hangs instead of failing. Say so rather than deadlocking.
        if max_available_batches < 1:
            raise ValueError(
                f"In-training evaluation cannot run: {len(eval_dataset)} evaluation "
                f"scenes split over {ddp_world_size} ranks at "
                f"{config.training.batch_size_per_gpu} per rank leaves some ranks with "
                f"no batch at all.\n"
                f"Disable it with `-s training.eval_every 0` and evaluate separately "
                f"with `python -m uttt_nvs.eval.evaluate`, or run on at most "
                f"{max(len(eval_dataset) // config.training.batch_size_per_gpu, 1)} ranks."
            )
        eval_steps = max(min(eval_steps, max_available_batches), 1)
    eval_first_step = config.training.get("eval_first_step", True)
    # By default we still evaluate the first step for activation plots, but long
    # tuning sweeps can disable this to shorten turnaround.
    if not eval_disabled and (fwdbwd_pass_step % eval_every == 0 or (
        eval_first_step and fwdbwd_pass_step == (start_fwdbwd_pass_step + 1)
    )):
        debug_utils.log_status = True
        print(
            f"Evaluating at iter {fwdbwd_pass_step}",
            "eval_steps",
            eval_steps,
        )
        model.eval()
        torch.cuda.empty_cache()
        with torch.no_grad():
            eval_datasampler.set_epoch(0)
            eval_dataloader_iter = iter(eval_dataloader)

            # Initialize accumulators
            name2loss = defaultdict(float)
            for i in range(eval_steps):
                batch = next(eval_dataloader_iter)
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    result = model(batch)

                result = edict(result)
                result.loss_metrics = edict(result.loss_metrics)

                for key in ["loss", "psnr", "psnr_loc", "moe_freq_max", "moe_freq_min"]:
                    if key in result.loss_metrics:
                        name2loss[key] = name2loss[key] + result.loss_metrics[key] / eval_steps
            
            # Save visuals for evaluation
            psnr_loc = result.loss_metrics["psnr_loc"].clone()
            dist.all_reduce(psnr_loc, op=dist.ReduceOp.AVG)     # Avg for better vis
            if ddp_rank == 0:
                result.loss_metrics["psnr_loc"] = psnr_loc
                visual_dict = unwrap_model(model).save_visuals(
                    os.path.join(out_dir, f"eval_{fwdbwd_pass_step:08d}"),
                    result, batch)
                if visual_dict is not None and len(visual_dict) > 0:
                    visual_dict = {"eval_" + k: wandb.Image(v) for k, v in visual_dict.items()}
                    wandb.log(visual_dict, step=fwdbwd_pass_step)
            torch.distributed.barrier()

            for name, loss in list(name2loss.items()):
                if name.endswith("_loc"):
                    for loc, v in enumerate(loss):
                        name2loss[f"{name}_{loc:03}"] = v
                    del name2loss[name]

            name2loss = dist_avg_loss_dict(name2loss)
            name2loss = {k: v.item() for k, v in name2loss.items()}
            wandb_dict = {
                **{f"eval/{k}": v for k, v in name2loss.items() if "_loc_" not in k},
                **{f"eval_loc/{k}": v for k, v in name2loss.items() if "_loc_" in k},
            }

            # Log metrics on rank 0
            if ddp_rank == 0:
                if "op2block2info" in result:
                    try:
                        wandb_additional_info, html_url = debug_utils.log_op2block2info(result.op2block2info, config.exp_name, fwdbwd_pass_step)
                        print("See activations plots at: ", html_url)
                        wandb_dict.update(wandb_additional_info)
                        wandb_dict["link/activations_html_url"] = wandb.Html(f'<a href="{html_url}">Activations</a>')
                    except Exception as e:
                        print(f"Warning: log_op2block2info failed (S3 upload?), skipping: {e}")


                wandb.log(
                    wandb_dict,
                    step=fwdbwd_pass_step,
                )
                print(
                    f"eval/loss: {name2loss['loss']}, eval/psnr: {name2loss['psnr']} at iter {fwdbwd_pass_step}"
                )
        torch.distributed.barrier()
        model.train()
        debug_utils.log_status = False
    if args.debug:
        break


# Print training summary (memory + timing)
import builtins as _builtins
torch.cuda.synchronize()
_total_steps = fwdbwd_pass_step - start_fwdbwd_pass_step
_total_time = time.time() - _train_loop_start_time
_avg_time = _total_time / max(_total_steps, 1)
_peak_alloc = torch.cuda.max_memory_allocated(device) / (1024**3)
_peak_resv = torch.cuda.max_memory_reserved(device) / (1024**3)
if ddp_rank == 0:
    _builtins.print(f"[TRAIN_SUMMARY] config={args.config} steps={_total_steps} total_time={_total_time:.1f}s avg_step={_avg_time:.3f}s peak_allocated={_peak_alloc:.2f}GB peak_reserved={_peak_resv:.2f}GB")

# in case of early stopping, save the final model
if ddp_rank == 0 and not args.debug:
    if not save_checkpoint:
        checkpoint_job(
            out_dir,
            model,
            optimizer,
            lr_scheduler,
            fwdbwd_pass_step,
            param_update_step,
        )

torch.distributed.barrier()
destroy_process_group()
