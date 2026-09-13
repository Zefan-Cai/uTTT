import os
import re
import tempfile

import torch
from easydict import EasyDict as edict
from uttt_nvs.train.optimizer import configure_optimizer, configure_lr_scheduler
import traceback
from rich import print

from uttt_nvs.train.distributed import print_rank0


def checkpoint_job(
    out_dir, model, optimizer, lr_scheduler, fwdbwd_pass_step, param_update_step
):
    """Publish a complete checkpoint atomically on the output filesystem."""
    if isinstance(model, torch.nn.parallel.distributed.DistributedDataParallel):
        model = model.module

    state_dict = {}
    for key, value in model.state_dict().items():
        key = key.replace("_checkpoint_wrapped_module.", "")
        key = key.replace("_orig_mod.", "")
        while key.startswith("module."):
            key = key[len("module."):]
        state_dict[key] = value

    checkpoint = {
        "model": state_dict,
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "fwdbwd_pass_step": fwdbwd_pass_step,
        "param_update_step": param_update_step,
    }

    os.makedirs(out_dir, exist_ok=True)
    ckpt_fpath = os.path.join(out_dir, f"ckpt_{fwdbwd_pass_step:016}.pt")
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=out_dir, prefix=".ckpt_", suffix=".tmp", delete=False
        ) as handle:
            temporary_path = handle.name
            torch.save(checkpoint, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, ckpt_fpath)
    finally:
        if temporary_path is not None and os.path.exists(temporary_path):
            os.remove(temporary_path)
    print(f"Saved checkpoint to {os.path.abspath(ckpt_fpath)}")


def find_checkpoints(out_dir):
    """Return complete checkpoint filenames in numeric step order."""
    if not os.path.isdir(out_dir):
        return []
    checkpoints = []
    for name in os.listdir(out_dir):
        match = re.fullmatch(r"ckpt_(\d+)\.pt", name)
        path = os.path.join(out_dir, name)
        if match and os.path.isfile(path):
            checkpoints.append((int(match.group(1)), path))
    return [path for step, path in sorted(checkpoints)]


def select_resume_path(out_dir, load_path="", config_load_path=""):
    """Prefer an existing run, then an explicit checkpoint, then config.load."""
    if find_checkpoints(out_dir):
        return out_dir
    return load_path or config_load_path or ""


def delete_previous_job(out_dir, fwdbwd_pass_step, save_last_n_ckpts=1):
    """Keep the newest N checkpoint files at or before the current step."""
    if type(save_last_n_ckpts) is not int or save_last_n_ckpts < 1:
        raise ValueError("save_last_n_ckpts must be a positive integer")
    if not os.path.exists(out_dir):
        print(
            "WARNING: The output directory does not exist. Skip deleting previous job."
        )
        return

    candidates = [
        path for path in find_checkpoints(out_dir)
        if int(os.path.basename(path)[5:-3]) <= fwdbwd_pass_step
    ]
    for ckpt_fpath in candidates[:-save_last_n_ckpts]:
        os.remove(ckpt_fpath)
        print(f"Deleted checkpoint {os.path.abspath(ckpt_fpath)}")


def resume_job(
    load_path,
    model,
    optimizer,
    lr_scheduler,
    reset_training_state=False,
    load_optimizer_state_when_reset_training_state=False,
):
    """Restore weights and optionally optimizer/scheduler/step state.

    An empty path means training from scratch. A supplied directory is searched
    newest-first, falling back to older files if deserialization fails. A missing
    or entirely unreadable source raises instead of silently training from scratch.
    Resetting training state returns zero counters but still loads model weights.
    Returns (optimizer, lr_scheduler, fwdbwd_pass_step, param_update_step).
    """
    if not load_path:
        return optimizer, lr_scheduler, 0, 0
    load_path = os.fspath(load_path)
    if os.path.isdir(load_path):
        all_ckpt_paths = find_checkpoints(load_path)

        # No checkpoint found in this directory
        if len(all_ckpt_paths) == 0:
            raise FileNotFoundError(f"No ckpt_<step>.pt checkpoints in {load_path}")
    else:
        # If file, assume that it is a checkpoint
        if not load_path.endswith(".pt"):
            raise ValueError(f"Checkpoint path must be a .pt file or directory: {load_path}")
        if not os.path.isfile(load_path):
            raise FileNotFoundError(f"Checkpoint does not exist: {load_path}")

        all_ckpt_paths = [load_path]

    # Load the latest checkpoint in the reverse order
    #   This is to avoid the last checkpoint corrupted (due to disk issue or sudden kill jobs)
    for ckpt_fpath in all_ckpt_paths[::-1]:
        try:
            # Load checkpoints to CPU, it can avoid double loading the params into a single GPU.
            checkpoint = torch.load(ckpt_fpath, map_location="cpu")
        except Exception:
            traceback.print_exc()
            print(
                f"Failed to load {ckpt_fpath}, we will continue to load the next ckpt in the reverse order"
            )
            continue
        else:
            break
    else:
        raise RuntimeError(
            f"Failed to load any checkpoint in {load_path}; all ckpt paths: {all_ckpt_paths}"
        )

    # Load model weights
    if model is not None:
        if isinstance(model, torch.nn.parallel.distributed.DistributedDataParallel):
            model = model.module

        state_dict = {}
        for key, value in checkpoint["model"].items():
            key = key.replace("_checkpoint_wrapped_module.", "")
            key = key.replace("_orig_mod.", "")
            while key.startswith("module."):
                key = key[len("module."):]
            state_dict[key] = value

        status = model.load_state_dict(state_dict, strict=False)
        print_rank0(
            f"Loaded model from {os.path.abspath(ckpt_fpath)}, the status is {status}"
        )

    # reset the training state
    if reset_training_state:
        if load_optimizer_state_when_reset_training_state:
            fresh_lrs = [pg["lr"] for pg in optimizer.param_groups]
            try:
                optimizer.load_state_dict(checkpoint["optimizer"])
                for pg, lr in zip(optimizer.param_groups, fresh_lrs):
                    pg["lr"] = lr
                print_rank0(
                    "Loaded optimizer state while keeping fresh lr_scheduler, "
                    "fwdbwd_pass_step, and param_update_step"
                )
            except (ValueError, KeyError) as e:
                print_rank0(
                    "[WARNING] Failed to load optimizer state dict while resetting "
                    f"training state: {e}"
                )
        print_rank0(
            f"Reset the training state to have fresh optimizer, lr_scheduler, fwdbwd_pass_step, param_update_step"
        )
        return optimizer, lr_scheduler, 0, 0

    ckpt_lr = checkpoint["optimizer"]["param_groups"][0]["lr"]
    print(f"checkpoint['optimizer']['param_groups'][0]['lr'] = {ckpt_lr}")
    try:
        optimizer.load_state_dict(checkpoint["optimizer"])
        print(f"optimizer.param_groups[0]['lr'] = {optimizer.param_groups[0]['lr']}")
        print_rank0(f"Loaded optimizer from {os.path.abspath(ckpt_fpath)}")
    except (ValueError, KeyError) as e:
        print_rank0(
            f"[WARNING] Failed to load optimizer state dict (likely param group mismatch due to model architecture change): {e}\n"
            f"Skipping optimizer state restore. Restoring LR={ckpt_lr} from checkpoint."
        )
        for pg in optimizer.param_groups:
            pg["lr"] = ckpt_lr

    lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    print_rank0(
        f"Loaded learning rate scheduler from {os.path.abspath(ckpt_fpath)}"
    )

    return (
        optimizer,
        lr_scheduler,
        checkpoint["fwdbwd_pass_step"],
        checkpoint["param_update_step"],
    )


def get_job_overview(
    num_gpus,
    num_train_samples,
    batch_size_per_gpu,
    gradient_accumulation_steps,
    max_fwdbwd_passes=int(1e10),
    round_max_fwdbwd_passes_to_epoch=True,
):
    """Compute the total number of training steps."""
    batch_size_per_fwdbwd_pass = batch_size_per_gpu * num_gpus
    num_fwdbwd_passes_per_epoch = max(
        1, int(num_train_samples / batch_size_per_fwdbwd_pass)
    )
    batch_size_per_param_update = (
        batch_size_per_fwdbwd_pass * gradient_accumulation_steps
    )
    num_param_updates_per_epoch = int(
        num_fwdbwd_passes_per_epoch / gradient_accumulation_steps
    )

    if round_max_fwdbwd_passes_to_epoch:
        num_epochs = int(max_fwdbwd_passes / num_fwdbwd_passes_per_epoch) + 1
        num_fwdbwd_passes = num_fwdbwd_passes_per_epoch * num_epochs
    else:
        num_fwdbwd_passes = max_fwdbwd_passes
        num_epochs = (
            num_fwdbwd_passes + num_fwdbwd_passes_per_epoch - 1
        ) // num_fwdbwd_passes_per_epoch
    num_param_updates = (
        num_fwdbwd_passes + gradient_accumulation_steps - 1
    ) // gradient_accumulation_steps
    overview = edict(
        batch_size_per_fwdbwd_pass=batch_size_per_fwdbwd_pass,
        batch_size_per_param_update=batch_size_per_param_update,
        num_fwdbwd_passes_per_epoch=num_fwdbwd_passes_per_epoch,
        num_param_updates_per_epoch=num_param_updates_per_epoch,
        num_fwdbwd_passes=num_fwdbwd_passes,
        num_param_updates=num_param_updates,
        num_epochs=num_epochs,
    )
    return overview
