import os

import torch
import torch.distributed as dist

from rich import print


def print_rank0(*args, **kwargs):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(*args, **kwargs)
    else:
        print(*args, **kwargs)


def get_rank():
    if dist.is_initialized():
        return dist.get_rank()
    else:
        return 0


def get_world_size():
    if dist.is_initialized():
        return dist.get_world_size()
    else:
        return 1


def get_local_rank():
    if dist.is_initialized():
        return int(os.environ["LOCAL_RANK"])
    else:
        return 0


def dist_avg_loss_dict(loss_dict, sort_keys=True):
    is_tensor = isinstance(list(loss_dict.values())[0], torch.Tensor)

    if sort_keys:
        # sort by key
        loss_dict = {k: loss_dict[k] for k in sorted(loss_dict)}

    if dist.is_initialized():
        flatten_values = torch.stack([
            v.flatten()[0].float() if is_tensor else torch.tensor(v).float().cuda()
            for v in loss_dict.values()
        ])
        dist.all_reduce(flatten_values, op=dist.ReduceOp.AVG)
        
        if is_tensor:
            return {k: v for k, v in zip(loss_dict.keys(), flatten_values)}
        else:
            return {k: v.item() for k, v in zip(loss_dict.keys(), flatten_values)}
    else:
        return loss_dict


def unwrap_model(model):
    if hasattr(model, "module"):
        return model.module
    else:
        return model
