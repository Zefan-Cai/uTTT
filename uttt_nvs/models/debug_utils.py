log_status = False

import os
import re
from typing import Dict, List, Tuple

import torch
import matplotlib
import numpy as np
matplotlib.use("Agg")
import matplotlib.pyplot as plt



@torch.no_grad()
def sketch_tensor(tensor):
    tensor = tensor.float()

    tensor_rms = (tensor ** 2).mean().sqrt()
    tensor_abs_mean = tensor.abs().mean()
    tensor_mean = tensor.mean()
    tensor_max = tensor.abs().max()

    tensor_sketch = torch.stack([tensor_rms, tensor_abs_mean, tensor_mean, tensor_max])
    return tensor_sketch


@torch.no_grad()
def log_weight_and_grad(model):
    """
    return: 
    a dict of {
        "weights/param_name/{metric}": float value,
        "grads/param_name/{metric}": float value,
    }
    metirc in [rms_norm, abs_max]

    Requirement:
        1. param_name.replace("blocks.", "") for all "blocks."
        2. param_name.replace(".weight", "") for param_name ending with ".weight"
        3. if param_name contains bias, ignore it.
    """
    results: Dict[str, float] = {}

    for name, param in model.named_parameters():
        if "bias" in name:
            continue

        normalized = name.replace("blocks.", "")
        if normalized.endswith(".weight"):
            normalized = normalized[:-len(".weight")]

        # Weights metrics
        w = param.data
        w_sketch = sketch_tensor(w)
        results[f"weights/{normalized}/rms_norm"] = float(w_sketch[0])
        results[f"weights/{normalized}/abs_max"] = float(w_sketch[3])

        # Gradients metrics (skip if no grad)
        if param.grad is not None:
            g = param.grad.data
            g_sketch = sketch_tensor(g)
            results[f"grads/{normalized}/rms_norm"] = float(g_sketch[0])
            results[f"grads/{normalized}/abs_max"] = float(g_sketch[3])

    return results


layer_idx = 0
op_idx = 0


def set_layer(i):
    global layer_idx
    layer_idx = i

def set_op(i):
    global op_idx
    op_idx = i


def save_tensor(name, tensor):
    global layer_idx, op_idx
    file_name = f"layer{layer_idx}_op{op_idx}_{name}.npy"
    vis_dir = "/tmp/uttt_nvs_debug/visualization/"
    os.makedirs(vis_dir, exist_ok=True)

    tensor = tensor[0].float().detach().cpu().numpy()

    np.save(os.path.join(vis_dir, file_name), tensor)

    