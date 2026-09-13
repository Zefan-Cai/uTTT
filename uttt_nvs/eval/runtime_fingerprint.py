"""Deterministic ML runtime identity for reproducible NVS evaluation."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
from hashlib import sha256
from typing import Any, Mapping


RUNTIME_FINGERPRINT_SCHEMA = "nvs-runtime-fingerprint-v1"


def _distribution_version(*names: str) -> str:
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return "not-installed"


def _nvidia_driver_version() -> str:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return "unavailable"
    versions = sorted({line.strip() for line in result.stdout.splitlines() if line.strip()})
    return ",".join(versions) if versions else "unavailable"


def deterministic_runtime_fingerprint(device: Any) -> dict[str, Any]:
    """Return only deterministic runtime properties, never host/GPU identity.

    GPU UUID, PCI bus, hostname, process, shard, and visible-device ordinal are
    deliberately absent. The model name and compute capability are included.
    """
    import numpy as np
    import torch

    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA runtime fingerprint requested without CUDA")
        ordinal = torch.cuda.current_device() if torch_device.index is None else torch_device.index
        gpu_model = torch.cuda.get_device_name(ordinal)
        capability = list(torch.cuda.get_device_capability(ordinal))
    else:
        gpu_model = None
        capability = None
    try:
        import torchvision

        torchvision_version = str(torchvision.__version__)
    except Exception:
        torchvision_version = _distribution_version("torchvision")
    cudnn_version = torch.backends.cudnn.version()
    return {
        "schema_version": RUNTIME_FINGERPRINT_SCHEMA,
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "torch": str(torch.__version__),
        "torchvision": torchvision_version,
        "numpy": str(np.__version__),
        "lpips": _distribution_version("lpips"),
        "pytorch_msssim": _distribution_version("pytorch-msssim", "pytorch_msssim"),
        "torch_cuda_build": str(torch.version.cuda),
        "cudnn": None if cudnn_version is None else int(cudnn_version),
        "gpu_model": gpu_model,
        "gpu_compute_capability": capability,
        "nvidia_driver": _nvidia_driver_version(),
        "numeric_flags": {
            "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "default_dtype": str(torch.get_default_dtype()),
        },
    }


def runtime_fingerprint_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return sha256(encoded).hexdigest()

_RUNTIME_KEYS = {
    "schema_version", "python", "python_implementation", "torch",
    "torchvision", "numpy", "lpips", "pytorch_msssim",
    "torch_cuda_build", "cudnn", "gpu_model",
    "gpu_compute_capability", "nvidia_driver", "numeric_flags",
}
_NUMERIC_FLAG_KEYS = {
    "deterministic_algorithms", "cudnn_deterministic", "cudnn_benchmark",
    "cuda_matmul_allow_tf32", "cudnn_allow_tf32", "default_dtype",
}
_FORBIDDEN_KEY_TOKENS = (
    "hostname", "uuid", "pci", "bus", "ordinal", "process", "shard",
)


def validate_runtime_fingerprint(
    payload: Mapping[str, Any], *, require_cuda: bool = True,
) -> None:
    """Validate the exact v1 schema and reject host/shard identity leakage."""
    if not isinstance(payload, Mapping):
        raise ValueError("runtime fingerprint must be an object")

    def walk_keys(value: Any) -> None:
        if isinstance(value, Mapping):
            for raw_key, nested in value.items():
                key = str(raw_key).lower()
                if any(token in key for token in _FORBIDDEN_KEY_TOKENS):
                    raise ValueError(f"forbidden runtime fingerprint key: {raw_key}")
                walk_keys(nested)
        elif isinstance(value, (list, tuple)):
            for nested in value:
                walk_keys(nested)

    walk_keys(payload)
    if set(payload) != _RUNTIME_KEYS:
        missing = sorted(_RUNTIME_KEYS - set(payload))
        extra = sorted(set(payload) - _RUNTIME_KEYS)
        raise ValueError(f"runtime fingerprint fields changed; missing={missing}, extra={extra}")
    if payload.get("schema_version") != RUNTIME_FINGERPRINT_SCHEMA:
        raise ValueError("runtime fingerprint schema version changed")
    for key in (
        "python", "python_implementation", "torch", "torchvision", "numpy",
        "lpips", "pytorch_msssim", "torch_cuda_build", "nvidia_driver",
    ):
        if not isinstance(payload.get(key), str) or not payload[key]:
            raise ValueError(f"runtime fingerprint {key} must be a nonempty string")
    flags = payload.get("numeric_flags")
    if not isinstance(flags, Mapping) or set(flags) != _NUMERIC_FLAG_KEYS:
        raise ValueError("runtime numeric flag schema changed")
    for key in _NUMERIC_FLAG_KEYS - {"default_dtype"}:
        if not isinstance(flags.get(key), bool):
            raise ValueError(f"runtime numeric flag {key} must be boolean")
    if not isinstance(flags.get("default_dtype"), str) or not flags["default_dtype"]:
        raise ValueError("runtime default_dtype must be a nonempty string")
    cudnn = payload.get("cudnn")
    if cudnn is not None and (not isinstance(cudnn, int) or isinstance(cudnn, bool)):
        raise ValueError("runtime cuDNN version must be integer or null")
    capability = payload.get("gpu_compute_capability")
    if require_cuda:
        if not isinstance(payload.get("gpu_model"), str) or not payload["gpu_model"]:
            raise ValueError("formal CUDA runtime lacks GPU model")
        if (
            not isinstance(capability, list) or len(capability) != 2
            or any(not isinstance(value, int) for value in capability)
        ):
            raise ValueError("formal CUDA runtime lacks compute capability")
        if payload["torch_cuda_build"] in {"None", "none", ""}:
            raise ValueError("formal CUDA runtime lacks torch CUDA build")
        if cudnn is None:
            raise ValueError("formal CUDA runtime lacks cuDNN")
        if payload["nvidia_driver"] == "unavailable":
            raise ValueError("formal CUDA runtime lacks NVIDIA driver")
    elif payload.get("gpu_model") is not None or capability is not None:
        if not isinstance(payload.get("gpu_model"), str):
            raise ValueError("GPU model must be a string or null")
