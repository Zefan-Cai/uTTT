"""Shared registry, configuration, and checkpoint helpers for NVS evaluation."""

from __future__ import annotations

import hashlib
import importlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_BATCH_KEYS = ("fxfycxcy", "c2w", "image", "index")
NATURAL_OUTPUT_IDENTITY_ASSET_KEYS = (
    "registry_sha256",
    "checkpoint_sha256",
    "config_sha256",
    "dataset_manifest_sha256",
    "view_manifest_sha256",
    "code_commit",
    "redundancy_code_sha256",
)


@dataclass(frozen=True)
class ResolvedExperiment:
    """One registry entry with all filesystem paths resolved."""

    raw: Mapping[str, Any]
    dataset: Mapping[str, Any]
    defaults: Mapping[str, Any]
    config_path: Path
    checkpoint_path: Path
    local_manifest_path: Path
    view_manifest_path: Path
    output_dir: Path

    @property
    def id(self) -> str:
        return str(self.raw["id"])

    @property
    def model_name(self) -> str:
        return str(self.raw.get("paper_model", self.raw.get("model", self.id)))

    @property
    def dataset_id(self) -> str:
        return str(self.raw["dataset"])


def _resolve_path(path: str | Path, base: Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else (base / value).resolve()


def load_registry(path: str | Path) -> dict[str, Any]:
    registry_path = Path(path).expanduser().resolve()
    registry = yaml.safe_load(registry_path.read_text())
    if not isinstance(registry, dict):
        raise ValueError(f"Registry must contain a mapping: {registry_path}")
    for required in ("defaults", "datasets", "experiments"):
        if required not in registry:
            raise ValueError(f"Registry is missing top-level key {required!r}")
    experiments = registry["experiments"]
    if not isinstance(experiments, list):
        raise ValueError("registry.experiments must be a list")
    datasets = registry["datasets"]
    if isinstance(datasets, list):
        dataset_ids = [str(item["id"]) for item in datasets]
        duplicates = sorted(
            {item for item in dataset_ids if dataset_ids.count(item) > 1}
        )
        if duplicates:
            raise ValueError(f"Duplicate dataset ids: {duplicates}")
        registry["datasets"] = {
            str(item["id"]): item for item in datasets
        }
    elif not isinstance(datasets, dict):
        raise ValueError("registry.datasets must be a mapping or list")
    ids = [str(item["id"]) for item in experiments]
    duplicates = sorted({item for item in ids if ids.count(item) > 1})
    if duplicates:
        raise ValueError(f"Duplicate experiment ids: {duplicates}")
    return registry


def resolve_experiment(
    registry_path: str | Path,
    experiment_id: str,
    output_root: str | Path | None = None,
) -> ResolvedExperiment:
    registry_path = Path(registry_path).expanduser().resolve()
    registry = load_registry(registry_path)
    defaults = registry["defaults"]
    datasets = registry["datasets"]
    candidates = [
        item for item in registry["experiments"] if str(item["id"]) == experiment_id
    ]
    if len(candidates) != 1:
        raise KeyError(f"Expected one registry entry for {experiment_id!r}")
    experiment = candidates[0]
    dataset_id = str(experiment["dataset"])
    if dataset_id not in datasets:
        raise KeyError(f"Unknown dataset {dataset_id!r} in {experiment_id!r}")
    dataset = datasets[dataset_id]

    checkpoint_root = _resolve_path(
        defaults["checkpoint_root"],
        REPO_ROOT,
    )
    resolved_output_root = _resolve_path(
        output_root or defaults["output_root"],
        REPO_ROOT,
    )
    config_path = _resolve_path(experiment["config"], REPO_ROOT)
    checkpoint_path = _resolve_path(experiment["checkpoint"], checkpoint_root)
    local_manifest_path = _resolve_path(dataset["local_manifest"], REPO_ROOT)
    view_manifest_path = _resolve_path(dataset["view_manifest"], REPO_ROOT)
    output_dir = resolved_output_root / dataset_id / experiment_id

    return ResolvedExperiment(
        raw=experiment,
        dataset=dataset,
        defaults=defaults,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        local_manifest_path=local_manifest_path,
        view_manifest_path=view_manifest_path,
        output_dir=output_dir,
    )


def validate_resolved_experiment(experiment: ResolvedExperiment) -> None:
    missing = [
        path
        for path in (
            experiment.config_path,
            experiment.checkpoint_path,
            experiment.local_manifest_path,
            experiment.view_manifest_path,
        )
        if not path.exists()
    ]
    if missing:
        rendered = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Required evaluation inputs are missing:\n{rendered}")


def validate_v_gap(experiment: ResolvedExperiment, config: Any) -> Any:
    """Validate the per-experiment attention-gap protocol declaration."""

    if isinstance(config, Mapping):
        params = config["model"]["block_config"]["params"]
    else:
        params = config.model.block_config.params
    actual = params.get("v_gap")
    expected = experiment.raw.get("expected_v_gap")
    if actual != expected:
        raise ValueError(
            f"{experiment.id}: expected v_gap={expected!r}, got {actual!r} "
            f"from {experiment.config_path}"
        )
    if actual is not None and not bool(
        experiment.raw.get("allow_nonnull_v_gap", False)
    ):
        raise ValueError(
            f"{experiment.id}: non-null v_gap={actual!r} requires an explicit "
            "allow_nonnull_v_gap registry declaration"
        )
    return actual


def load_model_config(experiment: ResolvedExperiment) -> Any:
    """Load the training YAML and replace only evaluation data settings."""

    from easydict import EasyDict
    from omegaconf import OmegaConf

    loaded = OmegaConf.load(experiment.config_path)
    container = OmegaConf.to_container(loaded, resolve=True)
    config = EasyDict(container)
    num_views = int(experiment.defaults.get("num_views", 24))
    config.training.dataset_path = str(experiment.local_manifest_path)
    config.training.num_views = num_views
    config.training.eval_num_views = num_views
    config.training.num_workers = int(experiment.defaults.get("num_workers", 4))
    config.training.batch_size_per_gpu = int(
        experiment.defaults.get("batch_size_per_gpu", 1)
    )
    validate_v_gap(experiment, config)
    return config


def import_model(config: Any) -> Any:
    module_name, class_name = str(config.model.class_name).rsplit(".", 1)
    model_class = getattr(importlib.import_module(module_name), class_name)
    return model_class(config)


def normalize_state_dict(
    state_dict: Mapping[str, Any],
) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for original_key, value in state_dict.items():
        key = original_key.replace("_checkpoint_wrapped_module.", "")
        key = key.replace("_orig_mod.", "")
        while key.startswith("module."):
            key = key[len("module.") :]
        normalized[key] = value
    return normalized


def load_checkpoint_strict(model: Any, checkpoint_path: Path) -> dict[str, Any]:
    """Load model weights and fail on every missing or unexpected tensor."""

    import torch

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")
    state_dict = normalize_state_dict(checkpoint["model"])
    model.load_state_dict(state_dict, strict=True)
    return {
        "fwdbwd_pass_step": checkpoint.get("fwdbwd_pass_step"),
        "param_update_step": checkpoint.get("param_update_step"),
    }


def model_batch(batch: Mapping[str, Any], device: Any) -> dict[str, Any]:
    """Strip metadata keys before calling models that slice every batch value."""

    missing = [key for key in MODEL_BATCH_KEYS if key not in batch]
    if missing:
        raise KeyError(f"Batch is missing model keys: {missing}")
    return {
        key: batch[key].to(device, non_blocking=True)
        for key in MODEL_BATCH_KEYS
    }


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluation_code_sha256() -> str:
    """Hash the evaluator implementation, including uncommitted source edits."""

    paths = sorted((REPO_ROOT / "eval").glob("*.py"))
    _launcher = REPO_ROOT / "scripts" / "launch_nvs_eval.py"
    if _launcher.exists():
        paths.append(_launcher)
    digest = hashlib.sha256()
    for path in paths:
        relative_path = path.relative_to(REPO_ROOT).as_posix()
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def natural_output_source_fingerprint_sha256(
    *,
    experiment_id: str,
    protocol_version: str,
    run_signature: str,
    identity_assets: Mapping[str, Any],
    runtime_sha256: str,
    scene_count: int,
    scene_file_sha256: Mapping[str, str],
) -> str:
    """Hash the exact, path-independent identity of one natural-output root."""
    scalar_fields = {
        "experiment_id": experiment_id,
        "protocol_version": protocol_version,
        "run_signature": run_signature,
        "runtime_sha256": runtime_sha256,
    }
    invalid_scalars = [
        key
        for key, value in scalar_fields.items()
        if not isinstance(value, str) or not value
    ]
    if invalid_scalars:
        raise ValueError(
            f"natural-output fingerprint has invalid fields {invalid_scalars}"
        )
    if not isinstance(identity_assets, Mapping):
        raise ValueError("natural-output identity_assets must be an object")
    expected_asset_keys = set(NATURAL_OUTPUT_IDENTITY_ASSET_KEYS)
    if set(identity_assets) != expected_asset_keys:
        raise ValueError(
            "natural-output identity_assets must contain exactly "
            f"{sorted(expected_asset_keys)}"
        )
    if any(
        not isinstance(identity_assets[key], str) or not identity_assets[key]
        for key in NATURAL_OUTPUT_IDENTITY_ASSET_KEYS
    ):
        raise ValueError(
            "natural-output identity asset values must be non-empty strings"
        )
    if (
        isinstance(scene_count, bool)
        or not isinstance(scene_count, int)
        or scene_count <= 0
    ):
        raise ValueError("natural-output scene_count must be a positive integer")
    if (
        not isinstance(scene_file_sha256, Mapping)
        or len(scene_file_sha256) != scene_count
    ):
        raise ValueError(
            "natural-output scene_file_sha256 must contain exactly scene_count entries"
        )
    for name, digest in scene_file_sha256.items():
        if (
            not isinstance(name, str)
            or not name
            or Path(name).name != name
            or not isinstance(digest, str)
            or not digest
        ):
            raise ValueError(
                "natural-output scene hashes require basename keys and non-empty strings"
            )
    payload = {
        **scalar_fields,
        "identity_assets": {
            key: identity_assets[key] for key in NATURAL_OUTPUT_IDENTITY_ASSET_KEYS
        },
        "scene_count": scene_count,
        "scene_file_sha256": dict(scene_file_sha256),
    }
    return canonical_json_sha256(payload)


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
    ).strip()


def select_experiments(
    registry_path: str | Path,
    datasets: Iterable[str] | None = None,
) -> list[str]:
    registry = load_registry(registry_path)
    allowed = set(datasets or [])
    return [
        str(item["id"])
        for item in registry["experiments"]
        if not allowed or str(item["dataset"]) in allowed
    ]
