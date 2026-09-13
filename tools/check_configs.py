"""Validate released configuration invariants without importing CUDA models."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import yaml

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "docs/configuration-catalog.md"


def positive(config: dict, key: str) -> int:
    value = config.get(key)
    if type(value) is not int or value <= 0:
        raise ValueError(f"{key} must be a positive integer, got {value!r}")
    return value


def divides(numerator: int, denominator: int, label: str) -> None:
    if numerator % denominator:
        raise ValueError(f"{label}: {numerator} is not divisible by {denominator}")


def source_class(dotted: str, root: Path) -> None:
    module, name = dotted.rsplit(".", 1)
    path = root.joinpath(*module.split(".")).with_suffix(".py")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    if not any(isinstance(node, ast.ClassDef) and node.name == name for node in tree.body):
        raise ValueError(f"Class {dotted} is not defined in {path}")


def llm_model_types(root: Path) -> set[str]:
    package = root / "uttt_llm/flame/custom_models"
    registry = ast.parse((package / "__init__.py").read_text(encoding="utf-8"))
    imported = set()
    for node in registry.body:
        if isinstance(node, ast.ImportFrom) and node.level == 1:
            if node.module:
                imported.add(node.module.split(".")[0])
            else:
                imported.update(alias.name for alias in node.names)
    model_types = {"transformer"}
    for name in sorted(imported):
        for path in (package / name).rglob("configuration*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
                    if any(isinstance(target, ast.Name) and target.id == "model_type" for target in node.targets):
                        if isinstance(node.value.value, str):
                            model_types.add(node.value.value)
    return model_types


def validate_nvs(config: dict, root: Path = ROOT) -> None:
    model, training = config["model"], config["training"]
    if not isinstance(config.get("exp_name"), str) or not config["exp_name"].strip():
        raise ValueError("exp_name must be a non-empty string")
    for key in ("image_size", "patch_size", "dim", "layers"):
        positive(model, key)
    divides(model["image_size"], model["patch_size"], "image_size / patch_size")
    for key in ("batch_size_per_gpu", "total_batch_size", "grad_accum_steps",
                "num_views", "eval_num_views", "max_fwdbwd_passes", "checkpoint_every"):
        positive(training, key)
    divides(training["total_batch_size"], training["batch_size_per_gpu"] * training["grad_accum_steps"],
            "global batch / (per-GPU batch * accumulation)")
    source_class(model["class_name"], root)
    source_class(training["dataset_name"], root)
    block = model["block_config"]
    source_class(block["type"], root)
    params = block.get("params", {})
    for key in ("attn_head_dim", "fw_head_dim"):
        if key in params:
            divides(model["dim"], positive(params, key), f"model.dim / {key}")
    if "num_experts" in params:
        experts = positive(params, "num_experts")
        active = positive(params, "num_active_experts")
        if active > experts:
            raise ValueError("num_active_experts cannot exceed num_experts")
    if "num_shared_pools" in params:
        divides(model["layers"], positive(params, "num_shared_pools"), "layers / num_shared_pools")
    if params.get("aggregate_write") and params.get("pool_mode", "shared") != "shared":
        raise ValueError("aggregate_write requires a shared pool")
    if params.get("pool_mode", "shared") not in {"shared", "per_block"}:
        raise ValueError("pool_mode must be shared or per_block")


def validate_llm(config: dict, model_types: set[str]) -> None:
    if config.get("model_type") not in model_types:
        raise ValueError(f"Unknown model_type: {config.get('model_type')!r}")
    for key in ("hidden_size", "num_hidden_layers", "vocab_size", "max_position_embeddings"):
        positive(config, key)
    for key in ("num_heads", "fw_num_heads"):
        if key in config:
            divides(config["hidden_size"], positive(config, key), f"hidden_size / {key}")
    if "chunk_size" in config:
        positive(config, "chunk_size")
    if "memory_num_experts" in config:
        experts = positive(config, "memory_num_experts")
        active = positive(config, "memory_num_active_experts")
        if active > experts:
            raise ValueError("memory_num_active_experts cannot exceed memory_num_experts")
    if config.get("memory_update_aggregate_write") and config.get("memory_pool_mode") != "shared":
        raise ValueError("memory_update_aggregate_write requires memory_pool_mode=shared")


def validate_experiment(config: dict) -> None:
    training = config["training"]
    for key in ("batch_size", "seq_len", "context_len", "gradient_accumulation_steps", "steps"):
        positive(training, key)
    if "expected_global_batch_size" in training:
        divides(positive(training, "expected_global_batch_size"),
                training["batch_size"] * training["gradient_accumulation_steps"],
                "global batch / (per-device batch * accumulation)")
    for key in ("config", "tokenizer_path"):
        if not isinstance(config["model"].get(key), str) or not config["model"][key]:
            raise ValueError(f"model.{key} must be a non-empty string")
    if config.get("eval", {}).get("per_token_loss_from_hidden"):
        positive(config["eval"], "lm_head_chunk_size")


def inspect_configs(root: Path = ROOT) -> tuple[list, list[str]]:
    entries, errors = [], []
    model_types = llm_model_types(root)
    for domain, extension in (("uttt_nvs", "yaml"), ("uttt_llm", "json"), ("uttt_llm", "toml")):
        paths = sorted((root / domain / "configs").rglob(f"*.{extension}"))
        if not paths:
            errors.append(f"{domain}: no .{extension} configurations found")
        for path in paths:
            try:
                text = path.read_text(encoding="utf-8")
                if extension == "yaml":
                    config = yaml.safe_load(text)
                    validate_nvs(config, root)
                elif extension == "json":
                    config = json.loads(text)
                    validate_llm(config, model_types)
                else:
                    config = tomllib.loads(text)
                    validate_experiment(config)
                entries.append((path.relative_to(root), config))
            except (ValueError, TypeError, KeyError, AttributeError, OSError, SyntaxError, yaml.YAMLError) as exc:
                errors.append(f"{path.relative_to(root)}: {exc}")
    return entries, errors


def render_catalog(entries: list) -> str:
    lines = ["# Configuration catalog", "",
             "Generated from the checked-in configurations by `python -m tools.check_configs --write-catalog`.",
             "Do not edit tables by hand. Counts describe configuration files, not validated training runs.", "",
             "See [configuration semantics](configuration.md) before changing ownership, routing or update schedules.", "",
             "## Novel view synthesis", "",
             "GPU count is total_batch_size / (batch_size_per_gpu × grad_accum_steps); it is not a memory-fit guarantee.",
             "Steps are the configured forward/backward pass limit, before optional epoch rounding.", "",
             "| Config | Block | Pool | Experts / active | Resolution | Layers × width | Global batch | GPUs | Steps |",
             "|---|---|---|---|---:|---:|---:|---:|---:|"]
    for path, config in entries:
        if path.suffix != ".yaml":
            continue
        model, training = config["model"], config["training"]
        block = model["block_config"]
        params = block.get("params", {})
        experts = f"{params['num_experts']} / {params['num_active_experts']}" if "num_experts" in params else "—"
        pool = params.get("pool_mode", "see block")
        if "num_shared_pools" in params:
            pool = f"{params['num_shared_pools']} shared pools"
        gpus = training["total_batch_size"] // (training["batch_size_per_gpu"] * training["grad_accum_steps"])
        lines.append(f"| [{path.relative_to('uttt_nvs/configs')}](../{path.as_posix()}) | `{block['type'].split('.')[-2]}` | {pool} | {experts} | {model['image_size']}² | {model['layers']} × {model['dim']} | {training['total_batch_size']} | {gpus} | {training['max_fwdbwd_passes']:,} |")
    lines.extend(["", "## Language-model architectures", "",
                  "Architecture JSONs must be paired with a training/evaluation TOML. E/K denotes routed pool size and active experts.", "",
                  "| Config | model_type | Layers × width | Context | Chunk | E / K | LB mode / update rate |",
                  "|---|---|---:|---:|---:|---:|---|"])
    for path, config in entries:
        if path.suffix != ".json":
            continue
        experts = f"{config['memory_num_experts']} / {config['memory_num_active_experts']}" if "memory_num_experts" in config else "—"
        balancing = f"{config.get('memory_lb_mode', 'default')} / {config.get('memory_loss_free_update_rate', '—')}"
        lines.append(f"| [{path.relative_to('uttt_llm/configs')}](../{path.as_posix()}) | `{config['model_type']}` | {config['num_hidden_layers']} × {config['hidden_size']} | {config['max_position_embeddings']:,} | {config.get('chunk_size', '—')} | {experts} | {balancing} |")
    lines.extend(["", "## Language-model jobs", "",
                  "| Config | Per-device batch | Accumulation | Expected global batch | Sequence length | Steps |",
                  "|---|---:|---:|---:|---:|---:|"])
    for path, config in entries:
        if path.suffix == ".toml":
            training = config["training"]
            lines.append(f"| [{path.name}](../{path.as_posix()}) | {training['batch_size']} | {training['gradient_accumulation_steps']} | {training.get('expected_global_batch_size', 'not enforced')} | {training['seq_len']:,} | {training['steps']:,} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--write-catalog", action="store_true")
    modes.add_argument("--check-catalog", action="store_true")
    args = parser.parse_args()
    entries, errors = inspect_configs()
    if not errors and args.write_catalog:
        CATALOG.write_text(render_catalog(entries), encoding="utf-8")
    if not errors and args.check_catalog:
        if not CATALOG.is_file() or CATALOG.read_text(encoding="utf-8") != render_catalog(entries):
            errors.append("Configuration catalog is stale; run with --write-catalog.")
    for error in errors:
        print(f"FAIL {error}")
    print(f"{len(entries)} configurations checked; {len(errors)} errors. Static checks only; CUDA/data/checkpoints are not validated.")
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
