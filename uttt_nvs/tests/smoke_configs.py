"""Construction smoke test for every shipped config.

For each YAML under ``configs/`` this resolves the model class path, builds the
model on CPU (meta-free, small override where possible) and reports failures.

It exists because config files name their model classes as fully qualified
import strings that are resolved at runtime, so a wrong path is only detected
when the model is constructed -- never at import time.

Usage:
    python -m uttt_nvs.tests.smoke_configs
    python -m uttt_nvs.tests.smoke_configs --construct   # also instantiate
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "uttt_nvs" / "configs"


def iter_configs():
    for path in sorted(CONFIG_ROOT.rglob("*.yaml")):
        if path.parent.name == "_base":
            continue
        yield path


def collect_class_paths(node, found):
    """Walk a parsed config and collect every dotted class path it names."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key in {"class_name", "type"} and isinstance(value, str) and "." in value:
                found.append(value)
            else:
                collect_class_paths(value, found)
    elif isinstance(node, list):
        for item in node:
            collect_class_paths(item, found)
    return found


OPTIONAL_DEPS = ("flash_attn",)


class OptionalDependencyMissing(Exception):
    """The class exists but an optional third-party package is unusable."""


def resolve(dotted: str):
    """Resolve 'pkg.mod.Class' the same way the trainer does at runtime."""
    parts = dotted.split(".")
    module_name = ".".join(parts[:-1])
    attr = parts[-1]
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        # Only a genuinely absent optional package counts as SKIP. A package
        # that is installed but broken (for example a flash-attn wheel built
        # against a different torch) must FAIL loudly -- silently skipping it
        # would defeat the reason this check exists.
        cause, seen = exc, set()
        while cause is not None and id(cause) not in seen:
            seen.add(id(cause))
            if isinstance(cause, ModuleNotFoundError):
                root = (getattr(cause, "name", "") or "").split(".")[0]
                if root in OPTIONAL_DEPS:
                    raise OptionalDependencyMissing(
                        f"optional dependency '{root}' is not installed"
                    ) from exc
            cause = cause.__cause__ or cause.__context__
        raise
    return getattr(module, attr)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--construct", action="store_true",
                        help="also build the model on CPU, not just resolve class paths")
    parser.add_argument("--strict", action="store_true",
                        help="treat skipped configs as failures (no optional dependency may be missing)")
    args = parser.parse_args()

    configs = list(iter_configs())
    print(f"Found {len(configs)} configs under {CONFIG_ROOT}\n")

    failures = []
    skips = []
    all_paths = set()
    for path in configs:
        rel = path.relative_to(CONFIG_ROOT)
        try:
            cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception as exc:
            failures.append((rel, "yaml", exc))
            print(f"  FAIL  {rel}  (yaml: {exc})")
            continue

        dotted_paths = collect_class_paths(cfg, [])
        if not dotted_paths:
            failures.append((rel, "no-class", "no class_name/type found"))
            print(f"  FAIL  {rel}  (no class path found)")
            continue

        bad = None
        skipped = None
        for dotted in dotted_paths:
            all_paths.add(dotted)
            try:
                resolve(dotted)
            except OptionalDependencyMissing as exc:
                skipped = (dotted, exc)
            except Exception as exc:
                bad = (dotted, exc)
                break

        if args.construct and not bad and not skipped:
            try:
                from easydict import EasyDict
                model_cls = resolve(cfg["model"]["class_name"])
                model = model_cls(EasyDict(cfg))
                n_params = sum(p.numel() for p in model.parameters())
                del model
                print(f"  ok    {rel}  ({n_params / 1e6:.1f}M params)")
                continue
            except OptionalDependencyMissing as exc:
                skipped = (cfg["model"]["class_name"], exc)
            except Exception as exc:
                bad = (cfg["model"]["class_name"], exc)

        if bad:
            failures.append((rel, bad[0], bad[1]))
            print(f"  FAIL  {rel}\n          {bad[0]}\n          {type(bad[1]).__name__}: {bad[1]}")
        elif skipped:
            skips.append((rel, skipped[0], skipped[1]))
            print(f"  SKIP  {rel}  ({skipped[1]})")
        else:
            print(f"  ok    {rel}  ({len(dotted_paths)} class paths)")

    print(f"\nDistinct class paths referenced ({len(all_paths)}):")
    for dotted in sorted(all_paths):
        print(f"  {dotted}")

    resolved = len(configs) - len(failures) - len(skips)
    print(f"\n{resolved}/{len(configs)} configs resolved.")
    if skips:
        print(f"{len(skips)} skipped (optional dependency unavailable):")
        for rel, dotted, exc in skips:
            print(f"  {rel}: {exc}")
    if failures:
        print(f"{len(failures)} FAILED.")
        return 1
    if skips and args.strict:
        print(f"{len(skips)} skipped and --strict was given, so this is a failure.")
        return 1
    if skips:
        print(f"No configs failed, but {len(skips)} could not be checked because an "
              f"optional dependency is missing. Coverage is {resolved}/{len(configs)}, "
              f"not {len(configs)}/{len(configs)}. Use --strict to make this an error.")
        return 0
    print(f"All {len(configs)} configs checked.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
