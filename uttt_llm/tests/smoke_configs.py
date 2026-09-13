"""Config smoke test for the language-modeling release.

Configs identify their model by a `model_type` string that is resolved through
the transformers Auto* registries at runtime, so a broken registration or a
renamed directory only surfaces when a model is actually built. This check
catches that up front:

    python -m tests.smoke_configs               # resolve every model_type
    python -m tests.smoke_configs --construct   # also build each model on CPU

Run from the uttt_llm directory. Requires flash-attn (the models import it at
module scope).
"""

from __future__ import annotations

import argparse
import glob
import json
import sys


def _install_flash_attn_stub():
    """SDPA-backed stand-in for flash_attn, for ABI-broken installs.

    The stub is numerically a different attention implementation; it makes the
    registry importable and models constructible, nothing more.
    """
    import importlib.machinery
    import types

    import torch
    import torch.nn.functional as F

    def _fa(q, k, v, *args, **kwargs):
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return out.transpose(1, 2)

    mod = types.ModuleType("flash_attn")
    mod.flash_attn_func = _fa
    mod.flash_attn_varlen_func = _fa
    mod.__spec__ = importlib.machinery.ModuleSpec("flash_attn", None)
    sys.modules["flash_attn"] = mod
    import transformers.utils.import_utils as iu
    orig = iu._is_package_available
    iu._is_package_available = (
        lambda name, *a, **k: False if name == "flash_attn" else orig(name, *a, **k)
    )
    iu.is_flash_attn_2_available = lambda: False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--construct", action="store_true",
                        help="also instantiate each model (CPU, real weights)")
    parser.add_argument("--stub-flash-attn", action="store_true",
                        help="replace flash-attn with an SDPA stub, for machines "
                             "where the installed wheel does not match torch")
    args = parser.parse_args()

    if args.stub_flash_attn:
        _install_flash_attn_stub()

    import flame.custom_models  # noqa: F401  (registers every model_type)
    from transformers import AutoModelForCausalLM
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    paths = sorted(glob.glob("configs/main/*/*.json")) + \
            sorted(glob.glob("configs/balancing/*/*.json"))
    print(f"Found {len(paths)} configs")

    failures = []
    built = {}
    for path in paths:
        cfg = json.load(open(path))
        model_type = cfg.get("model_type")
        if model_type not in CONFIG_MAPPING:
            failures.append((path, f"model_type {model_type!r} is not registered"))
            print(f"  FAIL  {path}: unregistered model_type {model_type!r}")
            continue
        if not args.construct:
            print(f"  ok    {path}  ({model_type})")
            continue
        try:
            config = CONFIG_MAPPING[model_type](**cfg)
            model = AutoModelForCausalLM.from_config(config)
            n_params = sum(p.numel() for p in model.parameters())
            built[path] = n_params
            del model
            print(f"  ok    {path}  ({n_params / 1e6:.1f}M params)")
        except Exception as exc:
            failures.append((path, f"{type(exc).__name__}: {exc}"))
            print(f"  FAIL  {path}\n          {type(exc).__name__}: {exc}")

    print(f"\n{len(paths) - len(failures)}/{len(paths)} configs passed.")
    if failures:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
