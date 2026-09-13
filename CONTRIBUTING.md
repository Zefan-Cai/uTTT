# Contributing to uTTT

This repository contains research implementations and reproducibility tooling.
Prefer focused changes that preserve the experimental interpretation of the
released configurations. See the [architecture guide](docs/architecture.md)
and [repository split](docs/migration.md) before renaming models or changing
remotes.

## Before editing

1. Identify the affected domain, configuration family and implementation.
2. State the expected behavior and a check that can fail before the fix.
3. Separate correctness, optimization and new research ablations. Do not mix
   a kernel rewrite with unrelated config or metric changes.
4. Preserve `model_type`, checkpoint keys, data splits and evaluation protocols
   unless the change explicitly includes a tested migration.

## Development environment

Use Python 3.10 or 3.11 and install CPU torch 2.8.0 plus
`requirements-dev.txt` for infrastructure work. Model/kernel work requires the
full domain CUDA environment. Do not add flash-attn to the CPU test path just
to import an unrelated configuration utility.

```bash
python -m tools.check_configs --check-catalog
python -m pytest -q
python -m compileall -q tools tests uttt_nvs uttt_llm
bash -n scripts/launch_common.sh
bash -n uttt_nvs/train/launch.sh
bash -n uttt_llm/launch/train.sh
git diff --check
```

When changing a released configuration, regenerate the catalog with
`python -m tools.check_configs --write-catalog` and review the generated diff.
The catalog is a description of configs, not a performance database.

## Tests and evidence

Add a small CPU regression for launcher, checkpoint, data or validation-tool
changes. Keep CUDA tests explicit. Kernel changes require numerical comparison
of outputs and gradients on representative shapes, dtypes, expert loads and
sequence lengths, followed by a bounded end-to-end training smoke test.

Report failures and skips rather than treating skipped tests as coverage.
If CUDA or real data is unavailable, say which behavior remains unverified;
do not replace that evidence with simulated benchmark numbers. Details are in
the [testing guide](docs/testing.md).

## Pull-request checklist

- Describe the bug/feature and affected configs/packages.
- Include the minimal reproducer, validation commands and results.
- Explain checkpoint/config compatibility and any intentional behavior change.
- Update the relevant guide and changelog for user-visible changes.
- Do not commit datasets, checkpoints, `.env`, API keys, private path registries
  or generated training outputs.
- Keep benchmark settings, evaluation coverage and seeds comparable across
  methods; report deviations explicitly.

Use the MIT license already in the repository. Preserve upstream attribution;
do not silently relicense code or redistribute datasets/checkpoints whose
licenses or provenance are unknown.
