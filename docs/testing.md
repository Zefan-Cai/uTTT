# Testing and evidence levels

## The validation ladder

| Level | Command / exercise | What it establishes | What it does not establish |
|---|---|---|---|
| Static configuration | `python -m tools.check_configs --check-catalog` | Released config structure and selected cross-field invariants | Model imports, data access or CUDA correctness |
| CPU regression | `python -m pytest -q` | Checkpoint/data/launcher/tool behavior | Full model execution or training quality |
| Syntax | `python -m compileall -q tools tests uttt_nvs uttt_llm` | Python source parses | Imports or runtime correctness |
| Construction | Domain `smoke_configs --construct` | Real classes import and initialize | Fused forward/backward correctness |
| GPU kernel | Explicit CUDA kernel tests | Covered primitive cases on the tested stack | All shapes, all hardware or end-to-end quality |
| Bounded training | Several real forward/backward/optimizer steps | End-to-end viability for that recipe | Convergence or published performance |
| Resume | Stop and continue a completed checkpoint | Restore behavior for that model/topology | Bitwise equivalence under all distributed settings |
| Full evaluation | Matched checkpoints/data/protocol | Reproducible metrics with recorded coverage | Generalization outside the measured setup |

## CPU test setup

From the repository root, create a Python 3.10 or 3.11 environment. Install
PyTorch 2.8.0 (CPU wheel on Linux, native wheel on macOS), then:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest tests/test_checkpoints.py -q
python -m pytest tests/test_launchers.py -q
python -m pytest tests/test_data_tools.py tests/test_configs.py -q
python -m pytest -q
```

`pytest.ini` selects only the root `tests/` CPU suite by default. Existing GPU
tests are not silently reclassified as CPU tests. The fake torchrun executable
in launcher tests checks exact argument forwarding and process exit status; it
does not create a distributed process group.

### Regression coverage

- Standalone/multi-node command construction, node-rank forwarding, invalid
  process counts, partial environments, absolute paths and space-containing args.
- Optional LLM job names without accidentally consuming the first trainer flag.
- Atomic checkpoint publication, failed-save cleanup, numeric discovery,
  count-based retention and corrupt-checkpoint fallback.
- Explicit missing/corrupt restore sources, fresh-run behavior and fine-tuning
  with reset counters but preserved weights.
- Deterministic toy images, sufficient default evaluation views, portable paths,
  right-handed camera frames and manifest validation failures.
- All released configurations, invalid expert counts, incompatible dimensions,
  batch arithmetic and missing source classes.
- Documentation links and catalog freshness.

## CUDA checks remain separate

In the full NVS environment:

```bash
python -m uttt_nvs.tests.smoke_configs --construct --strict
python -m pytest uttt_nvs/tests/test_triton_permute.py -q
```

The strict construction mode treats missing optional attention dependencies as
failure rather than full coverage. The permutation test covers that primitive,
not every kernel in the repository. For changes to grouped GEMM/SwiGLU or
fast-weight updates, add targeted reference comparisons for the changed path.

In the full LLM environment:

```bash
(cd uttt_llm && python -m tests.smoke_configs --construct)
```

Do not use `--stub-flash-attn` for claimed CUDA validation. That option replaces
the attention implementation for diagnosis and cannot establish fidelity of
the original kernels.

## Continuous integration

[CPU release checks](../.github/workflows/ci.yml) runs on Ubuntu with Python
3.10 and 3.11 for pushes and pull requests. It installs CPU torch, runs the
regression suite, checks all configs/catalog output and parses Python/shell
sources. It needs no GPU, private dataset, W&B secret or trained checkpoint.
Its token has read-only repository-content permissions.

The CI workflow does not run training, benchmark kernels, publish a project
page or deploy a model. A green badge should be described as CPU release
validation, not as paper reproduction.

## Recording local verification

Report the exact commands and environment, the number of passed/failed/skipped
tests, and the untested layers. If only CPU checks ran, explicitly say that
CUDA forward/backward, multi-node communication and paper metrics were not
rerun. Keep timing claims separate from compilation/autotuning and synchronize
the GPU when measuring kernels.
