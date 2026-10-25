"""Every NVS evaluation registry entry must resolve to files in the release."""

from pathlib import Path

import pytest

from uttt_nvs.eval import common

ROOT = Path(__file__).resolve().parents[1]
REGISTRIES = sorted((ROOT / "uttt_nvs" / "eval").glob("experiments*.yaml"))


def test_repo_root_is_the_repository_root():
    assert common.REPO_ROOT == ROOT
    assert common.DEFAULT_REGISTRY == ROOT / "uttt_nvs" / "eval" / "experiments.yaml"
    assert common.DEFAULT_REGISTRY.is_file()


@pytest.mark.parametrize("registry", REGISTRIES, ids=lambda p: p.name)
def test_registry_configs_resolve_to_released_files(registry):
    entries = common.load_registry(registry)["experiments"]
    assert entries
    for entry in entries:
        resolved = common.resolve_experiment(registry, str(entry["id"]))
        assert resolved.config_path.is_file(), f"{entry['id']}: {resolved.config_path}"
        assert resolved.config_path.is_relative_to(ROOT)


def test_evaluation_code_fingerprint_covers_the_evaluator():
    # Hashing an empty file list would silently fingerprint nothing.
    assert sorted(common.EVAL_DIR.glob("*.py"))
    assert len(common.evaluation_code_sha256()) == 64
