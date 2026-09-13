import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHERS = [
    ("uttt_nvs/train/launch.sh", ["uttt_nvs/configs/ownership/obj/uttt_moe_e64a1.yaml"]),
    ("uttt_llm/launch/train.sh", ["uttt_llm/configs/exp/train_124M_32k.toml",
                                "uttt_llm/configs/main/124M/uttt_moe.json"]),
]


def launch(script, args, environment=None, cwd=ROOT):
    env = dict(os.environ)
    for key in ("MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "RANK", "NODE_RANK",
                "NPROC_PER_NODE", "NODE_SYNC", "JOB_UUID"):
        env.pop(key, None)
    env.update(environment or {})
    return subprocess.run(["bash", str(ROOT / script), *args], cwd=cwd, env=env,
                          capture_output=True, text=True, timeout=15)


@pytest.mark.parametrize("script,configs", LAUNCHERS)
def test_standalone_dry_run(script, configs):
    result = launch(script, ["--dry-run", *configs])
    assert result.returncode == 0, result.stderr
    command = shlex.split(result.stdout.split("Command: ", 1)[1])
    assert command[0] == "torchrun"
    assert "--standalone" in command
    assert all(str(ROOT / config) in command for config in configs)


@pytest.mark.parametrize("script,configs", LAUNCHERS)
@pytest.mark.parametrize("value", ["0", "-1", "abc", "08", "2.5"])
def test_invalid_gpu_count(script, configs, value):
    result = launch(script, ["--dry-run", *configs], {"NPROC_PER_NODE": value})
    assert result.returncode == 2
    assert "NPROC_PER_NODE" in result.stderr


@pytest.mark.parametrize("script,configs", LAUNCHERS)
@pytest.mark.parametrize("updates", [
    {"MASTER_PORT": ""},
    {"MASTER_ADDR": ""},
    {"WORLD_SIZE": "0"},
    {"WORLD_SIZE": "7"},
    {"WORLD_SIZE": "garbage"},
    {"MASTER_PORT": "65536"},
    {"RANK": "2"},
    {"RANK": "-1"},
    {"RANK": ""},
])
def test_invalid_distributed_environment(script, configs, updates):
    environment = {"NPROC_PER_NODE": "4", "MASTER_ADDR": "localhost", "MASTER_PORT": "29500",
                   "WORLD_SIZE": "8", "RANK": "0"}
    environment.update(updates)
    result = launch(script, ["--dry-run", *configs], environment)
    assert result.returncode == 2, result.stdout


@pytest.mark.parametrize("script,configs", LAUNCHERS)
def test_multi_node_rank_and_world_size(script, configs):
    result = launch(script, ["--dry-run", *configs], {
        "NPROC_PER_NODE": "4", "WORLD_SIZE": "8", "MASTER_ADDR": "localhost",
        "MASTER_PORT": "29500", "NODE_RANK": "1", "JOB_UUID": "test-job",
    })
    assert result.returncode == 0, result.stderr
    assert "--nnodes=2" in result.stdout
    assert "--node-rank=1" in result.stdout
    assert "--rdzv-id=test-job" in result.stdout


@pytest.mark.parametrize("script,configs", LAUNCHERS)
def test_missing_config_fails_before_torchrun(script, configs):
    result = launch(script, ["--dry-run", "missing.yaml", *configs[1:]])
    assert result.returncode == 2
    assert "Cannot read configuration" in result.stderr


@pytest.mark.parametrize("script,configs", LAUNCHERS)
def test_absolute_paths_from_another_directory(script, configs, tmp_path):
    result = launch(script, ["--dry-run", *[str(ROOT / config) for config in configs]], cwd=tmp_path)
    assert result.returncode == 0, result.stderr


def test_llm_options_without_job_name():
    script, configs = LAUNCHERS[1]
    result = launch(script, ["--dry-run", *configs, "--training.gradient_accumulation_steps", "2"])
    assert result.returncode == 0, result.stderr
    command = shlex.split(result.stdout.split("Command: ", 1)[1])
    assert command[command.index("--job.dump_folder") + 1] == "runs/uttt_moe"
    assert command[-2:] == ["--training.gradient_accumulation_steps", "2"]


@pytest.mark.parametrize("script,configs", LAUNCHERS)
def test_real_dispatch_preserves_arguments_and_exit_status(script, configs, tmp_path):
    executable = tmp_path / "torchrun"
    output = tmp_path / "arguments.json"
    executable.write_text(f"#!{sys.executable}\nimport json, os, sys\n"
                          "with open(os.environ['CAPTURE'], 'w') as handle: json.dump(sys.argv[1:], handle)\n"
                          "raise SystemExit(7)\n")
    executable.chmod(0o755)
    result = launch(script, [*configs, "--example", "value with spaces"], {
        "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"], "CAPTURE": str(output),
    })
    assert result.returncode == 7, result.stderr
    assert json.loads(output.read_text())[-2:] == ["--example", "value with spaces"]


@pytest.mark.parametrize("script,configs", LAUNCHERS)
def test_help(script, configs):
    result = launch(script, ["--help"])
    assert result.returncode == 0
    assert "--dry-run" in result.stdout


@pytest.mark.parametrize("registry", ["", "train: null", "train: text", "train:\n  files: []", "train:\n  files: ''"])
def test_invalid_llm_dataset_registry(tmp_path, registry):
    domain = tmp_path / "uttt_llm"
    (domain / "launch").mkdir(parents=True)
    (tmp_path / "scripts").mkdir()
    shutil.copy2(ROOT / "scripts/launch_common.sh", tmp_path / "scripts/launch_common.sh")
    shutil.copy2(ROOT / "uttt_llm/launch/train.sh", domain / "launch/train.sh")
    (domain / "datasets.yaml").write_text(registry)
    configs = [str(ROOT / path) for path in LAUNCHERS[1][1]]
    result = launch(domain / "launch/train.sh", ["--dry-run", *configs], {"PYTHON": sys.executable})
    assert result.returncode != 0
    assert "non-empty train.files string" in result.stderr


def test_node_sync_failure_stops_launch_and_dry_run_does_not_execute(tmp_path):
    script, configs = LAUNCHERS[0]
    sync = tmp_path / "sync.py"
    sync.write_text("raise SystemExit(9)\n")
    environment = {"NPROC_PER_NODE": "4", "WORLD_SIZE": "8", "MASTER_ADDR": "localhost",
                   "MASTER_PORT": "29500", "NODE_RANK": "0", "NODE_SYNC": "sync.py", "PYTHON": sys.executable}
    absolute_configs = [str(ROOT / path) for path in configs]
    result = launch(script, absolute_configs, environment, cwd=tmp_path)
    assert result.returncode == 9
    result = launch(script, ["--dry-run", *absolute_configs], environment, cwd=tmp_path)
    assert result.returncode == 0, result.stderr
