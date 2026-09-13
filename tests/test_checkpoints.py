from pathlib import Path

import pytest
import torch

from uttt_nvs.train.checkpoint import (
    checkpoint_job, delete_previous_job, find_checkpoints, get_job_overview,
    resume_job, select_resume_path,
)


@pytest.fixture
def training_state():
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    scheduler.step()
    return model, optimizer, scheduler


def test_round_trip_and_numeric_steps(tmp_path, training_state):
    model, optimizer, scheduler = training_state
    expected = {name: value.clone() for name, value in model.state_dict().items()}
    checkpoint_job(tmp_path, model, optimizer, scheduler, 500, 125)
    with torch.no_grad():
        model.weight.zero_()
    result = resume_job(tmp_path, model, optimizer, scheduler)
    assert result[2:] == (500, 125)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, expected[name])
    assert len(list(tmp_path.iterdir())) == 1


def test_failed_save_leaves_previous_checkpoint_intact(tmp_path, training_state, monkeypatch):
    checkpoint_job(tmp_path, *training_state, 500, 125)
    path = Path(find_checkpoints(tmp_path)[0])
    original = path.read_bytes()

    def fail_save(checkpoint, handle):
        handle.write(b"partial checkpoint")
        raise OSError("disk full")

    monkeypatch.setattr(torch, "save", fail_save)
    with pytest.raises(OSError, match="disk full"):
        checkpoint_job(tmp_path, *training_state, 500, 125)
    assert path.read_bytes() == original
    assert len(list(tmp_path.iterdir())) == 1


def test_discovery_and_count_based_retention(tmp_path):
    for name in ("ckpt_1.pt", "ckpt_20.pt", "ckpt_300.pt", "ckpt_4000.pt", "ckpt_bad.pt", ".ckpt_500.tmp"):
        (tmp_path / name).touch()
    (tmp_path / "ckpt_30.pt").mkdir()
    assert [Path(path).name for path in find_checkpoints(tmp_path)] == [
        "ckpt_1.pt", "ckpt_20.pt", "ckpt_300.pt", "ckpt_4000.pt"]
    delete_previous_job(tmp_path, 300, save_last_n_ckpts=2)
    assert [Path(path).name for path in find_checkpoints(tmp_path)] == [
        "ckpt_20.pt", "ckpt_300.pt", "ckpt_4000.pt"]
    assert (tmp_path / "ckpt_bad.pt").exists()


@pytest.mark.parametrize("count", [0, -1, True, 1.5])
def test_invalid_retention(tmp_path, count):
    with pytest.raises(ValueError, match="positive integer"):
        delete_previous_job(tmp_path, 100, count)


def test_corrupt_newest_falls_back(tmp_path, training_state):
    checkpoint_job(tmp_path, *training_state, 100, 25)
    (tmp_path / "ckpt_200.pt").write_bytes(b"broken")
    assert resume_job(tmp_path, *training_state)[2:] == (100, 25)


def test_all_corrupt_raises(tmp_path, training_state):
    (tmp_path / "ckpt_200.pt").write_bytes(b"broken")
    with pytest.raises(RuntimeError, match="Failed to load any checkpoint"):
        resume_job(tmp_path, *training_state)


def test_missing_explicit_checkpoint_raises(tmp_path, training_state):
    with pytest.raises(FileNotFoundError):
        resume_job(tmp_path / "missing.pt", *training_state)
    with pytest.raises(FileNotFoundError, match="No ckpt_"):
        resume_job(tmp_path, *training_state)
    assert resume_job("", *training_state)[2:] == (0, 0)


def test_reset_keeps_weights_but_resets_counters(tmp_path, training_state):
    checkpoint_job(tmp_path, *training_state, 100, 25)
    expected = training_state[0].weight.detach().clone()
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    result = resume_job(tmp_path, model, optimizer, scheduler, reset_training_state=True)
    assert result[2:] == (0, 0)
    torch.testing.assert_close(model.weight, expected)
    assert optimizer.param_groups[0]["lr"] == 0.3
    assert not optimizer.state


def test_resume_source_selection(tmp_path):
    assert select_resume_path(tmp_path, "explicit.pt", "config.pt") == "explicit.pt"
    assert select_resume_path(tmp_path, "", "config.pt") == "config.pt"
    assert select_resume_path(tmp_path) == ""
    (tmp_path / "ckpt_0.pt").touch()
    assert select_resume_path(tmp_path, "explicit.pt", "config.pt") == tmp_path


def test_bounded_smoke_steps_are_not_epoch_rounded():
    overview = get_job_overview(4, 256, 32, 1, max_fwdbwd_passes=5,
                               round_max_fwdbwd_passes_to_epoch=False)
    assert overview.num_fwdbwd_passes == 5
    assert overview.num_param_updates == 5
