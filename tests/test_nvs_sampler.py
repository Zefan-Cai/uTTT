"""Resume semantics of the NVS training sampler (CPU only)."""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from uttt_nvs.train.sampler import ResumableDistributedSampler


def _dataset(n=40):
    return TensorDataset(torch.arange(n))


def _sampler(**kwargs):
    return ResumableDistributedSampler(_dataset(), num_replicas=2, rank=1, shuffle=True,
                                       drop_last=True, **kwargs)


def test_resume_offset_continues_the_same_permutation():
    full = _sampler()
    full.set_epoch(3)
    reference = list(full)

    resumed = _sampler()
    resumed.set_epoch(3)
    resumed.set_resume_offset(7)
    assert list(resumed) == reference[7:]
    # The skip applies once; the next epoch iteration is complete again.
    assert list(resumed) == reference


def test_resume_offset_matches_batches_consumed():
    batch_size = 4
    full = _sampler()
    full.set_epoch(0)
    loader = DataLoader(_dataset(), batch_size=batch_size, sampler=full, drop_last=True)
    batches = [b[0].tolist() for b in loader]

    consumed = 3
    resumed = _sampler()
    resumed.set_epoch(0)
    resumed.set_resume_offset(consumed * batch_size)
    loader = DataLoader(_dataset(), batch_size=batch_size, sampler=resumed, drop_last=True)
    assert [b[0].tolist() for b in loader] == batches[consumed:]


def test_offset_beyond_epoch_is_rejected():
    sampler = _sampler()
    sampler.set_resume_offset(len(sampler))
    with pytest.raises(ValueError):
        list(sampler)
    with pytest.raises(ValueError):
        sampler.set_resume_offset(-1)
