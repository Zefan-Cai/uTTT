"""Distributed sampler that can resume part-way through an epoch."""

from torch.utils.data import DistributedSampler


class ResumableDistributedSampler(DistributedSampler):
    """``DistributedSampler`` whose next iteration can skip already-consumed samples.

    The epoch permutation is a pure function of ``seed`` and ``epoch``, so a run
    resumed at forward/backward pass ``s`` can rebuild epoch ``s // passes_per_epoch``
    and skip the first ``(s % passes_per_epoch) * batch_size_per_gpu`` samples of
    this rank's slice instead of replaying the epoch from its beginning. The skip
    applies to the next ``__iter__`` only.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._skip = 0

    def set_resume_offset(self, num_samples: int) -> None:
        if num_samples < 0:
            raise ValueError("num_samples must be non-negative")
        self._skip = int(num_samples)

    def __iter__(self):
        indices = list(super().__iter__())
        skip, self._skip = self._skip, 0
        if skip:
            if skip >= len(indices):
                raise ValueError(
                    f"Resume offset {skip} is not smaller than this rank's epoch length {len(indices)}"
                )
            indices = indices[skip:]
        return iter(indices)

    def __len__(self) -> int:
        return max(self.num_samples - self._skip, 0)
