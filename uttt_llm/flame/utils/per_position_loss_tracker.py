import torch
import torch.distributed as dist
import pathlib
import json
import math
from typing import Optional


class AverageTracker:
    """
    Tracks the (weighted) running mean of a scalar or tensor.
        • shape‑agnostic: [seq_len], [], images, …
        • detaches values so the tracker never clutters autograd.
        • optional DDP sync so every rank has identical statistics.
    """
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device
        self.reset()

    # ------------------------------------------------------------
    def reset(self):
        self._sum   = None       # running sum (tensor)
        self._count = 0.0        # running weight (float)

    # ------------------------------------------------------------
    @torch.no_grad()
    def update(self, value: torch.Tensor | float, n: int | float = 1):
        """
        Args:
            value : tensor or float  — **already on the right device**.
            n     : weight of this update (e.g. batch‑size).
        """
        if not torch.is_tensor(value):
            value = torch.as_tensor(value, dtype=torch.float32,
                                    device=self.device)

        value = value.detach()                     # keep graph out
        if self._sum is None:
            self._sum = value.clone()
        else:
            self._sum = self._sum + value

        self._count += float(n)

    # ------------------------------------------------------------
    @torch.no_grad()
    def average(self) -> torch.Tensor:
        if self._count == 0:
            raise RuntimeError("AverageTracker: no samples seen yet.")
        return self._sum / self._count

    # ------------------------------------------------------------
    @torch.no_grad()
    def sync_ddp(self, group: Optional[dist.ProcessGroup] = None):
        """
        All‑reduce both sum and count across `group` (default WORLD)
        so that the running statistics become global averages.
        Call once **after** the last .update() you care about.
        """
        if not (dist.is_available() and dist.is_initialized()):
            return

        # 1) sum
        dist.all_reduce(self._sum, op=dist.ReduceOp.SUM, group=group)

        # 2) count  (scalar on all ranks)
        count_tensor = torch.tensor(self._count,
                                    device=self._sum.device,
                                    dtype=self._sum.dtype)
        dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM, group=group)
        self._count = count_tensor.item()

    @torch.no_grad()
    def per_chunk_average(self, num_chunks: int = 16) -> torch.Tensor:
        """
        Returns the mean of the tracked 1‑D tensor in `num_chunks` contiguous
        segments.  Useful for “block loss” plots on long sequences.

        Preconditions
        -------------
        • The tracked value must be 1‑D (shape [seq_len]).
        • Call after at least one .update() and, if needed, .sync_ddp().
        """
        mean = self.average()                  # shape [seq_len] or scalar
        if mean.ndim != 1:
            raise ValueError("per_chunk_average requires a 1‑D tracked tensor.")

        seq_len = mean.numel()
        if num_chunks > seq_len:
            raise ValueError(f"num_chunks ({num_chunks}) > seq_len ({seq_len}).")

        # Evenly split; last chunk may be shorter if seq_len % num_chunks ≠ 0
        chunk_size = math.ceil(seq_len / num_chunks)
        chunks = mean.split(chunk_size)
        chunk_means = torch.stack([c.mean() for c in chunks[:num_chunks]])
        return chunk_means     # shape [num_chunks]

    # --------------------------------------------------------------------- #
    @torch.no_grad()
    def save_to_json(self, save_path: str | pathlib.Path):
        """
        Write the current average() (tensor of any shape) to `save_path`
        as a flat list[float] using single‑precision numbers.
        """
        save_path = pathlib.Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        data = self.average().cpu().float().reshape(-1).tolist()
        with open(save_path, "w") as f:
            json.dump(data, f)
        # Optionally return the path for logging
        return save_path.as_posix()
    
    @torch.no_grad()
    @torch.no_grad()
    def plot_curve(
        self,
        save_path: str | pathlib.Path | None = None,
        show: bool = False,
        dpi: int = 120,
        drop_first_and_last: bool = True,
    ):
        """
        Draw a FiveThirtyEight‑style line chart of the per‑token loss.

        Parameters
        ----------
        save_path : str | Path | None
            If given, save the PNG to this location.
        show : bool
            Call `plt.show()` for interactive environments (e.g. notebooks).
        dpi : int
            Resolution of the saved figure.
        drop_first_and_last : bool
            If True, omit the first and last token positions from the plot.
        """
        mean = self.average()
        if mean.ndim != 1:
            raise ValueError("plot_curve expects a 1‑D tracked tensor.")

        if drop_first_and_last:
            if mean.numel() < 3:
                raise ValueError("Need at least 3 positions to drop first/last.")
            mean = mean[1:-1]
            offset = 1  # x‑axis starts at token position 1
        else:
            offset = 0

        import matplotlib.pyplot as plt

        plt.style.use("fivethirtyeight")
        fig, ax = plt.subplots(figsize=(8, 4), dpi=dpi)

        x = torch.arange(offset, offset + mean.numel()).cpu()
        ax.plot(x, mean.cpu())

        ax.set_xlabel("Token position")
        ax.set_ylabel("Average loss")

        original_seq_len = mean.numel() + (2 if drop_first_and_last else 0)
        total_tokens = int(original_seq_len * self._count)
        ax.set_title(f"Per‑position loss  •  N = {int(total_tokens/1e6)}M tokens")

        fig.tight_layout()

        if save_path is not None:
            save_path = pathlib.Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=dpi)

        if show:
            plt.show()

        plt.close(fig)
        return fig
    
    def num_of_tokens_tracked(self) -> int:
        return self.average().numel() * self._count