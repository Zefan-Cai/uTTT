"""Per-image PSNR, SSIM, and LPIPS-VGG for NVS evaluation."""

from __future__ import annotations

from typing import Any


def compute_psnr(ground_truth: Any, prediction: Any) -> Any:
    """Return one PSNR value per NCHW image, using a [0, 1] data range."""

    import torch

    ground_truth = ground_truth.float().clamp(0.0, 1.0)
    prediction = prediction.float().clamp(0.0, 1.0)
    mse = (ground_truth - prediction).square().mean(dim=(1, 2, 3))
    return -10.0 * torch.log10(mse.clamp_min(torch.finfo(mse.dtype).tiny))


def compute_ssim(ground_truth: Any, prediction: Any) -> Any:
    """Return Gaussian-window SSIM for each NCHW image."""

    from pytorch_msssim import ssim

    ground_truth = ground_truth.float().clamp(0.0, 1.0)
    prediction = prediction.float().clamp(0.0, 1.0)
    return ssim(
        ground_truth,
        prediction,
        data_range=1.0,
        size_average=False,
        win_size=11,
        win_sigma=1.5,
    )


class ImageMetricComputer:
    """Reusable GPU metric modules for a sequence of model batches."""

    def __init__(
        self,
        *,
        device: Any,
        lpips_batch_size: int = 32,
    ) -> None:
        if lpips_batch_size <= 0:
            raise ValueError("lpips_batch_size must be positive")
        from lpips import LPIPS

        self.device = device
        self.lpips_batch_size = lpips_batch_size
        self.lpips_model = LPIPS(net="vgg").to(device).eval()
        for parameter in self.lpips_model.parameters():
            parameter.requires_grad_(False)

    def compute_lpips(self, ground_truth: Any, prediction: Any) -> Any:
        """Return LPIPS-VGG per image with normalize=True for [0, 1] input."""

        import torch

        ground_truth = ground_truth.float().clamp(0.0, 1.0)
        prediction = prediction.float().clamp(0.0, 1.0)
        values = []
        with torch.no_grad():
            for start in range(0, ground_truth.shape[0], self.lpips_batch_size):
                stop = start + self.lpips_batch_size
                value = self.lpips_model(
                    ground_truth[start:stop],
                    prediction[start:stop],
                    normalize=True,
                )
                values.append(value.reshape(-1))
        return torch.cat(values)

    def __call__(self, ground_truth: Any, prediction: Any) -> dict[str, Any]:
        if ground_truth.ndim != 4 or prediction.ndim != 4:
            raise ValueError("metric inputs must both be NCHW tensors")
        if ground_truth.shape != prediction.shape:
            raise ValueError(
                f"metric input shapes differ: {tuple(ground_truth.shape)} vs "
                f"{tuple(prediction.shape)}"
            )
        return {
            "psnr": compute_psnr(ground_truth, prediction),
            "ssim": compute_ssim(ground_truth, prediction),
            "lpips": self.compute_lpips(ground_truth, prediction),
        }
