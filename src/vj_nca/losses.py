import torch
from torch import Tensor
from torch.nn import functional as F


def temporal_loss(prediction: Tensor, target: Tensor) -> Tensor:
    """Signed finite differences penalize frozen and reversed videos."""
    return F.mse_loss(prediction.diff(dim=1), target.diff(dim=1))


def video_loss(
    prediction: Tensor, target: Tensor, state: Tensor, spectral_weight: float = 0.02
) -> tuple[Tensor, dict[str, Tensor]]:
    pred, truth = prediction.flatten(0, 1), target.flatten(0, 1)
    pixel = F.mse_loss(pred, truth)
    coarse = F.mse_loss(F.avg_pool2d(pred, 2), F.avg_pool2d(truth, 2))
    spatial = F.mse_loss(pred.diff(dim=-1), truth.diff(dim=-1)) + F.mse_loss(
        pred.diff(dim=-2), truth.diff(dim=-2)
    )
    temporal = temporal_loss(prediction, target)
    # A frequency-domain objective, NOT FourierDiff-NCA or a diffusion sampler.
    spectral = pred.new_zeros(())
    if spectral_weight:
        a = torch.fft.rfft2(pred, norm="ortho").abs()
        b = torch.fft.rfft2(truth, norm="ortho").abs()
        spectral = F.l1_loss(torch.log1p(a), torch.log1p(b))
    overflow = (state.abs() - 2).relu().square().mean()
    total = pixel + 0.5 * coarse + 0.1 * spatial + 0.5 * temporal
    total = total + spectral_weight * spectral + 1e-3 * overflow
    return total, {"pixel": pixel, "temporal": temporal, "spectral": spectral, "overflow": overflow}
