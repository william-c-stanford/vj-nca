"""State damage used during training and evaluation; RGB black is -1, not zero."""

import torch
from torch import Tensor
from torch.nn import functional as F

REPAIR_KINDS = ("blackout", "glitch", "rgb-split", "tear")


def coordinates(state: Tensor, origins: Tensor | None, canvas_size: int) -> tuple[Tensor, Tensor]:
    b, _, h, w = state.shape
    if origins is None:
        origins = torch.zeros(b, 2, device=state.device)
    y = (
        torch.arange(h, device=state.device)[None, :, None] + origins[:, 0, None, None] + 0.5
    ) / canvas_size * 2 - 1
    x = (
        torch.arange(w, device=state.device)[None, None, :] + origins[:, 1, None, None] + 0.5
    ) / canvas_size * 2 - 1
    return y.expand(b, h, w), x.expand(b, h, w)


def perturb_repair(
    state: Tensor,
    kind: str,
    origins: Tensor | None = None,
    canvas_size: int | None = None,
    circle: tuple[float, float, float] | None = None,
) -> tuple[Tensor, Tensor]:
    """Return actual damaged cell state and the affected mask, without compositing video."""
    out = state.clone()
    b, _, h, w = state.shape
    size = canvas_size or h
    y, x = coordinates(state, origins, size)
    if kind == "blackout":
        if circle is None:
            centers = torch.rand(b, 2, 1, 1, device=state.device) * 0.8 - 0.4
            radius = 0.5 + torch.rand(b, 1, 1, device=state.device) * 0.3
            mask = (x - centers[:, 0]).square() + (y - centers[:, 1]).square() < radius.square()
        else:
            cx, cy, radius = circle
            mask = (x - cx).square() + (y - cy).square() < radius**2
        out[:, :3].masked_fill_(mask[:, None], -1)
    elif kind in {"glitch", "rgb-split", "tear"}:
        mask = torch.zeros(b, h, w, dtype=torch.bool, device=state.device)
        palette = state.new_tensor([[1.0, -0.55, 0.45], [-1.0, 1.0, 1.0], [1.0, -1.0, 1.0]])
        for band in range(3):
            center = torch.rand(b, 2, 1, 1, device=state.device) * 1.6 - 0.8
            half_width = 0.25 + torch.rand(b, 1, 1, device=state.device) * 0.65
            half_height = 0.04 + torch.rand(b, 1, 1, device=state.device) * 0.16
            region = (x - center[:, 0]).abs() < half_width
            region &= (y - center[:, 1]).abs() < half_height
            shift = int(torch.randint(8, 33, ()).item()) * (-1 if band % 2 else 1)
            if kind == "rgb-split":
                rgb = torch.cat(
                    [state[:, c : c + 1].roll(shift * (c - 1), -1) for c in range(3)], 1
                )
            else:
                rgb = state[:, :3].roll(shift, -1)
            tint = palette[band][None, :, None, None]
            rgb = (rgb * 0.45 + tint * 0.55) if kind != "tear" else (rgb * 0.7 + tint * 0.3)
            out[:, :3] = torch.where(region[:, None], rgb, out[:, :3])
            mask |= region
    elif kind in {"beat", "noise"}:
        if kind == "beat":
            out[:, :3] += 0.7
            out[:, 3:-1] += torch.randn_like(out[:, 3:-1]) * 0.15
        else:
            out[:, :-1] += torch.randn_like(out[:, :-1]) * 0.3
        return out, torch.ones_like(state[:, :1], dtype=torch.bool)
    elif kind == "full-reset":
        out.zero_()
        out[:, :3] = -1
        return out, torch.ones_like(state[:, :1], dtype=torch.bool)
    else:
        raise ValueError(f"Unknown repair perturbation {kind}")
    out[:, 3:].masked_fill_(mask[:, None], 0)
    return out, mask[:, None]


def repair_teacher_step(
    support: Tensor, rgb: Tensor, target: Tensor, model_config
) -> tuple[Tensor, Tensor]:
    """Training-only local growth curriculum. The network predicts this state at inference."""
    radius = model_config.repair_radius
    nearby = F.max_pool2d(F.pad(support, (radius,) * 4, mode="replicate"), 2 * radius + 1, 1)
    increment = ((nearby - support) * 0.35 + 0.002).clamp(0, model_config.repair_rate)
    support = (support + increment).clamp(0, 1)
    rate = ((support - model_config.repair_threshold) / (1 - model_config.repair_threshold)).clamp(
        0, 1
    )
    return support, rgb + rate * (target - rgb)
