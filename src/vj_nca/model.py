import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    num_loops: int
    channels: int = 19
    hidden: int = 64
    embedding_dim: int = 12
    phase_harmonics: int = 3
    perception_scales: tuple[int, ...] = (1, 2)
    fire_rate: float = 0.5
    hidden_norm: str = "none"
    norm_groups: int = 4
    use_phase: bool = True

    def __post_init__(self):
        self.perception_scales = tuple(self.perception_scales)
        if self.channels < 4 or self.num_loops < 1 or self.hidden < 1:
            raise ValueError("Need at least one loop, four state channels, and a hidden unit")
        if not 0 < self.fire_rate <= 1:
            raise ValueError("fire_rate must be in (0, 1]")
        if self.hidden_norm not in {"none", "unit_ball"}:
            raise ValueError("hidden_norm must be none or unit_ball")
        if self.hidden_norm == "unit_ball" and (self.channels - 3) % self.norm_groups:
            raise ValueError("Hidden state channels must divide evenly into norm_groups")
        if not self.perception_scales or min(self.perception_scales) < 1:
            raise ValueError("perception_scales are positive downsampling factors")


class ConditionedNCA(nn.Module):
    """Shared local residual rule; loop identity and phase are read-only inputs.

    RGB lives in the first three state channels, in [-1, 1]. There is no living
    cell mask: a completely erased region must be able to grow back. Coarse
    perception increases the communication radius without adding a large backbone.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.num_loops, config.embedding_dim)
        nn.init.normal_(self.embedding.weight, std=0.3)
        n_inputs = config.channels * 4 + config.embedding_dim + 2 + 2 * config.phase_harmonics
        self.update_in = nn.Conv2d(n_inputs, config.hidden, 1)
        self.update_out = nn.Conv2d(config.hidden, config.channels, 1, bias=False)
        nn.init.zeros_(self.update_out.weight)
        sx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32) / 8
        lap = torch.tensor([[1, 2, 1], [2, -12, 2], [1, 2, 1]], dtype=torch.float32) / 16
        identity = torch.zeros(3, 3)
        identity[1, 1] = 1
        kernels = torch.stack([identity, sx, sx.T, lap])[:, None]
        self.register_buffer("kernels", kernels.repeat(config.channels, 1, 1, 1))

    def seed(self, batch: int, size: int, device: torch.device | str) -> Tensor:
        return torch.zeros(batch, self.config.channels, size, size, device=device)

    def perceive(self, state: Tensor) -> Tensor:
        h, w = state.shape[-2:]
        levels = []
        for scale in self.config.perception_scales:
            x = state if scale == 1 else F.avg_pool2d(state, scale)
            p = F.conv2d(
                F.pad(x, (1, 1, 1, 1), mode="replicate"), self.kernels, groups=self.config.channels
            )
            if scale != 1:
                p = F.interpolate(p, size=(h, w), mode="bilinear", align_corners=False)
            levels.append(p)
        return torch.stack(levels).mean(0)

    def conditioning(self, ids: Tensor, phase: Tensor, height: int, width: int) -> Tensor:
        e = self.embedding(ids)
        f = torch.arange(1, self.config.phase_harmonics + 1, device=e.device)
        angle = (phase[:, None] % 1) * f * (2 * math.pi)
        time = torch.cat([angle.sin(), angle.cos()], dim=1)
        if not self.config.use_phase:
            time = time * 0
        cond = torch.cat([e, time], dim=1)[:, :, None, None].expand(-1, -1, height, width)
        ys = (torch.arange(height, device=e.device, dtype=e.dtype) + 0.5) / height * 2 - 1
        xs = (torch.arange(width, device=e.device, dtype=e.dtype) + 0.5) / width * 2 - 1
        y, x = torch.meshgrid(ys, xs, indexing="ij")
        xy = torch.stack([x, y])[None].expand(len(ids), -1, -1, -1)
        return torch.cat([cond, xy], dim=1)

    def forward(
        self, state: Tensor, ids: Tensor, phase: Tensor, mask: Tensor | None = None
    ) -> Tensor:
        p = self.perceive(state)
        cond = self.conditioning(ids, phase, *state.shape[-2:])
        delta = self.update_out(F.relu(self.update_in(torch.cat([p, cond], dim=1))))
        if mask is None:
            mask = (torch.rand_like(state[:, :1]) < self.config.fire_rate).to(state.dtype)
        updated = state + delta
        if self.config.hidden_norm == "unit_ball":
            b, _, h, w = updated.shape
            hidden = updated[:, 3:].reshape(b, self.config.norm_groups, -1, h, w)
            hidden = (hidden / hidden.norm(dim=2, keepdim=True).clamp_min(1)).flatten(1, 2)
            updated = torch.cat([updated[:, :3], hidden], dim=1)
        return state + mask * (updated - state)

    @staticmethod
    def rgb(state: Tensor) -> Tensor:
        return state[:, :3]
