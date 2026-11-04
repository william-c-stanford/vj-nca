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
    spatial_frequencies: tuple[float, ...] = ()
    hidden_layers: int = 1
    plane_channels: int = 0
    plane_resolution: int = 128
    plane_time_resolution: int = 64
    direct_rgb: bool = False

    def __post_init__(self):
        self.perception_scales = tuple(self.perception_scales)
        self.spatial_frequencies = tuple(self.spatial_frequencies)
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
        if self.hidden_layers < 1 or any(f <= 0 for f in self.spatial_frequencies):
            raise ValueError("Use positive hidden_layers and spatial frequencies")
        if self.plane_channels < 0 or min(self.plane_resolution, self.plane_time_resolution) < 2:
            raise ValueError("Use nonnegative plane channels and plane sizes >=2")


class ConditionedNCA(nn.Module):
    """Shared local residual rule; loop identity and phase are read-only inputs.

    RGB lives in the first three state channels, in [-1, 1]. There is no living
    cell mask: a completely erased region must be able to grow back. Coarse
    perception increases the communication radius without adding a large backbone.
    Optional shared feature planes provide learned position/time priors; they
    remain read-only during rollouts and are not per-video state or target frames.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.num_loops, config.embedding_dim)
        nn.init.normal_(self.embedding.weight, std=0.3)
        n_inputs = config.channels * 4 + config.embedding_dim + 2 + 2 * config.phase_harmonics
        n_inputs += 4 * len(config.spatial_frequencies)
        n_inputs += 4 * config.plane_channels
        self.planes = nn.ParameterDict()
        if config.plane_channels:
            c, r, t = config.plane_channels, config.plane_resolution, config.plane_time_resolution
            for key, shape in {"xy": (1, c, r, r), "xt": (1, c, r, t), "yt": (1, c, r, t)}.items():
                self.planes[key] = nn.Parameter(1 + torch.randn(shape) * 0.1)
        self.update_in = nn.Conv2d(n_inputs, config.hidden, 1)
        self.update_hidden = nn.Sequential(
            *[
                layer
                for _ in range(config.hidden_layers - 1)
                for layer in (nn.Conv2d(config.hidden, config.hidden, 1), nn.ReLU())
            ]
        )
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

    def conditioning(
        self,
        ids: Tensor,
        phase: Tensor,
        height: int,
        width: int,
        origins: Tensor | None = None,
        canvas_size: int | None = None,
    ) -> Tensor:
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
        if origins is not None:
            if canvas_size is None:
                raise ValueError("Patch origins require the full canvas size")
            # Origins are (y,x) in full-resolution pixels; Fourier features never
            # restart at a patch boundary. Align pixel centers with full rollouts.
            ys = torch.arange(height, device=e.device, dtype=e.dtype) + 0.5
            xs = torch.arange(width, device=e.device, dtype=e.dtype) + 0.5
            y = (ys[None, :, None] + origins[:, 0, None, None]) / canvas_size * 2 - 1
            x = (xs[None, None, :] + origins[:, 1, None, None]) / canvas_size * 2 - 1
            xy = torch.stack([x.expand(-1, height, -1), y.expand(-1, -1, width)], dim=1)
        features = [cond, xy]
        for frequency in self.config.spatial_frequencies:
            angles = xy * (math.pi * frequency)
            features.extend([angles.sin(), angles.cos()])
        if self.config.plane_channels:
            spatial = self.sample_border(
                self.planes["xy"].expand(len(ids), -1, -1, -1),
                xy.permute(0, 2, 3, 1),
            )
            plane_phase = phase if self.config.use_phase else torch.zeros_like(phase)
            xt = self.sample_time_plane(self.planes["xt"], xy[:, 0], plane_phase)
            yt = self.sample_time_plane(self.planes["yt"], xy[:, 1], plane_phase)
            # Every loop uses these same learned planes. Only the global loop
            # embedding selects the clip; there are no per-clip feature atlases.
            features.extend([spatial, xt, yt, spatial * xt * yt])
        return torch.cat(features, dim=1)

    @staticmethod
    def sample_border(plane: Tensor, grid: Tensor) -> Tensor:
        """Border-equivalent bilinear sampling, supported by Metal without CPU fallback.

        Clamping to the first/last pixel centers makes zero padding irrelevant.
        PyTorch 2.8's MPS grid sampler does not implement border padding.
        """
        height, width = plane.shape[-2:]
        bounded = torch.stack(
            [
                grid[..., 0].clamp(-1 + 1 / width, 1 - 1 / width),
                grid[..., 1].clamp(-1 + 1 / height, 1 - 1 / height),
            ],
            dim=-1,
        )
        return F.grid_sample(plane, bounded, padding_mode="zeros", align_corners=False)

    @staticmethod
    def sample_time_plane(plane: Tensor, coordinate: Tensor, phase: Tensor) -> Tensor:
        """Bilinear space-time features with periodic time, including the seam."""
        count = plane.shape[-1]
        wrapped = torch.cat([plane[..., -1:], plane, plane[..., :1]], dim=-1)
        time = (((phase % 1) * count + 1.5) / (count + 2) * 2 - 1)[:, None, None].expand_as(
            coordinate
        )
        grid = torch.stack([time, coordinate], dim=-1)
        return ConditionedNCA.sample_border(wrapped.expand(len(phase), -1, -1, -1), grid)

    def forward(
        self,
        state: Tensor,
        ids: Tensor,
        phase: Tensor,
        mask: Tensor | None = None,
        origins: Tensor | None = None,
        canvas_size: int | None = None,
    ) -> Tensor:
        p = self.perceive(state)
        cond = self.conditioning(ids, phase, *state.shape[-2:], origins, canvas_size)
        features = F.relu(self.update_in(torch.cat([p, cond], dim=1)))
        delta = self.update_out(self.update_hidden(features))
        if mask is None:
            mask = (torch.rand_like(state[:, :1]) < self.config.fire_rate).to(state.dtype)
        updated = state + delta
        if self.config.direct_rgb:
            # Visible channels track the current phase directly. Hidden channels
            # retain the residual local dynamics, without integrating RGB bias
            # for hundreds of steps into a color drift.
            updated = torch.cat([delta[:, :3], updated[:, 3:]], dim=1)
        if self.config.hidden_norm == "unit_ball":
            b, _, h, w = updated.shape
            hidden = updated[:, 3:].reshape(b, self.config.norm_groups, -1, h, w)
            hidden = (hidden / hidden.norm(dim=2, keepdim=True).clamp_min(1)).flatten(1, 2)
            updated = torch.cat([updated[:, :3], hidden], dim=1)
        return state + mask * (updated - state)

    @staticmethod
    def rgb(state: Tensor) -> Tensor:
        return state[:, :3]
