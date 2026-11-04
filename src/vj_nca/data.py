import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import torch
from torch import Tensor


class LoopDataset:
    """One normalized cycle per clip. Frames do not duplicate the endpoint."""

    def __init__(
        self, frames: Tensor, names: list[str], fps: float, sources: list[dict] | None = None
    ):
        if frames.ndim != 5 or frames.shape[2] != 3 or len(names) != len(frames):
            raise ValueError("Expected [loops, time, RGB, height, width] and one name per loop")
        if frames.shape[1] < 2 or frames.shape[-2] != frames.shape[-1] or fps <= 0:
            raise ValueError("Need at least two frames, square images and positive fps")
        if not torch.isfinite(frames).all():
            raise ValueError("Non-finite video data")
        if len(set(names)) != len(names):
            raise ValueError("Loop names must be unique")
        self.frames = frames.contiguous().float()
        self.names = list(names)
        self.fps = float(fps)
        self.sources = sources or [{} for _ in names]

    @property
    def period(self) -> int:
        return self.frames.shape[1]

    @property
    def size(self) -> int:
        return self.frames.shape[-1]

    def target(self, ids: Tensor, ticks: Tensor, steps_per_frame: int) -> Tensor:
        time = ticks.float() / steps_per_frame
        index = time.floor().long()
        alpha = (time - time.floor())[:, None, None, None]
        a = self.frames[ids, index % self.period]
        b = self.frames[ids, (index + 1) % self.period]
        return torch.lerp(a, b, alpha)

    def to(self, device: torch.device | str) -> "LoopDataset":
        return LoopDataset(self.frames.to(device), self.names, self.fps, self.sources)

    def fingerprint(self) -> str:
        h = hashlib.sha256(self.frames.detach().cpu().numpy().tobytes())
        h.update(json.dumps([self.names, self.fps, self.sources], sort_keys=True).encode())
        return h.hexdigest()

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        # Store float32, so save/load and checkpoint resume are lossless.
        np.savez_compressed(
            path,
            frames=self.frames.cpu().numpy(),
            names=np.array(self.names),
            fps=np.array(self.fps),
            sources=np.array(json.dumps(self.sources)),
        )

    @classmethod
    def load(cls, path: Path) -> "LoopDataset":
        with np.load(path, allow_pickle=False) as d:
            return cls(
                torch.from_numpy(d["frames"].copy()),
                d["names"].tolist(),
                float(d["fps"]),
                json.loads(str(d["sources"])),
            )

    def diagnostics(self) -> dict:
        seam = (self.frames[:, 0] - self.frames[:, -1]).square().mean((1, 2, 3))
        motion = self.frames.diff(dim=1).square().mean((1, 2, 3, 4))
        return {
            name: {
                "seam_mse": s.item(),
                "adjacent_mse": m.item(),
                "seam_ratio": (s / (m + 1e-8)).item(),
            }
            for name, s, m in zip(self.names, seam, motion, strict=True)
        }


def make_procedural(
    size: int = 32, frames: int = 32, count: int = 6, fps: float = 16
) -> LoopDataset:
    """Original, analytic VJ-style loops: free of third-party media dependencies."""
    if not 1 <= count <= 6 or size < 4 or frames < 4:
        raise ValueError("Choose 1..6 loops, size >= 4 and frames >= 4")
    y, x = torch.meshgrid(torch.linspace(-1, 1, size), torch.linspace(-1, 1, size), indexing="ij")
    x, y = x[None], y[None]
    t = torch.arange(frames)[:, None, None] / frames * 2 * math.pi
    r = (x.square() + y.square() + 1e-8).sqrt()
    angle = torch.atan2(y, x)
    fields = [
        torch.sin(2 * math.pi * x - t).expand(-1, size, -1),
        torch.cos(3 * math.pi * r - t),
        2 * torch.exp(-6 * ((x - 0.55 * t.cos()).square() + (y - 0.55 * t.sin()).square())) - 1,
        torch.cos(3 * angle - t) * torch.exp(-0.9 * r.square()),
        (torch.sin(3 * x + t) + torch.cos(3 * y - t) + torch.sin(3 * (x + y) + t)) / 3,
        torch.sin(2 * math.pi * (x * t.cos() + y * t.sin()))
        * torch.cos(math.pi * (y * t.cos() - x * t.sin())),
    ]
    names = ["drifting-bars", "expanding-rings", "orbit", "rotating-spokes", "plasma", "lattice"]
    colors = [
        ([0.1, 0.9, 0.9], [-0.8, -0.6, 0.4]),
        ([0.9, 0.1, 0.7], [-0.9, -0.8, -0.4]),
        ([0.8, 0.7, 0.1], [-0.9, -0.9, -0.7]),
        ([0.8, 0.1, 0.5], [-0.9, -0.7, -0.9]),
        ([0.4, 0.9, 0.2], [-0.6, -0.7, -0.9]),
        ([0.5, 0.3, 0.9], [-0.9, -0.8, -0.3]),
    ]
    videos = []
    for field, (bright, dark) in zip(fields[:count], colors[:count], strict=True):
        a = (field[:, None] + 1) / 2
        hi = torch.tensor(bright)[None, :, None, None]
        lo = torch.tensor(dark)[None, :, None, None]
        videos.append(lo + a * (hi - lo))
    sources = [
        {
            "kind": "procedural",
            "license": "CC0-1.0",
            "generator": "vj_nca.data",
            "description": name,
        }
        for name in names[:count]
    ]
    return LoopDataset(torch.stack(videos), names[:count], fps, sources)


def import_video(
    path: Path,
    name: str,
    size: int,
    frames: int,
    fps: float,
    start: float,
    duration: float,
    source_url: str,
    license_name: str,
    normalize_timestamps: bool = False,
) -> LoopDataset:
    """Decode one explicitly selected complete cycle and retime to a common period.

    Audio is never decoded. The caller chooses the actual cycle boundaries; this
    function does not invent a seam, ping-pong a clip, or duplicate its last frame.
    """
    if duration <= 0 or start < 0 or frames < 2:
        raise ValueError("Use nonnegative start, positive duration and at least two frames")
    vf = (
        f"scale={size}:{size}:force_original_aspect_ratio=increase,"
        f"crop={size}:{size},fps={frames / duration}"
    )
    if normalize_timestamps:
        vf = "setpts=PTS-STARTPTS," + vf + ":round=up:start_time=0"
    command = [
        "ffmpeg",
        "-v",
        "error",
        *(["-ss", str(start)] if start or not normalize_timestamps else []),
        "-i",
        str(path),
        "-t",
        str(duration),
        "-an",
        "-vf",
        vf,
        "-frames:v",
        str(frames),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    raw = subprocess.run(command, check=True, capture_output=True, timeout=120).stdout
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(-1, size, size, 3).copy()
    if len(arr) != frames:
        raise ValueError(f"Decoded {len(arr)} of {frames} frames; check the cycle boundaries")
    source = {
        "kind": "video",
        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "url": source_url,
        "license": license_name,
        "start_seconds": start,
        "duration_seconds": duration,
        "retimed_duration_seconds": frames / fps,
        "crop": "center-square",
        "audio": "discarded",
    }
    if normalize_timestamps:
        source["timestamp_normalization"] = "zero-origin, fps round-up; no zero-time AVI seek"
    video = torch.from_numpy(arr).permute(0, 3, 1, 2).float() / 127.5 - 1
    return LoopDataset(video[None], [name], fps, [source])


def merge_datasets(datasets: list[LoopDataset]) -> LoopDataset:
    if not datasets:
        raise ValueError("No datasets to merge")
    first = datasets[0]
    if any(
        d.frames.shape[1:] != first.frames.shape[1:] or d.fps != first.fps for d in datasets[1:]
    ):
        raise ValueError("Merge requires equal resolution, cycle frame count and playback fps")
    return LoopDataset(
        torch.cat([d.frames for d in datasets]),
        sum((d.names for d in datasets), []),
        first.fps,
        sum((d.sources for d in datasets), []),
    )
