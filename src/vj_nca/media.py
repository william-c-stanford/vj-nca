import subprocess
from itertools import pairwise
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor


def save_video(path: Path, video: Tensor, fps: float, zoom: int = 1):
    """Export [time, RGB, height, width] in [-1, 1], clipping only for display."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = ((video.detach().cpu().clamp(-1, 1) + 1) * 127.5).round().byte()
    arr = arr.permute(0, 2, 3, 1).numpy()
    if zoom > 1:
        arr = arr.repeat(zoom, axis=1).repeat(zoom, axis=2)
    if path.suffix == ".gif":
        images = [Image.fromarray(a) for a in arr]
        # GIF has 10 ms ticks; distribute rounding rather than speeding up every frame.
        boundaries = [round(index * 100 / fps) * 10 for index in range(len(images) + 1)]
        durations = [end - start for start, end in pairwise(boundaries)]
        images[0].save(
            path,
            save_all=True,
            append_images=images[1:],
            loop=0,
            duration=durations,
            optimize=False,
        )
    else:
        h, w = arr.shape[1:3]
        cmd = [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{w}x{h}",
            "-r",
            str(fps),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(path),
        ]
        subprocess.run(cmd, input=arr.tobytes(), capture_output=True, check=True, timeout=120)


def comparison_grid(target: Tensor, prediction: Tensor, damaged: Tensor) -> Tensor:
    """[loops,time,RGB,H,W] -> a video with target/clean/perturbed columns."""
    rows = [
        torch.cat([t, p, d], dim=-1) for t, p, d in zip(target, prediction, damaged, strict=True)
    ]
    return torch.cat(rows, dim=-2)


def save_contact_sheet(path: Path, video: Tensor, columns: int = 8):
    indices = torch.linspace(0, len(video) - 1, columns).long()
    arr = ((video[indices].cpu().clamp(-1, 1) + 1) * 127.5).round().byte()
    arr = arr.permute(0, 2, 3, 1).numpy()
    Image.fromarray(np.concatenate(list(arr), axis=1)).save(path)
