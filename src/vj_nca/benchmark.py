"""Synchronized native-resolution NCA timings, with a separate RGB readback path."""

import platform
import time

import numpy as np
import torch

from .model import ConditionedNCA


@torch.inference_mode()
def benchmark_model(
    model: ConditionedNCA,
    size: int = 512,
    samples: int = 120,
    warmup: int = 32,
    steps_per_frame: int = 1,
    period_frames: int = 64,
) -> dict:
    if min(samples, steps_per_frame, period_frames) < 1 or warmup < 0:
        raise ValueError("Use positive sample/step/period counts and nonnegative warmup")
    device = next(model.parameters()).device

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "mps":
            torch.mps.synchronize()

    ids = torch.zeros(1, dtype=torch.long, device=device)
    phase = torch.zeros(1, device=device)
    mask = torch.ones(1, 1, size, size, device=device) if model.config.fire_rate == 1 else None
    period = period_frames * steps_per_frame
    result = {}
    for readback in (False, True):
        state, tick, durations = model.seed(1, size, device), 0, []
        for frame in range(warmup + samples):
            synchronize()
            start = time.perf_counter_ns()
            for _ in range(steps_per_frame):
                tick = (tick + 1) % period
                phase.fill_(tick / period)
                state = model(state, ids, phase, mask=mask)
            if readback:
                # Include RGB conversion, synchronization and actual host image access.
                pixels = (model.rgb(state)[0].permute(1, 2, 0).clamp(-1, 1) + 1) * 127.5
                pixels = pixels.round().to(torch.uint8).cpu().numpy()
                if pixels.shape != (size, size, 3):
                    raise RuntimeError("Unexpected native RGB shape")
            synchronize()
            if frame >= warmup:
                durations.append((time.perf_counter_ns() - start) / 1e6)
        result["rgb_readback" if readback else "gpu_resident"] = {
            "mean_ms_per_frame": float(np.mean(durations)),
            "p50_ms_per_frame": float(np.median(durations)),
            "p95_ms_per_frame": float(np.percentile(durations, 95)),
            "frames_per_second": 1000 / float(np.mean(durations)),
            "mean_ms_per_update": float(np.mean(durations)) / steps_per_frame,
        }
    return {
        "device": str(device),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "size": size,
        "parameters": sum(p.numel() for p in model.parameters()),
        "samples": samples,
        "warmup": warmup,
        "steps_per_frame": steps_per_frame,
        "period_frames": period_frames,
        "scope": "NCA update(s) and optional native uint8 RGB CPU readback; excludes display, "
        "audio input, encoding, application overhead and frame pacing. No CPU fallback enabled.",
        **result,
    }
