"""Compare native shader motion bandwidth; these proxies do not prove NCA learning."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from vj_nca.shader_data import ShaderRenderer

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "datasets/shader-preview-v0.json")
    parser.add_argument(
        "--out", type=Path, default=ROOT / "artifacts/shader-preview-v0/learnability-probes.json"
    )
    args = parser.parse_args()
    torch.set_num_threads(2)
    manifest = json.loads(args.manifest.read_text())
    results = {
        "note": "Native float GLSL renders at 64 uniform phases. Learning-difficulty proxies, "
        "not trained NCA results. Spatial retention uses linear area-downsample/bilinear-"
        "upsample signed velocities; it is not a nonlinear expressivity bound.",
        "samples": 64,
        "resolution": 512,
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "probe_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "per_loop": {},
    }
    with ShaderRenderer(ROOT / "shaders", 512) as renderer:
        for clip in manifest["clips"]:
            x = (
                torch.from_numpy(np.stack([renderer.render(clip, i / 64) for i in range(64)]))
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            velocity = x - x.roll(1, 0)
            power = velocity.square().mean().item()
            if power < 1e-8:
                raise ValueError("A frozen target is not a valid motion study")
            energy = torch.fft.rfft(x, dim=0).abs().square().sum((1, 2, 3))
            velocity_energy = torch.fft.rfft(velocity, dim=0).abs().square().sum((1, 2, 3))
            energy[0] = velocity_energy[0] = 0
            item = {
                "phase_harmonic_energy": {
                    str(k): energy[1 : k + 1].sum().item() / energy.sum().item() for k in (4, 8, 16)
                },
                "velocity_harmonic_energy": {
                    str(k): velocity_energy[1 : k + 1].sum().item() / velocity_energy.sum().item()
                    for k in (4, 8, 16)
                },
                "spatial_motion_retention": {},
            }
            for size in (64, 128, 256):
                low = F.interpolate(
                    F.interpolate(velocity, size=(size, size), mode="area"),
                    size=(512, 512),
                    mode="bilinear",
                    align_corners=False,
                )
                dot, prediction_power = (low * velocity).mean().item(), low.square().mean().item()
                item["spatial_motion_retention"][str(size)] = {
                    "motion_gain": dot / power,
                    "motion_cosine": dot / (prediction_power * power) ** 0.5,
                }
            samples = F.interpolate(x, size=(32, 32), mode="area").flatten(1).double()
            samples -= samples.mean(0)
            eigenvalues = torch.linalg.eigvalsh(samples @ samples.T).clamp_min(0).flip(0)
            item["temporal_rank_90pct_at_32px"] = int(
                torch.searchsorted(eigenvalues.cumsum(0) / eigenvalues.sum(), 0.9).item() + 1
            )
            results["per_loop"][clip["name"]] = item
            print(clip["name"], json.dumps(item), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
