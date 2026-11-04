"""Render the learned Aperture cycle at a chosen tempo, optionally with state kicks."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from vj_nca.evaluate import perturb
from vj_nca.media import save_video
from vj_nca.model import ConditionedNCA, ModelConfig


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--bpm", type=float, default=120)
    parser.add_argument("--cycles", type=int, default=4)
    parser.add_argument("--beat", action="store_true")
    args = parser.parse_args()
    if not 30 <= args.bpm <= 300 or not 1 <= args.cycles <= 16:
        raise ValueError("Use 30..300BPM and 1..16 cycles")
    torch.set_num_threads(2)
    torch.manual_seed(6061)
    saved = torch.load(args.weights, map_location="cpu", weights_only=True)
    if saved["model_config"].get("repair_channel"):
        raise ValueError("Use evaluate_reactive.py for the faster teacher's native cadence")
    if saved["loop_names"] != ["aperture"]:
        raise ValueError("This preview recipe requires the single-preset Aperture model")
    model = ConditionedNCA(ModelConfig(**saved["model_config"])).to(args.device).eval()
    model.load_state_dict(saved["model"])
    spf = saved["steps_per_frame"]
    period = 64 * spf
    ids = torch.zeros(1, dtype=torch.long, device=args.device)
    state = model.seed(1, 512, args.device)
    mask = torch.ones_like(state[:, :1]) if model.config.fire_rate == 1 else None
    for tick in range(1, 2 * period + 1):
        phase = torch.full((1,), tick % period / period, device=args.device)
        state = model(state, ids, phase, mask=mask)
    frames = []
    for tick in range(args.cycles * period):
        if args.beat and tick % (period // 4) == 0:
            state = perturb(state, "beat")
        rgb = model.rgb(state)[0].cpu().clamp(-1, 1)
        frames.append(((rgb + 1) * 127.5).round().byte())
        phase = torch.full((1,), (tick + 1) % period / period, device=args.device)
        state = model(state, ids, phase, mask=mask)
    fps = period * args.bpm / 240
    save_video(args.out, torch.stack(frames).float() / 127.5 - 1, fps=fps)
    args.out.with_suffix(".json").write_text(
        json.dumps(
            {
                "weights_sha256": hashlib.sha256(args.weights.read_bytes()).hexdigest(),
                "resolution": 512,
                "bpm": args.bpm,
                "beats_per_cycle": 4,
                "cycle_seconds": 240 / args.bpm,
                "video_fps": fps,
                "cycles": args.cycles,
                "synthetic_state_kicks": args.beat,
                "seed": 6061,
                "note": "Offline export at a chosen playback tempo; inference timing is measured "
                "separately. Synthetic brightness/latent kicks, no audio analysis.",
            },
            indent=2,
        )
        + "\n"
    )
    print(args.out)


if __name__ == "__main__":
    main()
