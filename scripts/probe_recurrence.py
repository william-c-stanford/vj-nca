"""Measure whether a trained loop needs carried state rather than only coordinates/phase."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from vj_nca.data import LoopDataset
from vj_nca.model import ConditionedNCA, ModelConfig


@torch.inference_mode()
def probe(weights: Path, data: Path, device: str) -> dict:
    saved = torch.load(weights, map_location="cpu", weights_only=True)
    dataset = LoopDataset.load(data)
    if dataset.fingerprint() != saved["dataset_fingerprint"]:
        raise ValueError("Probe dataset differs from training")
    dataset = dataset.to(device)
    model = ConditionedNCA(ModelConfig(**saved["model_config"])).to(device).eval()
    model.load_state_dict(saved["model"])
    if model.config.fire_rate != 1:
        raise ValueError("This paired probe currently requires synchronous updates")
    spf = saved["steps_per_frame"]
    period = dataset.period * spf
    ids = torch.arange(len(dataset.names), device=device)
    state = model.seed(len(ids), dataset.size, device)
    empty = torch.zeros_like(state)
    mask = torch.ones_like(state[:, :1])
    sums = torch.zeros(6, device=device)
    previous = None
    count = 0
    for tick in range(1, 4 * period + 1):
        phase = torch.full((len(ids),), (tick % period) / period, device=device)
        state = model(state, ids, phase, mask=mask)
        if tick <= 2 * period or tick % spf:
            continue
        reset = model(empty, ids, phase, mask=mask)
        carried, erased = model.rgb(state), model.rgb(reset)
        target = dataset.target(ids, torch.full_like(ids, tick % period), spf)
        sums[:3] += torch.stack(
            [
                (carried - target).square().mean() / 4,
                (erased - target).square().mean() / 4,
                (carried - erased).square().mean() / 4,
            ]
        )
        if previous is not None:
            dv = erased - previous[0]
            dt = target - previous[1]
            sums[3:6] += torch.stack([dv.square().mean(), dt.square().mean(), (dv * dt).mean()])
        previous = erased.clone(), target.clone()
        count += 1
    a, b, paired, power, truth, dot = sums.cpu().tolist()
    return {
        "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "dataset_fingerprint": saved["dataset_fingerprint"],
        "device": device,
        "resolution": dataset.size,
        "measured_frames": count,
        "carried_state_mse": a / count,
        "reset_every_frame_mse": b / count,
        "reset_every_frame_psnr": -10 * math.log10(max(b / count, 1e-12)),
        "paired_mse": paired / count,
        "reset_to_carried_error_ratio": b / max(a, 1e-12),
        "reset_motion_cosine": dot / max(math.sqrt(power * truth), 1e-12),
        "reset_motion_gain": dot / max(truth, 1e-12),
        "scope": "Two burn-in and two measured cycles; compare carried state to one update "
        "from all-zero state at the same phase. A successful reset-every-frame result would "
        "limit claims about learned recurrent dynamics. Same training video, no unseen-loop test.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/aperture-fast.npz"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    torch.set_num_threads(2)
    result = probe(args.weights, args.data, args.device)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
