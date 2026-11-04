"""Prepare the selected Aperture cycle at four times the original preview pace."""

import argparse
import json
from pathlib import Path

import torch

from vj_nca.shader_data import prepare_shader_dataset

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--seconds", type=float, default=2)
    parser.add_argument("--out", type=Path, default=ROOT / "data/aperture-fast.npz")
    args = parser.parse_args()
    torch.set_num_threads(2)
    manifest = json.loads((ROOT / "datasets/shader-preview-v0.json").read_text())
    dataset = prepare_shader_dataset(
        ROOT / "shaders", manifest, ("aperture",), args.size, args.frames, args.seconds
    )
    dataset.save(args.out)
    print(
        json.dumps(
            {
                "shape": list(dataset.frames.shape),
                "fps": dataset.fps,
                "seconds": args.seconds,
                "fingerprint": dataset.fingerprint(),
                "bytes": args.out.stat().st_size,
                "diagnostics": dataset.diagnostics(),
            }
        )
    )


if __name__ == "__main__":
    main()
