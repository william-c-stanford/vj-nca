"""Evaluate and preview the fast teacher with learned blackouts and glitch recovery."""

import argparse
from pathlib import Path

import torch

from vj_nca.data import LoopDataset
from vj_nca.repair_eval import evaluate_repair


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/aperture-reactive.npz"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seed", type=int, default=4041)
    parser.add_argument("--frames", type=int, default=192)
    parser.add_argument("--phase-frame", type=int, default=0)
    parser.add_argument("--no-export", action="store_true")
    parser.add_argument(
        "--circle", nargs=3, type=float, default=(0.4, 0.05, 0.68), metavar=("X", "Y", "RADIUS")
    )
    parser.add_argument("--ablate-repair", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    evaluate_repair(
        args.checkpoint,
        LoopDataset.load(args.data),
        args.out,
        device=args.device,
        seed=args.seed,
        frames=args.frames,
        phase_frame=args.phase_frame,
        export=not args.no_export,
        circle=tuple(args.circle),
        ablate_repair=args.ablate_repair,
    )


if __name__ == "__main__":
    main()
