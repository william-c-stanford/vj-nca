"""The fixed autoresearch development / held-out evaluation command."""

import argparse
import json
from pathlib import Path

import torch

from vj_nca.data import LoopDataset
from vj_nca.evaluate import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/mixed.npz"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=4041)
    parser.add_argument("--cycles", type=int, default=4)
    parser.add_argument("--phase-frame", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--export", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    metrics = evaluate(
        args.checkpoint,
        LoopDataset.load(args.data),
        args.out,
        args.device,
        args.seed,
        cycles=args.cycles,
        export=args.export,
        phase_frame=args.phase_frame,
    )
    compact = {key: value for key, value in metrics.items() if key != "per_loop"}
    compact["per_loop"] = {
        name: {
            key: item[key]
            for key in (
                "psnr",
                "motion_cosine",
                "motion_gain",
                "relative_temporal_error",
                "quality_and_recovery_pass",
            )
        }
        for name, item in metrics["per_loop"].items()
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
