"""Build the actual faster teacher: four contour advances per second at 64fps."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch

from vj_nca.data import LoopDataset
from vj_nca.media import save_video


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=Path("artifacts/aperture-fast-v1/targets.npz")
    )
    parser.add_argument("--out", type=Path, default=Path("data/aperture-reactive.npz"))
    parser.add_argument(
        "--preview", type=Path, default=Path("runs/aperture-reactive/teacher-cycle.mp4")
    )
    args = parser.parse_args()
    torch.set_num_threads(2)
    parent = LoopDataset.load(args.source)
    if parent.period != 64 or parent.names != ["aperture"]:
        raise ValueError("Expected the exact 64-frame original Aperture cycle")
    sources = copy.deepcopy(parent.sources)
    sources[0].update(
        cycle_seconds=0.25,
        preview_speed_multiplier=32,
        parent_dataset_fingerprint=parent.fingerprint(),
        temporal_stride=4,
        contour_advances_per_second=4,
    )
    dataset = LoopDataset(parent.frames[:, ::4], parent.names, 64, sources)
    dataset.save(args.out)
    save_video(args.preview, dataset.frames[0], dataset.fps)
    print(
        json.dumps(
            {
                "shape": list(dataset.frames.shape),
                "fps": dataset.fps,
                "cycle_seconds": dataset.period / dataset.fps,
                "fingerprint": dataset.fingerprint(),
                "file_sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(),
                "diagnostics": dataset.diagnostics(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
