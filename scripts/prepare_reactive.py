"""Set Aperture to a four-second cycle: twice the original eight-second preview."""

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
    parser.add_argument("--out", type=Path, default=Path("data/aperture-reactive-2x.npz"))
    parser.add_argument(
        "--preview", type=Path, default=Path("runs/aperture-reactive-2x/teacher-cycle.mp4")
    )
    args = parser.parse_args()
    torch.set_num_threads(2)
    parent = LoopDataset.load(args.source)
    if parent.period != 64 or parent.names != ["aperture"]:
        raise ValueError("Expected the exact 64-frame original Aperture cycle")
    sources = copy.deepcopy(parent.sources)
    sources[0].update(
        cycle_seconds=4.0,
        original_preview_cycle_seconds=8.0,
        preview_speed_multiplier=2,
        parent_dataset_fingerprint=parent.fingerprint(),
        temporal_stride=1,
        contour_advances_per_second=0.25,
    )
    dataset = LoopDataset(parent.frames, parent.names, 16, sources)
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
