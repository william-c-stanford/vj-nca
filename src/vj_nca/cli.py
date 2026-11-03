import argparse
import json
from dataclasses import fields
from pathlib import Path

import torch

from .data import LoopDataset, import_video, make_procedural, merge_datasets
from .evaluate import evaluate
from .export import export_weights
from .media import comparison_grid, save_video
from .train import TrainConfig, train


def load_config(path: Path | None) -> TrainConfig:
    values = json.loads(path.read_text()) if path else {}
    unknown = set(values) - {f.name for f in fields(TrainConfig)}
    if unknown:
        raise ValueError(f"Unknown training settings: {sorted(unknown)}")
    return TrainConfig(**values)


def main():
    parser = argparse.ArgumentParser(description="Conditioned NCA video-loop training experiments")
    parser.add_argument("--threads", type=int, default=2)
    sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate", help="Create six original seamless procedural loops")
    generate.add_argument("--out", type=Path, default=Path("data/procedural.npz"))
    generate.add_argument("--size", type=int, default=32)
    generate.add_argument("--frames", type=int, default=32)
    generate.add_argument("--count", type=int, default=6)
    generate.add_argument("--fps", type=float, default=16)
    ingest = sub.add_parser("import-video", help="Import one complete, explicitly selected cycle")
    ingest.add_argument("--input", type=Path, required=True)
    ingest.add_argument("--out", type=Path, required=True)
    ingest.add_argument("--name", required=True)
    ingest.add_argument("--source-url", required=True)
    ingest.add_argument("--license", required=True)
    ingest.add_argument("--start", type=float, default=0)
    ingest.add_argument("--duration", type=float, required=True)
    ingest.add_argument("--size", type=int, default=32)
    ingest.add_argument("--frames", type=int, default=32)
    ingest.add_argument("--fps", type=float, default=16)
    merge = sub.add_parser("merge")
    merge.add_argument("inputs", nargs="+", type=Path)
    merge.add_argument("--out", type=Path, required=True)
    training = sub.add_parser("train")
    training.add_argument("--data", type=Path, required=True)
    training.add_argument("--config", type=Path)
    training.add_argument("--out", type=Path, required=True)
    training.add_argument("--resume", type=Path)
    training.add_argument("--device")
    training.add_argument("--iterations", type=int)
    training.add_argument("--max-seconds", type=float)
    testing = sub.add_parser("evaluate")
    testing.add_argument("--data", type=Path, required=True)
    testing.add_argument("--checkpoint", type=Path, required=True)
    testing.add_argument("--out", type=Path, required=True)
    testing.add_argument("--device", default="cpu")
    testing.add_argument("--seed", type=int, default=1234)
    testing.add_argument("--cycles", type=int, default=4)
    testing.add_argument("--no-export", action="store_true")
    exporting = sub.add_parser("export", help="Save small tensor-only weights for sharing")
    exporting.add_argument("--checkpoint", type=Path, required=True)
    exporting.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    if args.command == "generate":
        ds = make_procedural(args.size, args.frames, args.count, args.fps)
    elif args.command == "import-video":
        ds = import_video(
            args.input,
            args.name,
            args.size,
            args.frames,
            args.fps,
            args.start,
            args.duration,
            args.source_url,
            args.license,
        )
    elif args.command == "merge":
        ds = merge_datasets([LoopDataset.load(p) for p in args.inputs])
    elif args.command == "train":
        config = load_config(args.config)
        for name in ("device", "iterations", "max_seconds"):
            if getattr(args, name) is not None:
                setattr(config, name, getattr(args, name))
        print(train(LoopDataset.load(args.data), config, args.out, args.resume))
        return
    elif args.command == "export":
        print(export_weights(args.checkpoint, args.out))
        return
    else:
        metrics = evaluate(
            args.checkpoint,
            LoopDataset.load(args.data),
            args.out,
            args.device,
            args.seed,
            cycles=args.cycles,
            export=not args.no_export,
        )
        print(json.dumps({k: v for k, v in metrics.items() if k != "per_loop"}, indent=2))
        return
    ds.save(args.out)
    grid = comparison_grid(ds.frames, ds.frames, ds.frames)
    save_video(args.out.with_suffix(".gif"), grid, ds.fps)
    print(
        json.dumps(
            {
                "path": str(args.out),
                "fingerprint": ds.fingerprint(),
                "diagnostics": ds.diagnostics(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
