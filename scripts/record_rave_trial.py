"""Capture an immutable training trial without its large optimizer/replay state."""

import argparse
import json
import shutil
from pathlib import Path

from vj_nca.artifacts import archive_directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--trials-dir", type=Path, default=Path("experiments/rave-512/trials"))
    args = parser.parse_args()
    if args.name in {".", ".."} or Path(args.name).name != args.name:
        raise ValueError("Trial name must be a single directory name")
    receipt = json.loads((args.run / "receipt.json").read_text())
    if receipt["status"] != "complete":
        raise ValueError("Record failed runs separately; this command requires complete results")
    target = args.trials_dir / args.name
    target.mkdir(parents=True, exist_ok=False)
    for name in ("submission.json", "receipt.json"):
        shutil.copyfile(args.run / name, target / name)
    for name in ("config.json", "dataset.json", "training_report.json", "train.jsonl"):
        shutil.copyfile(args.run / "train" / name, target / name)
    for name in ("model.pt", "model.json"):
        shutil.copyfile(args.run / "weights" / name, target / name)
    metrics = json.loads((args.run / "eval/metrics.json").read_text())
    (target / "metrics-dev.json").write_text(json.dumps(metrics, separators=(",", ":")) + "\n")
    (target / "training-source.zip").write_bytes(archive_directory(args.run / "source"))
    print(target)


if __name__ == "__main__":
    main()
