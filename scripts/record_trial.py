"""Retain a pilot's weights, exact source, settings and metrics without replay state."""

import argparse
import json
import shutil
from pathlib import Path

from vj_nca.artifacts import archive_directory
from vj_nca.export import export_weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    if Path(args.name).name != args.name:
        raise ValueError("Trial name must be a single directory name")
    target = Path("experiments/autoresearch/trials") / args.name
    target.mkdir(parents=True, exist_ok=False)
    export_weights(args.run / "normalized/last.pt", target)
    for filename in ("submission.json", "receipt.json"):
        shutil.copyfile(args.run / filename, target / filename)
    for filename in ("config.json", "dataset.json", "training_report.json", "train.jsonl"):
        shutil.copyfile(args.run / "normalized" / filename, target / filename)
    metrics = json.loads(args.metrics.read_text())
    (target / "metrics-dev.json").write_text(json.dumps(metrics, separators=(",", ":")) + "\n")
    (target / "training-source.zip").write_bytes(archive_directory(args.run / "source"))
    print(target)


if __name__ == "__main__":
    main()
