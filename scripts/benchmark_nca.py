"""Benchmark the actual exported checkpoint, with synchronized frame latencies."""

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import torch

from vj_nca.benchmark import benchmark_model
from vj_nca.model import ConditionedNCA, ModelConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--samples", type=int, default=120)
    args = parser.parse_args()
    torch.set_num_threads(2)
    saved = torch.load(args.weights, map_location="cpu", weights_only=True)
    model = ConditionedNCA(ModelConfig(**saved["model_config"])).to(args.device).eval()
    model.load_state_dict(saved["model"])
    result = benchmark_model(
        model, size=args.size, samples=args.samples, steps_per_frame=saved["steps_per_frame"]
    )
    result.update(
        weights_sha256=hashlib.sha256(args.weights.read_bytes()).hexdigest(),
        training_step=saved["training_step"],
        dataset_fingerprint=saved["dataset_fingerprint"],
        timestamp_utc=datetime.now(UTC).isoformat(),
    )
    if platform.system() == "Darwin":
        result["hardware"] = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
