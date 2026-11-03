import json
from pathlib import Path

import torch


def export_weights(checkpoint: Path, output: Path) -> Path:
    """Strip optimizer, replay and Python/NumPy RNG objects from a trusted training checkpoint."""
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    output.mkdir(parents=True, exist_ok=True)
    target = output / "model.pt"
    if target.exists():
        raise FileExistsError("Export already exists; choose a new output directory")
    payload = {
        "format_version": 1,
        "model_config": saved["model_config"],
        "model": {key: value.cpu() for key, value in saved["model"].items()},
        "loop_names": saved["loop_names"],
        "training_step": saved["step"],
        "steps_per_frame": saved["train_config"]["steps_per_frame"],
        "dataset_fingerprint": saved["dataset_fingerprint"],
    }
    torch.save(payload, target)
    # The released artifact must load using PyTorch's restricted tensor loader.
    torch.load(target, map_location="cpu", weights_only=True)
    (output / "model.json").write_text(
        json.dumps({k: v for k, v in payload.items() if k != "model"}, indent=2) + "\n"
    )
    return target
