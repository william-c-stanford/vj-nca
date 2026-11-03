"""Ephemeral, single-GPU pilots; no endpoint, schedule or persistent volume."""

import json
import time
from collections.abc import Iterator
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .pip_install("torch==2.8.0", "numpy==2.3.5", "Pillow==12.0.0")
    .add_local_dir(ROOT / "src" / "vj_nca", "/root/vj_nca")
)
app = modal.App("vj-nca-pilot")

# Published rates checked 2026-11-03. CPU and memory are charged separately.
L4_USD_PER_SECOND = 0.000222
T4_USD_PER_SECOND = 0.000164
CPU_USD_PER_SECOND = 0.0000131
MEMORY_GIB_USD_PER_SECOND = 0.00000222
MAX_FUNCTION_SECONDS = 900


@app.function(
    image=image,
    gpu=["T4", "L4"],
    cpu=(1, 1),
    memory=(4096, 4096),
    timeout=MAX_FUNCTION_SECONDS,
    startup_timeout=180,
    max_containers=1,
    scaledown_window=2,
    single_use_containers=True,
)
def pilot(
    data_bytes: bytes,
    settings: dict,
    variants: list[str],
    train_seconds: int,
    iterations: int,
    source_digest: str,
) -> Iterator[bytes]:
    import hashlib
    import shutil
    import tempfile
    import traceback

    import torch

    from vj_nca.artifacts import archive_directory
    from vj_nca.data import LoopDataset
    from vj_nca.evaluate import evaluate
    from vj_nca.train import TrainConfig, train

    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    started = time.monotonic()
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        data_path = root / "dataset.npz"
        data_path.write_bytes(data_bytes)
        dataset = LoopDataset.load(data_path)
        shutil.copytree(
            "/root/vj_nca", root / "source/vj_nca", ignore=shutil.ignore_patterns("__pycache__")
        )
        shutil.copyfile(__file__, root / "source/modal_train.py")
        yield archive_directory(root)
        summaries = []
        for variant in variants:
            cfg = TrainConfig(**settings)
            cfg.device, cfg.iterations, cfg.max_seconds = "cuda", iterations, train_seconds
            if variant == "baseline":
                cfg.damage_probability = cfg.noise_probability = cfg.swap_probability = 0
                cfg.hidden_norm = "none"
            elif variant == "robust":
                cfg.hidden_norm = "none"
            elif variant == "normalized":
                cfg.hidden_norm = "unit_ball"
            elif variant == "no-phase":
                cfg.use_phase = False
            else:
                raise ValueError(f"Unknown variant: {variant}")
            cfg.__post_init__()
            torch.cuda.reset_peak_memory_stats()
            try:
                checkpoint = train(dataset, cfg, root / variant)
                metrics = evaluate(checkpoint, dataset, root / variant / "eval", device="cuda")
                report = json.loads((root / variant / "training_report.json").read_text())
                summary = {
                    "variant": variant,
                    "status": "complete",
                    **report,
                    **{
                        k: metrics[k]
                        for k in (
                            "mean_mse",
                            "mean_psnr",
                            "eligible_loops",
                            "single_step_reset_mse",
                        )
                    },
                }
            except Exception as exc:
                failure = {
                    "variant": variant,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
                (root / variant).mkdir(exist_ok=True)
                (root / variant / "failure.json").write_text(json.dumps(failure, indent=2))
                summary = {key: value for key, value in failure.items() if key != "traceback"}
            summaries.append(summary)
            print(json.dumps(summaries[-1]), flush=True)
            yield archive_directory(root / variant, prefix=variant)
            # No automatic retry/scale-up if a run exhausts its time budget.
        receipt = {
            "provider": "Modal",
            "gpu": torch.cuda.get_device_name(),
            "source_sha256": source_digest,
            "dataset_file_sha256": hashlib.sha256(data_bytes).hexdigest(),
            "function_seconds": time.monotonic() - started,
            "price_checked": "2026-11-03",
            "summaries": summaries,
            "cost_status": "resource-time estimate; consult Modal for settled billing",
        }
        gpu_rate = T4_USD_PER_SECOND if "T4" in receipt["gpu"] else L4_USD_PER_SECOND
        rate = gpu_rate + CPU_USD_PER_SECOND + 4 * MEMORY_GIB_USD_PER_SECOND
        receipt["estimated_function_usd"] = receipt["function_seconds"] * rate
        (root / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        yield archive_directory(root, paths=[root / "receipt.json"])


@app.local_entrypoint()
def main(
    data: str = "data/procedural.npz",
    out: str = "runs/modal-pilot",
    variants: str = "baseline,robust",
    train_seconds: int = 180,
    iterations: int = 1500,
    budget_usd: float = 1.0,
    config: str = "configs/pilot.json",
):
    import hashlib

    from vj_nca.artifacts import extract_archive

    selected = variants.split(",")
    allowed = {"baseline", "robust", "normalized", "no-phase"}
    if not selected or len(selected) > 3 or any(v not in allowed for v in selected):
        raise ValueError(f"Select at most three variants from {sorted(allowed)}")
    if len(set(selected)) != len(selected):
        raise ValueError("Duplicate variants would repeat paid training")
    if not 1 <= train_seconds <= 240 or not 1 <= iterations <= 5000:
        raise ValueError("Pilot limit: 1..240 training seconds and 1..5000 iterations per variant")
    if len(selected) * train_seconds > 720:
        raise ValueError("Leave time within the function timeout for evaluation and export")
    # Use the full function timeout, not the optimistic training duration, for the budget check.
    rate = L4_USD_PER_SECOND + CPU_USD_PER_SECOND + 4 * MEMORY_GIB_USD_PER_SECOND
    estimate = MAX_FUNCTION_SECONDS * rate
    if budget_usd < estimate:
        raise ValueError(f"Timeout-based function estimate ${estimate:.3f} exceeds --budget-usd")
    target = Path(out)
    if target.exists() and any(target.iterdir()):
        raise FileExistsError("Choose an empty output directory")
    data_bytes = Path(data).read_bytes()
    if len(data_bytes) > 32 * 1024**2:
        raise ValueError("Pilot transfer limit is 32 MiB; downsample before training")
    h = hashlib.sha256()
    for path in sorted((ROOT / "src").rglob("*.py")):
        h.update(str(path.relative_to(ROOT)).encode())
        h.update(path.read_bytes())
    settings = json.loads(Path(config).read_text())
    print(
        f"One T4 or L4, no retries; function timeout estimate <= ${estimate:.3f}. "
        "Image-build CPU, startup and egress are additional.",
        flush=True,
    )
    target.mkdir(parents=True, exist_ok=True)
    submission = {
        "source_sha256": h.hexdigest(),
        "variants": selected,
        "iterations": iterations,
        "train_seconds": train_seconds,
        "config": settings,
    }
    (target / "submission.json").write_text(json.dumps(submission, indent=2) + "\n")
    for index, artifact in enumerate(
        pilot.remote_gen(data_bytes, settings, selected, train_seconds, iterations, h.hexdigest())
    ):
        extract_archive(artifact, target)
        print(f"Saved artifact bundle {index + 1} in {target}", flush=True)
    receipt = json.loads((target / "receipt.json").read_text())
    print(json.dumps(receipt, indent=2))
    if any(row["status"] == "failed" for row in receipt["summaries"]):
        raise RuntimeError(f"Some variants failed; diagnostic artifacts saved in {target}")
