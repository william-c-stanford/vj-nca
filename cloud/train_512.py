"""Bounded native-512 research runs; source, checkpoint and eval stream separately."""

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("vj-nca-512")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .pip_install("torch==2.8.0", "numpy==2.3.5", "Pillow==12.0.0")
    .add_local_dir(ROOT / "src/vj_nca", "/root/vj_nca")
)
TIMEOUT = 2100
MAX_RATE = 0.000306 + 2 * 0.0000131 + 8 * 0.00000222


@app.function(
    image=image,
    gpu=["L4", "A10G", "T4"],
    cpu=(2, 2),
    memory=(8192, 8192),
    timeout=TIMEOUT,
    startup_timeout=180,
    max_containers=1,
    scaledown_window=2,
    single_use_containers=True,
)
def run_trial(
    files: dict[str, bytes],
    manifest: dict,
    settings: dict,
    seconds: int,
    iterations: int,
    source_hash: str,
    weights: bytes | None = None,
    eval_seed: int = 4041,
    eval_cycles: int = 2,
    eval_phase_frame: int = 0,
) -> Iterator[bytes]:
    import shutil
    import tempfile
    import time
    import traceback

    import torch

    from vj_nca.artifacts import archive_directory
    from vj_nca.export import export_weights
    from vj_nca.rave_data import prepare_rave_dataset
    from vj_nca.stream_eval import evaluate_streaming
    from vj_nca.train import TrainConfig, train

    torch.set_num_threads(2)
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    started = time.monotonic()
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        (root / "raw").mkdir()
        for entry in manifest["clips"]:
            content = files[entry["file"]]
            if hashlib.sha256(content).hexdigest() != entry["sha256"]:
                raise ValueError("Source checksum mismatch")
            (root / "raw" / entry["file"]).write_bytes(content)
        shutil.copytree(
            "/root/vj_nca", root / "source/vj_nca", ignore=shutil.ignore_patterns("__pycache__")
        )
        shutil.copyfile(__file__, root / "source/train_512.py")
        (root / "source/manifest.json").write_text(json.dumps(manifest, indent=2))
        yield archive_directory(root / "source", prefix="source")
        dataset = prepare_rave_dataset(root / "raw", manifest, 512)
        if weights is None:
            cfg = TrainConfig(**settings)
            cfg.device, cfg.iterations, cfg.max_seconds = "cuda", iterations, seconds
            torch.cuda.reset_peak_memory_stats()
            checkpoint = train(dataset, cfg, root / "train")
            export_weights(checkpoint, root / "weights")
            yield archive_directory(root / "train", prefix="train")
            yield archive_directory(root / "weights", prefix="weights")
        else:
            checkpoint = root / "input-model.pt"
            checkpoint.write_bytes(weights)
        status = "complete"
        try:
            metrics = evaluate_streaming(
                checkpoint,
                dataset,
                root / "eval",
                device="cuda",
                seed=eval_seed,
                cycles=eval_cycles,
                phase_frame=eval_phase_frame,
            )
            summary = {k: metrics[k] for k in ("mean_mse", "mean_psnr", "passing_loops")}
        except Exception as exc:
            status = "evaluation-failed"
            summary = {"error": str(exc), "traceback": traceback.format_exc()}
            (root / "eval").mkdir(exist_ok=True)
            (root / "eval/failure.json").write_text(json.dumps(summary, indent=2))
        yield archive_directory(root / "eval", prefix="eval")
        elapsed = time.monotonic() - started
        gpu = torch.cuda.get_device_name()
        rate = 0.000222 if "L4" in gpu else 0.000164 if "T4" in gpu else 0.000306
        receipt = {
            "status": status,
            "gpu": gpu,
            "source_sha256": source_hash,
            "dataset_fingerprint": dataset.fingerprint(),
            "resolution": 512,
            "input_weights_sha256": hashlib.sha256(weights).hexdigest() if weights else None,
            "evaluation": {
                "seed": eval_seed,
                "cycles": eval_cycles,
                "phase_frame": eval_phase_frame,
            },
            "function_seconds": elapsed,
            "estimated_function_usd": elapsed * (rate + 2 * 0.0000131 + 8 * 0.00000222),
            "tf32": True,
            "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
            "price_checked": "2026-11-03",
            "summary": summary,
        }
        (root / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        yield archive_directory(root, paths=[root / "receipt.json"])


@app.local_entrypoint()
def main(
    out: str = "runs/rave-512-preflight",
    config: str = "configs/rave-512.json",
    seconds: int = 90,
    iterations: int = 250,
    budget_usd: float = 0.8,
    weights: str = "",
    eval_seed: int = 4041,
    eval_cycles: int = 2,
    eval_phase_frame: int = 0,
):
    from vj_nca.artifacts import extract_archive

    if not 1 <= seconds <= 1500 or not 1 <= iterations <= 20000:
        raise ValueError("Trial limit: 1500 training seconds and 20000 updates")
    if not 2 <= eval_cycles <= 4 or not 0 <= eval_phase_frame < 64:
        raise ValueError("Use 2..4 evaluation cycles and a phase frame in 0..63")
    if TIMEOUT * MAX_RATE > budget_usd:
        raise ValueError("Timeout-based function estimate exceeds budget")
    target = Path(out)
    if target.exists() and any(target.iterdir()):
        raise FileExistsError("Choose an empty trial directory")
    manifest = json.loads((ROOT / "datasets/rave-loops.json").read_text())
    files = {
        entry["file"]: (ROOT / "data/raw/electric-sheep" / entry["file"]).read_bytes()
        for entry in manifest["clips"]
    }
    if sum(map(len, files.values())) > 32 * 1024**2:
        raise ValueError("Raw clip transfer limit is 32 MiB")
    settings = json.loads(Path(config).read_text())
    weight_bytes = Path(weights).read_bytes() if weights else None
    if weight_bytes is not None and len(weight_bytes) > 32 * 1024**2:
        raise ValueError("Weight transfer limit is 32 MiB")
    h = hashlib.sha256()
    for path in sorted((ROOT / "src").rglob("*.py")):
        h.update(str(path.relative_to(ROOT)).encode())
        h.update(path.read_bytes())
    target.mkdir(parents=True, exist_ok=True)
    (target / "submission.json").write_text(
        json.dumps(
            {
                "source_sha256": h.hexdigest(),
                "config": settings,
                "train_seconds": seconds,
                "requested_iterations": iterations,
                "timeout_estimate_usd": TIMEOUT * MAX_RATE,
                "source_bytes": sum(map(len, files.values())),
                "input_weights_sha256": hashlib.sha256(weight_bytes).hexdigest()
                if weight_bytes
                else None,
                "evaluation": {
                    "seed": eval_seed,
                    "cycles": eval_cycles,
                    "phase_frame": eval_phase_frame,
                },
            },
            indent=2,
        )
    )
    for i, bundle in enumerate(
        run_trial.remote_gen(
            files,
            manifest,
            settings,
            seconds,
            iterations,
            h.hexdigest(),
            weight_bytes,
            eval_seed,
            eval_cycles,
            eval_phase_frame,
        )
    ):
        extract_archive(bundle, target)
        print(f"Saved bundle {i + 1} to {target}", flush=True)
    receipt = json.loads((target / "receipt.json").read_text())
    print(json.dumps(receipt, indent=2))
    if receipt["status"] != "complete":
        raise RuntimeError("Evaluation failed; weights and diagnostics were saved")
