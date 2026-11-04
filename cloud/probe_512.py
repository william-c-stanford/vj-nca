"""Cheap native-canvas diagnostic: motion and reconstruction versus recurrent age."""

import json
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("vj-nca-512-horizon")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .pip_install("torch==2.8.0", "numpy==2.3.5", "Pillow==12.0.0")
    .add_local_dir(ROOT / "src/vj_nca", "/root/vj_nca")
)


@app.function(
    image=image,
    gpu=["L4", "A10G", "T4"],
    cpu=(2, 2),
    memory=(8192, 8192),
    timeout=180,
    startup_timeout=180,
    max_containers=1,
    scaledown_window=2,
    single_use_containers=True,
)
def probe(files: dict[str, bytes], manifest: dict, weights: bytes) -> dict:
    import tempfile
    import time

    import torch

    from vj_nca.model import ConditionedNCA, ModelConfig
    from vj_nca.rave_data import prepare_rave_dataset

    torch.set_num_threads(2)
    torch.backends.cudnn.allow_tf32 = True
    started = time.monotonic()
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        for name, content in files.items():
            (root / name).write_bytes(content)
        dataset = prepare_rave_dataset(root, manifest, 512)
        (root / "model.pt").write_bytes(weights)
        saved = torch.load(root / "model.pt", map_location="cpu", weights_only=True)
        if saved["dataset_fingerprint"] != dataset.fingerprint():
            raise ValueError("Dataset mismatch")
        dataset = dataset.to("cuda")
        model = ConditionedNCA(ModelConfig(**saved["model_config"])).cuda().eval()
        model.load_state_dict(saved["model"])
        spf = saved["steps_per_frame"]
        ticks = dataset.period * spf
        ids = torch.arange(len(dataset.names), device="cuda")
        state = model.seed(len(ids), 512, "cuda")
        old_rgb = old_target = None
        sums = torch.zeros(4, len(ids), 4, device="cuda")
        errors = []
        with torch.no_grad():
            for step in range(4 * ticks + 1):
                if step:
                    state = model(
                        state,
                        ids,
                        torch.full_like(ids, (step % ticks) / ticks, dtype=torch.float32),
                    )
                if step % spf:
                    continue
                rgb = model.rgb(state)
                target = dataset.target(ids, torch.full_like(ids, step % ticks), spf)
                errors.append(((rgb - target).square().mean((1, 2, 3)) / 4).cpu().tolist())
                if old_rgb is not None:
                    v, t = rgb - old_rgb, target - old_target
                    sums[(step - 1) // ticks] += torch.stack(
                        [
                            v.square().mean((1, 2, 3)),
                            t.square().mean((1, 2, 3)),
                            (v * t).mean((1, 2, 3)),
                            (v - t).square().mean((1, 2, 3)),
                        ],
                        dim=-1,
                    )
                old_rgb, old_target = rgb.clone(), target.clone()
        moments = sums.cpu().tolist()
        per_loop = {}
        for i, name in enumerate(dataset.names):
            per_loop[name] = []
            for cycle in range(4):
                p, t, d, e = moments[cycle][i]
                curve = [
                    row[i]
                    for row in errors[cycle * dataset.period + 1 : (cycle + 1) * dataset.period + 1]
                ]
                per_loop[name].append(
                    {
                        "cycle": cycle + 1,
                        "mse": sum(curve) / len(curve),
                        "motion_gain": d / max(t, 1e-12),
                        "motion_cosine": d / max((p * t) ** 0.5, 1e-12),
                    }
                )
        return {
            "gpu": torch.cuda.get_device_name(),
            "seconds": time.monotonic() - started,
            "resolution": 512,
            "per_loop": per_loop,
            "target_error_curve": errors,
        }


@app.local_entrypoint()
def main(
    weights: str = "runs/rave-512-motion/weights/model.pt",
    out: str = "experiments/rave-512/horizon-probe.json",
):
    manifest = json.loads((ROOT / "datasets/rave-loops.json").read_text())
    files = {
        e["file"]: (ROOT / "data/raw/electric-sheep" / e["file"]).read_bytes()
        for e in manifest["clips"]
    }
    result = probe.remote(files, manifest, Path(weights).read_bytes())
    Path(out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "target_error_curve"}, indent=2))
