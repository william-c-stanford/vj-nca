"""Full-resolution evaluation with bounded video memory, one loop at a time."""

import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from .data import LoopDataset
from .evaluate import perturb, recovery_frame
from .media import save_contact_sheet, save_video
from .model import ConditionedNCA, ModelConfig
from .train import seed_all


@torch.no_grad()
def evaluate_streaming(
    checkpoint: Path,
    dataset: LoopDataset,
    output: Path,
    device: str = "cuda",
    seed: int = 4041,
    burn_cycles: int = 2,
    cycles: int = 2,
    export: bool = True,
    phase_frame: int = 0,
) -> dict:
    """Same native-pixel motion/recovery gates; retain only one preview cycle.

    Motion sums and errors use unclipped float32 RGB. Byte previews are display
    artifacts only. No frame or NCA state is upscaled to reach the target size.
    """
    if cycles < 2 or burn_cycles < 0:
        raise ValueError("Use at least two measured cycles and nonnegative burn-in")
    if not 0 <= phase_frame < dataset.period:
        raise ValueError("Perturbation phase_frame must be inside the target cycle")
    if any(
        not name or name in {".", ".."} or "/" in name or "\\" in name for name in dataset.names
    ):
        raise ValueError("Loop names must be safe artifact filenames")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if saved["dataset_fingerprint"] != dataset.fingerprint():
        raise ValueError("Evaluation dataset differs from training")
    output.mkdir(parents=True, exist_ok=True)
    dataset = dataset.to(device)
    model = ConditionedNCA(ModelConfig(**saved["model_config"])).to(device).eval()
    model.load_state_dict(saved["model"])
    seed_all(seed)
    spf = saved.get("train_config", {}).get("steps_per_frame", saved.get("steps_per_frame"))
    period_ticks = dataset.period * spf
    ids = torch.arange(len(dataset.names), device=device)
    offset = phase_frame * spf
    tick = offset
    warm = model.seed(len(ids), dataset.size, device)
    for _ in range(burn_cycles * period_ticks):
        tick = (tick + 1) % period_ticks
        warm = model(warm, ids, torch.full_like(ids, tick / period_ticks, dtype=torch.float32))
    kinds = ("erase", "noise", "beat", "full-reset", "switch")
    per_loop = {}
    for index, name in enumerate(dataset.names):
        loop_id = ids[index : index + 1]
        clean_start = warm[index : index + 1]
        states = {"clean": clean_start.clone()}
        states.update({k: perturb(clean_start, k) for k in kinds if k != "switch"})
        states["switch"] = warm[(index - 1) % len(ids)].unsqueeze(0).clone()
        errors = {key: [] for key in states}
        paired = {key: [] for key in kinds}
        preview, reference = [], []
        mean_frame = dataset.frames[index].mean(0)
        baseline, drift, drift_count = 0.0, 0.0, 0
        motion = torch.zeros(4, device=device)
        last_clean = last_target = None
        for step in range(cycles * period_ticks + 1):
            if step:
                phase = torch.full(
                    (1,), ((offset + step) % period_ticks) / period_ticks, device=device
                )
                mask = (torch.rand_like(clean_start[:, :1]) < model.config.fire_rate).float()
                for key, state in states.items():
                    states[key] = model(state, loop_id, phase, mask=mask)
            if export and step < period_ticks:
                display_target = dataset.target(
                    loop_id, torch.tensor([(offset + step) % period_ticks], device=device), spf
                )
                display = torch.cat(
                    [
                        display_target[0],
                        model.rgb(states["clean"])[0],
                        model.rgb(states["erase"])[0],
                    ],
                    dim=-1,
                )
                preview.append(((display.cpu().clamp(-1, 1) + 1) * 127.5).round().byte())
            if step % spf:
                continue
            frame = step // spf
            target = dataset.target(
                loop_id, torch.tensor([(offset + step) % period_ticks], device=device), spf
            )
            rgbs = {k: model.rgb(v) for k, v in states.items()}
            clean = rgbs["clean"]
            if not all(torch.isfinite(v).all() for v in rgbs.values()):
                raise FloatingPointError("Non-finite full-canvas rollout")
            values = (
                torch.stack(
                    [
                        *((v - target).square().mean() / 4 for v in rgbs.values()),
                        *((rgbs[k] - clean).square().mean() / 4 for k in kinds),
                        (mean_frame - target).square().mean() / 4,
                    ]
                )
                .cpu()
                .tolist()
            )
            for k, v in zip(rgbs, values[: len(rgbs)], strict=True):
                errors[k].append(v)
            for k, v in zip(kinds, values[len(rgbs) : -1], strict=True):
                paired[k].append(v)
            baseline += values[-1]
            if last_clean is not None:
                velocity, truth = clean - last_clean, target - last_target
                motion += torch.stack(
                    [
                        velocity.square().mean(),
                        truth.square().mean(),
                        (velocity * truth).mean(),
                        (velocity - truth).square().mean(),
                    ]
                )
            last_clean, last_target = clean.clone(), target.clone()
            if 1 <= frame <= dataset.period:
                reference.append(clean.cpu())
            elif frame > dataset.period:
                drift += (
                    (clean.cpu() - reference[(frame - 1) % dataset.period]).square().mean() / 4
                ).item()
                drift_count += 1
        predicted_power, target_power, dot, temporal = motion.cpu().tolist()
        ratio = predicted_power / max(target_power, 1e-12)
        gain = dot / max(target_power, 1e-12)
        cosine = dot / max(math.sqrt(predicted_power * target_power), 1e-12)
        mse = sum(errors["clean"]) / len(errors["clean"])
        mean_error = baseline / len(errors["clean"])
        eligible = mse < 0.5 * mean_error and cosine >= 0.7 and 0.5 <= gain <= 1.5
        item = {
            "mse": mse,
            "psnr": -10 * math.log10(max(mse, 1e-12)),
            "mean_frame_mse": mean_error,
            "motion_cosine": cosine,
            "motion_gain": gain,
            "motion_energy_ratio": ratio,
            "relative_temporal_error": temporal / max(target_power, 1e-12),
            "temporal_mse": temporal / (cycles * dataset.period) / 4,
            "recovery_eligible": eligible,
            "cycle_drift_mse": drift / max(drift_count, 1),
            "perturbations": {},
        }
        for kind in kinds:
            recovered = recovery_frame(paired[kind], errors[kind], errors["clean"], eligible)
            item["perturbations"][kind] = {
                "recovery_frame": recovered,
                "recovery_seconds": recovered / dataset.fps if recovered is not None else None,
                "paired_curve": paired[kind],
                "target_curve": errors[kind],
                "tail_paired_mse": sum(paired[kind][-dataset.period :]) / dataset.period,
            }
        item["quality_and_recovery_pass"] = eligible and all(
            p["recovery_seconds"] is not None and p["recovery_seconds"] <= 1
            for p in item["perturbations"].values()
        )
        per_loop[name] = item
        if export:
            video = torch.stack(preview).float() / 127.5 - 1
            # Full 512-pixel panels, not enlarged low-resolution predictions.
            save_video(output / f"{name}-comparison.mp4", video, dataset.fps * spf)
            save_video(
                output / f"{name}.mp4",
                video[..., dataset.size : 2 * dataset.size],
                dataset.fps * spf,
            )
            sheet = F.interpolate(video, size=(192, 576), mode="area")
            save_contact_sheet(output / f"{name}-contact.png", sheet, columns=4)
        print(
            json.dumps(
                {
                    "loop": name,
                    "psnr": item["psnr"],
                    "motion_cosine": cosine,
                    "motion_gain": gain,
                    "pass": item["quality_and_recovery_pass"],
                }
            ),
            flush=True,
        )
    metrics = {
        "schema_version": 2,
        "evaluator": "streaming-native-pixels",
        "resolution": dataset.size,
        "target_sample_fps": dataset.fps,
        "preview_fps": dataset.fps * spf,
        "seed": seed,
        "burn_cycles": burn_cycles,
        "eval_cycles": cycles,
        "perturbation_phase_frame": phase_frame,
        "checkpoint_step": saved.get("step", saved.get("training_step")),
        "mean_mse": sum(v["mse"] for v in per_loop.values()) / len(per_loop),
        "mean_psnr": sum(v["psnr"] for v in per_loop.values()) / len(per_loop),
        "passing_loops": sum(v["quality_and_recovery_pass"] for v in per_loop.values()),
        "per_loop": per_loop,
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics
