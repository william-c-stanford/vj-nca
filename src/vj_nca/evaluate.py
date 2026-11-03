import json
import math
from pathlib import Path

import torch
from torch import Tensor

from .data import LoopDataset
from .media import comparison_grid, save_contact_sheet, save_video
from .model import ConditionedNCA, ModelConfig
from .train import seed_all


def motion_statistics(prediction: Tensor, target: Tensor) -> dict[str, Tensor]:
    """Phase-sensitive motion scores, reduced over time, RGB and space per loop.

    Energy alone rewards flicker. Signed velocity agreement measures whether the
    model changes the right pixels in the right direction at the right phase.
    Gain 1 and cosine 1 denote perfect motion; a frozen video has gain/cosine 0.
    """
    velocity, truth = prediction.diff(dim=1), target.diff(dim=1)
    dims = (1, 2, 3, 4)
    power = truth.square().mean(dims).clamp_min(1e-12)
    pred_power = velocity.square().mean(dims)
    dot = (velocity * truth).mean(dims)
    return {
        "target_motion_mse": power / 4,
        "motion_energy_ratio": pred_power / power,
        "motion_gain": dot / power,
        "motion_cosine": dot / (pred_power * power).sqrt().clamp_min(1e-12),
        "relative_temporal_error": (velocity - truth).square().mean(dims) / power,
    }


def recovery_frame(
    paired_error: list[float],
    target_error: list[float],
    clean_error: list[float],
    eligible: bool,
    tolerance: float = 0.0025,
    sustained: int = 4,
) -> int | None:
    """Require sustained paired and absolute agreement, plus a competent clean model."""
    if not eligible or len(paired_error) < sustained:
        return None
    limit = min(tolerance, max(paired_error[0] * 0.2, 1e-7))
    for index in range(len(paired_error) - sustained + 1):
        if all(
            paired_error[j] <= limit and target_error[j] <= clean_error[j] + 0.01
            for j in range(index, index + sustained)
        ):
            return index
    return None


def perturb(state: Tensor, kind: str) -> Tensor:
    out = state.clone()
    if kind == "erase":
        h, w = state.shape[-2:]
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, h, device=state.device),
            torch.linspace(-1, 1, w, device=state.device),
            indexing="ij",
        )
        out.masked_fill_((x.square() + y.square() < 0.6**2)[None, None], 0)
    elif kind == "noise":
        out += torch.randn_like(out) * 0.3
    elif kind == "beat":
        # Synthetic brightness/latent kick; this is not a learned audio mapping.
        out[:, :3] += 0.7
        out[:, 3:] += torch.randn_like(out[:, 3:]) * 0.15
    elif kind == "full-reset":
        out.zero_()
    else:
        raise ValueError(f"Unknown perturbation: {kind}")
    return out


@torch.no_grad()
def evaluate(
    checkpoint: Path,
    dataset: LoopDataset,
    output: Path,
    device: str = "cpu",
    seed: int = 1234,
    burn_cycles: int = 3,
    cycles: int = 4,
    kinds: tuple[str, ...] = ("erase", "noise", "beat", "full-reset"),
    export: bool = True,
    phase_frame: int = 0,
) -> dict:
    if burn_cycles < 0 or cycles < 2:
        raise ValueError("Use nonnegative burn-in and at least two evaluation cycles")
    if not 0 <= phase_frame < dataset.period:
        raise ValueError("Perturbation phase_frame must be inside one target cycle")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if saved["dataset_fingerprint"] != dataset.fingerprint():
        raise ValueError(
            "Evaluation dataset differs from the checkpoint's training loop identities"
        )
    output.mkdir(parents=True, exist_ok=True)
    dataset = dataset.to(device)
    model = ConditionedNCA(ModelConfig(**saved["model_config"])).to(device).eval()
    model.load_state_dict(saved["model"])
    seed_all(seed)
    # Both resumable checkpoints and the small released tensor-only artifacts work.
    spf = (
        saved["train_config"]["steps_per_frame"]
        if "train_config" in saved
        else saved["steps_per_frame"]
    )
    period_ticks = dataset.period * spf
    ids = torch.arange(len(dataset.names), device=device)
    ticks = torch.full_like(ids, phase_frame * spf)
    state = model.seed(len(ids), dataset.size, device)
    for _ in range(burn_cycles * period_ticks):
        ticks = (ticks + 1) % period_ticks
        state = model(state, ids, ticks.float() / period_ticks)
    states = {"clean": state.clone(), **{kind: perturb(state, kind) for kind in kinds}}
    # Switching preserves the source trajectory's latent state and phase.
    states["switch"] = state.roll(1, 0).clone()
    videos = {key: [model.rgb(x).cpu()] for key, x in states.items()}
    targets = [dataset.target(ids, ticks, spf).cpu()]
    for step in range(cycles * period_ticks):
        ticks = (ticks + 1) % period_ticks
        mask = (torch.rand_like(state[:, :1]) < model.config.fire_rate).float()
        for key, current in states.items():
            states[key] = model(current, ids, ticks.float() / period_ticks, mask=mask)
        if (step + 1) % spf == 0:
            targets.append(dataset.target(ids, ticks, spf).cpu())
            for key, current in states.items():
                videos[key].append(model.rgb(current).cpu())
    videos = {key: torch.stack(frames, dim=1) for key, frames in videos.items()}
    target = torch.stack(targets, dim=1)
    clean = videos["clean"]
    if not all(torch.isfinite(x).all() for x in videos.values()):
        raise FloatingPointError("Non-finite state during long-horizon evaluation")
    # All reported errors below use the [0,1] RGB convention; predictions are NOT clipped.
    clean_error = (clean - target).square().mean((2, 3, 4)) / 4
    mean = dataset.frames.mean(1).cpu()[:, None].expand_as(target)
    mean_error = (mean - target).square().mean((1, 2, 3, 4)) / 4
    motion = motion_statistics(clean, target)
    temporal = (clean.diff(dim=1) - target.diff(dim=1)).square().mean((1, 2, 3, 4)) / 4
    paired_errors = {
        kind: (videos[kind] - clean).square().mean((2, 3, 4)) / 4 for kind in (*kinds, "switch")
    }
    target_errors = {
        kind: (videos[kind] - target).square().mean((2, 3, 4)) / 4 for kind in (*kinds, "switch")
    }
    per_loop = {}
    for index, name in enumerate(dataset.names):
        mse = clean_error[index].mean().item()
        motion_pass = (
            motion["motion_cosine"][index].item() >= 0.7
            and 0.5 <= motion["motion_gain"][index].item() <= 1.5
        )
        eligible = mse < 0.5 * mean_error[index].item() and motion_pass
        item = {
            "mse": mse,
            "psnr": -10 * math.log10(max(mse, 1e-12)),
            "temporal_mse": temporal[index].item(),
            **{key: value[index].item() for key, value in motion.items()},
            "motion_pass": motion_pass,
            "mean_frame_mse": mean_error[index].item(),
            "recovery_eligible": eligible,
            "cycle_drift_mse": (
                (clean[index, -dataset.period :] - clean[index, 1 : dataset.period + 1])
                .square()
                .mean()
                / 4
            ).item(),
            "perturbations": {},
        }
        for kind in (*kinds, "switch"):
            paired, error = paired_errors[kind], target_errors[kind]
            frame = recovery_frame(
                paired[index].tolist(), error[index].tolist(), clean_error[index].tolist(), eligible
            )
            item["perturbations"][kind] = {
                "recovery_frame": frame,
                "recovery_seconds": frame / dataset.fps if frame is not None else None,
                "initial_paired_mse": paired[index, 0].item(),
                "tail_paired_mse": paired[index, -dataset.period :].mean().item(),
                "tail_target_mse": error[index, -dataset.period :].mean().item(),
                "paired_curve": paired[index].tolist(),
                "target_curve": error[index].tolist(),
            }
        per_loop[name] = item
        times = [p["recovery_seconds"] for p in item["perturbations"].values()]
        item["quality_and_recovery_pass"] = eligible and all(
            t is not None and t <= 1.0 for t in times
        )
    # Test that removing state changes the output: a useful warning for a phase-only fit.
    probe_ticks = (
        (torch.arange(1, dataset.period + 1, device=device) + phase_frame) * spf % period_ticks
    )
    reset_outputs = []
    for tick in probe_ticks:
        reset_outputs.append(
            model.rgb(
                model(
                    model.seed(len(ids), dataset.size, device),
                    ids,
                    tick.expand(len(ids)).float() / period_ticks,
                    mask=torch.ones_like(state[:, :1]),
                )
            ).cpu()
        )
    reset_outputs = torch.stack(reset_outputs, 1)
    reset_mse = ((reset_outputs - target[:, 1 : dataset.period + 1]).square().mean() / 4).item()
    metrics = {
        "schema_version": 2,
        "checkpoint_step": saved.get("step", saved.get("training_step")),
        "seed": seed,
        "burn_cycles": burn_cycles,
        "eval_cycles": cycles,
        "perturbation_phase_frame": phase_frame,
        "note": "Training-loop reconstruction on fresh states/masks; no unseen-clip claim.",
        "mean_mse": clean_error.mean().item(),
        "mean_psnr": sum(x["psnr"] for x in per_loop.values()) / len(per_loop),
        "mean_frame_baseline_mse": mean_error.mean().item(),
        "single_step_reset_mse": reset_mse,
        "eligible_loops": sum(x["recovery_eligible"] for x in per_loop.values()),
        "passing_loops": sum(x["quality_and_recovery_pass"] for x in per_loop.values()),
        "quality_gate": {
            "motion_cosine_min": 0.7,
            "motion_gain_range": [0.5, 1.5],
            "mse_max_fraction_of_mean_frame": 0.5,
            "all_perturbations_recovery_max_seconds": 1.0,
        },
        "per_loop": per_loop,
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    if export:
        grid = comparison_grid(target, clean, videos[kinds[0]])
        save_video(output / "comparison.gif", grid, dataset.fps, zoom=max(1, 96 // dataset.size))
        save_contact_sheet(output / "contact.png", grid)
        for index in range(len(dataset.names)):
            # Numeric filenames keep user-provided names out of filesystem paths.
            save_video(
                output / f"loop-{index:02d}.mp4",
                clean[index, -dataset.period :],
                dataset.fps,
                zoom=max(1, 128 // dataset.size),
            )
    return metrics
