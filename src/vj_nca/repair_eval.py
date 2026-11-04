"""Measure fast clean motion and deliberately progressive spatial recovery."""

import json
import math
from contextlib import ExitStack
from pathlib import Path

import torch

from .data import LoopDataset
from .media import VideoWriter
from .model import ConditionedNCA, ModelConfig
from .perturbations import coordinates, perturb_repair
from .train import seed_all


def sustained_recovery(paired, target, clean, eligible, period):
    if not eligible:
        return None
    limit = min(0.0025, max(max(paired[:period]) * 0.2, 1e-7))
    for start in range(len(paired) - period + 1):
        if all(
            paired[j] <= limit and target[j] <= clean[j] + 0.01
            for j in range(start, start + period)
        ):
            return start
    return None


@torch.inference_mode()
def evaluate_repair(
    checkpoint: Path,
    dataset: LoopDataset,
    output: Path,
    device="cuda",
    seed=4041,
    frames=192,
    phase_frame=0,
    export=True,
    circle=(0.4, 0.05, 0.68),
    ablate_repair=False,
) -> dict:
    if frames < 2 * dataset.period or len(dataset.names) != 1:
        raise ValueError("Use one loop and at least two cycles")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if saved["dataset_fingerprint"] != dataset.fingerprint():
        raise ValueError("Evaluation dataset differs from training")
    model = ConditionedNCA(ModelConfig(**saved["model_config"])).to(device).eval()
    if not model.config.repair_channel:
        raise ValueError("Use a model with the trained local repair state")
    model.load_state_dict(saved["model"])
    dataset = dataset.to(device)
    seed_all(seed)
    output.mkdir(parents=True, exist_ok=True)
    ids = torch.zeros(1, dtype=torch.long, device=device)
    state = model.seed(1, dataset.size, device)
    for tick in range(1, 4 * dataset.period + 1):
        phase = torch.tensor(
            [(phase_frame + tick) % dataset.period / dataset.period], device=device
        )
        state = model(state, ids, phase)
    kinds = ("blackout", "glitch", "rgb-split", "tear", "beat", "noise", "full-reset")
    states, masks = {"clean": state}, {}
    for kind in kinds:
        states[kind], masks[kind] = perturb_repair(
            state, kind, circle=circle if kind == "blackout" else None
        )
        if ablate_repair:
            states[kind][:, -1:] = 1
    y, x = coordinates(state, None, dataset.size)
    distance = ((x - circle[0]).square() + (y - circle[1]).square()).sqrt()
    ever_bright = dataset.frames[0].mean(1).amax(0) > -0.4
    regions = {
        "rim": (distance > circle[2] * 0.65) & (distance < circle[2]) & ever_bright,
        "core": (distance < circle[2] * 0.35) & ever_bright,
    }
    if any(not mask.any() for mask in regions.values()):
        raise ValueError("Morphology probe regions must contain visible target contours")
    errors = {k: [] for k in states}
    paired = {k: [] for k in kinds}
    support = {k: [] for k in states}
    regional = {k: [] for k in regions}
    motion = torch.zeros(4, device=device)
    baseline = 0.0
    previous = None
    with ExitStack() as stack:
        writers = {}
        if export:
            for kind in ("clean", "blackout", "glitch", "rgb-split", "tear", "beat"):
                writers[kind] = stack.enter_context(
                    VideoWriter(output / f"{kind}.mp4", dataset.size, dataset.size, dataset.fps)
                )
            writers["comparison"] = stack.enter_context(
                VideoWriter(output / "comparison.mp4", 4 * dataset.size, dataset.size, dataset.fps)
            )
        for step in range(frames + 1):
            target = dataset.frames[0, (phase_frame + step) % dataset.period][None]
            rgbs = {k: model.rgb(v) for k, v in states.items()}
            clean = rgbs["clean"]
            if not all(torch.isfinite(v).all() for v in states.values()):
                raise FloatingPointError("Non-finite repair rollout")
            values = (
                torch.stack(
                    [
                        *((rgb - target).square().mean() / 4 for rgb in rgbs.values()),
                        *((rgbs[k] - clean).square().mean() / 4 for k in kinds),
                        *(v[:, -1:].mean() for v in states.values()),
                        *(
                            (rgbs["blackout"] - clean).square().mean(1)[m].mean() / 4
                            for m in regions.values()
                        ),
                        (dataset.frames[0].mean(0)[None] - target).square().mean() / 4,
                    ]
                )
                .cpu()
                .tolist()
            )
            cursor = 0
            for collection, keys in (
                (errors, states),
                (paired, kinds),
                (support, states),
                (regional, regions),
            ):
                for k in keys:
                    collection[k].append(values[cursor])
                    cursor += 1
            baseline += values[-1]
            if previous is not None:
                dv, dt = clean - previous[0], target - previous[1]
                motion += torch.stack(
                    [
                        dv.square().mean(),
                        dt.square().mean(),
                        (dv * dt).mean(),
                        (dv - dt).square().mean(),
                    ]
                )
            previous = clean.clone(), target.clone()
            if export and step < frames:
                for kind, writer in writers.items():
                    rgb = (
                        torch.cat([target[0], clean[0], rgbs["blackout"][0], rgbs["glitch"][0]], -1)
                        if kind == "comparison"
                        else rgbs[kind][0]
                    )
                    writer.write(rgb)
            if step < frames:
                phase = torch.tensor(
                    [(phase_frame + step + 1) % dataset.period / dataset.period], device=device
                )
                states = {k: model(v, ids, phase) for k, v in states.items()}
                if ablate_repair:
                    for v in states.values():
                        v[:, -1:] = 1
    power, truth, dot, diff = motion.cpu().tolist()
    mse = sum(errors["clean"]) / len(errors["clean"])
    cosine = dot / max(math.sqrt(power * truth), 1e-12)
    gain = dot / max(truth, 1e-12)
    eligible = (
        mse < min(0.01, 0.5 * baseline / len(errors["clean"]))
        and cosine >= 0.8
        and 0.7 <= gain <= 1.3
    )
    perturbations = {}
    for kind in kinds:
        recovery = sustained_recovery(
            paired[kind], errors[kind], errors["clean"], eligible, dataset.period
        )
        perturbations[kind] = dict(
            recovery_frame=recovery,
            recovery_seconds=recovery / dataset.fps if recovery is not None else None,
            paired_curve=paired[kind],
            target_curve=errors[kind],
            support_curve=support[kind],
        )
    regional_times = {}
    for region, curve in regional.items():
        recovery = sustained_recovery(curve, curve, [0.0] * len(curve), True, dataset.period)
        regional_times[region] = recovery / dataset.fps if recovery is not None else None
    spatial = (
        regional_times["rim"] is not None
        and regional_times["core"] is not None
        and regional_times["core"] - regional_times["rim"] >= 0.125
    )
    progressive = all(
        perturbations[k]["recovery_seconds"] is not None
        and (0.25 if k == "blackout" else 0.1) <= perturbations[k]["recovery_seconds"] <= 2.5
        for k in ("blackout", "glitch", "rgb-split", "tear")
    )
    result = dict(
        evaluator="learned-progressive-repair",
        resolution=dataset.size,
        device=str(device),
        fps=dataset.fps,
        cycle_seconds=dataset.period / dataset.fps,
        seed=seed,
        frames=frames,
        phase_frame=phase_frame,
        blackout_circle=list(circle),
        repair_ablation=ablate_repair,
        checkpoint_step=saved.get("step", saved.get("training_step")),
        clean=dict(
            mse=mse,
            psnr=-10 * math.log10(max(mse, 1e-12)),
            motion_cosine=cosine,
            motion_gain=gain,
            relative_temporal_error=diff / max(truth, 1e-12),
            eligible=eligible,
        ),
        perturbations=perturbations,
        blackout_regional_recovery_seconds=regional_times,
        blackout_regional_curves=regional,
        spatial_rebuilding_pass=spatial,
        progressive_damage_pass=progressive,
        recovery_sustained_frames=dataset.period,
        quality_and_recovery_pass=eligible
        and spatial
        and progressive
        and all(
            perturbations[k]["recovery_seconds"] is not None
            and perturbations[k]["recovery_seconds"] <= 2.5
            for k in kinds
            if k != "full-reset"
        ),
        note=(
            "Recovery is sustained for a complete fast teacher cycle. Local support is "
            "predicted by the NCA; the dilation curriculum is used only in training."
        ),
    )
    (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                **result,
                "perturbations": {k: v["recovery_seconds"] for k, v in perturbations.items()},
                "blackout_regional_curves": None,
            }
        ),
        flush=True,
    )
    return result
