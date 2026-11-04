"""Train a local repair state and progressive RGB recovery alongside fast loop motion."""

import hashlib
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import LoopDataset
from .losses import video_loss
from .model import ConditionedNCA, ModelConfig
from .perturbations import REPAIR_KINDS, perturb_repair, repair_teacher_step
from .train import (
    ReplayPool,
    TrainConfig,
    crop_targets,
    save_checkpoint,
    seed_all,
    supervised_window,
)


class RepairPool(ReplayPool):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_rgb = torch.zeros_like(self.states[:, :3])
        self.teacher_support = torch.ones_like(self.states[:, :1])

    def state_dict(self):
        return {
            **super().state_dict(),
            "teacher_rgb": self.teacher_rgb.cpu(),
            "teacher_support": self.teacher_support.cpu(),
        }

    def load_state_dict(self, saved):
        super().load_state_dict(saved)
        self.teacher_rgb.copy_(saved["teacher_rgb"])
        self.teacher_support.copy_(saved["teacher_support"])


def initialize_repair(
    model: ConditionedNCA, path: Path, loop_names: list[str] | None = None
) -> str:
    """Warm-start an existing repair rule, or add repair state to the image rule."""
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if loop_names is not None and saved.get("loop_names") != loop_names:
        raise ValueError("Initialization loop IDs differ from the new dataset")
    old = ModelConfig(**saved["model_config"])
    if old.repair_channel and asdict(old) == asdict(model.config):
        model.load_state_dict(saved["model"])
        return hashlib.sha256(path.read_bytes()).hexdigest()
    if (
        old.repair_channel
        or not model.config.repair_channel
        or old.channels + 1 != model.config.channels
    ):
        raise ValueError("Initialization needs the old rule plus one new repair state channel")
    for key in (
        "num_loops",
        "hidden",
        "hidden_layers",
        "embedding_dim",
        "phase_harmonics",
        "spatial_frequencies",
        "plane_channels",
        "plane_resolution",
        "plane_time_resolution",
        "perception_scales",
        "hidden_norm",
        "norm_groups",
        "direct_rgb",
        "use_phase",
        "fire_rate",
    ):
        if getattr(old, key) != getattr(model.config, key):
            raise ValueError(f"Warm-start architecture differs: {key}")
    current = model.state_dict()
    with torch.no_grad():
        for name, tensor in saved["model"].items():
            if name in current and current[name].shape == tensor.shape:
                current[name].copy_(tensor)
        model.update_in.weight.zero_()
        old_perception, new_perception = old.channels * 4, model.config.channels * 4
        weight = saved["model"]["update_in.weight"]
        model.update_in.weight[:, :old_perception].copy_(weight[:, :old_perception])
        model.update_in.weight[
            :, new_perception : new_perception + weight.shape[1] - old_perception
        ].copy_(weight[:, old_perception:])
        model.update_out.weight.zero_()
        model.update_out.weight[: old.channels].copy_(saved["model"]["update_out.weight"])
    return hashlib.sha256(path.read_bytes()).hexdigest()


def train_repair(
    dataset: LoopDataset,
    config: TrainConfig,
    output: Path,
    resume: Path | None = None,
    initialize: Path | None = None,
) -> Path:
    config.__post_init__()
    if not config.repair_channel or config.fire_rate != 1:
        raise ValueError("Repair recipe requires repair_channel and synchronous updates")
    period_ticks = dataset.period * config.steps_per_frame
    if resume and initialize:
        raise ValueError("Resume and initialization are different operations")
    size = config.patch_size or dataset.size
    alignment = math.lcm(*config.perception_scales)
    if (
        size > dataset.size
        or size % alignment
        or dataset.size % alignment
        or size <= 2 * config.loss_margin
    ):
        raise ValueError("Invalid native patch geometry")
    capacity = len(dataset.names) * config.pool_per_loop
    if config.batch_size > capacity:
        raise ValueError("Batch exceeds replay capacity")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "last.pt").exists() and resume is None:
        raise FileExistsError("Choose a new output directory or resume")
    fingerprint = dataset.fingerprint()
    seed_all(config.seed)
    device = torch.device(config.device)
    dataset = dataset.to(device)
    fields = ModelConfig.__dataclass_fields__
    model = ConditionedNCA(
        ModelConfig(
            num_loops=len(dataset.names), **{k: v for k, v in asdict(config).items() if k in fields}
        )
    ).to(device)
    parent_sha = initialize_repair(model, initialize, dataset.names) if initialize else None
    optimizer = torch.optim.Adam(
        [
            {
                "params": [
                    p for name, p in model.named_parameters() if not name.startswith("repair_head.")
                ]
            },
            {"params": model.repair_head.parameters(), "lr": config.repair_learning_rate},
        ],
        lr=config.learning_rate,
    )
    pool = RepairPool(len(dataset.names), config.pool_per_loop, config.channels, size, device)
    pool.states.copy_(model.seed(capacity, size, device))
    pool.ticks.random_(0, period_ticks)

    def origins(count):
        maximum = (dataset.size - size) // alignment
        return torch.randint(maximum + 1, (count, 2), device=device) * alignment

    pool.origins.copy_(origins(capacity))
    start_step, last_loss = 0, float("inf")
    if resume:
        saved = torch.load(resume, map_location="cpu", weights_only=False)
        if saved["dataset_fingerprint"] != fingerprint or "teacher_support" not in saved["pool"]:
            raise ValueError("Resume requires the same dataset and a repair checkpoint")
        previous = asdict(TrainConfig(**saved["train_config"]))
        mutable = {"iterations", "device", "max_seconds", "log_every", "checkpoint_every"}
        if any(
            value != previous[key] for key, value in asdict(config).items() if key not in mutable
        ):
            raise ValueError("Resume cannot change model or optimization settings")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        pool.load_state_dict(saved["pool"])
        start_step, last_loss = saved["step"], saved["last_loss"]
        torch.set_rng_state(saved["rng"]["torch"])
        random.setstate(saved["rng"]["python"])
        np.random.set_state(saved["rng"]["numpy"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(saved["rng"]["cuda"])
    else:
        with torch.no_grad():
            ids = pool.loop_ids(torch.arange(capacity, device=device))
            for _ in range(2 * period_ticks):
                pool.ticks = (pool.ticks + 1) % period_ticks
                pool.states.copy_(
                    model(
                        pool.states,
                        ids,
                        pool.ticks.float() / period_ticks,
                        origins=pool.origins,
                        canvas_size=dataset.size,
                    )
                )
            pool.teacher_rgb.copy_(
                crop_targets(
                    dataset.target(ids, pool.ticks, config.steps_per_frame), pool.origins, size
                )
            )
    (output / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    (output / "dataset.json").write_text(
        json.dumps(
            dict(
                names=dataset.names,
                fingerprint=fingerprint,
                sources=dataset.sources,
                shape=list(dataset.frames.shape),
                fps=dataset.fps,
            ),
            indent=2,
        )
        + "\n"
    )
    started, completed = time.monotonic(), start_step
    with (output / "train.jsonl").open("a" if resume else "w") as log:
        for step in range(start_step, config.iterations):
            if time.monotonic() - started >= config.max_seconds:
                break
            slots = torch.randperm(capacity, device=device)[: config.batch_size]
            ids = pool.loop_ids(slots)
            state, ticks = pool.states[slots].clone(), pool.ticks[slots].clone()
            location = pool.origins[slots].clone()
            teacher_rgb, support = (
                pool.teacher_rgb[slots].clone(),
                pool.teacher_support[slots].clone(),
            )
            fresh = torch.rand(len(slots), device=device) < config.fresh_fraction
            if fresh.any():
                state[fresh] = model.seed(int(fresh.sum()), size, device)
                ticks[fresh] = torch.randint(period_ticks, (int(fresh.sum()),), device=device)
                location[fresh] = origins(int(fresh.sum()))
                support[fresh] = 1
                teacher_rgb[fresh] = crop_targets(
                    dataset.target(ids[fresh], ticks[fresh], config.steps_per_frame),
                    location[fresh],
                    size,
                )
            for i in range(len(slots)):
                healthy = support[i].mean().item() > 0.985
                if healthy and random.random() < config.damage_probability:
                    kind = random.choice(REPAIR_KINDS)
                    circle = None
                    if kind == "blackout" and random.random() < 0.5:
                        oy, ox = location[i].tolist()
                        circle = (
                            (ox + size / 2) / dataset.size * 2 - 1,
                            (oy + size / 2) / dataset.size * 2 - 1,
                            size / dataset.size * random.uniform(0.4, 0.7),
                        )
                    state[i : i + 1], mask = perturb_repair(
                        state[i : i + 1], kind, location[i : i + 1], dataset.size, circle=circle
                    )
                    teacher_rgb[i : i + 1] = state[i : i + 1, :3].detach()
                    support[i : i + 1] = support[i : i + 1].masked_fill(mask, 0)
                elif healthy and random.random() < config.noise_probability:
                    state[i : i + 1], _ = perturb_repair(
                        state[i : i + 1], random.choice(("beat", "noise"))
                    )
            optimizer.zero_grad(set_to_none=True)
            predictions, targets, supports, support_targets = [], [], [], []
            unroll = random.randint(config.unroll_min, config.unroll_max)
            for _ in range(unroll):
                ticks = (ticks + 1) % period_ticks
                state = model(
                    state,
                    ids,
                    ticks.float() / period_ticks,
                    origins=location,
                    canvas_size=dataset.size,
                )
                clean_target = crop_targets(
                    dataset.target(ids, ticks, config.steps_per_frame), location, size
                )
                support, teacher_rgb = repair_teacher_step(
                    support, teacher_rgb, clean_target, model.config
                )
                predictions.append(model.rgb(state))
                targets.append(teacher_rgb)
                supports.append(state[:, -1:])
                support_targets.append(support)
            prediction, target = torch.stack(predictions, 1), torch.stack(targets, 1)
            predicted_support, truth_support = (
                torch.stack(supports, 1),
                torch.stack(support_targets, 1),
            )
            losses, support_losses = [], []
            for i, origin in enumerate(location.cpu().tolist()):
                y, x = supervised_window(origin, size, dataset.size, config.loss_margin)
                loss, _ = video_loss(
                    prediction[i : i + 1, ..., y, x],
                    target[i : i + 1, ..., y, x],
                    state[i : i + 1],
                    config.spectral_weight,
                    config.temporal_weight,
                )
                losses.append(loss)
                support_losses.append(
                    F.mse_loss(
                        predicted_support[i : i + 1, ..., y, x], truth_support[i : i + 1, ..., y, x]
                    )
                )
            image_loss, support_loss = (
                torch.stack(losses).mean(),
                torch.stack(support_losses).mean(),
            )
            loss = image_loss + 2 * support_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite repair loss at {step + 1}")
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.grad_clip, error_if_nonfinite=True
            )
            optimizer.step()
            pool.write(slots, state, ticks, pool.ages[slots] + unroll)
            pool.origins[slots] = location
            pool.teacher_rgb[slots], pool.teacher_support[slots] = (
                teacher_rgb.detach(),
                support.detach(),
            )
            completed, last_loss = step + 1, loss.item()
            if completed % config.log_every == 0 or completed == start_step + 1:
                row = dict(
                    step=completed,
                    loss=last_loss,
                    image_loss=image_loss.item(),
                    support_loss=support_loss.item(),
                    grad_norm=grad.item(),
                    elapsed_seconds=time.monotonic() - started,
                )
                log.write(json.dumps(row) + "\n")
                log.flush()
                print(json.dumps(row), flush=True)
            if completed % config.checkpoint_every == 0:
                save_checkpoint(
                    output / "last.pt",
                    model,
                    optimizer,
                    pool,
                    config,
                    dataset,
                    fingerprint,
                    completed,
                    last_loss,
                )
    if not np.isfinite(last_loss):
        raise RuntimeError("No training update completed")
    save_checkpoint(
        output / "last.pt",
        model,
        optimizer,
        pool,
        config,
        dataset,
        fingerprint,
        completed,
        last_loss,
    )
    report = dict(
        recipe="learned-local-repair",
        completed_iterations=completed,
        requested_iterations=config.iterations,
        stop_reason="iterations" if completed >= config.iterations else "time_budget",
        training_seconds=time.monotonic() - started,
        parameters=sum(p.numel() for p in model.parameters()),
        device=str(device),
        parent_weights_sha256=parent_sha,
        torch=torch.__version__,
        gpu=torch.cuda.get_device_name() if device.type == "cuda" else None,
        peak_cuda_bytes=torch.cuda.max_memory_allocated() if device.type == "cuda" else 0,
        curriculum=dict(front_rate=0.35, spontaneous_growth=0.002, support_loss_weight=2),
    )
    (output / "training_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return output / "last.pt"
