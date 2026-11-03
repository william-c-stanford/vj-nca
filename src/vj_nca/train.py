import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from .data import LoopDataset
from .losses import video_loss
from .model import ConditionedNCA, ModelConfig


@dataclass
class TrainConfig:
    iterations: int = 2000
    batch_size: int = 6
    pool_per_loop: int = 8
    channels: int = 19
    hidden: int = 64
    embedding_dim: int = 12
    phase_harmonics: int = 3
    perception_scales: tuple[int, ...] = (1, 2)
    fire_rate: float = 0.5
    hidden_norm: str = "none"
    norm_groups: int = 4
    use_phase: bool = True
    steps_per_frame: int = 2
    unroll_min: int = 8
    unroll_max: int = 16
    warmup_max: int = 8
    fresh_fraction: float = 0.2
    damage_probability: float = 0.3
    noise_probability: float = 0.3
    noise_std: float = 0.12
    swap_probability: float = 0.15
    perturb_warmup: int = 200
    spectral_weight: float = 0.02
    learning_rate: float = 0.002
    grad_clip: float = 1.0
    seed: int = 42
    device: str = "cpu"
    max_seconds: float = 300
    log_every: int = 50
    checkpoint_every: int = 250

    def __post_init__(self):
        self.perception_scales = tuple(self.perception_scales)
        if not 2 <= self.unroll_min <= self.unroll_max or self.warmup_max < 0:
            raise ValueError("Need 2 <= unroll_min <= unroll_max and nonnegative warmup")
        if (
            min(
                self.iterations,
                self.batch_size,
                self.pool_per_loop,
                self.steps_per_frame,
                self.log_every,
                self.checkpoint_every,
            )
            < 1
            or self.max_seconds <= 0
        ):
            raise ValueError(
                "Iteration counts, batch sizes, intervals and time limit must be positive"
            )
        for value in (
            self.fresh_fraction,
            self.damage_probability,
            self.noise_probability,
            self.swap_probability,
        ):
            if not 0 <= value <= 1:
                raise ValueError("Probabilities must be in [0, 1]")


class ReplayPool:
    """Loop-stratified states with exact phase and age, detached between chunks."""

    def __init__(
        self, num_loops: int, per_loop: int, channels: int, size: int, device: torch.device
    ):
        self.per_loop = per_loop
        self.states = torch.zeros(num_loops * per_loop, channels, size, size, device=device)
        self.ticks = torch.zeros(num_loops * per_loop, dtype=torch.long, device=device)
        self.ages = torch.zeros_like(self.ticks)

    def loop_ids(self, slots: Tensor) -> Tensor:
        return slots // self.per_loop

    def write(self, slots: Tensor, states: Tensor, ticks: Tensor, ages: Tensor):
        self.states[slots] = states.detach()
        self.ticks[slots] = ticks.detach()
        self.ages[slots] = ages.detach()

    def state_dict(self) -> dict:
        return {key: getattr(self, key).cpu() for key in ("states", "ticks", "ages")}

    def load_state_dict(self, saved: dict):
        for key in ("states", "ticks", "ages"):
            getattr(self, key).copy_(saved[key])


def damage_state(state: Tensor, probability: float) -> Tensor:
    b, _, h, w = state.shape
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, h, device=state.device),
        torch.linspace(-1, 1, w, device=state.device),
        indexing="ij",
    )
    centers = torch.rand(b, 2, 1, 1, device=state.device) * 1.2 - 0.6
    radius = 0.2 + 0.5 * torch.rand(b, 1, 1, device=state.device)
    circle = ((x - centers[:, 0]).square() + (y - centers[:, 1]).square()) < radius.square()
    chosen = torch.rand(b, 1, 1, device=state.device) < probability
    return state.masked_fill((circle & chosen)[:, None], 0)


def noise_state(state: Tensor, probability: float, std: float) -> Tensor:
    chosen = torch.rand(len(state), 1, 1, 1, device=state.device) < probability
    spatial = torch.rand_like(state[:, :1]) < 0.25
    return state + torch.randn_like(state) * std * chosen * spatial


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(
    path: Path,
    model: ConditionedNCA,
    optimizer: torch.optim.Optimizer,
    pool: ReplayPool,
    config: TrainConfig,
    dataset: LoopDataset,
    fingerprint: str,
    step: int,
    loss: float,
):
    checkpoint = {
        "format_version": 1,
        "step": step,
        "last_loss": loss,
        "model_config": asdict(model.config),
        "train_config": asdict(config),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "pool": pool.state_dict(),
        "loop_names": dataset.names,
        "dataset_fingerprint": fingerprint,
        "rng": {
            "torch": torch.get_rng_state(),
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        },
    }
    tmp = path.with_suffix(".tmp")
    torch.save(checkpoint, tmp)
    tmp.replace(path)


def train(
    dataset: LoopDataset, config: TrainConfig, output: Path, resume: Path | None = None
) -> Path:
    config.__post_init__()
    if dataset.size < max(config.perception_scales) * 2:
        raise ValueError("Canvas too small for the chosen perception scales")
    if config.batch_size > len(dataset.names) * config.pool_per_loop:
        raise ValueError("batch_size cannot exceed the replay pool capacity")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "last.pt").exists() and resume is None:
        raise FileExistsError("Run already exists; use --resume or a new output directory")
    fingerprint = dataset.fingerprint()
    seed_all(config.seed)
    device = torch.device(config.device)
    dataset = dataset.to(device)
    model_fields = ModelConfig.__dataclass_fields__
    kwargs = {k: v for k, v in asdict(config).items() if k in model_fields}
    model = ConditionedNCA(ModelConfig(num_loops=len(dataset.names), **kwargs)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    pool = ReplayPool(
        len(dataset.names), config.pool_per_loop, config.channels, dataset.size, device
    )
    period_ticks = dataset.period * config.steps_per_frame
    pool.ticks.random_(0, period_ticks)
    first_step, last_loss = 0, float("inf")
    if resume is not None:
        saved = torch.load(resume, map_location="cpu", weights_only=False)
        if saved["dataset_fingerprint"] != fingerprint:
            raise ValueError("Checkpoint dataset differs (including loop ID ordering)")
        mutable = {"iterations", "device", "max_seconds", "log_every", "checkpoint_every"}
        if any(
            v != saved["train_config"][k] for k, v in asdict(config).items() if k not in mutable
        ):
            raise ValueError("Resume cannot change model, optimizer or training settings")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        pool.load_state_dict(saved["pool"])
        first_step, last_loss = saved["step"], saved["last_loss"]
        torch.set_rng_state(saved["rng"]["torch"])
        random.setstate(saved["rng"]["python"])
        np.random.set_state(saved["rng"]["numpy"])
        if device.type == "cuda" and saved["rng"]["cuda"]:
            torch.cuda.set_rng_state_all(saved["rng"]["cuda"])
    (output / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    (output / "dataset.json").write_text(
        json.dumps(
            {
                "names": dataset.names,
                "fingerprint": fingerprint,
                "sources": dataset.sources,
                "shape": list(dataset.frames.shape),
                "fps": dataset.fps,
            },
            indent=2,
        )
        + "\n"
    )
    start = time.monotonic()
    completed = first_step
    with (output / "train.jsonl").open("a" if resume else "w") as log:
        for step in range(first_step, config.iterations):
            if time.monotonic() - start >= config.max_seconds:
                break
            slots = torch.randperm(len(pool.states), device=device)[: config.batch_size]
            ids = pool.loop_ids(slots)
            state, ticks, ages = (
                pool.states[slots].clone(),
                pool.ticks[slots].clone(),
                pool.ages[slots].clone(),
            )
            fresh = torch.rand(len(slots), device=device) < config.fresh_fraction
            state[fresh] = 0
            ages[fresh] = 0
            ticks[fresh] = torch.randint(period_ticks, (int(fresh.sum()),), device=device)
            ramp = min(1.0, (step + 1) / max(1, config.perturb_warmup))
            if len(dataset.names) > 1 and config.swap_probability:
                switched = torch.rand(len(slots), device=device) < config.swap_probability * ramp
                source_ids = ids + torch.randint(1, len(dataset.names), ids.shape, device=device)
                source_ids %= len(dataset.names)
                source_slots = source_ids * config.pool_per_loop + torch.randint(
                    config.pool_per_loop, ids.shape, device=device
                )
                state[switched] = pool.states[source_slots[switched]]
                ticks[switched] = pool.ticks[source_slots[switched]]
                ages[switched] = 0
            state = damage_state(state, config.damage_probability * ramp)
            state = noise_state(state, config.noise_probability * ramp, config.noise_std)
            # Prefix advances trajectories without retaining an ever-growing graph.
            prefix = random.randint(0, config.warmup_max)
            with torch.no_grad():
                for _ in range(prefix):
                    ticks = (ticks + 1) % period_ticks
                    state = model(state, ids, ticks.float() / period_ticks)
            unroll = random.randint(config.unroll_min, config.unroll_max)
            predictions, targets = [], []
            optimizer.zero_grad(set_to_none=True)
            for _ in range(unroll):
                ticks = (ticks + 1) % period_ticks
                state = model(state, ids, ticks.float() / period_ticks)
                if config.noise_probability:
                    state = noise_state(
                        state, config.noise_probability * ramp * 0.15, config.noise_std
                    )
                predictions.append(model.rgb(state))
                targets.append(dataset.target(ids, ticks, config.steps_per_frame))
            loss, parts = video_loss(
                torch.stack(predictions, 1), torch.stack(targets, 1), state, config.spectral_weight
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at iteration {step + 1}")
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.grad_clip, error_if_nonfinite=True
            )
            optimizer.step()
            pool.write(slots, state, ticks, ages + prefix + unroll)
            completed, last_loss = step + 1, loss.item()
            if completed % config.log_every == 0 or completed == first_step + 1:
                row = {
                    "step": completed,
                    "loss": last_loss,
                    "elapsed_seconds": time.monotonic() - start,
                    "grad_norm": grad.item(),
                    "pool_max_age": pool.ages.max().item(),
                    **{k: v.item() for k, v in parts.items()},
                }
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
    if completed == first_step and not np.isfinite(last_loss):
        raise RuntimeError("Training time limit expired before any iteration completed")
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
    report = {
        "completed_iterations": completed,
        "requested_iterations": config.iterations,
        "stop_reason": "iterations" if completed >= config.iterations else "time_budget",
        "training_seconds": time.monotonic() - start,
        "parameters": sum(p.numel() for p in model.parameters()),
        "device": str(device),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else 0,
    }
    (output / "training_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return output / "last.pt"
