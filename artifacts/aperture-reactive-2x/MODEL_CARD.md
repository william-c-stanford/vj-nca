# Aperture at 2× the original pace

Native **512×512**, with a **four-second cycle**, twice the original eight-second
shader preview. This replaces the excessive quarter-second pace of reactive-v1.
The model was retrained against the corrected teacher on November 4, 2026.

Watch the [clean loop](clean.mp4), [large blackout](blackout.mp4),
[chromatic glitch](glitch.mp4), or [preview gallery](index.html).
Every clip lasts eight seconds and shows two full cycles at its actual training
cadence. The [comparison](comparison.mp4) shows teacher, clean, blackout and glitch.

## Timing and training

The [exact float32 teacher](targets.npz) retains all 64 original source frames at
16 fps. Four synchronous NCA updates interpolate each source interval, including
the seam: **64 updates/second, 256 updates/cycle**. The original shader period is
eight seconds; the teacher period is four. This changes the training trajectory,
not just the video export rate.

Repair keeps the previous 64 Hz clock. The model retains RGB, eight bounded hidden
channels, a local integrity channel and its learned repair head: **144,629 parameters**.
Healthy cells track phase; damaged cells rebuild locally. Blackouts clear hidden
state/integrity and set RGB to actual black. Pink/cyan/magenta glitches, RGB splitting
and horizontal tears also damage integrity. Beat/noise kicks preserve it.

Initialize from the complete reactive-v1 rule, including its repair head, and reset
optimizer/replay for the changed dataset. One small A10 pilot completed **1,500
updates in 110.84 training seconds**. Estimated function compute: **$0.0575**;
build/startup/transfer and any account charges are separate. The app is stopped.
[Receipt](receipt.json), [configuration](config.json), [training source](training-source.zip).

## Frozen validation

Weights were frozen after the first passing development run, before a fresh
corruption seed and phase. Native M4 Max validation covers eight seconds after
four cycles of warmup. No tuning followed validation.

- Clean PSNR: **28.99 dB**; signed motion cosine **0.827**; motion gain **0.841**.
- Large blackout: **0.469 s**, with rim **0.266 s** and visible core **0.578 s**.
- Chromatic glitch / RGB split / horizontal tear: **0.141 / 0.203 / 0.141 s**.
- Beat / noise: **0.031 / 0.016 s**.
- Full reset, measured separately: **3.19 s**.

All existing clean, progressive-repair and spatial-order gates pass. Recovery must
stay within tolerance for an entire four-second cycle (256 updates). The tested
blackout is a roughly 348-pixel-diameter circle. [Development metrics](metrics-dev.json),
[validation metrics](validation-9091/metrics.json), [weight freeze](selection.json).

The repair head sees only local integrity and neighboring integrity summaries.
Training uses an engineered, supervised growth curriculum. These are raw NCA RGB
outputs, without video crossfades; they are not a claim of unguided morphogenesis.

## Reproduce

```sh
PYTHONPATH=src python scripts/evaluate_reactive.py \
  --checkpoint artifacts/aperture-reactive-2x/model.pt \
  --data artifacts/aperture-reactive-2x/targets.npz \
  --out runs/reactive-2x-reproduction --device mps \
  --seed 9091 --phase-frame 16 --frames 512
```

Use `--device cuda` for CUDA, `--circle X Y RADIUS` for a different normalized
blackout placement, or `--no-export` for metrics only. FFmpeg is needed for media.
The previous BPM exporter rejects repair models; use this dataset-aware evaluator.

```sh
PYTHONPATH=src python scripts/prepare_reactive.py
PYTHONPATH=src modal run --env main cloud/train_aperture.py \
  --config configs/aperture-reactive-2x.json --data data/aperture-reactive-2x.npz \
  --initialize artifacts/aperture-reactive-v1/model.pt --out runs/reactive-2x-repeat \
  --iterations 1500 --seconds 150 --budget-usd 0.35
```

Load [model.pt](model.pt) with `torch.load(..., weights_only=True)`, construct
`ConditionedNCA(ModelConfig(**saved['model_config']))`, load `saved['model']` and
initialize with `model.seed(...)`. Supply phase `(tick % 256) / 256` at 64 Hz.
Use `perturb_repair` for consistent damage to RGB, hidden state and integrity.
[Runtime metadata](runtime.json), [dataset provenance](dataset.json),
[experiment report](../../experiments/aperture-pace-2x/REPORT.md).

One training seed and one shader preset. This validates reconstruction and recovery
under fresh corruptions, not unseen-video generalization. Code/text conditioning,
audio input and broader shader families remain future work. Code/weights: MIT.
Original teacher data and rendered media: CC0-1.0. No external artist footage.
