# aperture-reactive-v1

Native 512×512 Aperture with a genuinely faster training target, black-circle damage,
pink/cyan/magenta glitches and learned spatial repair. **144,629 parameters**,
12 state channels: RGB, eight bounded hidden channels and one bounded integrity
channel. Trained November 3, 2026 from the previous Aperture weights.

Watch [blackout rebuilding](blackout.mp4), [colored glitch recovery](glitch.mp4),
[RGB splitting](rgb-split.mp4), [horizontal tears](tear.mp4), or the
[teacher / clean / blackout / glitch comparison](comparison.mp4).
The [gallery](index.html) also includes the [actual faster teacher](teacher.mp4).
All previews use their training cadence, without a playback speed multiplier.

## Faster teacher

The new training dataset contains 16 exact native 512×512 frames at **64 fps**:
**0.25 seconds per contour advance**, or four advances/second. The previous teacher
advanced one contour every two seconds. This is **8× faster physical motion** and
**4× the phase advance per recurrent update**. The NCA was trained against that
larger step; this change is not just a retimed preview.

The original 64-frame teacher is sampled every fourth frame. Shader geometry and
palette stay the same. Exact [float32 targets](targets.npz) and
[provenance](dataset.json) are included. There is one shader preset, with an
external phase clock; code/text conditioning is still future work.

## Recovery behavior

Three frozen-weight validation cases use new seeds, phases and blackout placements.
All pass the fixed clean-motion and progressive-recovery gates:

| Damage | Recovery onset, sustained for a complete 16-frame cycle |
|---|---:|
| Large black circle | 0.36–0.47s |
| Pink/cyan/magenta glitches | 0.13–0.25s |
| RGB-split distortion | 0.17–0.23s |
| Horizontal tears | 0.11–0.25s |
| Brightness/latent beat kick | 0.031s |
| Noise | 0.016s |
| Full black reset, measured separately | 1.63–1.70s |

Blackout rims recover in 0.25–0.30s; visible core contours recover later,
in 0.50–0.64s. The evaluator excludes pixels that never show teacher light, so the
dark central opening cannot falsely count as a repaired core. At64 fps these are
many actual recurrent steps, while the undamaged pattern keeps moving quickly.

Clean reconstruction is about **25.33 dB PSNR**, signed motion cosine **0.947**,
projected gain **0.858**. [Validation1](validation-9091/metrics.json),
[validation2](validation-9092/metrics.json), [validation3](validation-9093/metrics.json).
This is reconstruction of the training loop under new corruptions, not unseen-video
generalization. No inference-speed claim is made for this extension.

The additional [32-second rollout](validation-long/metrics.json), 2,048 frames at
the actual training cadence, also passes all gates: 25.33 dB, cosine 0.9474 and
the same spatial repair time course.

## How spatial repair is trained

Blackouts set RGB to actual black (-1 in the normalized state), clear hidden state
and reset integrity. Glitches alter RGB channels/positions and reset affected
hidden state/integrity. Beat and noise perturbations preserve integrity, allowing
quick rhythmic transients.

A small neural repair head sees local integrity, its neighborhood maximum and
neighborhood average. It has **no phase or coordinate input**. Its bounded positive
growth controls RGB relaxation inside the NCA update. Hidden unit-ball normalization
is retained; integrity is separately constrained to [0,1].

The training curriculum supplies progressively rebuilding targets using local
growth from surviving neighbors and a small spontaneous-growth term. At inference,
the neural head predicts the evolving integrity field. This is an **engineered,
supervised growth prior**, not a claim of unguided emergent morphogenesis.
The exported clips show raw NCA RGB states; no compositor or global fade is applied.

[Repair ablation](ablation-repair/metrics.json) holds integrity at 1 for the same
weights. Blackout recovery collapses to 0.047s, glitch recovery to 0.031s, and the
spatial/progressive gates fail. The mutable repair field causes the slower behavior;
the full pattern is not globally slowed down.

## Reproduce

```sh
PYTHONPATH=src python scripts/evaluate_reactive.py \
  --checkpoint artifacts/aperture-reactive-v1/model.pt \
  --data artifacts/aperture-reactive-v1/targets.npz \
  --out runs/reactive-reproduction --device mps --seed 9091 --phase-frame 4
```

Use `--device cuda` on a CUDA GPU. Use `--circle X Y RADIUS` for normalized
blackout coordinates, `--frames 2048 --no-export` for a 32-second rollout,
or `--ablate-repair` for the control. FFmpeg is required for previews.
Use this dataset-aware renderer rather than the previous BPM-retimed Aperture script.

Load with `torch.load(..., weights_only=True)`, construct
`ConditionedNCA(ModelConfig(**saved['model_config']))`, and load `saved['model']`.
Initialize using `model.seed(...)`. Supply phase `tick % 16 / 16`, one update per
64 fps frame. [Runtime metadata](runtime.json). Apply damage with
`vj_nca.perturbations.perturb_repair(state,kind)` so integrity is reset consistently.

To train from the previous model:

```sh
PYTHONPATH=src python scripts/prepare_reactive.py
PYTHONPATH=src modal run --env main cloud/train_aperture.py \
  --config configs/aperture-reactive.json --data data/aperture-reactive.npz \
  --initialize artifacts/aperture-fast-v1/model.pt --out runs/reactive-repeat \
  --iterations 1500 --seconds 150 --budget-usd 0.35
```

Training: **1,500 updates**, 112.34 training seconds on one A10, native 128-pixel
patches with global coordinates. [Configuration](config.json),
[training report](training_report.json), [training source](training-source.zip),
[corrected evaluation source](evaluation-source.zip), [freeze](selection.json),
and [experiment report](../../experiments/aperture-reactive/REPORT.md).
The initial cloud evaluator had a region-mask indexing bug after successful
checkpoint download; the exact weights were reevaluated locally after the fix.
Both the failure and correction are retained in the trial record.

Original target data/media: CC0-1.0. Shader-only weights and code: MIT.
No external artist assets or Electric Sheep data are used. This remains a
single-seed, single-preset training result. Real audio input, semantic controls,
multi-family transitions and installation-scale validation remain future work.
