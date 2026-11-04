# Faster native-512 Aperture

2026-11-03. Accepted artifact: [aperture-fast-v1](../../artifacts/aperture-fast-v1/MODEL_CARD.md).
The selected original shader now has a trained 512×512 NCA, faithful moving
contours, perturbation recovery and measured local throughput above its playback
requirements. Both teacher and recurrent state are native resolution; no upscaling
or coarse decoder is used.

## Experiment and decisions

The [protocol](PROTOCOL.md) fixed the image/motion/recovery gates, development
seed 4041 and frozen 32-second seed 9091 test before training. The new dataset has
64 float32 frames at 32 fps: a two-second cycle, four times the old preview pace.
One original preset, one loop ID. Metadata is retained for later structured conditioning.

| Trial | Total updates | Parameters | Dev PSNR | Motion cosine | Projected gain | All gates | Function USD estimate |
|---|---:|---:|---:|---:|---:|---|---:|
| Compact preflight | 3,000 | 12,236 | 12.56 | 0.076 | 0.021 | Fail | 0.0381 |
| Compact extended, resumed | 18,000 | 12,236 | 17.27 | 0.510 | 0.339 | Fail | 0.1763 |
| Compact + learned planes | 4,000 | 144,332 | 22.06 | 0.723 | 0.610 | Pass | 0.0483 |
| Planes refined, resumed | 11,546 | 144,332 | 28.01 | 0.902 | 0.838 | Pass | 0.1096 |

Trial counts are cumulative within each resumed lineage. The refinement requested
12,000 updates and stopped after 300 training seconds at 11,546. It was not stopped
by looking at test quality. Optimizer, pool, origins and RNG are preserved on resume.
Every trial's [weights, source snapshot, configuration, log and metrics](trials)
are retained, including failures. [Machine-readable table](trials.json).

More optimization alone left the small coordinate/Fourier rule short of the motion
gate. Learned XY/XT/YT priors improved the oblique contours while the local update
backbone stayed small. Most parameters in the passing model are sampled feature
planes, so parameter count alone poorly predicts per-frame compute.
This is an adaptation of the existing baseline, not an exact DyNCA reproduction.

## Frozen evaluation

[Weights were frozen](../../artifacts/aperture-fast-v1/selection.json) before the
long test. On the local M4 Max/MPS, the [32-second result](../../artifacts/aperture-fast-v1/held-out/metrics.json)
is **28.015 dB**, MSE 0.001579, motion cosine 0.90172, gain 0.837997. All four meaningful
perturbations recover: erase 0.09375s, noise 0.03125s, beat 0.0625s, full reset 0.125s.
Recovery must remain under paired and target-error limits for four consecutive
frames, and is only counted when clean reconstruction and signed motion pass.
Quarter/half/three-quarter phase tests also pass with recovery 0.03125–0.09375s.
Cycle drift MSE over the long test is 1.26e-15. Switch-to-self is trivial for one loop.

[Reset-every-frame control](../../artifacts/aperture-fast-v1/recurrence.json): MSE
0.014319 versus carried-state MSE 0.001579, a 9.07× increase; motion cosine 0.437
and gain 0.310. The learned state contributes to reconstruction. This is not an
autonomous oscillator claim: coordinates, learned priors and an external phase
clock remain inputs.

## Runtime and playback

| Checkpoint / hardware | Updates per frame | Synchronized mean ms/update, resident | RGB readback fps | Readback p95 ms/frame |
|---|---:|---:|---:|---:|
| Previous deep 512 / M4 Max | 2 | 29.03 | 17.32 | 58.63 |
| Compact preflight / M4 Max | 1 | 7.02 | 151.82 | 7.00 |
| Selected planes model / M4 Max | 1 | 7.27 | 136.37 | 7.69 |

The selected [600-frame timing](../../artifacts/aperture-fast-v1/benchmark.json)
uses 32 warmup frames per path, float32, native 512×512 RGB conversion and actual CPU
image access. [Previous deep baseline](benchmark-previous-deep.json) and
[compact pilot](benchmark-preflight.json) use 120 measured frames per path. All are
PyTorch 2.8 MPS with no CPU fallback. Runs are sequential. The deep baseline was
trained on a different dataset and failed quality; this is a pipeline-cost
comparison, not a quality-matched speed benchmark. One versus two updates/frame
accounts for part of the frame-rate gain; per-update compute improves about 4×.

The old plane sampler initially failed on Metal's unsupported border padding.
Clamping sample coordinates to pixel centers and using zero padding fixes this
on GPU. A CPU reference test checks values and plane gradients against border
sampling, and the trained native model executes on Metal.

The default training playback is 120 BPM/four beats per two-second cycle. The
exported 150 BPM preview has a 1.6-second cycle at 40 fps. Tempo changes retime the
same 64-state trajectory. Inference benchmarks exclude display, audio, encoding,
frame pacing and host-application overhead; MP4 playback is not performance proof.

## Cost, reproducibility and scope

Four development runs used an estimated **$0.3722** in function compute, using
[Modal rates](https://modal.com/pricing) checked November 3, 2026. This includes GPU,
two CPU cores and 8GiB RAM during measured function lifetimes. Image builds,
startup and transfer are separate; this is not an account billing statement.
The selected lineage accounts for about $0.1578 of that estimate. Each function
had a 900-second timeout and $0.315 maximum resource-time estimate; no deployment,
volume or automatic retry. [All four cloud apps are stopped](cloud-status.json).
The $3 / six-cycle research budget was not exhausted: four cycles were sufficient.

The exact 33.3 MiB float32 target dataset is packaged with the model to preserve its
fingerprint across graphics drivers. Original targets/media are CC0; code and
shader-only weights are MIT. No reference-artist media is used.

All 25 tests pass, including exact patch resume with the new sampling distribution
and border-equivalent plane sampling; Ruff checks and formatting pass. The accepted
artifact and negative results live on `codex/conditioned-vj-training`. The
[hypothesis tree](.arbor/tree.json) records the search and held-out merge gate.

This validates one training loop, not the full multi-program proposal. Next work
is parameter-conditioned Aperture sweeps with held-out control combinations, then
program/text conditioning and controlled transitions. Direct shader rendering,
memoryless controls and multi-seed trials remain required before paper claims.
