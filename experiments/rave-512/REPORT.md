# Native-512 exploration — source direction superseded

2026-11-03. **No accepted 512×512 model.** The user rejected the Electric Sheep
aesthetic and proposed shader-generated data with code/description conditioning.
The active run was stopped, and all seven Modal apps are recorded as stopped in
[modal-apps-after-pivot.json](modal-apps-after-pivot.json). The proposed next
experiment is in [SHADER_CONDITIONING.md](../../docs/SHADER_CONDITIONING.md).

## What works

Native 512×512 source preparation, global-coordinate patch BPTT, shared cyclic
feature planes, deeper update rules and bounded-memory full-canvas evaluation.
The evaluator reproduces the earlier accepted 32×32 artifact's seven passing
loops; see [evaluator-regression.json](evaluator-regression.json). Native-pixel
targets and actual 512×512 recurrent rollouts are used, not enlarged predictions.
All 24 CPU tests pass, including exact patch/plane optimizer resume, periodic
sampling and firing-mask semantics for the optional direct RGB rule.

## Measured trials

Fixed gates and source provenance are in [PROTOCOL.md](PROTOCOL.md). A pass requires
native image/motion fidelity and five perturbation recoveries within one second.
All completed trials use training seed 42 and development seed 4041.

| Intervention | Updates | Parameters | Training seconds | Mean PSNR | Pass | Estimated function USD |
|---|---:|---:|---:|---:|---:|---:|
| Native-patch preflight | 250 | 85,872 | 8.44 | 10.88 | 0/3 | 0.0368 |
| Wider/deeper rule | 15,000 | 196,656 | 497.72 | 20.19 | 0/3 | 0.2174 |
| Shared planes preflight | 1,000 | 737,328 | 43.31 | 14.10 | 0/3 | 0.0667 |
| Planes + stronger motion loss | 20,000 | 737,328 | 759.88 | 18.53 | 0/3 | 0.3165 |

Each completed trial retains portable weights, config, exact source, provenance,
training log, metrics and receipt under `trials/`. The total of these four function
estimates is $0.6374. These exclude image builds, startup, transfer, the short
horizon probe and interrupted jobs; they are not final account billing.

The deeper rule mostly froze: motion gains 0.0028, 0.0005 and 0.0141. Learned planes
and stronger motion supervision improved gains to 0.0479, 0.0498 and 0.0899, still
far below the fixed 0.5 minimum. Native previews also show color drift and blur.
An improvement in PSNR or a completed GPU job is not an accepted result.

The [four-cycle horizon diagnostic](horizon-probe.json) shows weak motion from the
first cycle and growing RGB error later. [Spatial low-pass controls](spatial-motion-spectrum.json)
retain substantially more signed motion than the model, so loss of native detail
alone does not explain the failure. These controls are not expressivity bounds on
nonlinear fields. A direct-visible-RGB ablation was interrupted after 8,250 observed
updates when the user changed the source direction. It has no evaluated model or
score; see [readout-interrupted.json](readout-interrupted.json).

The first horizon-probe attempt failed during startup because a local package
import was unavailable remotely. It was stopped, the standalone image declaration
was corrected, and the next short probe completed. No quality result is attributed
to that failed startup.

## Limits and next direction

No held-out native-512 evaluation was run because no candidate passed development.
The 32×32 `mixed-v1` artifact remains the previously accepted result. Do not treat
these detailed fractal clips as a validated training set for the user's preferred
VJ aesthetic, or these weights as a finished 512×512 release. They are CC BY-NC
research artifacts; the implementation remains MIT.

The shader pivot should first establish an accepted visual style, then test
parameter/code/text conditioning against parameter-only and clip-ID baselines.
Preserve the reproducible high-resolution pipeline and negative evidence while
testing that new objective. No additional GPU training is queued.
