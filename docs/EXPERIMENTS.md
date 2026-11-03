# Initial VJ-loop NCA experiments

November 3, 2026. All training used a Tesla T4 through Modal. These are small,
single-training-seed pilots, with new random masks/states used for evaluation.

## Delivered models

| Model | Training loops | Parameters | Updates | Training seconds | Mean PSNR | Mean RGB MSE | Loops passing quality and recovery |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [Procedural v0](../artifacts/procedural-v0/MODEL_CARD.md) | Six original analytic loops | 7,496 | 5,000 | 197.1 | 25.47 dB | 0.004392 | 5 / 6 |
| [Mixed v0](../artifacts/mixed-v0/MODEL_CARD.md) | Those six plus a downloaded color-cycling plasma loop | 11,220 | 5,000 | 189.1 | 24.73 dB | 0.005110 | 6 / 7 |

Training resolution is 32×32, with 32 frames per two-second cycle and two NCA updates
per frame. Both models use persistent phase-aware replay, fresh seeds, circular damage,
state noise, condition switching, a periodic clock and hidden group unit-ball projection.
One shared update rule and embedding table generate every loop in each model.

MSE is measured on unclipped RGB in [0,1] units. PSNR is averaged per loop; it is not
computed from the pooled MSE. The mixed model's temporal-mean-image baseline MSE is
0.02780, compared with 0.00511 for the recurrent model. A single NCA step from zero
at each requested phase has MSE 0.02940, showing a substantial benefit from its evolving
state in this diagnostic. This does not replace a separately trained feed-forward baseline.

GPU peak allocations were approximately 121 MiB and 139 MiB respectively. This is
PyTorch's allocated-memory measurement, not the entire GPU reservation. No live
frame-rate measurement has been made.

## Recovery in the seven-loop model

Evaluation starts with three burn-in cycles, then branches the same state into clean,
erased, noisy, brightness/latent-kicked, fully reset and switched-condition rollouts.
Each branch receives the same per-step firing mask. The table reports seconds at the
16 fps dataset playback rate, not wall-clock GPU inference latency.

| Loop | PSNR | Motion energy / target | Erase | Noise | Kick | Full reset | Condition switch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Drifting bars | 26.61 | 1.62 | 0.188 | 0.188 | 0.250 | 0.313 | 0.375 |
| Expanding rings | 22.13 | 1.58 | 0.313 | 0.250 | 0.313 | 0.438 | 0.625 |
| Orbit | 27.83 | 1.30 | 0.188 | 0.188 | 0.250 | 0.250 | 0.313 |
| Rotating spokes | 27.37 | 1.63 | 0.250 | 0.188 | 0.250 | 0.250 | 0.313 |
| Procedural plasma | 30.36 | 1.66 | 0.125 | 0.188 | 0.250 | 0.188 | 0.250 |
| Rotating lattice | 19.04 | 0.09 | fail | fail | fail | fail | fail |
| Public plasma | 19.75 | 0.23 | 0.188 | 0.188 | 0.250 | 0.313 | 0.313 |

Success requires the clean model to beat its temporal-mean baseline by 2× and have
nontrivial motion. Recovery then requires four consecutive frames with paired MSE
at most 0.0025 and at most 20% of the initial disturbance error, while remaining close
to the clean branch's target error. Times identify the beginning of that sustained
window. The complete criteria and curves are in the evaluator and raw metrics.

The lattice largely froze, so no recovery success is credited even when a disturbed
branch returns to that output. The public plasma passes the stated threshold but
captures only about 23% of target motion energy and remains visibly underfit. The
other loops have excess motion energy partly associated with stochastic updating;
their motion is not an exact frame-by-frame replay.

## Longer rollout with another random seed

The mixed model was also evaluated on CPU with seed 2026 for **16 cycles (32 seconds)**
after three warmup cycles. MSE remained 0.005111 and mean PSNR 24.72 dB, with the same
six successful loops. Recovery times were similar; the lattice still failed. This
provides a longer trajectory check, not an indefinite-stability proof or another
independent training seed. See
[the full longer-run metrics](../experiments/2026-11-03/mixed-final/metrics-long.json).

## Initial comparison and failures

The first numerical comparison used 7,496 parameters and training seed 42:

| Configuration | Updates | Seconds | Mean PSNR | Mean MSE | Clean loops eligible for recovery |
| --- | ---: | ---: | ---: | ---: | ---: |
| Phase-conditioned baseline, replay without perturbations | 5,000 | 149.5 | 19.88 dB | 0.01546 | 3 / 6 |
| Add damage, noise and condition switching | 5,000 | 161.6 | 21.82 dB | 0.00809 | 5 / 6 |
| Add hidden unit-ball projection | 4,596 | 180.0 | 24.68 dB | 0.00480 | 5 / 6 |

The normalization run hit its time cap, so this table does not have identical update
budgets across all rows. GPU operations were not configured for bitwise determinism.
The separately retained 5,000-update procedural model reaches 25.47 dB; this is a
development result, not a significance test.

Two engineering failures are preserved rather than omitted:

- The first unit-length hidden normalization produced non-finite gradients near the
  zero seed. Unit-ball projection resolved that failure in the tested runs.
- The initial wrapper lost output bundles on a later exception, then a source archive
  failed because Modal mounts source files with Unix-epoch timestamps. The logs retain
  aggregate comparison results, but **those initial checkpoints and per-loop curves
  were lost**. Portable archive timestamps and streaming per-variant downloads were
  verified with a real two-update GPU round trip before obtaining the retained models.

Raw initial logs, including failures, are under
[experiments/2026-11-03](../experiments/2026-11-03). The retained final runs contain
full metrics, configs, training logs, dataset provenance, resource reports, receipts
and exact source snapshots. Model exports load with PyTorch's restricted tensor loader
and were checked against their full training checkpoint outputs.

## Cost and stopped jobs

Modal's workspace billing API reported **$0.21037203 metered usage and $0 billed after
credits/allowances** at the final check. Billing data can settle later. The workspace
reported zero metered usage before this work; the figure includes the failed pilots
and setup/transfer activity visible at that check. Function-time estimates for the
two retained runs were $0.0385 and $0.0372.

All six Modal apps created for this work, including the capacity-blocked L4 attempt,
were verified stopped with zero active tasks. No endpoint, recurring schedule or
persistent volume was deployed. The original L4 job never began training; T4 capacity
was used instead.

## What this establishes

A very small shared NCA can learn several simple phase-locked moving patterns and
recover under the tested corruptions. It can also fit part of an openly licensed
external animation. This justifies a more varied **small** dataset experiment.

It does not establish unseen-video generalization, audio responsiveness, high-resolution
quality, arbitrary tempo control, real-time rendering, an autonomous learned limit
cycle, or a new state of the art. A periodic global clock is always present. A paper
still needs multiple training seeds, stronger footage, per-clip and feed-forward
baselines, optical-flow/perceptual comparisons, and a current novelty review. The
research decisions and extension order are in [RESEARCH.md](RESEARCH.md).
