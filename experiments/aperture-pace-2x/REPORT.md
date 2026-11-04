# Aperture pace correction

2026-11-04. The user requested about **2× the original loop speed**. The original
shader preview had an eight-second cycle. The prior reactive experiment's
quarter-second teacher greatly exceeded that request. The active teacher and
released model now have a **four-second cycle**.

## Change

Keep the original 64 float32 frames, at 16 fps, and perform four NCA updates per
source frame using cyclic interpolation. Both training and evaluation now support
multiple updates per frame in the repair recipe. A cycle takes 256 updates at
64 Hz. This keeps the integrity-growth clock separate from motion pace; simply
slowing the previous video would also have stretched repair time by 16×.

Warm initialization now accepts an existing compatible repair model, preserving
its entire learned rule and head. Exact resume remains restricted to an unchanged
dataset and optimization settings. Portable weights carry period, fps and steps
per frame, and the cloud evaluator uses that cadence when choosing rollout length.

The teacher builder and current evaluation CLI default to the corrected dataset.
The separate reactive-v1 config and immutable weights retain the earlier experiment.
The active README and gallery point to the corrected artifact.

## Experiment and validation

One A10 pilot, initialized from reactive-v1: 1,500 updates, 110.84 training seconds,
1.34 GiB peak allocated CUDA memory. Whole function time including evaluation:
164.23 seconds; estimated running compute **$0.05747**. No second development run
was needed. Build/startup/transfer are excluded from that estimate. Modal app
`ap-8mMgS6m9blC0gI1j9AknlU` is stopped with zero tasks.

| Measure | Development, CUDA | Frozen validation, MPS |
|---|---:|---:|
| Seed / source phase frame | 4041 / 0 | 9091 / 16 |
| Clean PSNR | 28.980 dB | 28.986 dB |
| Signed motion cosine | 0.82724 | 0.82726 |
| Motion gain | 0.84144 | 0.84143 |
| Blackout recovery | 0.484 s | 0.469 s |
| Blackout rim / core | 0.281 / 0.578 s | 0.266 / 0.578 s |
| Glitch recovery | 0.297 s | 0.141 s |
| RGB splitting | 0.203 s | 0.203 s |
| Horizontal tear | 0.188 s | 0.141 s |
| Beat / noise | 0.031 / 0.016 s | 0.031 / 0.016 s |
| Full reset, separate probe | 3.156 s | 3.188 s |

Both evaluations pass the predeclared gates. Each measures eight seconds after
four full warmup cycles. Recovery must hold for one complete four-second cycle,
preventing transient agreement with a dark target frame from qualifying alone.
The large circle's visible core recovers after its rim, consistent with spatial
development rather than a uniformly delayed reveal. Native contact sheets confirm
that visual sequence. The exported clips contain two cycles at 64 fps; metadata
was checked with FFprobe.

Validation used frozen weights and a fresh corruption seed/phase, without another
selection step. This is a narrow single-preset transfer of an existing repair rule,
not a new generalization result or an inference-throughput benchmark. Full-state
reset takes longer than localized damage. No model-size change was needed.

## Retained evidence

- [Predeclared protocol](PROTOCOL.md) and [.arbor tree](.arbor/tree.json).
- [Pilot configuration, weights, loss log, source and receipt](trials/pilot).
- [Released artifact](../../artifacts/aperture-reactive-2x/MODEL_CARD.md), exact
  teacher, portable weights, frozen metrics and eight-second previews.
- [Cloud status](cloud-status.json), [validation summary](validation-summary.json).

Verification: 31 CPU tests pass, including four-update interpolation cadence,
compatible repair warm start and exact optimizer/replay resume. Ruff lint and
format checks pass. GPU training and native CUDA/MPS evaluation both complete.

Future work stays unchanged: more shader controls, code/text conditioning and
additional visual families follow this pace correction. The local-growth prior
remains engineered supervision, not unrestricted emergent morphogenesis.
