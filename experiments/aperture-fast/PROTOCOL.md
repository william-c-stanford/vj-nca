# Faster Aperture training and measured inference

2026-11-03, starting at `8be546d`. User authorized training and a speed increase.
Treat speed as both animation pacing and computation: the original eight-second
shader cycle becomes **two seconds**, four times faster, with 64 samples at 32fps.
All target pixels and measured model rollouts remain native 512×512.

Begin with one shader preset, one compact shared update rule and one loop code.
This is not yet multi-family code/text conditioning. Source and parameter metadata
remain attached. Start small before adding a decoder, feature planes or capacity.

Research budget: **$3 incremental GPU compute**, at most six development cycles.
Each trial has <=600 training seconds, <=20000 updates and a 900-second function
timeout. The timeout function estimate at verified rates is $0.314964 including
one A10, two CPU cores and 8GiB RAM using [Modal's posted rates](https://modal.com/pricing)
checked November 3, 2026; build/startup/transfer are separate. No deployment,
volume, automatic retry or background schedule. Download weights before evaluation.

## Gates fixed before training

Development: seed4041, two burn-in cycles, two measured cycles. Same native-pixel
quality gate: MSE < half static mean, signed motion cosine >=0.7, projected gain
0.5..1.5. Erase, noise, beat and full reset must recover in <=1s under the existing
four-consecutive-frame criterion. The evaluator's switch-to-self case is trivial
for a single loop; do not claim that it demonstrates loop switching.

Held-out: freeze selected weights, seed9091, sixteen measured cycles (32 seconds).
Do not choose hyperparameters using this result. Test other perturbation phases
after selection. No unseen-video generalization or novel semantic latent claim.

Speed: benchmark one real recurrent update plus native RGB decoding on the local
Apple M4 Max, with warmup, synchronization and both resident and CPU-readback paths.
Minimum target is 32 updates/frames per second at one update per display frame;
60fps is desirable. Report p50/p95 latency and timing scope, not only GPU kernel time.
An offline 32fps MP4 is not proof of real-time inference.

Compare with the prior 196656-parameter native-512 deep model on the same machine.
At setup, the larger feature-plane model could not execute on PyTorch2.8 MPS
because border grid sampling was unsupported. The subsequent sampler repair
clamps coordinates to pixel centers before zero-padded GPU sampling; values and
plane gradients are tested against border sampling. No CPU fallback is enabled.
First profile the compact rule. Consider cached spatial conditioning or coarse
recurrent state only if measured performance justifies the change.

Dataset fingerprint and exact local-render preparation metadata are retained in
`preparation.json`. The old source rendering was eight seconds; these weights must
be evaluated against the new two-second timing. State dependence should also be
probed by resetting state every update, so recovery is not confused with a purely
memoryless conditional renderer.
