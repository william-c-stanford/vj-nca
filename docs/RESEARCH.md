# A shared NCA for periodic VJ loops

Research checked on November 3, 2026. This repository is an initial training experiment,
not a reproduction of DyNCA's published results or a demonstrated general video model.

## What the supplied research supports

| Work | Relevant result | Decision here |
| --- | --- | --- |
| [DyNCA, CVPR 2023](https://arxiv.org/html/2211.11417v2) · [implementation](https://github.com/IVRL/DyNCA) | A compact residual NCA combines fixed spatial derivatives, coarse perception, coordinates, appearance supervision and optical-flow supervision to learn ongoing texture dynamics. This objective need not reproduce the phase of an input video. | Use the small residual rule, coarse perception and coordinates. For exact short cycles, start with phase-aligned reconstruction and signed temporal differences. This is a DyNCA-inspired baseline with different losses, not an exact DyNCA baseline. |
| [Multi-Texture Synthesis through Signal Responsive NCA](https://arxiv.org/html/2407.05991v2) | A shared NCA uses genomic conditioning to synthesize multiple static textures, retaining regeneration and enabling interpolation. | Multi-texture conditioning already has prior art. The research question is shared **periodic dynamics**, recovery to the intended phase, and continuous adaptation to changed conditioning. Our embedding is supplied at every step so damage cannot erase the requested loop identity. |
| [Reasoning with NCA, 2026](https://arxiv.org/html/2609.36126v1) · [reproduction code](https://github.com/paradigms-of-intelligence/reasoning-nca) | Replay with fresh seeds, state noise, damage and target swaps trains persistent adaptive computation. Local channel normalization and position-dependent fixed attention help some reasoning tasks. Its ARC setup also uses temporal embeddings. | Incorporate replay, noise, circular damage and condition switching. Test hidden-only local group L2 normalization separately. Use a periodic global clock for deliberate musical synchronization. Fixed attention, local firing clocks and solution-search pruning are not necessary for this raster-video pilot; their benefit would require separate ablations. |
| [Neural Particle Automata, SIGGRAPH 2026](https://dl.acm.org/doi/full/10.1145/3799902.3811052) · [project](https://selforg-npa.github.io/) · [code](https://github.com/TheDevilWillBeBee/NPA) | Moving particles use differentiable SPH neighborhood perception and a shared update rule. Texture experiments supervise Gaussian-splatted color and density maps; CUDA neighborhood kernels make training practical. | Promising for sparse points, ribbons, swarms and transparent particle visuals. A raster-video objective additionally needs a renderer, occupancy/density supervision and efficient neighbor queries. There is no demonstrated result here that NPA improves video-loop encoding. Keep it as a separate model family rather than replacing the cheap raster baseline. |
| [Frequency-Time Diffusion with NCA, ICML 2024](https://arxiv.org/html/2401.06291v2) | Diff-NCA and FourierDiff-NCA are image denoising diffusion models. FourierDiff-NCA begins diffusion in a frequency representation before moving to image space to address global structure. “Time” includes diffusion time, not necessarily video playback. | A full diffusion sampler changes the training objective and inference cost. First test a cheap log-spectrum loss alongside coarse perception. That auxiliary loss is **not** FourierDiff-NCA, diffusion, or global Fourier communication inside the update rule. |
| [IEEE document 10663061](https://ieeexplore.ieee.org/abstract/document/10663061) | SMC-NCA concerns semi-supervised temporal action segmentation. NCA there denotes a Neighbourhood-Consistency-Aware unit. | This is not a neural cellular automaton paper and is not an architectural dependency. |

The DyNCA code declares CC BY-NC-SA 4.0 in its README. This project uses an original
PyTorch implementation of the published ideas and does not vendor its code or weights.
Reasoning-NCA's repository declares Apache-2.0. See [ATTRIBUTION.md](../ATTRIBUTION.md)
for code and training-media provenance.

## Initial model and training contract

One model has a learned embedding per training loop, a shared cell update MLP and 19
mutable state channels. The first three channels are RGB; the other 16 are latent state.
Every update receives read-only loop conditioning, Cartesian coordinates and three
sine/cosine harmonics of loop phase. Phase zero and phase one encode the same moment.
The clock and coordinates are external inputs, not a claim of autonomous oscillation
or fully decentralized inference.

Each NCA update uses identity, horizontal/vertical derivative and Laplacian perception,
averaged across full and half resolution. A stochastic per-cell mask applies the
shared residual update. Optional group normalization caps the L2 norm of each firing
cell's hidden group at one; it never normalizes RGB across pixels or across a batch.
This is an adaptation of the reasoning paper's normalization: initial testing found
that projecting zero-initialized groups to unit length caused non-finite gradients.
The unit-ball projection leaves small groups unchanged and bounds larger groups.

Training has no target frame as a model input and no teacher forcing. Stored replay
states carry loop identity, exact cycle tick and age. A chunk includes a short randomly
chosen gradient-free prefix and 8–16 differentiable updates. Fresh seeds start at random
phases. Damage affects mutable state; the embedding and clock remain available. For
condition-switch examples, a state from another loop is transplanted into a destination
loop's replay slot and trained toward the destination at the source state's phase.

The loss combines pixel reconstruction, coarse reconstruction, spatial differences,
signed temporal differences, optional spatial log-spectrum matching and an overflow
penalty. The temporal term penalizes reversed and frozen motion. An actual pretrained
optical-flow/perceptual loss remains an important comparison for textured footage;
finite differences are a low-cost starting objective, not an equivalent replacement.

All clips are retimed to a common cycle duration for this pilot. A future variable-tempo
model should also condition on phase velocity or physical time step and train across
velocities. Simply changing phase speed at inference is not yet validated.

## Evaluation that matters

Evaluate from fresh zero states and new stochastic masks, after three warmup cycles,
for at least four additional cycles. These are new trajectories of **training clips**;
there is no held-out clip generalization claim from a lookup embedding.

Measure unclipped RGB reconstruction, PSNR, signed temporal error, motion energy,
cycle-to-cycle drift and error relative to each clip's temporal-mean image. Pair clean
and perturbed branches with identical firing masks. Test central erasure, global state
noise, a synthetic brightness/latent kick, full state reset and a loop-identity switch.

A recovery success requires four consecutive output frames close to the clean branch
and close to its target error. The clean model must also beat the temporal-mean baseline
by at least 2× with nontrivial motion. Report the full curves and null recovery times
when this prerequisite fails. A poor model returning to its own blur is not success.
Compare a single-update-from-zero probe against the recurrent rollout to check whether
the model is using its evolving state rather than only fitting coordinates and phase.
The probe is diagnostic; it is not a parameter-matched feed-forward baseline.

## Experiments before scaling

1. Six original analytic loops at 32×32, 32 frames, 16 fps: translating bars, expanding
   rings, orbiting blob, rotating spokes, plasma and rotating lattice. Compare baseline,
   robustness training, and robustness plus local hidden normalization with the same
   initialization seed and architecture.
2. Add a checked, downloaded color-cycling plasma animation. Its real source is 640×480,
   64 frames and 5.12 seconds. Preserve attribution and its CC BY-SA 3.0 terms. Retiming
   makes its pilot cycle two seconds; this is recorded in the dataset.
3. Examine the long-rollout outputs before choosing a winner. Keep failed runs in the
   experimental record. Do not launch a large training run because short training loss
   decreased.
4. Only after stable recovery: train at 64×64, increase loop diversity and model capacity
   separately, repeat at least three seeds, and compare with an independent model per
   clip, a parameter-matched phase-conditioned MLP, and the official DyNCA objectives.

## Possible paper, with limits

A defensible target is a small open benchmark and shared phase-conditioned automaton
for dynamic loops, including controlled perturbation and transition evaluations.
Neither conditioning nor NCA regeneration by itself is new. A paper needs evidence of
what the shared dynamical model buys over per-loop models and a direct coordinate/phase
decoder at matched compute and capacity.
For video encoding claims, also compare H.264/AV1 and simple analytic generators at
matched quality and stored size. Do not infer a compression advantage from parameter
count alone, or from an easily generated synthetic target.

Required comparisons include replay with/without perturbations, phase conditioning,
hidden normalization, spectral loss, loop count, resolution, recovery severity and
long-horizon drift. Keep held-out **trajectories**, held-out **clips with a frozen model
and fitted code**, and held-out **clip families** distinct. Study code interpolation,
unseen condition combinations and repeated perturbations only after measuring the
training-loop behavior. Record training time, memory, parameter count, data licenses
and failure rates, and release seeds, configs, raw curves and weights.

Useful extensions, in order of the failure they address:

- Failure to represent texture motion: pretrained optical-flow and appearance losses.
- Long-range coordination failure: compare coarser perception with a small spectral
  communication branch; measure the added per-step FFT cost explicitly.
- High-frequency recovery failure: experiment with corruption-frequency curricula,
  while keeping the target periodic dynamics and sampling process clearly specified.
- Sparse particle footage: an NPA branch with SPH kernels, splatting and density loss;
  compare raster quality at matched runtime before making it a new default.
- Audio control: train impulse severity and repeated disturbance schedules, then train
  recovery at randomized timing. The present synthetic kick tests a requirement; it
  does not implement audio processing or a live VJ application.
