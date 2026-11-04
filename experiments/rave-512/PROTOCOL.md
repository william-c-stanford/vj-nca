# Native 512×512 psychedelic VJ training

Started 2026-11-03 from `9ba269f`, following the user's request for 512×512 and
stronger rave footage. This is a new dataset, not a resolution change to the
earlier toy benchmark. GPU research budget: **$5**, staged behind small pilots.
The hypothesis tree in `.arbor/` records changes and failures. Execution remains
sequential in the main chat as required by the workspace instructions.

## Sources and preparation

Three real Electric Sheep self-loops, generation 248, serials **27561, 27681,
27741**, by **Scott Draves and the Electric Sheep**, from the
[artist-uploaded archive](https://archive.org/details/electricsheep-flock-248-27500-1).
The per-item license is **CC BY-NC 3.0 US**. These data, reconstructions and model
weights are a noncommercial research release. The Python code remains MIT.
See the [artist's reuse policy](https://electricsheep.org/license/) and
[checksum manifest](../../datasets/rave-loops.json). Generation/serial numbers
are preserved in output video names for attribution.

Originals: 800×592, 160 frames, 30 fps, 5⅓ seconds. Center-square crop and downsample
to 512×512; retain the complete cycle as 64 frames at 12 fps, with no duration
change. No ping-pong, invented seam or enlarged low-resolution source is used.
Timestamp normalization avoids an H.264-in-AVI zero-time seek issue. This is an
opt-in import mode, preserving the earlier public-data preparation behavior.

The native dataset fingerprint is
`f16d63cc361446f96b5a5d8bbf6765b811027db2e716bee95908f4cfa9ca77aa`.
Native seam/adjacent-MSE ratios are 1.29, 1.56 and 1.50: the seam is comparable to
an ordinary sampled transition. Sources are decoded locally and on the GPU
worker by the same preparation function, and fingerprints must agree.

Beeple's artist page permits broad reuse, but its tested Vimeo/MediaFire hosts
returned unavailable/forbidden responses. Neb Motion's free pack has a specific
bundled license prohibiting animation redistribution, despite broad page copy;
it was not used. These are source-selection facts, not legal advice.

## Native-pixel training

Targets stay at 512×512. The replay pool holds 96×96 windows with their global
pixel origins, phase and age. Origins align to all perception scales. A fresh
sample changes its window and resets mutable state; state swaps retain the
destination window and requested clip code. The rule sees the correct full-image
pixel-center coordinates and their Fourier features. It never sees a target frame
as an input. Every clip uses the same update rule and global embedding table.

Loss excludes a 24-pixel halo at artificial crop edges. Genuine canvas edges
remain supervised, and edge origins are sampled explicitly. This avoids leaving
an untrained border. Crop BPTT is an approximation to long full-canvas histories;
its acceptability is established by full-canvas evaluation, not assumed from a
single-step equivalence test.

Optional learned XY, X-time and Y-time planes are **shared across all clips**.
The conditional inputs concatenate the three bilinearly sampled feature vectors
and their elementwise product. Time planes wrap cyclically, including interpolation
across phase one/zero. These are read-only learned priors during generation;
mutable recurrent state and local perception remain the NCA component. This is a
**hybrid feature-field/NCA**, inspired by planar factorization in
[K-Planes](https://sarafridov.github.io/K-Planes/), not a reproduction of that
paper, FourierDiff-NCA or a diffusion sampler. No upstream implementation is copied.

A subsequent readout ablation replaces RGB residual integration with a direct
visible-channel prediction from the same local rule; hidden channels still update
residually and undergo unit-ball projection. Firing masks apply to both parts.
This is tested because the four-cycle horizon diagnostic found accumulating RGB
bias. The change is opt-in (`direct_rgb`); older checkpoints keep their exact rule.

## Fixed evaluation and selection

Development uses seed **4041**, two full burn-in cycles and two measured cycles.
The acceptance gate is carried over from the earlier benchmark, unchanged:

- Unclipped native-pixel MSE below half the static temporal-mean baseline.
- Signed motion cosine ≥0.7 and projected motion gain 0.5–1.5 per clip.
- All five perturbations recover within one simulated video second, using four
  consecutive qualifying frames under the paired/absolute error criterion.

The five cases are erasure, noise, brightness/latent beat, full reset, and loop
switch. Sharp still images do not pass. Source imagery and model imagery are
inspected together. Full-canvas state is created directly at 512×512 and evolved
there; no prediction or recurrent state is enlarged from a lower resolution.

The native evaluator accumulates unclipped errors and signed motion sums online,
one loop at a time, retaining only a reference cycle and a display cycle. Its
quality/motion/drift scores match the existing full-video evaluator on deterministic
fixtures. It exports a 512×512 model video and a 1536×512 target/model/recovery
comparison for each loop. Current previews record **every NCA update at 24 fps**;
training target samples are 12 fps, with linear interpolation at the intermediate
updates. The first two runs exported sampled frames at 12 fps. Neither rate is a
real-time performance measurement.

After development selection, freeze the weights and run seed **9091**, four
measured cycles (21⅓ seconds). This is a holdout of rollout/corruption noise and
horizon, not unseen videos, new training seeds, or clip codes. With synchronous
firing the clean trajectory is deterministic. Do not use the held-out score to
choose hyperparameters.

## Resource and evidence controls

One ephemeral GPU, L4/A10G/T4 fallback, two CPU cores, 8 GiB host RAM, no automatic
retries, endpoint, volume or schedule. A trial has at most 1,500 training seconds,
20,000 updates and a 2,100-second function timeout. At the verified
[published rates](https://modal.com/pricing), the maximum requested function rate
is $0.00034996/second, a timeout estimate of $0.734916. Startup, builds and transfer
are additional; this is not an account-level billing cap. Check receipts and billing
before further trials, and verify all apps stopped at the end.

Raw inputs total under 6 MB and are checksum-verified before remote decoding. The
worker streams exact source first, checkpoint/portable weights after training,
and evaluation/receipt separately. A failed evaluation cannot lose an already
downloaded checkpoint. Full resumable state remains in ignored local `runs/`;
portable weights, config, log, data provenance, metrics and source snapshots are
retained in the experiment record. Do not report a training pilot as success just
because it completed or its average pixel loss fell.
