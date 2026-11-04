# Fast teacher and progressive spatial repair

2026-11-03. Starting from `761508e`, the user requested actual faster training
motion, true large blackouts, pink/cyan/magenta glitches and visibly slower,
morphological recovery. Inference acceleration is outside this experiment's goal.

## Teacher and state

Use every fourth exact float32 frame of the original 64-frame Aperture cycle,
at **64 fps**: 16 frames, **0.25 seconds per contour advance**, four advances/second.
This is 8× the physical speed of the previous two-second training target and
4× its phase advance per recurrent update. The new dataset/fingerprint and
training loss use this cadence. It is not only a faster video export.

Retain the original shader, palette and native 512×512 pixels. Initialize the image
rule and learned priors from the passing Aperture weights. Add one mutable local
integrity channel, a small neural repair head that sees only neighboring integrity,
and integrity-dependent RGB relaxation. Integrity is excluded from hidden unit-ball
normalization. Healthy cells follow the fast target; damaged cells rebuild.

Teach a local growth curriculum: max-neighbor propagation at radius 8, front
coefficient 0.35, spontaneous increment 0.002, bounded by the configured rate0.12.
Teacher integrity drives a progressively rebuilding RGB target during training.
At inference the neural head predicts integrity; no curriculum operator or video
crossfade is applied. This is an engineered spatial-repair prior, not a claim
of unguided emergent morphogenesis. Whole-state damage resets local integrity;
brightness beats preserve it and can recover quickly.

Training damage includes large native-coordinate circles (RGB=-1, hidden state=0),
smaller circles for local front examples, pink/cyan/magenta block glitches,
channel-split distortions and horizontal tears. Native patches use global positions;
full-canvas validation checks whether their learned local kinetics scale to a
large circle. Keep corruption seeds separate between development and validation.

## Fixed development gates

Seed4041, 192 measured frames (three seconds), four-cycle warmup, native 512×512 output.
Clean MSE<0.01 and half the static mean; signed motion cosine>=0.8; gain0.7..1.3.
Recovery must satisfy paired and absolute limits for **16 consecutive frames**,
an entire fast cycle, so a temporarily dark frame cannot count as repair.
Paired tolerance is min(0.0025,20% of the maximum initial-cycle error).

Blackout recovery: 0.25..2.5s. Glitch, RGB-split and tear: 0.1..2.5s. Beat and noise
may recover quickly, but must recover within2.5s. Full reset is measured separately.
Blackout rim must recover at least 0.125s before its core, evaluated only on pixels
that show target light somewhere in the cycle. This distinguishes spatial rebuilding
from a uniform delayed reveal. Inspect actual native video as well as scores.

## Validation, budget and retention

Freeze selected weights before validation. Seed9091 and additional phase/placement
probes use the same metrics without subsequent hyperparameter selection. Test the
faster teacher itself, not only the student's playback metadata. Retain old accepted
artifacts and all new negative results.

At most six development cycles and **$2 incremental function compute**. Begin with
a 150-second, 1500-update warm-start pilot. Existing Modal cap: <=600 training
seconds, 900-second function timeout, <=$0.315 resource-time estimate per function
at the previously verified rates. Build/startup/transfer are separate. One GPU,
no deployment, volume, automatic retry or background task. Download checkpoints
before evaluation. Run experiments sequentially in accordance with repository instructions.
