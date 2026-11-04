# Fast training motion and learned spatial rebuilding

2026-11-03. User correction: speed meant the **training teacher**, not inference.
The new teacher moves four contours per second at 64 fps. True black circles and
chromatic distortions now train a local integrity field with visible rebuilding.
Accepted artifact: [aperture-reactive-v1](../../artifacts/aperture-reactive-v1/MODEL_CARD.md).

## Method and development

The previous teacher advanced one contour in two seconds, with 64 frames at 32 fps.
The new dataset samples every fourth exact frame and uses 16 frames at 64 fps,
giving a 0.25-second cycle. Temporal supervision and every recurrent training step
use that larger phase jump. This produces 8× physical speed, not only changed
MP4 playback metadata. Geometry and original ivory/acid-green palette are retained.

The previous model's zero-valued erasure was gray in normalized RGB, and its
recovery was too quick for the requested effect. New blackouts use RGB=-1 and
erase all hidden/integrity state within large native-coordinate circles. Other
training events include pink/cyan/magenta block glitches, RGB-channel splits and
horizontal tears. Beat/noise retain integrity and may recover quickly.

One extra mutable integrity channel and a 41-parameter local neural repair head
separate repair kinetics from fast image tracking. The head sees only local
integrity and its max/average neighborhood at radius 8; it does not see position
or phase. The main rule can see integrity through perception. Integrity governs
local RGB relaxation; healthy cells still follow the fast teacher every update.
The old hidden unit-ball normalization is retained, excluding the bounded
integrity scalar. Total parameters: 144,629 versus 144,332 in the parent.

The progressive curriculum grows teacher integrity from surviving neighbors with
coefficient 0.35, spontaneous increment 0.002 and rate cap 0.12. Corresponding RGB
targets rebuild locally while the teacher keeps moving. The learned head predicts
integrity during inference. This is a designed local-growth prior taught to the
model; it is not a claim that unrestricted DyNCA naturally developed morphogenesis.

The [fixed protocol](PROTOCOL.md) requires clean MSE<0.01 and half static mean,
motion cosine>=0.8, gain0.7..1.3. Recovery must hold over an entire 16-frame cycle.
Blackout recovery must take 0.25..2.5s; glitches 0.1..2.5s. The visible core must
recover at least 0.125s later than the rim. This rules out rewarding instant repair
or counting a temporarily dark frame as reconstruction.

One small warm-start pilot was sufficient: **1,500 updates**, 112.34 training seconds
on an A10, 128-pixel patches and native 512×512 evaluation. It passed development with
PSNR 25.33 dB, cosine 0.9474, gain 0.8575, blackout 0.453s and chromatic repair 0.156–0.219s.
The blackout rim/core times were 0.281/0.578s. There were no subsequent training
changes selected from validation results. [Exact trial record](trials/preflight).

## Frozen validation and control

[Freeze](../../artifacts/aperture-reactive-v1/selection.json) preceded validation.

| Seed / phase | Blackout circle(x,y,r) | Blackout recovery | Rim / core | Glitch / split / tear | Pass |
|---|---|---:|---:|---|---|
|9091 /4|0.4,0.05,0.68|0.469s|0.281 /0.578s|0.219 /0.234 /0.141s|Yes|
|9092 /8|-0.3,0.1,0.72|0.469s|0.297 /0.641s|0.250 /0.219 /0.250s|Yes|
|9093 /12|0,-0.4,0.6|0.359s|0.250 /0.500s|0.125 /0.172 /0.109s|Yes|

All cases maintain clean PSNR 25.33–25.34 dB, signed motion cosine 0.9474 and gain 0.8575.
Beat recovery is 0.031s and noise 0.016s. Full reset, separately measured, takes 1.63–1.70s.
These are simulated seconds at the actual 64 fps training cadence, not wall-clock
inference timings. [Machine-readable validation summary](validation-summary.json).

An additional frozen 32-second rollout (2,048 frames at 64 fps) passes all gates,
with PSNR 25.3345 dB, motion cosine 0.94744 and gain 0.85751. Its blackout rim/core
times remain 0.281/0.578s. [Long result](../../artifacts/aperture-reactive-v1/validation-long/metrics.json).

Holding integrity at 1 with the **same weights** reduces blackout repair to 0.047s
and color distortions to 0.031s. Rim/core become 0.047/0.063s and both progressive
and spatial gates fail. [Ablation](../../artifacts/aperture-reactive-v1/ablation-repair/metrics.json).
This directly tests that mutable repair state causes the desired time course.

## Checks, provenance and cost

The initial cloud evaluation failed with a region-mask shape error after training,
checkpoint and portable-weight download had succeeded. It added an extra batch
axis to a native spatial mask. The fix was covered by a small evaluator regression
test, and the exact saved weights were reevaluated on MPS. The original failure,
receipt, training source and corrected evaluation source are all preserved.

All 31 tests pass, including RGB-black damage, palette corruption, local-growth
ordering, clean warm-start equivalence, exact replay/teacher/optimizer resume and
native-region evaluation. Ruff lint/format pass. Native 512×512 MP4s were inspected;
their actual 64 fps cadence matches the new teacher, without a speed multiplier.

Estimated cloud function compute: **$0.0408**, including its GPU, two CPU cores
and 8GiB RAM. This excludes image build/startup/transfer and is not an account
billing statement. One function, 150-second training cap, 900-second timeout,
no deployment, volume, retries or background task. [App stopped](cloud-status.json).
The $2/six-development-cycle budget needed only one pilot. Validation ran locally.

The new teacher is derived solely from the original CC0 shader renders. Code and
shader-only weights are MIT. No new artist media, pretrained encoder or external
training asset was introduced. All original accepted artifacts remain unchanged.

The remaining scope is one preset, one training seed, external phase, and designed
state perturbations. Multi-program/text conditioning, control sweeps, measured
audio mappings and paper-level comparisons remain future work. Further artistic
work can explore pattern-specific branching growth and more structural distortions
once these basic spatial recovery controls are useful.
