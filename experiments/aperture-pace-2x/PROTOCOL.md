# Correct the pace to 2× the original preview

2026-11-04. The user clarified that the desired speed is about twice the original
preview. The original Aperture cycle was eight seconds. The new teacher is
therefore **four seconds**, one contour advance per cycle. The previous reactive
experiment's quarter-second cycle was excessive and is superseded for playback.

Retain all 64 exact 512×512 source frames, at 16 fps. Use cyclic linear interpolation
and four NCA updates per source frame: **64 updates per second, 256 per cycle**.
This separates the target's motion speed from local repair kinetics. The integrity
head, corruption types and growth curriculum retain their previous physical clock.
Warm-start the complete 144,629-parameter repair model; reset optimization and replay
for the changed teacher. This is a new training dataset, not an altered export rate.

One initial 1500-update, 150-second Modal pilot. At most two development runs and
$0.70 estimated function compute, using the existing 900-second timeout/resource
cap. No persistent deployment. Retain negative results and stop the GPU after use.

Keep the existing clean-quality and progressive-repair gates. Evaluate seed 4041,
phase frame 0, eight measured seconds after four cycles of warmup. Recovery must
remain within tolerance for one entire four-second cycle (256 NCA updates).
Compare target and student at the actual 64 Hz interpolation/update cadence.

After selecting weights, freeze them and validate seed 9091 at source phase frame
16, with the default blackout circle. Inspect a native-resolution video with two
complete cycles; confirm its metadata and elapsed time. No tuning on validation.
Record observed quality and repair limitations honestly even if some original
repair gates do not transfer to the slower motion.
