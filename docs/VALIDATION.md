# Validation record

## Scope

Original code in `src/vj_nca`, the Modal pilot wrapper, public-data preparation,
configs and tests. Work is contained in the initially empty `vj-nca` repository on
`codex/conditioned-vj-training`.

## Checks

- Sixteen unit/integration tests pass using PyTorch 2.8.0 on CPU. They cover periodic
  conditioning, cyclic interpolation across a seam, conditioning gradients, firing
  masks, local hidden bounds, finite gradients from zero, replay identity/phase/age,
  exact resume, dataset/config mismatch rejection, recovery eligibility, epoch-dated
  source archives, artifact path containment and average GIF playback timing.
- New motion-score tests distinguish target motion from freezing, reversed motion
  and unrelated flicker with exactly the same motion energy. Released tensor-only
  weights and full checkpoints reproduce identical evaluation; phase-offset probes
  match shifted targets and reject invalid phases.
- A 30-update 32×32 CPU smoke run completed, followed by a much longer rollout. The
  model's poor extrapolation was detected rather than hidden by clipping or a static
  output being counted as recovery.
- FFmpeg decoded a real 5.12-second source cycle into 32 frames. The decoded seam
  error was comparable to an ordinary temporal transition (ratio approximately 0.83).
- The tensor-only export loaded with `weights_only=True` and matched a full
  checkpoint's forward result exactly under the same inputs and firing mask.
- The evaluation reuse refactor produced a byte-for-byte identical metrics file on
  the fixed CPU smoke checkpoint.
- Ruff lint and formatting checks pass. No static type checker is configured.

See [EXPERIMENTS.md](EXPERIMENTS.md) for actual GPU results and known model limitations.
The [autoresearch follow-up](../experiments/autoresearch/REPORT.md) adds three actual
5,000-update GPU trials, a frozen-candidate 32-second held-out comparison from a
fresh worktree, and corruption tests at three additional quarter-cycle phases.
Every new trial preserves its source, portable weights, config, logs and curves.

## Review

Code review: skipped (ce-code-review unavailable). Its scope helper could not resolve
a valid base commit in this unborn repository (`invalid base endpoint`); there was
also no standalone harness-native local review tool. A manual source review followed,
using the correctness, reliability, testing, reuse and efficiency review rubrics. No
independent reviewer or cross-model review is claimed. Project instructions require
review work to remain sequential in the main chat.

### Actionable findings addressed

- Unit-length projection amplified gradients around zero-initialized hidden groups.
  The experiment failed with non-finite gradients. Replaced it with local projection
  onto the unit ball and added zero-state gradient coverage.
- The initial cloud wrapper lost completed results when a later variant failed.
  Variant exceptions now produce explicit failure artifacts; earlier outputs return
  with the receipt, and the CLI reports a failure after saving diagnostics.
- ZIP packaging rejected Modal's epoch-dated mounted source files after training.
  Portable timestamps now round-trip in a regression test. The worker streams source,
  each variant, and the final receipt separately. A real two-update GPU test verified
  successful download and extraction before the final training validations.
- Per-loop evaluation recomputed errors for the entire batch repeatedly. Compute each
  batch's error tensors once; the fixed-checkpoint equivalence check passed.
- GIF timestamps have 10 ms precision. Rounding every frame to 60 ms sped up 16 fps
  previews; cumulative rounding now preserves average playback rate. Retained MP4
  previews were explicitly retimed to 16 fps. Numerical metrics were unaffected.

### Coverage limits

GPU scheduling, hard process termination and provider-level failures are not mocked
by unit tests. Actual small GPU invocations exercise the cloud path. A hard container
failure or overall function timeout can still prevent the artifact bundle returning;
there is no persistent volume in this low-cost pilot. Already downloaded variants
remain available if a later stage fails. Numerical failure within a variant is
captured. No live audio, latency, unseen-clip generalization or
publication-quality multi-seed result has been validated.

### Verdict

Suitable for reproducible pilot experiments, with model quality and scientific claims
limited to the measured cases in the experiment report. Larger training is not
automatically authorized or launched by the wrapper.

## Autoresearch follow-up review

Manually reviewed the current diff from `c69dcac` in the main chat, including
motion-score reductions/units, schema compatibility, shifted-phase target/probe
alignment, export/checkpoint loading equivalence, immutable trial capture, cost
receipts and reported claims. The new helper scripts only read trusted local
training artifacts. No independent reviewer or cross-model review is claimed.
The initial unavailable-review-tool record above describes the original commit.

All sixteen tests and Ruff checks pass. The selected portable artifact was the
actual input to the held-out evaluation, and its preview has 129 frames at exactly
16 fps (8.0625 seconds). All three new Modal apps are stopped. The selected model
passes the fixed seven-loop pilot objective; publication-level multi-seed evidence
and live audio/inference validation remain outside these results.
