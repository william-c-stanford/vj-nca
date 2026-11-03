# Seven-loop motion and recovery search

Started 2026-11-03 from `c69dcac` and `artifacts/mixed-v0`. The user requested
a slightly larger model first, followed by autonomous research if it failed.

The material is the shared phase-conditioned NCA and the existing seven-loop,
32×32, 32-frame dataset (fingerprint
`2188365ed68f53871b7652e09aaf84abe32c7d5d7a3cd33f92acbfbbf9fe7b30`).
All experiments reconstruct these training clips. A different rollout seed is
a trajectory holdout, **not** an unseen-video or unseen-condition test.

The objective is correct moving patterns and rapid recovery for every loop.
Development evaluation uses seed **4041**, three burn-in cycles and four measured
cycles. The held-out gate uses seed **9091** and sixteen measured cycles, after
development selection. The same evaluator is used for the initial and final
model. The scalar score is the number of loops satisfying all these fixed gates:

- Clean MSE below half the static temporal-mean baseline.
- Cosine similarity of signed pixel velocities at least 0.7.
- Motion gain (predicted/target velocity projection) between 0.5 and 1.5.
- All five perturbations (erase, noise, brightness/latent beat, full reset, loop
  switch) recover within one second, with the existing four-frame sustained
  paired/absolute agreement rule.

Cosine distinguishes target motion from flicker; gain distinguishes correct
motion amplitude from a near-static approximation. PSNR and all raw errors are
also retained. Historical schema-1 eligibility used only motion energy and is
not directly comparable with this stricter schema-2 score. No thresholds are
tuned after seeing candidate results. Successful completion requires 7/7 on
development and on the held-out rollout.

Start with 12 coordinator cycles and a **$2 incremental GPU resource budget**.
Each training pilot is initially at most 5,000 updates / 240 seconds on one
ephemeral T4 (L4 fallback), with a 900-second function timeout. Timeouts bound
function resource exposure; startup, image builds and transfer are additional.
Check billing and stop all apps at the end. Do not launch a large training job.

The Arbor tree in `.arbor/` records hypotheses, evidence and selection. Per the
workspace instructions, execution is sequential in the main agent. Each trial
has a unique run directory, immutable config and exact uploaded source snapshot;
the original model remains available throughout the search.

Reproduce the fixed evaluator with:

```sh
PYTHONPATH=src python scripts/score_motion.py \
  --checkpoint runs/TRIAL/normalized/last.pt \
  --out runs/TRIAL/dev --seed 4041 --cycles 4
```

Use `--seed 9091 --cycles 16` only for the final gate. The script reports compact
scores and saves the complete metrics and recovery curves.
