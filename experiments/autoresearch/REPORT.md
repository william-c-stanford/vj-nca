# Motion and recovery autoresearch — 2026-11-03

The selected model is [mixed-v1](../../artifacts/mixed-v1/MODEL_CARD.md), on branch
`codex/conditioned-vj-training`. It passes all seven loops on the fixed development
and held-out motion/recovery gates. The original model passes four on development
and three on the longer held-out evaluation under the same stricter gates.

Three short T4 training trials were sufficient. Increasing parameters alone did
not solve the issue. Increasing phase bandwidth helped, then synchronous updates
produced the largest improvement. No larger training job was launched.

## Fixed experiment

See [PROTOCOL.md](PROTOCOL.md) for the objective, data fingerprint and gates,
declared before the first trial. Six procedural loops and one attributed public
color-cycling clip are reconstructed at 32×32, 32 frames, 16 fps. All runs use one
shared rule with learned loop embeddings, seed 42, 5,000 optimizer updates, replay,
damage/noise/state-swap training and local hidden unit-ball normalization. No new
loss, teacher-forced target input, diffusion process or particle model was needed.

Development: seed 4041, three burn-in cycles, four measured cycles. Pass requires
per-loop MSE below half the temporal-mean baseline, signed velocity cosine ≥0.7,
motion gain 0.5–1.5, and all five perturbations recovering within one second.
These gates were fixed before the candidates returned. Historical schema-1
eligibility counted energy, which could mistake flicker for motion; its counts
are not comparable. All results below use schema 2.

| Model / intervention | Parameters | Phase harmonics | Fire rate | Dev pass | Dev mean PSNR | Dev MSE | Training seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original mixed-v0 | 11,220 | 3 | 0.5 | 4/7 | 24.70 dB | 0.005123 | 189.1 |
| [Capacity only](trials/capacity-001/config.json) | 25,172 | 3 | 0.5 | 3/7 | 24.62 dB | 0.004863 | 198.3 |
| [Richer clock](trials/phase-002/config.json) | 26,452 | 8 | 0.5 | 4/7 | 25.71 dB | 0.003561 | 186.8 |
| [Synchronous updates](trials/synchronous-003/config.json), selected | 26,452 | 8 | 1.0 | **7/7** | **28.75 dB** | **0.001790** | 190.0 |

The capacity trial changes channels 19→35 and hidden width 96→128, preserving the
other recipe settings. The next two trials each change one setting from their
predecessor. All finish the requested update count before their 240-second caps.
Maximum allocated CUDA memory for the selected training run was 210,816,000 bytes.

## How the hypothesis changed

The larger model improves the hard clips modestly but leaves lattice motion gain
at 0.200 and public plasma at 0.317. Capacity alone is insufficient in this budget.

The [target spectrum diagnostic](target-phase-spectrum.json) shows that three
temporal harmonics cover 50% of the lattice's non-DC energy, versus 99% with eight.
Eight explicit clock harmonics improve lattice motion gain to 0.352. This is a
representation diagnostic, not an expressivity bound: nonlinear recurrent models
can generate harmonics absent from their inputs.

Changing firing probability 0.5→1 improves lattice gain to 0.810 and public gain to
0.566 without adding parameters. It reduces random RGB jitter and doubles the
expected number of active cell updates per video frame. **This experiment does
not isolate jitter reduction from effective update rate.** Damage/noise/swap
training remains enabled, and recovery survives the change.

We have not tested the original 11,220-parameter model with both improved settings,
so these results do not establish that the final parameter increase is necessary.
They establish that widening alone failed and that this modest model plus the
improved recipe meets the pilot objective.

## Held-out admission

After selecting the 7/7 development candidate, freeze its weights and evaluate
both original and candidate in a fresh detached worktree. The
[manifest](heldout-manifest.json) records source, weight and data hashes before
the evaluation. The [exact evaluation source](heldout-evaluation-source.zip)
matches those hashes. Seed 9091 supplies fresh corruption noise and stochastic masks;
16 measured cycles span **32 seconds** after the six-second burn-in. No gradients
or model changes are applied. The temporary worktree was removed afterwards.

| Fixed held-out evaluator | Original | Selected |
|---|---:|---:|
| Passing loops | 3/7 | **7/7** |
| Mean per-loop PSNR | 24.71 dB | **28.76 dB** |
| Mean MSE | 0.005117 | **0.001788** |
| Lattice motion cosine / gain | 0.277 / 0.084 | **0.917 / 0.810** |
| Public plasma motion cosine / gain | 0.342 / 0.162 | **0.737 / 0.566** |

The selected model reduces held-out MSE by **65.1%** and raises mean per-loop PSNR
by **4.04 dB**. All 35 loop/perturbation cases recover. Recovery onset ranges from
0.0625 to 0.4375 seconds, followed by three more qualifying frames to confirm the
four-frame sustained criterion. These are simulated video seconds, not measured
real-time inference latency. Recovery conditions use unclipped RGB errors.

| Loop | Held-out PSNR | Motion cosine | Gain | Slowest recovery onset |
|---|---:|---:|---:|---:|
| drifting-bars | 30.37 dB | 0.931 | 0.969 | 0.2500 s |
| expanding-rings | 26.72 dB | 0.904 | 0.948 | 0.3750 s |
| orbit | 33.44 dB | 0.958 | 0.954 | 0.1250 s |
| rotating-spokes | 29.65 dB | 0.849 | 0.947 | 0.1875 s |
| plasma | 31.87 dB | 0.870 | 0.977 | 0.1250 s |
| lattice | 26.08 dB | 0.917 | 0.810 | 0.3750 s |
| commons-plasma | 23.17 dB | 0.737 | 0.566 | 0.4375 s |

The selected clean cycle drift is below 1e-12 MSE. With synchronous firing, its
clean trajectory is deterministic: a new seed holds out corruption noise, not
clean firing masks. This is a conditional driven periodic trajectory, not an
autonomous oscillator. A one-update reset probe has MSE 0.03025, versus 0.001788
with recurrent state; it is a dependency diagnostic, not a trained MLP baseline.

Post-selection stress tests repeat all perturbations at phase frames 8, 16 and
24, with seeds 5059, 5067 and 5075. Every phase passes 7/7, with maximum recovery
onset 0.375 seconds. Together with phase zero this covers **140 tested cases**.
These checks did not guide model selection. Full curves are retained in
`trials/synchronous-003/metrics-phase-*.json`.

## Artifacts and cost

The [hypothesis tree](.arbor/tree.json) records three executed candidates, the
capacity-only branch pruned with its evidence, and promotion of n4 after its test
score improves 3→7. The initial material is retained as n0. The search ends after
three of twelve available cycles because its fixed success criterion is met.

Each trial directory contains portable tensor-only weights, model/config/data
metadata, exact training source, training logs, receipt, submission and complete
development recovery curves. The selected trial additionally includes held-out
and phase-stress metrics. Full resumable optimizer/replay checkpoints remain in
the ignored local `runs/` directories. Code is MIT; mixed-dataset trial weights
and previews use CC BY-SA 3.0, with [dataset attribution](../../ATTRIBUTION.md).

The three function resource estimates total **$0.11307**. The account's
[metered snapshot](billing-final.json) increased from $0.21037203 to $0.33103976,
an incremental **$0.12066773**, including startup/transfer differences. Billed
usage is $0 after credits/allowances. This is a usage snapshot, not an invoice.
The $2 batch budget was not approached. [All apps stopped](modal-apps-final.json);
no endpoint, volume, schedule or other continuing GPU service was created.

## Limits and next useful evidence

This is a successful **seven-training-loop, 32×32 pilot**. The public clip still
has weaker motion amplitude than its target. Only one training seed, one public
clip and one resolution were tested. No unseen-video capability, audio mapping,
tempo change, real-time benchmark or compression advantage is established.

Before a paper or larger dataset run: test the original parameter count with the
improved recipe; compare synchronous firing against more steps per frame at 0.5
firing; repeat training seeds; add licensed nonprocedural VJ loops and higher
resolutions; and compare against phase-conditioned neural fields and per-clip
NCA baselines. Keep the held-out trajectory gate separate from future search.
