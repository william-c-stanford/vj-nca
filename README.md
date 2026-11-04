# vj-nca

Train one small, globally conditioned neural cellular automaton on multiple video
loops, then measure whether its motion survives and recovers from disturbances.
The first implementation targets training, reproducible experiments and visual
inspection. It does not yet implement a live VJ application or audio input.

Read [the research and design notes](docs/RESEARCH.md) for the DyNCA, Reasoning-NCA,
neural particle automata and frequency-time diffusion investigation.

Current trained artifact: [mixed-v1](artifacts/mixed-v1/MODEL_CARD.md), with 26,452
parameters. All seven loops pass the stricter motion/recovery check over a
32-second held-out rollout; mean PSNR is 28.76 dB. Richer phase conditioning and
synchronous updates fixed the earlier weak-motion result after widening alone
failed. See the [three-trial autoresearch report](experiments/autoresearch/REPORT.md).
This remains a 32×32 training-loop pilot, with one public clip and six procedural loops.

The subsequent [native 512×512 exploration](experiments/rave-512/REPORT.md) implements
native-pixel patch training and full-canvas evaluation, but has no accepted model.
The detailed Electric Sheep targets were rejected on aesthetic grounds; GPU runs
are stopped. The next proposed direction is
[shader-generated data with program/text/control conditioning](docs/SHADER_CONDITIONING.md).

Selected visual direction: **kinetic line sculptures**, starting with **Aperture**.
See the [512×512 shader preview gallery](artifacts/shader-preview-v0/index.html),
[aesthetic research and learning-difficulty comparison](docs/SHADER_AESTHETICS.md),
and [deferred directions](docs/FUTURE_WORK.md). These original CC0 shader renders
are training targets; they are not a new trained NCA artifact.

Earlier artifacts: [mixed-v0](artifacts/mixed-v0/MODEL_CARD.md),
[six procedural loops](artifacts/procedural-v0/MODEL_CARD.md), and
[initial results](docs/EXPERIMENTS.md).

## Install

Use Python 3.11+ and install FFmpeg for video ingestion/export.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[cloud,dev]'
```

PyTorch 2.8.0, NumPy 2.3.5 and Pillow 12.x were used for the initial checks. The Modal
image pins its dependencies. Use `PYTHONPATH=src python -m vj_nca` instead of `vj-nca`
when running from a source checkout without installing the package.

## Start locally

Render the original shader studies locally with the optional OpenGL dependency:

```bash
python -m pip install -e '.[shaders]'
PYTHONPATH=src python scripts/render_shader_previews.py --stills-only --out runs/shader-stills
PYTHONPATH=src python scripts/render_shader_previews.py --out runs/shader-previews
```

The tested renderer uses the local graphics GPU, with no cloud service or neural
checkpoint. The [preset manifest](datasets/shader-preview-v0.json) declares the
control ranges; outputs retain source, descriptions, normalized controls and
hashes for later conditioning experiments. See the
[preview record](artifacts/shader-preview-v0/README.md) for platform limits and checks.

Create six original, smooth, seamless VJ-style test loops. Each has 32 frames at 32×32,
played at 16 fps. One shared network learns all six with separate conditioning codes.

```bash
vj-nca generate --out data/procedural.npz
vj-nca train --data data/procedural.npz --config configs/pilot.json \
  --out runs/local-pilot --iterations 30 --max-seconds 15
vj-nca evaluate --data data/procedural.npz --checkpoint runs/local-pilot/last.pt \
  --out runs/local-pilot/eval
```

Thirty updates are an integration smoke test, **not** enough training for good videos.
The evaluation deliberately runs much longer than the training backpropagation chunk.

Resume without changing the dataset, model or optimization settings:

```bash
vj-nca train --data data/procedural.npz --config configs/pilot.json \
  --out runs/local-pilot --resume runs/local-pilot/last.pt --iterations 1500
```

## Run a cheap GPU pilot

Authenticate Modal using its own CLI if it is not already configured. This launches
an ephemeral job, not a deployment. It requests one T4, allowing an L4 fallback,
uses no persistent volume, disables retries and stops after the function returns.

```bash
PYTHONPATH=src modal run --env main cloud/modal_train.py \
  --data data/procedural.npz --out runs/modal-pilot \
  --variants baseline,robust,normalized \
  --train-seconds 180 --iterations 1500 --budget-usd 1
```

`baseline` has no damage/noise/switch training. `robust` adds those perturbations.
`normalized` also caps each hidden-channel group's L2 norm at one, locally per cell.
This avoids expanding near-zero states during initialization. All share the
same architecture, initialization seed, phase signal, losses and replay configuration.
`no-phase` is an additional ablation with the periodic signal zeroed.

The wrapper allows at most three variants, 240 training seconds and 5,000 iterations
per variant. A 900-second function timeout bounds this invocation's running compute.
At the prices checked November 3, 2026, its worst-case L4 function compute estimate
is approximately **$0.22** for GPU, one CPU core and 4 GiB RAM. Image-build compute,
startup, scheduling behavior, egress and any account-level charges are separate.
`--budget-usd` checks that estimate; it is not an account-wide hard billing cap.
Read the actual `receipt.json` and Modal billing after a run. Do not run the command
repeatedly while a previous job is queued.

## Use real clips

Prepare the checked, small public color-cycling loop and a mixed seven-loop dataset:

```bash
PYTHONPATH=src python scripts/prepare_public.py
```

Train the selected seven-loop recipe on a short GPU pilot:

```bash
PYTHONPATH=src modal run --env main cloud/modal_train.py \
  --data data/mixed.npz --config configs/synchronous-003.json \
  --out runs/mixed-v1-repeat --variants normalized \
  --train-seconds 240 --iterations 5000 --budget-usd 0.25
```

The preparation script downloads an 8.1 MB source only if missing, verifies its SHA-256, discards audio,
center-crops and downsamples it, and records the source and changes. The public media
is CC BY-SA 3.0; see [attribution](ATTRIBUTION.md) and the
[source manifest](datasets/public-loops.json). The six original procedural loops are CC0.

Import your own **complete cycle** with explicit boundaries and provenance:

```bash
vj-nca import-video --input /path/to/loop.mp4 --name my-loop \
  --start 0 --duration 4 --source-url https://example.com/source \
  --license 'your-license' --size 32 --frames 32 --fps 16 --out data/my-loop.npz
vj-nca merge data/procedural.npz data/my-loop.npz --out data/mixed-own.npz
```

All clips in a dataset share a normalized cycle duration. A four-second source becomes
two seconds with 32 frames at 16 fps. This is intentional for the first experiment,
and the original and retimed durations are recorded. Inspect `seam_ratio`: the last-to-
first transition should resemble an ordinary adjacent transition. A low ratio does
not prove that an arbitrary trimmed clip is a valid loop.

## Inspect the artifacts

- `last.pt`: model, optimizer, replay states with phase/age, RNG state and dataset hash.
- `config.json`, `dataset.json`, `training_report.json`: settings, provenance and resources.
- `train.jsonl`: losses, gradient norm, state overflow and replay age.
- `eval/metrics.json`: per-loop quality, motion, drift, perturbation curves and recovery.
- `eval/comparison.gif`: target, clean rollout and erased-then-recovering rollout columns;
  one row per loop. The erase happens at the beginning after the warmup cycles.
- `eval/loop-XX.mp4`: the final cycle for each numeric loop ID in `dataset.json` order.
- `receipt.json` from Modal: source/data hashes, GPU, elapsed time and cost estimate.
- `source/` from Modal: the exact code used in that run. Source/data, each completed
  variant, and the final receipt download as separate bundles.

Export a checkpoint without optimizer/replay/RNG objects for sharing:

```bash
vj-nca export --checkpoint runs/modal-pilot/normalized/last.pt --out artifacts/my-model
```

The small `model.pt` artifact loads with `torch.load(..., weights_only=True)`.
Keep its `model.json`, source version, dataset provenance and evaluation alongside it.

Metrics use unclipped outputs in [0,1] RGB units; display exports are clipped. A static
or inaccurate model is ineligible for a recovery success. Schema 2 requires signed
motion cosine ≥0.7 and gain 0.5–1.5, plus reconstruction better than half the static
mean-frame baseline. `passing_loops` additionally requires every tested perturbation
to recover within one second. `--phase-frame` selects another perturbation phase.
The current evaluator accepts exported `model.pt` files as well as resumable checkpoints.
Evaluation uses new states and corruption noise for the training clips (also fresh
firing masks for stochastic models); it does not show generalization to unseen clips.
Conditioning embeddings alone do not encode new videos without optimization.

## Verify

```bash
python -m unittest discover -s tests -v
ruff check src tests cloud scripts
ruff format --check src tests cloud scripts
```

Tests cover periodic phase, cyclic interpolation, conditioning gradients, firing-mask
semantics, local hidden normalization, replay metadata, exact CPU resume and recovery
criteria. A small training run plus long-rollout evaluation exercises the full pipeline.

## Research direction

Start with the shared phase-conditioned raster model. Measure quality, conditioning
capacity and recovery before scaling resolution or adding expensive losses. Particle
automata are a promising separate branch for sparse particle footage. Full diffusion
and Fourier-space generation require a different objective and sampling process;
the optional spectrum loss here is not a diffusion implementation.

The paper opportunity is the dynamic-loop benchmark, training method and evidence,
not a claim that conditional NCA or regeneration is new. Required comparisons and
limitations are listed in [docs/RESEARCH.md](docs/RESEARCH.md).
