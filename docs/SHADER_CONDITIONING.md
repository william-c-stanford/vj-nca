# Program-conditioned VJ dynamics — roadmap

2026-11-03. The user rejected the Electric Sheep source aesthetic and suggested
synthetic shader/TouchDesigner loops, conditioned by source-code and/or description
embeddings. This changes the dataset direction; it is not evidence that the failed
native-512 models have been solved. The current Electric Sheep GPU run was stopped.
The later [Aperture training experiment](../experiments/aperture-fast/REPORT.md)
now passes native 512×512 motion/recovery checks with measured local throughput of 136 fps
including RGB readback. This is a single-preset learned-ID baseline; semantic and
numeric-control conditioning remain untrained.

The subsequent [reactive Aperture model](../artifacts/aperture-reactive-v1/MODEL_CARD.md)
trains against a much faster teacher (four contour advances/second at 64 fps) and
adds true blackouts, chromatic glitches and supervised local rebuilding. This
separates motion pace from repair duration; it does not add semantic conditioning.

The user subsequently delegated visual selection. The chosen direction is
**kinetic line sculptures**, starting with **Aperture**, based on
[the deeper aesthetic investigation and native motion probes](SHADER_AESTHETICS.md).
[512×512 shader previews](../artifacts/shader-preview-v0/index.html), exact source,
descriptions and normalized controls are now rendered and recorded. Other visual
directions are filed in [FUTURE_WORK.md](FUTURE_WORK.md). The original eight-second
preview is now trained as a two-second cycle, with faster tempo exports available.
No semantic encoder, parameter sweep or unseen-program result is claimed.

## What this would let us test

Distill families of controllable visual programs into one shared NCA, with an
explorable conditioning space and local perturbation recovery. Render targets at
512×512, starting with a controlled sweep within the selected Aperture family.
Broader multi-family conditioning follows a successful parameter-conditioned
baseline. Liquid/chrome, complex tunnels and dense moiré are deferred.

For each sample retain the exact shader source, program version/hash, numerical
uniforms, random seed, loop period, description, renderer version and license.
Author original shaders for reproducible, redistributable data. Do not assume
arbitrary public Shadertoy code has a common reuse license.

Prefer GLSL for the first controlled dataset: source and parameters are easy to
record and batch-render. TouchDesigner can be an authoring source, provided its
node graph, expressions, assets and exposed controls are captured reproducibly.

## Conditioning hypothesis

Proposed global conditioning:

`z = projector(code_embedding, description_embedding, normalized_shader_controls)`

Supply the existing periodic phase signal separately. The shared NCA update rule
receives z at every cell; eliminate the requirement for an arbitrary learned ID
per rendered clip. Initially freeze the semantic encoders, cache their embeddings,
and train only the projector and NCA. Encoder/model choice is not settled.

Numerical controls are essential: many videos share exactly the same program and
comments while using different parameters. A source-code embedding alone cannot
distinguish those renders. Text/code similarity also need not imply similar
rendered motion. Normalizing comments and names, and testing paraphrases, helps
detect superficial shortcuts but does not establish visual alignment.

Known controls provide a supervised continuous space within each shader family.
An optional later extension aligns program/text embeddings with rendered-video
features using matched and mismatched pairs, making motion-aware similarity an
explicit learning target. This is a research hypothesis, not demonstrated behavior.

## Small falsifiable experiment

1. Completed: render a small 512×512 style/contact preview, research the aesthetic,
   and select the first family before bulk data generation or paid training.
   Subsequently completed: train one Aperture preset, check carried-state dependence,
   pass a frozen 32-second motion/recovery test and benchmark native inference speed.
2. Generate a modest parameter sweep within Aperture, then expand families. Make all time
   dependence genuinely periodic; check both image and temporal-derivative seams.
   Start with simple geometry, then add distortion/detail as a curriculum.
3. Compare the same NCA capacity with learned IDs, parameters only, code/text only,
   and combined conditioning. Keep data, phase, training budget and recovery gates
   fixed. Structured conditioning should earn its complexity against parameters.
4. Hold out parameter combinations and whole trajectories, not only corruption
   seeds. Later hold out entire shader programs/families for a separate, harder
   generalization claim. Text prompts alone do not establish zero-shot generation.
5. Train on changing controls as well as fixed clips. Test interpolated controls,
   abrupt switches, loops of controls and state corruption. Within-family parameter
   paths have exact shader targets. Cross-family interpolation needs separate
   plausibility/motion/recovery evaluation; it has no automatic ground-truth blend.

Measure image fidelity, signed motion fidelity, long-rollout stability, seam
continuity and recovery while controls change. Retain native-512 previews for
inspection. Sharp frozen textures must not count as motion or recovery success.
Some procedural coordinate warps can supply analytic motion; arbitrary shader
brightness changes do not imply a unique optical-flow ground truth.

Direct shader rendering is a required baseline: it is already a natural way to
make controllable visuals. The NCA must demonstrate added value through persistent
local state, spatial perturbations, recovery and transitions between dynamics.
Train/evaluate state dependence so a memoryless phase-conditioned renderer cannot
silently stand in for the intended recurrent behavior.

## Research grounding and limits

- [Procedural Image Programs for Representation Learning / Shaders-21k](https://mbaradad.github.io/shaders21k/)
  (NeurIPS 2022): 21,000 small OpenGL programs for synthetic-image representation
  learning. Supports scalable program-generated data; not evidence for conditional
  NCA video learning or a useful code-to-motion latent space.
- [Multi-texture synthesis through signal responsive neural cellular automata](https://pmc.ncbi.nlm.nih.gov/articles/PMC12623773/)
  (2025): encoded conditioning channels select textures and permit interpolation.
  Establishes relevant prior conditional-NCA behavior, not the proposed
  shader-code-conditioned dynamic-loop result.
- [Neural Cellular Automata: From Cells to Pixels](https://cells2pixels.github.io/)
  (SIGGRAPH 2026): coarse recurrent cells plus a local implicit appearance decoder
  produce high-resolution outputs. Relevant alternative to one recurrent cell per
  output pixel; any adoption must be described as coarse-state/high-resolution
  decoding, distinct from the existing native-512 recurrent-state experiments.

A possible contribution is a reproducible program/video/control dataset and a
behaviorally aligned, perturbation-recovering conditional NCA. Synthetic shader
datasets and conditional NCA textures already exist; novelty requires a closer
literature check and comparative experiments. No paper or zero-shot claim is made.
