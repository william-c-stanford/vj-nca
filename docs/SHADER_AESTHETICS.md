# Choosing the first shader family

2026-11-03. Selected direction: **kinetic line sculptures**, beginning with
**Aperture**. Black space, ivory luminous contours, one restrained acid-green
accent, a stable composition and continuous motion with a gentle change of pace.
The [original previews](../artifacts/shader-preview-v0/index.html) are 512×512,
24 fps, eight seconds. They are shader targets, **not trained NCA outputs**.

The user asked for aesthetic research beyond prominent artists and a single blog
post. I followed festival/community documentation into artist portfolios, specific
works and a technical interview. The scope includes working VJs and smaller
audiovisual practices alongside some established artists. This is a curatorial
choice for this experiment, not a universal definition of good taste or an
exhaustive survey of rave visuals. Historical works are useful design evidence;
they do not establish a 2026 trend.

## Evidence and what it changes

| Artist / primary source read | Documented approach | Consequence for this project |
|---|---|---|
| Cadie Desbiens-Desmeules / Push 1 stop & Wiklow, [Membrane](https://desbiens-desmeules.com/membrane/) | Custom generative performance, projected forms in haze, rehearsed scenes with room for improvisation. | Light should describe a form in black space. A few legible gestures can sustain a scene; scene structure matters as much as detail. |
| Julien Bayle, [STRUCTURE.INSTALLATION](https://julienbayle.net/works/structure-installation/) | Repetition, architectural abstraction, and context changes structure the work; it includes intense flashes. | Use repetition and a strong spatial frame. We select continuous contour motion for the first target, without adopting the flashing technique. |
| Florence To, [ALUCIIN](https://florence-to.com/ALUCIIN) | Evolving repetitive abstract phrases explore movement and time. | A recognizable motif should evolve over a phrase, rather than switching to unrelated effects each frame. |
| Nicolas Lavoie / DATA DADA, [MUTEK artist profile](https://montreal.mutek.org/en/artists/data-dada) | Visual grammar, typography, shape, texture and opposition are part of live collage practice. | Deliberate light/dark hierarchy and a distinct accent should do compositional work. Random decoration is not a substitute for a visual grammar. |
| VJ BunBun, [MUTEK's VJ curation and interview](https://montreal.mutek.org/en/news/vj-mutek-2022) | Dense color and texture, context-sensitive improvisation, and curating complementary local VJ styles. The article links practitioners including Kitana, Line Katcho and Susy.technology. | Monochrome is our learning constraint, not the only credible club aesthetic. Keep colorful, textured families as a later expansion. |
| Olivia Jack, [Hydra](https://ojack.xyz/work/hydra/) | Modular visual functions, feedback, networked collaboration and chaos in live performance. | Keep shader controls composable and record exact programs. Feedback is artistically valuable, but it adds state/history complexity, so it is deferred. |
| Yannick Jacquet & Laurent Delforge, [Remote Memories](https://www.yannickjacquet.net/works/remote-memories) | Layered textures with subtle motion; sound and image explicitly evolve independently. | Musicality need not mean a brightness flash on every beat. Preserve a visual phrase and allow later audio perturbations to interact with it. |
| Nicolas Sassoon & Rick Silva, [SIGNALS / RIPPLES](https://nicolassassoon.com/SIGNALS_works.html) | Looping altered landscapes and interleaved digital patterning. The RIPPLES sample visibly uses dense, fine repeated marks. | Optical interference is a useful later vocabulary. Dense pixel-scale patterning is a poor first learning target. |
| Marcel Weber / MFO, [Nervous System 2020](https://mfoptik.de/n2020/) | A staged spatial installation/performance with translucent chambers, projected imagery and no single audience focus. | Venue, light and viewing position shape the experience. A flat preview cannot reproduce the installation; design a legible flat composition for this stage. |
| Jemma Woolmore, [Poetically Based Rendering](https://jemmawoolmore.com/project/poetically-based-rendering/) | Site research, material textures, scanning and speculative digital environments. | Material specificity is a rich alternative to generic effects. It belongs in future organic/material families after a controllable geometric baseline. |

Further context read: [Bayle's technical interview on the.collapse](https://derivative.ca/community-post/julien-bayles-thecollapse/64062)
describes a different, materially grounded workflow using failed Polaroids and
granular scanning; it would be inaccurate to reduce his practice to line art.
[Florence To's EOAN](https://florence-to.com/EOAN) documents participatory sound/light
relationships, while [INTRVL](https://florence-to.com/INTRVL-1) provides a route into
an independent performance community. EOAN's speculative physical explanations
are not adopted as scientific evidence here. The
[SHAPE artist profile](https://shapeplatform.eu/artist/julien-bayle-2/) and
[LPM's Stanislav Glazov profile](https://liveperformersmeeting.net/editions/2026-rome/artists/glazov)
provide additional performance/programming context, not a quantitative aesthetic
ranking.

I inspected primary imagery from Membrane, STRUCTURE and Remote Memories in the
browser, plus the animated RIPPLES sample. I sampled the embedded Membrane player;
this is not a complete frame-by-frame performance analysis. Remote Memories'
embedded video was unavailable, so its timing claims above rely on the artist's
description. None of these artists' media, code or textures is used as training
data or bundled with the previews. These are references for general design choices,
not reconstructions or artist-style imitation targets.

## The resulting aesthetic brief

1. **A form with a clear silhouette.** Large dark areas separate the luminous
   structure from its surroundings. The viewer should read it at a distance.
2. **A small visual vocabulary.** Repeated contours establish continuity; a limited
   accent introduces direction and tension. The particular ivory/green palette is
   our choice, not a palette prescribed by the references.
3. **Continuous, meaningful motion.** A contour advances, a sheet folds, or two
   fields interfere. The motion changes the form instead of merely changing its
   exposure. Use a small change of speed across the phrase.
4. **Enough restraint to vary intensity later.** Preserve room to increase density,
   deformation and contrast with future musical controls. Avoid placing every
   effect at its maximum in every target.
5. **Complexity introduced deliberately.** Start with broad, softly antialiased
   edges and controlled motion; add finer interference, texture and history once
   the NCA actually learns the first family.

These are design inferences from the evidence and the training objective. The
references also provide counterexamples to simplistic claims that all strong VJ
work must be minimal, monochrome or synchronized to every beat.

## Why Aperture goes first

Three original studies were rendered and visually compared. The first pass had
thinner lines; the second made the line bodies wider. A final pass introduced an
explicit soft edge, keeping the graphic silhouette while reducing high-frequency
motion at its boundaries. The three studies share the chosen visual direction:

- **Aperture:** rounded architectural contours advance through a dark opening.
- **Fold:** a ribbed sheet undergoes a travelling bend and squeeze.
- **Interference:** two line fields produce moving regions of reinforcement.

The [probe](../artifacts/shader-preview-v0/learnability-probes.json) renders 64
uniform phases directly at 512×512 as float pixels. It measures temporal Fourier
energy, signed-velocity spatial retention and coarse temporal rank. These are
diagnostics, not a theorem about neural expressivity or a trained-model result.

| Study | Dynamic RGB energy in first 16 harmonics | Signed-motion energy in first 16 harmonics | Components for 90% temporal energy at 32px |
|---|---:|---:|---:|
| **Aperture** | **99.95%** | **97.46%** | **3** |
| Fold | 73.46% | 39.35% | 20 |
| Interference | 88.23% | 55.33% | 14 |

Aperture offers the simplest temporal structure while preserving visible geometric
motion. That matters because inadequate phase representation was a major failure
mode in the earlier experiments. It is not easiest under every metric: its sharp
moving contours still require spatial accuracy. A linear 128px spatial control
retains only 45.7% of its projected native motion, versus 72.6% at 256px. Do not
assume that training a tiny raster and enlarging it will solve native fidelity.

The final render checks verify actual phase-zero/one continuity, numerical
derivative continuity, non-finite rejection and ordinary last-to-first sampled
transitions. The endpoint is not duplicated. Exact per-study results, controls,
source hashes, renderer versions and source snapshots accompany the videos.

## Scope of the next training experiment

Begin with **Aperture parameter variations**, fixed composition and palette, then
vary density, deformation, width and softness. Keep a small sweep and hold out
parameter combinations. Keep the existing reconstruction, motion, long-rollout
and recovery checks; add targets while controls change.

Within a single shader family the code and base description are constant, so a
code embedding cannot identify different uniform settings. Numerical controls
must accompany it. First establish a parameter-conditioned baseline; code/text
conditioning earns a separate multi-family ablation later. The renderer stores
the source, descriptions and normalized controls, but no semantic encoder has
been selected, run or trained in this preview step.

The other visual directions and research extensions are recorded in
[FUTURE_WORK.md](FUTURE_WORK.md). No paid cloud training was launched for this
aesthetic/preview task. Local rendering used the Mac's graphics GPU; this is not
an NCA inference-speed measurement.
