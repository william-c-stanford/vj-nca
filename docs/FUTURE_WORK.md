# Deferred visual directions and experiments

2026-11-03. Active direction: **kinetic line sculptures**, starting with Aperture.
See [the aesthetic decision](SHADER_AESTHETICS.md). This list preserves alternatives;
none is queued for a paid training run.

The [first Aperture preset now has a trained native 512×512 baseline](../artifacts/aperture-fast-v1/MODEL_CARD.md),
with passing motion/recovery checks and measured 136 fps inference plus RGB readback
on M4 Max. Structured multi-preset conditioning is the next training question;
the broader visual directions below remain deferred.

| Direction | Why retain it | Why defer it / return condition |
|---|---|---|
| Fold and controlled interference | Original previews already exist in [the gallery](../artifacts/shader-preview-v0/index.html); add bending, squeezing and interactions within the same restrained visual language. | Much higher measured temporal rank than Aperture. Add after parameter-conditioned Aperture learns motion and recovery. |
| Dense optical moiré and pixel interference | [Sassoon & Silva's SIGNALS](https://nicolassassoon.com/SIGNALS_works.html) is a useful reference for sustained patterned motion. | Fine moving marks demand spatial fidelity. Return after a high-resolution decoder or native model passes the simpler contours. |
| Liquid chrome and reflective surfaces | Material identity, highlights and flow could make strong club imagery. [Woolmore's material/site-based process](https://jemmawoolmore.com/project/poetically-based-rendering/) is an alternative to arbitrary generic effects, not a chrome reference. | Changing reflections, occlusions and shading entangle appearance with motion. Start with one periodic height field and fixed lighting once geometry works. |
| Psychedelic 3D tunnels and fractal spaces | Useful future high-intensity visual vocabulary. | The rejected Electric Sheep trial already shows the cost of beginning with complex fractal motion. Return with a difficulty curriculum and explicit spatial/motion checks. The simple 2D Aperture contours are not a claim to have solved this branch. |
| Color-rich collage and textured imagery | [BunBun's MUTEK interview](https://montreal.mutek.org/en/news/vj-mutek-2022) documents a strong alternative to minimal monochrome. | Adds palette, material and compositing factors at once. Introduce one factor at a time after the geometric baseline. |
| Feedback and live-coded visual systems | [Olivia Jack's Hydra](https://ojack.xyz/work/hydra/) supplies a natural program-composition model and history-dependent visuals. | Feedback adds hidden history and possible nonperiodicity. Use bounded, recorded initial states and explicitly measured attractors. |
| Sparse particles | Potentially a good fit for neural particle automata and local audio perturbations. | Keep separate from dense raster studies. Return with a particle target dataset, neighborhood method and particle-specific motion metrics. |
| Physical material / photographic textures | [Bayle's the.collapse interview](https://derivative.ca/community-post/julien-bayles-thecollapse/64062) demonstrates materially grounded abstraction. | Requires newly authored/cleared material sources and more appearance capacity. Do not reuse reference-artist assets as training data. |
| Typography | [DATA DADA's visual grammar](https://montreal.mutek.org/en/artists/data-dada) makes this a relevant later visual vocabulary. | Letter consistency creates a different accuracy requirement. Use owned fonts and a separate readability check. |

## Training and conditioning extensions

- **Code/text-conditioned multi-family NCAs:** retain code, comments, descriptions
  and controls now. Compare learned IDs, parameters only, semantic embeddings only
  and combined signals on a controlled shared dataset. Freeze/cache encoders first.
  Hold out parameter combinations, then whole programs/families; do not confuse a
  new corruption seed with unseen-program generalization.
- **Behaviorally aligned embeddings:** align program/text codes with rendered-video
  motion if raw code similarity does not organize useful transitions. Test comment
  paraphrases, renamed variables and identical code with different parameters.
- **Transitions while recovering:** train smooth control paths and abrupt changes,
  then perturb mutable state during those transitions. Within-family shader control
  paths have targets; cross-family interpolation needs a separate evaluation.
- **Coarse recurrent state with local high-resolution decoding:** investigate
  [Cells to Pixels](https://cells2pixels.github.io/) after the first baseline. State
  grid size and rendered output resolution must be reported separately.
- **TouchDesigner authoring:** support canonical graph/expressions/assets and exposed
  control capture once the GLSL dataset contract is useful. Avoid making an opaque
  project file the sole conditioning/provenance artifact.
- **Frequency-time diffusion and neural particle automata:** retain the separate
  branches in [RESEARCH.md](RESEARCH.md); neither is needed to render these targets.
- **Audio:** map measured musical features to controls and spatial disturbances only
  after motion and recovery are established. Direct shader rendering is a required
  comparison, since the NCA must add useful persistent state and local dynamics.

These are hypotheses and return conditions, not validated capabilities or a claim
that any one of them is novel enough for a paper.
