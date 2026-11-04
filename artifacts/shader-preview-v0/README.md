# Kinetic line studies — shader targets

First training family: **Aperture**. Fold and Interference are later curriculum
studies. The [gallery](index.html), [triptych](triptych.mp4) and individual
[Aperture](aperture.mp4), [Fold](fold.mp4), [Interference](interference.mp4) videos
show the original shaders, **not a trained NCA**. Each panel is actually rendered
at 512×512, 192 frames, 24 fps, eight seconds; the triptych places three native
panels side by side without enlargement.

The [aesthetic research and selection](../../docs/SHADER_AESTHETICS.md) explains the
artist references, design judgments, competing styles and difficulty measurements.
The [future-work file](../../docs/FUTURE_WORK.md) preserves the deferred directions.

Render again from the repository root:

```bash
python -m pip install -e '.[shaders]'
PYTHONPATH=src python scripts/render_shader_previews.py
PYTHONPATH=src python scripts/probe_shader_learnability.py
```

FFmpeg is required for video export. The renderer was verified on Apple M4 Max,
OpenGL 4.1, ModernGL 5.13.0 and glcontext 3.1.0. Linux/EGL has not been tested.
`--stills-only --out runs/my-style-study` renders a small style check first.
Edit a copy of [the preset manifest](../../datasets/shader-preview-v0.json) to
explore the declared parameter ranges. No cloud service or neural checkpoint is
required to render these targets.

[manifest.json](manifest.json) and per-study JSON files retain controls, normalized
controls, source/description hashes, actual renderer/library versions and seam
checks. `source/` retains the exact assembled GLSL, vertex shader, renderer and probe
code used for this version. These files are audit snapshots; use the repository
layout and commands above to rerun. The probe records its own source and input
manifest hashes. It uses native float renders, not compressed MP4 pixels.

The renderer checks finite/range-valid output, raw phase-zero/one continuity and
numerical derivative continuity. Frames sample phases `0 .. 191/192`; no duplicate
endpoint is inserted. These checks do not establish NCA learnability or aesthetic
quality by themselves. Source/code embeddings have not yet been computed or trained.

Original shader and renderer code: [MIT](../../LICENSE). Rendered videos and stills:
[CC0-1.0](https://creativecommons.org/publicdomain/zero/1.0/), dedicated by this
project. No reference artist's media, textures, code or pretrained weights are used.
The separate Electric Sheep experiments retain their noncommercial restrictions;
those restrictions do not apply to these original shader renders.
