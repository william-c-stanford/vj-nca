# mixed-v0

Experimental phase-conditioned neural cellular automaton, trained November 3, 2026.

Model weights and previews: [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/).
Training data: six original CC0 loops plus **-PLASMA-ColorCycling.Gif**, by **Ósemkowski**,
[Wikimedia Commons](https://commons.wikimedia.org/wiki/File:-PLASMA-ColorCycling.Gif), CC BY-SA 3.0.
Source changes: square crop, 32×32, 32 frames, retimed from 5.12 seconds to two seconds.

The Python implementation is MIT licensed. This is a pilot model, not a general video model.
Load the tensor-only weights using `torch.load(path, map_location="cpu", weights_only=True)`,
construct `ConditionedNCA(ModelConfig(**payload["model_config"]))`, and load `payload["model"]`.
Loop ID ordering and steps per frame are in `model.json`. Supply a normalized loop phase at every update.
No target image is supplied during generation. Playback is 16 fps; the phase period is 32 frames.

See [preview.mp4](preview.mp4), [metrics.json](metrics.json), and the exact
[training source](training-source.zip). Full settings and provenance are in
[the experiment record](../../experiments/2026-11-03/mixed-final).

The rotating lattice did not learn adequate motion. The public plasma clip, where present,
also under-represents motion. Results concern training-loop reconstruction from new trajectories;
unseen videos, audio input, continuous tempo control and live rendering have not been validated.
