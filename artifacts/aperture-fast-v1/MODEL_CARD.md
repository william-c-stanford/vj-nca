# aperture-fast-v1

Native **512×512** Aperture reconstruction with perturbation recovery, trained
November 3, 2026. **144,332 parameters**, including 131,072 learned feature-plane
parameters and a 13,260-parameter local update/conditioning rule. One update per
display frame, 11 state channels, two 64-unit hidden layers, 16 phase harmonics,
shared XY/XT/YT planes and local hidden unit-ball normalization.

Watch the [150 BPM learned preview](preview-150bpm.mp4), [synthetic beat kicks and
recovery](beat-recovery.mp4), or [target / clean / erased comparison](comparison.mp4).
The [gallery](index.html) lets you retime the previews. These are model outputs;
[target.mp4](target.mp4) is the original GLSL teacher.

## Quality and speed

Frozen weights passed [32 seconds of native-pixel evaluation](held-out/metrics.json)
on an Apple M4 Max: **28.02 dB PSNR**, signed motion cosine **0.902**, projected
motion gain **0.838**, MSE **0.001579** versus static-mean MSE **0.051491**.
Erase, noise, synthetic beat and full reset recovered in **0.094, 0.031, 0.063
and 0.125 seconds** respectively, sustained over four consecutive frames.
Tests at [quarter](phase-16/metrics.json), [half](phase-32/metrics.json) and
[three-quarter](phase-48/metrics.json) cycle phases also pass. Recovery times use
the training playback rate of 32 fps. Switch-to-self is excluded from interpretation.

The [synchronized benchmark](benchmark.json) measured **136.4 frames/second**
including native uint8 RGB CPU readback on the M4 Max, PyTorch 2.8 MPS, float32,
600 measured frames per path after warmup. Readback p50/p95: **7.35/7.69ms**.
GPU-resident throughput: 137.6fps. Display, audio, video encoding and application
overhead are excluded; this is not a completed live VJ application benchmark.

The training cycle is **two seconds**, 4× faster than the original eight-second
shader preview. The 150 BPM export plays four beats per cycle in **1.6 seconds**,
5× the original pace, at 40 fps. This is explicit retiming of the same 64-state
trajectory. A video file's frame rate is separate from measured inference speed.

[Resetting all state before every frame](recurrence.json) raises reconstruction
error **9.07×** and lowers motion cosine to 0.437, so carried state contributes
substantially. This does not establish an autonomous limit cycle or superiority
over directly running the shader.

## Reproduce and change tempo

The exact [float32 target dataset](targets.npz), 33.3 MiB compressed, is included.
It avoids fingerprint changes caused by renderer/version metadata or GPU rounding.
The frames are original CC0 renders, not reference-artist assets. The
[dataset record](dataset.json) retains GLSL hashes, description and numeric controls.
To create a new local dataset instead, run `scripts/prepare_aperture.py`; newly
rendered datasets can have different fingerprints and require their own training.

```sh
PYTHONPATH=src python -m vj_nca evaluate --streaming \
  --data artifacts/aperture-fast-v1/targets.npz \
  --checkpoint artifacts/aperture-fast-v1/model.pt \
  --out runs/reproduce-aperture --device mps --seed 9091 --cycles 16 --no-export

PYTHONPATH=src python scripts/benchmark_nca.py \
  --weights artifacts/aperture-fast-v1/model.pt \
  --out runs/aperture-benchmark.json --device mps --samples 600

PYTHONPATH=src python scripts/render_aperture.py \
  --weights artifacts/aperture-fast-v1/model.pt \
  --out runs/aperture-150bpm.mp4 --bpm 150 --device mps
```

Add `--beat` for synthetic brightness/latent kicks. Change `--bpm` to retime
playback. The script exports offline; it does not promise frame pacing at every
requested tempo. Use `--device cuda` on a CUDA machine. FFmpeg is required for video.

Load the tensor-only weights using `torch.load(path, weights_only=True)`, construct
`ConditionedNCA(ModelConfig(**saved['model_config']))`, then load `saved['model']`.
Each update takes a carried state, loop ID0 and phase `tick % 64 / 64`.
Keep `fire_rate=1`; initialize with `model.seed(1,512,device)` and warm up two cycles.
No target frames are supplied during inference. Read-only planes are trained
parameters sampled at coordinates and phase; they are not mutable recurrent state.

## Training, license and limits

The selected lineage trained **11,546 total updates**, first 4,000 on an A10,
then 7,546 on an A10G with exact optimizer/replay/RNG resume. The refinement
stopped at its 300-second cap, short of the requested 12,000 updates. Combined
lineage training time: about seven minutes. [Configuration](config.json),
[training report](training_report.json), [exact training source](training-source.zip),
[selection freeze](selection.json), [development metrics](metrics-dev.json), and
[four-trial research report](../../experiments/aperture-fast/REPORT.md) are retained.

Code and Aperture-only weights: MIT. Original target renders and media: CC0-1.0.
No Electric Sheep data or weights are used in this artifact. See
[attribution](../../ATTRIBUTION.md).

This is **one shader preset and one loop code**. Numeric controls and code/text
embeddings have not been trained as conditioning signals. There is no unseen-loop,
unseen-program or interpolation result. The phase clock is external. Synthetic
beat kicks are not audio analysis. Only one training seed is represented, motion
amplitude is still attenuated, and long-term installation behavior is untested.
The [program-conditioning roadmap](../../docs/SHADER_CONDITIONING.md) remains future work.
