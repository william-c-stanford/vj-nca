# mixed-v1

One shared phase-conditioned neural cellular automaton, trained November 3, 2026.
**26,452 parameters**, 35 state channels, 128 hidden units, eight phase harmonics,
and synchronous updates. The loop embedding has 12 dimensions. Local hidden
unit-ball normalization, replay, damage, noise and state-swap training are retained.
Training: 5,000 updates, 190 seconds on one T4, seven loops at 32×32 and 16 fps.

The [preview](preview.mp4) shows target / clean rollout / erased-and-recovering
rollout in columns, one loop per row. [Contact sheet](contact.png).

The fixed [held-out evaluation](metrics.json) spans 32 seconds after warmup:
7/7 loops pass phase-correct motion, reconstruction and all five perturbation
recovery checks. Mean per-loop PSNR is 28.76 dB; MSE is 0.001788, 65.1% lower than
mixed-v0 under the same evaluation. Recovery onset is 0.06–0.44 simulated video
seconds, confirmed over four consecutive frames. Separate quarter-cycle phase
tests also pass 7/7. See the [full report](../../experiments/autoresearch/REPORT.md)
for gates, controls, raw evidence, costs and limitations.

## Load and evaluate

```python
import torch
from vj_nca.model import ConditionedNCA, ModelConfig

payload = torch.load("artifacts/mixed-v1/model.pt", map_location="cpu", weights_only=True)
model = ConditionedNCA(ModelConfig(**payload["model_config"])).eval()
model.load_state_dict(payload["model"])
```

Supply the requested loop ID and normalized cycle phase at every update. There
are **two NCA updates per displayed frame** and 32 frames per cycle. Initialize
with `model.seed(...)`; no target frames are inputs. Preserve `fire_rate=1.0` for
the reported behavior. Loop ordering and metadata are in [model.json](model.json).

After preparing `data/mixed.npz` using `scripts/prepare_public.py`:

```sh
PYTHONPATH=src python scripts/score_motion.py \
  --checkpoint artifacts/mixed-v1/model.pt --data data/mixed.npz \
  --out runs/reproduce-mixed-v1 --seed 9091 --cycles 16
```

The exact [training source](training-source.zip), [configuration](../../configs/synchronous-003.json),
[development metrics](metrics-dev.json) and [trial record](../../experiments/autoresearch/trials/synchronous-003)
are retained. The current evaluator also accepts the released tensor-only artifact.

## Data, license and limits

Weights and previews: [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/).
Training data: six original CC0 loops plus **-PLASMA-ColorCycling.Gif**, by
**Ósemkowski**, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:-PLASMA-ColorCycling.Gif),
CC BY-SA 3.0. Source changes: square crop, 32×32, 32 frames, retimed from 5.12 to
two seconds. [Attribution](../../ATTRIBUTION.md). Python implementation: MIT.

This is reconstruction of seven training loops from new states/corruptions.
The public plasma's motion amplitude remains lower than the target's. No unseen
videos, audio control, tempo variation, production resolution or real-time speed
has been validated. The phase clock is external; this is not an autonomous
limit-cycle claim. A single training seed is insufficient for publication claims.
