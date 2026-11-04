import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import torch

from vj_nca.data import make_procedural
from vj_nca.evaluate import evaluate, motion_statistics, recovery_frame
from vj_nca.model import ConditionedNCA, ModelConfig
from vj_nca.stream_eval import evaluate_streaming


class EvaluationTests(unittest.TestCase):
    def test_streaming_scores_match_full_video_evaluator(self):
        torch.set_num_threads(1)
        torch.manual_seed(8)
        ds = make_procedural(size=8, frames=8, count=2)
        model = ConditionedNCA(ModelConfig(num_loops=2, channels=7, hidden=8, fire_rate=1))
        with torch.no_grad():
            model.update_out.weight.normal_(0, 0.02)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "model.pt"
            torch.save(
                {
                    "model_config": asdict(model.config),
                    "model": model.state_dict(),
                    "dataset_fingerprint": ds.fingerprint(),
                    "training_step": 0,
                    "steps_per_frame": 2,
                },
                path,
            )
            old = evaluate(path, ds, root / "full", burn_cycles=1, cycles=2, export=False)
            new = evaluate_streaming(
                path, ds, root / "stream", device="cpu", burn_cycles=1, cycles=2, export=False
            )
            for name in ds.names:
                for key in (
                    "mse",
                    "psnr",
                    "motion_cosine",
                    "motion_gain",
                    "relative_temporal_error",
                    "cycle_drift_mse",
                ):
                    self.assertAlmostEqual(
                        old["per_loop"][name][key], new["per_loop"][name][key], places=5
                    )

    def test_released_weights_reproduce_full_checkpoint_evaluation(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        ds = make_procedural(size=4, frames=4, count=1)
        model = ConditionedNCA(ModelConfig(num_loops=1, channels=7, hidden=8))
        # A phase-dependent output makes a misaligned reset probe observable.
        with torch.no_grad():
            model.update_out.weight.normal_(0, 0.1)
        shared = {
            "dataset_fingerprint": ds.fingerprint(),
            "model_config": asdict(model.config),
            "model": model.state_dict(),
        }
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            torch.save(
                {**shared, "step": 5, "train_config": {"steps_per_frame": 2}}, root / "full.pt"
            )
            torch.save({**shared, "training_step": 5, "steps_per_frame": 2}, root / "export.pt")
            results = [
                evaluate(root / name, ds, root / str(i), burn_cycles=0, cycles=2, export=False)
                for i, name in enumerate(("full.pt", "export.pt"))
            ]
            self.assertEqual(results[0], results[1])
            shifted = evaluate(
                root / "export.pt",
                ds,
                root / "shifted",
                burn_cycles=0,
                cycles=2,
                export=False,
                phase_frame=1,
            )
            self.assertEqual(shifted["perturbation_phase_frame"], 1)
            # The one-step reset probe must compare against the matching shifted phase.
            self.assertAlmostEqual(
                shifted["single_step_reset_mse"], results[0]["single_step_reset_mse"]
            )
            with self.assertRaisesRegex(ValueError, "phase_frame"):
                evaluate(root / "export.pt", ds, root / "invalid", phase_frame=ds.period)

    def test_motion_scores_reject_freezing_reversal_and_unrelated_flicker(self):
        target = torch.arange(9.0).reshape(1, 9, 1, 1, 1).expand(1, 9, 3, 2, 2)
        perfect = motion_statistics(target, target)
        self.assertAlmostEqual(perfect["motion_gain"].item(), 1)
        self.assertAlmostEqual(perfect["motion_cosine"].item(), 1)
        frozen = motion_statistics(target * 0, target)
        self.assertEqual(frozen["motion_gain"].item(), 0)
        self.assertEqual(frozen["motion_cosine"].item(), 0)
        reverse = motion_statistics(target.flip(1), target)
        self.assertAlmostEqual(reverse["motion_cosine"].item(), -1)
        flicker = torch.tensor([0.0, 1] * 4 + [0]).reshape(1, 9, 1, 1, 1).expand_as(target)
        noise = motion_statistics(flicker, target)
        self.assertAlmostEqual(noise["motion_energy_ratio"].item(), 1)
        self.assertAlmostEqual(noise["motion_cosine"].item(), 0)

    def test_static_incompetent_model_cannot_claim_recovery(self):
        self.assertIsNone(recovery_frame([0] * 20, [0.1] * 20, [0.1] * 20, False))

    def test_recovery_requires_sustained_agreement_and_target_accuracy(self):
        paired = [0.1, 0.05, 0.002, 0.001, 0.001, 0.001]
        self.assertEqual(recovery_frame(paired, [0.01] * 6, [0.01] * 6, True), 2)
        self.assertIsNone(recovery_frame(paired, [0.2] * 6, [0.01] * 6, True))
        self.assertIsNone(recovery_frame([0.1, 0.001, 0.1, 0.001], [0.01] * 4, [0.01] * 4, True))


if __name__ == "__main__":
    unittest.main()
