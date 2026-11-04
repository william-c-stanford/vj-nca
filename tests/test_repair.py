import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import torch

from vj_nca.data import make_procedural
from vj_nca.model import ConditionedNCA, ModelConfig
from vj_nca.perturbations import perturb_repair, repair_teacher_step
from vj_nca.repair_eval import evaluate_repair, sustained_recovery
from vj_nca.repair_train import initialize_repair, train_repair
from vj_nca.train import TrainConfig


class RepairTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(9)

    def test_blackout_is_black_and_removes_hidden_state_only_inside_circle(self):
        state = torch.ones(1, 12, 64, 64)
        out, mask = perturb_repair(state, "blackout", circle=(0, 0, 0.6))
        self.assertTrue((out[:, :3].masked_select(mask) == -1).all())
        self.assertTrue((out[:, 3:].masked_select(mask) == 0).all())
        self.assertTrue((out.masked_select(~mask) == 1).all())
        self.assertGreater(mask.float().mean().item(), 0.2)

    def test_glitches_damage_support_and_preserve_healthy_cells(self):
        state = torch.ones(1, 12, 64, 64)
        for kind in ("glitch", "rgb-split", "tear"):
            out, mask = perturb_repair(state, kind)
            self.assertTrue(mask.any())
            self.assertTrue((out[:, -1:].masked_select(mask) == 0).all())
            self.assertTrue((out.masked_select(~mask) == 1).all())
            self.assertGreater(out[:, :3].std().item(), 0.1)
        beat, _ = perturb_repair(state, "beat")
        torch.testing.assert_close(beat[:, -1:], state[:, -1:])

    def test_growth_curriculum_rebuilds_from_edges_before_the_core(self):
        cfg = ModelConfig(
            num_loops=1,
            channels=12,
            hidden=8,
            repair_channel=True,
            direct_rgb=True,
            repair_radius=4,
            repair_rate=0.12,
        )
        state = torch.ones(1, 12, 64, 64)
        out, _ = perturb_repair(state, "blackout", circle=(0, 0, 0.6))
        support, rgb = out[:, -1:], out[:, :3]
        for _ in range(4):
            support, rgb = repair_teacher_step(support, rgb, state[:, :3], cfg)
        self.assertGreater(support[0, 0, 32, 49].item(), support[0, 0, 32, 32].item())
        self.assertEqual(rgb[0, 0, 32, 32].item(), -1)

    def test_warm_start_preserves_clean_rule_and_support_stays_in_bounds(self):
        old = ConditionedNCA(
            ModelConfig(
                num_loops=1,
                channels=11,
                hidden=8,
                direct_rgb=True,
                hidden_norm="unit_ball",
                norm_groups=2,
                fire_rate=1,
            )
        )
        with torch.no_grad():
            old.update_out.weight.normal_(0, 0.1)
        new = ConditionedNCA(
            ModelConfig(
                num_loops=1,
                channels=12,
                hidden=8,
                direct_rgb=True,
                hidden_norm="unit_ball",
                norm_groups=2,
                fire_rate=1,
                repair_channel=True,
                repair_radius=2,
                repair_rate=0.12,
            )
        )
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / "model.pt"
            torch.save(dict(model_config=asdict(old.config), model=old.state_dict()), p)
            initialize_repair(new, p)
        state = torch.randn(1, 11, 16, 16) * 0.1
        repaired = torch.cat([state, torch.ones_like(state[:, :1])], 1)
        ids, phase = torch.tensor([0]), torch.tensor([0.2])
        torch.testing.assert_close(new(repaired, ids, phase)[:, :11], old(state, ids, phase))
        damaged, _ = perturb_repair(repaired, "full-reset")
        next_state = new(damaged, ids, phase)
        self.assertTrue((next_state[:, -1:] >= 0).all())
        self.assertTrue((next_state[:, -1:] <= 0.12).all())
        torch.testing.assert_close(next_state[:, :3], damaged[:, :3])
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / "repair.pt"
            torch.save(dict(model_config=asdict(new.config), model=new.state_dict()), p)
            restored = ConditionedNCA(new.config)
            initialize_repair(restored, p)
            for key in new.state_dict():
                torch.testing.assert_close(new.state_dict()[key], restored.state_dict()[key])

    def test_repair_resume_restores_curriculum_and_optimizer(self):
        dataset = make_procedural(size=16, frames=4, count=1)
        cfg = TrainConfig(
            iterations=4,
            channels=8,
            hidden=8,
            repair_channel=True,
            repair_radius=2,
            repair_rate=0.12,
            direct_rgb=True,
            patch_size=8,
            loss_margin=2,
            batch_size=2,
            pool_per_loop=2,
            unroll_min=2,
            unroll_max=2,
            damage_probability=1,
            fresh_fraction=0.1,
            log_every=4,
            checkpoint_every=4,
            steps_per_frame=4,
            fire_rate=1,
        )
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)
            full = train_repair(dataset, cfg, p / "full")
            cfg.iterations = 2
            train_repair(dataset, cfg, p / "split")
            cfg.iterations = 4
            split = train_repair(dataset, cfg, p / "split", resume=p / "split/last.pt")
            a, b = [torch.load(x, weights_only=False) for x in (full, split)]
            for key in a["model"]:
                torch.testing.assert_close(a["model"][key], b["model"][key], atol=0, rtol=0)
            for key in a["pool"]:
                torch.testing.assert_close(a["pool"][key], b["pool"][key], atol=0, rtol=0)
            self.assertTrue((a["pool"]["ticks"] < dataset.period * 4).all())

    def test_evaluator_measures_spatial_regions_and_sustains_a_cycle(self):
        data = make_procedural(size=16, frames=4, count=1)
        model = ConditionedNCA(
            ModelConfig(
                num_loops=1,
                channels=8,
                hidden=8,
                repair_channel=True,
                direct_rgb=True,
                repair_radius=2,
                fire_rate=1,
            )
        )
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)
            torch.save(
                dict(
                    model_config=asdict(model.config),
                    model=model.state_dict(),
                    dataset_fingerprint=data.fingerprint(),
                    steps_per_frame=4,
                ),
                p / "weights.pt",
            )
            result = evaluate_repair(
                p / "weights.pt", data, p / "eval", device="cpu", frames=32, export=False
            )
            self.assertEqual(result["fps"], 64)
            self.assertEqual(result["cycle_seconds"], 0.25)
            self.assertEqual(result["recovery_sustained_frames"], 16)
            self.assertIn("rim", result["blackout_regional_recovery_seconds"])
            self.assertFalse(result["quality_and_recovery_pass"])
        curve = [0.0, 0.04, 0.03, 0.04] * 4 + [0.0] * 8
        self.assertEqual(sustained_recovery(curve, curve, [0.0] * len(curve), True, 4), 16)


if __name__ == "__main__":
    unittest.main()
