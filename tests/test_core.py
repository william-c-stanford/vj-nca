import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from vj_nca.data import LoopDataset, make_procedural
from vj_nca.losses import temporal_loss
from vj_nca.model import ConditionedNCA, ModelConfig
from vj_nca.train import ReplayPool, TrainConfig, train


class CoreTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(12)

    def test_phase_is_periodic_and_mask_freezes_state(self):
        model = ConditionedNCA(ModelConfig(num_loops=2, channels=11, hidden=16))
        x = torch.randn(2, 11, 8, 8) * 0.1
        ids = torch.tensor([0, 1])
        phase = torch.tensor([0.13, 0.6])
        mask = torch.ones(2, 1, 8, 8)
        with torch.no_grad():
            model.update_out.weight.normal_(0, 0.1)
        a = model(x, ids, phase, mask=mask)
        b = model(x, ids, phase + 1, mask=mask)
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(model(x, ids, phase, mask=mask * 0), x)
        self.assertGreater((a - x).abs().mean().item(), 0)

    def test_condition_reaches_every_cell_and_receives_gradients(self):
        model = ConditionedNCA(ModelConfig(num_loops=2, channels=11, hidden=16))
        with torch.no_grad():
            model.update_out.weight.normal_(0, 0.1)
        x = torch.zeros(2, 11, 8, 8)
        out = model(x, torch.tensor([0, 1]), torch.zeros(2), mask=torch.ones(2, 1, 8, 8))
        self.assertGreater((out[0] - out[1]).abs().mean().item(), 1e-4)
        out.square().mean().backward()
        self.assertTrue(torch.isfinite(model.embedding.weight.grad).all())
        self.assertGreater(model.embedding.weight.grad.abs().sum().item(), 0)

    def test_hidden_normalization_is_per_cell_and_preserves_rgb_and_nonfiring_cells(self):
        model = ConditionedNCA(
            ModelConfig(num_loops=1, channels=11, hidden=16, hidden_norm="unit_ball", norm_groups=2)
        )
        x = torch.randn(1, 11, 8, 8)
        mask = torch.zeros(1, 1, 8, 8)
        mask[:, :, 0, 0] = 1
        out = model(x, torch.tensor([0]), torch.zeros(1), mask=mask)
        torch.testing.assert_close(out[:, :3], x[:, :3])
        torch.testing.assert_close(out[:, :, 1:, :], x[:, :, 1:, :])
        norms = out[:, 3:, 0, 0].reshape(1, 2, 4).norm(dim=2)
        self.assertTrue((norms <= 1.000001).all())
        small = x * 1e-4
        unchanged = model(small, torch.tensor([0]), torch.zeros(1), mask=torch.ones_like(mask))
        torch.testing.assert_close(unchanged, small)

    def test_hidden_normalization_has_finite_gradients_near_zero(self):
        model = ConditionedNCA(
            ModelConfig(num_loops=1, channels=11, hidden=16, hidden_norm="unit_ball", norm_groups=2)
        )
        with torch.no_grad():
            model.update_out.weight[:3].normal_(0, 0.01)
        state = torch.zeros(1, 11, 8, 8)
        for step in range(16):
            state = model(
                state, torch.tensor([0]), torch.tensor([step / 32]), mask=torch.ones(1, 1, 8, 8)
            )
        (state[:, :3] - 0.5).square().mean().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_cyclic_sampler_interpolates_across_seam(self):
        frames = torch.tensor([0.0, 1.0, 2.0, 3.0]).reshape(1, 4, 1, 1, 1)
        frames = frames.expand(1, 4, 3, 2, 2)
        ds = LoopDataset(frames, ["ramp"], fps=4)
        got = ds.target(torch.tensor([0, 0, 0]), torch.tensor([0, 7, 8]), 2)
        torch.testing.assert_close(got[:, 0, 0, 0], torch.tensor([0.0, 1.5, 0.0]))

    def test_temporal_loss_rejects_frozen_and_reversed_motion(self):
        target = torch.arange(4.0).reshape(1, 4, 1, 1, 1)
        self.assertEqual(temporal_loss(target, target).item(), 0)
        self.assertGreater(temporal_loss(target * 0, target).item(), 0)
        self.assertGreater(temporal_loss(target.flip(1), target).item(), 1)

    def test_procedural_loops_are_distinct_and_closed(self):
        ds = make_procedural(size=16, frames=16, count=4)
        self.assertEqual(tuple(ds.frames.shape), (4, 16, 3, 16, 16))
        self.assertTrue(torch.isfinite(ds.frames).all())
        self.assertGreater((ds.frames[0] - ds.frames[1]).abs().mean().item(), 0.1)
        seam = (ds.frames[:, 0] - ds.frames[:, -1]).square().mean()
        motion = ds.frames.diff(dim=1).square().mean()
        self.assertLess(seam.item(), 4 * motion.item())
        with tempfile.TemporaryDirectory() as d:
            ds.save(Path(d) / "loops.npz")
            loaded = LoopDataset.load(Path(d) / "loops.npz")
            torch.testing.assert_close(ds.frames, loaded.frames)

    def test_replay_keeps_phase_identity_and_detaches_graph(self):
        pool = ReplayPool(2, 4, 11, 8, torch.device("cpu"))
        slots = torch.tensor([0, 5])
        states = torch.ones(2, 11, 8, 8, requires_grad=True)
        pool.write(slots, states, torch.tensor([17, 42]), torch.tensor([8, 9]))
        torch.testing.assert_close(pool.ticks[slots], torch.tensor([17, 42]))
        torch.testing.assert_close(pool.ages[slots], torch.tensor([8, 9]))
        torch.testing.assert_close(pool.loop_ids(slots), torch.tensor([0, 1]))
        self.assertFalse(pool.states.requires_grad)

    def test_training_resume_restores_rng_optimizer_and_pool(self):
        ds = make_procedural(size=8, frames=8, count=2)
        cfg = TrainConfig(
            iterations=4,
            batch_size=2,
            pool_per_loop=2,
            unroll_min=2,
            unroll_max=2,
            warmup_max=0,
            log_every=1,
            checkpoint_every=2,
            channels=11,
            hidden=16,
            seed=5,
        )
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            full = train(ds, cfg, root / "full")
            cfg.iterations = 2
            train(ds, cfg, root / "split")
            cfg.iterations = 4
            resumed = train(ds, cfg, root / "split", resume=root / "split" / "last.pt")
            a = torch.load(full, weights_only=False, map_location="cpu")
            b = torch.load(resumed, weights_only=False, map_location="cpu")
            for key in a["model"]:
                torch.testing.assert_close(a["model"][key], b["model"][key], atol=0, rtol=0)
            torch.testing.assert_close(a["pool"]["ticks"], b["pool"]["ticks"])
            self.assertTrue(np.isfinite(b["last_loss"]))
            wrong_order = LoopDataset(ds.frames.flip(0), ds.names[::-1], ds.fps, ds.sources[::-1])
            with self.assertRaisesRegex(ValueError, "Checkpoint dataset differs"):
                train(wrong_order, cfg, root / "wrong-data", resume=resumed)
            cfg.learning_rate *= 2
            with self.assertRaisesRegex(ValueError, "Resume cannot change"):
                train(ds, cfg, root / "wrong-settings", resume=resumed)


if __name__ == "__main__":
    unittest.main()
