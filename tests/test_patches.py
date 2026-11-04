import tempfile
import unittest
from pathlib import Path

import torch

from vj_nca.data import make_procedural
from vj_nca.model import ConditionedNCA, ModelConfig
from vj_nca.train import TrainConfig, crop_targets, supervised_window, train


class PatchTests(unittest.TestCase):
    def test_supervision_covers_canvas_edges_but_excludes_artificial_edges(self):
        self.assertEqual(supervised_window([0, 416], 96, 512, 24), (slice(0, 72), slice(24, 96)))
        self.assertEqual(supervised_window([20, 20], 96, 512, 24), (slice(24, 72), slice(24, 72)))

    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(4)

    def test_patch_rule_matches_full_canvas_interior(self):
        model = ConditionedNCA(
            ModelConfig(
                num_loops=1,
                channels=7,
                hidden=16,
                hidden_layers=2,
                spatial_frequencies=(1, 4, 8),
                plane_channels=4,
                plane_resolution=16,
                plane_time_resolution=8,
                fire_rate=1,
            )
        )
        with torch.no_grad():
            model.update_out.weight.normal_(0, 0.1)
        state = torch.randn(1, 7, 32, 32) * 0.1
        ids, phase = torch.tensor([0]), torch.tensor([0.37])
        full = model(state, ids, phase)
        patch = model(
            state[..., 8:24, 8:24], ids, phase, origins=torch.tensor([[8, 8]]), canvas_size=32
        )
        torch.testing.assert_close(patch[..., 4:-4, 4:-4], full[..., 12:20, 12:20])
        wrong = model(state[..., 8:24, 8:24], ids, phase)
        self.assertGreater((wrong - patch).square().mean().item(), 1e-5)

    def test_targets_use_native_pixels_and_per_sample_origins(self):
        frames = torch.arange(2 * 3 * 16 * 16).reshape(2, 3, 16, 16)
        origins = torch.tensor([[2, 4], [8, 6]])
        crops = crop_targets(frames, origins, 4)
        torch.testing.assert_close(crops[0], frames[0, :, 2:6, 4:8])
        torch.testing.assert_close(crops[1], frames[1, :, 8:12, 6:10])

    def test_direct_visible_readout_keeps_hidden_residual_and_firing_mask(self):
        model = ConditionedNCA(ModelConfig(num_loops=1, channels=7, hidden=8, direct_rgb=True))
        state = torch.randn(1, 7, 8, 8)
        mask = torch.zeros(1, 1, 8, 8)
        mask[..., :4, :] = 1
        out = model(state, torch.tensor([0]), torch.tensor([0.2]), mask=mask)
        torch.testing.assert_close(out[:, :3, :4], torch.zeros_like(out[:, :3, :4]))
        torch.testing.assert_close(out[:, 3:], state[:, 3:])
        torch.testing.assert_close(out[..., 4:, :], state[..., 4:, :])

    def test_time_planes_wrap_and_receive_gradients(self):
        plane = torch.arange(4.0).reshape(1, 1, 1, 4).expand(1, 1, 2, 4).clone().requires_grad_()
        phases = torch.tensor([0.0, 0.25, 0.875, 1.0])
        values = ConditionedNCA.sample_time_plane(plane, torch.zeros(4, 1, 1), phases)
        torch.testing.assert_close(values.flatten(), torch.tensor([0.0, 1.0, 1.5, 0.0]))
        values.sum().backward()
        self.assertTrue(torch.isfinite(plane.grad).all())
        self.assertGreater(plane.grad.abs().sum().item(), 0)

        model = ConditionedNCA(
            ModelConfig(
                num_loops=2,
                channels=7,
                hidden=8,
                plane_channels=4,
                plane_resolution=8,
                plane_time_resolution=8,
            )
        )
        with torch.no_grad():
            model.update_out.weight.normal_(0, 0.02)
        ids, phase = torch.tensor([0, 1]), torch.tensor([0.13, 0.64])
        a = model(model.seed(2, 8, "cpu"), ids, phase, mask=torch.ones(2, 1, 8, 8))
        b = model(model.seed(2, 8, "cpu"), ids, phase + 1, mask=torch.ones(2, 1, 8, 8))
        torch.testing.assert_close(a, b)
        a.square().mean().backward()
        for parameter in model.planes.values():
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
        model.config.use_phase = False
        torch.testing.assert_close(
            model.conditioning(ids, phase, 8, 8), model.conditioning(ids, phase + 0.2, 8, 8)
        )

    def test_patch_resume_preserves_origins_rng_and_weights(self):
        dataset = make_procedural(size=16, frames=8, count=2)
        cfg = TrainConfig(
            iterations=4,
            channels=7,
            hidden=8,
            patch_size=8,
            plane_channels=2,
            plane_resolution=8,
            plane_time_resolution=8,
            plane_lr_multiplier=3,
            temporal_weight=5,
            loss_margin=2,
            batch_size=2,
            pool_per_loop=2,
            unroll_min=2,
            unroll_max=2,
            warmup_max=0,
            fresh_fraction=1,
            log_every=4,
        )
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            full = train(dataset, cfg, root / "full")
            cfg.iterations = 2
            train(dataset, cfg, root / "split")
            cfg.iterations = 4
            resumed = train(dataset, cfg, root / "split", resume=root / "split/last.pt")
            a, b = [torch.load(p, map_location="cpu", weights_only=False) for p in (full, resumed)]
            for key in a["model"]:
                torch.testing.assert_close(a["model"][key], b["model"][key], atol=0, rtol=0)
            torch.testing.assert_close(a["pool"]["origins"], b["pool"]["origins"])
            self.assertTrue((a["pool"]["origins"] % 2 == 0).all())
            self.assertGreater(a["pool"]["origins"].max().item(), 0)


if __name__ == "__main__":
    unittest.main()
