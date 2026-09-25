"""CPU regressions for paper Equations 1 and 5; run with unittest discovery."""

import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

from pretrain_models.UniEncoder.mask_tools import (
    _sample_keep_mask_torch,
    _sample_modal_mask_torch,
    apply_mask,
)


def _load_source_module(name, relative_path):
    # Avoid source.pretrain.__init__ importing the entire training stack for
    # these focused, CPU-only checks.
    path = Path(__file__).resolve().parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ReconstructionLoss = _load_source_module(
    "paper_masking_loss", "source/pretrain/loss_function.py"
).ReconstructionLoss
PretrainConfig = _load_source_module(
    "paper_masking_config", "source/pretrain/parse.py"
).PretrainConfig


class PaperMaskingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1024)
        self.device = torch.device("cpu")

    def test_all_15_nonempty_available_subsets_have_equal_probability(self):
        masks = _sample_modal_mask_torch(30000, 4, 3, self.device)
        ids = (masks.long() * torch.tensor([1, 2, 4, 8])).sum(dim=1)
        frequencies = torch.bincount(ids, minlength=16).float() / len(ids)
        self.assertEqual(frequencies[15].item(), 0.0)
        self.assertTrue(torch.all(frequencies[:15] > 0))
        self.assertTrue(torch.all((frequencies[:15] - 1 / 15).abs() < 0.01))

    def test_nondefault_probability_matches_conditioned_bernoulli(self):
        p = 0.2
        masks = _sample_modal_mask_torch(30000, 4, 3, self.device, p)
        ids = (masks.long() * torch.tensor([1, 2, 4, 8])).sum(dim=1)
        frequencies = torch.bincount(ids, minlength=16).float() / len(ids)
        expected = torch.tensor([
            p ** i.bit_count() * (1 - p) ** (4 - i.bit_count()) / (1 - p ** 4)
            for i in range(15)
        ])
        self.assertTrue(torch.all((frequencies[:15] - expected).abs() < 0.01))
        self.assertFalse(masks.all(dim=1).any())

    def test_probability_edges_and_legacy_cap(self):
        self.assertFalse(_sample_modal_mask_torch(100, 4, 3, self.device, 0).any())
        self.assertFalse(_sample_modal_mask_torch(100, 4, 0, self.device).any())
        capped = _sample_modal_mask_torch(100, 4, 1, self.device)
        self.assertTrue(torch.all(capped.sum(dim=1) <= 1))
        for invalid in (-0.1, 1.0):
            with self.assertRaises(ValueError):
                _sample_modal_mask_torch(1, 4, 3, self.device, invalid)

    def test_default_cap_adapts_to_modality_count(self):
        masks = _sample_modal_mask_torch(1000, 2, None, self.device)
        self.assertEqual(len(torch.unique(masks, dim=0)), 3)
        self.assertFalse(masks.all(dim=1).any())
        self.assertFalse(_sample_modal_mask_torch(10, 1, None, self.device).any())

    def test_patch_keep_count_has_bernoulli_mean_and_variance(self):
        keep = _sample_keep_mask_torch(
            4096, 64, 4, 0, True, 0.75, self.device
        )
        counts = keep.sum(dim=1).float()
        self.assertAlmostEqual(counts.mean().item(), 16.0, delta=0.3)
        self.assertAlmostEqual(counts.var().item(), 12.0, delta=1.0)
        joint_rate = (keep[:, 0] & keep[:, 1]).float().mean().item()
        self.assertAlmostEqual(joint_rate, 0.25 ** 2, delta=0.015)

    def test_volume_mask_keeps_modalities_coherent_and_patch_edges(self):
        volume = torch.ones(1, 4, 4, 4, 4)
        masked = apply_mask(128, volume, patch_size=2, crop_size=4, use_patch_mask=False)
        flattened = masked.flatten(2)
        self.assertTrue(torch.all(flattened == flattened[:, :, :1]))
        self.assertTrue(torch.all(flattened[:, :, 0].sum(dim=1) >= 1))
        full = apply_mask(
            1, volume, patch_size=2, crop_size=4, modality_mask_prob=0, patch_mask_ratio=0
        )
        torch.testing.assert_close(full, volume)
        empty = apply_mask(1, volume, patch_size=2, crop_size=4, patch_mask_ratio=1)
        self.assertFalse(empty.any())

    def test_dataclass_and_cli_use_paper_defaults(self):
        with patch.object(sys, "argv", ["pretrain.py"]):
            cli = PretrainConfig.from_args()
        for config in (PretrainConfig(), cli):
            self.assertEqual(config.base_lr, 3e-4)
            self.assertEqual(config.min_lr, 1e-6)
            self.assertEqual(config.crop_size, 96)
            self.assertEqual(config.modality_mask_prob, 0.5)
            self.assertEqual(config.num_mask_modalities, 3)
            self.assertEqual(config.mask_ratio, 0.75)


class PaperRegularizerTests(unittest.TestCase):
    def test_constant_mask_is_penalized_with_raw_norm(self):
        prior = torch.full((2, 4, 2, 2, 2), 3.0, requires_grad=True)
        target = torch.zeros_like(prior)
        loss = ReconstructionLoss()(target, prior, target)
        expected = 0.005 * (3 * (8 ** 0.5) + 1e-6)
        self.assertAlmostEqual(loss.item(), expected, places=7)
        loss.backward()
        self.assertTrue(torch.all(prior.grad > 0))

    def test_norm_reduction_averages_batch_and_channels(self):
        prior = torch.tensor([3.0, 4.0]).reshape(1, 2, 1, 1, 1)
        target = torch.zeros_like(prior)
        loss = ReconstructionLoss()(target, prior, target)
        self.assertAlmostEqual(loss.item(), 0.005 * (3.5 + 1e-6), places=7)

    def test_epsilon_adds_offset_without_changing_raw_norm_gradient(self):
        prior = torch.tensor(
            [3.0, 4.0, 0.0, 0.0], dtype=torch.float64
        ).reshape(1, 2, 1, 1, 2).requires_grad_()
        target = torch.zeros_like(prior)
        rate, epsilon = 0.2, 0.125
        with_epsilon = ReconstructionLoss(regulization_rate=rate, eps=epsilon)
        self.assertEqual(with_epsilon.eps, epsilon)
        base_loss = ReconstructionLoss(regulization_rate=rate, eps=0.0)(target, prior, target)
        offset_loss = with_epsilon(target, prior, target)
        self.assertAlmostEqual(base_loss.item(), rate * 2.5, places=14)
        self.assertAlmostEqual((offset_loss - base_loss).item(), rate * epsilon, places=14)
        base_gradient, = torch.autograd.grad(base_loss, prior)
        offset_gradient, = torch.autograd.grad(offset_loss, prior)
        torch.testing.assert_close(offset_gradient, base_gradient, rtol=0, atol=0)

    def test_zero_regularization_rate_ignores_epsilon(self):
        prior = torch.zeros((1, 2, 1, 1, 1))
        target = torch.zeros_like(prior)
        recon = torch.ones_like(prior)
        loss = ReconstructionLoss(regulization_rate=0.0, eps=0.125)(recon, prior, target)
        self.assertEqual(loss.item(), 1.0)

    def test_zero_mask_has_finite_zero_gradient(self):
        prior = torch.zeros((1, 4, 2, 2, 2), requires_grad=True)
        target = torch.zeros_like(prior)
        loss = ReconstructionLoss()(target, prior, target)
        self.assertAlmostEqual(loss.item(), 0.005 * 1e-6, places=14)
        loss.backward()
        self.assertTrue(torch.isfinite(prior.grad).all())
        self.assertFalse(prior.grad.any())


if __name__ == "__main__":
    unittest.main()
