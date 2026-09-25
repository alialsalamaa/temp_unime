"""Prevent mismatched volume/prior axes and random-crop indexing failures."""

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from source.dataset import augmentation


def _load_source_module(name, relative_path):
    path = Path(__file__).resolve().parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Keep CPU geometry tests independent of the full training stack imports.
config_module = _load_source_module("paper_prior_config", "source/pretrain/parse.py")
with patch.dict(sys.modules, {"source.pretrain.parse": config_module}):
    dataset_module = _load_source_module("paper_prior_dataset", "source/pretrain/dataset.py")


class PriorGeometryTests(unittest.TestCase):
    def setUp(self):
        temporary_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_dir.cleanup)
        self.root = Path(temporary_dir.name)
        (self.root / "vol").mkdir()
        self.train_file = self.root / "train.txt"
        self.train_file.write_text("case_0\n", encoding="utf-8")
        self.volume = np.arange(5 * 6 * 7 * 4, dtype=np.float32).reshape(5, 6, 7, 4)
        np.save(self.root / "vol" / "case_0_vol.npy", self.volume)

    def dataset(self, prior_shape=(7, 5, 6)):
        dataset = dataset_module.PretrainDataset(
            str(self.root), str(self.train_file), crop_size=3,
            include_auxiliary=False, prior_shape=prior_shape,
        )
        self.addCleanup(dataset.close)
        return dataset

    def test_cli_and_dataclass_use_official_original_pipeline_dhw_shape(self):
        with patch.object(sys, "argv", ["pretrain.py"]):
            args = config_module.PretrainConfig.from_args()
        self.assertEqual(tuple(args.original_shape), (160, 180, 210))
        self.assertEqual(tuple(args.original_shape), config_module.PretrainConfig().original_shape)

    def test_asymmetric_volume_crop_locations_match_dhw_prior_at_far_edge(self):
        dataset = self.dataset()
        with patch.object(augmentation.random, "randint", side_effect=lambda low, high: high):
            tensor, location = dataset[0]
        self.assertEqual(location, ((4, 7), (2, 5), (3, 6)))
        self.assertEqual(tuple(tensor.shape), (4, 3, 3, 3))
        expected = self.volume[2:5, 3:6, 4:7].transpose(3, 2, 0, 1)
        np.testing.assert_array_equal(tensor.numpy(), expected)

    def test_validates_all_training_headers_before_first_crop(self):
        self.train_file.write_text("case_0\ncase_1\n", encoding="utf-8")
        np.save(self.root / "vol" / "case_1_vol.npy", np.zeros((5, 8, 7, 4), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, r"case_1.*D,H,W shape \(7, 5, 8\).*original_shape"):
            self.dataset()

    def test_does_not_read_validation_or_test_volumes(self):
        # These names deliberately have no saved volumes.
        (self.root / "val.txt").write_text("missing_validation_case\n", encoding="utf-8")
        (self.root / "test.txt").write_text("missing_test_case\n", encoding="utf-8")
        self.assertEqual(len(self.dataset()), 1)

    def test_rejects_short_volumes_and_invalid_prior(self):
        for prior_shape in ((0, 5, 6), (-1, 5, 6), (7, 5)):
            with self.subTest(prior_shape=prior_shape), self.assertRaisesRegex(ValueError, "positive D,H,W"):
                self.dataset(prior_shape)
        np.save(self.root / "vol" / "case_0_vol.npy", np.zeros((2, 6, 7, 4), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "smaller than crop_size"):
            self.dataset()

    def test_loader_wires_prior_validation_before_training(self):
        args = config_module.PretrainConfig(
            data_path=str(self.root.parent), dataset_name=self.root.name,
            crop_size=3, original_shape=(7, 4, 6), num_workers=0,
        )
        with self.assertRaisesRegex(ValueError, "exceeding original_shape"):
            dataset_module.setup_pretrain_dataloader(args)


if __name__ == "__main__":
    unittest.main()
