"""CPU checks for explicit, opt-in native and AVMUR-compatible metric helpers."""
import ast
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import nibabel as nib
import torch
from torch import nn

from scripts import brats23_process as processing
from source.benchmark_metrics import (
    OfflineForwardFP32, avmur_region_dice, canonical_mask, native_metrics,
    native_region_metrics, raw_geometry, restore_full_grid,
)
from source.config.masks import MASKS
from source.core.evaluation import evaluate_single_sample


def avmur_reference():
    """Load only the two actual pure-Torch scoring functions, not heavy study imports."""
    root = Path(__file__).resolve().parents[3]
    path = root / "avmur_v11_joint_spatial_fusion_parallel/src/study.py"
    if not path.exists():
        def reference(logits, labels, eps=1e-6):
            def regions(x):
                return torch.stack(((x > 0), ((x == 1) | (x == 3)), (x == 3)), 1).float()
            p, g = regions(logits.argmax(1)), regions(labels)
            dims = tuple(range(2, p.ndim))
            return ((2 * (p * g).sum(dims) + eps) / ((p + g).sum(dims) + eps)).mean(0)
        return reference
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [node for node in tree.body
             if isinstance(node, ast.FunctionDef) and node.name in {"labels_to_regions", "batch_dice"}]
    if len(nodes) != 2:
        raise AssertionError("AVMUR reference functions changed")
    scope = {"torch": torch}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    return scope["batch_dice"]


class BenchmarkMetricTests(unittest.TestCase):
    def test_all_fifteen_masks_are_the_native_permutation(self):
        mapped = {canonical_mask(value) for value in range(1, 16)}
        self.assertEqual(mapped, {tuple(mask) for mask in MASKS})
        self.assertEqual(canonical_mask(1), (False, False, True, False))
        self.assertEqual(canonical_mask(2), (False, True, False, False))
        self.assertEqual(canonical_mask(4), (False, False, False, True))
        self.assertEqual(canonical_mask(8), (True, False, False, False))
        self.assertEqual(canonical_mask(15), (True, True, True, True))
        for bad in (0, 16, -1, True, 1.0, "1"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                canonical_mask(bad)

    def test_region_semantics_and_native_delegation(self):
        pred = torch.tensor([[[1, 2, 3, 0]]])
        target = torch.tensor([[[1, 3, 2, 0]]])
        scores = avmur_region_dice(pred, target)
        self.assertEqual(scores["WT"], 1.0)
        self.assertEqual(scores["TC"], float((torch.tensor(2.) + 1e-6) / (torch.tensor(4.) + 1e-6)))
        self.assertEqual(scores["ET"], float(torch.tensor(1e-6) / (torch.tensor(2.) + 1e-6)))
        actual = native_metrics(pred, target)
        expected = evaluate_single_sample(pred.unsqueeze(0), target.unsqueeze(0))
        for a, b in zip(actual, expected):
            np.testing.assert_array_equal(a, b)
        self.assertEqual(native_region_metrics(pred, target)["WT"], 1.0)

    def test_empty_and_disjoint_exact_avmur_reference(self):
        reference = avmur_reference()
        cases = [
            (torch.zeros((2, 3, 4), dtype=torch.long), torch.zeros((2, 3, 4), dtype=torch.long)),
            (torch.ones((2, 3, 4), dtype=torch.long), torch.zeros((2, 3, 4), dtype=torch.long)),
            (torch.zeros((2, 3, 4), dtype=torch.long), torch.full((2, 3, 4), 3, dtype=torch.long)),
            (torch.tensor([[[3, 0, 1, 2]]]), torch.tensor([[[0, 3, 2, 1]]])),
        ]
        for pred, target in cases:
            logits = torch.nn.functional.one_hot(pred.unsqueeze(0), 4).movedim(-1, 1).float()
            expected = reference(logits, target.unsqueeze(0)).tolist()
            self.assertEqual(list(avmur_region_dice(pred, target).values()), expected)
        empty = cases[0][0]
        self.assertEqual(avmur_region_dice(empty, empty), {"WT": 1., "TC": 1., "ET": 1.})
        self.assertEqual(native_region_metrics(empty, empty),
                         {"WT": 1., "TC": 1., "ET": 1., "ET_postpro": 1.})
        self.assertGreater(avmur_region_dice(*cases[1])["WT"], 0.0)

    def test_invalid_labels_shapes_and_batch_are_rejected(self):
        good = torch.zeros((2, 3, 4), dtype=torch.long)
        for bad in (good.float(), good.bool(), good + 4, good - 1,
                    torch.zeros((2, 2, 3, 4), dtype=torch.long)):
            with self.assertRaises(ValueError):
                avmur_region_dice(bad, good)
        with self.assertRaises(ValueError):
            avmur_region_dice(good, torch.zeros((2, 3, 5), dtype=torch.long))


class GeometryTests(unittest.TestCase):
    def test_numpy_torch_restore_and_padding_removal(self):
        crop = np.arange(2 * 3 * 4, dtype=np.int64).reshape(2, 3, 4)
        padded = np.pad(crop, ((1, 2), (0, 1), (2, 0)), constant_values=99)
        bounds, padding = ((1, 3), (2, 5), (1, 5)), ((1, 2), (0, 1), (2, 0))
        expected = np.zeros((4, 6, 7), dtype=np.int64)
        expected[1:3, 2:5, 1:5] = crop
        for prediction in (padded, torch.from_numpy(padded)):
            actual = restore_full_grid(prediction, expected.shape, bounds, padding)
            np.testing.assert_array_equal(np.asarray(actual), expected)
            self.assertEqual(type(actual), type(prediction))
            self.assertEqual(actual.dtype, prediction.dtype)
        for shape, bad_bounds, bad_padding in (
            ((0, 6, 7), bounds, padding),
            (expected.shape, ((1, 5), (2, 5), (1, 5)), padding),
            (expected.shape, bounds, ((-1, 2), (0, 1), (2, 0))),
            (expected.shape, bounds, ((0, 0), (0, 0), (0, 0))),
        ):
            with self.assertRaises(ValueError):
                restore_full_grid(padded, shape, bad_bounds, bad_padding)

    def test_raw_geometry_exact_mapping_and_outside_gt_is_not_ignored(self):
        volume = np.zeros((4, 5, 5, 5), dtype=np.float32)
        volume[:, 2:4, 2:4, 2:4] = 1
        target = np.zeros((5, 5, 5), dtype=np.uint8)
        target[2, 2, 2] = 4
        target[0, 0, 0] = 3  # Outside native image crop, still a full-grid false negative.
        processed = np.zeros((2, 2, 2), dtype=np.uint8)
        processed[0, 0, 0] = 3
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "seg.npy"
            np.save(path, processed)
            with patch.object(processing, "load_modalities", return_value=(volume, target)):
                geometry = raw_geometry("synthetic", path, min_size=2, check_raw_headers=False)
                np.save(path, processed + 1)
                with self.assertRaisesRegex(ValueError, "exactly match"):
                    raw_geometry("synthetic", path, min_size=2, check_raw_headers=False)
        self.assertEqual(geometry["crop_bounds"], ((2, 4), (2, 4), (2, 4)))
        self.assertEqual(geometry["outside_crop_tumor_voxels"], 1)
        self.assertEqual(int(geometry["target"][2, 2, 2]), 3)
        self.assertEqual(int(geometry["target"][0, 0, 0]), 3)
        restored = restore_full_grid(processed, geometry["source_shape"], geometry["crop_bounds"], geometry["padding"])
        self.assertEqual(avmur_region_dice(processed, processed)["ET"], 1.)
        self.assertLess(avmur_region_dice(restored, geometry["target"])["ET"], 1.)

    def test_raw_geometry_verifies_symmetric_padding(self):
        volume = np.ones((4, 2, 3, 4), dtype=np.float32)
        target = np.ones((2, 3, 4), dtype=np.uint8)
        expected = np.pad(target, ((1, 2), (1, 1), (0, 1)))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "seg.npy"
            np.save(path, expected)
            with patch.object(processing, "load_modalities", return_value=(volume, target)):
                geometry = raw_geometry("synthetic", path, min_size=5, check_raw_headers=False)
        self.assertEqual(geometry["padding"], ((1, 2), (1, 1), (0, 1)))
        restored = restore_full_grid(expected, geometry["source_shape"], geometry["crop_bounds"], geometry["padding"])
        np.testing.assert_array_equal(restored, target)

    def test_restored_short_axis_crop_and_padding_preserve_offcenter_tumor_coordinates(self):
        # Cover a short axis, an axis exactly 128, and a longer axis. The
        # official crop keeps only the foreground interval on the first two;
        # the supplement's padding then places that crop into a 128^3 array.
        volume = np.zeros((4, 7, 128, 129), dtype=np.float32)
        volume[:, 2:6, 13:18, 21:25] = np.arange(1, 5)[:, None, None, None]
        target = np.zeros((7, 128, 129), dtype=np.uint8)
        target[3, 14, 22] = 4  # Legacy ET label; deliberately off-center.
        target[5, 17, 24] = 1
        target[2, 13, 21] = 2
        expected_target = target.copy()
        expected_target[expected_target == 4] = 3

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            seg_path = root / "processed_seg.npy"
            vol_path = root / "processed_vol.npy"
            with patch.object(processing, "load_modalities", return_value=(volume, target)):
                processing.process_case(Path("synthetic"), "processed", root, root, 128, False)
                geometry = raw_geometry(
                    "synthetic", seg_path, min_size=128,
                    processed_vol_path=vol_path, check_raw_headers=False,
                )
            processed = np.load(seg_path)

        # These independent expected offsets catch accidental restoration of
        # the previous whole-short-axis behavior and odd-padding side swaps.
        self.assertEqual(geometry["crop_bounds"], ((2, 6), (13, 18), (0, 128)))
        self.assertEqual(geometry["padding"], ((62, 62), (61, 62), (0, 0)))
        self.assertEqual(geometry["source_shape"], target.shape)
        self.assertTrue(geometry["processed_volume_verified"])
        self.assertEqual(geometry["outside_crop_tumor_voxels"], 0)
        expected_processed = np.pad(
            expected_target[2:6, 13:18, :128], ((62, 62), (61, 62), (0, 0))
        )
        np.testing.assert_array_equal(processed, expected_processed)
        self.assertEqual(int(processed[63, 62, 22]), 3)
        np.testing.assert_array_equal(geometry["target"], expected_target)

        # Predictions in artificial padding must not enter original-grid Dice.
        prediction = processed.copy()
        prediction[0, 0, 0] = prediction[-1, -1, -1] = 3
        for array in (prediction, torch.from_numpy(prediction)):
            with self.subTest(backend=type(array).__name__):
                restored = restore_full_grid(
                    array, geometry["source_shape"], geometry["crop_bounds"], geometry["padding"]
                )
                np.testing.assert_array_equal(np.asarray(restored), expected_target)
                self.assertEqual(
                    avmur_region_dice(restored, expected_target), {"WT": 1., "TC": 1., "ET": 1.}
                )


class RawNiftiGeometryTests(unittest.TestCase):
    def setUp(self):
        self.temporary_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_dir.cleanup)
        self.root = Path(self.temporary_dir.name)
        self.case = self.root / "BraTS-GLI-00001-000"
        self.case.mkdir()
        # Native UniME does not require RAS; consistent LAS is also valid.
        self.affine = np.diag([-1., 1., 1., 1.])
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        self.volume = np.stack((values + 1, values ** 2 + 1,
                                np.sin(values) + 2, np.cos(values) + 2))
        self.target = np.zeros((2, 3, 4), dtype=np.uint8)
        self.target[0, 0, 0] = 4
        for channel, suffix in zip(self.volume, ("t2f", "t1c", "t1n", "t2w")):
            self.write_nifti(suffix, channel)
        self.write_nifti("seg", self.target)
        processing.process_case(self.case, "processed", self.root, self.root, 5, False)
        self.seg_path = self.root / "processed_seg.npy"
        self.vol_path = self.root / "processed_vol.npy"

    def write_nifti(self, suffix, array, affine=None):
        path = self.case / f"{self.case.name}-{suffix}.nii.gz"
        nib.save(nib.Nifti1Image(array, self.affine if affine is None else affine), path)

    def geometry(self):
        return raw_geometry(self.case, self.seg_path, min_size=5,
                            processed_vol_path=self.vol_path)

    def test_verified_native_preprocessing_and_non_ras_geometry(self):
        geometry = self.geometry()
        self.assertTrue(geometry["raw_headers_verified"])
        self.assertTrue(geometry["processed_volume_verified"])
        self.assertEqual(geometry["source_orientation"], ("L", "A", "S"))
        np.testing.assert_array_equal(geometry["source_affine"], self.affine)
        self.assertEqual(int(geometry["target"][0, 0, 0]), 3)

    def test_raw_affine_and_orientation_mismatch_fail_before_native_loading(self):
        translated = self.affine.copy()
        translated[0, 3] = 1
        self.write_nifti("t1c", self.volume[1], translated)
        with patch.object(processing, "load_modalities") as loader:
            with self.assertRaisesRegex(ValueError, "affine/orientation mismatch"):
                self.geometry()
            loader.assert_not_called()
        self.write_nifti("t1c", self.volume[1], np.eye(4))
        with self.assertRaisesRegex(ValueError, "affine/orientation mismatch"):
            self.geometry()

    def test_fractional_negative_nonfinite_and_wrapped_segmentation_values_rejected(self):
        for invalid in (0.5, -1., 256., np.nan, np.inf, 1.0000000001):
            with self.subTest(invalid=invalid):
                target = self.target.astype(np.float64)
                target[0, 1, 0] = invalid
                self.write_nifti("seg", target)
                with patch.object(processing, "load_modalities") as loader:
                    with self.assertRaisesRegex(ValueError, "before casting"):
                        self.geometry()
                    loader.assert_not_called()

    def test_raw_spatial_shape_mismatch_is_rejected(self):
        self.write_nifti("t2w", self.volume[3, :, :, :3])
        with self.assertRaisesRegex(ValueError, "shape/affine/orientation mismatch"):
            self.geometry()

    def test_processed_channel_order_normalization_and_dtype_are_verified(self):
        correct = np.load(self.vol_path)
        for bad in (correct[..., [1, 0, 2, 3]], correct + np.float32(0.01), correct.astype(np.float64)):
            with self.subTest(dtype=bad.dtype):
                np.save(self.vol_path, bad)
                with self.assertRaisesRegex(ValueError, "Processed volume does not exactly match"):
                    self.geometry()


class ForwardPrecisionTests(unittest.TestCase):
    def test_model_only_autocast_and_output_conversion(self):
        events = []

        class Context:
            def __enter__(self):
                events.append("enter")

            def __exit__(self, *args):
                events.append("exit")

        class Model(nn.Module):
            def forward(self, x):
                self.assert_active = events[-1] == "enter"
                events.append("model")
                return {"logits": x.half(), "extra": (x.bfloat16(), torch.tensor(1))}

        model = Model()
        wrapper = OfflineForwardFP32(model, dtype=torch.float16)
        with patch("source.benchmark_metrics.torch.autocast", return_value=Context()) as autocast:
            result = wrapper(torch.ones((1, 4, 2, 2, 2)))
        self.assertTrue(model.assert_active)
        self.assertEqual(events, ["enter", "model", "exit"])
        autocast.assert_called_once_with(device_type="cuda", dtype=torch.float16, enabled=True)
        self.assertEqual(result["logits"].dtype, torch.float32)
        self.assertEqual(result["extra"][0].dtype, torch.float32)
        self.assertEqual(result["extra"][1].dtype, torch.int64)
        self.assertEqual(torch.softmax(result["logits"], dim=1).dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
