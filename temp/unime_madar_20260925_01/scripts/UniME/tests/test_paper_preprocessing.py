"""Checks for paper preprocessing and explicitly retained implementation choices."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts import brats23_process as processing


class PaperPreprocessingTests(unittest.TestCase):
    def test_official_crop_retains_foreground_interval_on_short_axis(self):
        self.assertEqual(processing.sup_128(20, 100, 128, 127), (20, 100))

    def test_official_crop_retains_foreground_interval_on_exact_128_axis(self):
        self.assertEqual(processing.sup_128(30, 115, 128, 128), (30, 115))

    def test_crop_bounds_use_foreground_on_short_and_exact_128_axes(self):
        mask = np.zeros((127, 128, 129), dtype=bool)
        mask[20:100, 30:115, 5:124] = True
        bounds = processing.find_crop_bounds(mask, 128)
        self.assertEqual(bounds, (slice(20, 100), slice(30, 115), slice(1, 129)))

    def test_normalization_uses_union_including_negative_and_cancelling_values(self):
        # The first two nonzero positions have channel sums of zero and -1.
        volume = np.array(
            [[2, -2, 4, 0], [-2, 1, 3, 0]], dtype=np.float32
        ).reshape(2, 1, 1, 4)
        normalized = processing.normalize_channels(volume)
        for channel in normalized:
            brain = channel[0, 0, :3]
            self.assertAlmostEqual(float(brain.mean()), 0.0, places=6)
            self.assertAlmostEqual(float(brain.std()), 1.0, places=6)
        np.testing.assert_array_equal(volume[:, 0, 0, 0], [2, -2])

    def test_padding_preserves_voxels_labels_and_centers_short_axes(self):
        volume = np.arange(2 * 3 * 5 * 4, dtype=np.float32).reshape(2, 3, 5, 4)
        labels = (np.arange(2 * 3 * 5).reshape(2, 3, 5) % 4).astype(np.uint8)
        padded_volume, padded_labels = processing.pad_to_min_size(volume, labels, 6)
        self.assertEqual(padded_volume.shape, (6, 6, 6, 4))
        self.assertEqual(padded_labels.shape, (6, 6, 6))
        # Deficits 4/3/1 are split 2+2, 1+2, and 0+1.
        original_region = (slice(2, 4), slice(1, 4), slice(0, 5))
        np.testing.assert_array_equal(padded_volume[original_region], volume)
        np.testing.assert_array_equal(padded_labels[original_region], labels)
        padded_volume[original_region] = 0
        padded_labels[original_region] = 0
        self.assertFalse(np.any(padded_volume))
        self.assertFalse(np.any(padded_labels))
        self.assertEqual(padded_volume.dtype, np.float32)
        self.assertEqual(padded_labels.dtype, np.uint8)

    def test_processing_crops_foreground_then_pads_short_axes_without_changing_labels(self):
        volume = np.zeros((4, 127, 128, 129), dtype=np.float32)
        volume[:, 1:126, 1:127, 1:128] = np.arange(1, 5)[:, None, None, None]
        labels = np.zeros((127, 128, 129), dtype=np.uint8)
        labels[40, 50, 60] = 1
        labels[41, 50, 60] = 2
        labels[42, 50, 60] = 3
        labels[43, 50, 60] = 4  # Existing legacy BraTS conversion remains intact.
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir)
            with patch.object(processing, "load_modalities", return_value=(volume, labels)):
                processing.process_case(
                    Path("synthetic-case"), "case", output_dir, output_dir, 128, False
                )
            actual_volume = np.load(output_dir / "case_vol.npy")
            actual_labels = np.load(output_dir / "case_seg.npy")
        self.assertEqual(actual_volume.shape, (128, 128, 128, 4))
        # Official short-axis cropping keeps 1:126 and 1:127. Retained local
        # padding adds (1, 2) and (1, 1), respectively, after normalization.
        expected_labels = np.zeros((128, 128, 128), dtype=np.uint8)
        expected_labels[1:126, 1:127] = labels[1:126, 1:127, 1:129]
        expected_labels[expected_labels == 4] = 3
        np.testing.assert_array_equal(actual_labels, expected_labels)
        cropped = volume[:, 1:126, 1:127, 1:129]
        normalized = processing.normalize_channels(cropped).transpose(1, 2, 3, 0)
        expected_volume = np.pad(normalized, ((1, 2), (1, 1), (0, 0), (0, 0)))
        np.testing.assert_array_equal(actual_volume, expected_volume)
        self.assertFalse(np.any(actual_volume[0]))
        self.assertFalse(np.any(actual_volume[126:]))
        self.assertFalse(np.any(actual_volume[:, 0]))
        self.assertFalse(np.any(actual_volume[:, 127]))
        self.assertFalse(np.any(actual_labels[0]))
        self.assertFalse(np.any(actual_labels[126:]))


class ExistingSplitTests(unittest.TestCase):
    def setUp(self):
        self.temporary_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_dir.cleanup)
        self.root = Path(self.temporary_dir.name)
        self.split_json = self.root / "split.json"
        self.mapping = {
            "case_0": "BraTS-GLI-00001-000",
            "case_1": "BraTS-GLI-00001-001",
            "case_2": "BraTS-GLI-00002-000",
            "case_3": "BraTS-GLI-00003-000",
        }
        self.source_splits = {
            "train": [self.mapping["case_1"], self.mapping["case_0"]],
            "val": [self.mapping["case_2"]],
            "test": [self.mapping["case_3"]],
        }

    def write_source(self, source):
        self.split_json.write_text(json.dumps(source), encoding="utf-8")

    def test_preserves_memberships_order_and_repeated_patient_scans(self):
        self.write_source(self.source_splits)
        splits = processing.load_existing_splits(self.split_json, self.mapping)
        processing.write_existing_splits(self.root, splits)
        self.assertEqual((self.root / "train.txt").read_text(), "case_1\ncase_0\n")
        self.assertEqual((self.root / "val.txt").read_text(), "case_2\n")
        self.assertEqual((self.root / "test.txt").read_text(), "case_3\n")

    def test_rejects_patient_leakage_across_partitions(self):
        source = {
            "train": [self.mapping["case_0"]],
            "val": [self.mapping["case_1"], self.mapping["case_2"]],
            "test": [self.mapping["case_3"]],
        }
        self.write_source(source)
        with self.assertRaisesRegex(ValueError, "Patient overlap"):
            processing.load_existing_splits(self.split_json, self.mapping)

    def test_rejects_malformed_lists_duplicates_and_missing_coverage(self):
        invalid = [
            ({**self.source_splits, "val": "case_2"}, "nonempty list"),
            ({**self.source_splits, "test": []}, "nonempty list"),
            ({**self.source_splits, "train": [None]}, "invalid source case"),
            ({**self.source_splits, "train": [self.mapping["case_0"]] * 2}, "Duplicate case"),
            ({**self.source_splits, "train": [self.mapping["case_0"]]}, "exactly cover"),
        ]
        for source, error in invalid:
            with self.subTest(error=error):
                self.write_source(source)
                with self.assertRaisesRegex(ValueError, error):
                    processing.load_existing_splits(self.split_json, self.mapping)

    def test_partial_limit_fails_before_processing_or_output_creation(self):
        raw_dir = self.root / "raw"
        raw_dir.mkdir()
        for case in self.mapping.values():
            (raw_dir / case).mkdir()
        self.write_source(self.source_splits)
        out_dir = self.root / "processed"
        argv = [
            "brats23_process.py", "--raw-dir", str(raw_dir), "--out-dir", str(out_dir),
            "--split-json", str(self.split_json), "--limit", "2",
        ]
        with patch("sys.argv", argv), patch.object(processing, "process_case") as process_case:
            with self.assertRaisesRegex(ValueError, "partial --limit"):
                processing.main()
            process_case.assert_not_called()
        self.assertFalse(out_dir.exists())

    def test_split_json_and_skip_splits_are_mutually_exclusive(self):
        argv = [
            "brats23_process.py", "--raw-dir", str(self.root), "--split-json",
            str(self.split_json), "--skip-splits",
        ]
        with patch("sys.argv", argv), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                processing.parse_args()
        self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
