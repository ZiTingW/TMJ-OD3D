import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tmj_dataset_viewer import (
    ValidationIssue,
    build_arg_parser,
    build_dicom_index,
    CLASS_NAMES,
    load_annotations,
    parse_label,
    process_dataset,
    validate_args,
)
from tmj_dataset_read_example import read_sample


DATASET_ROOT = Path(r"E:\tmj")


class LabelParserTests(unittest.TestCase):
    def parse(self, label, expected_side="L"):
        issues = []
        parsed = parse_label(
            label,
            sample_id="TMJTEST",
            json_path=Path("L-label.json"),
            expected_side=expected_side,
            issues=issues,
        )
        return parsed, issues

    def test_canonical_label(self):
        parsed, issues = self.parse("4-L-295-325")
        self.assertEqual(parsed, ((4,), "L", 295, 325, "4-L-295-325"))
        self.assertEqual([i.severity for i in issues], [])

    def test_multi_class_label(self):
        parsed, _ = self.parse("1,3-R-275-305", expected_side="R")
        self.assertEqual(parsed, ((1, 3), "R", 275, 305, "1,3-R-275-305"))

    def test_missing_side_is_inferred(self):
        parsed, issues = self.parse("1,3-275-305", expected_side="R")
        self.assertEqual(parsed, ((1, 3), "R", 275, 305, "1,3-R-275-305"))
        self.assertTrue(any("missing side" in issue.message for issue in issues))

    def test_trailing_character_is_removed(self):
        parsed, issues = self.parse(r"1-L-260-290\\")
        self.assertEqual(parsed, ((1,), "L", 260, 290, "1-L-260-290"))
        self.assertTrue(any("removed trailing" in issue.message for issue in issues))

    def test_valid_label_side_is_kept_when_filename_differs(self):
        parsed, issues = self.parse("1,3-R-285-305", expected_side="L")
        self.assertEqual(parsed, ((1, 3), "R", 285, 305, "1,3-R-285-305"))
        self.assertTrue(any("differs from JSON filename" in issue.message for issue in issues))

    def test_invalid_side_uses_filename(self):
        parsed, issues = self.parse("0-RL-245-275", expected_side="L")
        self.assertEqual(parsed, ((0,), "L", 245, 275, "0-L-245-275"))
        self.assertTrue(any("invalid side token" in issue.message for issue in issues))

    def test_ambiguous_malformed_label_is_skipped(self):
        parsed, issues = self.parse("3-L-28310")
        self.assertIsNone(parsed)
        self.assertTrue(any(issue.severity == "error" for issue in issues))


@unittest.skipUnless(DATASET_ROOT.exists(), "E:\\tmj is not available")
class RealDatasetSmokeTests(unittest.TestCase):
    def test_load_annotations_handles_multiple_and_missing_side_cases(self):
        issues = []
        boxes = load_annotations(DATASET_ROOT / "TMJ00660", issues)
        sides = {box.side for box in boxes}
        self.assertIn("L", sides)
        self.assertIn("R", sides)
        self.assertGreaterEqual(len(boxes), 2)

        missing_issues = []
        missing_boxes = load_annotations(DATASET_ROOT / "TMJ00069", missing_issues)
        self.assertGreaterEqual(len(missing_boxes), 1)
        self.assertTrue(any("not present" in issue.message for issue in missing_issues))

    def test_dicom_index_uses_instance_number(self):
        issues = []
        index = build_dicom_index(DATASET_ROOT / "TMJ00622", issues)
        self.assertIn(315, index)
        self.assertEqual(index[315].path.name, "file_24081315.dcm")

    def test_reference_and_slice_render_create_nonempty_pngs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            parser = build_arg_parser()
            args = parser.parse_args(
                [
                    "--dataset-root",
                    str(DATASET_ROOT),
                    "--sample",
                    "TMJ00622",
                    "--mode",
                    "reference",
                    "--output-dir",
                    tmpdir,
                ]
            )
            validate_args(args)
            results, issues = process_dataset(args)
            self.assertGreaterEqual(len(results), 1)
            self.assertTrue(results[0].output_path.exists())
            self.assertGreater(results[0].output_path.stat().st_size, 1000)

            args = parser.parse_args(
                [
                    "--dataset-root",
                    str(DATASET_ROOT),
                    "--sample",
                    "TMJ00622",
                    "--mode",
                    "slice",
                    "--slice",
                    "315",
                    "--output-dir",
                    tmpdir,
                ]
            )
            validate_args(args)
            results, issues = process_dataset(args)
            self.assertEqual(len(results), 1)
            self.assertTrue(results[0].output_path.exists())
            self.assertGreater(results[0].output_path.stat().st_size, 1000)
            self.assertGreaterEqual(results[0].boxes_drawn, 1)

    def test_slice_outside_available_range_fails_clearly(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            parser = build_arg_parser()
            args = parser.parse_args(
                [
                    "--dataset-root",
                    str(DATASET_ROOT),
                    "--sample",
                    "TMJ00622",
                    "--mode",
                    "slice",
                    "--slice",
                    "9999",
                    "--output-dir",
                    tmpdir,
                ]
            )
            validate_args(args)
            with self.assertRaisesRegex(ValueError, "available range"):
                process_dataset(args)

    def test_read_example_returns_cbct_and_annotation_summary(self):
        summary = read_sample(DATASET_ROOT, "TMJ00622", side="all")
        volume = summary["volume"]
        first_pixels = volume["first_slice_raw_pixels"]
        annotations = summary["annotations"]

        self.assertGreater(volume["dicom_slice_count"], 0)
        self.assertLessEqual(volume["instance_number_min"], volume["instance_number_max"])
        self.assertEqual(len(first_pixels["pixel_array_shape"]), 2)
        self.assertTrue(first_pixels["pixel_array_dtype"])
        self.assertLessEqual(first_pixels["pixel_min"], first_pixels["pixel_max"])

        self.assertGreaterEqual(len(annotations), 1)
        record = annotations[0]
        self.assertIn(record["class_ids"][0], CLASS_NAMES)
        self.assertIn(record["class_names"][0], CLASS_NAMES.values())
        self.assertLessEqual(record["slice_start"], record["slice_end"])
        self.assertEqual(len(record["bbox_xyxy"]), 4)
        self.assertIsNotNone(record["reference_instance_number"])
        self.assertIsNotNone(record["reference_raw_pixels"])


if __name__ == "__main__":
    unittest.main()
