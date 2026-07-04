#!/usr/bin/env python
"""Example: read one TMJ CBCT sample and its 3D bounding-box labels.

This script is intentionally small and explicit. It demonstrates how to:
1. sort the DICOM series by InstanceNumber,
2. read raw CBCT pixel data with pydicom,
3. parse side-specific JSON labels into 3D boxes,
4. connect each box to its reference DICOM slice.

Example:
    conda run -n base python tmj_dataset_read_example.py ^
        --dataset-root E:\tmj ^
        --sample TMJ00622 ^
        --side all ^
        --output-json sample_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from tmj_dataset_viewer import (
    CLASS_NAMES,
    AnnotationBox,
    DicomSlice,
    ValidationIssue,
    build_dicom_index,
    filter_boxes,
    find_dicom_by_name,
    import_pydicom,
    load_annotations,
    require_dependencies,
)


def number_or_none(value):
    if value is None:
        return None
    try:
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, float):
            return float(value)
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return str(value)


def list_or_none(value):
    if value is None:
        return None
    if isinstance(value, str):
        return value
    try:
        return [number_or_none(item) for item in value]
    except TypeError:
        return str(value)


def read_raw_pixel_summary(dicom_path: Path) -> Dict[str, object]:
    """Read original DICOM pixels and return simple raw-data statistics."""
    pydicom = import_pydicom()
    ds = pydicom.dcmread(str(dicom_path), force=True)
    array = ds.pixel_array
    return {
        "dicom_path": str(dicom_path),
        "instance_number": int(getattr(ds, "InstanceNumber")),
        "pixel_array_shape": list(array.shape),
        "pixel_array_dtype": str(array.dtype),
        "pixel_min": number_or_none(array.min()),
        "pixel_max": number_or_none(array.max()),
        "rescale_slope": number_or_none(getattr(ds, "RescaleSlope", None)),
        "rescale_intercept": number_or_none(getattr(ds, "RescaleIntercept", None)),
        "rows": number_or_none(getattr(ds, "Rows", None)),
        "columns": number_or_none(getattr(ds, "Columns", None)),
        "pixel_spacing": list_or_none(getattr(ds, "PixelSpacing", None)),
        "slice_thickness": number_or_none(getattr(ds, "SliceThickness", None)),
        "photometric_interpretation": str(getattr(ds, "PhotometricInterpretation", "")),
    }


def summarize_volume(sample_dir: Path, dicom_index: Dict[int, DicomSlice]) -> Dict[str, object]:
    instance_numbers = sorted(dicom_index)
    if not instance_numbers:
        raise ValueError(f"No readable DICOM slices found in {sample_dir}")
    first_slice = dicom_index[instance_numbers[0]]
    first_pixels = read_raw_pixel_summary(first_slice.path)
    return {
        "sample_id": sample_dir.name,
        "sample_dir": str(sample_dir),
        "dicom_slice_count": len(instance_numbers),
        "instance_number_min": instance_numbers[0],
        "instance_number_max": instance_numbers[-1],
        "first_dicom": str(first_slice.path),
        "first_slice_raw_pixels": first_pixels,
    }


def annotation_to_record(
    sample_dir: Path,
    box: AnnotationBox,
    dicom_index: Dict[int, DicomSlice],
    pixel_cache: Dict[Path, Dict[str, object]],
) -> Dict[str, object]:
    reference_slice: Optional[DicomSlice] = None
    if box.image_path:
        reference_slice = find_dicom_by_name(sample_dir, box.image_path, dicom_index)

    pixel_summary = None
    if reference_slice is not None:
        if reference_slice.path not in pixel_cache:
            pixel_cache[reference_slice.path] = read_raw_pixel_summary(reference_slice.path)
        pixel_summary = pixel_cache[reference_slice.path]

    x0, y0, x1, y1 = box.bbox_xyxy
    class_names = [CLASS_NAMES.get(class_id, str(class_id)) for class_id in box.classes]
    return {
        "sample_id": box.sample_id,
        "side": box.side,
        "class_ids": list(box.classes),
        "class_names": class_names,
        "class_id": ",".join(str(class_id) for class_id in box.classes),
        "class_name": "+".join(class_names),
        "slice_start": box.slice_start,
        "slice_end": box.slice_end,
        "bbox_xyxy": [x0, y0, x1, y1],
        "points": [list(point) for point in box.points],
        "reference_dicom": str(reference_slice.path) if reference_slice else None,
        "reference_instance_number": reference_slice.instance_number if reference_slice else None,
        "reference_raw_pixels": pixel_summary,
        "json_path": str(box.json_path),
        "raw_label": box.raw_label,
        "normalized_label": box.normalized_label,
    }


def read_sample(dataset_root: Path, sample_id: str, side: str = "all") -> Dict[str, object]:
    require_dependencies()
    sample_dir = dataset_root / sample_id
    if not sample_dir.is_dir():
        raise FileNotFoundError(f"Sample folder not found: {sample_dir}")

    issues: List[ValidationIssue] = []
    dicom_index = build_dicom_index(sample_dir, issues)
    volume = summarize_volume(sample_dir, dicom_index)
    boxes = load_annotations(sample_dir, issues)
    boxes = filter_boxes(boxes, side=side, class_filter=None)

    pixel_cache: Dict[Path, Dict[str, object]] = {}
    annotations = [
        annotation_to_record(sample_dir, box, dicom_index, pixel_cache)
        for box in boxes
    ]
    return {
        "volume": volume,
        "annotations": annotations,
        "validation_issues": [
            {
                "severity": issue.severity,
                "sample_id": issue.sample_id,
                "source": issue.source,
                "message": issue.message,
                "raw_label": issue.raw_label,
            }
            for issue in issues
        ],
    }


def print_summary(summary: Dict[str, object]) -> None:
    volume = summary["volume"]
    first_pixels = volume["first_slice_raw_pixels"]
    print("CBCT volume")
    print(f"  sample_id: {volume['sample_id']}")
    print(f"  sample_dir: {volume['sample_dir']}")
    print(f"  dicom_slice_count: {volume['dicom_slice_count']}")
    print(f"  instance_number_range: {volume['instance_number_min']} - {volume['instance_number_max']}")
    print(
        "  first_slice_pixels: "
        f"shape={first_pixels['pixel_array_shape']} "
        f"dtype={first_pixels['pixel_array_dtype']} "
        f"min={first_pixels['pixel_min']} "
        f"max={first_pixels['pixel_max']}"
    )
    print()
    print("Annotations")
    header = [
        "sample_id",
        "side",
        "class_id",
        "class_name",
        "slice_start",
        "slice_end",
        "reference_instance_number",
        "bbox_xyxy",
        "reference_dicom",
    ]
    print("\t".join(header))
    for record in summary["annotations"]:
        row = [
            str(record["sample_id"]),
            str(record["side"]),
            str(record["class_id"]),
            str(record["class_name"]),
            str(record["slice_start"]),
            str(record["slice_end"]),
            str(record["reference_instance_number"]),
            json.dumps(record["bbox_xyxy"]),
            Path(str(record["reference_dicom"])).name if record["reference_dicom"] else "",
        ]
        print("\t".join(row))

    issues = summary["validation_issues"]
    print()
    print(f"Validation issues: {len(issues)}")
    for issue in issues[:10]:
        print(f"  [{issue['severity']}] {issue['source']}: {issue['message']} {issue['raw_label']}")
    if len(issues) > 10:
        print(f"  ... {len(issues) - 10} more")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read one TMJ CBCT sample and print raw DICOM plus parsed annotation-box information.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-root", type=Path, required=True, help="Dataset root containing TMJ* folders.")
    parser.add_argument("--sample", required=True, help="Sample id, for example TMJ00622.")
    parser.add_argument("--side", choices=["L", "R", "all"], default="all", help="Annotation side filter.")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional JSON output path.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        summary = read_sample(args.dataset_root, args.sample, args.side)
        print_summary(summary)
        if args.output_json is not None:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            with args.output_json.open("w", encoding="utf-8") as handle:
                json.dump(summary, handle, indent=2)
            print(f"\nWrote JSON: {args.output_json}")
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
