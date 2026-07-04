#!/usr/bin/env python
"""TMJ CBCT dataset reader and 3D bounding-box visualizer.

Example:
    conda run -n base python tmj_dataset_viewer.py ^
        --dataset-root E:\tmj ^
        --sample TMJ00622 ^
        --mode slice --slice 315 ^
        --side all ^
        --output-dir tmj_preview
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


CLASS_NAMES = {
    0: "normal",
    1: "cortical erosion",
    2: "subchondral sclerosis",
    3: "subchondral cystic changes",
    4: "condylar flattening",
    5: "osteophyte",
    6: "other",
}

CLASS_COLORS = {
    0: (0, 180, 255),
    1: (255, 80, 80),
    2: (255, 185, 60),
    3: (190, 90, 255),
    4: (80, 220, 120),
    5: (255, 110, 210),
    6: (80, 180, 255),
}


@dataclass
class ValidationIssue:
    sample_id: str
    source: str
    severity: str
    message: str
    raw_label: str = ""


@dataclass
class DicomSlice:
    instance_number: int
    path: Path
    rows: Optional[int] = None
    columns: Optional[int] = None


@dataclass
class AnnotationBox:
    sample_id: str
    json_path: Path
    json_side: str
    side: str
    classes: Tuple[int, ...]
    slice_start: int
    slice_end: int
    points: Tuple[Tuple[float, float], Tuple[float, float]]
    image_path: Optional[str]
    width: Optional[int]
    height: Optional[int]
    shape_index: int
    raw_label: str
    normalized_label: str

    @property
    def bbox_xyxy(self) -> Tuple[float, float, float, float]:
        (x0, y0), (x1, y1) = self.points
        return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)

    @property
    def class_text(self) -> str:
        return "+".join(CLASS_NAMES.get(c, str(c)) for c in self.classes)

    def contains_slice(self, instance_number: int) -> bool:
        return self.slice_start <= instance_number <= self.slice_end


@dataclass
class RenderResult:
    sample_id: str
    output_path: Path
    mode: str
    instance_number: int
    dicom_path: Path
    boxes_drawn: int


def require_dependencies() -> None:
    missing = []
    for module in ("pydicom", "numpy", "PIL"):
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    if missing:
        raise RuntimeError(
            "Missing required Python packages: "
            + ", ".join(missing)
            + ". Install them or run with an environment that provides them, "
            "for example: conda run -n base python tmj_dataset_viewer.py ..."
        )


def add_issue(
    issues: List[ValidationIssue],
    sample_id: str,
    source: Path | str,
    severity: str,
    message: str,
    raw_label: str = "",
) -> None:
    issues.append(
        ValidationIssue(
            sample_id=sample_id,
            source=str(source),
            severity=severity,
            message=message,
            raw_label=raw_label,
        )
    )


def parse_class_filter(text: Optional[str]) -> Optional[Set[int]]:
    if not text:
        return None
    values: Set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if not part.isdigit():
            raise argparse.ArgumentTypeError(f"Invalid class id in --class-filter: {part!r}")
        values.add(int(part))
    return values


def class_filter_matches(box: AnnotationBox, class_filter: Optional[Set[int]]) -> bool:
    return class_filter is None or bool(set(box.classes) & class_filter)


def normalize_label_text(raw_label: str) -> Tuple[str, List[str]]:
    warnings: List[str] = []
    label = str(raw_label).strip()
    normalized = (
        label.replace("_", "-")
        .replace("\u2010", "-")
        .replace("\u2011", "-")
        .replace("\u2012", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2212", "-")
    )
    stripped = re.sub(r"[^0-9A-Za-z,\-\s]+$", "", normalized)
    if stripped != normalized:
        warnings.append(f"removed trailing non-label characters: {normalized!r} -> {stripped!r}")
        normalized = stripped
    normalized = re.sub(r"\s+", "-", normalized.strip())
    normalized = re.sub(r"-+", "-", normalized)
    return normalized, warnings


def parse_label(
    raw_label: str,
    *,
    sample_id: str = "",
    json_path: Path | str = "",
    expected_side: Optional[str] = None,
    issues: Optional[List[ValidationIssue]] = None,
) -> Optional[Tuple[Tuple[int, ...], str, int, int, str]]:
    """Parse one LabelMe shape label.

    Returns (classes, side, slice_start, slice_end, normalized_label), or None
    when the label is ambiguous and should be skipped.
    """
    local_issues: List[ValidationIssue] = issues if issues is not None else []
    source = json_path or "<label>"
    normalized, warnings = normalize_label_text(raw_label)
    for warning in warnings:
        add_issue(local_issues, sample_id, source, "warning", warning, raw_label)

    parts = [p for p in re.split(r"[-\s]+", normalized) if p]
    class_part: Optional[str] = None
    side_part: Optional[str] = None
    start_part: Optional[str] = None
    end_part: Optional[str] = None

    if len(parts) == 4:
        class_part, side_part, start_part, end_part = parts
    elif len(parts) == 3:
        if parts[1].upper() in {"L", "R", "LR", "RL"}:
            add_issue(
                local_issues,
                sample_id,
                source,
                "error",
                "label has a side token but does not have separate slice_start and slice_end fields",
                raw_label,
            )
            return None
        class_part, start_part, end_part = parts
        if not expected_side:
            add_issue(
                local_issues,
                sample_id,
                source,
                "error",
                "label is missing side and no JSON filename side is available",
                raw_label,
            )
            return None
        side_part = expected_side
        add_issue(
            local_issues,
            sample_id,
            source,
            "warning",
            f"label is missing side; inferred side {expected_side} from JSON filename",
            raw_label,
        )
    else:
        add_issue(
            local_issues,
            sample_id,
            source,
            "error",
            f"label does not match class-side-start-end format after normalization: {normalized!r}",
            raw_label,
        )
        return None

    assert class_part is not None and side_part is not None
    assert start_part is not None and end_part is not None

    if not re.fullmatch(r"\d+(?:,\d+)*", class_part):
        add_issue(local_issues, sample_id, source, "error", "class field is malformed", raw_label)
        return None

    classes = tuple(int(value) for value in class_part.split(","))
    for class_id in classes:
        if class_id not in CLASS_NAMES:
            add_issue(
                local_issues,
                sample_id,
                source,
                "warning",
                f"class id {class_id} is outside the documented 0-6 range",
                raw_label,
            )

    side_token = side_part.upper()
    if side_token in {"L", "R"}:
        side = side_token
        if expected_side and side != expected_side:
            add_issue(
                local_issues,
                sample_id,
                source,
                "warning",
                f"label side {side} differs from JSON filename side {expected_side}; using label side",
                raw_label,
            )
    elif expected_side and expected_side in side_token:
        side = expected_side
        add_issue(
            local_issues,
            sample_id,
            source,
            "warning",
            f"invalid side token {side_token!r}; inferred side {expected_side} from JSON filename",
            raw_label,
        )
    elif expected_side:
        side = expected_side
        add_issue(
            local_issues,
            sample_id,
            source,
            "warning",
            f"invalid side token {side_token!r}; inferred side {expected_side} from JSON filename",
            raw_label,
        )
    else:
        add_issue(local_issues, sample_id, source, "error", "side field is malformed", raw_label)
        return None

    if not start_part.isdigit() or not end_part.isdigit():
        add_issue(local_issues, sample_id, source, "error", "slice range is malformed", raw_label)
        return None

    slice_start = int(start_part)
    slice_end = int(end_part)
    if slice_start > slice_end:
        add_issue(
            local_issues,
            sample_id,
            source,
            "warning",
            f"slice_start {slice_start} is greater than slice_end {slice_end}; values were swapped",
            raw_label,
        )
        slice_start, slice_end = slice_end, slice_start

    normalized_label = f"{class_part}-{side}-{slice_start}-{slice_end}"
    return classes, side, slice_start, slice_end, normalized_label


def parse_points(points: object) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    if not isinstance(points, list) or len(points) != 2:
        return None
    parsed: List[Tuple[float, float]] = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            return None
        try:
            parsed.append((float(point[0]), float(point[1])))
        except (TypeError, ValueError):
            return None
    return parsed[0], parsed[1]


def iter_shape_records(annotation: dict) -> Iterable[Tuple[dict, dict, int]]:
    shape_index = 0
    for shape in annotation.get("shapes") or []:
        yield annotation, shape, shape_index
        shape_index += 1
    for frame in annotation.get("frames") or []:
        for shape in frame.get("shapes") or []:
            yield frame, shape, shape_index
            shape_index += 1


def load_annotations(sample_dir: Path, issues: List[ValidationIssue]) -> List[AnnotationBox]:
    sample_id = sample_dir.name
    boxes: List[AnnotationBox] = []
    json_paths = sorted(sample_dir.glob("*-label.json"))
    json_paths.extend(path for path in sorted(sample_dir.glob("* label.json")) if path not in json_paths)

    if not json_paths:
        add_issue(issues, sample_id, sample_dir, "info", "no side-specific JSON annotation files found")
        return boxes

    seen_standard_sides = set()
    for json_path in json_paths:
        match = re.match(r"^([LR])[- ]label\.json$", json_path.name, flags=re.IGNORECASE)
        expected_side = match.group(1).upper() if match else None
        if expected_side:
            seen_standard_sides.add(expected_side)
        else:
            add_issue(
                issues,
                sample_id,
                json_path,
                "warning",
                "annotation filename does not match L-label.json or R-label.json",
            )

        try:
            with json_path.open("r", encoding="utf-8") as handle:
                annotation = json.load(handle)
        except Exception as exc:
            add_issue(issues, sample_id, json_path, "error", f"failed to read JSON: {exc}")
            continue

        root_image_path = annotation.get("image_path")
        for container, shape, shape_index in iter_shape_records(annotation):
            raw_label = str(shape.get("label", ""))
            parsed_label = parse_label(
                raw_label,
                sample_id=sample_id,
                json_path=json_path,
                expected_side=expected_side,
                issues=issues,
            )
            if parsed_label is None:
                continue

            points = parse_points(shape.get("points"))
            if points is None:
                add_issue(issues, sample_id, json_path, "error", "shape points are malformed", raw_label)
                continue

            shape_type = shape.get("shape_type")
            if shape_type and shape_type != "rectangle":
                add_issue(
                    issues,
                    sample_id,
                    json_path,
                    "warning",
                    f"shape_type is {shape_type!r}, expected 'rectangle'",
                    raw_label,
                )

            classes, side, slice_start, slice_end, normalized_label = parsed_label
            boxes.append(
                AnnotationBox(
                    sample_id=sample_id,
                    json_path=json_path,
                    json_side=expected_side or "",
                    side=side,
                    classes=classes,
                    slice_start=slice_start,
                    slice_end=slice_end,
                    points=points,
                    image_path=container.get("image_path") or root_image_path,
                    width=container.get("width") or annotation.get("imageWidth"),
                    height=container.get("height") or annotation.get("imageHeight"),
                    shape_index=shape_index,
                    raw_label=raw_label,
                    normalized_label=normalized_label,
                )
            )

    for side in ("L", "R"):
        if side not in seen_standard_sides:
            add_issue(
                issues,
                sample_id,
                sample_dir / f"{side}-label.json",
                "info",
                f"{side}-label.json is not present; this is allowed for one-sided annotations",
            )
    return boxes


def import_pydicom():
    try:
        import pydicom  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "pydicom is required to read DICOM files. "
            "Try: conda run -n base python tmj_dataset_viewer.py ..."
        ) from exc
    return pydicom


def build_dicom_index(sample_dir: Path, issues: List[ValidationIssue]) -> Dict[int, DicomSlice]:
    pydicom = import_pydicom()
    sample_id = sample_dir.name
    index: Dict[int, DicomSlice] = {}
    for dicom_path in sorted(sample_dir.glob("*.dcm")):
        try:
            ds = pydicom.dcmread(str(dicom_path), stop_before_pixels=True, force=True)
        except Exception as exc:
            add_issue(issues, sample_id, dicom_path, "error", f"failed to read DICOM header: {exc}")
            continue
        instance_number = getattr(ds, "InstanceNumber", None)
        if instance_number is None:
            add_issue(issues, sample_id, dicom_path, "error", "DICOM has no InstanceNumber")
            continue
        try:
            instance_number_int = int(instance_number)
        except (TypeError, ValueError):
            add_issue(issues, sample_id, dicom_path, "error", f"invalid InstanceNumber: {instance_number!r}")
            continue
        if instance_number_int in index:
            add_issue(
                issues,
                sample_id,
                dicom_path,
                "warning",
                f"duplicate InstanceNumber {instance_number_int}; keeping first file {index[instance_number_int].path.name}",
            )
            continue
        index[instance_number_int] = DicomSlice(
            instance_number=instance_number_int,
            path=dicom_path,
            rows=int(getattr(ds, "Rows", 0) or 0) or None,
            columns=int(getattr(ds, "Columns", 0) or 0) or None,
        )
    if not index:
        add_issue(issues, sample_id, sample_dir, "error", "no readable DICOM slices found")
    return dict(sorted(index.items()))


def find_dicom_by_name(sample_dir: Path, image_path: str, index: Dict[int, DicomSlice]) -> Optional[DicomSlice]:
    wanted = Path(image_path).name
    for dicom_slice in index.values():
        if dicom_slice.path.name == wanted:
            return dicom_slice
    direct = sample_dir / wanted
    if direct.exists():
        pydicom = import_pydicom()
        ds = pydicom.dcmread(str(direct), stop_before_pixels=True, force=True)
        return DicomSlice(instance_number=int(ds.InstanceNumber), path=direct)
    return None


def read_dicom_as_uint8(dicom_path: Path, lower_percentile: float, upper_percentile: float):
    import numpy as np
    from PIL import Image

    pydicom = import_pydicom()
    ds = pydicom.dcmread(str(dicom_path), force=True)
    array = ds.pixel_array.astype("float32")
    slope = float(getattr(ds, "RescaleSlope", 1) or 1)
    intercept = float(getattr(ds, "RescaleIntercept", 0) or 0)
    array = array * slope + intercept

    finite = array[np.isfinite(array)]
    if finite.size == 0:
        raise ValueError(f"DICOM pixel array has no finite values: {dicom_path}")
    low, high = np.percentile(finite, [lower_percentile, upper_percentile])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(finite.min())
        high = float(finite.max())
    if high <= low:
        high = low + 1.0

    windowed = np.clip((array - low) / (high - low), 0.0, 1.0)
    image_array = (windowed * 255.0).astype("uint8")
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        image_array = 255 - image_array
    return Image.fromarray(image_array, mode="L").convert("RGB"), int(getattr(ds, "InstanceNumber", -1))


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def draw_boxes(image, boxes: Sequence[AnnotationBox], issues: List[ValidationIssue], dicom_path: Path) -> None:
    from PIL import ImageDraw, ImageFont

    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    image_width, image_height = image.size

    for box in boxes:
        x0, y0, x1, y1 = box.bbox_xyxy
        if box.width and box.height and (box.width != image_width or box.height != image_height):
            add_issue(
                issues,
                box.sample_id,
                box.json_path,
                "warning",
                f"annotation size {box.width}x{box.height} differs from DICOM size {image_width}x{image_height}; coordinates were scaled",
                box.raw_label,
            )
            x_scale = image_width / float(box.width)
            y_scale = image_height / float(box.height)
            x0, x1 = x0 * x_scale, x1 * x_scale
            y0, y1 = y0 * y_scale, y1 * y_scale

        if x1 < 0 or y1 < 0 or x0 >= image_width or y0 >= image_height:
            add_issue(
                issues,
                box.sample_id,
                box.json_path,
                "warning",
                f"box is outside image bounds for {dicom_path.name}",
                box.raw_label,
            )
            continue

        x0 = clamp(x0, 0, image_width - 1)
        x1 = clamp(x1, 0, image_width - 1)
        y0 = clamp(y0, 0, image_height - 1)
        y1 = clamp(y1, 0, image_height - 1)

        color = CLASS_COLORS.get(box.classes[0], (255, 255, 0))
        for offset in range(3):
            draw.rectangle([x0 - offset, y0 - offset, x1 + offset, y1 + offset], outline=color)

        label = f"{box.class_text} {box.side} {box.slice_start}-{box.slice_end}"
        text_x = int(x0)
        text_y = max(0, int(y0) - 14)
        try:
            text_bbox = draw.textbbox((text_x, text_y), label, font=font)
        except (AttributeError, ValueError):
            text_width, text_height = draw.textsize(label, font=font)
            text_bbox = (text_x, text_y, text_x + text_width, text_y + text_height)
        if text_bbox[2] >= image_width:
            shift = text_bbox[2] - image_width + 2
            text_x = max(0, text_x - shift)
            try:
                text_bbox = draw.textbbox((text_x, text_y), label, font=font)
            except (AttributeError, ValueError):
                text_width, text_height = draw.textsize(label, font=font)
                text_bbox = (text_x, text_y, text_x + text_width, text_y + text_height)
        draw.rectangle(text_bbox, fill=(0, 0, 0))
        draw.text((text_x, text_y), label, fill=color, font=font)


def sanitize_filename(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def render_image(
    *,
    sample_id: str,
    dicom_path: Path,
    output_path: Path,
    boxes: Sequence[AnnotationBox],
    issues: List[ValidationIssue],
    mode: str,
    lower_percentile: float,
    upper_percentile: float,
) -> RenderResult:
    image, instance_number = read_dicom_as_uint8(dicom_path, lower_percentile, upper_percentile)
    draw_boxes(image, boxes, issues, dicom_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return RenderResult(
        sample_id=sample_id,
        output_path=output_path,
        mode=mode,
        instance_number=instance_number,
        dicom_path=dicom_path,
        boxes_drawn=len(boxes),
    )


def select_samples(dataset_root: Path, sample: Optional[str], all_samples: bool) -> List[Path]:
    if sample and all_samples:
        raise ValueError("Use either --sample or --all, not both")
    if not sample and not all_samples:
        raise ValueError("Specify --sample TMJxxxxx or --all")
    if sample:
        sample_dir = dataset_root / sample
        if not sample_dir.is_dir():
            raise FileNotFoundError(f"Sample folder not found: {sample_dir}")
        return [sample_dir]
    sample_dirs = [path for path in sorted(dataset_root.glob("TMJ*")) if path.is_dir()]
    if not sample_dirs:
        raise FileNotFoundError(f"No TMJ* sample folders found under {dataset_root}")
    return sample_dirs


def filter_boxes(
    boxes: Sequence[AnnotationBox],
    *,
    side: str,
    class_filter: Optional[Set[int]],
) -> List[AnnotationBox]:
    selected: List[AnnotationBox] = []
    for box in boxes:
        if side != "all" and box.side != side:
            continue
        if not class_filter_matches(box, class_filter):
            continue
        selected.append(box)
    return selected


def render_sample_reference(
    sample_dir: Path,
    boxes: Sequence[AnnotationBox],
    dicom_index: Dict[int, DicomSlice],
    args: argparse.Namespace,
    issues: List[ValidationIssue],
) -> List[RenderResult]:
    sample_id = sample_dir.name
    grouped: Dict[str, List[AnnotationBox]] = {}
    for box in boxes:
        if not box.image_path:
            add_issue(issues, sample_id, box.json_path, "error", "annotation has no image_path", box.raw_label)
            continue
        grouped.setdefault(Path(box.image_path).name, []).append(box)

    results: List[RenderResult] = []
    for image_name, image_boxes in sorted(grouped.items()):
        dicom_slice = find_dicom_by_name(sample_dir, image_name, dicom_index)
        if dicom_slice is None:
            add_issue(issues, sample_id, sample_dir, "error", f"referenced DICOM image_path not found: {image_name}")
            continue
        output_name = f"{sample_id}_reference_{sanitize_filename(image_name)}.png"
        output_path = args.output_dir / sample_id / output_name if args.all else args.output_dir / output_name
        results.append(
            render_image(
                sample_id=sample_id,
                dicom_path=dicom_slice.path,
                output_path=output_path,
                boxes=image_boxes,
                issues=issues,
                mode="reference",
                lower_percentile=args.window_low,
                upper_percentile=args.window_high,
            )
        )
    return results


def render_sample_slice(
    sample_dir: Path,
    boxes: Sequence[AnnotationBox],
    dicom_index: Dict[int, DicomSlice],
    args: argparse.Namespace,
    issues: List[ValidationIssue],
) -> List[RenderResult]:
    sample_id = sample_dir.name
    instance_number = args.slice
    if instance_number not in dicom_index:
        available = sorted(dicom_index)
        if not available:
            raise ValueError(f"{sample_id}: no readable DICOM slices")
        raise ValueError(
            f"{sample_id}: requested slice InstanceNumber {instance_number} is not available; "
            f"available range is {available[0]}-{available[-1]}"
        )

    image_boxes = [box for box in boxes if box.contains_slice(instance_number)]
    if args.skip_empty and not image_boxes:
        return []
    output_name = f"{sample_id}_slice_{instance_number}_{args.side}.png"
    output_path = args.output_dir / sample_id / output_name if args.all else args.output_dir / output_name
    return [
        render_image(
            sample_id=sample_id,
            dicom_path=dicom_index[instance_number].path,
            output_path=output_path,
            boxes=image_boxes,
            issues=issues,
            mode="slice",
            lower_percentile=args.window_low,
            upper_percentile=args.window_high,
        )
    ]


def write_validation_report(output_dir: Path, issues: Sequence[ValidationIssue]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "validation_report.csv"
    with report_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["severity", "sample_id", "source", "message", "raw_label"],
        )
        writer.writeheader()
        for issue in issues:
            writer.writerow(
                {
                    "severity": issue.severity,
                    "sample_id": issue.sample_id,
                    "source": issue.source,
                    "message": issue.message,
                    "raw_label": issue.raw_label,
                }
            )
    return report_path


def write_render_manifest(output_dir: Path, results: Sequence[RenderResult]) -> Path:
    manifest_path = output_dir / "render_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "mode",
                "instance_number",
                "dicom_path",
                "boxes_drawn",
                "output_path",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "sample_id": result.sample_id,
                    "mode": result.mode,
                    "instance_number": result.instance_number,
                    "dicom_path": str(result.dicom_path),
                    "boxes_drawn": result.boxes_drawn,
                    "output_path": str(result.output_path),
                }
            )
    return manifest_path


def process_dataset(args: argparse.Namespace) -> Tuple[List[RenderResult], List[ValidationIssue]]:
    require_dependencies()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = args.dataset_root
    sample_dirs = select_samples(dataset_root, args.sample, args.all)
    results: List[RenderResult] = []
    issues: List[ValidationIssue] = []

    for sample_dir in sample_dirs:
        annotations = load_annotations(sample_dir, issues)
        annotations = filter_boxes(annotations, side=args.side, class_filter=args.class_filter)
        dicom_index = build_dicom_index(sample_dir, issues)
        if not dicom_index:
            continue
        if args.mode == "reference":
            results.extend(render_sample_reference(sample_dir, annotations, dicom_index, args, issues))
        elif args.mode == "slice":
            results.extend(render_sample_slice(sample_dir, annotations, dicom_index, args, issues))
        else:
            raise ValueError(f"Unsupported mode: {args.mode}")

    return results, issues


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read the TMJ CBCT dataset and export PNG visualizations of 3D bounding boxes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-root", type=Path, required=True, help="Dataset root containing TMJ* folders.")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--sample", help="Single sample id, for example TMJ00622.")
    target.add_argument("--all", action="store_true", help="Process all TMJ* sample folders.")
    parser.add_argument("--side", choices=["L", "R", "all"], default="all", help="Side filter.")
    parser.add_argument("--mode", choices=["reference", "slice"], default="reference", help="Visualization mode.")
    parser.add_argument(
        "--slice",
        type=int,
        dest="slice",
        help="DICOM InstanceNumber to visualize when --mode slice is used.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("tmj_visualizations"), help="PNG output directory.")
    parser.add_argument(
        "--class-filter",
        type=parse_class_filter,
        default=None,
        help="Comma-separated class ids to draw, for example 1,3,5.",
    )
    parser.add_argument(
        "--window-low",
        type=float,
        default=1.0,
        help="Lower percentile for grayscale windowing.",
    )
    parser.add_argument(
        "--window-high",
        type=float,
        default=99.0,
        help="Upper percentile for grayscale windowing.",
    )
    parser.add_argument(
        "--skip-empty",
        action="store_true",
        help="In slice mode, do not save images when no filtered box covers the requested slice.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.mode == "slice" and args.slice is None:
        raise ValueError("--slice INSTANCE_NUMBER is required when --mode slice is used")
    if args.mode == "reference" and args.slice is not None:
        raise ValueError("--slice is only valid with --mode slice")
    if args.window_low < 0 or args.window_high > 100 or args.window_low >= args.window_high:
        raise ValueError("--window-low and --window-high must satisfy 0 <= low < high <= 100")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        validate_args(args)
        results, issues = process_dataset(args)
        validation_report = write_validation_report(args.output_dir, issues)
        manifest = write_render_manifest(args.output_dir, results)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    counts = {
        "info": sum(1 for issue in issues if issue.severity == "info"),
        "warning": sum(1 for issue in issues if issue.severity == "warning"),
        "error": sum(1 for issue in issues if issue.severity == "error"),
    }
    print(f"Rendered {len(results)} PNG file(s).")
    print(f"Render manifest: {manifest}")
    print(f"Validation report: {validation_report}")
    print(f"Validation issues: info={counts['info']} warning={counts['warning']} error={counts['error']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
