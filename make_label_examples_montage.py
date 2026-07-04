#!/usr/bin/env python
"""Create a GitHub-ready montage with one real example for each TMJ label.

Example:
    conda run -n base python make_label_examples_montage.py ^
        --dataset-root E:\tmj ^
        --output tmj_label_examples.png
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

from tmj_dataset_viewer import (
    CLASS_COLORS,
    CLASS_NAMES,
    AnnotationBox,
    build_dicom_index,
    find_dicom_by_name,
    import_pydicom,
    load_annotations,
    read_dicom_as_uint8,
    require_dependencies,
)


def choose_font(size: int):
    for font_name in ("arial.ttf", "segoeui.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(font_name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def text_size(draw: ImageDraw.ImageDraw, text: str, font) -> Tuple[int, int]:
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    except (AttributeError, ValueError):
        return draw.textsize(text, font=font)


def pick_examples(dataset_root: Path, seed: int) -> Dict[int, Tuple[Path, AnnotationBox]]:
    rng = random.Random(seed)
    candidates: Dict[int, List[Tuple[Path, AnnotationBox]]] = {class_id: [] for class_id in CLASS_NAMES}
    single_class_candidates: Dict[int, List[Tuple[Path, AnnotationBox]]] = {
        class_id: [] for class_id in CLASS_NAMES
    }

    sample_dirs = [path for path in dataset_root.glob("TMJ*") if path.is_dir()]
    rng.shuffle(sample_dirs)
    for sample_dir in sample_dirs:
        issues = []
        for box in load_annotations(sample_dir, issues):
            for class_id in box.classes:
                if class_id in candidates and box.image_path:
                    candidates[class_id].append((sample_dir, box))
                    if box.classes == (class_id,):
                        single_class_candidates[class_id].append((sample_dir, box))

    selected: Dict[int, Tuple[Path, AnnotationBox]] = {}
    missing = []
    for class_id in sorted(CLASS_NAMES):
        if single_class_candidates[class_id]:
            selected[class_id] = rng.choice(single_class_candidates[class_id])
        elif candidates[class_id]:
            selected[class_id] = rng.choice(candidates[class_id])
        else:
            missing.append(class_id)
    if missing:
        raise RuntimeError(f"No annotation examples found for class id(s): {missing}")
    return selected


def crop_around_box(image: Image.Image, box: AnnotationBox, output_size: int) -> Image.Image:
    x0, y0, x1, y1 = box.bbox_xyxy
    image_w, image_h = image.size
    box_w = max(1.0, x1 - x0)
    box_h = max(1.0, y1 - y0)
    margin = max(box_w, box_h) * 2.2
    crop_w = max(box_w + margin, 190.0)
    crop_h = max(box_h + margin, 190.0)
    crop_w = min(float(image_w), crop_w)
    crop_h = min(float(image_h), crop_h)

    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    left = max(0.0, min(float(image_w) - crop_w, cx - crop_w / 2.0))
    top = max(0.0, min(float(image_h) - crop_h, cy - crop_h / 2.0))
    right = left + crop_w
    bottom = top + crop_h
    crop = image.crop((int(left), int(top), int(right), int(bottom))).resize(
        (output_size, output_size),
        resample=Image.BICUBIC,
    )

    scale_x = output_size / (right - left)
    scale_y = output_size / (bottom - top)
    box._montage_xyxy = (  # type: ignore[attr-defined]
        (x0 - left) * scale_x,
        (y0 - top) * scale_y,
        (x1 - left) * scale_x,
        (y1 - top) * scale_y,
    )
    return crop


def draw_tile(
    class_id: int,
    sample_dir: Path,
    box: AnnotationBox,
    tile_w: int,
    tile_h: int,
    image_size: int,
) -> Tuple[Image.Image, Dict[str, object]]:
    issues = []
    dicom_index = build_dicom_index(sample_dir, issues)
    reference_slice = find_dicom_by_name(sample_dir, box.image_path or "", dicom_index)
    if reference_slice is None:
        raise RuntimeError(f"Could not find reference DICOM {box.image_path} for {sample_dir.name}")

    image, instance_number = read_dicom_as_uint8(reference_slice.path, 1.0, 99.0)
    crop = crop_around_box(image, box, image_size)
    tile = Image.new("RGB", (tile_w, tile_h), (248, 248, 248))
    draw = ImageDraw.Draw(tile)
    title_font = choose_font(20)
    meta_font = choose_font(14)
    small_font = choose_font(12)

    color = CLASS_COLORS.get(class_id, (255, 220, 0))
    title = f"{class_id}. {CLASS_NAMES[class_id]}"
    meta = f"{sample_dir.name} | {box.side} | slices {box.slice_start}-{box.slice_end}"
    img_x = (tile_w - image_size) // 2
    img_y = 76

    draw.text((18, 14), title, fill=(20, 20, 20), font=title_font)
    draw.text((18, 44), meta, fill=(80, 80, 80), font=meta_font)
    tile.paste(crop, (img_x, img_y))
    draw.rectangle([img_x, img_y, img_x + image_size - 1, img_y + image_size - 1], outline=(190, 190, 190))

    x0, y0, x1, y1 = box._montage_xyxy  # type: ignore[attr-defined]
    x0 += img_x
    x1 += img_x
    y0 += img_y
    y1 += img_y
    for offset in range(4):
        draw.rectangle([x0 - offset, y0 - offset, x1 + offset, y1 + offset], outline=color)

    label = CLASS_NAMES[class_id]
    tw, th = text_size(draw, label, small_font)
    label_x = int(max(img_x, min(x0, img_x + image_size - tw - 8)))
    label_y = int(max(img_y, y0 - th - 8))
    draw.rectangle([label_x, label_y, label_x + tw + 8, label_y + th + 6], fill=(0, 0, 0))
    draw.text((label_x + 4, label_y + 3), label, fill=color, font=small_font)

    record = {
        "class_id": class_id,
        "class_name": CLASS_NAMES[class_id],
        "sample_id": sample_dir.name,
        "side": box.side,
        "slice_start": box.slice_start,
        "slice_end": box.slice_end,
        "reference_instance_number": instance_number,
        "reference_dicom": str(reference_slice.path),
        "bbox_xyxy": list(box.bbox_xyxy),
        "raw_label": box.raw_label,
    }
    return tile, record


def make_montage(
    dataset_root: Path,
    output_path: Path,
    seed: int,
    manifest_path: Optional[Path],
) -> None:
    require_dependencies()
    selected = pick_examples(dataset_root, seed)
    cols = 4
    rows = 2
    tile_w = 420
    tile_h = 410
    image_size = 300
    margin = 24
    gap = 18
    header_h = 80
    sheet_w = margin * 2 + cols * tile_w + (cols - 1) * gap
    sheet_h = margin * 2 + header_h + rows * tile_h + (rows - 1) * gap

    sheet = Image.new("RGB", (sheet_w, sheet_h), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    title_font = choose_font(34)
    subtitle_font = choose_font(17)
    draw.text((margin, 18), "TMJ CBCT osseous abnormality labels", fill=(20, 20, 20), font=title_font)
    draw.text(
        (margin, 56),
        "One real reference-slice example for each annotation category. Boxes are 2D rectangles with 3D slice ranges.",
        fill=(85, 85, 85),
        font=subtitle_font,
    )

    records = []
    for idx, class_id in enumerate(sorted(CLASS_NAMES)):
        sample_dir, box = selected[class_id]
        tile, record = draw_tile(class_id, sample_dir, box, tile_w, tile_h, image_size)
        row = idx // cols
        col = idx % cols
        x = margin + col * (tile_w + gap)
        y = margin + header_h + row * (tile_h + gap)
        sheet.paste(tile, (x, y))
        records.append(record)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    if manifest_path is not None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "class_id",
                    "class_name",
                    "sample_id",
                    "side",
                    "slice_start",
                    "slice_end",
                    "reference_instance_number",
                    "reference_dicom",
                    "bbox_xyxy",
                    "raw_label",
                ],
            )
            writer.writeheader()
            for record in records:
                writer.writerow(record)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a montage containing one real TMJ CBCT example for each label category.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-root", type=Path, required=True, help="Dataset root containing TMJ* folders.")
    parser.add_argument("--output", type=Path, required=True, help="Output PNG path.")
    parser.add_argument("--manifest", type=Path, default=None, help="Optional CSV describing selected examples.")
    parser.add_argument("--seed", type=int, default=20260704, help="Random seed for choosing examples.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        make_montage(args.dataset_root, args.output, args.seed, args.manifest)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote montage: {args.output}")
    if args.manifest:
        print(f"Wrote manifest: {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
