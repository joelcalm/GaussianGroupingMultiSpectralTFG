#!/usr/bin/env python3
from __future__ import annotations

import argparse
import colorsys
import csv
import json
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

TARGET_LABELS = {
    0: {"name": "background_or_ignore", "class_name": "background"},
    1: {"name": "target_vine_a_leaf", "class_name": "vine_leaf"},
    2: {"name": "target_vine_a_trunk", "class_name": "vine_trunk"},
    3: {"name": "target_vine_b_leaf", "class_name": "vine_leaf"},
    4: {"name": "target_vine_b_trunk", "class_name": "vine_trunk"},
    5: {"name": "reference_wooden_post", "class_name": "wooden_post"},
    6: {"name": "ground", "class_name": "ground"},
    7: {"name": "building_wall", "class_name": "building_wall"},
    8: {"name": "sky", "class_name": "sky"},
    9: {"name": "other_vegetation", "class_name": "other_vegetation"},
}
SEMANTIC_SOURCE_TO_TARGET = {
    805: 6,
    806: 7,
    807: 8,
    808: 9,
}
FIELD_TO_TARGET = {
    "target_vine_a_leaf_ids": 1,
    "target_vine_a_trunk_ids": 2,
    "target_vine_b_leaf_ids": 3,
    "target_vine_b_trunk_ids": 4,
    "reference_wooden_post_ids": 5,
}


def parse_ids(value: str) -> list[int]:
    return [int(v) for v in re.findall(r"\d+", value or "")]


def id2rgb(idx: int) -> tuple[int, int, int]:
    if idx <= 0:
        return (0, 0, 0)
    h = (idx * 1.6180339887) % 1
    s = 0.5 + (idx % 2) * 0.5
    l = 0.5
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return (int(r * 255), int(g * 255), int(b * 255))


def colorize(mask: np.ndarray) -> np.ndarray:
    color = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for idx in np.unique(mask):
        color[mask == idx] = id2rgb(int(idx))
    return color


def save_index(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8), mode="L").save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply stable target-vine ID mappings to selected manual-eval masks.")
    parser.add_argument("--subset_dir", type=Path, required=True)
    parser.add_argument("--mapping_csv", type=Path, default=None)
    parser.add_argument("--image_dir", type=Path, default=Path("data/vinyes_20260509_pinhole/images_rgb"))
    parser.add_argument(
        "--input_mask_dir",
        type=Path,
        default=None,
        help="Indexed masks to relabel. Defaults to corrected_masks/ when present, else reference_masks/.",
    )
    args = parser.parse_args()

    subset = args.subset_dir
    mapping_csv = args.mapping_csv or subset / "target_id_mapping_template.csv"
    ref_dir = args.input_mask_dir or (subset / "corrected_masks" if (subset / "corrected_masks").is_dir() else subset / "reference_masks")
    out_dir = subset / "stable_eval_masks"
    color_dir = subset / "stable_eval_color"
    overlay_dir = subset / "stable_eval_overlay"
    out_dir.mkdir(parents=True, exist_ok=True)
    color_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    with mapping_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))

    report = []
    for row in rows:
        frame = row["frame"]
        ref_path = ref_dir / frame
        if not ref_path.exists():
            raise FileNotFoundError(ref_path)
        ref = np.array(Image.open(ref_path))
        if ref.ndim == 3:
            ref = ref[..., 0]

        out = np.zeros(ref.shape, dtype=np.uint8)
        for src_id, target_id in SEMANTIC_SOURCE_TO_TARGET.items():
            out[ref == src_id] = target_id

        for field, target_id in FIELD_TO_TARGET.items():
            source_ids = parse_ids(row.get(field, ""))
            if source_ids:
                out[np.isin(ref, source_ids)] = target_id
            report.append({
                "frame": frame,
                "field": field,
                "target_id": target_id,
                "source_ids": " ".join(str(v) for v in source_ids),
                "pixels": int(np.isin(ref, source_ids).sum()) if source_ids else 0,
            })

        save_index(out, out_dir / frame)
        color = colorize(out)
        Image.fromarray(color).save(color_dir / frame)

        image_path = args.image_dir / frame
        if image_path.exists():
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            color_bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
            overlay = image.copy()
            active = out > 0
            blended = cv2.addWeighted(image, 0.55, color_bgr, 0.45, 0)
            overlay[active] = blended[active]
            cv2.imwrite(str(overlay_dir / frame.replace(".png", ".jpg")), overlay)

    (subset / "stable_eval_label_map.json").write_text(json.dumps({str(k): v for k, v in TARGET_LABELS.items()}, indent=2) + "\n")
    with (subset / "stable_eval_report.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame", "field", "target_id", "source_ids", "pixels"])
        writer.writeheader()
        writer.writerows(report)

    print(f"Wrote stable masks: {out_dir}")
    print(f"Wrote overlays: {overlay_dir}")


if __name__ == "__main__":
    main()
