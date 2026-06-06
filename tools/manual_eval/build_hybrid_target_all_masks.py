#!/usr/bin/env python3
from __future__ import annotations

import argparse
import colorsys
import csv
import json
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

TARGET_LABELS = {
    901: {"class_id": 1, "class_name": "vine_leaf", "label": "target_vine_a_leaf", "source": "manual_target_mapping", "label_kind": "corrected_target"},
    902: {"class_id": 2, "class_name": "vine_trunk", "label": "target_vine_a_trunk", "source": "manual_target_mapping", "label_kind": "corrected_target"},
    903: {"class_id": 1, "class_name": "vine_leaf", "label": "target_vine_b_leaf", "source": "manual_target_mapping", "label_kind": "corrected_target"},
    904: {"class_id": 2, "class_name": "vine_trunk", "label": "target_vine_b_trunk", "source": "manual_target_mapping", "label_kind": "corrected_target"},
    905: {"class_id": 3, "class_name": "wooden_post", "label": "reference_wooden_post", "source": "manual_target_mapping", "label_kind": "corrected_target"},
}

FIELD_TO_TARGET = {
    "target_vine_a_leaf_ids": 901,
    "target_vine_a_trunk_ids": 902,
    "target_vine_b_leaf_ids": 903,
    "target_vine_b_trunk_ids": 904,
    "reference_wooden_post_ids": 905,
}

LEGEND_COLUMNS = ["label_id", "rgb", "class_id", "class_name", "label", "source", "label_kind", "sam3_track_id"]


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
    Image.fromarray(mask.astype(np.uint16)).save(path)


def save_overlay(image_path: Path, color: np.ndarray, active: np.ndarray, path: Path) -> None:
    if not image_path.exists():
        return
    image = np.array(Image.open(image_path).convert("RGB"))
    if image.shape[:2] != color.shape[:2]:
        image = np.array(Image.fromarray(image).resize((color.shape[1], color.shape[0]), Image.BILINEAR))
    overlay = image.copy()
    blended = (image.astype(np.float32) * 0.55 + color.astype(np.float32) * 0.45).astype(np.uint8)
    overlay[active] = blended[active]
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(path)


def read_mapping_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def build_legend(base_legend: Path, out_csv: Path, out_json: Path) -> None:
    rows = []
    if base_legend.exists():
        with base_legend.open(newline="") as f:
            rows = list(csv.DictReader(f))

    existing = {int(row["label_id"]) for row in rows if row.get("label_id", "").isdigit()}
    for label_id, meta in TARGET_LABELS.items():
        if label_id in existing:
            raise ValueError(f"Hybrid target label {label_id} already exists in {base_legend}")
        rgb = "#{:02x}{:02x}{:02x}".format(*id2rgb(label_id))
        rows.append({"label_id": str(label_id), "rgb": rgb, "sam3_track_id": "", **{k: str(v) for k, v in meta.items()}})

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LEGEND_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in LEGEND_COLUMNS})

    label_map = {}
    for row in rows:
        label_id = row.get("label_id", "")
        if not label_id:
            continue
        label_map[label_id] = {key: row.get(key, "") for key in LEGEND_COLUMNS if key != "label_id"}
    out_json.write_text(json.dumps(label_map, indent=2) + "\n")


def copy_final_subset(selected_frames: list[str], scene_dir: Path, mask_dir: Path, color_dir: Path, overlay_dir: Path, final_dir: Path, legend_csv: Path, label_map_json: Path) -> None:
    if final_dir.exists():
        shutil.rmtree(final_dir)
    (final_dir / "object_mask").mkdir(parents=True)
    (final_dir / "gt_objects_color").mkdir(parents=True)
    (final_dir / "gt_objects_overlay").mkdir(parents=True)
    (final_dir / "rgb").mkdir(parents=True)

    image_dir = scene_dir / "images_rgb"
    for frame in selected_frames:
        stem = Path(frame).stem
        shutil.copy2(mask_dir / frame, final_dir / "object_mask" / frame)
        shutil.copy2(color_dir / frame, final_dir / "gt_objects_color" / frame)
        overlay_path = overlay_dir / f"{stem}.jpg"
        if overlay_path.exists():
            shutil.copy2(overlay_path, final_dir / "gt_objects_overlay" / f"{stem}.jpg")
        rgb_path = image_dir / frame
        if rgb_path.exists():
            shutil.copy2(rgb_path, final_dir / "rgb" / frame)

    with (final_dir / "selected_frames.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame"])
        writer.writeheader()
        for frame in selected_frames:
            writer.writerow({"frame": frame})

    shutil.copy2(legend_csv, final_dir / "gt_objects_label_legend_hybrid.csv")
    shutil.copy2(label_map_json, final_dir / "hybrid_label_map.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build full SAM3 object masks with manual target-vine IDs overlaid.")
    parser.add_argument("--scene_dir", type=Path, default=Path("data/vinyes_20260509_pinhole"))
    parser.add_argument("--subset_dir", type=Path, default=Path("data/vinyes_20260509_pinhole/manual_eval_gt/target_two_vines_all31"))
    parser.add_argument("--mapping_csv", type=Path, default=None)
    parser.add_argument("--reference_mask_dir", type=Path, default=None)
    parser.add_argument("--selected_frames_csv", type=Path, default=None)
    parser.add_argument("--output_mask_name", default="object_mask_hybrid_target_all")
    parser.add_argument("--final_output_name", default="final_15_all_objects")
    args = parser.parse_args()

    scene = args.scene_dir
    manual_root = scene / "manual_eval_gt"
    subset = args.subset_dir
    mapping_csv = args.mapping_csv or subset / "target_id_mapping_template.csv"
    reference_mask_dir = args.reference_mask_dir or manual_root / "object_mask_reference"
    selected_frames_csv = args.selected_frames_csv or subset / "final_15" / "selected_frames.csv"

    if not mapping_csv.exists():
        raise FileNotFoundError(mapping_csv)
    if not reference_mask_dir.is_dir():
        raise FileNotFoundError(reference_mask_dir)
    if not selected_frames_csv.exists():
        raise FileNotFoundError(selected_frames_csv)

    out_mask_dir = manual_root / args.output_mask_name
    out_color_dir = manual_root / f"{args.output_mask_name}_color"
    out_overlay_dir = manual_root / f"{args.output_mask_name}_overlay"
    for directory in (out_mask_dir, out_color_dir, out_overlay_dir):
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True)

    rows = read_mapping_rows(mapping_csv)
    rows_by_frame = {row["frame"]: row for row in rows}
    report_rows = []
    conflict_rows = []
    image_dir = scene / "images_rgb"

    for ref_path in sorted(reference_mask_dir.glob("rgb_*.png")):
        frame = ref_path.name
        mask = np.array(Image.open(ref_path)).astype(np.uint16)
        if mask.ndim == 3:
            mask = mask[..., 0].astype(np.uint16)
        out = mask.copy()

        row = rows_by_frame.get(frame)
        assignments: dict[int, tuple[str, int]] = {}
        if row is not None:
            for field, target_id in FIELD_TO_TARGET.items():
                for source_id in parse_ids(row.get(field, "")):
                    previous = assignments.get(source_id)
                    if previous is not None and previous != (field, target_id):
                        conflict_rows.append({
                            "frame": frame,
                            "source_id": source_id,
                            "first_field": previous[0],
                            "first_target_id": previous[1],
                            "overriding_field": field,
                            "overriding_target_id": target_id,
                        })
                    assignments[source_id] = (field, target_id)

            for source_id, (field, target_id) in assignments.items():
                pixels = int(np.count_nonzero(mask == source_id))
                if pixels:
                    out[mask == source_id] = target_id
                report_rows.append({
                    "frame": frame,
                    "field": field,
                    "source_id": source_id,
                    "target_id": target_id,
                    "pixels": pixels,
                })

        save_index(out, out_mask_dir / frame)
        color = colorize(out)
        Image.fromarray(color).save(out_color_dir / frame)
        save_overlay(image_dir / frame, color, out > 0, out_overlay_dir / f"{Path(frame).stem}.jpg")

    legend_csv = manual_root / "gt_objects_label_legend_hybrid_target_all.csv"
    label_map_json = manual_root / "hybrid_target_all_label_map.json"
    build_legend(manual_root / "gt_objects_label_legend.csv", legend_csv, label_map_json)

    with (manual_root / "hybrid_target_all_mapping_report.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame", "field", "source_id", "target_id", "pixels"])
        writer.writeheader()
        writer.writerows(report_rows)

    with (manual_root / "hybrid_target_all_mapping_conflicts.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame", "source_id", "first_field", "first_target_id", "overriding_field", "overriding_target_id"])
        writer.writeheader()
        writer.writerows(conflict_rows)

    selected_frames = [row["frame"] for row in csv.DictReader(selected_frames_csv.open())]
    final_dir = subset / args.final_output_name
    copy_final_subset(selected_frames, scene, out_mask_dir, out_color_dir, out_overlay_dir, final_dir, legend_csv, label_map_json)

    summary = {
        "mask_dir": str(out_mask_dir),
        "color_dir": str(out_color_dir),
        "overlay_dir": str(out_overlay_dir),
        "final_dir": str(final_dir),
        "mapping_report": str(manual_root / "hybrid_target_all_mapping_report.csv"),
        "mapping_conflicts": str(manual_root / "hybrid_target_all_mapping_conflicts.csv"),
        "legend_csv": str(legend_csv),
        "label_map_json": str(label_map_json),
        "target_labels": TARGET_LABELS,
        "frames_with_manual_target_mapping": len(rows_by_frame),
        "selected_final_frames": len(selected_frames),
        "conflicts": len(conflict_rows),
    }
    (manual_root / "hybrid_target_all_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
