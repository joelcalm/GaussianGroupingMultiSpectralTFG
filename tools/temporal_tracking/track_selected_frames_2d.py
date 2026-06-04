#!/usr/bin/env python3
"""Create a selected-frame 2D temporal vine tracking visualization."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import cv2
except ImportError:  # pragma: no cover - handled at runtime
    cv2 = None

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent))

from config import load_config, resolve_mask, resolve_selected_image  # noqa: E402


def load_label_map(source_path: Path) -> dict[int, dict]:
    path = source_path / "metadata" / "instance_label_map.json"
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    return {int(k): v for k, v in raw.items()}


def load_index_mask(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path))
    if arr.ndim == 2:
        return arr.astype(np.int64)
    if arr.ndim == 3 and arr.shape[2] == 1:
        return arr[:, :, 0].astype(np.int64)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        rgb = arr[:, :, :3].astype(np.int64)
        if np.all(rgb[:, :, 0] == rgb[:, :, 1]) and np.all(rgb[:, :, 0] == rgb[:, :, 2]):
            return rgb[:, :, 0]
        return (rgb[:, :, 0] << 16) + (rgb[:, :, 1] << 8) + rgb[:, :, 2]
    raise ValueError(f"Unsupported mask shape {arr.shape} for {path}")


def is_vine_label(entry: dict) -> bool:
    object_type = str(entry.get("object_type", "")).lower()
    class_name = str(entry.get("class_name", "")).lower()
    label = str(entry.get("label", "")).lower()
    return object_type == "vine" or class_name.startswith("vine") or label.startswith("vine")


def vine_label_ids(mask: np.ndarray, label_map: dict[int, dict]) -> list[int]:
    labels = [int(v) for v in np.unique(mask) if int(v) > 0]
    if not label_map:
        return labels
    return [label for label in labels if is_vine_label(label_map.get(label, {}))]


def label_groups(mask: np.ndarray, label_map: dict[int, dict]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    if not label_map:
        return {str(label): [label] for label in vine_label_ids(mask, label_map)}

    for label in vine_label_ids(mask, label_map):
        entry = label_map.get(label, {})
        group_id = entry.get("instance_id") or entry.get("label") or str(label)
        if str(group_id).lower() in {"", "none", "background"}:
            group_id = str(label)
        groups.setdefault(str(group_id), []).append(label)
    return groups


def _row_from_pixels(
    scene_name: str,
    image_name: str,
    local_id: str,
    source_label_ids: list[int],
    xs: np.ndarray,
    ys: np.ndarray,
    width: int,
    height: int,
) -> dict:
    area = int(xs.size)
    centroid_x = float(xs.mean())
    centroid_y = float(ys.mean())
    bottom_idx = np.lexsort((np.abs(xs - centroid_x), -ys))[0]
    bottom_x = int(xs[bottom_idx])
    bottom_y = int(ys[bottom_idx])
    return {
        "scene": scene_name,
        "image_name": image_name,
        "local_instance_id": local_id,
        "source_label_ids": " ".join(str(label) for label in sorted(source_label_ids)),
        "temporal_id": "",
        "match_score": "",
        "centroid_x": centroid_x,
        "centroid_y": centroid_y,
        "bbox_xmin": int(xs.min()),
        "bbox_ymin": int(ys.min()),
        "bbox_xmax": int(xs.max()),
        "bbox_ymax": int(ys.max()),
        "area": area,
        "bottom_x": bottom_x,
        "bottom_y": bottom_y,
        "image_width": int(width),
        "image_height": int(height),
        "norm_bottom_x": float(bottom_x / max(width - 1, 1)),
        "norm_bottom_y": float(bottom_y / max(height - 1, 1)),
    }


def metadata_group_rows(scene_name: str, image_name: str, mask: np.ndarray, label_map: dict[int, dict], min_area: int, max_area_fraction: float) -> list[dict]:
    height, width = mask.shape[:2]
    max_area = height * width * max_area_fraction
    rows = []
    for group_id, labels in sorted(label_groups(mask, label_map).items()):
        ys, xs = np.nonzero(np.isin(mask, labels))
        area = int(xs.size)
        if area < min_area or area > max_area:
            continue
        rows.append(_row_from_pixels(scene_name, image_name, group_id, labels, xs, ys, width, height))
    return sorted(rows, key=lambda r: (r["bottom_x"], r["centroid_x"], -r["area"]))


def connected_component_rows(
    scene_name: str,
    image_name: str,
    mask: np.ndarray,
    label_map: dict[int, dict],
    min_area: int,
    max_area_fraction: float,
    close_kernel: int,
) -> list[dict]:
    if cv2 is None:
        raise RuntimeError("OpenCV is required for --extraction-mode connected_components.")
    height, width = mask.shape[:2]
    labels = vine_label_ids(mask, label_map)
    binary = np.isin(mask, labels).astype(np.uint8)
    if close_kernel > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    num_labels, components, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    max_area = height * width * max_area_fraction
    rows = []
    for component_id in range(1, num_labels):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        ys, xs = np.nonzero(components == component_id)
        overlapping = sorted(int(v) for v in np.unique(mask[components == component_id]) if int(v) in labels)
        local_id = f"cc_{component_id:03d}"
        rows.append(_row_from_pixels(scene_name, image_name, local_id, overlapping, xs, ys, width, height))
    return sorted(rows, key=lambda r: (r["bottom_x"], r["centroid_x"], -r["area"]))


def extract_instance_rows(
    scene_name: str,
    image_name: str,
    mask: np.ndarray,
    label_map: dict[int, dict],
    min_area: int,
    max_area_fraction: float,
    extraction_mode: str,
    close_kernel: int,
) -> list[dict]:
    if extraction_mode == "metadata_groups":
        return metadata_group_rows(scene_name, image_name, mask, label_map, min_area, max_area_fraction)
    if extraction_mode == "connected_components":
        return connected_component_rows(scene_name, image_name, mask, label_map, min_area, max_area_fraction, close_kernel)
    raise ValueError(f"Unknown extraction mode: {extraction_mode}")


def assign_temporal_ids(
    rows_by_scene: dict[str, list[dict]],
    reference_scene: str,
    max_match_distance: float,
    y_weight: float,
    min_scenes_per_id: int,
) -> list[dict]:
    ref_rows = rows_by_scene[reference_scene]
    ref_sorted = sorted(ref_rows, key=lambda r: (r["bottom_x"], r["bottom_y"]))
    for idx, row in enumerate(ref_sorted):
        row["temporal_id"] = f"vine_{idx + 1:03d}"
        row["match_score"] = 0.0

    for scene_name, rows in rows_by_scene.items():
        if scene_name == reference_scene:
            continue
        available = set(range(len(ref_sorted)))
        source_rows = sorted(rows, key=lambda r: (-r["area"], r["bottom_x"]))
        for row in source_rows:
            if not available:
                break
            best_idx = None
            best_score = None
            for idx in available:
                ref = ref_sorted[idx]
                score = abs(row["norm_bottom_x"] - ref["norm_bottom_x"]) + y_weight * abs(row["norm_bottom_y"] - ref["norm_bottom_y"])
                if best_score is None or score < best_score:
                    best_idx = idx
                    best_score = score
            if best_idx is not None and best_score is not None and best_score <= max_match_distance:
                row["temporal_id"] = ref_sorted[best_idx]["temporal_id"]
                row["match_score"] = float(best_score)
                available.remove(best_idx)

    counts: dict[str, int] = {}
    for rows in rows_by_scene.values():
        for row in rows:
            temporal_id = row.get("temporal_id")
            if temporal_id:
                counts[temporal_id] = counts.get(temporal_id, 0) + 1
    kept = {temporal_id for temporal_id, count in counts.items() if count >= min_scenes_per_id}
    for rows in rows_by_scene.values():
        for row in rows:
            if row.get("temporal_id") not in kept:
                row["temporal_id"] = ""
                row["match_score"] = ""

    association_rows = []
    for temporal_id in sorted(kept):
        for scene_name, rows in rows_by_scene.items():
            match = next((row for row in rows if row.get("temporal_id") == temporal_id), None)
            association_rows.append({
                "temporal_id": temporal_id,
                "scene": scene_name,
                "local_instance_id": "" if match is None else match["local_instance_id"],
                "source_label_ids": "" if match is None else match["source_label_ids"],
                "match_score": "" if match is None else match["match_score"],
                "centroid_x": "" if match is None else match["centroid_x"],
                "centroid_y": "" if match is None else match["centroid_y"],
                "bottom_x": "" if match is None else match["bottom_x"],
                "bottom_y": "" if match is None else match["bottom_y"],
            })
    return association_rows


def color_for_id(temporal_id: str) -> tuple[int, int, int]:
    idx = int(temporal_id.split("_")[-1])
    hue = (idx * 0.61803398875) % 1.0
    r, g, b = colorsys_hsv_to_rgb(hue, 0.72, 0.95)
    return int(r * 255), int(g * 255), int(b * 255)


def colorsys_hsv_to_rgb(h: float, s: float, v: float) -> tuple[float, float, float]:
    i = int(math.floor(h * 6.0))
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i %= 6
    if i == 0:
        return v, t, p
    if i == 1:
        return q, v, p
    if i == 2:
        return p, v, t
    if i == 3:
        return p, q, v
    if i == 4:
        return t, p, v
    return v, p, q


def draw_panel(image_path: Path, rows: list[dict], max_width: int = 900, show_unmatched: bool = False) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    scale = min(1.0, max_width / image.width)
    if scale < 1.0:
        image = image.resize((int(image.width * scale), int(image.height * scale)))
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default()

    for row in rows:
        temporal_id = row.get("temporal_id", "")
        if not temporal_id and not show_unmatched:
            continue
        color = color_for_id(temporal_id) if temporal_id else (160, 160, 160)
        xy = [
            int(row["bbox_xmin"] * scale),
            int(row["bbox_ymin"] * scale),
            int(row["bbox_xmax"] * scale),
            int(row["bbox_ymax"] * scale),
        ]
        draw.rectangle(xy, outline=color + (255,), width=3)
        label = temporal_id or "unmatched"
        text_pos = (xy[0] + 3, max(0, xy[1] - 14))
        text_box = draw.textbbox(text_pos, label, font=font)
        draw.rectangle(text_box, fill=(0, 0, 0, 150))
        draw.text(text_pos, label, fill=color + (255,), font=font)
        draw.ellipse(
            [
                int(row["bottom_x"] * scale) - 4,
                int(row["bottom_y"] * scale) - 4,
                int(row["bottom_x"] * scale) + 4,
                int(row["bottom_y"] * scale) + 4,
            ],
            fill=color + (220,),
        )
    return image


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def save_figure(panels: list[tuple[str, Image.Image]], out_path: Path) -> None:
    title_h = 28
    gap = 10
    widths = [panel.width for _, panel in panels]
    heights = [panel.height for _, panel in panels]
    canvas = Image.new("RGB", (sum(widths) + gap * (len(panels) - 1), max(heights) + title_h), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    x = 0
    for scene_name, panel in panels:
        draw.text((x + 6, 8), scene_name, fill=(20, 20, 20), font=font)
        canvas.paste(panel, (x, title_h))
        x += panel.width + gap
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Temporal tracking YAML config.")
    parser.add_argument("--min-area", type=int, default=1000, help="Minimum connected vine area to keep.")
    parser.add_argument("--max-area-fraction", type=float, default=0.15, help="Drop components larger than this image fraction.")
    parser.add_argument("--extraction-mode", choices=["connected_components", "metadata_groups"], default="connected_components")
    parser.add_argument("--close-kernel", type=int, default=0, help="Optional morphology close kernel for connected components.")
    parser.add_argument("--max-match-distance", type=float, default=0.22, help="Maximum normalized reference-position match score.")
    parser.add_argument("--y-weight", type=float, default=0.2, help="Weight for normalized bottom-y mismatch during matching.")
    parser.add_argument("--min-scenes-per-id", type=int, default=2, help="Only display IDs seen in at least this many scenes.")
    parser.add_argument("--show-unmatched", action="store_true", help="Draw unmatched detections in gray.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    rows_by_scene: dict[str, list[dict]] = {}
    selected_images: dict[str, Path] = {}
    for scene in cfg.scenes.values():
        image_path = resolve_selected_image(scene)
        mask_path = resolve_mask(scene, image_path)
        mask = load_index_mask(mask_path)
        label_map = load_label_map(scene.source_path)
        rows = extract_instance_rows(
            scene.name,
            image_path.name,
            mask,
            label_map,
            args.min_area,
            args.max_area_fraction,
            args.extraction_mode,
            args.close_kernel,
        )
        if not rows:
            raise RuntimeError(
                f"{scene.name}: no vine candidates survived filtering in {mask_path}. "
                "Try lowering --min-area, increasing --max-area-fraction, or using --extraction-mode metadata_groups."
            )
        selected_images[scene.name] = image_path
        rows_by_scene[scene.name] = rows

    association_rows = assign_temporal_ids(
        rows_by_scene,
        cfg.reference_scene,
        args.max_match_distance,
        args.y_weight,
        args.min_scenes_per_id,
    )

    instance_fields = [
        "scene",
        "image_name",
        "local_instance_id",
        "source_label_ids",
        "temporal_id",
        "match_score",
        "centroid_x",
        "centroid_y",
        "bbox_xmin",
        "bbox_ymin",
        "bbox_xmax",
        "bbox_ymax",
        "area",
        "bottom_x",
        "bottom_y",
        "image_width",
        "image_height",
        "norm_bottom_x",
        "norm_bottom_y",
    ]
    for scene_name, rows in rows_by_scene.items():
        write_csv(cfg.output_dir / "instances" / f"instances_{scene_name}.csv", rows, instance_fields)

    write_csv(
        cfg.output_dir / "temporal_vine_ids_2d.csv",
        association_rows,
        [
            "temporal_id",
            "scene",
            "local_instance_id",
            "source_label_ids",
            "match_score",
            "centroid_x",
            "centroid_y",
            "bottom_x",
            "bottom_y",
        ],
    )

    panels = [
        (scene_name, draw_panel(selected_images[scene_name], rows, show_unmatched=args.show_unmatched))
        for scene_name, rows in rows_by_scene.items()
    ]
    figure_path = cfg.output_dir / "figures" / "temporal_vine_tracking_2d.png"
    save_figure(panels, figure_path)
    print(f"Wrote temporal associations: {cfg.output_dir / 'temporal_vine_ids_2d.csv'}")
    print(f"Wrote 3-panel figure: {figure_path}")


if __name__ == "__main__":
    main()
