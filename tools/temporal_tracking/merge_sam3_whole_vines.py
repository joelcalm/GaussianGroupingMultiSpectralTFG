#!/usr/bin/env python3
"""Merge SAM3 part tracks into whole-vine boxes and match them across frames."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageStat

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


class DSU:
    def __init__(self, ids: list[int]):
        self.parent = {i: i for i in ids}

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def x_overlap_ratio(a: dict, b: dict) -> float:
    overlap = max(0, min(a["x2"], b["x2"]) - max(a["x1"], b["x1"]))
    return overlap / max(1, min(a["x2"] - a["x1"], b["x2"] - b["x1"]))


def y_overlap_ratio(a: dict, b: dict) -> float:
    overlap = max(0, min(a["y2"], b["y2"]) - max(a["y1"], b["y1"]))
    return overlap / max(1, min(a["y2"] - a["y1"], b["y2"] - b["y1"]))


def y_gap(a: dict, b: dict) -> int:
    return max(0, max(a["y1"], b["y1"]) - min(a["y2"], b["y2"]))


def should_merge(a: dict, b: dict, image_w: int, image_h: int, x_overlap_min: float, y_gap_frac: float, center_gap_frac: float) -> bool:
    xo = x_overlap_ratio(a, b)
    yo = y_overlap_ratio(a, b)
    gap = y_gap(a, b)
    center_dx = abs(a["cx"] - b["cx"]) / max(image_w, 1)
    center_dy = abs(a["cy"] - b["cy"]) / max(image_h, 1)
    near_center = math.hypot(center_dx, center_dy) <= center_gap_frac
    vertically_stacked = xo >= x_overlap_min and gap <= y_gap_frac * image_h
    overlapping_parts = xo >= x_overlap_min and yo > 0
    return vertically_stacked or overlapping_parts or near_center


def load_images(images_dir: Path) -> list[Path]:
    return sorted(p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def load_tracks(mask_path: Path, min_area: int, max_track_area_fraction: float, args: argparse.Namespace) -> tuple[np.ndarray, list[dict]]:
    mask = np.asarray(Image.open(mask_path))
    max_track_area = mask.shape[0] * mask.shape[1] * max_track_area_fraction
    tracks = []
    for track_id in sorted(int(v) for v in np.unique(mask) if int(v) > 0):
        pixels = mask == track_id
        area = int(pixels.sum())
        if area < min_area or area > max_track_area:
            continue
        ys, xs = np.nonzero(pixels)
        x1, y1, x2, y2 = bbox_from_mask(pixels)
        if (x2 - x1) / max(mask.shape[1], 1) > args.max_track_width_fraction:
            continue
        tracks.append({
            "track_id": track_id,
            "area": area,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "cx": float(xs.mean()),
            "cy": float(ys.mean()),
        })
    return mask, tracks


def build_whole_vines(mask: np.ndarray, tracks: list[dict], args: argparse.Namespace, image_name: str, frame_index: int) -> list[dict]:
    h, w = mask.shape[:2]
    dsu = DSU([t["track_id"] for t in tracks])
    by_id = {t["track_id"]: t for t in tracks}
    for i, a in enumerate(tracks):
        for b in tracks[i + 1:]:
            if should_merge(a, b, w, h, args.x_overlap_min, args.y_gap_frac, args.center_gap_frac):
                dsu.union(a["track_id"], b["track_id"])

    groups: dict[int, list[int]] = {}
    for track in tracks:
        groups.setdefault(dsu.find(track["track_id"]), []).append(track["track_id"])

    rows = []
    for group_idx, track_ids in enumerate(sorted(groups.values(), key=lambda ids: min(by_id[i]["x1"] for i in ids)), start=1):
        group_mask = np.isin(mask, track_ids)
        area = int(group_mask.sum())
        if area < args.min_group_area:
            continue
        if area > args.max_area_fraction * h * w:
            continue
        ys, xs = np.nonzero(group_mask)
        x1, y1, x2, y2 = bbox_from_mask(group_mask)
        bottom_idx = np.lexsort((np.abs(xs - xs.mean()), -ys))[0]
        bottom_x = int(xs[bottom_idx])
        bottom_y = int(ys[bottom_idx])
        norm_bottom_x = float(bottom_x / max(w - 1, 1))
        norm_bottom_y = float(bottom_y / max(h - 1, 1))
        norm_centroid_x = float(xs.mean() / max(w - 1, 1))
        if norm_bottom_y < args.min_norm_bottom_y or norm_bottom_y > args.max_norm_bottom_y:
            continue
        if norm_centroid_x < args.min_norm_centroid_x or norm_centroid_x > args.max_norm_centroid_x:
            continue
        row = {
            "frame_index": frame_index,
            "image_name": image_name,
            "whole_vine_id": f"whole_{group_idx:03d}",
            "sam3_track_ids": " ".join(str(i) for i in sorted(track_ids)),
            "num_sam3_tracks": len(track_ids),
            "temporal_id": "",
            "match_method": "",
            "match_score": "",
            "bbox_xmin": x1,
            "bbox_ymin": y1,
            "bbox_xmax": x2,
            "bbox_ymax": y2,
            "area": area,
            "centroid_x": float(xs.mean()),
            "centroid_y": float(ys.mean()),
            "bottom_x": bottom_x,
            "bottom_y": bottom_y,
            "image_width": w,
            "image_height": h,
            "norm_bottom_x": norm_bottom_x,
            "norm_bottom_y": norm_bottom_y,
            "norm_centroid_x": norm_centroid_x,
        }
        rows.append(row)
    return rows


def track_set(row: dict) -> set[int]:
    return {int(v) for v in str(row["sam3_track_ids"]).split() if v}


def assign_temporal(rows_by_frame: list[list[dict]], reference_index: int, max_position_distance: float, y_weight: float) -> list[dict]:
    ref_rows = sorted(rows_by_frame[reference_index], key=lambda r: (r["bottom_x"], r["bottom_y"]))
    for idx, row in enumerate(ref_rows, start=1):
        row["temporal_id"] = f"vine_{idx:03d}"
        row["match_method"] = "reference"
        row["match_score"] = 0.0

    for frame_idx, rows in enumerate(rows_by_frame):
        if frame_idx == reference_index:
            continue
        available = set(range(len(ref_rows)))
        for row in sorted(rows, key=lambda r: -r["area"]):
            row_tracks = track_set(row)
            shared = []
            for ref_idx in available:
                overlap = len(row_tracks & track_set(ref_rows[ref_idx]))
                if overlap:
                    shared.append((overlap, ref_idx))
            if shared:
                _, ref_idx = max(shared)
                ref = ref_rows[ref_idx]
                row["temporal_id"] = ref["temporal_id"]
                row["match_method"] = "shared_sam3_track"
                row["match_score"] = len(row_tracks & track_set(ref))
                available.remove(ref_idx)
                continue

            best = None
            for ref_idx in available:
                ref = ref_rows[ref_idx]
                score = abs(row["norm_bottom_x"] - ref["norm_bottom_x"]) + y_weight * abs(row["norm_bottom_y"] - ref["norm_bottom_y"])
                if best is None or score < best[0]:
                    best = (score, ref_idx)
            if best and best[0] <= max_position_distance:
                score, ref_idx = best
                row["temporal_id"] = ref_rows[ref_idx]["temporal_id"]
                row["match_method"] = "position"
                row["match_score"] = float(score)
                available.remove(ref_idx)

    rows = [row for frame_rows in rows_by_frame for row in frame_rows]
    return rows


def color_for_id(temporal_id: str) -> tuple[int, int, int]:
    if not temporal_id:
        return (170, 170, 170)
    idx = int(temporal_id.split("_")[-1])
    hue = (idx * 0.61803398875) % 1.0
    i = int(math.floor(hue * 6.0))
    f = hue * 6.0 - i
    v = 0.96
    s = 0.75
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    rgb = [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i % 6]
    return tuple(int(c * 255) for c in rgb)


def draw_contact(images: list[Path], rows_by_frame: list[list[dict]], out_path: Path, show_unmatched: bool) -> None:
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
        small = ImageFont.truetype("DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
        small = font
    panels = []
    for image_path, rows in zip(images, rows_by_frame):
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image, "RGBA")
        for row in rows:
            if not row["temporal_id"] and not show_unmatched:
                continue
            color = color_for_id(row["temporal_id"])
            xy = [row["bbox_xmin"], row["bbox_ymin"], row["bbox_xmax"], row["bbox_ymax"]]
            draw.rectangle(xy, outline=color + (255,), width=5)
            label = row["temporal_id"] or "unmatched"
            if row["num_sam3_tracks"] > 1:
                label += f" ({row['num_sam3_tracks']} parts)"
            text_pos = (xy[0] + 4, max(0, xy[1] - 22))
            box = draw.textbbox(text_pos, label, font=small)
            draw.rectangle(box, fill=(0, 0, 0, 170))
            draw.text(text_pos, label, fill=color + (255,), font=small)
        scale = min(1.0, 900 / image.width)
        if scale < 1.0:
            image = image.resize((int(image.width * scale), int(image.height * scale)))
        title_h = 32
        panel = Image.new("RGB", (image.width, image.height + title_h), "white")
        pd = ImageDraw.Draw(panel)
        pd.text((6, 7), image_path.name, fill=(20, 20, 20), font=font)
        panel.paste(image, (0, title_h))
        panels.append(panel)
    canvas = Image.new("RGB", (sum(p.width for p in panels) + 10 * (len(panels) - 1), max(p.height for p in panels)), "white")
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width + 10
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=92)
    stat = ImageStat.Stat(canvas)
    print(f"Wrote {out_path} size={canvas.size} stddev={[round(v, 2) for v in stat.stddev]}")


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "frame_index", "image_name", "whole_vine_id", "sam3_track_ids", "num_sam3_tracks", "temporal_id",
        "match_method", "match_score", "bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax", "area",
        "centroid_x", "centroid_y", "bottom_x", "bottom_y", "image_width", "image_height", "norm_bottom_x", "norm_bottom_y", "norm_centroid_x",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sam3-dir", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-index", type=int, default=1, help="Frame index used to seed temporal IDs; default is April.")
    parser.add_argument("--min-track-area", type=int, default=200)
    parser.add_argument("--max-track-area-fraction", type=float, default=0.022, help="Drop individual SAM3 tracks larger than this frame fraction before merging.")
    parser.add_argument("--max-track-width-fraction", type=float, default=0.30, help="Drop individual SAM3 tracks wider than this frame fraction before merging.")
    parser.add_argument("--min-group-area", type=int, default=500)
    parser.add_argument("--max-area-fraction", type=float, default=0.18)
    parser.add_argument("--x-overlap-min", type=float, default=0.18)
    parser.add_argument("--y-gap-frac", type=float, default=0.09)
    parser.add_argument("--center-gap-frac", type=float, default=0.0)
    parser.add_argument("--min-norm-bottom-y", type=float, default=0.34, help="Keep only groups whose bottom point is below this normalized y value.")
    parser.add_argument("--max-norm-bottom-y", type=float, default=1.0)
    parser.add_argument("--min-norm-centroid-x", type=float, default=0.02)
    parser.add_argument("--max-norm-centroid-x", type=float, default=0.98)
    parser.add_argument("--max-position-distance", type=float, default=0.18)
    parser.add_argument("--y-weight", type=float, default=0.2)
    parser.add_argument("--show-unmatched", action="store_true")
    args = parser.parse_args()

    images = load_images(args.images_dir)
    rows_by_frame = []
    for frame_index, image_path in enumerate(images):
        mask_path = args.sam3_dir / "semantic_instance_masks" / f"{image_path.stem}.png"
        if not mask_path.exists():
            raise FileNotFoundError(mask_path)
        mask, tracks = load_tracks(mask_path, args.min_track_area, args.max_track_area_fraction, args)
        rows_by_frame.append(build_whole_vines(mask, tracks, args, image_path.name, frame_index))

    all_rows = assign_temporal(rows_by_frame, args.reference_index, args.max_position_distance, args.y_weight)
    write_csv(args.output_dir / "sam3_whole_vine_boxes.csv", all_rows)
    draw_contact(images, rows_by_frame, args.output_dir / "sam3_whole_vine_boxes_contact_sheet.jpg", args.show_unmatched)
    summary = {
        "num_frames": len(images),
        "groups_per_frame": {images[i].name: len(rows_by_frame[i]) for i in range(len(images))},
        "matched_groups_per_frame": {images[i].name: sum(1 for r in rows_by_frame[i] if r["temporal_id"]) for i in range(len(images))},
        "multi_part_groups_per_frame": {images[i].name: sum(1 for r in rows_by_frame[i] if int(r["num_sam3_tracks"]) > 1) for i in range(len(images))},
        "num_temporal_ids": len({r["temporal_id"] for r in all_rows if r["temporal_id"]}),
    }
    (args.output_dir / "sam3_whole_vine_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
