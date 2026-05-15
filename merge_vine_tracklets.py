#!/usr/bin/env python3
"""Suggest merges between SAM3 vine tracklets that likely form one physical vine.

The script is intentionally conservative: by default it writes ranked merge
candidates and an automatic component map, but it does not rewrite masks.
Use --write-remapped-masks after reviewing the suggested groups.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from select_gaussians import find_iteration, id_to_rgb, load_scene_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build physical-vine merge candidates from SAM3 vine tracklets.")
    parser.add_argument("-m", "--model-path", required=True, type=Path)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--scene-dir", type=Path, default=None, help="Prepared scene with object_mask/. Auto-detected when omitted.")
    parser.add_argument("--mask-dir", type=Path, default=None, help="Instance mask directory. Defaults to <scene-dir>/object_mask.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--class-name", default="vine_plant")
    parser.add_argument("--include-semantic-class", action="store_true", help="Include the semantic class label itself, not only SAM3 track IDs.")

    parser.add_argument("--min-points", type=int, default=20, help="Minimum predicted Gaussians for a tracklet.")
    parser.add_argument("--min-mask-area", type=int, default=25, help="Minimum per-frame mask pixels used for 2D statistics.")
    parser.add_argument("--min-opacity", type=float, default=0.0)
    parser.add_argument("--sample-points", type=int, default=256, help="Per-tracklet points sampled for nearest-neighbor distances.")

    parser.add_argument("--dilate-pixels", type=int, default=9, help="2D dilation radius used to detect adjacent masks.")
    parser.add_argument("--max-centroid-dist", type=float, default=0.80)
    parser.add_argument("--max-bbox-gap", type=float, default=0.20)
    parser.add_argument("--max-nn-dist", type=float, default=0.20)
    parser.add_argument("--max-frame-gap", type=int, default=10)
    parser.add_argument("--max-2d-center-dist", type=float, default=80.0)
    parser.add_argument("--max-endpoint-center-dist", type=float, default=140.0)
    parser.add_argument("--candidate-threshold", type=float, default=0.60)
    parser.add_argument("--merge-threshold", type=float, default=0.75)
    parser.add_argument("--min-merge-adjacent-frames", type=int, default=2)
    parser.add_argument("--strict-merge-nn-dist", type=float, default=0.03)
    parser.add_argument("--strict-merge-centroid-dist", type=float, default=0.45)
    parser.add_argument("--strict-merge-endpoint-dist", type=float, default=80.0)
    parser.add_argument("--top-k", type=int, default=300)

    parser.add_argument("--merge-map", type=Path, default=None, help="Use an existing merge map for --write-remapped-masks.")
    parser.add_argument("--write-remapped-masks", action="store_true")
    parser.add_argument("--remapped-mask-dir", type=Path, default=None)
    return parser.parse_args()


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open() as f:
        return json.load(f)


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def frame_number(path: Path) -> int:
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    return int(digits) if digits else 0


def resolve_scene_dir(model_path: Path, scene_dir: Path | None) -> Path:
    if scene_dir is not None:
        return scene_dir

    summary = load_json(model_path / "registered_images_summary.json", {})
    scene_name = summary.get("scene_name")
    candidates = []
    if scene_name:
        candidates.extend([
            Path.cwd().parent / "vineyard_posematch" / scene_name,
            Path.cwd() / "vineyard_posematch" / scene_name,
            model_path.parent / scene_name,
        ])
    for candidate in candidates:
        if (candidate / "object_mask").is_dir():
            return candidate
    raise FileNotFoundError("Could not auto-detect the prepared scene. Pass --scene-dir or --mask-dir.")


def load_instance_map(model_path: Path) -> dict[int, dict[str, Any]]:
    raw = load_json(model_path / "instance_label_map.json", {})
    out = {}
    for key, row in raw.items():
        row = dict(row)
        row["instance_id"] = int(key)
        row["class_id"] = int(row.get("class_id", key))
        row["class_name"] = str(row.get("class_name", row["class_id"]))
        row["label"] = str(row.get("label", f"{row['class_name']}_{int(key):04d}"))
        out[int(key)] = row
    return out


def is_tracklet(row: dict[str, Any], class_name: str, include_semantic_class: bool) -> bool:
    if row.get("class_name") != class_name:
        return False
    if include_semantic_class:
        return True
    return row.get("source") not in {"semantic_class", "background"}


def box_gap(a_min: np.ndarray, a_max: np.ndarray, b_min: np.ndarray, b_max: np.ndarray) -> float:
    delta = np.maximum(0.0, np.maximum(a_min - b_max, b_min - a_max))
    return float(np.linalg.norm(delta))


def make_tracklet_stats(
    instance_map: dict[int, dict[str, Any]],
    data: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    rng = np.random.default_rng(17)
    stats = {}
    skipped = []

    for inst_id, row in sorted(instance_map.items()):
        if not is_tracklet(row, args.class_name, args.include_semantic_class):
            continue
        mask = data["pred_instance"] == inst_id
        if args.min_opacity > 0:
            mask &= data["opacity"] >= args.min_opacity
        point_count = int(mask.sum())
        if point_count < args.min_points:
            skipped.append({
                "instance_id": inst_id,
                "label": row["label"],
                "points": point_count,
                "reason": "too_few_points",
            })
            continue

        pts = data["xyz"][mask]
        sample = pts
        if pts.shape[0] > args.sample_points:
            keep = rng.choice(pts.shape[0], args.sample_points, replace=False)
            sample = pts[keep]

        stats[inst_id] = {
            "instance_id": inst_id,
            "label": row["label"],
            "sam3_track_id": row.get("sam3_track_id"),
            "points": point_count,
            "centroid": pts.mean(axis=0),
            "bbox_min": pts.min(axis=0),
            "bbox_max": pts.max(axis=0),
            "sample_points": sample.astype(np.float32),
        }

    return stats, skipped


def nearest_sample_distance(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("inf")
    best = float("inf")
    chunk = 128
    for start in range(0, a.shape[0], chunk):
        diff = a[start:start + chunk, None, :] - b[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        best = min(best, float(dist.min()))
    return best


def analyze_masks(
    mask_dir: Path,
    vine_ids: set[int],
    args: argparse.Namespace,
) -> tuple[dict[int, list[dict[str, Any]]], dict[tuple[int, int], dict[str, Any]]]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for 2D adjacency analysis") from exc

    tracks: dict[int, list[dict[str, Any]]] = defaultdict(list)
    pair_stats: dict[tuple[int, int], dict[str, Any]] = defaultdict(lambda: {
        "adjacent_frames": 0,
        "adjacent_pixels": 0,
        "co_visible_frames": 0,
        "min_same_frame_center_dist": None,
    })
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (args.dilate_pixels * 2 + 1, args.dilate_pixels * 2 + 1),
    )

    for mask_path in sorted(mask_dir.glob("*.png"), key=frame_number):
        arr = np.array(Image.open(mask_path))
        labels = [int(v) for v in np.unique(arr) if int(v) in vine_ids]
        if not labels:
            continue
        frame = frame_number(mask_path)
        per_frame = {}
        for label in labels:
            ys, xs = np.where(arr == label)
            area = int(xs.size)
            if area < args.min_mask_area:
                continue
            center = (float(xs.mean()), float(ys.mean()))
            bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
            item = {"frame": frame, "area": area, "center": center, "bbox": bbox}
            tracks[label].append(item)
            per_frame[label] = item

        present = sorted(per_frame)
        for a, b in itertools.combinations(present, 2):
            pair = (a, b)
            row = pair_stats[pair]
            row["co_visible_frames"] += 1
            ca = np.array(per_frame[a]["center"], dtype=np.float32)
            cb = np.array(per_frame[b]["center"], dtype=np.float32)
            dist = float(np.linalg.norm(ca - cb))
            old = row["min_same_frame_center_dist"]
            row["min_same_frame_center_dist"] = dist if old is None else min(old, dist)

        touched_pairs = set()
        for label in present:
            binary = (arr == label).astype(np.uint8)
            dilated = cv2.dilate(binary, kernel, iterations=1) > 0
            touched = np.unique(arr[dilated & (arr != label)])
            for other in touched:
                other = int(other)
                if other not in per_frame or other == label:
                    continue
                pair = tuple(sorted((label, other)))
                if pair in touched_pairs:
                    continue
                touched_pairs.add(pair)
                overlap_pixels = int(np.count_nonzero(dilated & (arr == other)))
                pair_stats[pair]["adjacent_frames"] += 1
                pair_stats[pair]["adjacent_pixels"] += overlap_pixels

    for items in tracks.values():
        items.sort(key=lambda row: row["frame"])
    return tracks, pair_stats


def temporal_endpoint_signal(
    a_track: list[dict[str, Any]],
    b_track: list[dict[str, Any]],
    max_frame_gap: int,
) -> tuple[int | None, float | None]:
    if not a_track or not b_track:
        return None, None

    ranges = [
        (a_track[-1], b_track[0]),
        (b_track[-1], a_track[0]),
    ]
    best_gap = None
    best_dist = None
    for earlier, later in ranges:
        gap = int(later["frame"]) - int(earlier["frame"])
        if gap < 0 or gap > max_frame_gap:
            continue
        dist = float(np.linalg.norm(np.array(earlier["center"]) - np.array(later["center"])))
        if best_gap is None or gap < best_gap or (gap == best_gap and dist < best_dist):
            best_gap = gap
            best_dist = dist
    return best_gap, best_dist


def dist_score(value: float | None, max_value: float) -> float:
    if value is None or not math.isfinite(value) or max_value <= 0:
        return 0.0
    return max(0.0, 1.0 - float(value) / max_value)


def build_candidates(
    stats: dict[int, dict[str, Any]],
    tracks: dict[int, list[dict[str, Any]]],
    pair_2d: dict[tuple[int, int], dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    candidates = []
    ids = sorted(stats)
    for a, b in itertools.combinations(ids, 2):
        sa = stats[a]
        sb = stats[b]
        centroid_dist = float(np.linalg.norm(sa["centroid"] - sb["centroid"]))
        bbox_dist = box_gap(sa["bbox_min"], sa["bbox_max"], sb["bbox_min"], sb["bbox_max"])
        two_d = pair_2d.get((a, b), {})
        temporal_gap, temporal_px = temporal_endpoint_signal(
            tracks.get(a, []),
            tracks.get(b, []),
            args.max_frame_gap,
        )

        cheap_candidate = (
            centroid_dist <= args.max_centroid_dist
            or bbox_dist <= args.max_bbox_gap
            or two_d.get("adjacent_frames", 0) > 0
            or (
                temporal_gap is not None
                and temporal_px is not None
                and temporal_px <= args.max_endpoint_center_dist
            )
        )
        if not cheap_candidate:
            continue

        nn_dist = nearest_sample_distance(sa["sample_points"], sb["sample_points"])
        adj_score = min(float(two_d.get("adjacent_frames", 0)) / 3.0, 1.0)
        same_frame_score = dist_score(two_d.get("min_same_frame_center_dist"), args.max_2d_center_dist)
        temporal_score = 0.0
        if temporal_gap is not None and temporal_px is not None:
            gap_score = max(0.0, 1.0 - float(temporal_gap) / float(args.max_frame_gap + 1))
            px_score = dist_score(temporal_px, args.max_endpoint_center_dist)
            temporal_score = 0.4 * gap_score + 0.6 * px_score

        score = (
            0.38 * dist_score(nn_dist, args.max_nn_dist)
            + 0.22 * dist_score(bbox_dist, args.max_bbox_gap)
            + 0.15 * dist_score(centroid_dist, args.max_centroid_dist)
            + 0.15 * adj_score
            + 0.05 * same_frame_score
            + 0.05 * temporal_score
        )
        if score < args.candidate_threshold:
            continue

        candidates.append({
            "a": a,
            "b": b,
            "a_label": sa["label"],
            "b_label": sb["label"],
            "score": score,
            "centroid_dist": centroid_dist,
            "bbox_gap": bbox_dist,
            "nn_dist": nn_dist,
            "adjacent_frames": int(two_d.get("adjacent_frames", 0)),
            "adjacent_pixels": int(two_d.get("adjacent_pixels", 0)),
            "co_visible_frames": int(two_d.get("co_visible_frames", 0)),
            "min_same_frame_center_dist": two_d.get("min_same_frame_center_dist"),
            "temporal_frame_gap": temporal_gap,
            "temporal_endpoint_center_dist": temporal_px,
            "a_points": int(sa["points"]),
            "b_points": int(sb["points"]),
        })

    candidates.sort(key=lambda row: row["score"], reverse=True)
    return candidates


class UnionFind:
    def __init__(self, values: list[int]) -> None:
        self.parent = {v: v for v in values}

    def find(self, value: int) -> int:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if rb < ra:
            ra, rb = rb, ra
        self.parent[rb] = ra


def build_merge_map(
    ids: list[int],
    stats: dict[int, dict[str, Any]],
    candidates: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    uf = UnionFind(ids)
    accepted_edges = []
    for row in candidates:
        strong_2d = int(row.get("adjacent_frames", 0)) >= args.min_merge_adjacent_frames
        strong_3d = (
            float(row.get("nn_dist", float("inf"))) <= args.strict_merge_nn_dist
            and float(row.get("centroid_dist", float("inf"))) <= args.strict_merge_centroid_dist
        )
        temporal_gap = row.get("temporal_frame_gap")
        temporal_dist = row.get("temporal_endpoint_center_dist")
        strong_temporal = (
            temporal_gap is not None
            and temporal_dist is not None
            and float(temporal_dist) <= args.strict_merge_endpoint_dist
        )
        if row["score"] < args.merge_threshold or not (strong_2d or strong_3d or strong_temporal):
            continue
        uf.union(int(row["a"]), int(row["b"]))
        accepted = dict(row)
        accepted["merge_evidence"] = {
            "strong_2d_adjacency": strong_2d,
            "strong_3d_proximity": strong_3d,
            "strong_temporal_handoff": strong_temporal,
        }
        accepted_edges.append(accepted)

    groups = defaultdict(list)
    for inst_id in ids:
        groups[uf.find(inst_id)].append(inst_id)

    physical_vines = []
    for physical_id, (_, members) in enumerate(sorted(groups.items(), key=lambda item: min(item[1])), start=1):
        member_rows = []
        for inst_id in sorted(members):
            member_rows.append({
                "instance_id": inst_id,
                "label": stats[inst_id]["label"],
                "points": int(stats[inst_id]["points"]),
                "sam3_track_id": stats[inst_id].get("sam3_track_id"),
            })
        physical_vines.append({
            "physical_vine_id": physical_id,
            "member_instance_ids": sorted(members),
            "num_tracklets": len(members),
            "points": int(sum(stats[i]["points"] for i in members)),
            "members": member_rows,
        })

    return {
        "source": "merge_vine_tracklets.py",
        "merge_threshold": args.merge_threshold,
        "merge_evidence_thresholds": {
            "min_merge_adjacent_frames": args.min_merge_adjacent_frames,
            "strict_merge_nn_dist": args.strict_merge_nn_dist,
            "strict_merge_centroid_dist": args.strict_merge_centroid_dist,
            "strict_merge_endpoint_dist": args.strict_merge_endpoint_dist,
        },
        "num_input_tracklets": len(ids),
        "num_physical_vine_candidates": len(physical_vines),
        "num_multi_tracklet_groups": int(sum(1 for row in physical_vines if row["num_tracklets"] > 1)),
        "accepted_edges": accepted_edges,
        "physical_vines": physical_vines,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in keys})


def write_report(path: Path, merge_map: dict[str, Any], candidates: list[dict[str, Any]], top_k: int) -> None:
    lines = []
    lines.append(f"input tracklets: {merge_map['num_input_tracklets']}")
    lines.append(f"physical-vine candidates: {merge_map['num_physical_vine_candidates']}")
    lines.append(f"multi-tracklet groups: {merge_map['num_multi_tracklet_groups']}")
    lines.append(f"accepted edges: {len(merge_map['accepted_edges'])}")
    lines.append("")
    lines.append("Top candidate edges:")
    for row in candidates[:top_k]:
        lines.append(
            f"  {row['score']:.3f}  {row['a']}:{row['a_label']} <-> {row['b']}:{row['b_label']}  "
            f"nn={row['nn_dist']:.3f} bbox_gap={row['bbox_gap']:.3f} "
            f"centroid={row['centroid_dist']:.3f} adj_frames={row['adjacent_frames']}"
        )
    lines.append("")
    lines.append("Multi-tracklet groups:")
    for group in merge_map["physical_vines"]:
        if group["num_tracklets"] <= 1:
            continue
        labels = ", ".join(member["label"] for member in group["members"])
        lines.append(
            f"  physical_vine_{group['physical_vine_id']:04d}: "
            f"{group['num_tracklets']} tracklets, {group['points']} points: {labels}"
        )
    path.write_text("\n".join(lines) + "\n")


def load_or_build_merge_map(args: argparse.Namespace, auto_map: dict[str, Any]) -> dict[str, Any]:
    if args.merge_map is None:
        return auto_map
    return load_json(args.merge_map, auto_map)


def write_remapped_masks(
    mask_dir: Path,
    output_dir: Path,
    merge_map: dict[str, Any],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    color_dir = output_dir.parent / f"{output_dir.name}_color"
    color_dir.mkdir(parents=True, exist_ok=True)
    member_to_physical = {}
    label_map = {"0": {"label": "background", "source": "background"}}
    for group in merge_map["physical_vines"]:
        physical_id = int(group["physical_vine_id"])
        label_map[str(physical_id)] = {
            "label": f"physical_vine_{physical_id:04d}",
            "source": "vine_tracklet_merge",
            "member_instance_ids": [int(v) for v in group["member_instance_ids"]],
            "num_tracklets": int(group["num_tracklets"]),
            "points": int(group["points"]),
        }
        for inst_id in group["member_instance_ids"]:
            member_to_physical[int(inst_id)] = physical_id

    if not member_to_physical:
        raise ValueError("Merge map has no member_instance_ids")
    max_id = max(member_to_physical)
    lut = np.zeros(max_id + 1, dtype=np.uint16)
    for inst_id, physical_id in member_to_physical.items():
        lut[inst_id] = physical_id

    for mask_path in sorted(mask_dir.glob("*.png"), key=frame_number):
        arr = np.array(Image.open(mask_path))
        remapped = np.zeros(arr.shape, dtype=np.uint16)
        valid = arr <= max_id
        remapped[valid] = lut[arr[valid]]
        Image.fromarray(remapped, mode="I;16").save(output_dir / mask_path.name)
        color = np.zeros((*remapped.shape, 3), dtype=np.uint8)
        for label in np.unique(remapped):
            label = int(label)
            if label <= 0:
                continue
            color[remapped == label] = id_to_rgb(label)
        Image.fromarray(color, mode="RGB").save(color_dir / mask_path.name)

    label_path = output_dir.parent / "physical_vine_label_map.json"
    label_path.write_text(json.dumps(label_map, indent=2, default=json_default))
    return output_dir, label_path


def strip_arrays(stats: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for inst_id, row in sorted(stats.items()):
        rows.append({
            "instance_id": inst_id,
            "label": row["label"],
            "sam3_track_id": row.get("sam3_track_id"),
            "points": int(row["points"]),
            "centroid": row["centroid"],
            "bbox_min": row["bbox_min"],
            "bbox_max": row["bbox_max"],
        })
    return rows


def main() -> None:
    args = parse_args()
    model_path = args.model_path
    iteration = find_iteration(model_path, args.iteration)
    out_dir = args.output_dir or model_path / "vine_tracklet_merges" / f"iteration_{iteration}"
    out_dir.mkdir(parents=True, exist_ok=True)

    scene_dir = resolve_scene_dir(model_path, args.scene_dir)
    mask_dir = args.mask_dir or scene_dir / "object_mask"
    if not mask_dir.is_dir():
        raise FileNotFoundError(mask_dir)

    instance_map = load_instance_map(model_path)
    data = load_scene_data(model_path, iteration)
    stats, skipped = make_tracklet_stats(instance_map, data, args)
    vine_ids = set(stats)
    if not vine_ids:
        raise RuntimeError(f"No {args.class_name} tracklets survived --min-points {args.min_points}")

    tracks, pair_2d = analyze_masks(mask_dir, vine_ids, args)
    candidates = build_candidates(stats, tracks, pair_2d, args)
    merge_map = build_merge_map(sorted(stats), stats, candidates, args)

    stats_path = out_dir / "vine_tracklet_stats.json"
    candidates_path = out_dir / "vine_tracklet_merge_candidates.json"
    candidates_csv_path = out_dir / "vine_tracklet_merge_candidates.csv"
    map_path = out_dir / "vine_tracklet_merge_map.json"
    report_path = out_dir / "vine_tracklet_merge_report.txt"

    stats_payload = {
        "model_path": str(model_path),
        "scene_dir": str(scene_dir),
        "mask_dir": str(mask_dir),
        "iteration": iteration,
        "class_name": args.class_name,
        "num_tracklets": len(stats),
        "num_skipped": len(skipped),
        "tracklets": strip_arrays(stats),
        "skipped": skipped,
    }
    stats_path.write_text(json.dumps(stats_payload, indent=2, default=json_default))
    candidates_path.write_text(json.dumps(candidates, indent=2, default=json_default))
    write_csv(candidates_csv_path, candidates)
    map_path.write_text(json.dumps(merge_map, indent=2, default=json_default))
    write_report(report_path, merge_map, candidates, args.top_k)

    remapped_info = None
    if args.write_remapped_masks:
        selected_map = load_or_build_merge_map(args, merge_map)
        remapped_dir = args.remapped_mask_dir or out_dir / "physical_vine_mask"
        mask_out, label_out = write_remapped_masks(mask_dir, remapped_dir, selected_map)
        remapped_info = {"mask_dir": str(mask_out), "label_map": str(label_out)}

    print(f"Analyzed {len(stats)} vine tracklets at iteration {iteration}")
    print(f"Merge candidates: {len(candidates)}")
    print(f"Auto groups: {merge_map['num_physical_vine_candidates']} physical-vine candidates")
    print(f"Multi-tracklet groups: {merge_map['num_multi_tracklet_groups']}")
    print(f"Wrote stats:      {stats_path}")
    print(f"Wrote candidates: {candidates_path}")
    print(f"Wrote CSV:        {candidates_csv_path}")
    print(f"Wrote merge map:  {map_path}")
    print(f"Wrote report:     {report_path}")
    if remapped_info:
        print(f"Wrote remapped masks: {remapped_info['mask_dir']}")
        print(f"Wrote label map:      {remapped_info['label_map']}")


if __name__ == "__main__":
    main()
