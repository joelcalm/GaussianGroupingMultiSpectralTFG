#!/usr/bin/env python3
"""Prepare the vinyes_partial200 shared COLMAP scene with SAM3 labels.

The script keeps the existing COLMAP reconstruction intact, links/copies the
registered images, aligns RGB SAM3 semantic masks, optionally builds weak
instance tracks from semantic connected components, and writes metadata used by
the multichannel active-channel training path.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


BAND_CHANNELS = {
    "rgbp": [0, 1, 2],
    "RGB": [0, 1, 2],
    "b470": [3],
    "b505": [4],
    "b525": [5],
    "b590": [6],
    "b635": [7],
    "b660": [8],
    "b850": [9],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_scene", type=Path, default=Path("vineyard_posematch/vinyes_partial200"))
    parser.add_argument("--sam3_dir", type=Path, default=Path("vineyard_posematch/sam3_video_vinyes"))
    parser.add_argument("--output_scene", type=Path, default=Path("vineyard_posematch/vinyes_sam3_200"))
    parser.add_argument("--scene_name", default="vinyes_sam3_200")
    parser.add_argument("--label_mode", choices=["instance", "semantic"], default="instance")
    parser.set_defaults(link=True)
    parser.add_argument("--link", dest="link", action="store_true", help="Symlink source image/sparse folders into the prepared scene.")
    parser.add_argument("--copy", dest="link", action="store_false", help="Copy source image/sparse folders instead of symlinking them.")
    parser.add_argument("--instance_min_area", type=int, default=15000)
    parser.add_argument("--instance_iou_threshold", type=float, default=0.10)
    parser.add_argument(
        "--instance_classes",
        nargs="*",
        default=None,
        help=(
            "Optional semantic class names to keep as per-object SAM3 track IDs. "
            "Other tracked pixels are folded back to their semantic class IDs."
        ),
    )
    parser.add_argument("--max_overlay_frames", type=int, default=12)
    parser.add_argument("--config_out", type=Path, default=Path("config/gaussian_dataset/vinyes_sam3_200.json"))
    return parser.parse_args()


def resolve_sam3_dir(path: Path) -> Path:
    if path.exists():
        return path
    for fallback in [
        path.parent / "sam3_video_vine_semantic",
        Path("vineyard_posematch/sam3_video_vine_semantic"),
        Path("../vineyard_posematch/sam3_video_vine_semantic"),
    ]:
        if fallback.exists():
            print(f"Warning: {path} not found; using {fallback}")
            return fallback
    return path


def read_registered_images(images_txt: Path) -> list[str]:
    names = []
    for line in images_txt.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 10:
            names.append(parts[9])
    if not names:
        raise RuntimeError(f"No registered images found in {images_txt}")
    return names


def frame_number(stem: str) -> int | None:
    match = re.search(r"(\d+)$", stem)
    return int(match.group(1)) if match else None


def band_key(stem: str) -> str:
    return stem.split("_", 1)[0]


def active_channels_for(stem: str) -> list[int]:
    key = band_key(stem)
    if key in BAND_CHANNELS:
        return BAND_CHANNELS[key]
    raise KeyError(f"Cannot infer active channels for {stem}")


def link_or_copy(src: Path, dst: Path, use_link: bool) -> None:
    if dst.exists() or dst.is_symlink():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if use_link:
        os.symlink(os.path.abspath(src), dst)
    elif src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)


def load_class_map(sam3_dir: Path) -> dict[str, int]:
    class_map_path = sam3_dir / "metadata" / "class_map.json"
    if class_map_path.exists():
        return json.loads(class_map_path.read_text())
    return {"background": 0, "vine_plant": 1}


def load_class_colors(sam3_dir: Path, class_map: dict[str, int]) -> dict[int, tuple[int, int, int]]:
    colors_path = sam3_dir / "metadata" / "class_colors.json"
    if colors_path.exists():
        raw = json.loads(colors_path.read_text())
        return {int(k): tuple(v) for k, v in raw.items()}
    return {idx: (idx * 37 % 255, idx * 67 % 255, idx * 97 % 255) for idx in class_map.values()}


def semantic_mask_path(sam3_dir: Path, image_name: str) -> Path | None:
    stem = Path(image_name).stem
    indexed = sam3_dir / "semantic_index_masks" / f"{stem}.png"
    if indexed.exists():
        return indexed
    color = sam3_dir / "semantic_color_masks" / f"{stem}.png"
    if color.exists():
        return color
    binary = sam3_dir / "vine_binary_masks" / f"{stem}.png"
    if binary.exists():
        return binary
    return None


def instance_mask_path(sam3_dir: Path, image_name: str) -> Path | None:
    stem = Path(image_name).stem
    indexed = sam3_dir / "semantic_instance_masks" / f"{stem}.png"
    if indexed.exists():
        return indexed
    return None


def color_to_index_mask(path: Path, colors: dict[int, tuple[int, int, int]]) -> np.ndarray:
    arr = np.array(Image.open(path))
    if arr.ndim == 2:
        if arr.max() > 8:
            return (arr > 0).astype(np.uint8)
        return arr.astype(np.uint16)
    rgb = arr[..., :3]
    out = np.zeros(rgb.shape[:2], dtype=np.uint16)
    for cls_id, color in colors.items():
        out[(rgb == np.array(color, dtype=np.uint8)).all(axis=-1)] = int(cls_id)
    return out


def save_label_png(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if arr.max(initial=0) > 255:
        Image.fromarray(arr.astype(np.uint16), mode="I;16").save(path)
    else:
        Image.fromarray(arr.astype(np.uint8), mode="L").save(path)


def load_instance_label_map(sam3_dir: Path) -> dict[str, dict] | None:
    path = sam3_dir / "metadata" / "instance_label_map.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def compact_tracked_instance_masks(
    tracked_instance_items,
    object_dir: Path,
    sam3_instance_map: dict[str, dict],
    class_map: dict[str, int],
    keep_instance_classes: set[str],
):
    class_entries = {
        str(class_id): {
            "class_id": int(class_id),
            "class_name": class_name,
            "label": class_name,
            "source": "semantic_class",
        }
        for class_name, class_id in sorted(class_map.items(), key=lambda item: item[1])
    }
    next_id = max(int(v) for v in class_map.values()) + 1
    remap = {0: 0}
    compact_map = class_entries.copy()
    kept_by_class = Counter()
    folded_by_class = Counter()

    for raw_id_str, entry in sorted(sam3_instance_map.items(), key=lambda item: int(item[0])):
        raw_id = int(raw_id_str)
        if raw_id == 0:
            continue
        class_name = entry.get("class_name", "")
        class_id = int(entry.get("class_id", class_map.get(class_name, 0)))
        if class_name in keep_instance_classes:
            compact_id = next_id
            next_id += 1
            remap[raw_id] = compact_id
            kept_by_class[class_name] += 1
            compact_entry = dict(entry)
            compact_entry.update({
                "class_id": class_id,
                "class_name": class_name,
                "source_sam3_instance_id": raw_id,
                "source": entry.get("source", "sam3_video_tracker"),
            })
            compact_map[str(compact_id)] = compact_entry
        else:
            remap[raw_id] = class_id
            folded_by_class[class_name] += 1

    lut = np.zeros(max(remap.keys()) + 1, dtype=np.uint16)
    for raw_id, compact_id in remap.items():
        lut[raw_id] = compact_id

    for image_name, path in tracked_instance_items:
        inst = np.array(Image.open(path))
        if inst.max(initial=0) >= len(lut):
            raise ValueError(f"{path} contains an instance id outside the SAM3 label map")
        save_label_png(lut[inst], object_dir / image_name)

    report = {
        "source": "sam3_video_tracker_compact",
        "keep_instance_classes": sorted(keep_instance_classes),
        "kept_instances_by_class": dict(sorted(kept_by_class.items())),
        "folded_tracks_by_class": dict(sorted(folded_by_class.items())),
        "num_rgb_instance_masks_aligned": len(tracked_instance_items),
        "raw_num_instances": len(sam3_instance_map) - 1,
        "compact_num_classes": max(int(k) for k in compact_map.keys()) + 1,
    }
    return compact_map, report


def connected_components(mask: np.ndarray):
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for instance mask generation") from exc
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    for comp_id in range(1, num):
        area = int(stats[comp_id, cv2.CC_STAT_AREA])
        x = int(stats[comp_id, cv2.CC_STAT_LEFT])
        y = int(stats[comp_id, cv2.CC_STAT_TOP])
        w = int(stats[comp_id, cv2.CC_STAT_WIDTH])
        h = int(stats[comp_id, cv2.CC_STAT_HEIGHT])
        yield comp_id, area, labels == comp_id, (x, y, x + w, y + h)


def bbox_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter / max(1, area_a + area_b - inter)


def build_instance_masks(mask_items, out_dir: Path, class_id_to_name: dict[int, str], min_area: int, iou_threshold: float):
    out_dir.mkdir(parents=True, exist_ok=True)
    next_instance_id = 1
    prev_by_class: dict[int, list[dict]] = defaultdict(list)
    instance_map = {
        "0": {"class_id": 0, "class_name": class_id_to_name.get(0, "background"), "label": "background"}
    }
    report_rows = []

    for image_name, semantic_path in sorted(mask_items, key=lambda item: (frame_number(Path(item[0]).stem) or 0, item[0])):
        sem = np.array(Image.open(semantic_path))
        out = np.zeros_like(sem, dtype=np.uint16)
        current_by_class: dict[int, list[dict]] = defaultdict(list)
        created = reused = kept_components = 0

        for class_id in sorted(int(v) for v in np.unique(sem) if int(v) != 0):
            comps = []
            for _, area, comp_mask, bbox in connected_components(sem == class_id):
                if area < min_area:
                    continue
                comps.append((area, comp_mask, bbox))
            comps.sort(reverse=True, key=lambda item: item[0])

            available_prev = prev_by_class.get(class_id, []).copy()
            for area, comp_mask, bbox in comps:
                best_idx = -1
                best_iou = 0.0
                for idx, prev in enumerate(available_prev):
                    iou = bbox_iou(bbox, prev["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = idx
                if best_idx >= 0 and best_iou >= iou_threshold:
                    instance_id = available_prev.pop(best_idx)["instance_id"]
                    reused += 1
                else:
                    instance_id = next_instance_id
                    next_instance_id += 1
                    class_name = class_id_to_name.get(class_id, f"class_{class_id}")
                    instance_map[str(instance_id)] = {
                        "class_id": class_id,
                        "class_name": class_name,
                        "label": f"{class_name}_{instance_id:04d}",
                        "source": "weak_connected_component_tracking",
                    }
                    created += 1
                out[comp_mask] = instance_id
                current_by_class[class_id].append({"instance_id": instance_id, "bbox": bbox, "area": area})
                kept_components += 1

        save_label_png(out, out_dir / image_name)
        prev_by_class = current_by_class
        report_rows.append({
            "image_name": image_name,
            "kept_components": kept_components,
            "reused_instances": reused,
            "created_instances": created,
            "max_instance_id": next_instance_id - 1,
        })

    return instance_map, report_rows


def make_contact_sheet(rows, scene_dir: Path, labels_dir: Path, out_path: Path, colors: dict[int, tuple[int, int, int]], max_frames: int) -> None:
    samples = rows[:max_frames]
    if not samples:
        return
    tiles = []
    for row in samples:
        img = Image.open(scene_dir / "images" / row["image_name"]).convert("RGB")
        lbl = np.array(Image.open(labels_dir / row["image_name"]))
        color = np.zeros((*lbl.shape, 3), dtype=np.uint8)
        for idx in np.unique(lbl):
            if idx == 0:
                continue
            fallback = ((int(idx) * 37) % 255, (int(idx) * 67) % 255, (int(idx) * 97) % 255)
            color[lbl == idx] = colors.get(int(idx), fallback)
        overlay = Image.blend(img.resize((370, 187)), Image.fromarray(color).resize((370, 187), Image.NEAREST), 0.45)
        draw = ImageDraw.Draw(overlay)
        draw.text((8, 8), row["image_name"], fill=(255, 255, 255))
        tiles.append(overlay)
    cols = 3
    rows_n = int(np.ceil(len(tiles) / cols))
    sheet = Image.new("RGB", (cols * 370, rows_n * 187), (0, 0, 0))
    for idx, tile in enumerate(tiles):
        sheet.paste(tile, ((idx % cols) * 370, (idx // cols) * 187))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    source_scene = args.source_scene
    sam3_dir = resolve_sam3_dir(args.sam3_dir)
    output_scene = args.output_scene
    metadata_dir = output_scene / "metadata"

    if not (source_scene / "sparse" / "0" / "images.txt").exists():
        raise FileNotFoundError(f"COLMAP images.txt not found under {source_scene}")
    if not sam3_dir.exists():
        raise FileNotFoundError(f"SAM3 directory not found: {sam3_dir}")

    output_scene.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    for name in ["images", "images_rgb", "sparse"]:
        src = source_scene / name
        if src.exists():
            link_or_copy(src, output_scene / name, args.link)
    for name in ["frame_info.json", "band_info.json", "partial_channels_summary.json"]:
        src = source_scene / name
        if src.exists():
            shutil.copy2(src, output_scene / name)

    class_map = load_class_map(sam3_dir)
    class_id_to_name = {int(v): k for k, v in class_map.items()}
    class_colors = load_class_colors(sam3_dir, class_map)
    keep_instance_classes = set(args.instance_classes or [])
    unknown_instance_classes = sorted(keep_instance_classes - set(class_map.keys()))
    if unknown_instance_classes:
        raise ValueError(f"Unknown --instance_classes values: {unknown_instance_classes}")
    registered = read_registered_images(output_scene / "sparse" / "0" / "images.txt")
    registered_stems = {Path(name).stem for name in registered}

    active_channels = {}
    summary = Counter()
    alignment_rows = []
    semantic_items = []
    tracked_instance_items = []
    semantic_dir = output_scene / "semantic_mask"
    semantic_dir.mkdir(parents=True, exist_ok=True)
    tracked_instance_dir = output_scene / "sam3_instance_mask"
    tracked_instance_dir.mkdir(parents=True, exist_ok=True)

    for image_name in registered:
        stem = Path(image_name).stem
        channels = active_channels_for(stem)
        active_channels[stem] = channels
        summary[band_key(stem)] += 1
        is_rgb = band_key(stem) == "rgbp"
        mask_path = semantic_mask_path(sam3_dir, image_name) if is_rgb else None
        tracked_instance_path = instance_mask_path(sam3_dir, image_name) if is_rgb else None
        row = {
            "image_name": image_name,
            "band": band_key(stem),
            "registered": True,
            "active_channels": " ".join(str(c) for c in channels),
            "has_sam3_mask": mask_path is not None,
            "has_sam3_instance_mask": tracked_instance_path is not None,
            "mask_source": str(mask_path) if mask_path else "",
            "instance_mask_source": str(tracked_instance_path) if tracked_instance_path else "",
            "dimension_match": "",
            "warning": "",
        }
        if mask_path is not None:
            mask = color_to_index_mask(mask_path, class_colors)
            image_size = Image.open(output_scene / "images" / image_name).size
            if mask.shape[::-1] != image_size:
                row["dimension_match"] = False
                row["warning"] = f"mask size {mask.shape[::-1]} != image size {image_size}"
            else:
                row["dimension_match"] = True
            out_mask = semantic_dir / image_name
            save_label_png(mask, out_mask)
            semantic_items.append((image_name, out_mask))
        if tracked_instance_path is not None:
            inst = np.array(Image.open(tracked_instance_path))
            image_size = Image.open(output_scene / "images" / image_name).size
            if inst.shape[::-1] != image_size:
                row["dimension_match"] = False
                row["warning"] = f"instance mask size {inst.shape[::-1]} != image size {image_size}"
            out_inst = tracked_instance_dir / image_name
            save_label_png(inst, out_inst)
            tracked_instance_items.append((image_name, out_inst))
        alignment_rows.append(row)

    all_sam_masks = {
        p.stem for sub in ["semantic_index_masks", "semantic_color_masks", "vine_binary_masks"]
        for p in (sam3_dir / sub).glob("*.png")
    }
    missing_registered = sorted(all_sam_masks - registered_stems)
    for stem in missing_registered:
        alignment_rows.append({
            "image_name": f"{stem}.png",
            "band": band_key(stem),
            "registered": False,
            "active_channels": "",
            "has_sam3_mask": True,
            "has_sam3_instance_mask": False,
            "mask_source": "SAM3 mask not present in COLMAP registered image list",
            "instance_mask_source": "",
            "dimension_match": "",
            "warning": "mask_without_registered_colmap_image",
        })

    object_dir = output_scene / "object_mask"
    if object_dir.is_symlink():
        object_dir.unlink()
    elif object_dir.exists():
        shutil.rmtree(object_dir)
    object_dir.mkdir(parents=True, exist_ok=True)

    if args.label_mode == "instance":
        sam3_instance_map = load_instance_label_map(sam3_dir)
        if sam3_instance_map is not None and tracked_instance_items:
            if args.instance_classes is None:
                for image_name, path in tracked_instance_items:
                    shutil.copy2(path, object_dir / image_name)
                instance_map = sam3_instance_map
                report_path = sam3_dir / "metadata" / "instance_tracking_report.json"
                report = json.loads(report_path.read_text()) if report_path.exists() else {}
                report.update({
                    "source": "sam3_video_tracker",
                    "num_rgb_instance_masks_aligned": len(tracked_instance_items),
                    "raw_num_instances": len(instance_map) - 1,
                })
            else:
                instance_map, report = compact_tracked_instance_masks(
                    tracked_instance_items,
                    object_dir,
                    sam3_instance_map,
                    class_map,
                    keep_instance_classes,
                )
            num_classes = max(int(k) for k in instance_map.keys()) + 1
            (metadata_dir / "instance_label_map.json").write_text(json.dumps(instance_map, indent=2))
            (metadata_dir / "instance_tracking_report.json").write_text(json.dumps(report, indent=2))
        else:
            instance_map, instance_rows = build_instance_masks(
                semantic_items,
                object_dir,
                class_id_to_name,
                min_area=args.instance_min_area,
                iou_threshold=args.instance_iou_threshold,
            )
            num_classes = max(int(k) for k in instance_map.keys()) + 1
            (metadata_dir / "instance_label_map.json").write_text(json.dumps(instance_map, indent=2))
            (metadata_dir / "instance_tracking_report.json").write_text(json.dumps({
                "source": "semantic_index_masks_connected_components",
                "warning": "Weak pixel-space association; SAM3 semantic_instance_masks were unavailable.",
                "min_area": args.instance_min_area,
                "iou_threshold": args.instance_iou_threshold,
                "num_instances": num_classes - 1,
                "frames": instance_rows,
            }, indent=2))
    else:
        for image_name, path in semantic_items:
            shutil.copy2(path, object_dir / image_name)
        num_classes = max(class_map.values()) + 1

    (metadata_dir / "active_channels.json").write_text(json.dumps(active_channels, indent=2))
    (metadata_dir / "class_map.json").write_text(json.dumps(class_map, indent=2))
    (metadata_dir / "class_colors.json").write_text(json.dumps({str(k): list(v) for k, v in class_colors.items()}, indent=2))
    (metadata_dir / "registered_images_summary.json").write_text(json.dumps({
        "scene_name": args.scene_name,
        "source_scene": str(source_scene),
        "sam3_dir": str(sam3_dir),
        "label_mode": args.label_mode,
        "instance_classes": sorted(keep_instance_classes) if args.instance_classes is not None else None,
        "num_registered_images": len(registered),
        "registered_per_band": dict(sorted(summary.items())),
        "num_rgb_registered": summary.get("rgbp", 0),
        "num_rgb_masks_aligned": len(semantic_items),
        "num_rgb_instance_masks_aligned": len(tracked_instance_items),
        "num_sam3_masks_not_registered": len(missing_registered),
        "channels": BAND_CHANNELS,
    }, indent=2))
    (metadata_dir / "mask_alignment_report.json").write_text(json.dumps(alignment_rows, indent=2))
    write_csv(metadata_dir / "mask_alignment_report.csv", alignment_rows)

    make_contact_sheet(
        [row for row in alignment_rows if row["registered"] and row["has_sam3_mask"]],
        output_scene,
        object_dir,
        metadata_dir / "mask_contact_sheet.jpg",
        class_colors,
        args.max_overlay_frames,
    )

    config = {
        "densify_until_iter": 6000,
        "densify_grad_threshold": 0.00005,
        "num_classes": num_classes,
        "num_objects": 16,
        "max_num_points": 1000000,
        "reg3d_interval": 5,
        "reg3d_k": 5,
        "reg3d_lambda_val": 2,
        "reg3d_max_points": 150000,
        "reg3d_sample_size": 500,
        "use_color_embed": True,
        "color_embed_dim": 32,
        "color_decoder_hidden_dim": 128,
        "color_decoder_num_hidden_layers": 3,
        "color_decoder_lr": 0.001,
        "num_channels": 10,
        "single_channel_mode": False,
        "label_mode": args.label_mode,
    }
    args.config_out.parent.mkdir(parents=True, exist_ok=True)
    args.config_out.write_text(json.dumps(config, indent=4))

    print(f"Prepared scene: {output_scene}")
    print(f"Registered images per band: {dict(sorted(summary.items()))}")
    print(f"RGB masks aligned: {len(semantic_items)}")
    print(f"SAM3 masks not registered: {len(missing_registered)}")
    print(f"Label mode: {args.label_mode}; num_classes={num_classes}")
    print(f"Training config: {args.config_out}")


if __name__ == "__main__":
    main()
