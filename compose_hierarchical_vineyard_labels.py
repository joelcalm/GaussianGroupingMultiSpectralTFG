#!/usr/bin/env python3
"""Compose hierarchical vineyard object/part labels for Gaussian training.

The trainer consumes one flat indexed PNG per supervised image. This script keeps
that interface, but writes metadata that records each flat ID as:
object_type + physical instance + part.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


DEFAULT_PART_CLASSES = {
    "leaf": "vine_leaf",
    "trunk": "vine_trunk",
}

DEFAULT_SEMANTIC_ORDER = [
    "ground",
    "sky",
    "tree",
    "stone_wall",
    "shrub_or_other_vegetation",
    "building",
]

TRAIN_CONFIG_DEFAULTS = {
    "densify_until_iter": 6000,
    "densify_grad_threshold": 0.00005,
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
    "label_mode": "hierarchical_composite",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene_dir", required=True, type=Path, help="Prepared scene containing images/, sparse/, metadata/.")
    parser.add_argument("--sam3_dir", required=True, type=Path, help="SAM3 output with class_instance_masks/ and class_binary_masks/.")
    parser.add_argument("--scene_name", default=None)
    parser.add_argument("--config_out", type=Path, default=None)
    parser.add_argument("--whole_vine_class", default="vine_plant")
    parser.add_argument("--post_class", default="wooden_post")
    parser.add_argument("--part_class", action="append", default=[], metavar="PART=CLASS", help="Part mapping, e.g. leaf=vine_leaf. Can be repeated.")
    parser.add_argument("--semantic_class", action="append", default=[], help="Semantic classes to retain as non-instance labels.")
    parser.add_argument("--association_dilate_pixels", type=int, default=7)
    parser.add_argument("--min_part_overlap_pixels", type=int, default=20)
    parser.add_argument("--min_instance_pixels", type=int, default=100)
    parser.add_argument("--max_overlay_frames", type=int, default=12)
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing object_mask directory.")
    return parser.parse_args()


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open() as f:
        return json.load(f)


def save_label_png(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if arr.max(initial=0) > 255:
        Image.fromarray(arr.astype(np.uint16), mode="I;16").save(path)
    else:
        Image.fromarray(arr.astype(np.uint8), mode="L").save(path)


def frame_number(path: Path) -> int:
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    return int(digits) if digits else 0


def parse_part_classes(items: list[str]) -> dict[str, str]:
    out = dict(DEFAULT_PART_CLASSES)
    for item in items:
        if "=" not in item:
            raise ValueError(f"--part_class must be PART=CLASS, got {item!r}")
        part, cls = item.split("=", 1)
        part = part.strip()
        cls = cls.strip()
        if not part or not cls:
            raise ValueError(f"Invalid --part_class value: {item!r}")
        out[part] = cls
    return out


def read_registered_rgb_images(scene_dir: Path) -> list[str]:
    images_txt = scene_dir / "sparse" / "0" / "images.txt"
    if not images_txt.exists():
        raise FileNotFoundError(images_txt)
    names = []
    for line in images_txt.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 10 and Path(parts[9]).stem.startswith("rgb"):
            names.append(parts[9])
    if not names:
        raise RuntimeError(f"No registered RGB images found in {images_txt}")
    return sorted(set(names), key=lambda name: (frame_number(Path(name)), name))


def mask_path(root: Path, class_name: str, image_name: str) -> Path:
    return root / class_name / f"{Path(image_name).stem}.png"


def load_mask(path: Path, shape: tuple[int, int] | None = None, dtype=np.uint16) -> np.ndarray:
    if not path.exists():
        if shape is None:
            raise FileNotFoundError(path)
        return np.zeros(shape, dtype=dtype)
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(dtype, copy=False)


def connected_components(mask: np.ndarray):
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for part-to-vine association") from exc
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    for comp_id in range(1, num):
        area = int(stats[comp_id, cv2.CC_STAT_AREA])
        yield area, labels == comp_id


def dilate_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
    if pixels <= 0:
        return mask
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for dilated part-to-vine association") from exc
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pixels * 2 + 1, pixels * 2 + 1))
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1) > 0


def choose_instance(component: np.ndarray, whole_inst: np.ndarray, args: argparse.Namespace) -> tuple[int, int]:
    overlap_ids, overlap_counts = np.unique(whole_inst[component], return_counts=True)
    pairs = [(int(i), int(c)) for i, c in zip(overlap_ids, overlap_counts) if int(i) > 0]
    if pairs:
        inst_id, count = max(pairs, key=lambda item: item[1])
        if count >= args.min_part_overlap_pixels:
            return inst_id, count

    expanded = dilate_mask(component, args.association_dilate_pixels)
    overlap_ids, overlap_counts = np.unique(whole_inst[expanded], return_counts=True)
    pairs = [(int(i), int(c)) for i, c in zip(overlap_ids, overlap_counts) if int(i) > 0]
    if not pairs:
        return 0, 0
    inst_id, count = max(pairs, key=lambda item: item[1])
    if count < args.min_part_overlap_pixels:
        return 0, count
    return inst_id, count


def id_color(idx: int) -> tuple[int, int, int]:
    if idx <= 0:
        return (0, 0, 0)
    return ((idx * 37) % 255, (idx * 67) % 255, (idx * 97) % 255)


def discover_instances(
    image_names: list[str],
    root: Path,
    class_name: str,
    min_pixels: int,
) -> list[int]:
    counts: Counter[int] = Counter()
    for image_name in image_names:
        path = mask_path(root, class_name, image_name)
        if not path.exists():
            continue
        arr = load_mask(path)
        ids, nums = np.unique(arr, return_counts=True)
        for inst_id, count in zip(ids, nums):
            inst_id = int(inst_id)
            if inst_id > 0:
                counts[inst_id] += int(count)
    return [inst_id for inst_id, count in sorted(counts.items()) if count >= min_pixels]


def build_label_maps(
    vine_source_ids: list[int],
    post_source_ids: list[int],
    part_classes: dict[str, str],
    semantic_classes: list[str],
) -> tuple[dict[str, int], dict[int, dict[str, Any]], dict[tuple[str, int, str], int]]:
    class_map = {"background": 0}
    for name in semantic_classes:
        class_map.setdefault(name, len(class_map))
    for part in part_classes:
        class_map.setdefault(f"vine_{part}", len(class_map))
    class_map.setdefault("vine_other", len(class_map))
    class_map.setdefault("wooden_post", len(class_map))

    instance_map: dict[int, dict[str, Any]] = {
        0: {
            "class_id": 0,
            "class_name": "background",
            "label": "background",
            "source": "background",
            "object_type": "background",
            "instance_id": "background",
            "part_id": "none",
        }
    }
    label_lookup: dict[tuple[str, int, str], int] = {}

    for name in semantic_classes:
        label_id = len(instance_map)
        instance_map[label_id] = {
            "class_id": class_map[name],
            "class_name": name,
            "label": name,
            "source": "semantic_class",
            "object_type": "background",
            "instance_id": "background",
            "part_id": "none",
        }
        label_lookup[("semantic", class_map[name], "none")] = label_id

    for vine_index, source_id in enumerate(vine_source_ids, start=1):
        instance_key = f"vine_{vine_index:04d}"
        for part in list(part_classes) + ["other"]:
            class_name = f"vine_{part}" if part != "other" else "vine_other"
            label_id = len(instance_map)
            label_lookup[("vine", source_id, part)] = label_id
            instance_map[label_id] = {
                "class_id": class_map[class_name],
                "class_name": class_name,
                "label": f"{instance_key}_{part}",
                "source": "hierarchical_composite",
                "object_type": "vine",
                "instance_id": instance_key,
                "instance_index": vine_index,
                "part_id": part,
                "source_whole_instance_id": source_id,
            }

    for post_index, source_id in enumerate(post_source_ids, start=1):
        instance_key = f"post_{post_index:04d}"
        label_id = len(instance_map)
        label_lookup[("post", source_id, "whole")] = label_id
        instance_map[label_id] = {
            "class_id": class_map["wooden_post"],
            "class_name": "wooden_post",
            "label": f"{instance_key}_whole",
            "source": "hierarchical_composite",
            "object_type": "post",
            "instance_id": instance_key,
            "instance_index": post_index,
            "part_id": "whole",
            "source_whole_instance_id": source_id,
        }

    return class_map, instance_map, label_lookup


def write_contact_sheet(scene_dir: Path, object_dir: Path, image_names: list[str], out_path: Path, max_frames: int) -> None:
    samples = image_names[:max_frames]
    if not samples:
        return
    tiles = []
    for image_name in samples:
        image = Image.open(scene_dir / "images" / image_name).convert("RGB")
        labels = np.array(Image.open(object_dir / image_name))
        color = np.zeros((*labels.shape, 3), dtype=np.uint8)
        for idx in np.unique(labels):
            idx = int(idx)
            color[labels == idx] = id_color(idx)
        overlay = Image.blend(image.resize((370, 187)), Image.fromarray(color).resize((370, 187), Image.NEAREST), 0.45)
        draw = ImageDraw.Draw(overlay)
        draw.text((8, 8), image_name, fill=(255, 255, 255))
        tiles.append(overlay)
    cols = 3
    rows = int(np.ceil(len(tiles) / cols))
    sheet = Image.new("RGB", (cols * 370, rows * 187), (0, 0, 0))
    for idx, tile in enumerate(tiles):
        sheet.paste(tile, ((idx % cols) * 370, (idx // cols) * 187))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    scene_dir = args.scene_dir
    sam3_dir = args.sam3_dir
    metadata_dir = scene_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    class_instance_root = sam3_dir / "class_instance_masks"
    class_binary_root = sam3_dir / "class_binary_masks"
    if not class_instance_root.is_dir() or not class_binary_root.is_dir():
        raise FileNotFoundError(
            f"Expected {class_instance_root} and {class_binary_root}. Run sam3_vine_video.py with --save_class_outputs."
        )

    part_classes = parse_part_classes(args.part_class)
    semantic_classes = args.semantic_class or [name for name in DEFAULT_SEMANTIC_ORDER if (load_json(sam3_dir / "metadata" / "class_map.json", {})).get(name) is not None]
    sam3_class_map = load_json(sam3_dir / "metadata" / "class_map.json", {})
    semantic_class_ids = {name: int(sam3_class_map[name]) for name in semantic_classes if name in sam3_class_map}

    image_names = read_registered_rgb_images(scene_dir)
    vine_source_ids = discover_instances(image_names, class_instance_root, args.whole_vine_class, args.min_instance_pixels)
    post_source_ids = discover_instances(image_names, class_instance_root, args.post_class, args.min_instance_pixels)
    if not vine_source_ids:
        raise RuntimeError(f"No vine instances found in class_instance_masks/{args.whole_vine_class}")

    class_map, instance_map, label_lookup = build_label_maps(vine_source_ids, post_source_ids, part_classes, semantic_classes)
    object_dir = scene_dir / "object_mask"
    if object_dir.exists() or object_dir.is_symlink():
        if not args.overwrite:
            raise FileExistsError(f"{object_dir} exists; pass --overwrite to replace it")
        if object_dir.is_symlink() or object_dir.is_file():
            object_dir.unlink()
        else:
            shutil.rmtree(object_dir)
    object_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    label_pixel_counts: Counter[int] = Counter()
    part_assignment_counts = defaultdict(Counter)
    semantic_dir = scene_dir / "semantic_mask"

    for image_name in image_names:
        image_size = Image.open(scene_dir / "images" / image_name).size
        shape = (image_size[1], image_size[0])
        out = np.zeros(shape, dtype=np.uint16)

        semantic_path = semantic_dir / image_name
        if semantic_path.exists():
            semantic = load_mask(semantic_path, shape=shape, dtype=np.uint16)
            for name, source_class_id in semantic_class_ids.items():
                label_id = label_lookup.get(("semantic", class_map[name], "none"))
                if label_id is not None:
                    out[semantic == source_class_id] = label_id

        whole_vine = load_mask(mask_path(class_instance_root, args.whole_vine_class, image_name), shape=shape, dtype=np.uint16)
        post = load_mask(mask_path(class_instance_root, args.post_class, image_name), shape=shape, dtype=np.uint16)

        for source_id in vine_source_ids:
            label_id = label_lookup[("vine", source_id, "other")]
            out[whole_vine == source_id] = label_id

        for source_id in post_source_ids:
            label_id = label_lookup[("post", source_id, "whole")]
            out[post == source_id] = label_id

        unassigned_parts = 0
        assigned_parts = 0
        for part, class_name in part_classes.items():
            part_mask = load_mask(mask_path(class_binary_root, class_name, image_name), shape=shape, dtype=np.uint8) > 0
            if not part_mask.any():
                continue
            for area, component in connected_components(part_mask):
                if area < args.min_part_overlap_pixels:
                    continue
                source_id, overlap = choose_instance(component, whole_vine, args)
                if source_id <= 0 or ("vine", source_id, part) not in label_lookup:
                    unassigned_parts += int(area)
                    part_assignment_counts[part]["unassigned_components"] += 1
                    continue
                out[component] = label_lookup[("vine", source_id, part)]
                assigned_parts += int(area)
                part_assignment_counts[part]["assigned_components"] += 1
                part_assignment_counts[part]["assigned_pixels"] += int(area)
                part_assignment_counts[part]["overlap_pixels"] += int(overlap)

        save_label_png(out, object_dir / image_name)
        unique, counts = np.unique(out, return_counts=True)
        for label_id, count in zip(unique, counts):
            label_pixel_counts[int(label_id)] += int(count)
        rows.append({
            "image_name": image_name,
            "vine_pixels": int(np.count_nonzero(whole_vine)),
            "post_pixels": int(np.count_nonzero(post)),
            "assigned_part_pixels": int(assigned_parts),
            "unassigned_part_pixels": int(unassigned_parts),
            "num_labels": int(len(unique)),
        })

    for label_id, count in label_pixel_counts.items():
        if label_id in instance_map:
            instance_map[label_id]["training_pixels"] = int(count)

    compact_instance_map = {str(k): v for k, v in sorted(instance_map.items())}
    metadata_payload = {
        "source": "compose_hierarchical_vineyard_labels.py",
        "scene_name": args.scene_name or scene_dir.name,
        "sam3_dir": str(sam3_dir),
        "label_mode": "hierarchical_composite",
        "flat_label_note": "Training uses one indexed label per pixel; hierarchy is represented in instance_label_map.json metadata.",
        "whole_vine_class": args.whole_vine_class,
        "post_class": args.post_class,
        "part_classes": part_classes,
        "semantic_classes": semantic_classes,
        "num_vine_instances": len(vine_source_ids),
        "num_post_instances": len(post_source_ids),
        "num_classes": max(instance_map) + 1,
        "num_flat_labels": max(instance_map) + 1,
        "vine_source_instance_ids": vine_source_ids,
        "post_source_instance_ids": post_source_ids,
        "part_assignment_counts": {k: dict(v) for k, v in part_assignment_counts.items()},
    }

    (metadata_dir / "class_map.json").write_text(json.dumps(class_map, indent=2))
    (metadata_dir / "class_colors.json").write_text(json.dumps({str(v): list(id_color(v)) for v in class_map.values()}, indent=2))
    (metadata_dir / "instance_label_map.json").write_text(json.dumps(compact_instance_map, indent=2))
    (metadata_dir / "hierarchical_label_schema.json").write_text(json.dumps(metadata_payload, indent=2))
    (metadata_dir / "hierarchical_label_report.json").write_text(json.dumps(rows, indent=2))
    write_csv(metadata_dir / "hierarchical_label_report.csv", rows)
    write_contact_sheet(scene_dir, object_dir, image_names, metadata_dir / "hierarchical_mask_contact_sheet.jpg", args.max_overlay_frames)

    summary_path = metadata_dir / "registered_images_summary.json"
    summary = load_json(summary_path, {})
    summary.update({
        "scene_name": args.scene_name or scene_dir.name,
        "label_mode": "hierarchical_composite",
        "num_rgb_hierarchical_masks": len(image_names),
        "num_vine_instances": len(vine_source_ids),
        "num_post_instances": len(post_source_ids),
        "num_flat_labels": max(instance_map) + 1,
    })
    summary_path.write_text(json.dumps(summary, indent=2))

    if args.config_out is not None:
        config = dict(TRAIN_CONFIG_DEFAULTS)
        config["num_classes"] = max(instance_map) + 1
        args.config_out.parent.mkdir(parents=True, exist_ok=True)
        args.config_out.write_text(json.dumps(config, indent=4))

    print(f"Wrote hierarchical object masks: {object_dir}")
    print(f"Vine instances: {len(vine_source_ids)}; post instances: {len(post_source_ids)}")
    print(f"Flat labels / num_classes: {max(instance_map) + 1}")
    if args.config_out is not None:
        print(f"Training config: {args.config_out}")


if __name__ == "__main__":
    main()
