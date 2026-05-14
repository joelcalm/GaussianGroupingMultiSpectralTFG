#!/usr/bin/env python3
"""Run SAM3 video semantic prompting for vineyard scenes and save:
- binary masks for all classes
- vine binary masks
- vine overlays
- semantic indexed masks
- semantic instance masks with stable SAM3 tracker IDs
- semantic color masks
- semantic overlays
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm


# ----------------------------
# Class setup
# ----------------------------

CLASS_MAP = {
    "background": 0,
    "vine_plant": 1,
    "wooden_post": 2,
    "ground": 3,
    "sky": 4,
    "tree": 5,
    "stone_wall": 6,
    "shrub_or_other_vegetation": 7,
    "building": 8,
}

# One prompt per class
CLASS_PROMPTS = {
    "vine_plant": "grapevine trunk and branches",
    "wooden_post": "wooden vineyard post",
    "ground": "bare soil ground in a vineyard",
    "sky": "blue sky",
    "tree": "tree canopy or tree trunk",
    "stone_wall": "stone wall made of rocks",
    "shrub_or_other_vegetation": "shrub, bush, or other vegetation",
    "building": "building wall or building facade",
}

# RGB colors for visualizations
CLASS_COLORS = {
    0: (0, 0, 0),          # background - black
    1: (0, 255, 0),        # vine - green
    2: (139, 69, 19),      # wooden_post - brown
    3: (210, 180, 140),    # ground - tan
    4: (135, 206, 235),    # sky - sky blue
    5: (34, 139, 34),      # tree - dark green
    6: (128, 128, 128),    # stone_wall - gray
    7: (50, 205, 50),      # shrub/other vegetation - lime green
    8: (255, 182, 193),    # building - pink-ish
}

# Merge order = low priority first, high priority last
# Later classes overwrite earlier ones.
MERGE_ORDER = [
    "sky",
    "ground",
    "building",
    "stone_wall",
    "tree",
    "shrub_or_other_vegetation",
    "wooden_post",
    "vine_plant",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

CONF_THRESHOLD = 0.25
MASK_THRESHOLD = 0.5
MIN_COMPONENT_AREA = 20
MORPH_KERNEL_SIZE = 11


# ----------------------------
# Args
# ----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SAM3 video semantic prompting on an ordered image sequence."
    )
    parser.add_argument("--images_dir", type=Path, default=Path("vinyes_partial200/images_rgb"))
    parser.add_argument("--output_dir", type=Path, default=Path("sam3_video_vine_semantic"))
    parser.add_argument("--model", type=str, default="weights/sam3.pt")
    parser.add_argument(
        "--box",
        type=float,
        nargs=4,
        action="append",
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Optional initial-frame box prompt in original image pixels. Used only for vine_plant.",
    )
    parser.add_argument("--device", type=str, default="2")
    parser.add_argument("--half", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=CONF_THRESHOLD)

    parser.add_argument("--score_threshold_detection", type=float, default=0.5)
    parser.add_argument("--new_det_thresh", type=float, default=0.0)
    parser.add_argument("--assoc_iou_thresh", type=float, default=0.5)
    parser.add_argument("--trk_assoc_iou_thresh", type=float, default=0.5)
    parser.add_argument("--init_trk_keep_alive", type=int, default=10)
    parser.add_argument("--max_trk_keep_alive", type=int, default=10)
    parser.add_argument("--min_trk_keep_alive", type=int, default=-4)
    parser.add_argument(
        "--max_num_objects",
        type=int,
        default=96,
        help="Maximum active SAM3 video track objects per semantic class. -1 disables the cap.",
    )

    parser.add_argument("--mask_threshold", type=float, default=MASK_THRESHOLD)
    parser.add_argument("--min_component_area", type=int, default=MIN_COMPONENT_AREA)
    parser.add_argument("--morph_kernel_size", type=int, default=MORPH_KERNEL_SIZE)
    parser.add_argument(
        "--dilate_kernel_size",
        type=int,
        default=0,
        help="Optional final dilation kernel size. 0 disables it.",
    )
    parser.add_argument("--dilate_iterations", type=int, default=1)

    parser.add_argument("--start_index", type=int, default=None)
    parser.add_argument("--end_index", type=int, default=None)
    parser.add_argument("--max_images", type=int, default=None)

    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--keep_video", type=Path, default=None)
    parser.add_argument("--overlay_alpha", type=float, default=0.45)

    return parser.parse_args()


# ----------------------------
# Utility functions
# ----------------------------

def frame_number(path: Path) -> int | None:
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    return int(digits) if digits else None


def load_images(
    images_dir: Path,
    start_index: int | None = None,
    end_index: int | None = None,
    max_images: int | None = None,
) -> list[Path]:
    image_paths = sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if start_index is not None or end_index is not None:
        filtered = []
        for path in image_paths:
            n = frame_number(path)
            if n is None:
                continue
            if start_index is not None and n < start_index:
                continue
            if end_index is not None and n > end_index:
                continue
            filtered.append(path)
        image_paths = filtered

    if max_images is not None:
        image_paths = image_paths[:max_images]

    if not image_paths:
        raise FileNotFoundError(f"No images selected from {images_dir}")

    return image_paths


def make_video(image_paths: list[Path], video_path: Path, fps: float) -> None:
    first = cv2.imread(str(image_paths[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise ValueError(f"Could not read image: {image_paths[0]}")

    height, width = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {video_path}")

    try:
        for image_path in image_paths:
            frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError(f"Could not read image: {image_path}")
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(frame)
    finally:
        writer.release()


def build_video_predictor(args: argparse.Namespace) -> Any:
    from ultralytics.models.sam import SAM3VideoSemanticPredictor

    return SAM3VideoSemanticPredictor(
        overrides={
            "conf": args.conf,
            "task": "segment",
            "mode": "predict",
            "model": args.model,
            "device": args.device,
            "half": args.half,
            "imgsz": args.imgsz,
            "save": False,
            "verbose": False,
        },
        score_threshold_detection=args.score_threshold_detection,
        new_det_thresh=args.new_det_thresh,
        assoc_iou_thresh=args.assoc_iou_thresh,
        trk_assoc_iou_thresh=args.trk_assoc_iou_thresh,
        init_trk_keep_alive=args.init_trk_keep_alive,
        max_trk_keep_alive=args.max_trk_keep_alive,
        min_trk_keep_alive=args.min_trk_keep_alive,
        max_num_objects=args.max_num_objects,
    )


def result_to_mask(result: Any, original_shape: tuple[int, int], mask_threshold: float) -> np.ndarray:
    mask_union = np.zeros(original_shape, dtype=np.uint8)
    masks_obj = getattr(result, "masks", None)
    if masks_obj is None or masks_obj.data is None:
        return mask_union

    masks = masks_obj.data
    if hasattr(masks, "detach"):
        masks = masks.detach().cpu().numpy()
    masks = np.asarray(masks)

    if masks.ndim == 2:
        masks = masks[None]

    for mask in masks:
        mask = (mask > mask_threshold).astype(np.uint8)
        if mask.shape[:2] != original_shape:
            mask = cv2.resize(
                mask,
                (original_shape[1], original_shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        mask_union |= mask

    return mask_union


def _result_masks_array(result: Any) -> np.ndarray | None:
    masks_obj = getattr(result, "masks", None)
    if masks_obj is None or masks_obj.data is None:
        return None
    masks = masks_obj.data
    if hasattr(masks, "detach"):
        masks = masks.detach().cpu().numpy()
    masks = np.asarray(masks)
    if masks.ndim == 2:
        masks = masks[None]
    return masks


def _result_boxes_array(result: Any) -> np.ndarray:
    boxes_obj = getattr(result, "boxes", None)
    if boxes_obj is None or boxes_obj.data is None:
        return np.empty((0, 0), dtype=np.float32)
    boxes = boxes_obj.data
    if hasattr(boxes, "detach"):
        boxes = boxes.detach().cpu().numpy()
    return np.asarray(boxes)


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return (0, 0, 0, 0)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def result_to_instances(
    result: Any,
    original_shape: tuple[int, int],
    mask_threshold: float,
) -> list[dict[str, Any]]:
    """Return per-object masks and SAM3 tracker IDs from one Results object.

    Ultralytics SAM3 video postprocess stores boxes as:
    [x1, y1, x2, y2, track_id, score, class].
    """
    masks = _result_masks_array(result)
    if masks is None:
        return []

    boxes = _result_boxes_array(result)
    instances = []
    for idx, raw_mask in enumerate(masks):
        mask = (raw_mask > mask_threshold).astype(np.uint8)
        if mask.shape[:2] != original_shape:
            mask = cv2.resize(
                mask,
                (original_shape[1], original_shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        if not mask.any():
            continue

        if idx < len(boxes) and boxes.shape[1] >= 7:
            track_id = int(boxes[idx, 4])
            score = float(boxes[idx, 5])
            pred_cls = int(boxes[idx, 6])
            bbox = tuple(int(round(v)) for v in boxes[idx, :4])
        elif idx < len(boxes) and boxes.shape[1] >= 6:
            track_id = int(boxes[idx, 4])
            score = float(boxes[idx, 5])
            pred_cls = 0
            bbox = tuple(int(round(v)) for v in boxes[idx, :4])
        else:
            track_id = idx + 1
            score = 1.0
            pred_cls = 0
            bbox = mask_bbox(mask)

        instances.append({
            "local_mask_index": idx,
            "track_id": track_id,
            "score": score,
            "pred_cls": pred_cls,
            "bbox": bbox,
            "mask": mask,
        })

    return instances


def remove_small_components(mask: np.ndarray, min_component_area: int) -> np.ndarray:
    if min_component_area <= 0:
        return mask

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    cleaned = np.zeros_like(mask, dtype=np.uint8)

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_component_area:
            cleaned[labels == label] = 1

    return cleaned


def postprocess_mask(
    raw_mask: np.ndarray,
    output_shape: tuple[int, int],
    min_component_area: int,
    morph_kernel_size: int,
) -> np.ndarray:
    mask = (raw_mask > 0).astype(np.uint8)

    if mask.shape[:2] != output_shape:
        mask = cv2.resize(
            mask,
            (output_shape[1], output_shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    if morph_kernel_size > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (morph_kernel_size, morph_kernel_size),
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    mask = remove_small_components(mask, min_component_area)
    return mask


def maybe_dilate_mask(mask: np.ndarray, kernel_size: int, iterations: int) -> np.ndarray:
    if kernel_size <= 1 or iterations <= 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.dilate(mask, kernel, iterations=iterations)


def save_binary_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), (mask > 0).astype(np.uint8) * 255)


def save_index_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), mask.astype(np.uint8))


def save_instance_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), mask.astype(np.uint16))


def colorize_semantic_mask(
    index_mask: np.ndarray,
    class_colors: dict[int, tuple[int, int, int]]
) -> np.ndarray:
    color = np.zeros((index_mask.shape[0], index_mask.shape[1], 3), dtype=np.uint8)
    for class_id, rgb in class_colors.items():
        color[index_mask == class_id] = rgb
    return color


def save_color_mask(mask_rgb: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR))


def save_overlay(image_bgr: np.ndarray, binary_mask: np.ndarray, path: Path, alpha: float = 0.45) -> None:
    """Green overlay for vine binary mask."""
    path.parent.mkdir(parents=True, exist_ok=True)
    overlay = image_bgr.copy()
    green = np.zeros_like(image_bgr)
    green[:, :] = (0, 255, 0)  # BGR green
    mask_bool = binary_mask > 0
    overlay[mask_bool] = cv2.addWeighted(image_bgr, 1 - alpha, green, alpha, 0)[mask_bool]
    cv2.imwrite(str(path), overlay)


def save_semantic_overlay(
    image_bgr: np.ndarray,
    color_mask_rgb: np.ndarray,
    path: Path,
    alpha: float = 0.45,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    color_mask_bgr = cv2.cvtColor(color_mask_rgb, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(image_bgr, 1 - alpha, color_mask_bgr, alpha, 0)
    cv2.imwrite(str(path), overlay)


def write_metadata(output_dir: Path) -> None:
    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    with (metadata_dir / "class_map.json").open("w", encoding="utf-8") as f:
        json.dump(CLASS_MAP, f, indent=2)

    with (metadata_dir / "class_prompts.json").open("w", encoding="utf-8") as f:
        json.dump(CLASS_PROMPTS, f, indent=2)

    with (metadata_dir / "class_colors.json").open("w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in CLASS_COLORS.items()}, f, indent=2)

    with (metadata_dir / "merge_order.json").open("w", encoding="utf-8") as f:
        json.dump(MERGE_ORDER, f, indent=2)


# ----------------------------
# Main processing
# ----------------------------

def run_class_predictions(
    args: argparse.Namespace,
    video_path: Path,
    image_paths: list[Path],
    temp_class_root: Path,
    temp_instance_root: Path,
    instance_label_map: dict[str, dict[str, Any]],
    track_to_instance_id: dict[tuple[str, int], int],
) -> None:
    """
    For each class:
      - run SAM3 on the full video
      - temporarily save one binary mask per frame in temp_class_root/<class_name>/
      - only save final vine_binary_masks/ and vine_overlays/ permanently
    """
    vine_mask_dir = args.output_dir / "vine_binary_masks"
    vine_overlay_dir = args.output_dir / "vine_overlays"

    for class_name in MERGE_ORDER:
        prompt = CLASS_PROMPTS[class_name]
        predictor = build_video_predictor(args)

        predict_kwargs: dict[str, Any] = {
            "source": str(video_path),
            "text": [prompt],
            "stream": True,
        }

        # Optional box prompt only for vines
        if class_name == "vine_plant" and args.box:
            predict_kwargs["bboxes"] = args.box

        results = predictor(**predict_kwargs)

        temp_out_dir = temp_class_root / class_name
        temp_out_dir.mkdir(parents=True, exist_ok=True)
        temp_instance_dir = temp_instance_root / class_name
        temp_instance_dir.mkdir(parents=True, exist_ok=True)

        desc = f"Class: {class_name}"

        for image_path, result in tqdm(
            zip(image_paths, results),
            total=len(image_paths),
            unit="frame",
            desc=desc,
        ):
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")

            class_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            instance_mask = np.zeros(image.shape[:2], dtype=np.uint16)
            for instance in result_to_instances(result, image.shape[:2], args.mask_threshold):
                mask = postprocess_mask(
                    instance["mask"],
                    image.shape[:2],
                    min_component_area=args.min_component_area,
                    morph_kernel_size=args.morph_kernel_size,
                )
                mask = maybe_dilate_mask(
                    mask,
                    args.dilate_kernel_size,
                    args.dilate_iterations,
                )
                if not mask.any():
                    continue

                key = (class_name, int(instance["track_id"]))
                if key not in track_to_instance_id:
                    global_instance_id = len(instance_label_map)
                    track_to_instance_id[key] = global_instance_id
                    instance_label_map[str(global_instance_id)] = {
                        "class_id": int(CLASS_MAP[class_name]),
                        "class_name": class_name,
                        "label": f"{class_name}_track_{int(instance['track_id']):04d}",
                        "source": "sam3_video_tracker",
                        "sam3_track_id": int(instance["track_id"]),
                    }
                else:
                    global_instance_id = track_to_instance_id[key]

                mask_bool = mask > 0
                class_mask[mask_bool] = 1
                instance_mask[mask_bool] = global_instance_id

            stem = image_path.stem

            # Temporary class mask, only used to build semantic outputs
            temp_mask_path = temp_out_dir / f"{stem}.png"
            save_binary_mask(class_mask, temp_mask_path)
            save_instance_mask(instance_mask, temp_instance_dir / f"{stem}.png")

            # Permanent outputs only for vine
            if class_name == "vine_plant":
                save_binary_mask(class_mask, vine_mask_dir / f"{stem}.png")
                save_overlay(
                    image,
                    class_mask,
                    vine_overlay_dir / f"{stem}.jpg",
                    alpha=args.overlay_alpha,
                )

def build_semantic_outputs(
    args: argparse.Namespace,
    image_paths: list[Path],
    temp_class_root: Path,
    temp_instance_root: Path,
) -> None:
    """
    Read temporary per-class binary masks and assemble:
      - semantic_index_masks/
      - semantic_instance_masks/
      - semantic_color_masks/
      - semantic_overlays/

    Temporary class masks are not saved in the final output folder.
    """
    semantic_index_dir = args.output_dir / "semantic_index_masks"
    semantic_instance_dir = args.output_dir / "semantic_instance_masks"
    semantic_color_dir = args.output_dir / "semantic_color_masks"
    semantic_overlay_dir = args.output_dir / "semantic_overlays"

    for image_path in tqdm(image_paths, desc="Assembling semantic masks", unit="frame"):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        h, w = image.shape[:2]
        semantic_mask = np.zeros((h, w), dtype=np.uint8)
        semantic_instance_mask = np.zeros((h, w), dtype=np.uint16)

        stem = image_path.stem

        # Low priority first, high priority last.
        # Later classes overwrite earlier classes.
        for class_name in MERGE_ORDER:
            class_id = CLASS_MAP[class_name]
            class_mask_path = temp_class_root / class_name / f"{stem}.png"

            if not class_mask_path.exists():
                continue

            class_mask = cv2.imread(str(class_mask_path), cv2.IMREAD_GRAYSCALE)
            if class_mask is None:
                continue

            class_mask = class_mask > 127
            semantic_mask[class_mask] = class_id
            class_instance_path = temp_instance_root / class_name / f"{stem}.png"
            if class_instance_path.exists():
                class_instance = cv2.imread(str(class_instance_path), cv2.IMREAD_UNCHANGED)
                if class_instance is not None:
                    class_instance = class_instance.astype(np.uint16)
                    instance_pixels = class_instance > 0
                    semantic_instance_mask[instance_pixels] = class_instance[instance_pixels]

        color_mask = colorize_semantic_mask(semantic_mask, CLASS_COLORS)

        save_index_mask(semantic_mask, semantic_index_dir / f"{stem}.png")
        save_instance_mask(semantic_instance_mask, semantic_instance_dir / f"{stem}.png")
        save_color_mask(color_mask, semantic_color_dir / f"{stem}.png")
        save_semantic_overlay(
            image,
            color_mask,
            semantic_overlay_dir / f"{stem}.jpg",
            alpha=args.overlay_alpha,
        )

def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_metadata(args.output_dir)

    image_paths = load_images(
        args.images_dir,
        start_index=args.start_index,
        end_index=args.end_index,
        max_images=args.max_images,
    )

    with tempfile.TemporaryDirectory(prefix="sam3_semantic_work_") as work_tmpdir:
        work_tmpdir = Path(work_tmpdir)

        temp_class_root = work_tmpdir / "temp_class_binary_masks"
        temp_class_root.mkdir(parents=True, exist_ok=True)
        temp_instance_root = work_tmpdir / "temp_class_instance_masks"
        temp_instance_root.mkdir(parents=True, exist_ok=True)

        instance_label_map: dict[str, dict[str, Any]] = {
            "0": {
                "class_id": 0,
                "class_name": "background",
                "label": "background",
                "source": "background",
            }
        }
        track_to_instance_id: dict[tuple[str, int], int] = {}

        if args.keep_video is not None:
            video_path = args.keep_video
            video_path.parent.mkdir(parents=True, exist_ok=True)
            make_video(image_paths, video_path, args.fps)
        else:
            video_path = work_tmpdir / "sequence.mp4"
            make_video(image_paths, video_path, args.fps)

        run_class_predictions(
            args=args,
            video_path=video_path,
            image_paths=image_paths,
            temp_class_root=temp_class_root,
            temp_instance_root=temp_instance_root,
            instance_label_map=instance_label_map,
            track_to_instance_id=track_to_instance_id,
        )

        build_semantic_outputs(
            args=args,
            image_paths=image_paths,
            temp_class_root=temp_class_root,
            temp_instance_root=temp_instance_root,
        )

    metadata_dir = args.output_dir / "metadata"
    with (metadata_dir / "instance_label_map.json").open("w", encoding="utf-8") as f:
        json.dump(instance_label_map, f, indent=2)
    with (metadata_dir / "sam3_track_id_map.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                f"{class_name}:{track_id}": instance_id
                for (class_name, track_id), instance_id in sorted(track_to_instance_id.items())
            },
            f,
            indent=2,
        )
    with (metadata_dir / "instance_tracking_report.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "source": "sam3_video_tracker",
                "num_instances": len(instance_label_map) - 1,
                "note": "Global IDs are assigned from SAM3 per-class video track IDs.",
            },
            f,
            indent=2,
        )

    print(f"Done. Wrote outputs to: {args.output_dir}")

if __name__ == "__main__":
    main()
