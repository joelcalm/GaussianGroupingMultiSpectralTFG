#!/usr/bin/env python3
"""Run SAM3 video semantic prompts for the basement tabletop scene.

Writes colorized semantic masks under data/basement/color_masks.
"""

from __future__ import annotations

import argparse
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
    "table": 1,
    "colors_panel": 2,
    "iron_forge": 3,
    "mazinger_z_toy": 4,
    "among_us_purple_teddy": 5,
    "banana": 6,
    "kiwi": 7,
    "lemon": 8,
    "orange": 9,
    "green_dark_orange": 10,
}

# One prompt per class (background is implicit).
CLASS_PROMPTS = {
    "table": "black table surface or table cloth",
    "colors_panel": "color calibration chart or color checker panel",
    "iron_forge": "vintage clothes iron or cast iron",
    "mazinger_z_toy": "Mazinger Z robot toy figure",
    "among_us_purple_teddy": "purple Among Us plush toy",
    "banana": "banana fruit on the table",
    "kiwi": "kiwi fruit on the table",
    "lemon": "lemon fruit on the table",
    "orange": "orange fruit on the table",
    "green_dark_orange": "green or dark orange fruit on the table",
}

# RGB colors for semantic visualization
CLASS_COLORS = {
    0: (0, 0, 0),
    1: (70, 70, 70),
    2: (0, 255, 255),
    3: (120, 60, 20),
    4: (0, 128, 255),
    5: (160, 32, 240),
    6: (255, 230, 40),
    7: (0, 170, 60),
    8: (255, 250, 80),
    9: (255, 125, 0),
    10: (110, 130, 0),
}

# Merge order = low priority first, high priority last
MERGE_ORDER = [
    "table",
    "colors_panel",
    "iron_forge",
    "mazinger_z_toy",
    "among_us_purple_teddy",
    "banana",
    "kiwi",
    "lemon",
    "orange",
    "green_dark_orange",
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
        description="Run SAM3 video semantic prompting on the basement image sequence."
    )
    parser.add_argument("--images_dir", type=Path, default=Path("data/basement/input"))
    parser.add_argument("--output_dir", type=Path, default=Path("data/basement/color_masks"))
    parser.add_argument("--overlay_dir", type=Path, default=Path("data/basement/overlays"))
    parser.add_argument("--model", type=str, default="weights/sam3.pt")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--half", dest="half", action="store_true", default=True)
    parser.add_argument("--no-half", dest="half", action="store_false")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=CONF_THRESHOLD)
    parser.add_argument("--overlay_alpha", type=float, default=0.45)

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
        default=1,
        help="Maximum active SAM3 video track objects per semantic class.",
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
        filtered: list[Path] = []
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



def install_torch_attention_compat() -> None:
    """Provide torch.nn.attention for Ultralytics SAM3 on Torch 2.1."""
    import sys
    import types

    try:
        import torch.nn.attention  # type: ignore  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    import torch

    compiler = getattr(torch, "compiler", None)
    if compiler is not None and not hasattr(compiler, "is_dynamo_compiling"):
        dynamo = getattr(torch, "_dynamo", None)
        compiler.is_dynamo_compiling = getattr(dynamo, "is_compiling", lambda: False)

    if not getattr(torch.Tensor, "_jcalm_tuple_any_compat", False):
        original_tensor_any = torch.Tensor.any

        def tuple_dim_any(self, dim=None, keepdim=False, *args, **kwargs):
            if "dim" in kwargs:
                dim = kwargs.pop("dim")
            if "keepdim" in kwargs:
                keepdim = kwargs.pop("keepdim")
            if isinstance(dim, tuple):
                result = self
                ndim = result.dim()
                dims = sorted({d if d >= 0 else d + ndim for d in dim})
                if keepdim:
                    for d in dims:
                        result = original_tensor_any(result, d, True)
                else:
                    for d in reversed(dims):
                        result = original_tensor_any(result, d, False)
                return result
            if dim is None:
                return original_tensor_any(self, *args, **kwargs)
            return original_tensor_any(self, dim, keepdim, *args, **kwargs)

        torch.Tensor.any = tuple_dim_any
        torch.Tensor._jcalm_tuple_any_compat = True

    if not torch.cuda.is_available():
        return

    sdp_backend = getattr(torch.backends.cuda, "SDPBackend", None)
    legacy_sdp_kernel = getattr(torch.backends.cuda, "sdp_kernel", None)
    if sdp_backend is None or legacy_sdp_kernel is None:
        return

    def sdpa_kernel(backends):
        if not isinstance(backends, (list, tuple, set)):
            selected = {backends}
        else:
            selected = set(backends)
        return legacy_sdp_kernel(
            enable_flash=sdp_backend.FLASH_ATTENTION in selected,
            enable_math=sdp_backend.MATH in selected,
            enable_mem_efficient=sdp_backend.EFFICIENT_ATTENTION in selected,
        )

    attention_module = types.ModuleType("torch.nn.attention")
    attention_module.SDPBackend = sdp_backend
    attention_module.sdpa_kernel = sdpa_kernel
    sys.modules["torch.nn.attention"] = attention_module


def build_video_predictor(args: argparse.Namespace) -> Any:
    install_torch_attention_compat()

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

    return remove_small_components(mask, min_component_area)


def maybe_dilate_mask(mask: np.ndarray, kernel_size: int, iterations: int) -> np.ndarray:
    if kernel_size <= 1 or iterations <= 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.dilate(mask, kernel, iterations=iterations)


def save_binary_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), (mask > 0).astype(np.uint8) * 255)


def colorize_semantic_mask(
    index_mask: np.ndarray,
    class_colors: dict[int, tuple[int, int, int]],
) -> np.ndarray:
    color = np.zeros((index_mask.shape[0], index_mask.shape[1], 3), dtype=np.uint8)
    for class_id, rgb in class_colors.items():
        color[index_mask == class_id] = rgb
    return color


def save_color_mask(mask_rgb: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR))


def save_semantic_overlay(
    image_bgr: np.ndarray,
    color_mask_rgb: np.ndarray,
    path: Path,
    alpha: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    color_mask_bgr = cv2.cvtColor(color_mask_rgb, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(image_bgr, 1 - alpha, color_mask_bgr, alpha, 0)
    cv2.imwrite(str(path), overlay)


# ----------------------------
# Main processing
# ----------------------------

def run_class_predictions(
    args: argparse.Namespace,
    video_path: Path,
    image_paths: list[Path],
    temp_class_root: Path,
) -> None:
    for class_name in MERGE_ORDER:
        prompt = CLASS_PROMPTS[class_name]
        predictor = build_video_predictor(args)

        results = predictor(
            source=str(video_path),
            text=[prompt],
            stream=True,
        )

        temp_out_dir = temp_class_root / class_name
        temp_out_dir.mkdir(parents=True, exist_ok=True)

        desc = f"Class: {class_name}"

        try:
            for image_path, result in tqdm(
                zip(image_paths, results),
                total=len(image_paths),
                unit="frame",
                desc=desc,
            ):
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f"Could not read image: {image_path}")

                class_mask = result_to_mask(result, image.shape[:2], args.mask_threshold)
                class_mask = postprocess_mask(
                    class_mask,
                    image.shape[:2],
                    min_component_area=args.min_component_area,
                    morph_kernel_size=args.morph_kernel_size,
                )
                class_mask = maybe_dilate_mask(
                    class_mask,
                    args.dilate_kernel_size,
                    args.dilate_iterations,
                )

                stem = image_path.stem
                temp_mask_path = temp_out_dir / f"{stem}.png"
                save_binary_mask(class_mask, temp_mask_path)
        finally:
            if hasattr(results, "close"):
                results.close()


def build_color_masks(
    image_paths: list[Path],
    temp_class_root: Path,
    output_dir: Path,
    overlay_dir: Path,
    overlay_alpha: float,
) -> None:
    for image_path in tqdm(image_paths, desc="Saving color masks", unit="frame"):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        h, w = image.shape[:2]
        semantic_mask = np.zeros((h, w), dtype=np.uint8)
        stem = image_path.stem

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

        color_mask = colorize_semantic_mask(semantic_mask, CLASS_COLORS)
        save_color_mask(color_mask, output_dir / f"{stem}.png")
        save_semantic_overlay(
            image,
            color_mask,
            overlay_dir / f"{stem}.jpg",
            alpha=overlay_alpha,
        )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.overlay_dir.mkdir(parents=True, exist_ok=True)

    image_paths = load_images(
        args.images_dir,
        start_index=args.start_index,
        end_index=args.end_index,
        max_images=args.max_images,
    )

    with tempfile.TemporaryDirectory(prefix="sam3_basement_work_") as work_tmpdir:
        work_tmpdir = Path(work_tmpdir)

        temp_class_root = work_tmpdir / "temp_class_binary_masks"
        temp_class_root.mkdir(parents=True, exist_ok=True)

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
        )

        build_color_masks(
            image_paths=image_paths,
            temp_class_root=temp_class_root,
            output_dir=args.output_dir,
            overlay_dir=args.overlay_dir,
            overlay_alpha=args.overlay_alpha,
        )

    print(f"Done. Wrote color masks to: {args.output_dir}")
    print(f"Done. Wrote overlays to: {args.overlay_dir}")


if __name__ == "__main__":
    main()
