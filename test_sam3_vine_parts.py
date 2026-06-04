#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import tempfile
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


DEFAULT_PROMPTS = [
    # Leaves
    {
        "part": "leaves",
        "label": "leaves_simple",
        "text": "grapevine leaves",
        "color": [0, 255, 0],
    },
    {
        "part": "leaves",
        "label": "green_vine_leaves",
        "text": "green grapevine leaves on vineyard plants",
        "color": [0, 255, 0],
    },
    {
        "part": "leaves",
        "label": "leaf_canopy",
        "text": "leaf canopy of grapevines",
        "color": [0, 255, 0],
    },

    # Trunk / woody parts
    {
        "part": "trunk",
        "label": "trunk_simple",
        "text": "grapevine trunk",
        "color": [255, 140, 0],
    },
    {
        "part": "trunk",
        "label": "dark_woody_trunk",
        "text": "dark woody grapevine trunk",
        "color": [255, 140, 0],
    },
    {
        "part": "trunk",
        "label": "old_vine_wood",
        "text": "old woody vine trunk and branches",
        "color": [255, 140, 0],
    },

    # Whole vine, useful as comparison only
    {
        "part": "whole_vine",
        "label": "whole_grapevine",
        "text": "whole grapevine plant including trunk branches and leaves",
        "color": [0, 180, 255],
    },
]


def parse_args():
    p = argparse.ArgumentParser("Test SAM3 prompts for vineyard leaves/trunks")

    p.add_argument("--images_dir", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--model", type=str, default="weights/sam3.pt")
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--half", action="store_true", default=True)
    p.add_argument("--no-half", dest="half", action="store_false")

    p.add_argument("--prompt_part", choices=["all", "leaves", "trunk", "whole_vine"], default="all")
    p.add_argument("--prompts_json", type=Path, default=None)

    p.add_argument("--conf_values", type=float, nargs="+", default=[0.25])
    p.add_argument("--mask_thresholds", type=float, nargs="+", default=[0.50])

    p.add_argument("--score_threshold_detection", type=float, default=0.5)
    p.add_argument("--new_det_thresh", type=float, default=0.0)
    p.add_argument("--assoc_iou_thresh", type=float, default=0.5)
    p.add_argument("--trk_assoc_iou_thresh", type=float, default=0.5)
    p.add_argument("--max_num_objects", type=int, default=96)

    p.add_argument("--min_area", type=int, default=80)
    p.add_argument("--morph", type=int, default=5)
    p.add_argument("--overlay_alpha", type=float, default=0.45)

    p.add_argument("--start_index", type=int, default=None)
    p.add_argument("--end_index", type=int, default=None)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--max_images", type=int, default=30)
    p.add_argument("--fps", type=float, default=5.0)

    p.add_argument("--contact_n", type=int, default=5)
    p.add_argument("--contact_cols", type=int, default=3)

    return p.parse_args()


def frame_number(path: Path) -> int | None:
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    return int(digits) if digits else None


def load_images(args) -> list[Path]:
    paths = sorted(
        p for p in args.images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )

    if args.start_index is not None or args.end_index is not None:
        selected = []
        for p in paths:
            n = frame_number(p)
            if n is None:
                continue
            if args.start_index is not None and n < args.start_index:
                continue
            if args.end_index is not None and n > args.end_index:
                continue
            selected.append(p)
        paths = selected

    paths = paths[::max(1, args.stride)]

    if args.max_images:
        paths = paths[:args.max_images]

    if not paths:
        raise FileNotFoundError(f"No images found in {args.images_dir}")

    return paths


def make_video(image_paths: list[Path], video_path: Path, fps: float):
    first = cv2.imread(str(image_paths[0]))
    if first is None:
        raise ValueError(f"Could not read {image_paths[0]}")

    h, w = first.shape[:2]
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )

    for p in image_paths:
        img = cv2.imread(str(p))
        if img is None:
            raise ValueError(f"Could not read {p}")
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        writer.write(img)

    writer.release()


def slug(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def build_predictor(args, conf: float):
    try:
        from sam3_vine_video import install_torch_attention_compat
    except ImportError:
        from sam3_basement_video import install_torch_attention_compat

    try:
        install_torch_attention_compat()
    except Exception as exc:
        raise RuntimeError("Failed to install Torch/SAM3 compatibility patch") from exc

    from ultralytics.models.sam import SAM3VideoSemanticPredictor

    return SAM3VideoSemanticPredictor(
        overrides={
            "conf": conf,
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
        max_num_objects=args.max_num_objects,
    )


def result_to_mask(result, shape, mask_threshold: float) -> np.ndarray:
    h, w = shape
    out = np.zeros((h, w), dtype=np.uint8)

    masks_obj = getattr(result, "masks", None)
    if masks_obj is None or masks_obj.data is None:
        return out

    masks = masks_obj.data
    if hasattr(masks, "detach"):
        masks = masks.detach().cpu().numpy()

    masks = np.asarray(masks)
    if masks.ndim == 2:
        masks = masks[None]

    for m in masks:
        m = (m > mask_threshold).astype(np.uint8)
        if m.shape[:2] != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        out |= m

    return out


def clean_mask(mask: np.ndarray, min_area: int, morph: int) -> np.ndarray:
    mask = (mask > 0).astype(np.uint8)

    if morph > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph, morph))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)

    if min_area > 0:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        cleaned = np.zeros_like(mask)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                cleaned[labels == i] = 1
        mask = cleaned

    return mask


def save_overlay(image_bgr, mask, color_rgb, label, path, alpha):
    path.parent.mkdir(parents=True, exist_ok=True)

    color_bgr = tuple(reversed(color_rgb))
    color_img = np.zeros_like(image_bgr)
    color_img[:] = color_bgr

    overlay = image_bgr.copy()
    blended = cv2.addWeighted(image_bgr, 1 - alpha, color_img, alpha, 0)
    overlay[mask > 0] = blended[mask > 0]

    cv2.rectangle(overlay, (10, 10), (10 + 13 * len(label), 42), (0, 0, 0), -1)
    cv2.putText(
        overlay,
        label,
        (18, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.imwrite(str(path), overlay)


def run_prompt(args, video_path, image_paths, prompt_cfg, conf, mask_thr, csv_writer):
    run_name = f"{prompt_cfg['part']}__{prompt_cfg['label']}__conf{conf:g}__m{mask_thr:g}"
    run_name = slug(run_name)

    masks_dir = args.output_dir / "runs" / run_name / "masks"
    overlays_dir = args.output_dir / "runs" / run_name / "overlays"
    masks_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    predictor = build_predictor(args, conf)

    results = predictor(
        source=str(video_path),
        text=[prompt_cfg["text"]],
        stream=True,
    )

    desc = f"{prompt_cfg['label']} | conf={conf:g} | m={mask_thr:g}"

    try:
        for image_path, result in tqdm(zip(image_paths, results), total=len(image_paths), desc=desc):
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not read {image_path}")

            mask = result_to_mask(result, image.shape[:2], mask_thr)
            mask = clean_mask(mask, args.min_area, args.morph)

            stem = image_path.stem
            cv2.imwrite(str(masks_dir / f"{stem}.png"), mask * 255)

            label = f"{prompt_cfg['part']} | {prompt_cfg['label']}"
            save_overlay(
                image,
                mask,
                prompt_cfg["color"],
                label,
                overlays_dir / f"{stem}.jpg",
                args.overlay_alpha,
            )

            csv_writer.writerow({
                "frame": image_path.name,
                "run": run_name,
                "part": prompt_cfg["part"],
                "label": prompt_cfg["label"],
                "prompt": prompt_cfg["text"],
                "conf": conf,
                "mask_threshold": mask_thr,
                "mask_pixels": int(mask.sum()),
                "mask_area_pct": float(mask.mean()),
            })
    finally:
        close = getattr(results, "close", None)
        if close is not None:
            close()

    return run_name


def make_contact_sheets(args, image_paths, run_names):
    out_dir = args.output_dir / "contact_sheets"
    out_dir.mkdir(parents=True, exist_ok=True)

    for image_path in image_paths[:args.contact_n]:
        tiles = []

        for run in run_names:
            p = args.output_dir / "runs" / run / "overlays" / f"{image_path.stem}.jpg"
            img = cv2.imread(str(p))
            if img is None:
                continue

            target_w = 420
            scale = target_w / img.shape[1]
            img = cv2.resize(img, (target_w, int(img.shape[0] * scale)))
            tiles.append(img)

        if not tiles:
            continue

        cols = max(1, args.contact_cols)
        rows = []

        for i in range(0, len(tiles), cols):
            row_tiles = tiles[i:i + cols]
            while len(row_tiles) < cols:
                row_tiles.append(np.zeros_like(row_tiles[0]))
            rows.append(np.hstack(row_tiles))

        sheet = np.vstack(rows)
        cv2.imwrite(str(out_dir / f"{image_path.stem}.jpg"), sheet)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.prompts_json:
        prompts = json.loads(args.prompts_json.read_text())
    else:
        prompts = DEFAULT_PROMPTS

    if args.prompt_part != "all":
        prompts = [p for p in prompts if p["part"] == args.prompt_part]

    image_paths = load_images(args)

    with tempfile.TemporaryDirectory(prefix="sam3_prompt_test_") as tmp:
        video_path = Path(tmp) / "sequence.mp4"
        make_video(image_paths, video_path, args.fps)

        summary_path = args.output_dir / "summary.csv"
        run_names = []

        with summary_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "frame",
                    "run",
                    "part",
                    "label",
                    "prompt",
                    "conf",
                    "mask_threshold",
                    "mask_pixels",
                    "mask_area_pct",
                ],
            )
            writer.writeheader()

            for prompt_cfg in prompts:
                for conf in args.conf_values:
                    for mask_thr in args.mask_thresholds:
                        run = run_prompt(
                            args=args,
                            video_path=video_path,
                            image_paths=image_paths,
                            prompt_cfg=prompt_cfg,
                            conf=conf,
                            mask_thr=mask_thr,
                            csv_writer=writer,
                        )
                        run_names.append(run)

        make_contact_sheets(args, image_paths, run_names)

    print(f"Done. Results written to: {args.output_dir}")
    print(f"Open: {args.output_dir / 'contact_sheets'}")
    print(f"Summary CSV: {args.output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()