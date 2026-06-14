#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment


def latest_method_dir(model_path: Path, split: str) -> Path:
    split_dir = model_path / split
    methods = sorted(split_dir.glob("ours_*"), key=lambda p: int(p.name.split("_", 1)[1]) if p.name.split("_", 1)[1].isdigit() else -1)
    if not methods:
        raise FileNotFoundError(f"No render methods under {split_dir}")
    return methods[-1]


def load_index(path: Path) -> np.ndarray:
    return np.array(Image.open(path)).astype(np.int64)


def resize_nearest(arr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if arr.shape == shape:
        return arr
    im = Image.fromarray(arr.astype(np.uint16))
    im = im.resize((shape[1], shape[0]), Image.NEAREST)
    return np.array(im).astype(np.int64)


def id2rgb(idx: int) -> tuple[int, int, int]:
    if idx <= 0:
        return (0, 0, 0)
    h = (idx * 1.6180339887) % 1
    s = 0.5 + (idx % 2) * 0.5
    l = 0.5

    import colorsys

    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return (int(r * 255), int(g * 255), int(b * 255))


def colorize_index(mask: np.ndarray) -> np.ndarray:
    color = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for idx in np.unique(mask):
        color[mask == idx] = id2rgb(int(idx))
    return color


def load_or_colorize_gt(gt_path: Path, shape: tuple[int, int]) -> np.ndarray:
    color_path = gt_path.parent.parent / "gt_objects_color" / gt_path.name
    if color_path.exists():
        color = np.array(Image.open(color_path).convert("RGB"))
        if color.shape[:2] != shape:
            color = np.array(Image.fromarray(color).resize((shape[1], shape[0]), Image.NEAREST))
        return color.astype(np.uint8)
    return colorize_index(resize_nearest(load_index(gt_path), shape))


def write_mapped_visualizations(method_dir: Path, pairs: list[tuple[str, Path, Path]], pred_to_gt: dict[int, int]) -> dict[str, str]:
    mapped_index_dir = method_dir / "objects_pred_mapped_index"
    mapped_color_dir = method_dir / "objects_pred_mapped_color"
    compare_dir = method_dir / "object_id_compare_mapped"
    mapped_index_dir.mkdir(parents=True, exist_ok=True)
    mapped_color_dir.mkdir(parents=True, exist_ok=True)
    compare_dir.mkdir(parents=True, exist_ok=True)

    for image_name, pred_path, gt_path in pairs:
        pred = load_index(pred_path)
        mapped = np.zeros_like(pred, dtype=np.uint16)
        for pred_id, gt_id in pred_to_gt.items():
            mapped[pred == int(pred_id)] = int(gt_id)

        stem = pred_path.name
        pred_color = colorize_index(mapped)
        gt_color = load_or_colorize_gt(gt_path, pred.shape)

        Image.fromarray(mapped.astype(np.uint16)).save(mapped_index_dir / stem)
        Image.fromarray(pred_color).save(mapped_color_dir / stem)

        panels = []
        rgb_path = method_dir / "gt" / stem
        if rgb_path.exists():
            rgb = np.array(Image.open(rgb_path).convert("RGB"))
            if rgb.shape[:2] != pred.shape:
                rgb = np.array(Image.fromarray(rgb).resize((pred.shape[1], pred.shape[0]), Image.BILINEAR))
            panels.append(rgb.astype(np.uint8))
        panels.extend([gt_color, pred_color])
        Image.fromarray(np.hstack(panels).astype(np.uint8)).save(compare_dir / stem)

    return {
        "objects_pred_mapped_index": str(mapped_index_dir),
        "objects_pred_mapped_color": str(mapped_color_dir),
        "object_id_compare_mapped": str(compare_dir),
    }


def label_metrics(gt: np.ndarray, pred_mapped: np.ndarray, labels: list[int]) -> tuple[dict[int, dict], dict]:
    rows = {}
    ious, dices = [], []
    for lab in labels:
        g = gt == lab
        p = pred_mapped == lab
        inter = int(np.logical_and(g, p).sum())
        union = int(np.logical_or(g, p).sum())
        gsum = int(g.sum())
        psum = int(p.sum())
        iou = inter / union if union else None
        dice = (2 * inter) / (gsum + psum) if (gsum + psum) else None
        rows[lab] = {"gt_pixels": gsum, "pred_pixels": psum, "intersection": inter, "union": union, "iou": iou, "dice": dice}
        if gsum > 0:
            ious.append(0.0 if iou is None else iou)
            dices.append(0.0 if dice is None else dice)
    macro = {"miou": float(np.mean(ious)) if ious else 0.0, "mdice": float(np.mean(dices)) if dices else 0.0}
    return rows, macro


def default_class_map_path(gt_dir: Path, label_mode: str) -> Path | None:
    eval_dir = gt_dir.parent
    candidates = (
        ["hybrid_label_map.json", "stable_eval_label_map.json"]
        if label_mode == "all_gt"
        else ["stable_eval_label_map.json", "hybrid_label_map.json"]
    )
    for name in candidates:
        path = eval_dir / name
        if path.exists():
            return path
    return None


def load_class_labels(path: Path | None) -> dict[int, str]:
    if path is None:
        return {}
    raw = json.loads(path.read_text())
    labels = {}
    for key, row in raw.items():
        try:
            label_id = int(key)
        except ValueError:
            continue
        class_name = row.get("class_name")
        class_id = row.get("class_id")
        if class_name:
            labels[label_id] = str(class_name)
        elif class_id is not None:
            labels[label_id] = str(class_id)
    return labels


def class_metrics(gt: np.ndarray, pred_mapped: np.ndarray, label_to_class: dict[int, str]) -> tuple[dict[str, dict], dict]:
    if not label_to_class:
        return {}, {"miou": None, "mdice": None}

    gt_class = np.array([label_to_class.get(int(v), "__unknown__") for v in gt], dtype=object)
    pred_class = np.array([label_to_class.get(int(v), "__background__") for v in pred_mapped], dtype=object)
    classes = sorted({str(v) for v in gt_class if v not in {"background", "__unknown__"}})

    rows = {}
    ious, dices = [], []
    for cls in classes:
        g = gt_class == cls
        p = pred_class == cls
        inter = int(np.logical_and(g, p).sum())
        union = int(np.logical_or(g, p).sum())
        gsum = int(g.sum())
        psum = int(p.sum())
        iou = inter / union if union else None
        dice = (2 * inter) / (gsum + psum) if (gsum + psum) else None
        rows[cls] = {"gt_pixels": gsum, "pred_pixels": psum, "intersection": inter, "union": union, "iou": iou, "dice": dice}
        if gsum > 0:
            ious.append(0.0 if iou is None else iou)
            dices.append(0.0 if dice is None else dice)
    macro = {"miou": float(np.mean(ious)) if ious else None, "mdice": float(np.mean(dices)) if dices else None}
    return rows, macro


def evaluate_model(
    model_path: Path,
    gt_dir: Path,
    selected_frames: list[str],
    iteration: int | None,
    split: str,
    label_mode: str = "target_compact",
    explicit_labels: list[int] | None = None,
) -> dict:
    method_dir = model_path / split / f"ours_{iteration}" if iteration is not None else latest_method_dir(model_path, split)
    pred_dir = method_dir / "objects_pred_index"
    if not pred_dir.is_dir():
        raise FileNotFoundError(pred_dir)
    frames_index_path = method_dir / "frames_index.json"
    if not frames_index_path.exists():
        raise FileNotFoundError(frames_index_path)
    frames_index = json.loads(frames_index_path.read_text())
    selected_set = {Path(x).stem for x in selected_frames}

    pairs = []
    for row in frames_index:
        image_name = row["image_name"]
        if image_name not in selected_set:
            continue
        stem = row.get("file_stem", f"{int(row['index']):05d}")
        pred_path = pred_dir / f"{stem}.png"
        gt_path = gt_dir / f"{image_name}.png"
        if pred_path.exists() and gt_path.exists():
            pairs.append((image_name, pred_path, gt_path))
    if not pairs:
        raise RuntimeError(f"No selected final-15 RGB frames found in {method_dir}")

    labels = list(range(1, 10)) if explicit_labels is None else explicit_labels
    pred_ids = set()
    gt_all = []
    pred_all = []
    per_frame = []
    raw_counts_by_gt = defaultdict(Counter)

    for image_name, pred_path, gt_path in pairs:
        pred = load_index(pred_path)
        gt = resize_nearest(load_index(gt_path), pred.shape)
        valid = gt > 0
        gt_v = gt[valid]
        pred_v = pred[valid]
        gt_all.append(gt_v)
        pred_all.append(pred_v)
        pred_ids.update(int(v) for v in np.unique(pred_v))
        per_frame.append({"frame": image_name + ".png", "valid_pixels": int(valid.sum())})

    gt_cat = np.concatenate(gt_all) if gt_all else np.array([], dtype=np.int64)
    pred_cat = np.concatenate(pred_all) if pred_all else np.array([], dtype=np.int64)
    if explicit_labels is None and label_mode == "all_gt":
        labels = sorted(int(v) for v in np.unique(gt_cat) if int(v) > 0)
    for image_name, pred_path, gt_path in pairs:
        pred = load_index(pred_path)
        gt = resize_nearest(load_index(gt_path), pred.shape)
        for lab in labels:
            vals = pred[gt == lab]
            raw_counts_by_gt[lab].update(int(v) for v in vals.tolist())
    pred_ids = sorted(pred_ids)
    pred_index = {pid: i for i, pid in enumerate(pred_ids)}
    gt_index = {lab: i for i, lab in enumerate(labels)}

    if labels and pred_ids and gt_cat.size:
        gt_codes = np.fromiter((gt_index[int(v)] for v in gt_cat), dtype=np.int64, count=gt_cat.size)
        pred_codes = np.fromiter((pred_index[int(v)] for v in pred_cat), dtype=np.int64, count=pred_cat.size)
        flat_codes = gt_codes * len(pred_ids) + pred_codes
        intersections = np.bincount(flat_codes, minlength=len(labels) * len(pred_ids)).reshape(len(labels), len(pred_ids)).astype(np.float64)
        gt_sizes = np.bincount(gt_codes, minlength=len(labels)).astype(np.float64)
        pred_sizes = np.bincount(pred_codes, minlength=len(pred_ids)).astype(np.float64)
    else:
        intersections = np.zeros((len(labels), len(pred_ids)), dtype=np.float64)
        gt_sizes = np.zeros(len(labels), dtype=np.float64)
        pred_sizes = np.zeros(len(pred_ids), dtype=np.float64)
    unions = gt_sizes[:, None] + pred_sizes[None, :] - intersections
    iou_matrix = np.divide(intersections, unions, out=np.zeros_like(intersections), where=unions > 0)

    if pred_ids:
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    else:
        row_ind, col_ind = np.array([], dtype=int), np.array([], dtype=int)
    pred_to_gt = {}
    gt_to_pred = {}
    for r, c in zip(row_ind.tolist(), col_ind.tolist()):
        if gt_sizes[r] <= 0:
            continue
        gt_lab = labels[r]
        pred_lab = pred_ids[c]
        pred_to_gt[pred_lab] = gt_lab
        gt_to_pred[gt_lab] = pred_lab

    pred_mapped = np.zeros_like(pred_cat)
    for pid, glab in pred_to_gt.items():
        pred_mapped[pred_cat == pid] = glab

    correct = pred_mapped == gt_cat
    pixel_accuracy = float(correct.mean()) if correct.size else 0.0
    eps = 1e-6
    k = len(labels)
    hard_ce = float(np.mean(np.where(correct, -np.log(1 - eps), -np.log(eps / max(1, k - 1))))) if correct.size else 0.0

    per_label, macro = label_metrics(gt_cat, pred_mapped, labels)
    class_map_path = default_class_map_path(gt_dir, label_mode)
    label_to_class = load_class_labels(class_map_path)
    per_class, class_macro = class_metrics(gt_cat, pred_mapped, label_to_class)
    consistency_rows = {}
    consistency_scores = []
    for lab in labels:
        counts = raw_counts_by_gt[lab]
        total = sum(counts.values())
        if total == 0:
            consistency_rows[lab] = {"gt_pixels": 0, "dominant_pred_id": None, "dominant_fraction": None, "unique_pred_ids": 0}
            continue
        dominant_id, dominant_count = counts.most_common(1)[0]
        frac = dominant_count / total
        consistency_rows[lab] = {
            "gt_pixels": total,
            "dominant_pred_id": dominant_id,
            "dominant_fraction": frac,
            "unique_pred_ids": len(counts),
        }
        consistency_scores.append(frac)

    visualization_dirs = write_mapped_visualizations(method_dir, pairs, pred_to_gt)

    result = {
        "model_path": str(model_path),
        "method_dir": str(method_dir),
        "frames_evaluated": len(pairs),
        "gt_dir": str(gt_dir),
        "visualization_dirs": visualization_dirs,
        "label_mode": label_mode,
        "label_ids": labels,
        "class_map_path": str(class_map_path) if class_map_path else None,
        "pred_to_gt_hungarian": {str(k): int(v) for k, v in pred_to_gt.items()},
        "gt_to_pred_hungarian": {str(k): int(v) for k, v in gt_to_pred.items()},
        "metrics": {
            "mIoU": macro["miou"],
            "Dice_Coefficient_F1": macro["mdice"],
            "class_mIoU": class_macro["miou"],
            "class_Dice_Coefficient_F1": class_macro["mdice"],
            "pixel_accuracy": pixel_accuracy,
        },
        "per_label": {str(k): v for k, v in per_label.items()},
        "per_class": per_class,
        "id_consistency_per_label": {str(k): v for k, v in consistency_rows.items()},
        "per_frame": per_frame,
    }
    return result


def write_outputs(result: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "object_id_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    with (out_dir / "object_id_per_label.csv").open("w", newline="") as f:
        cols = ["label", "gt_pixels", "pred_pixels", "intersection", "union", "iou", "dice", "dominant_pred_id", "dominant_fraction", "unique_pred_ids"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for lab, row in result["per_label"].items():
            cons = result["id_consistency_per_label"].get(lab, {})
            out = {"label": lab, **row, **cons}
            w.writerow(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate rendered object IDs against final-15 GT masks.")
    parser.add_argument("--model_paths", nargs="+", type=Path, required=True)
    parser.add_argument("--gt_dir", type=Path, default=Path("data/vinyes_20260509_pinhole/manual_eval_gt/target_two_vines_all31/final_15/object_mask"))
    parser.add_argument("--selected_frames_csv", type=Path, default=Path("data/vinyes_20260509_pinhole/manual_eval_gt/target_two_vines_all31/final_15/selected_frames.csv"))
    parser.add_argument("--iteration", type=int, default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--label_mode", choices=["target_compact", "all_gt"], default="target_compact")
    parser.add_argument("--labels", nargs="+", type=int, default=None, help="Explicit GT label IDs to evaluate. Overrides --label_mode.")
    parser.add_argument("--per_model_output_name", default="manual_eval_final15")
    parser.add_argument("--output_dir", type=Path, default=Path("output/vinyes_20260509_pinhole_object_eval"))
    args = parser.parse_args()

    selected = [row["frame"] for row in csv.DictReader(args.selected_frames_csv.open())]
    summary = []
    for model in args.model_paths:
        result = evaluate_model(model, args.gt_dir, selected, args.iteration, args.split, args.label_mode, args.labels)
        write_outputs(result, model / args.per_model_output_name)
        row = {"model_path": str(model), **result["metrics"], "frames_evaluated": result["frames_evaluated"]}
        summary.append(row)
        print(json.dumps(row, indent=2))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if summary:
        with (args.output_dir / "summary.csv").open("w", newline="") as f:
            cols = list(summary[0].keys())
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(summary)
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
