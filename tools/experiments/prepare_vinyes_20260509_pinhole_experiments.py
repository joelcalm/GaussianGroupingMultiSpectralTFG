#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path


def frame_index(name: str) -> str | None:
    stem = Path(name).stem
    if "_" not in stem:
        return None
    return stem.split("_", 1)[1]


def relink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.symlink(os.path.relpath(src, dst.parent), dst)
    except OSError:
        shutil.copy2(src, dst)


def write_config(path: Path, variant: str, photometric_channels: list[int], use_color_embed: bool, disable_color: bool, photometric_weight: float) -> None:
    cfg = {
        "densify_until_iter": 10000,
        "densify_grad_threshold": 0.00005,
        "num_classes": 1024,
        "num_objects": 16,
        "reg3d_interval": 5,
        "reg3d_k": 5,
        "reg3d_max_points": 200000,
        "reg3d_sample_size": 1000,
        "reg3d_lambda_val": 2,
        "use_color_embed": use_color_embed,
        "disable_color": disable_color,
        "color_embed_dim": 32,
        "color_decoder_hidden_dim": 128,
        "color_decoder_num_hidden_layers": 3,
        "color_decoder_lr": 0.001,
        "num_channels": 10,
        "single_channel_mode": False,
        "rgb_oversample_factor": 1,
        "photometric_channels": photometric_channels,
        "photometric_loss_weight": photometric_weight,
        "experiment_variant": variant,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare vinyes_20260509_pinhole 185/15 RGB-mask experiment split.")
    parser.add_argument("--scene_dir", type=Path, default=Path("data/vinyes_20260509_pinhole"))
    parser.add_argument("--manual_eval_dir", type=Path, default=Path("data/vinyes_20260509_pinhole/manual_eval_gt/target_two_vines_all31/final_15_all_objects"))
    parser.add_argument("--object_mask_dir", type=Path, default=None, help="Full RGB mask source for both train and held-out frames. Defaults to manual_eval_gt/object_mask_hybrid_target_all.")
    parser.add_argument("--config_dir", type=Path, default=Path("config/gaussian_dataset"))
    args = parser.parse_args()

    scene = args.scene_dir
    final = args.manual_eval_dir
    selected_csv = final / "selected_frames.csv"
    if not selected_csv.exists():
        raise FileNotFoundError(selected_csv)
    selected = [row["frame"] for row in csv.DictReader(selected_csv.open())]
    heldout_indices = {frame_index(name) for name in selected}
    heldout_indices.discard(None)
    if len(heldout_indices) != len(selected):
        raise RuntimeError(f"Expected unique selected frame indices, got {len(heldout_indices)} for {len(selected)} frames")

    object_source = args.object_mask_dir or scene / "manual_eval_gt" / "object_mask_hybrid_target_all"
    if not object_source.is_dir():
        raise FileNotFoundError(f"Missing object mask source {object_source}. Run tools/manual_eval/build_hybrid_target_all_masks.py first.")

    out_masks = scene / "object_mask_experiments"
    shutil.rmtree(out_masks, ignore_errors=True)
    out_masks.mkdir(parents=True, exist_ok=True)

    rgb_names = sorted(p.name for p in (scene / "images_rgb").glob("rgb_*.png"))
    train_rgb = []
    test_rgb = []
    for name in rgb_names:
        idx = frame_index(name)
        src = object_source / name
        if idx in heldout_indices:
            test_rgb.append(name)
        else:
            train_rgb.append(name)
        if not src.exists():
            raise FileNotFoundError(src)
        shutil.copy2(src, out_masks / name)

    images_train = scene / "images_train"
    shutil.rmtree(images_train, ignore_errors=True)
    images_train.mkdir(parents=True, exist_ok=True)
    image_files = sorted((scene / "images").glob("*.png"))
    train_image_count = 0
    test_image_count = 0
    for src in image_files:
        idx = frame_index(src.name)
        if idx in heldout_indices:
            test_image_count += 1
            continue
        relink(src, images_train / src.name)
        train_image_count += 1

    metadata = scene / "metadata"
    metadata.mkdir(exist_ok=True)
    split = {
        "scene_dir": str(scene),
        "mask_dir": str(out_masks),
        "images_train_dir": str(images_train),
        "manual_eval_final_15": str(final),
        "object_mask_source": str(object_source),
        "heldout_rgb_frames": selected,
        "heldout_frame_indices": sorted(heldout_indices),
        "train_rgb_mask_count": len(train_rgb),
        "test_rgb_mask_count": len(test_rgb),
        "train_image_count_all_modalities": train_image_count,
        "test_image_count_all_modalities": test_image_count,
        "active_channels": {
            "rgb": [0, 1, 2],
            "ms": [3, 4, 5, 6, 7, 8],
            "rgb_ms": [0, 1, 2, 3, 4, 5, 6, 7, 8],
            "inactive": [9],
        },
        "note": "Hybrid target-all SAM3/object masks are RGB-only. MS views have no object masks and contribute only photometric loss in MS/RGB+MS variants.",
    }
    (metadata / "vinyes_20260509_pinhole_experiment_split.json").write_text(json.dumps(split, indent=2) + "\n")

    write_config(args.config_dir / "vinyes_20260509_pinhole_no_color.json", "no_color", [], False, True, 0.0)
    write_config(args.config_dir / "vinyes_20260509_pinhole_rgb.json", "rgb", [0, 1, 2], True, False, 1.0)
    write_config(args.config_dir / "vinyes_20260509_pinhole_ms.json", "ms", [3, 4, 5, 6, 7, 8], True, False, 1.0)
    write_config(args.config_dir / "vinyes_20260509_pinhole_rgb_ms.json", "rgb_ms", [0, 1, 2, 3, 4, 5, 6, 7, 8], True, False, 1.0)

    print(json.dumps(split, indent=2))


if __name__ == "__main__":
    main()
