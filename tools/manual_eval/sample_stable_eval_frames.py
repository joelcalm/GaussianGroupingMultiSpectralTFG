#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path


def copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample corrected stable eval frames into a final subset.")
    parser.add_argument("--subset_dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=15)
    parser.add_argument("--seed", type=int, default=20260509)
    parser.add_argument("--output_name", default="final_15")
    args = parser.parse_args()

    subset = args.subset_dir
    stable_dir = subset / "stable_eval_masks"
    if not stable_dir.is_dir():
        raise FileNotFoundError(f"Missing {stable_dir}. Run apply_target_vine_corrections.py first.")

    frames = sorted(p.name for p in stable_dir.glob("*.png"))
    if len(frames) < args.count:
        raise ValueError(f"Requested {args.count} frames but only found {len(frames)} stable masks")

    rng = random.Random(args.seed)
    selected = sorted(rng.sample(frames, args.count))
    out = subset / args.output_name
    out.mkdir(parents=True, exist_ok=True)

    sources = [
        (subset / "rgb", out / "rgb", ".png"),
        (subset / "stable_eval_masks", out / "object_mask", ".png"),
        (subset / "stable_eval_color", out / "gt_objects_color", ".png"),
        (subset / "stable_eval_overlay", out / "gt_objects_overlay", ".jpg"),
    ]
    for frame in selected:
        stem = Path(frame).stem
        for src_dir, dst_dir, suffix in sources:
            src_name = frame if suffix == ".png" else f"{stem}{suffix}"
            copy_if_exists(src_dir / src_name, dst_dir / src_name)

    with (out / "selected_frames.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame"])
        writer.writeheader()
        for frame in selected:
            writer.writerow({"frame": frame})

    for name in ["stable_eval_label_map.json", "stable_eval_report.csv"]:
        copy_if_exists(subset / name, out / name)

    print(f"Selected {len(selected)} frames into {out}")
    print(" ".join(selected))


if __name__ == "__main__":
    main()
