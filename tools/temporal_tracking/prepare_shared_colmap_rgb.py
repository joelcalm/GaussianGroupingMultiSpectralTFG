#!/usr/bin/env python3
"""Prepare and verify a shared RGB COLMAP workspace for temporal vineyard scenes."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import Counter
from pathlib import Path

from shared_colmap_utils import (
    choose_rgb_image_dir,
    default_scenes,
    iter_image_files,
    normalize_sparse_model,
    prefixed_image_name,
    read_images,
    resolve_repo_path,
    split_prefixed_image_name,
)


def select_images(images: list[Path], stride: int, max_frames: int | None) -> list[Path]:
    selected = images[:: max(1, stride)]
    if max_frames is not None:
        selected = selected[:max_frames]
    return selected


def link_or_copy(src: Path, dst: Path, mode: str, overwrite: bool) -> None:
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        os.link(src, dst)
    elif mode == "symlink":
        dst.symlink_to(src.resolve())
    else:
        raise ValueError(f"Unknown link mode: {mode}")


def prepare_workspace(args: argparse.Namespace) -> list[dict]:
    workspace = resolve_repo_path(args.workspace)
    images_out = workspace / "images"
    lists_out = workspace / "lists"
    sparse_out = workspace / "sparse"
    manifest_csv = workspace / "image_manifest.csv"
    manifest_json = workspace / "image_manifest.json"

    rows: list[dict] = []
    for scene in default_scenes():
        image_dir = choose_rgb_image_dir(scene.source_path)
        if image_dir is None:
            print(f"WARNING: {scene.key}: could not identify an RGB image folder under {scene.source_path}")
            continue
        source_images = iter_image_files(image_dir)
        selected = select_images(source_images, args.stride, args.max_frames_per_scene)
        print(f"{scene.key}: selected {len(selected)} / {len(source_images)} RGB images from {image_dir}")
        for index, image_path in enumerate(selected):
            target_name = prefixed_image_name(scene.date_prefix, image_path.name)
            rows.append(
                {
                    "scene": scene.key,
                    "date_prefix": scene.date_prefix,
                    "source_path": str(image_path),
                    "source_image_dir": str(image_dir),
                    "source_name": image_path.name,
                    "target_name": target_name,
                    "target_path": str(images_out / target_name),
                    "selection_index": index,
                }
            )

    if args.dry_run:
        print(f"Dry run: would create {len(rows)} staged images under {images_out}")
        for row in rows[:10]:
            print(f"  {row['source_path']} -> {row['target_path']}")
        if len(rows) > 10:
            print(f"  ... {len(rows) - 10} more")
        return rows

    images_out.mkdir(parents=True, exist_ok=True)
    lists_out.mkdir(parents=True, exist_ok=True)
    sparse_out.mkdir(parents=True, exist_ok=True)

    for row in rows:
        link_or_copy(Path(row["source_path"]), Path(row["target_path"]), args.link_mode, args.overwrite)

    with manifest_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scene",
                "date_prefix",
                "source_path",
                "source_image_dir",
                "source_name",
                "target_name",
                "target_path",
                "selection_index",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    manifest_json.write_text(json.dumps(rows, indent=2))
    (lists_out / "all_images.txt").write_text("\n".join(row["target_name"] for row in rows) + "\n")

    print(f"Wrote image manifest: {manifest_csv}")
    print(f"Wrote image list: {lists_out / 'all_images.txt'}")
    print_colmap_instructions(workspace)
    return rows


def print_colmap_instructions(workspace: Path) -> None:
    images = workspace / "images"
    database = workspace / "database.db"
    image_list = workspace / "lists" / "all_images.txt"
    sparse = workspace / "sparse"
    print("")
    print("Run COLMAP after review, for example:")
    print(f"  colmap feature_extractor --database_path {database} --image_path {images} --image_list_path {image_list} --ImageReader.camera_model OPENCV --SiftExtraction.max_image_size 3200 --SiftExtraction.max_num_features 12000")
    print(f"  colmap exhaustive_matcher --database_path {database} --SiftMatching.use_gpu 0")
    print(f"  colmap mapper --database_path {database} --image_path {images} --image_list_path {image_list} --output_path {sparse} --Mapper.multiple_models 1 --Mapper.ba_refine_principal_point 0")
    print("")
    print("After mapping, rerun this script with --verify to write registration_report.md.")


def load_manifest(workspace: Path) -> list[dict]:
    manifest = workspace / "image_manifest.json"
    if not manifest.exists():
        return []
    return json.loads(manifest.read_text())


def verify_registration(workspace: Path, model_dir: Path | None = None) -> Path | None:
    workspace = resolve_repo_path(workspace)
    model = normalize_sparse_model(resolve_repo_path(model_dir) if model_dir else workspace / "sparse" / "0")
    report_path = workspace / "registration_report.md"
    manifest_rows = load_manifest(workspace)

    if not model.exists():
        print(f"Shared COLMAP model not found yet: {model}")
        return None
    try:
        images = read_images(model)
    except FileNotFoundError as exc:
        print(f"Cannot verify registration: {exc}")
        return None

    registered_names = {image.name for image in images.values()}
    expected_names = {row["target_name"] for row in manifest_rows}
    registered_by_date = Counter()
    unexpected_by_date = Counter()
    for name in registered_names:
        prefix, _ = split_prefixed_image_name(name)
        if prefix is None:
            unexpected_by_date["unprefixed"] += 1
        else:
            registered_by_date[prefix] += 1
    expected_by_date = Counter(row["date_prefix"] for row in manifest_rows)
    missing = sorted(expected_names - registered_names)
    present_dates = sorted(prefix for prefix, count in registered_by_date.items() if count > 0)
    connected = len(present_dates) >= 3

    lines = [
        "# Shared RGB COLMAP Registration Report",
        "",
        f"- Model inspected: `{model}`",
        f"- Registered images: {len(registered_names)}",
        f"- Expected staged images: {len(expected_names)}",
        f"- All three dates in this model: {'yes' if connected else 'no'}",
        "",
        "## Registered Images By Date",
        "",
        "| Date prefix | Expected | Registered | Missing |",
        "| --- | ---: | ---: | ---: |",
    ]
    for prefix in sorted(set(expected_by_date) | set(registered_by_date)):
        lines.append(
            f"| `{prefix}` | {expected_by_date[prefix]} | {registered_by_date[prefix]} | "
            f"{expected_by_date[prefix] - registered_by_date[prefix]} |"
        )
    if unexpected_by_date:
        lines.extend(["", "## Unexpected Image Names", ""])
        for key, count in sorted(unexpected_by_date.items()):
            lines.append(f"- {key}: {count}")
    lines.extend(["", "## Images Not Registered", ""])
    if missing:
        for name in missing[:500]:
            lines.append(f"- `{name}`")
        if len(missing) > 500:
            lines.append(f"- ... {len(missing) - 500} more omitted from this report")
    else:
        lines.append("- None detected from the manifest.")

    report_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote registration report: {report_path}")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("output/temporal_tracking/shared_colmap_rgb"),
        help="Shared COLMAP workspace root.",
    )
    parser.add_argument("--stride", type=int, default=1, help="Frame stride per scene.")
    parser.add_argument("--max-frames-per-scene", type=int, default=None, help="Optional cap per scene.")
    parser.add_argument("--link-mode", choices=("symlink", "copy", "hardlink"), default="symlink")
    parser.add_argument("--overwrite", action="store_true", help="Replace already staged image links/copies.")
    parser.add_argument("--dry-run", action="store_true", help="Show staged images without writing files.")
    parser.add_argument("--verify", action="store_true", help="Only verify an existing shared COLMAP reconstruction.")
    parser.add_argument("--model-dir", type=Path, default=None, help="COLMAP model dir to verify. Defaults to <workspace>/sparse/0.")
    args = parser.parse_args()

    workspace = resolve_repo_path(args.workspace)
    if args.verify:
        verify_registration(workspace, args.model_dir)
        return

    prepare_workspace(args)
    verify_registration(workspace, args.model_dir)


if __name__ == "__main__":
    main()
