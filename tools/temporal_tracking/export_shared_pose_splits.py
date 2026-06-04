#!/usr/bin/env python3
"""Export per-date COLMAP pose folders from one shared RGB COLMAP model."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from shared_colmap_utils import (
    copy_best_points_file,
    default_scenes,
    normalize_sparse_model,
    read_cameras,
    read_images,
    resolve_repo_path,
    split_prefixed_image_name,
    write_cameras_text,
    write_images_text,
)


def load_manifest(workspace: Path) -> dict[str, dict]:
    manifest = workspace / "image_manifest.json"
    if not manifest.exists():
        return {}
    rows = json.loads(manifest.read_text())
    return {row["target_name"]: row for row in rows}


def default_prefix_to_scene() -> dict[str, str]:
    return {scene.date_prefix: scene.key for scene in default_scenes()}


def export_splits(args: argparse.Namespace) -> list[Path]:
    workspace = resolve_repo_path(args.workspace)
    shared_model = normalize_sparse_model(resolve_repo_path(args.shared_model))
    output_root = resolve_repo_path(args.output_root)
    cameras = read_cameras(shared_model)
    images = read_images(shared_model)
    manifest = load_manifest(workspace)
    prefix_to_scene = default_prefix_to_scene()

    by_prefix = defaultdict(list)
    name_overrides = {}
    skipped_unprefixed = []
    for image in images.values():
        prefix, original_name = split_prefixed_image_name(image.name)
        if prefix is None:
            skipped_unprefixed.append(image.name)
            continue
        by_prefix[prefix].append(image)
        if not args.keep_prefixed_names:
            manifest_row = manifest.get(image.name, {})
            name_overrides[image.id] = manifest_row.get("source_name", original_name)

    written = []
    split_summary = []
    for prefix, split_images in sorted(by_prefix.items()):
        scene_name = prefix_to_scene.get(prefix, f"date_{prefix}")
        model_out = output_root / scene_name / "sparse" / "0"
        model_out.mkdir(parents=True, exist_ok=True)
        used_camera_ids = {image.camera_id for image in split_images}
        split_cameras = {camera_id: cameras[camera_id] for camera_id in used_camera_ids if camera_id in cameras}
        if len(split_cameras) != len(used_camera_ids):
            missing = sorted(used_camera_ids - set(split_cameras))
            raise KeyError(f"{scene_name}: shared model is missing camera ids: {missing}")

        write_cameras_text(model_out / "cameras.txt", split_cameras)
        write_images_text(model_out / "images.txt", split_images, name_overrides=name_overrides)
        points_path = copy_best_points_file(shared_model, model_out)

        written.append(model_out)
        split_summary.append(
            {
                "date_prefix": prefix,
                "scene": scene_name,
                "registered_images": len(split_images),
                "camera_count": len(split_cameras),
                "output_sparse": str(model_out),
                "points_file": str(points_path) if points_path else None,
                "image_names": [name_overrides.get(image.id, image.name) for image in split_images],
            }
        )
        print(f"Wrote {scene_name}: {len(split_images)} images, {len(split_cameras)} cameras -> {model_out}")

    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "split_manifest.json"
    summary_path.write_text(json.dumps({"splits": split_summary, "skipped_unprefixed": skipped_unprefixed}, indent=2))
    print(f"Wrote split manifest: {summary_path}")
    if skipped_unprefixed:
        print(f"WARNING: skipped {len(skipped_unprefixed)} unprefixed registered image(s).")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("output/temporal_tracking/shared_colmap_rgb"),
        help="Shared workspace containing image_manifest.json.",
    )
    parser.add_argument(
        "--shared-model",
        type=Path,
        default=Path("output/temporal_tracking/shared_colmap_rgb/sparse/0"),
        help="Shared COLMAP model folder, or workspace root with sparse/0.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("output/temporal_tracking/shared_pose_splits"),
        help="Root for per-date COLMAP split folders.",
    )
    parser.add_argument(
        "--keep-prefixed-names",
        action="store_true",
        help="Keep YYYYMMDD__ image names in split images.txt instead of restoring original image names.",
    )
    args = parser.parse_args()
    export_splits(args)


if __name__ == "__main__":
    main()
