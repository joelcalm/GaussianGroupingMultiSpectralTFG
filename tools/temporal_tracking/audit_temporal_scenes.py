#!/usr/bin/env python3
"""Audit temporal vineyard scenes for shared RGB COLMAP pose preparation."""

from __future__ import annotations

import argparse
from pathlib import Path

from shared_colmap_utils import (
    COLMAP_CAMERA_FILES,
    COLMAP_IMAGE_FILES,
    COLMAP_POINT_FILES,
    candidate_image_dirs,
    default_scenes,
    find_sparse_models,
    has_colmap_cameras,
    has_colmap_images,
    has_colmap_points,
    iter_image_files,
    resolve_repo_path,
)


def yes_no(value: bool) -> str:
    return "yes" if value else "no"


def describe_sparse_model(path: Path) -> str:
    files = []
    for name in (*COLMAP_CAMERA_FILES, *COLMAP_IMAGE_FILES, *COLMAP_POINT_FILES):
        if (path / name).exists():
            files.append(name)
    registered = ""
    images_txt = path / "images.txt"
    if images_txt.exists():
        count = sum(1 for line in images_txt.read_text().splitlines() if line and not line.startswith("#")) // 2
        registered = f"; registered images in text model: {count}"
    elif (path / "images.bin").exists():
        registered = "; binary images model present"
    return f"{path} ({', '.join(files) or 'no standard files'}{registered})"


def audit_scene(scene) -> dict:
    image_dirs = []
    for image_dir in candidate_image_dirs(scene.source_path):
        images = iter_image_files(image_dir)
        image_dirs.append((image_dir, len(images), images[:3]))

    sparse_models = find_sparse_models(scene.source_path)
    missing = []
    if not scene.source_path.is_dir():
        missing.append(f"source scene missing: {scene.source_path}")
    if not image_dirs:
        missing.append("no RGB candidate image folders found")
    if not sparse_models:
        missing.append("no COLMAP sparse/model folders found")
    if not scene.output_path.is_dir():
        missing.append(f"trained output missing: {scene.output_path}")

    trained_files = []
    if scene.output_path.is_dir():
        for name in ("results.json", "cfg_args", "input.ply", "cameras.json", "registered_images_summary.json"):
            if (scene.output_path / name).exists():
                trained_files.append(name)
        pc_root = scene.output_path / "point_cloud"
        if pc_root.is_dir():
            trained_files.extend(sorted(p.name for p in pc_root.iterdir() if p.is_dir()))

    return {
        "scene": scene,
        "image_dirs": image_dirs,
        "sparse_models": sparse_models,
        "trained_files": trained_files,
        "missing": missing,
    }


def write_summary(path: Path, audits: list[dict]) -> None:
    lines = [
        "# Temporal Vineyard Shared COLMAP Audit",
        "",
        "This report inspects the date-specific vineyard scenes before building a shared RGB COLMAP coordinate system.",
        "",
    ]
    for audit in audits:
        scene = audit["scene"]
        lines.extend(
            [
                f"## {scene.key}",
                "",
                f"- Date prefix: `{scene.date_prefix}`",
                f"- Source scene: `{scene.source_path}`",
                f"- Trained output: `{scene.output_path}`",
                "",
                "### RGB Image Folders",
                "",
            ]
        )
        if audit["image_dirs"]:
            for image_dir, count, sample in audit["image_dirs"]:
                sample_names = ", ".join(p.name for p in sample) if sample else "none"
                lines.append(f"- `{image_dir}`: {count} images; sample: {sample_names}")
        else:
            lines.append("- None found.")

        lines.extend(["", "### COLMAP Sparse Models", ""])
        if audit["sparse_models"]:
            for sparse_model in audit["sparse_models"]:
                lines.append(f"- `{describe_sparse_model(sparse_model)}`")
            primary = audit["sparse_models"][0]
            lines.extend(
                [
                    "",
                    "Primary model file presence:",
                    f"- cameras.bin/txt: {yes_no(has_colmap_cameras(primary))}",
                    f"- images.bin/txt: {yes_no(has_colmap_images(primary))}",
                    f"- points3D.bin/txt/ply: {yes_no(has_colmap_points(primary))}",
                ]
            )
        else:
            lines.append("- None found.")

        lines.extend(["", "### Existing Trained Output", ""])
        if audit["trained_files"]:
            for name in audit["trained_files"]:
                lines.append(f"- `{name}`")
        else:
            lines.append("- No expected trained-output artifacts found.")

        lines.extend(["", "### Warnings", ""])
        if audit["missing"]:
            for item in audit["missing"]:
                lines.append(f"- {item}")
        else:
            lines.append("- No obvious missing files.")
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/temporal_tracking/audit_summary.md"),
        help="Markdown summary path. Defaults to output/temporal_tracking/audit_summary.md.",
    )
    args = parser.parse_args()

    audits = [audit_scene(scene) for scene in default_scenes()]
    output = resolve_repo_path(args.output)
    write_summary(output, audits)
    print(f"Wrote audit summary: {output}")
    for audit in audits:
        scene = audit["scene"]
        rgb_count = audit["image_dirs"][0][1] if audit["image_dirs"] else 0
        print(
            f"{scene.key}: {rgb_count} images in primary RGB candidate, "
            f"{len(audit['sparse_models'])} sparse/model folders, "
            f"{len(audit['missing'])} warning(s)"
        )


if __name__ == "__main__":
    main()
