#!/usr/bin/env python3
"""Rebuild a grayscale multispectral COLMAP scene from an existing prepared scene.

This keeps the RGB-only sparse model as the fixed anchor, re-extracts
multispectral descriptors from grayscale staged images, and registers the
selected bands against all RGB frames. It is meant for cases where the original
joint/mixed reconstruction drifted into separate camera islands.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import struct
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np


DEFAULT_COLMAP = Path("/home/msiau/workspace/.conda/envs/colmap_gpu/bin/colmap")
PAIR_ID_MULTIPLIER = 2147483647


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_scene", type=Path, required=True)
    parser.add_argument("--output_scene", type=Path, required=True)
    parser.add_argument("--bands", default="rgb,b470,b505,b525,b590,b635,b660")
    parser.add_argument("--exclude_bands", default="b850")
    parser.add_argument("--colmap_bin", type=Path, default=DEFAULT_COLMAP)
    parser.add_argument("--cuda_visible_devices", default="1")
    parser.add_argument("--colmap_gpu_index", default="0")
    parser.add_argument("--max_image_size", type=int, default=3200)
    parser.add_argument("--max_num_features", type=int, default=16000)
    parser.add_argument("--sift_num_threads", type=int, default=8)
    parser.add_argument("--estimate_affine_shape", type=int, default=0)
    parser.add_argument("--domain_size_pooling", type=int, default=0)
    parser.add_argument("--matching_use_gpu", type=int, default=1)
    parser.add_argument("--max_num_matches", type=int, default=32768)
    parser.add_argument("--min_num_inliers", type=int, default=30)
    parser.add_argument("--guided_matching", type=int, default=1)
    parser.add_argument("--abs_pose_min_num_inliers", type=int, default=250)
    parser.add_argument("--abs_pose_max_error", type=float, default=2.0)
    parser.add_argument("--mapper_min_num_matches", type=int, default=100)
    parser.add_argument("--copy_images", action="store_true", help="Copy images instead of symlinking them.")
    return parser.parse_args()


def band_of_name(name: str) -> str:
    return Path(name).stem.split("_", 1)[0]


def parse_band_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def clean_dir(path: Path) -> None:
    if path.exists() or path.is_symlink():
        if path.is_symlink() or path.is_file():
            path.unlink()
        else:
            shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path, copy: bool = False) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        shutil.copy2(src, dst)
    else:
        os.symlink(src.resolve(strict=True), dst)


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open() as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def filter_scene_json(source: Path, output: Path, active_bands: set[str]) -> None:
    band_info = load_json(source / "band_info.json", {})
    write_json(
        output / "band_info.json",
        {k: v for k, v in band_info.items() if band_of_name(k) in active_bands},
    )

    frame_info = load_json(source / "frame_info.json", {})
    write_json(
        output / "frame_info.json",
        {k: v for k, v in frame_info.items() if band_of_name(k) in active_bands},
    )

    extraction = load_json(source / "extraction_report.json", [])
    if isinstance(extraction, list):
        extraction = [row for row in extraction if row.get("band") in active_bands]
    write_json(output / "extraction_report.json", extraction)

    manifest = load_json(source / "videos_manifest.json", [])
    if isinstance(manifest, list):
        manifest = [row for row in manifest if row.get("band") in active_bands]
    write_json(output / "videos_manifest.json", manifest)

    summary = load_json(source / "partial_channels_summary.json", {})
    if isinstance(summary, dict):
        summary = dict(summary)
        summary["output_dir"] = str(output.resolve())
        if isinstance(summary.get("bands"), list):
            summary["bands"] = [b for b in summary["bands"] if b in active_bands]
        if isinstance(summary.get("channels"), dict):
            summary["channels"] = {k: v for k, v in summary["channels"].items() if k in active_bands}
        if isinstance(summary.get("extraction"), list):
            summary["extraction"] = [row for row in summary["extraction"] if row.get("band") in active_bands]
    write_json(output / "partial_channels_summary.json", summary)


def prepare_scene(source: Path, output: Path, bands: list[str], copy_images: bool) -> None:
    active_bands = set(bands)
    output.mkdir(parents=True, exist_ok=True)
    clean_dir(output / "frames_raw")
    clean_dir(output / "images")
    clean_dir(output / "images_rgb")

    for band in bands:
        src_raw = source / "frames_raw" / band
        if not src_raw.exists():
            raise FileNotFoundError(src_raw)
        dst_raw = output / "frames_raw" / band
        dst_raw.mkdir(parents=True, exist_ok=True)
        for src in sorted(src_raw.iterdir()):
            if src.name.startswith("."):
                continue
            link_or_copy(src, dst_raw / src.name, copy=copy_images)

    for src in sorted((source / "images").iterdir()):
        if band_of_name(src.name) in active_bands:
            link_or_copy(src, output / "images" / src.name, copy=copy_images)

    for src in sorted((output / "images").iterdir()):
        if band_of_name(src.name) == "rgb":
            link_or_copy(src, output / "images_rgb" / src.name, copy=copy_images)

    for dirname in ["object_mask", "metadata", "sam3_vine_parts_posts"]:
        src = source / dirname
        dst = output / dirname
        if src.exists():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst, symlinks=True)

    filter_scene_json(source, output, active_bands)
    write_json(
        output / "clean_scene_manifest.json",
        {
            "source_scene": str(source.resolve()),
            "active_bands": bands,
            "excluded_bands": sorted(set(["b850"]) - active_bands),
            "images": len(list((output / "images").iterdir())),
            "rgb_images": len(list((output / "images_rgb").iterdir())),
        },
    )


def run(cmd: list[str | Path], *, env: dict[str, str], cwd: Path) -> None:
    printable = " ".join(str(part) for part in cmd)
    print(f"\n$ {printable}", flush=True)
    subprocess.run([str(part) for part in cmd], check=True, env=env, cwd=cwd)


def convert_model(colmap: Path, input_path: Path, output_path: Path, env: dict[str, str], cwd: Path) -> None:
    clean_dir(output_path)
    run(
        [
            colmap,
            "model_converter",
            "--input_path",
            input_path,
            "--output_path",
            output_path,
            "--output_type",
            "TXT",
        ],
        env=env,
        cwd=cwd,
    )


def read_camera_params(cameras_txt: Path) -> tuple[int, int, bytes]:
    for line in cameras_txt.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split()
        width = int(parts[2])
        height = int(parts[3])
        params = [float(v) for v in parts[4:]]
        return width, height, struct.pack(f"{len(params)}d", *params)
    raise RuntimeError(f"No camera row in {cameras_txt}")


def stage_grayscale_images(scene: Path, workspace: Path) -> Path:
    image_dir = workspace / "images"
    clean_dir(image_dir)
    for src in sorted((scene / "images").iterdir()):
        image = cv2.imread(str(src.resolve(strict=True)), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read {src}")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        dst = image_dir / src.name
        if not cv2.imwrite(str(dst), gray):
            raise RuntimeError(f"Could not write {dst}")
    return image_dir


def pair_id(image_id1: int, image_id2: int) -> int:
    a, b = sorted((image_id1, image_id2))
    return a * PAIR_ID_MULTIPLIER + b


def prepare_database(source_db: Path, dst_db: Path, active_bands: set[str], camera_blob: bytes, width: int, height: int) -> dict[str, int]:
    shutil.copy2(source_db, dst_db)
    con = sqlite3.connect(dst_db)
    cur = con.cursor()

    active_ids = {
        image_id: name
        for image_id, name in cur.execute("select image_id, name from images")
        if band_of_name(name) in active_bands
    }
    delete_ids = [
        image_id
        for image_id, name in cur.execute("select image_id, name from images")
        if band_of_name(name) not in active_bands
    ]
    ms_ids = [image_id for image_id, name in active_ids.items() if band_of_name(name) != "rgb"]

    cur.execute("delete from matches")
    cur.execute("delete from two_view_geometries")
    for image_id in delete_ids:
        cur.execute("delete from keypoints where image_id=?", (image_id,))
        cur.execute("delete from descriptors where image_id=?", (image_id,))
        cur.execute("delete from images where image_id=?", (image_id,))
    for image_id in ms_ids:
        cur.execute("delete from keypoints where image_id=?", (image_id,))
        cur.execute("delete from descriptors where image_id=?", (image_id,))
    cur.execute("update images set camera_id=1")
    cur.execute("delete from cameras where camera_id != 1")
    cur.execute(
        "update cameras set model=1, width=?, height=?, params=?, prior_focal_length=1 where camera_id=1",
        (width, height, camera_blob),
    )
    con.commit()
    counts = {
        "cameras": cur.execute("select count(*) from cameras").fetchone()[0],
        "images": cur.execute("select count(*) from images").fetchone()[0],
        "rgb_images": cur.execute("select count(*) from images where name like 'rgb_%'").fetchone()[0],
    }
    con.close()
    return counts


def write_image_lists(scene: Path, workspace: Path, bands: list[str]) -> dict[str, list[str]]:
    lists_dir = workspace / "lists"
    clean_dir(lists_dir)
    names_by_band: dict[str, list[str]] = {}
    for band in bands:
        names = sorted(p.name for p in (scene / "images").iterdir() if band_of_name(p.name) == band)
        names_by_band[band] = names
        (lists_dir / f"{band}_images.txt").write_text("\n".join(names) + "\n")
    all_names = [name for band in bands for name in names_by_band[band]]
    (lists_dir / "all_images.txt").write_text("\n".join(all_names) + "\n")
    pairs = [
        f"{ms_name} {rgb_name}"
        for band in bands
        if band != "rgb"
        for ms_name in names_by_band[band]
        for rgb_name in names_by_band["rgb"]
    ]
    (lists_dir / "match_pairs_ms_to_rgb_exhaustive.txt").write_text("\n".join(pairs) + "\n")
    return names_by_band


def run_feature_extraction(args: argparse.Namespace, db: Path, image_dir: Path, lists_dir: Path, bands: list[str], env: dict[str, str], cwd: Path) -> None:
    for band in bands:
        if band == "rgb":
            continue
        run(
            [
                args.colmap_bin,
                "feature_extractor",
                "--database_path",
                db,
                "--image_path",
                image_dir,
                "--image_list_path",
                lists_dir / f"{band}_images.txt",
                "--ImageReader.single_camera",
                "1",
                "--ImageReader.existing_camera_id",
                "1",
                "--ImageReader.camera_model",
                "PINHOLE",
                "--SiftExtraction.use_gpu",
                "1",
                "--SiftExtraction.gpu_index",
                args.colmap_gpu_index,
                "--SiftExtraction.num_threads",
                str(args.sift_num_threads),
                "--SiftExtraction.max_image_size",
                str(args.max_image_size),
                "--SiftExtraction.max_num_features",
                str(args.max_num_features),
                "--SiftExtraction.estimate_affine_shape",
                str(args.estimate_affine_shape),
                "--SiftExtraction.domain_size_pooling",
                str(args.domain_size_pooling),
            ],
            env=env,
            cwd=cwd,
        )


def run_matching_and_registration(args: argparse.Namespace, db: Path, workspace: Path, output_scene: Path, env: dict[str, str], cwd: Path) -> Path:
    run(
        [
            args.colmap_bin,
            "matches_importer",
            "--database_path",
            db,
            "--match_list_path",
            workspace / "lists" / "match_pairs_ms_to_rgb_exhaustive.txt",
            "--match_type",
            "pairs",
            "--SiftMatching.use_gpu",
            str(args.matching_use_gpu),
            "--SiftMatching.gpu_index",
            args.colmap_gpu_index,
            "--SiftMatching.max_num_matches",
            str(args.max_num_matches),
            "--SiftMatching.min_num_inliers",
            str(args.min_num_inliers),
            "--SiftMatching.guided_matching",
            str(args.guided_matching),
        ],
        env=env,
        cwd=cwd,
    )

    registered = workspace / "sparse_registered" / "0"
    clean_dir(registered)
    run(
        [
            args.colmap_bin,
            "image_registrator",
            "--database_path",
            db,
            "--input_path",
            workspace / "sparse_rgb" / "0",
            "--output_path",
            registered,
            "--Mapper.fix_existing_images",
            "1",
            "--Mapper.ba_refine_focal_length",
            "0",
            "--Mapper.ba_refine_principal_point",
            "0",
            "--Mapper.ba_refine_extra_params",
            "0",
            "--Mapper.ba_global_images_freq",
            "100000000",
            "--Mapper.ba_global_points_freq",
            "100000000",
            "--Mapper.abs_pose_min_num_inliers",
            str(args.abs_pose_min_num_inliers),
            "--Mapper.abs_pose_max_error",
            str(args.abs_pose_max_error),
            "--Mapper.min_num_matches",
            str(args.mapper_min_num_matches),
        ],
        env=env,
        cwd=cwd,
    )

    final_sparse = output_scene / "sparse" / "0"
    convert_model(args.colmap_bin, registered, final_sparse, env, cwd)
    return final_sparse


def qvec_to_rotmat(q: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = q
    return np.array(
        [
            [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
            [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qx * qw],
            [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx * qx - 2 * qy * qy],
        ]
    )


def parse_model_centers(images_txt: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    lines = [line.strip() for line in images_txt.read_text().splitlines() if line.strip() and not line.startswith("#")]
    for idx in range(0, len(lines), 2):
        parts = lines[idx].split()
        if len(parts) < 10:
            continue
        qvec = np.array([float(v) for v in parts[1:5]])
        tvec = np.array([float(v) for v in parts[5:8]])
        center = -(qvec_to_rotmat(qvec).T @ tvec)
        rows.append({"image_id": int(parts[0]), "name": parts[9], "band": band_of_name(parts[9]), "center": center})
    return rows


def audit_model(output_scene: Path, workspace: Path, bands: list[str]) -> None:
    rows = parse_model_centers(output_scene / "sparse" / "0" / "images.txt")
    counts = Counter(row["band"] for row in rows)
    rgb_centers = np.array([row["center"] for row in rows if row["band"] == "rgb"])
    nearest: dict[str, dict[str, float]] = {}
    far_examples: list[dict[str, Any]] = []
    if len(rgb_centers):
        for band in bands:
            if band == "rgb":
                continue
            distances = []
            for row in rows:
                if row["band"] != band:
                    continue
                d = float(np.linalg.norm(rgb_centers - row["center"], axis=1).min())
                distances.append(d)
                far_examples.append({"name": row["name"], "band": band, "nearest_rgb_distance": d})
            if distances:
                nearest[band] = {
                    "min": float(np.min(distances)),
                    "median": float(np.median(distances)),
                    "p95": float(np.percentile(distances, 95)),
                    "max": float(np.max(distances)),
                }

    summary = {
        "registered_images": len(rows),
        "registered_per_band": dict(sorted(counts.items())),
        "expected_bands": bands,
        "nearest_rgb_distance_by_band": nearest,
        "farthest_non_rgb_images": sorted(far_examples, key=lambda row: row["nearest_rgb_distance"], reverse=True)[:25],
        "registered_names": [row["name"] for row in rows],
    }
    write_json(output_scene / "metadata" / "registered_images_summary.json", summary)
    write_json(workspace / "registration_audit.json", summary)
    csv_lines = ["name,band,x,y,z"]
    for row in rows:
        x, y, z = row["center"]
        csv_lines.append(f"{row['name']},{row['band']},{x:.9f},{y:.9f},{z:.9f}")
    (workspace / "camera_centers.csv").write_text("\n".join(csv_lines) + "\n")


def main() -> None:
    args = parse_args()
    source = args.source_scene.resolve()
    output = args.output_scene.resolve()
    bands = [band for band in parse_band_list(args.bands) if band not in set(parse_band_list(args.exclude_bands))]
    if "rgb" not in bands:
        raise ValueError("The RGB band must be included as the fixed anchor.")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    print(f"Preparing {output} from {source}", flush=True)
    print(f"Active bands: {bands}", flush=True)
    prepare_scene(source, output, bands, args.copy_images)

    workspace = output / "colmap_gray_exhaustive"
    clean_dir(workspace)
    image_dir = stage_grayscale_images(output, workspace)
    shutil.copytree(source / "colmap_shared" / "sparse_rgb", workspace / "sparse_rgb")
    shutil.copytree(source / "colmap_shared" / "sparse_rgb", output / "colmap_shared" / "sparse_rgb")

    rgb_txt = workspace / "sparse_rgb_txt"
    convert_model(args.colmap_bin, workspace / "sparse_rgb" / "0", rgb_txt, env, output.parent)
    width, height, camera_blob = read_camera_params(rgb_txt / "cameras.txt")

    db = workspace / "database.db"
    db_counts = prepare_database(
        source / "colmap_shared" / "database.db",
        db,
        set(bands),
        camera_blob,
        width,
        height,
    )
    write_json(workspace / "database_prepare_summary.json", db_counts)

    names_by_band = write_image_lists(output, workspace, bands)
    write_json(workspace / "image_counts.json", {band: len(names) for band, names in names_by_band.items()})
    run_feature_extraction(args, db, image_dir, workspace / "lists", bands, env, output.parent)
    final_sparse = run_matching_and_registration(args, db, workspace, output, env, output.parent)
    audit_model(output, workspace, bands)
    print(f"Finished. Final sparse model: {final_sparse}", flush=True)


if __name__ == "__main__":
    main()
