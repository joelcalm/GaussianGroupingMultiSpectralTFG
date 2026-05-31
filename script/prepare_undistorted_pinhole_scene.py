#!/usr/bin/env python3
"""Create a pinhole, undistorted copy of a COLMAP vineyard scene."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


def parse_camera(path: Path) -> tuple[int, int, float, float, float, float, np.ndarray]:
    cameras = [
        line.split()
        for line in path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if len(cameras) != 1:
        raise ValueError(f"Expected one shared camera in {path}, found {len(cameras)}")
    parts = cameras[0]
    model = parts[1]
    width = int(parts[2])
    height = int(parts[3])
    params = [float(x) for x in parts[4:]]
    if model == "PINHOLE":
        fx, fy, cx, cy = params[:4]
        dist = np.zeros(4, dtype=np.float64)
    elif model == "OPENCV":
        fx, fy, cx, cy, k1, k2, p1, p2 = params[:8]
        dist = np.array([k1, k2, p1, p2], dtype=np.float64)
    else:
        raise ValueError(f"Unsupported camera model {model}; expected OPENCV or PINHOLE")
    return width, height, fx, fy, cx, cy, dist


def iter_image_records(path: Path):
    lines = path.read_text().splitlines()
    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()
        if not line or line.startswith("#"):
            idx += 1
            continue
        parts = line.split()
        if len(parts) >= 10 and parts[9].endswith(".png"):
            yield line, parts[9]
            idx += 2
        else:
            idx += 1


def write_images_txt(src: Path, dst: Path, image_count: int) -> None:
    records = list(iter_image_records(src))
    with dst.open("w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {image_count}\n")
        for line, _name in records:
            f.write(line + "\n\n")


def ensure_clean_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} already exists; pass --overwrite")
        shutil.rmtree(path)
    path.mkdir(parents=True)


def copy_json_sidecars(source: Path, output: Path) -> None:
    for name in [
        "band_info.json",
        "clean_scene_manifest.json",
        "colmap_quality.json",
        "extraction_report.json",
        "frame_info.json",
        "partial_channels_summary.json",
        "videos_manifest.json",
    ]:
        src = source / name
        if src.exists():
            shutil.copy2(src, output / name)
    if (source / "metadata").exists():
        shutil.copytree(source / "metadata", output / "metadata", dirs_exist_ok=True)


def undistort_dir(
    source_dir: Path,
    output_dir: Path,
    names: list[str],
    map1: np.ndarray,
    map2: np.ndarray,
    interpolation: int,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for name in names:
        src = source_dir / name
        if not src.exists():
            continue
        image = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Could not read {src}")
        undistorted = cv2.remap(
            image,
            map1,
            map2,
            interpolation=interpolation,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        cv2.imwrite(str(output_dir / name), undistorted)
        written += 1
    return written


def parse_points3d(path: Path):
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        xyz = np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float64)
        rgb = np.array([int(parts[4]), int(parts[5]), int(parts[6])], dtype=np.uint8)
        error = float(parts[7])
        yield xyz, rgb, error, line


def write_points3d_txt(path: Path, rows: list[tuple[np.ndarray, np.ndarray, float, str]]) -> None:
    with path.open("w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {len(rows)}\n")
        for new_id, (xyz, rgb, error, _line) in enumerate(rows, start=1):
            f.write(
                f"{new_id} {xyz[0]:.12g} {xyz[1]:.12g} {xyz[2]:.12g} "
                f"{int(rgb[0])} {int(rgb[1])} {int(rgb[2])} {error:.12g}\n"
            )


def write_ply(path: Path, rows: list[tuple[np.ndarray, np.ndarray, float, str]]) -> None:
    with path.open("w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(rows)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property float nx\n")
        f.write("property float ny\n")
        f.write("property float nz\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for xyz, rgb, _error, _line in rows:
            f.write(
                f"{xyz[0]:.9g} {xyz[1]:.9g} {xyz[2]:.9g} "
                f"0 0 0 {int(rgb[0])} {int(rgb[1])} {int(rgb[2])}\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max_point_radius", type=float, default=30.0)
    parser.add_argument("--max_point_error", type=float, default=2.5)
    args = parser.parse_args()

    source = args.source
    output = args.output
    sparse_src = source / "sparse" / "0"
    sparse_out = output / "sparse" / "0"

    width, height, fx, fy, cx, cy, dist = parse_camera(sparse_src / "cameras.txt")
    k = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    map1, map2 = cv2.initUndistortRectifyMap(
        k,
        dist,
        None,
        k,
        (width, height),
        cv2.CV_32FC1,
    )

    records = list(iter_image_records(sparse_src / "images.txt"))
    registered_names = [name for _line, name in records]
    rgb_names = [name for name in registered_names if name.startswith("rgb_")]

    ensure_clean_dir(output, args.overwrite)
    sparse_out.mkdir(parents=True, exist_ok=True)
    copy_json_sidecars(source, output)

    undistort_dir(source / "images", output / "images", registered_names, map1, map2, cv2.INTER_LINEAR)
    undistort_dir(source / "images_rgb", output / "images_rgb", rgb_names, map1, map2, cv2.INTER_LINEAR)
    undistort_dir(source / "object_mask", output / "object_mask", rgb_names, map1, map2, cv2.INTER_NEAREST)
    undistort_dir(source / "semantic_mask", output / "semantic_mask", rgb_names, map1, map2, cv2.INTER_NEAREST)

    (sparse_out / "cameras.txt").write_text(
        "# Camera list with one line of data per camera:\n"
        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        "# Number of cameras: 1\n"
        f"1 PINHOLE {width} {height} {fx:.12g} {fy:.12g} {width / 2:.12g} {height / 2:.12g}\n"
    )
    write_images_txt(sparse_src / "images.txt", sparse_out / "images.txt", len(registered_names))

    rows = []
    for xyz, rgb, error, line in parse_points3d(sparse_src / "points3D.txt"):
        if np.linalg.norm(xyz) <= args.max_point_radius and error <= args.max_point_error:
            rows.append((xyz, rgb, error, line))
    write_points3d_txt(sparse_out / "points3D.txt", rows)
    write_ply(sparse_out / "points3D.ply", rows)

    report = {
        "source": str(source),
        "output": str(output),
        "camera_model": "PINHOLE",
        "width": width,
        "height": height,
        "fx": fx,
        "fy": fy,
        "cx": width / 2,
        "cy": height / 2,
        "distortion_removed": dist.tolist(),
        "registered_images": len(registered_names),
        "rgb_images": len(rgb_names),
        "max_point_radius": args.max_point_radius,
        "max_point_error": args.max_point_error,
        "kept_points": len(rows),
    }
    (output / "undistort_pinhole_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
