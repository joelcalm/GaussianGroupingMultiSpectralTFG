#!/usr/bin/env python3
"""Export reference and Sim(3)-aligned COLMAP sparse points as PLY files."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

if __package__ in (None, ""):
    this_dir = Path(__file__).resolve().parent
    sys.path.append(str(this_dir))
else:
    this_dir = Path(__file__).resolve().parent

from config import load_config  # noqa: E402

repo_root = this_dir.parents[1]
_colmap_loader_path = repo_root / "scene" / "colmap_loader.py"
_colmap_spec = importlib.util.spec_from_file_location("temporal_colmap_loader", _colmap_loader_path)
if _colmap_spec is None or _colmap_spec.loader is None:
    raise ImportError(f"Could not load COLMAP loader from {_colmap_loader_path}")
_colmap_loader = importlib.util.module_from_spec(_colmap_spec)
_colmap_spec.loader.exec_module(_colmap_loader)
read_points3D_binary = _colmap_loader.read_points3D_binary
read_points3D_text = _colmap_loader.read_points3D_text


def read_ply_points(path: Path) -> tuple[np.ndarray, np.ndarray]:
    plydata = PlyData.read(str(path))
    vertices = plydata["vertex"]
    xyz = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T.astype(np.float64)
    if {"red", "green", "blue"}.issubset(vertices.data.dtype.names or []):
        rgb = np.vstack([vertices["red"], vertices["green"], vertices["blue"]]).T.astype(np.uint8)
    else:
        rgb = np.full((xyz.shape[0], 3), 255, dtype=np.uint8)
    return xyz, rgb


def write_ply_points(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"), ("nx", "f4"), ("ny", "f4"), ("nz", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    normals = np.zeros_like(xyz, dtype=np.float32)
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    elements = np.empty(xyz.shape[0], dtype=dtype)
    elements[:] = list(map(tuple, np.concatenate([xyz.astype(np.float32), normals, rgb], axis=1)))
    PlyData([PlyElement.describe(elements, "vertex")]).write(str(path))


def read_colmap_points(colmap_path: Path) -> tuple[np.ndarray, np.ndarray]:
    ply_path = colmap_path / "points3D.ply"
    if ply_path.exists():
        return read_ply_points(ply_path)
    bin_path = colmap_path / "points3D.bin"
    if bin_path.exists():
        xyz, rgb, _ = read_points3D_binary(str(bin_path))
        return xyz, rgb.astype(np.uint8)
    txt_path = colmap_path / "points3D.txt"
    if txt_path.exists():
        xyz, rgb, _ = read_points3D_text(str(txt_path))
        return xyz, rgb.astype(np.uint8)
    raise FileNotFoundError(f"No points3D.ply/bin/txt found in {colmap_path}")


def load_transform(path: Path) -> tuple[float, np.ndarray, np.ndarray]:
    payload = json.loads(path.read_text())
    return (
        float(payload["scale"]),
        np.asarray(payload["rotation"], dtype=float),
        np.asarray(payload["translation"], dtype=float),
    )


def apply_transform(xyz: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return scale * (xyz @ rotation.T) + translation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Temporal tracking YAML config.")
    parser.add_argument(
        "--transforms-dir",
        default=None,
        help="Directory containing <source>_to_<reference>.json. Defaults to <output_dir>/transforms.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    transforms_dir = Path(args.transforms_dir) if args.transforms_dir else cfg.output_dir / "transforms"
    out_dir = cfg.output_dir / "aligned_colmap"
    out_dir.mkdir(parents=True, exist_ok=True)

    ref = cfg.reference
    xyz, rgb = read_colmap_points(ref.colmap_path)
    ref_out = out_dir / f"{ref.name}_points_reference.ply"
    write_ply_points(ref_out, xyz, rgb)
    print(f"Wrote reference points: {ref_out}")

    for scene in cfg.scenes.values():
        if scene.name == cfg.reference_scene:
            continue
        transform_path = transforms_dir / f"{scene.name}_to_{cfg.reference_scene}.json"
        if not transform_path.exists():
            print(f"Skipping {scene.name}: missing transform {transform_path}")
            continue
        xyz, rgb = read_colmap_points(scene.colmap_path)
        scale, rotation, translation = load_transform(transform_path)
        aligned = apply_transform(xyz, scale, rotation, translation)
        out_path = out_dir / f"{scene.name}_points_aligned.ply"
        write_ply_points(out_path, aligned, rgb)
        print(f"Wrote aligned points: {out_path}")


if __name__ == "__main__":
    main()
