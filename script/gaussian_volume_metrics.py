#!/usr/bin/env python3
"""Estimate implicit-object volumes from a Gaussian Splatting PLY.

The volume is defined by thresholding the Gaussian density field:

    Omega = {x : rho(x) >= tau}

This script implements three practical estimators for that thresholded object:
voxel occupancy, Monte Carlo integration, and marching-cubes mesh volume.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


@dataclass
class GaussianScene:
    xyz: np.ndarray
    opacity: np.ndarray
    scales: np.ndarray
    rotations: np.ndarray
    object_features: np.ndarray | None


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def field_names(vertex) -> set[str]:
    return {prop.name for prop in vertex.properties}


def numbered_fields(vertex, prefix: str) -> list[str]:
    names = [prop.name for prop in vertex.properties if prop.name.startswith(prefix)]
    return sorted(names, key=lambda name: int(name.split("_")[-1]))


def load_gaussian_ply(path: Path) -> GaussianScene:
    ply = PlyData.read(path)
    vertex = ply["vertex"]
    names = field_names(vertex)
    required = {"x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"}
    missing = sorted(required - names)
    if missing:
        raise ValueError(f"{path} is missing Gaussian Splat fields: {', '.join(missing)}")

    xyz = np.column_stack([np.asarray(vertex[name], dtype=np.float64) for name in ("x", "y", "z")])
    opacity = sigmoid(np.asarray(vertex["opacity"], dtype=np.float64))
    scales = np.exp(np.column_stack([np.asarray(vertex[f"scale_{idx}"], dtype=np.float64) for idx in range(3)]))
    rotations = np.column_stack([np.asarray(vertex[f"rot_{idx}"], dtype=np.float64) for idx in range(4)])

    obj_names = numbered_fields(vertex, "obj_dc_")
    object_features = None
    if obj_names:
        object_features = np.column_stack([np.asarray(vertex[name], dtype=np.float64) for name in obj_names])

    rotations = normalize_quaternions(rotations)
    return GaussianScene(xyz=xyz, opacity=opacity, scales=scales, rotations=rotations, object_features=object_features)


def normalize_quaternions(quaternions: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return quaternions / norms


def quaternion_to_rotation_matrix(quaternions: np.ndarray) -> np.ndarray:
    q = normalize_quaternions(quaternions)
    r, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    matrix = np.empty((q.shape[0], 3, 3), dtype=np.float64)
    matrix[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    matrix[:, 0, 1] = 2.0 * (x * y - r * z)
    matrix[:, 0, 2] = 2.0 * (x * z + r * y)
    matrix[:, 1, 0] = 2.0 * (x * y + r * z)
    matrix[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    matrix[:, 1, 2] = 2.0 * (y * z - r * x)
    matrix[:, 2, 0] = 2.0 * (x * z - r * y)
    matrix[:, 2, 1] = 2.0 * (y * z + r * x)
    matrix[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return matrix


def filter_scene(
    scene: GaussianScene,
    min_opacity: float,
    object_argmax: int | None,
    object_channel: int | None,
    object_min: float,
) -> GaussianScene:
    mask = scene.opacity >= min_opacity
    if object_argmax is not None:
        if scene.object_features is None:
            raise ValueError("--object-argmax requires obj_dc_* fields in the PLY")
        if object_argmax < 0 or object_argmax >= scene.object_features.shape[1]:
            raise ValueError(f"--object-argmax must be in [0, {scene.object_features.shape[1] - 1}]")
        mask &= np.argmax(scene.object_features, axis=1) == object_argmax
    if object_channel is not None:
        if scene.object_features is None:
            raise ValueError("--object-channel requires obj_dc_* fields in the PLY")
        if object_channel < 0 or object_channel >= scene.object_features.shape[1]:
            raise ValueError(f"--object-channel must be in [0, {scene.object_features.shape[1] - 1}]")
        mask &= scene.object_features[:, object_channel] >= object_min

    if not np.any(mask):
        raise ValueError("Filtering removed all Gaussians")

    return GaussianScene(
        xyz=scene.xyz[mask],
        opacity=scene.opacity[mask],
        scales=scene.scales[mask],
        rotations=scene.rotations[mask],
        object_features=None if scene.object_features is None else scene.object_features[mask],
    )


def density_weights(scene: GaussianScene, density_mode: str) -> np.ndarray:
    if density_mode == "amplitude":
        return scene.opacity
    if density_mode == "normalized":
        denom = ((2.0 * math.pi) ** 1.5) * np.prod(scene.scales, axis=1)
        return scene.opacity / np.maximum(denom, 1e-18)
    raise ValueError(f"Unknown density mode: {density_mode}")


def evaluate_density(
    points: np.ndarray,
    scene: GaussianScene,
    density_mode: str,
    point_chunk: int,
    gaussian_chunk: int,
) -> np.ndarray:
    rotations = quaternion_to_rotation_matrix(scene.rotations)
    weights = density_weights(scene, density_mode)
    density = np.zeros(points.shape[0], dtype=np.float64)

    for point_start in range(0, points.shape[0], point_chunk):
        point_stop = min(point_start + point_chunk, points.shape[0])
        pts = points[point_start:point_stop]
        chunk_density = np.zeros(pts.shape[0], dtype=np.float64)
        for gaussian_start in range(0, scene.xyz.shape[0], gaussian_chunk):
            gaussian_stop = min(gaussian_start + gaussian_chunk, scene.xyz.shape[0])
            diff = pts[:, None, :] - scene.xyz[None, gaussian_start:gaussian_stop, :]
            local = np.einsum("pgi,gij->pgj", diff, rotations[gaussian_start:gaussian_stop], optimize=True)
            local /= np.maximum(scene.scales[None, gaussian_start:gaussian_stop, :], 1e-12)
            mahalanobis = np.sum(local * local, axis=2)
            chunk_density += np.exp(-0.5 * mahalanobis) @ weights[gaussian_start:gaussian_stop]
        density[point_start:point_stop] = chunk_density
    return density


def bounding_box(scene: GaussianScene, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    radius = sigma * np.max(scene.scales, axis=1)
    bbox_min = np.min(scene.xyz - radius[:, None], axis=0)
    bbox_max = np.max(scene.xyz + radius[:, None], axis=0)
    return bbox_min, bbox_max


def voxel_size_from_resolution(bbox_min: np.ndarray, bbox_max: np.ndarray, grid_resolution: int) -> float:
    longest = float(np.max(bbox_max - bbox_min))
    return longest / float(grid_resolution)


def grid_shape(bbox_min: np.ndarray, bbox_max: np.ndarray, voxel_size: float) -> np.ndarray:
    return np.maximum(np.ceil((bbox_max - bbox_min) / voxel_size).astype(np.int64), 1)


def voxel_volume(
    scene: GaussianScene,
    threshold: float,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    voxel_size: float,
    density_mode: str,
    point_chunk: int,
    gaussian_chunk: int,
) -> dict:
    shape = grid_shape(bbox_min, bbox_max, voxel_size)
    axes = [bbox_min[dim] + (np.arange(shape[dim], dtype=np.float64) + 0.5) * voxel_size for dim in range(3)]
    occupied = 0
    total = int(np.prod(shape))

    for x in axes[0]:
        yy, zz = np.meshgrid(axes[1], axes[2], indexing="ij")
        points = np.column_stack(
            [
                np.full(yy.size, x, dtype=np.float64),
                yy.ravel(),
                zz.ravel(),
            ]
        )
        occupied += int(
            np.count_nonzero(evaluate_density(points, scene, density_mode, point_chunk, gaussian_chunk) >= threshold)
        )

    return {
        "voxel_size_scene_units": float(voxel_size),
        "grid_shape": shape.astype(int).tolist(),
        "sampled_voxel_count": total,
        "occupied_voxel_count": int(occupied),
        "volume_scene_units_cubed": float(occupied * voxel_size**3),
    }


def monte_carlo_volume(
    scene: GaussianScene,
    threshold: float,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    sample_count: int,
    seed: int,
    density_mode: str,
    point_chunk: int,
    gaussian_chunk: int,
) -> dict:
    rng = np.random.default_rng(seed)
    span = bbox_max - bbox_min
    inside = 0
    processed = 0
    while processed < sample_count:
        count = min(point_chunk, sample_count - processed)
        points = bbox_min + rng.random((count, 3)) * span
        inside += int(np.count_nonzero(evaluate_density(points, scene, density_mode, point_chunk, gaussian_chunk) >= threshold))
        processed += count

    fraction = inside / float(sample_count)
    box_volume = float(np.prod(span))
    return {
        "sample_count": int(sample_count),
        "seed": int(seed),
        "inside_count": int(inside),
        "inside_fraction": float(fraction),
        "bbox_volume_scene_units_cubed": box_volume,
        "volume_scene_units_cubed": float(box_volume * fraction),
    }


def density_grid(
    scene: GaussianScene,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    voxel_size: float,
    density_mode: str,
    point_chunk: int,
    gaussian_chunk: int,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    shape = grid_shape(bbox_min, bbox_max, voxel_size) + 1
    axes = tuple(bbox_min[dim] + np.arange(shape[dim], dtype=np.float64) * voxel_size for dim in range(3))
    grid = np.empty(tuple(shape), dtype=np.float32)

    for ix, x in enumerate(axes[0]):
        yy, zz = np.meshgrid(axes[1], axes[2], indexing="ij")
        points = np.column_stack(
            [
                np.full(yy.size, x, dtype=np.float64),
                yy.ravel(),
                zz.ravel(),
            ]
        )
        grid[ix, :, :] = evaluate_density(points, scene, density_mode, point_chunk, gaussian_chunk).reshape(
            len(axes[1]), len(axes[2])
        )
    return grid, axes


def mesh_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    tri = vertices[faces]
    signed = np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])) / 6.0
    return float(abs(np.sum(signed)))


def marching_cubes_volume(
    scene: GaussianScene,
    threshold: float,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    voxel_size: float,
    density_mode: str,
    point_chunk: int,
    gaussian_chunk: int,
    mesh_path: Path | None,
) -> dict:
    try:
        from skimage import measure
    except ImportError as exc:
        raise RuntimeError("Marching cubes requires scikit-image. Activate the project env or install scikit-image.") from exc

    grid, _ = density_grid(scene, bbox_min, bbox_max, voxel_size, density_mode, point_chunk, gaussian_chunk)
    grid_min = float(np.min(grid))
    grid_max = float(np.max(grid))
    if not (grid_min <= threshold <= grid_max):
        return {
            "voxel_size_scene_units": float(voxel_size),
            "grid_shape": list(grid.shape),
            "density_min": grid_min,
            "density_max": grid_max,
            "mesh_vertex_count": 0,
            "mesh_face_count": 0,
            "volume_scene_units_cubed": 0.0,
            "warning": "Threshold is outside the sampled density range; no isosurface was extracted.",
        }

    vertices, faces, _normals, _values = measure.marching_cubes(
        grid,
        level=threshold,
        spacing=(voxel_size, voxel_size, voxel_size),
    )
    vertices = vertices + bbox_min[None, :]
    volume = mesh_volume(vertices, faces)
    if mesh_path is not None:
        mesh_path.parent.mkdir(parents=True, exist_ok=True)
        write_mesh(mesh_path, vertices, faces.astype(np.int32))

    return {
        "voxel_size_scene_units": float(voxel_size),
        "grid_shape": list(grid.shape),
        "density_min": grid_min,
        "density_max": grid_max,
        "mesh_vertex_count": int(vertices.shape[0]),
        "mesh_face_count": int(faces.shape[0]),
        "volume_scene_units_cubed": float(volume),
        "mesh_ply": None if mesh_path is None else str(mesh_path),
    }


def write_mesh(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    vertex_data = np.empty(vertices.shape[0], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    vertex_data["x"], vertex_data["y"], vertex_data["z"] = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    face_data = np.empty(faces.shape[0], dtype=[("vertex_indices", "i4", (3,))])
    face_data["vertex_indices"] = faces
    PlyData([PlyElement.describe(vertex_data, "vertex"), PlyElement.describe(face_data, "face")], text=False).write(path)


def resolve_threshold(
    scene: GaussianScene,
    threshold: str,
    percentile: float,
    density_mode: str,
    point_chunk: int,
    gaussian_chunk: int,
) -> tuple[float, dict]:
    if threshold != "auto":
        value = float(threshold)
        return value, {"mode": "manual", "value": value}

    center_density = evaluate_density(scene.xyz, scene, density_mode, point_chunk, gaussian_chunk)
    value = float(np.percentile(center_density, percentile))
    return value, {
        "mode": "auto_center_density_percentile",
        "percentile": float(percentile),
        "value": value,
        "center_density_min": float(np.min(center_density)),
        "center_density_p50": float(np.percentile(center_density, 50)),
        "center_density_p90": float(np.percentile(center_density, 90)),
        "center_density_max": float(np.max(center_density)),
    }


def summarize_input(scene: GaussianScene, bbox_min: np.ndarray, bbox_max: np.ndarray) -> dict:
    return {
        "gaussian_count": int(scene.xyz.shape[0]),
        "bbox": {
            "min": bbox_min.astype(float).tolist(),
            "max": bbox_max.astype(float).tolist(),
            "size": (bbox_max - bbox_min).astype(float).tolist(),
            "volume_scene_units_cubed": float(np.prod(bbox_max - bbox_min)),
        },
        "opacity": {
            "min": float(np.min(scene.opacity)),
            "p50": float(np.percentile(scene.opacity, 50)),
            "p90": float(np.percentile(scene.opacity, 90)),
            "max": float(np.max(scene.opacity)),
        },
        "scale_scene_units": {
            "min": float(np.min(scene.scales)),
            "p50": float(np.percentile(scene.scales, 50)),
            "p90": float(np.percentile(scene.scales, 90)),
            "max": float(np.max(scene.scales)),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ply", required=True, help="Gaussian Splatting PLY with x/y/z, opacity, scale_*, rot_* fields.")
    parser.add_argument("--out", required=True, help="Output JSON report path.")
    parser.add_argument("--mesh-ply", help="Optional marching-cubes mesh PLY output path.")
    parser.add_argument("--methods", nargs="+", default=["voxel", "monte_carlo", "marching_cubes"])
    parser.add_argument("--density-mode", choices=["amplitude", "normalized"], default="amplitude")
    parser.add_argument("--threshold", default="auto", help="Density threshold tau, or 'auto'.")
    parser.add_argument("--threshold-percentile", type=float, default=10.0)
    parser.add_argument("--grid-resolution", type=int, default=96, help="Voxel count along the longest bbox axis.")
    parser.add_argument("--voxel-size", type=float, help="Override --grid-resolution with an explicit scene-unit voxel size.")
    parser.add_argument("--bbox-sigma", type=float, default=3.0, help="Extend bbox by this many max Gaussian scales.")
    parser.add_argument("--mc-samples", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--point-chunk", type=int, default=8192)
    parser.add_argument("--gaussian-chunk", type=int, default=2048)
    parser.add_argument("--min-opacity", type=float, default=0.0)
    parser.add_argument("--object-argmax", type=int, help="Keep Gaussians whose largest obj_dc_* channel is this index.")
    parser.add_argument("--object-channel", type=int, help="Keep Gaussians whose selected obj_dc_* channel is >= --object-min.")
    parser.add_argument("--object-min", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods = set(args.methods)
    valid_methods = {"voxel", "monte_carlo", "marching_cubes"}
    unknown = sorted(methods - valid_methods)
    if unknown:
        raise ValueError(f"Unknown methods: {', '.join(unknown)}")

    scene_all = load_gaussian_ply(Path(args.ply))
    scene = filter_scene(scene_all, args.min_opacity, args.object_argmax, args.object_channel, args.object_min)
    bbox_min, bbox_max = bounding_box(scene, args.bbox_sigma)
    voxel_size = args.voxel_size or voxel_size_from_resolution(bbox_min, bbox_max, args.grid_resolution)
    threshold, threshold_report = resolve_threshold(
        scene,
        args.threshold,
        args.threshold_percentile,
        args.density_mode,
        args.point_chunk,
        args.gaussian_chunk,
    )

    report = {
        "input_ply": str(Path(args.ply).resolve()),
        "density_definition": {
            "mode": args.density_mode,
            "amplitude": "sum opacity_i * exp(-0.5 * mahalanobis_i)",
            "normalized": "sum opacity_i * N(x | mean_i, covariance_i)",
            "threshold": threshold_report,
        },
        "filter": {
            "input_gaussian_count": int(scene_all.xyz.shape[0]),
            "kept_gaussian_count": int(scene.xyz.shape[0]),
            "min_opacity": float(args.min_opacity),
            "object_argmax": args.object_argmax,
            "object_channel": args.object_channel,
            "object_min": float(args.object_min),
        },
        "input_summary": summarize_input(scene, bbox_min, bbox_max),
        "methods": {},
    }

    if "voxel" in methods:
        report["methods"]["voxel"] = voxel_volume(
            scene,
            threshold,
            bbox_min,
            bbox_max,
            voxel_size,
            args.density_mode,
            args.point_chunk,
            args.gaussian_chunk,
        )
    if "monte_carlo" in methods:
        report["methods"]["monte_carlo"] = monte_carlo_volume(
            scene,
            threshold,
            bbox_min,
            bbox_max,
            args.mc_samples,
            args.seed,
            args.density_mode,
            args.point_chunk,
            args.gaussian_chunk,
        )
    if "marching_cubes" in methods:
        report["methods"]["marching_cubes"] = marching_cubes_volume(
            scene,
            threshold,
            bbox_min,
            bbox_max,
            voxel_size,
            args.density_mode,
            args.point_chunk,
            args.gaussian_chunk,
            None if args.mesh_ply is None else Path(args.mesh_ply),
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
