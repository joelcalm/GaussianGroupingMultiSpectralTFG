#!/usr/bin/env python3
"""Estimate arbitrary-unit geometry metrics from a cleaned vineyard PLY."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from plyfile import PlyData, PlyElement
from scipy.spatial import Delaunay, cKDTree


def load_ply_xyz_opacity(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    ply = PlyData.read(path)
    vertex = ply["vertex"]
    xyz = np.column_stack([np.asarray(vertex[name], dtype=np.float64) for name in ("x", "y", "z")])
    names = {prop.name for prop in vertex.properties}
    if "opacity" not in names:
        return xyz, None
    opacity_logit = np.asarray(vertex["opacity"], dtype=np.float64)
    opacity = 1.0 / (1.0 + np.exp(-opacity_logit))
    return xyz, opacity


def summarize(values: np.ndarray) -> dict:
    return {
        "min": float(np.min(values)),
        "p10": float(np.percentile(values, 10)),
        "p50": float(np.percentile(values, 50)),
        "mean": float(np.mean(values)),
        "p90": float(np.percentile(values, 90)),
        "max": float(np.max(values)),
    }


def nearest_neighbor_stats(xyz: np.ndarray) -> dict:
    distances, _ = cKDTree(xyz).query(xyz, k=2)
    return summarize(distances[:, 1])


def voxel_stats(xyz: np.ndarray, voxel_size: float) -> dict:
    bbox_min = xyz.min(axis=0)
    bbox_max = xyz.max(axis=0)
    indices = np.floor((xyz - bbox_min) / voxel_size).astype(np.int64)
    occupied = np.unique(indices, axis=0).shape[0]
    return {
        "voxel_size_scene_units": float(voxel_size),
        "occupied_voxel_count": int(occupied),
        "occupied_volume_scene_units_cubed": float(occupied * voxel_size**3),
    }


def tetra_circumradius(points: np.ndarray) -> float:
    p0 = points[0]
    matrix = 2.0 * (points[1:] - p0)
    rhs = np.sum(points[1:] ** 2 - p0**2, axis=1)
    try:
        center = np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        return math.inf
    return float(np.linalg.norm(center - p0))


def alpha_shape_faces(xyz: np.ndarray, alpha_radius: float) -> np.ndarray:
    delaunay = Delaunay(xyz)
    face_hits: dict[tuple[int, int, int], list[tuple[tuple[int, int, int], int]]] = {}
    for simplex in delaunay.simplices:
        if tetra_circumradius(xyz[simplex]) > alpha_radius:
            continue
        i, j, k, l = [int(idx) for idx in simplex]
        for face, opposite in (
            ((i, j, k), l),
            ((i, l, j), k),
            ((j, l, k), i),
            ((i, k, l), j),
        ):
            face_hits.setdefault(tuple(sorted(face)), []).append((face, opposite))

    faces = []
    for candidates in face_hits.values():
        if len(candidates) != 1:
            continue
        face, opposite = candidates[0]
        a, b, c = face
        pa, pb, pc = xyz[[a, b, c]]
        if np.dot(np.cross(pb - pa, pc - pa), xyz[opposite] - pa) > 0:
            b, c = c, b
        faces.append((a, b, c))
    return np.asarray(faces, dtype=np.int32)


def triangle_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = vertices[faces]
    return 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)


def mesh_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    tri = vertices[faces]
    signed = np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])) / 6.0
    return float(abs(np.sum(signed)))


def sample_mesh(vertices: np.ndarray, faces: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    areas = triangle_areas(vertices, faces)
    face_ids = rng.choice(len(faces), size=count, p=areas / areas.sum())
    tri = vertices[faces[face_ids]]
    u = rng.random(count)
    v = rng.random(count)
    flip = u + v > 1.0
    u[flip] = 1.0 - u[flip]
    v[flip] = 1.0 - v[flip]
    return tri[:, 0] + u[:, None] * (tri[:, 1] - tri[:, 0]) + v[:, None] * (tri[:, 2] - tri[:, 0])


def poisson_like_sample(
    vertices: np.ndarray,
    faces: np.ndarray,
    target_count: int,
    rng: np.random.Generator,
    oversample: int,
) -> tuple[np.ndarray, float]:
    area = float(triangle_areas(vertices, faces).sum())
    base_radius = math.sqrt(area / (2.0 * math.sqrt(3.0) * target_count))
    candidates = sample_mesh(vertices, faces, target_count * oversample, rng)
    best = candidates[:0]
    best_radius = base_radius
    for scale in (1.25, 1.0, 0.85, 0.7, 0.55, 0.4, 0.3, 0.2):
        radius = base_radius * scale
        radius_sq = radius * radius
        accepted = []
        for point in candidates:
            if accepted:
                existing = np.asarray(accepted)
                if np.min(np.sum((existing - point) ** 2, axis=1)) < radius_sq:
                    continue
            accepted.append(point)
            if len(accepted) >= target_count:
                break
        accepted = np.asarray(accepted, dtype=np.float64)
        if accepted.shape[0] > best.shape[0]:
            best = accepted
            best_radius = radius
        if accepted.shape[0] >= target_count:
            break
    return best[:target_count], best_radius


def write_points(path: Path, xyz: np.ndarray, color: tuple[int, int, int]) -> None:
    data = np.empty(
        xyz.shape[0],
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")],
    )
    data["x"], data["y"], data["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    data["red"], data["green"], data["blue"] = color
    PlyData([PlyElement.describe(data, "vertex")], text=False).write(path)


def write_mesh(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    vertex_data = np.empty(vertices.shape[0], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    vertex_data["x"], vertex_data["y"], vertex_data["z"] = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    face_data = np.empty(faces.shape[0], dtype=[("vertex_indices", "i4", (3,))])
    face_data["vertex_indices"] = faces
    PlyData([PlyElement.describe(vertex_data, "vertex"), PlyElement.describe(face_data, "face")], text=False).write(path)


def set_equal_axes(ax, points: np.ndarray) -> None:
    center = points.mean(axis=0)
    half = float(np.max(points.max(axis=0) - points.min(axis=0))) / 2.0
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")


def write_overview(path: Path, xyz: np.ndarray, faces: np.ndarray, sampled: np.ndarray) -> None:
    fig = plt.figure(figsize=(15, 5.8))
    axes = [fig.add_subplot(1, 3, idx + 1, projection="3d") for idx in range(3)]
    axes[0].scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=5, c="#555555", alpha=0.75)
    axes[0].set_title("Original Gaussian centers", pad=18)
    mesh = Poly3DCollection(xyz[faces], alpha=0.35, facecolor="#2e86ab", edgecolor="#1b4965", linewidths=0.15)
    axes[1].add_collection3d(mesh)
    axes[1].scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=2, c="#333333", alpha=0.25)
    axes[1].set_title("Alpha surface triangles", pad=18)
    axes[2].scatter(sampled[:, 0], sampled[:, 1], sampled[:, 2], s=7, c="#d1495b", alpha=0.9)
    axes[2].set_title("Poisson-like surface samples", pad=18)
    for ax in axes:
        set_equal_axes(ax, np.vstack([xyz, sampled]))
        ax.view_init(elev=20, azim=-60)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.03, top=0.88, wspace=0.08)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ply", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--voxel_sizes", nargs="+", type=float, default=[0.1, 0.05, 0.025, 0.01])
    parser.add_argument("--alpha_multiplier", type=float, default=4.0)
    parser.add_argument("--sample_count", type=int)
    parser.add_argument("--oversample", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    xyz, opacity = load_ply_xyz_opacity(Path(args.ply))
    sample_count = args.sample_count or xyz.shape[0]
    nn_stats = nearest_neighbor_stats(xyz)
    alpha_radius = nn_stats["p50"] * args.alpha_multiplier
    faces = alpha_shape_faces(xyz, alpha_radius)
    if faces.shape[0] == 0:
        raise RuntimeError("Alpha shape produced no triangles; increase --alpha_multiplier")

    areas = triangle_areas(xyz, faces)
    sampled, poisson_radius = poisson_like_sample(xyz, faces, sample_count, np.random.default_rng(args.seed), args.oversample)
    bbox_size = xyz.max(axis=0) - xyz.min(axis=0)

    write_points(out_dir / "original_gaussian_centers.ply", xyz, (85, 85, 85))
    write_mesh(out_dir / "alpha_surface_mesh.ply", xyz, faces)
    write_points(out_dir / "poisson_surface_points.ply", sampled, (209, 73, 91))
    write_overview(out_dir / "surface_sampling_overview.png", xyz, faces, sampled)

    report = {
        "input_ply": str(Path(args.ply).resolve()),
        "gaussian_count": int(xyz.shape[0]),
        "opacity_counts": None
        if opacity is None
        else {
            "gt_0p1": int((opacity > 0.1).sum()),
            "gt_0p5": int((opacity > 0.5).sum()),
            "gt_0p9": int((opacity > 0.9).sum()),
        },
        "bbox": {
            "min": xyz.min(axis=0).astype(float).tolist(),
            "max": xyz.max(axis=0).astype(float).tolist(),
            "size": bbox_size.astype(float).tolist(),
            "volume_scene_units_cubed": float(np.prod(bbox_size)),
        },
        "nearest_neighbor_distance_scene_units": nn_stats,
        "voxel_count": [voxel_stats(xyz, size) for size in args.voxel_sizes],
        "alpha_surface": {
            "alpha_multiplier": float(args.alpha_multiplier),
            "alpha_radius_scene_units": float(alpha_radius),
            "surface_triangle_count": int(faces.shape[0]),
            "surface_area_scene_units_squared": float(areas.sum()),
            "closed_mesh_volume_scene_units_cubed": mesh_volume(xyz, faces),
        },
        "surface_resample": {
            "requested_sample_count": int(sample_count),
            "sampled_point_count": int(sampled.shape[0]),
            "poisson_radius_scene_units": float(poisson_radius),
            "oversample": int(args.oversample),
            "seed": int(args.seed),
        },
        "outputs": {
            "original_gaussian_centers_ply": str(out_dir / "original_gaussian_centers.ply"),
            "alpha_surface_mesh_ply": str(out_dir / "alpha_surface_mesh.ply"),
            "poisson_surface_points_ply": str(out_dir / "poisson_surface_points.ply"),
            "overview_png": str(out_dir / "surface_sampling_overview.png"),
        },
    }
    with (out_dir / "geometry_metrics_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
