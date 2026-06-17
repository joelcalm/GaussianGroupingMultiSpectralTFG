#!/usr/bin/env python3
"""Sweep density thresholds and plot Gaussian Splatting volume estimates."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gaussian_volume_metrics import (
    bounding_box,
    density_grid,
    evaluate_density,
    filter_scene,
    grid_shape,
    load_gaussian_ply,
    mesh_volume,
    resolve_threshold,
    voxel_size_from_resolution,
)


def center_density_grid(scene, bbox_min, bbox_max, voxel_size, density_mode, point_chunk, gaussian_chunk):
    shape = grid_shape(bbox_min, bbox_max, voxel_size)
    axes = [bbox_min[dim] + (np.arange(shape[dim], dtype=np.float64) + 0.5) * voxel_size for dim in range(3)]
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
    return grid


def monte_carlo_densities(scene, bbox_min, bbox_max, sample_count, seed, density_mode, point_chunk, gaussian_chunk):
    rng = np.random.default_rng(seed)
    span = bbox_max - bbox_min
    densities = np.empty(sample_count, dtype=np.float32)
    start = 0
    while start < sample_count:
        stop = min(start + point_chunk, sample_count)
        points = bbox_min + rng.random((stop - start, 3)) * span
        densities[start:stop] = evaluate_density(points, scene, density_mode, point_chunk, gaussian_chunk)
        start = stop
    return densities


def thresholds_from_scene(
    scene,
    density_mode,
    point_chunk,
    gaussian_chunk,
    threshold_count,
    threshold_min,
    threshold_max,
):
    center_density = evaluate_density(scene.xyz, scene, density_mode, point_chunk, gaussian_chunk)
    positive = center_density[center_density > 0]
    if positive.size == 0:
        raise ValueError("All center densities are zero; cannot build a threshold sweep")
    lo = float(threshold_min) if threshold_min is not None else float(np.percentile(positive, 1))
    hi = float(threshold_max) if threshold_max is not None else float(np.percentile(positive, 99))
    if lo <= 0 or hi <= 0 or lo >= hi:
        raise ValueError(f"Invalid threshold range: min={lo}, max={hi}")
    thresholds = np.geomspace(lo, hi, threshold_count)
    return thresholds, {
        "center_density_min": float(np.min(center_density)),
        "center_density_p01": float(np.percentile(positive, 1)),
        "center_density_p10": float(np.percentile(positive, 10)),
        "center_density_p50": float(np.percentile(positive, 50)),
        "center_density_p90": float(np.percentile(positive, 90)),
        "center_density_p99": float(np.percentile(positive, 99)),
        "center_density_max": float(np.max(center_density)),
    }


def voxel_sweep(center_grid, thresholds, voxel_size):
    densities = center_grid.ravel()
    voxel_volume = voxel_size**3
    return np.asarray([np.count_nonzero(densities >= threshold) * voxel_volume for threshold in thresholds])


def monte_carlo_sweep(densities, thresholds, bbox_volume):
    return np.asarray([bbox_volume * (np.count_nonzero(densities >= threshold) / densities.size) for threshold in thresholds])


def marching_cubes_sweep(node_grid, thresholds, bbox_min, voxel_size):
    from skimage import measure

    grid_min = float(np.min(node_grid))
    grid_max = float(np.max(node_grid))
    volumes = np.full(thresholds.shape, np.nan, dtype=np.float64)
    mesh_sizes = []
    for idx, threshold in enumerate(thresholds):
        if threshold < grid_min or threshold > grid_max:
            mesh_sizes.append({"vertices": 0, "faces": 0})
            continue
        vertices, faces, _normals, _values = measure.marching_cubes(
            node_grid,
            level=float(threshold),
            spacing=(voxel_size, voxel_size, voxel_size),
        )
        vertices = vertices + bbox_min[None, :]
        volumes[idx] = mesh_volume(vertices, faces)
        mesh_sizes.append({"vertices": int(vertices.shape[0]), "faces": int(faces.shape[0])})
    return volumes, mesh_sizes


def knee_threshold(thresholds, volumes):
    valid = np.isfinite(volumes)
    x = np.log10(thresholds[valid])
    y = volumes[valid]
    if x.size < 3 or np.max(y) <= np.min(y):
        return None
    x_norm = (x - x.min()) / (x.max() - x.min())
    y_norm = (y - y.min()) / (y.max() - y.min())
    start = np.array([x_norm[0], y_norm[0]])
    end = np.array([x_norm[-1], y_norm[-1]])
    line = end - start
    line_norm = np.linalg.norm(line)
    if line_norm == 0:
        return None
    points = np.column_stack([x_norm, y_norm])
    distances = np.abs(np.cross(line, points - start) / line_norm)
    idx = int(np.argmax(distances))
    return {
        "threshold": float(thresholds[valid][idx]),
        "volume_scene_units_cubed": float(y[idx]),
        "index": idx,
        "distance_to_chord": float(distances[idx]),
    }


def write_csv(path, thresholds, voxel, monte_carlo, marching_cubes):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["threshold", "voxel_volume", "monte_carlo_volume", "marching_cubes_volume"])
        for row in zip(thresholds, voxel, monte_carlo, marching_cubes):
            writer.writerow([float(value) if np.isfinite(value) else "" for value in row])


def plot_sweep(path, thresholds, voxel, monte_carlo, marching_cubes, auto_threshold, knees):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    series = [
        ("voxel", voxel, "#1f77b4"),
        ("monte carlo", monte_carlo, "#2ca02c"),
        ("marching cubes", marching_cubes, "#d62728"),
    ]
    for label, values, color in series:
        axes[0].plot(thresholds, values, marker="o", markersize=3, linewidth=1.6, label=label, color=color)
        vmax = np.nanmax(values)
        if vmax > 0:
            axes[1].plot(thresholds, values / vmax, marker="o", markersize=3, linewidth=1.6, label=label, color=color)

    for ax in axes:
        ax.set_xscale("log")
        ax.grid(True, which="both", alpha=0.28)
        ax.axvline(auto_threshold, color="#111111", linestyle="--", linewidth=1.2, label="auto p10")
        for method, knee in knees.items():
            if knee is not None:
                ax.axvline(knee["threshold"], color="#888888", linestyle=":", linewidth=0.9)
        ax.set_xlabel("density threshold tau")
    axes[0].set_ylabel("volume (scene units^3)")
    axes[1].set_ylabel("normalized volume")
    axes[0].set_title("Absolute volume")
    axes[1].set_title("Shape of the threshold response")
    axes[0].legend(loc="best")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ply", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--density-mode", choices=["amplitude", "normalized"], default="amplitude")
    parser.add_argument("--threshold-count", type=int, default=50)
    parser.add_argument("--threshold-min", type=float)
    parser.add_argument("--threshold-max", type=float)
    parser.add_argument("--auto-threshold-percentile", type=float, default=10.0)
    parser.add_argument("--grid-resolution", type=int, default=72)
    parser.add_argument("--voxel-size", type=float)
    parser.add_argument("--bbox-sigma", type=float, default=3.0)
    parser.add_argument("--mc-samples", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--point-chunk", type=int, default=8192)
    parser.add_argument("--gaussian-chunk", type=int, default=2048)
    parser.add_argument("--min-opacity", type=float, default=0.0)
    parser.add_argument("--object-argmax", type=int)
    parser.add_argument("--object-channel", type=int)
    parser.add_argument("--object-min", type=float, default=0.0)
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    scene_all = load_gaussian_ply(Path(args.ply))
    scene = filter_scene(scene_all, args.min_opacity, args.object_argmax, args.object_channel, args.object_min)
    bbox_min, bbox_max = bounding_box(scene, args.bbox_sigma)
    voxel_size = args.voxel_size or voxel_size_from_resolution(bbox_min, bbox_max, args.grid_resolution)
    bbox_volume = float(np.prod(bbox_max - bbox_min))

    thresholds, density_summary = thresholds_from_scene(
        scene,
        args.density_mode,
        args.point_chunk,
        args.gaussian_chunk,
        args.threshold_count,
        args.threshold_min,
        args.threshold_max,
    )
    auto_threshold, auto_report = resolve_threshold(
        scene,
        "auto",
        args.auto_threshold_percentile,
        args.density_mode,
        args.point_chunk,
        args.gaussian_chunk,
    )

    center_grid = center_density_grid(
        scene, bbox_min, bbox_max, voxel_size, args.density_mode, args.point_chunk, args.gaussian_chunk
    )
    mc_density = monte_carlo_densities(
        scene, bbox_min, bbox_max, args.mc_samples, args.seed, args.density_mode, args.point_chunk, args.gaussian_chunk
    )
    node_grid, _axes = density_grid(
        scene, bbox_min, bbox_max, voxel_size, args.density_mode, args.point_chunk, args.gaussian_chunk
    )

    voxel = voxel_sweep(center_grid, thresholds, voxel_size)
    monte_carlo = monte_carlo_sweep(mc_density, thresholds, bbox_volume)
    marching_cubes, mesh_sizes = marching_cubes_sweep(node_grid, thresholds, bbox_min, voxel_size)
    knees = {
        "voxel": knee_threshold(thresholds, voxel),
        "monte_carlo": knee_threshold(thresholds, monte_carlo),
        "marching_cubes": knee_threshold(thresholds, marching_cubes),
    }

    csv_path = out_dir / "threshold_sweep.csv"
    png_path = out_dir / "threshold_sweep_volume.png"
    json_path = out_dir / "threshold_sweep_report.json"
    write_csv(csv_path, thresholds, voxel, monte_carlo, marching_cubes)
    plot_sweep(png_path, thresholds, voxel, monte_carlo, marching_cubes, auto_threshold, knees)

    report = {
        "input_ply": str(Path(args.ply).resolve()),
        "density_mode": args.density_mode,
        "threshold_generation": density_summary,
        "auto_threshold": auto_report,
        "filter": {
            "input_gaussian_count": int(scene_all.xyz.shape[0]),
            "kept_gaussian_count": int(scene.xyz.shape[0]),
            "min_opacity": float(args.min_opacity),
            "object_argmax": args.object_argmax,
            "object_channel": args.object_channel,
            "object_min": float(args.object_min),
        },
        "bbox": {
            "min": bbox_min.astype(float).tolist(),
            "max": bbox_max.astype(float).tolist(),
            "volume_scene_units_cubed": bbox_volume,
        },
        "grid": {
            "voxel_size_scene_units": float(voxel_size),
            "center_grid_shape": list(center_grid.shape),
            "node_grid_shape": list(node_grid.shape),
        },
        "monte_carlo": {
            "sample_count": int(args.mc_samples),
            "seed": int(args.seed),
        },
        "knees": knees,
        "marching_cubes_mesh_sizes": mesh_sizes,
        "outputs": {
            "csv": str(csv_path),
            "plot_png": str(png_path),
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
