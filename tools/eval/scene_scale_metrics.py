#!/usr/bin/env python3
"""Estimate scene-to-metric scale and simple plant-level reference volumes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from plyfile import PlyData


def load_xyz(path: Path) -> np.ndarray:
    ply = PlyData.read(path)
    vertex = ply["vertex"]
    names = {prop.name for prop in vertex.properties}
    missing = {"x", "y", "z"} - names
    if missing:
        raise ValueError(f"{path} is missing coordinate fields: {', '.join(sorted(missing))}")
    return np.column_stack([np.asarray(vertex[name], dtype=np.float64) for name in ("x", "y", "z")])


def pca_frame(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centered = xyz - np.mean(xyz, axis=0)
    values, vectors = np.linalg.eigh(np.cov(centered.T))
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]
    projected = centered @ vectors
    return values, vectors, projected


def projection_summary(projected: np.ndarray) -> dict:
    full_span = np.ptp(projected, axis=0)
    robust = {}
    for low, high in ((0.1, 99.9), (0.5, 99.5), (1.0, 99.0), (2.0, 98.0), (5.0, 95.0)):
        key = f"p{low:g}_p{high:g}"
        robust[key] = (np.percentile(projected, high, axis=0) - np.percentile(projected, low, axis=0)).tolist()
    return {
        "span_scene_units": full_span.tolist(),
        "robust_span_scene_units": robust,
    }


def metric_scale_estimates(length_scene: float, width_scene: float, real_length_m: float, real_width_m: float) -> dict:
    length_scale = real_length_m / length_scene
    width_scale = real_width_m / width_scene
    least_squares = (real_length_m * length_scene + real_width_m * width_scene) / (
        length_scene**2 + width_scene**2
    )
    geometric_area = math.sqrt((real_length_m * real_width_m) / (length_scene * width_scene))
    return {
        "length_m_per_scene_unit": float(length_scale),
        "width_m_per_scene_unit": float(width_scale),
        "mean_m_per_scene_unit": float((length_scale + width_scale) / 2.0),
        "least_squares_m_per_scene_unit": float(least_squares),
        "geometric_area_m_per_scene_unit": float(geometric_area),
        "length_width_relative_difference": float(abs(length_scale - width_scale) / ((length_scale + width_scale) / 2.0)),
        "predicted_width_from_length_scale_m": float(width_scene * length_scale),
        "predicted_length_from_width_scale_m": float(length_scene * width_scale),
    }


def robust_metric_scale_estimates(projected: np.ndarray, real_length_m: float, real_width_m: float) -> dict:
    estimates = {}
    for low, high in ((0.1, 99.9), (0.5, 99.5), (1.0, 99.0), (2.0, 98.0), (5.0, 95.0)):
        span = np.percentile(projected, high, axis=0) - np.percentile(projected, low, axis=0)
        key = f"p{low:g}_p{high:g}"
        estimates[key] = metric_scale_estimates(float(span[0]), float(span[1]), real_length_m, real_width_m)
    return estimates


def length_width_scale_range(scales: dict) -> dict:
    values = [float(scales["length_m_per_scene_unit"]), float(scales["width_m_per_scene_unit"])]
    low = min(values)
    high = max(values)
    midpoint = (low + high) / 2.0
    return {
        "min_m_per_scene_unit": low,
        "max_m_per_scene_unit": high,
        "midpoint_m_per_scene_unit": midpoint,
        "relative_width": 0.0 if midpoint == 0.0 else float((high - low) / midpoint),
    }


def scale_range_collection(scales: dict, robust_scales: dict) -> dict:
    ranges = {"full_span_length_width": length_width_scale_range(scales)}
    ranges.update({f"robust_{key}": length_width_scale_range(value) for key, value in robust_scales.items()})
    return ranges


def scaled_length_range(scene_length: float, scale_range: dict) -> dict:
    low = scene_length * scale_range["min_m_per_scene_unit"]
    high = scene_length * scale_range["max_m_per_scene_unit"]
    midpoint = scene_length * scale_range["midpoint_m_per_scene_unit"]
    return {
        "min_m": float(low),
        "max_m": float(high),
        "midpoint_m": float(midpoint),
        "relative_width": scale_range["relative_width"],
    }


def cylinder_range_report(height_scene: float, diameter_m: float, scale_ranges: dict) -> dict:
    area_m2 = math.pi * (diameter_m / 2.0) ** 2
    ranges = {}
    for key, scale_range in scale_ranges.items():
        height = scaled_length_range(height_scene, scale_range)
        ranges[key] = {
            **height,
            "volume_m3_min": float(area_m2 * height["min_m"]),
            "volume_m3_max": float(area_m2 * height["max_m"]),
            "volume_m3_midpoint": float(area_m2 * height["midpoint_m"]),
            "volume_liters_min": float(area_m2 * height["min_m"] * 1000.0),
            "volume_liters_max": float(area_m2 * height["max_m"] * 1000.0),
            "volume_liters_midpoint": float(area_m2 * height["midpoint_m"] * 1000.0),
        }
    return ranges


def select_scale(scales: dict, source: str) -> float:
    key = f"{source}_m_per_scene_unit"
    if key not in scales:
        raise ValueError(f"Unknown scale source {source!r}")
    return float(scales[key])


def cylinder_report(
    cylinder_xyz: np.ndarray,
    diameter_m: float,
    scale_m_per_scene_unit: float,
    scale_source: str,
    scale_ranges: dict | None = None,
) -> dict:
    values, vectors, projected = pca_frame(cylinder_xyz)
    span = np.ptp(projected, axis=0)
    height_scene = float(span[0])
    height_m = height_scene * scale_m_per_scene_unit
    volume_m3 = math.pi * (diameter_m / 2.0) ** 2 * height_m
    report = {
        "gaussian_count": int(cylinder_xyz.shape[0]),
        "pca_eigenvalues": values.tolist(),
        "pca_axes_columns": vectors.tolist(),
        "span_scene_units": span.tolist(),
        "height_scene_units": height_scene,
        "scale_source": scale_source,
        "scale_m_per_scene_unit": float(scale_m_per_scene_unit),
        "diameter_m": float(diameter_m),
        "height_m": float(height_m),
        "volume_m3": float(volume_m3),
        "volume_liters": float(volume_m3 * 1000.0),
    }
    if scale_ranges is not None:
        report["height_volume_ranges"] = cylinder_range_report(height_scene, diameter_m, scale_ranges)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field-ply", required=True, help="PLY containing the measured field crop.")
    parser.add_argument("--real-length-m", type=float, required=True, help="Measured real length of the field crop.")
    parser.add_argument("--real-width-m", type=float, required=True, help="Measured real width of the field crop.")
    parser.add_argument("--scale-source", default="length", choices=["length", "width", "mean", "least_squares", "geometric_area"])
    parser.add_argument("--cylinder-ply", help="Optional PLY selection for a cylinder-like object.")
    parser.add_argument("--cylinder-diameter-m", type=float, help="Real cylinder diameter in meters.")
    parser.add_argument("--out", help="Optional JSON report path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    field_path = Path(args.field_ply)
    field_xyz = load_xyz(field_path)
    values, vectors, projected = pca_frame(field_xyz)
    span = np.ptp(projected, axis=0)
    length_scene = float(span[0])
    width_scene = float(span[1])
    scales = metric_scale_estimates(length_scene, width_scene, args.real_length_m, args.real_width_m)
    robust_scales = robust_metric_scale_estimates(projected, args.real_length_m, args.real_width_m)
    scale_ranges = scale_range_collection(scales, robust_scales)
    selected_scale = select_scale(scales, args.scale_source)

    report = {
        "field_ply": str(field_path.resolve()),
        "field": {
            "gaussian_count": int(field_xyz.shape[0]),
            "xyz_bbox": {
                "min": np.min(field_xyz, axis=0).tolist(),
                "max": np.max(field_xyz, axis=0).tolist(),
                "span_scene_units": np.ptp(field_xyz, axis=0).tolist(),
            },
            "pca_eigenvalues": values.tolist(),
            "pca_axes_columns": vectors.tolist(),
            **projection_summary(projected),
            "interpreted_length_scene_units": length_scene,
            "interpreted_width_scene_units": width_scene,
        },
        "real_measurement_m": {
            "length": float(args.real_length_m),
            "width": float(args.real_width_m),
        },
        "scale_estimates": scales,
        "robust_scale_estimates": robust_scales,
        "scale_ranges": scale_ranges,
        "selected_scale": {
            "source": args.scale_source,
            "m_per_scene_unit": selected_scale,
        },
    }

    if args.cylinder_ply:
        if args.cylinder_diameter_m is None:
            raise ValueError("--cylinder-diameter-m is required with --cylinder-ply")
        cylinder_path = Path(args.cylinder_ply)
        report["cylinder_ply"] = str(cylinder_path.resolve())
        report["cylinder"] = cylinder_report(
            load_xyz(cylinder_path),
            args.cylinder_diameter_m,
            selected_scale,
            args.scale_source,
            scale_ranges,
        )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
