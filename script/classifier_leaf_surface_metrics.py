#!/usr/bin/env python3
"""Leaf surface metrics and class-distribution plots using the saved classifier."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from gaussian_volume_metrics import (
    GaussianScene,
    analytical_leaf_area,
    attach_metric_value,
    bounding_box,
    load_gaussian_ply,
    load_metric_scale_report,
    metric_scale_context,
    pca_basis,
    poisson_leaf_area,
    summarize_input,
)


@dataclass
class VineInput:
    name: str
    ply: Path


def load_classifier(path: Path) -> tuple[np.ndarray, np.ndarray]:
    state = torch.load(path, map_location="cpu")
    if hasattr(state, "state_dict"):
        state = state.state_dict()
    weights = state["weight"].squeeze(-1).squeeze(-1).numpy()
    bias_tensor = state.get("bias")
    bias = np.zeros(weights.shape[0], dtype=np.float32) if bias_tensor is None else bias_tensor.numpy()
    return weights.astype(np.float32), bias.astype(np.float32)


def classifier_predictions(scene: GaussianScene, weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
    if scene.object_features is None:
        raise ValueError("PLY does not contain obj_dc_* object features")
    if scene.object_features.shape[1] != weights.shape[1]:
        raise ValueError(
            f"Classifier expects {weights.shape[1]} object features, got {scene.object_features.shape[1]}"
        )
    logits = scene.object_features.astype(np.float32) @ weights.T + bias
    return np.argmax(logits, axis=1).astype(np.int64)


def filter_by_mask(scene: GaussianScene, mask: np.ndarray) -> GaussianScene:
    if not np.any(mask):
        raise ValueError("Classifier filter removed all Gaussians")
    return GaussianScene(
        xyz=scene.xyz[mask],
        opacity=scene.opacity[mask],
        scales=scene.scales[mask],
        rotations=scene.rotations[mask],
        object_features=None if scene.object_features is None else scene.object_features[mask],
    )


def metadata_by_label(path: Path) -> dict[int, dict]:
    raw = json.load(path.open("r", encoding="utf-8"))
    return {int(key): value for key, value in raw.items()}


def class_name_counts(predicted_labels: np.ndarray, label_metadata: dict[int, dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for label in predicted_labels:
        class_name = label_metadata.get(int(label), {}).get("class_name", f"label_{int(label)}")
        counts[class_name] = counts.get(class_name, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def flat_label_counts(predicted_labels: np.ndarray, label_metadata: dict[int, dict]) -> list[dict]:
    rows = []
    labels, counts = np.unique(predicted_labels, return_counts=True)
    for label, count in sorted(zip(labels, counts), key=lambda item: (-item[1], item[0])):
        info = label_metadata.get(int(label), {})
        rows.append(
            {
                "label_id": int(label),
                "count": int(count),
                "label": info.get("label", f"label_{int(label)}"),
                "class_name": info.get("class_name", "unknown"),
                "object_type": info.get("object_type", "unknown"),
                "part_id": info.get("part_id", "unknown"),
            }
        )
    return rows


def class_color_lookup(class_map_path: Path, class_colors_path: Path) -> dict[str, tuple[float, float, float]]:
    class_map = json.load(class_map_path.open("r", encoding="utf-8"))
    class_colors = json.load(class_colors_path.open("r", encoding="utf-8"))
    colors = {}
    for class_name, class_id in class_map.items():
        rgb = class_colors.get(str(class_id), [120, 120, 120])
        colors[class_name] = tuple(float(channel) / 255.0 for channel in rgb)
    return colors


def write_report(
    vine: VineInput,
    scene_all: GaussianScene,
    leaf_scene: GaussianScene,
    predicted_labels: np.ndarray,
    leaf_mask: np.ndarray,
    label_metadata: dict[int, dict],
    scale_context: dict | None,
    out_path: Path,
    poisson_mesh_path: Path,
    raster_resolution: int,
    ellipse_sigma: float,
    occupancy_threshold: float,
    poisson_depth: int,
    poisson_density_quantile: float,
) -> dict:
    bbox_min, bbox_max = bounding_box(leaf_scene, 3.0)
    report = {
        "input_ply": str(vine.ply.resolve()),
        "classifier_filter": {
            "target_class_name": "vine_leaf",
            "input_gaussian_count": int(scene_all.xyz.shape[0]),
            "kept_gaussian_count": int(leaf_scene.xyz.shape[0]),
            "classifier_leaf_fraction": float(np.mean(leaf_mask)),
            "class_name_counts_all_gaussians": class_name_counts(predicted_labels, label_metadata),
            "flat_label_counts_all_gaussians": flat_label_counts(predicted_labels, label_metadata),
        },
        "input_summary": summarize_input(leaf_scene, bbox_min, bbox_max),
        "methods": {},
        "metric_scale": scale_context,
    }

    analytical = analytical_leaf_area(leaf_scene, raster_resolution, ellipse_sigma, occupancy_threshold)
    attach_metric_value(analytical, "alpha_composited_area_scene_units_squared", 2, "alpha_composited_area_m2", scale_context)
    attach_metric_value(
        analytical,
        "thresholded_occupancy_area_scene_units_squared",
        2,
        "thresholded_occupancy_area_m2",
        scale_context,
    )
    report["methods"]["analytical_leaf_area"] = analytical

    poisson = poisson_leaf_area(leaf_scene, poisson_depth, poisson_density_quantile, poisson_mesh_path)
    attach_metric_value(poisson, "one_sided_area_scene_units_squared", 2, "one_sided_area_m2", scale_context)
    report["methods"]["poisson_leaf_area"] = poisson

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report


def plot_classifier_splats(
    vines: list[VineInput],
    scenes: list[GaussianScene],
    predicted: list[np.ndarray],
    label_metadata: dict[int, dict],
    colors: dict[str, tuple[float, float, float]],
    out_png: Path,
    out_pdf: Path | None,
) -> None:
    fig, axes = plt.subplots(1, len(vines), figsize=(7.2 * len(vines), 6.2), constrained_layout=True)
    if len(vines) == 1:
        axes = [axes]
    fig.suptitle("Classifier-predicted Gaussian classes in selected vines", fontsize=16, fontweight="bold")

    legend_classes = set()
    for ax, vine, scene, labels in zip(axes, vines, scenes, predicted):
        _values, _basis, projected = pca_basis(scene.xyz)
        xy = projected[:, :2]
        class_names = np.array([label_metadata.get(int(label), {}).get("class_name", "unknown") for label in labels])
        sizes = np.clip(np.max(scene.scales, axis=1) * 900.0, 9.0, 95.0)
        draw_order = np.argsort(sizes)[::-1]
        for idx in draw_order:
            class_name = str(class_names[idx])
            legend_classes.add(class_name)
            ax.scatter(
                xy[idx, 0],
                xy[idx, 1],
                s=sizes[idx],
                c=[colors.get(class_name, (0.55, 0.55, 0.55))],
                alpha=float(np.clip(scene.opacity[idx], 0.25, 0.9)),
                edgecolors="none",
            )
        counts = class_name_counts(labels, label_metadata)
        top_counts = ", ".join(f"{name}: {count}" for name, count in list(counts.items())[:5])
        ax.set_title(f"{vine.name}\n{scene.xyz.shape[0]} Gaussians; {top_counts}", fontsize=10)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    handles = []
    for class_name in sorted(legend_classes):
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor=colors.get(class_name, (0.55, 0.55, 0.55)),
                markeredgecolor="none",
                markersize=8,
                label=class_name,
            )
        )
    fig.legend(handles=handles, loc="lower center", ncol=min(5, max(1, len(handles))), frameon=False)
    fig.text(
        0.5,
        0.045,
        "2D PCA projection of Gaussian centers. Marker area follows maximum Gaussian scale; opacity follows learned Gaussian opacity.",
        ha="center",
        fontsize=9,
        color="#333333",
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    if out_pdf is not None:
        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def write_summary(rows: list[dict], out_csv: Path, out_json: Path, scale_context: dict | None) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with out_json.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "filter": "saved classifier predicted class_name == vine_leaf",
                "metric_scale": scale_context,
                "rows": rows,
            },
            handle,
            indent=2,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classifier-pth", required=True)
    parser.add_argument("--label-map", required=True)
    parser.add_argument("--class-map", required=True)
    parser.add_argument("--class-colors", required=True)
    parser.add_argument("--metric-scale-report", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--surface-raster-resolution", type=int, default=768)
    parser.add_argument("--surface-ellipse-sigma", type=float, default=1.0)
    parser.add_argument("--surface-occupancy-threshold", type=float, default=0.5)
    parser.add_argument("--poisson-depth", type=int, default=8)
    parser.add_argument("--poisson-density-quantile", type=float, default=0.01)
    parser.add_argument("--vine", nargs=2, action="append", metavar=("NAME", "PLY"), required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vines = [VineInput(name, Path(ply)) for name, ply in args.vine]
    out_dir = Path(args.out_dir)
    weights, bias = load_classifier(Path(args.classifier_pth))
    label_metadata = metadata_by_label(Path(args.label_map))
    colors = class_color_lookup(Path(args.class_map), Path(args.class_colors))
    scale_context = metric_scale_context(load_metric_scale_report(Path(args.metric_scale_report)))

    scenes = []
    predictions = []
    summary_rows = []
    for vine in vines:
        scene_all = load_gaussian_ply(vine.ply)
        predicted = classifier_predictions(scene_all, weights, bias)
        leaf_mask = np.array(
            [label_metadata.get(int(label), {}).get("class_name") == "vine_leaf" for label in predicted],
            dtype=bool,
        )
        leaf_scene = filter_by_mask(scene_all, leaf_mask)
        report = write_report(
            vine=vine,
            scene_all=scene_all,
            leaf_scene=leaf_scene,
            predicted_labels=predicted,
            leaf_mask=leaf_mask,
            label_metadata=label_metadata,
            scale_context=scale_context,
            out_path=out_dir / f"{vine.name}_classifier_leaf_surface_report.json",
            poisson_mesh_path=out_dir / f"{vine.name}_classifier_leaf_poisson_mesh.ply",
            raster_resolution=args.surface_raster_resolution,
            ellipse_sigma=args.surface_ellipse_sigma,
            occupancy_threshold=args.surface_occupancy_threshold,
            poisson_depth=args.poisson_depth,
            poisson_density_quantile=args.poisson_density_quantile,
        )
        analytical = report["methods"]["analytical_leaf_area"]
        poisson = report["methods"]["poisson_leaf_area"]
        summary_rows.append(
            {
                "vine": vine.name,
                "input_gaussians": int(scene_all.xyz.shape[0]),
                "classifier_leaf_gaussians": int(leaf_scene.xyz.shape[0]),
                "classifier_leaf_fraction": float(np.mean(leaf_mask)),
                "analytical_alpha_composited_area_m2": analytical.get("alpha_composited_area_m2"),
                "analytical_thresholded_occupancy_area_m2": analytical.get("thresholded_occupancy_area_m2"),
                "poisson_one_sided_area_m2": poisson.get("one_sided_area_m2"),
                "poisson_one_sided_area_scene_units2": poisson.get("one_sided_area_scene_units_squared"),
                "poisson_mesh_vertices": poisson.get("mesh_vertex_count"),
                "poisson_mesh_faces": poisson.get("mesh_face_count"),
            }
        )
        scenes.append(scene_all)
        predictions.append(predicted)

    write_summary(
        summary_rows,
        out_dir / "leaf_surface_classifier_summary.csv",
        out_dir / "leaf_surface_classifier_summary.json",
        scale_context,
    )
    plot_classifier_splats(
        vines,
        scenes,
        predictions,
        label_metadata,
        colors,
        out_dir / "classifier_gaussian_class_splats_2d.png",
        out_dir / "classifier_gaussian_class_splats_2d.pdf",
    )


if __name__ == "__main__":
    main()
