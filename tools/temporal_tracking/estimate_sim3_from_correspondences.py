#!/usr/bin/env python3
"""Estimate COLMAP scene Sim(3) transforms from manual 3D correspondences."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent))

from config import load_config  # noqa: E402


def umeyama(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, float]:
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source and target must both have shape Nx3.")
    if source.shape[0] < 3:
        raise ValueError("At least 3 correspondences are required for Sim(3) alignment.")

    n = source.shape[0]
    mu_src = source.mean(axis=0)
    mu_dst = target.mean(axis=0)
    src_centered = source - mu_src
    dst_centered = target - mu_dst

    cov = (dst_centered.T @ src_centered) / n
    u, singular_values, vt = np.linalg.svd(cov)
    d = np.ones(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        d[-1] = -1
    rotation = u @ np.diag(d) @ vt
    variance = np.mean(np.sum(src_centered**2, axis=1))
    if variance <= 0:
        raise ValueError("Source correspondences are degenerate.")
    scale = float(np.sum(singular_values * d) / variance)
    translation = mu_dst - scale * rotation @ mu_src
    aligned = scale * (source @ rotation.T) + translation
    rmse = float(np.sqrt(np.mean(np.sum((aligned - target) ** 2, axis=1))))
    return scale, rotation, translation, rmse


def load_correspondences(path: Path, reference_scene: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    grouped: dict[str, list[tuple[list[float], list[float]]]] = defaultdict(list)
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"source_scene", "source_x", "source_y", "source_z", "ref_x", "ref_y", "ref_z"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        for row in reader:
            source_scene = row["source_scene"]
            if source_scene == reference_scene:
                continue
            src = [float(row["source_x"]), float(row["source_y"]), float(row["source_z"])]
            dst = [float(row["ref_x"]), float(row["ref_y"]), float(row["ref_z"])]
            grouped[source_scene].append((src, dst))
    return {
        scene: (np.asarray([p[0] for p in pairs], dtype=float), np.asarray([p[1] for p in pairs], dtype=float))
        for scene, pairs in grouped.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Temporal tracking YAML config.")
    parser.add_argument("--correspondences", required=True, help="CSV of manual 3D correspondences.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    corr = load_correspondences(Path(args.correspondences), cfg.reference_scene)
    out_dir = cfg.output_dir / "transforms"
    out_dir.mkdir(parents=True, exist_ok=True)

    for source_scene, (source, target) in corr.items():
        if source_scene not in cfg.scenes:
            raise ValueError(f"Correspondence source_scene '{source_scene}' is not in config.")
        scale, rotation, translation, rmse = umeyama(source, target)
        payload = {
            "source_scene": source_scene,
            "reference_scene": cfg.reference_scene,
            "scale": scale,
            "rotation": rotation.tolist(),
            "translation": translation.tolist(),
            "rmse": rmse,
            "num_correspondences": int(source.shape[0]),
        }
        out_path = out_dir / f"{source_scene}_to_{cfg.reference_scene}.json"
        out_path.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"Wrote {out_path} with RMSE {rmse:.6g} from {source.shape[0]} correspondences")


if __name__ == "__main__":
    main()
