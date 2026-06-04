#!/usr/bin/env python3
"""List available selected-frame candidates for each configured scene."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent))

from config import iter_image_files, load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Temporal tracking YAML config.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = cfg.output_dir / "frame_lists"
    out_dir.mkdir(parents=True, exist_ok=True)

    for scene in cfg.scenes.values():
        rows = []
        for image_path in iter_image_files(scene.image_dir):
            rows.append(
                {
                    "scene": scene.name,
                    "image_name": image_path.name,
                    "image_path": str(image_path),
                }
            )
        out_path = out_dir / f"{scene.name}_frames.csv"
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["scene", "image_name", "image_path"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {len(rows)} frames: {out_path}")


if __name__ == "__main__":
    main()
