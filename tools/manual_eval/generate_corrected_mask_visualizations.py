#!/usr/bin/env python3
from __future__ import annotations

import argparse
import colorsys
import csv
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def id_to_rgb(label_id: int) -> tuple[int, int, int]:
    if label_id <= 0:
        return (0, 0, 0)
    hue = (label_id * 1.6180339887) % 1
    saturation = 0.5 + (label_id % 2) * 0.5
    red, green, blue = colorsys.hls_to_rgb(hue, 0.5, saturation)
    return (int(red * 255), int(green * 255), int(blue * 255))


def load_index_mask(path: Path) -> np.ndarray:
    mask = np.array(Image.open(path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask


def colorize(mask: np.ndarray) -> np.ndarray:
    color = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for label_id in np.unique(mask):
        color[mask == label_id] = id_to_rgb(int(label_id))
    return color


def build_overlay(rgb_path: Path, color: np.ndarray, active: np.ndarray) -> Image.Image:
    image = Image.open(rgb_path).convert("RGB")
    if image.size != (color.shape[1], color.shape[0]):
        image = image.resize((color.shape[1], color.shape[0]), Image.Resampling.BILINEAR)
    rgb = np.array(image)
    blended = (rgb.astype(np.float32) * 0.55 + color.astype(np.float32) * 0.45).astype(np.uint8)
    overlay = rgb.copy()
    overlay[active] = blended[active]
    return Image.fromarray(overlay)


def make_contact_sheet(
    image_paths: list[Path],
    output_path: Path,
    title: str,
    columns: int = 4,
    tile_size: tuple[int, int] = (384, 216),
) -> None:
    font = ImageFont.load_default()
    label_height = 24
    title_height = 42
    gap = 10
    rows = math.ceil(len(image_paths) / columns)
    width = columns * tile_size[0] + (columns + 1) * gap
    height = title_height + rows * (tile_size[1] + label_height + gap) + gap
    sheet = Image.new("RGB", (width, height), (28, 28, 28))
    draw = ImageDraw.Draw(sheet)
    draw.text((gap, 13), title, fill=(255, 255, 255), font=font)

    for index, path in enumerate(image_paths):
        row, column = divmod(index, columns)
        x = gap + column * (tile_size[0] + gap)
        y = title_height + gap + row * (tile_size[1] + label_height + gap)
        image = Image.open(path).convert("RGB")
        image.thumbnail(tile_size, Image.Resampling.LANCZOS)
        tile = Image.new("RGB", tile_size, (0, 0, 0))
        tile.paste(image, ((tile_size[0] - image.width) // 2, (tile_size[1] - image.height) // 2))
        sheet.paste(tile, (x, y))
        draw.text((x + 4, y + tile_size[1] + 5), path.stem, fill=(240, 240, 240), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=94, subsampling=0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create color instance masks, RGB overlays, and contact sheets for corrected masks."
    )
    parser.add_argument(
        "--subset_dir",
        type=Path,
        default=Path("data/vinyes_20260509_pinhole/manual_eval_gt/target_two_vines_all31"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/target_two_vines_all31_corrected_visualizations"),
    )
    args = parser.parse_args()

    mask_dir = args.subset_dir / "corrected_masks"
    rgb_dir = args.subset_dir / "rgb"
    mask_paths = sorted(mask_dir.glob("*.png"))
    if not mask_paths:
        raise FileNotFoundError(f"No corrected masks found in {mask_dir}")

    color_dir = args.output_dir / "color_instance_masks"
    overlay_dir = args.output_dir / "overlays"
    contact_dir = args.output_dir / "contact_sheets"
    for directory in (color_dir, overlay_dir, contact_dir):
        directory.mkdir(parents=True, exist_ok=True)

    label_ids: set[int] = set()
    color_paths: list[Path] = []
    overlay_paths: list[Path] = []
    for mask_path in mask_paths:
        rgb_path = rgb_dir / mask_path.name
        if not rgb_path.exists():
            raise FileNotFoundError(rgb_path)

        mask = load_index_mask(mask_path)
        label_ids.update(int(value) for value in np.unique(mask) if int(value) > 0)
        color = colorize(mask)

        color_path = color_dir / mask_path.name
        Image.fromarray(color).save(color_path)
        color_paths.append(color_path)

        overlay_path = overlay_dir / f"{mask_path.stem}.jpg"
        build_overlay(rgb_path, color, mask > 0).save(overlay_path, quality=95, subsampling=0)
        overlay_paths.append(overlay_path)

    make_contact_sheet(
        overlay_paths,
        contact_dir / "corrected_overlays_all31.jpg",
        "Corrected instance masks: RGB overlays (31 views)",
    )
    make_contact_sheet(
        color_paths,
        contact_dir / "corrected_color_instance_masks_all31.jpg",
        "Corrected color instance masks (31 views)",
    )

    with (args.output_dir / "color_legend.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["label_id", "red", "green", "blue", "hex"])
        writer.writeheader()
        for label_id in sorted(label_ids):
            red, green, blue = id_to_rgb(label_id)
            writer.writerow(
                {
                    "label_id": label_id,
                    "red": red,
                    "green": green,
                    "blue": blue,
                    "hex": f"#{red:02x}{green:02x}{blue:02x}",
                }
            )

    print(f"Wrote {len(color_paths)} color instance masks to {color_dir}")
    print(f"Wrote {len(overlay_paths)} overlays to {overlay_dir}")
    print(f"Wrote contact sheets to {contact_dir}")


if __name__ == "__main__":
    main()
