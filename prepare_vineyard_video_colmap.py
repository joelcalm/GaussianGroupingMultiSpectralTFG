#!/usr/bin/env python3
"""Prepare a shared COLMAP scene from raw vineyard RGB/multispectral videos.

The default workflow is designed for VINIA/2026/20260418-Augustus++:
- extract N frames from the main RGB video and band videos, with an optional
  higher RGB-only count or extra dense RGB ranges around the turn/end-of-row;
- stage them into one image folder with stable names (rgb_00001, b470_00001, ...);
- build RGB-only COLMAP variants first (exhaustive/sequential-loop matching,
  OPENCV/FOV camera models, stronger SIFT);
- optionally register multispectral band frames against the selected RGB model;
- write a training-compatible scene under output_dir with images/, images_rgb/,
  sparse/0/, band_info.json, frame_info.json, and COLMAP diagnostics.

The direct all-band reconstruction path is still available with
--registration_mode direct for comparison, but the robust path treats this as a
wrong-geometry RGB SfM problem rather than a multiple-model problem.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sqlite3
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_VIDEO_DIR = Path("/home/msiau/data/datasets/VINIA/2026/20260418-Augustus++")
DEFAULT_OUTPUT_DIR = Path("data/vinyes_fulles_1")
DEFAULT_COLMAP_CANDIDATES = [
    Path("/home/msiau/workspace/.conda/envs/colmap_gpu/bin/colmap"),
    Path("/home/msiau/workspace/.conda/envs/gaussian_grouping/bin/colmap"),
    Path("/home/msiau/.conda/envs/colmap-env/bin/colmap"),
]

BAND_CHANNELS = {
    "rgb": [0, 1, 2],
    "b470": [3],
    "b505": [4],
    "b525": [5],
    "b590": [6],
    "b635": [7],
    "b660": [8],
    "b850": [9],
}

VIDEO_SPECS = [
    {"band": "rgb", "contains": ["RGB_DJI"], "exclude": ["RGB_UP", "RGB_DOWN", "RGB_SOIL"]},
    {"band": "b470", "contains": ["_470_"], "exclude": []},
    {"band": "b505", "contains": ["_505_"], "exclude": []},
    {"band": "b525", "contains": ["_525_"], "exclude": []},
    {"band": "b590", "contains": ["_590_"], "exclude": []},
    {"band": "b635", "contains": ["_635_"], "exclude": []},
    {"band": "b660", "contains": ["_660_"], "exclude": []},
    {"band": "b850", "contains": ["_850_"], "exclude": []},
]

IMAGE_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".MP4", ".MOV", ".AVI", ".MKV"}
STAGES = ["extract", "features", "match", "map", "register", "finalize", "diagnose"]
COLMAP_PAIR_ID_MULTIPLIER = 2147483647


@dataclass(frozen=True)
class VideoItem:
    band: str
    path: Path
    segment_index: int = 0
    start_output_index: int = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video_dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--frames_per_video", type=int, default=200)
    parser.add_argument("--rgb_frames_per_video", type=int, default=None, help="Optional RGB-only extraction count for COLMAP. Use 400-800 here while keeping band extraction at --frames_per_video.")
    parser.add_argument("--rgb_dense_ranges", default="", help="Comma-separated RGB frame ranges to oversample, as start:end[:count]. Values <=1 are fractions of the video, values >1 are absolute frame indices.")
    parser.add_argument("--rgb_dense_range_frames", type=int, default=120, help="Extra frames per --rgb_dense_ranges interval when the interval omits an explicit count.")
    parser.add_argument("--image_ext", choices=["png", "jpg"], default="png")
    parser.add_argument("--jpeg_quality", type=int, default=95)
    parser.add_argument("--stage", choices=STAGES + ["all"], default="all")
    parser.add_argument("--bands", default=None, help="Comma-separated bands to include. Defaults to all known bands.")
    parser.add_argument("--exclude_bands", default="", help="Comma-separated bands to exclude before extraction/reconstruction.")
    parser.add_argument("--rgb_contains", default=None, help="Optional comma-separated filename tokens for selecting RGB videos, e.g. Wide_RGB_DJI.")
    parser.add_argument("--registration_mode", choices=["direct", "rgb_register", "rgb_only"], default="rgb_register")
    parser.add_argument("--rgb_variants", default="exhaustive:OPENCV,exhaustive:FOV,sequential_loop:OPENCV,sequential_loop:FOV", help="Comma-separated matcher:camera_model RGB COLMAP variants to run. Use an empty string to run --matcher_type/--camera_model only.")
    parser.add_argument("--selected_rgb_variant", default="auto", help="Variant folder name to finalize/register from, or auto to choose by diagnostics.")
    parser.add_argument("--matcher_type", choices=["exhaustive", "sequential_loop"], default="exhaustive")
    parser.add_argument("--sequential_overlap", type=int, default=50, help="Sequential matcher overlap for sequential_loop variants.")
    parser.add_argument("--turn_frame_range", default="", help="Optional weak-registration diagnostic frame range start:end using RGB output frame numbers or fractions.")
    parser.add_argument("--turn_weak_window", type=int, default=20, help="Number of RGB frames around the detected/declared turn to list in weak diagnostics.")
    parser.add_argument("--bad_trajectory_line_angle", type=float, default=35.0, help="Flag folded/V-shaped RGB trajectories when fitted vineyard-pass line angle exceeds this many degrees.")
    add_bool_arg(parser, "--allow_bad_colmap", default=False, help="Allow finalization even when diagnostics mark the selected RGB trajectory as bad.")
    add_bool_arg(parser, "--fail_on_folded_trajectory", default=True, help="Mark sharp V-shaped two-pass RGB trajectories as bad.")
    parser.add_argument("--colmap_bin", type=Path, default=None)
    parser.add_argument("--gpu_index", default="0")
    parser.add_argument("--camera_model", default="PINHOLE")
    parser.add_argument("--default_focal_length_factor", type=float, default=1.2)
    parser.add_argument("--max_image_size", type=int, default=3200)
    parser.add_argument("--max_num_features", type=int, default=12000)
    parser.add_argument("--sift_num_threads", type=int, default=8)
    parser.add_argument("--estimate_affine_shape", type=int, default=1)
    parser.add_argument("--domain_size_pooling", type=int, default=1)
    parser.add_argument("--sequential_window", type=int, default=8, help="RGB/RGB and same-band neighboring frame match window.")
    parser.add_argument("--cross_band_window", type=int, default=12, help="Match band frame k to RGB frames k +/- this value.")
    parser.add_argument("--direct_intra_band_radius", type=int, default=8, help="Same-band temporal pair radius for direct joint reconstruction.")
    parser.add_argument("--direct_cross_band_radius", type=int, default=3, help="Band frame k to RGB frames k +/- radius for direct joint reconstruction.")
    parser.add_argument("--long_stride", type=int, default=25, help="Extra RGB loop pairs every N frames. 0 disables.")
    parser.add_argument("--min_num_matches", type=int, default=30)
    parser.add_argument("--min_num_inliers", type=int, default=10, help="Geometric verification inliers for match import/mapping.")
    parser.add_argument("--abs_pose_min_num_inliers", type=int, default=50, help="Minimum 2D-3D inliers for registering a new image.")
    parser.add_argument("--abs_pose_max_error", type=float, default=4.0, help="Maximum absolute pose reprojection error for registering a new image.")
    parser.add_argument("--max_num_matches", type=int, default=32768)
    parser.add_argument("--matching_num_threads", type=int, default=16, help="COLMAP SiftMatching.num_threads cap for memory-sensitive matching.")
    parser.add_argument("--guided_matching", type=int, default=1)
    parser.add_argument("--matching_use_gpu", type=int, default=0, help="Use GPU for COLMAP matching. Defaults to 0 because headless GPU matching can require an OpenGL context.")
    parser.add_argument("--mapper_min_model_size", type=int, default=30)
    add_bool_arg(parser, "--skip_existing", default=True)
    add_bool_arg(
        parser,
        "--copy_images",
        default=False,
        help="Copy staged images instead of symlinking them.",
    )
    add_bool_arg(
        parser,
        "--grayscale_colmap_bands",
        default=False,
        help="Write non-RGB staged images as single-channel grayscale for COLMAP/training while keeping frames_raw unchanged.",
    )
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def add_bool_arg(
    parser: argparse.ArgumentParser,
    name: str,
    *,
    default: bool,
    help: str | None = None,
) -> None:
    try:
        action = argparse.BooleanOptionalAction
    except AttributeError:
        action = None
    if action is not None:
        parser.add_argument(name, action=action, default=default, help=help)
        return
    dest = name.lstrip("-").replace("-", "_")
    parser.add_argument(name, dest=dest, action="store_true", default=default, help=help)
    parser.add_argument(f"--no-{dest}", dest=dest, action="store_false", help=argparse.SUPPRESS)


def stage_enabled(args: argparse.Namespace, stage: str) -> bool:
    return args.stage == "all" or args.stage == stage


def ordered_known_bands() -> list[str]:
    return [spec["band"] for spec in VIDEO_SPECS]


def parse_band_list(value: str | None) -> list[str] | None:
    if value is None or not value.strip():
        return None
    return [part.strip().lower() for part in value.split(",") if part.strip()]


def parse_token_list(value: str | None) -> list[str] | None:
    if value is None or not value.strip():
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def resolve_active_bands(args: argparse.Namespace) -> list[str]:
    known = ordered_known_bands()
    requested = parse_band_list(args.bands)
    excluded = set(parse_band_list(args.exclude_bands) or [])
    unknown = set(requested or []) | excluded
    unknown -= set(known)
    if unknown:
        raise ValueError(f"Unknown band(s): {', '.join(sorted(unknown))}. Known bands: {', '.join(known)}")
    bands = requested if requested is not None else known
    bands = [band for band in bands if band not in excluded]
    if "rgb" not in bands:
        bands.insert(0, "rgb")
    if not bands:
        raise ValueError("Need at least one band. Use --bands rgb for an RGB-only diagnostic reconstruction.")
    return [band for band in known if band in bands]


def make_active_band_channels(active_bands: list[str]) -> dict[str, list[int]]:
    channels: dict[str, list[int]] = {"rgb": [0, 1, 2]}
    next_channel = 3
    for band in active_bands:
        if band == "rgb":
            continue
        channels[band] = [next_channel]
        next_channel += 1
    return channels


def run(cmd: list[str], *, dry_run: bool = False, cwd: Path | None = None) -> None:
    print("+ " + " ".join(str(c) for c in cmd), flush=True)
    if dry_run:
        return
    env = os.environ.copy()
    # COLMAP's CLI tools still construct a Qt application. In headless/nohup
    # runs, conda OpenCV can point Qt at an unusable xcb plugin and abort before
    # any SfM work starts, so force an offscreen platform for all subprocesses.
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    env.pop("QT_PLUGIN_PATH", None)
    subprocess.run([str(c) for c in cmd], check=True, cwd=str(cwd) if cwd else None, env=env)


def resolve_colmap(path: Path | None) -> Path:
    if path is not None:
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    found = shutil.which("colmap")
    if found:
        return Path(found)
    for candidate in DEFAULT_COLMAP_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not find COLMAP. Pass --colmap_bin /path/to/colmap")


def discover_videos(args: argparse.Namespace, video_dir: Path, active_bands: list[str]) -> list[VideoItem]:
    files = sorted(p for p in video_dir.iterdir() if p.is_file() and p.suffix in IMAGE_EXTENSIONS)
    items: list[VideoItem] = []
    used: set[Path] = set()
    rgb_contains = parse_token_list(args.rgb_contains)
    for spec in VIDEO_SPECS:
        band = spec["band"]
        if band not in active_bands:
            continue
        contains = rgb_contains if band == "rgb" and rgb_contains else spec["contains"]
        matches = []
        for path in files:
            name = path.name
            if path in used:
                continue
            if any(token not in name for token in contains):
                continue
            if any(token in name for token in spec["exclude"]):
                continue
            matches.append(path)
        if not matches:
            raise RuntimeError(f"Expected at least one video for {band}; found none")
        for segment_index, path in enumerate(matches):
            start_output_index = segment_index * frames_requested_for_band(args, band) + 1
            items.append(VideoItem(band, path, segment_index, start_output_index))
            used.add(path)
    return items


def frame_indices(total_frames: int, count: int) -> list[int]:
    if total_frames <= 0:
        raise ValueError("Video reports no frames")
    if count <= 1:
        return [0]
    if total_frames <= count:
        return list(range(total_frames))
    return [int(round(v)) for v in linspace(0, total_frames - 1, count)]


def frames_requested_for_band(args: argparse.Namespace, band: str) -> int:
    if band == "rgb" and args.rgb_frames_per_video is not None:
        return int(args.rgb_frames_per_video)
    return int(args.frames_per_video)


def parse_frame_range_spec(spec: str, total_frames: int, default_count: int) -> tuple[int, int, int]:
    parts = [part.strip() for part in spec.split(":") if part.strip()]
    if len(parts) not in {2, 3}:
        raise ValueError(f"Invalid frame range '{spec}', expected start:end[:count]")

    def to_index(value: str) -> int:
        raw = float(value)
        if 0.0 <= raw <= 1.0:
            return int(round(raw * (total_frames - 1)))
        return int(round(raw))

    start = max(0, min(total_frames - 1, to_index(parts[0])))
    end = max(0, min(total_frames - 1, to_index(parts[1])))
    if end < start:
        start, end = end, start
    count = int(parts[2]) if len(parts) == 3 else default_count
    return start, end, max(0, count)


def frame_indices_for_item(args: argparse.Namespace, item: VideoItem, total_frames: int) -> list[int]:
    count = frames_requested_for_band(args, item.band)
    indices = set(frame_indices(total_frames, count))
    if item.band == "rgb" and args.rgb_dense_ranges.strip():
        for spec in args.rgb_dense_ranges.split(","):
            spec = spec.strip()
            if not spec:
                continue
            start, end, dense_count = parse_frame_range_spec(spec, total_frames, args.rgb_dense_range_frames)
            indices.update(int(round(v)) for v in linspace(start, end, dense_count))
    return sorted(i for i in indices if 0 <= i < total_frames)


def linspace(start: float, stop: float, count: int) -> list[float]:
    if count == 1:
        return [start]
    step = (stop - start) / float(count - 1)
    return [start + i * step for i in range(count)]


def extract_video_frames(args: argparse.Namespace, item: VideoItem, raw_dir: Path, images_dir: Path, images_rgb_dir: Path) -> dict[str, Any]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for frame extraction") from exc

    cap = cv2.VideoCapture(str(item.path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {item.path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    indices = frame_indices_for_item(args, item, total)

    band_raw = raw_dir / item.band
    band_raw.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    if item.band == "rgb":
        images_rgb_dir.mkdir(parents=True, exist_ok=True)

    ext = "." + args.image_ext
    targets = []
    for local_idx, frame_idx in enumerate(indices):
        out_idx = item.start_output_index + local_idx
        stem = f"{item.band}_{out_idx:05d}"
        out_path = band_raw / f"{stem}{ext}"
        staged_path = images_dir / out_path.name
        rgb_path = images_rgb_dir / out_path.name
        targets.append((frame_idx, out_path, staged_path, rgb_path))

    written = []
    pending = []
    for frame_idx, out_path, staged_path, rgb_path in targets:
        if args.skip_existing and out_path.exists() and staged_path.exists() and (item.band != "rgb" or rgb_path.exists()):
            written.append(out_path.name)
        else:
            pending.append((frame_idx, out_path, staged_path, rgb_path))

    pending_by_frame = {frame_idx: (out_path, staged_path, rgb_path) for frame_idx, out_path, staged_path, rgb_path in pending}
    next_pending = iter(sorted(pending_by_frame))
    next_frame = next(next_pending, None)
    current_frame = 0
    while next_frame is not None:
        ok, frame = cap.read()
        if not ok or frame is None:
            print(f"Warning: failed to read {item.path.name} frame {next_frame}")
            break
        if current_frame == next_frame:
            out_path, staged_path, rgb_path = pending_by_frame[next_frame]
            params = []
            if args.image_ext == "jpg":
                params = [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)]
            if not cv2.imwrite(str(out_path), frame, params):
                raise RuntimeError(f"Could not write {out_path}")
            if args.grayscale_colmap_bands and item.band != "rgb":
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if not cv2.imwrite(str(staged_path), gray, params):
                    raise RuntimeError(f"Could not write {staged_path}")
            else:
                link_or_copy(out_path, staged_path, copy=args.copy_images)
            if item.band == "rgb":
                link_or_copy(out_path, rgb_path, copy=args.copy_images)
            written.append(out_path.name)
            next_frame = next(next_pending, None)
        current_frame += 1
    cap.release()
    return {
        "band": item.band,
        "video": str(item.path),
        "total_frames": total,
        "fps": fps,
        "width": width,
        "height": height,
        "segment_index": item.segment_index,
        "start_output_index": item.start_output_index,
        "requested_frames": frames_requested_for_band(args, item.band),
        "dense_ranges": args.rgb_dense_ranges if item.band == "rgb" else "",
        "grayscale_colmap_staging": bool(args.grayscale_colmap_bands and item.band != "rgb"),
        "written_frames": len(written),
        "source_frame_indices": indices[:len(written)],
        "image_names": written,
    }


def link_or_copy(src: Path, dst: Path, *, copy: bool) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        shutil.copy2(src, dst)
    else:
        os.symlink(os.path.abspath(src), dst)


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def image_names(images_dir: Path, band: str | None = None) -> list[str]:
    files = sorted(p.name for p in images_dir.iterdir() if p.is_file() or p.is_symlink())
    if band is not None:
        prefix = f"{band}_"
        files = [name for name in files if name.startswith(prefix)]
    return files


def make_pairs(args: argparse.Namespace, images_dir: Path, lists_dir: Path) -> dict[str, int]:
    all_pairs: set[tuple[str, str]] = set()
    rgb = image_names(images_dir, "rgb")
    bands = args.active_bands

    def add_pair(a: str, b: str) -> None:
        if a == b:
            return
        all_pairs.add(tuple(sorted((a, b))))

    if args.registration_mode == "direct":
        intra_radius = args.direct_intra_band_radius
        cross_radius = args.direct_cross_band_radius
    else:
        intra_radius = args.sequential_window
        cross_radius = args.cross_band_window

    per_band_counts: dict[str, int] = {}
    for band in bands:
        names = image_names(images_dir, band)
        per_band_counts[band] = len(names)
        for i, name in enumerate(names):
            for j in range(i + 1, min(len(names), i + intra_radius + 1)):
                add_pair(name, names[j])
            if band == "rgb" and args.long_stride > 0:
                for j in range(i + args.long_stride, len(names), args.long_stride):
                    add_pair(name, names[j])

    for band in bands:
        if band == "rgb":
            continue
        names = image_names(images_dir, band)
        for i, name in enumerate(names):
            rgb_center = min(i, len(rgb) - 1)
            lo = max(0, rgb_center - cross_radius)
            hi = min(len(rgb), rgb_center + cross_radius + 1)
            for rgb_name in rgb[lo:hi]:
                add_pair(name, rgb_name)

    pairs = sorted(all_pairs)
    write_lines(lists_dir / "match_pairs.txt", [f"{a} {b}" for a, b in pairs])
    return {
        "registration_mode": args.registration_mode,
        "num_pairs": len(pairs),
        "num_images_per_band": per_band_counts,
        "direct_intra_band_radius": args.direct_intra_band_radius,
        "direct_cross_band_radius": args.direct_cross_band_radius,
        "sequential_window": args.sequential_window,
        "cross_band_window": args.cross_band_window,
    }

def write_image_lists(output_dir: Path, active_bands: list[str]) -> None:
    images_dir = output_dir / "images"
    lists_dir = output_dir / "colmap_shared" / "lists"
    lists_dir.mkdir(parents=True, exist_ok=True)
    all_names: list[str] = []
    for band in active_bands:
        names = image_names(images_dir, band)
        all_names.extend(names)
        write_lines(lists_dir / f"{band}_images.txt", names)
    write_lines(lists_dir / "all_images.txt", sorted(all_names))


def run_feature_extraction(args: argparse.Namespace, colmap: Path, output_dir: Path) -> None:
    db = output_dir / "colmap_shared" / "database.db"
    images = output_dir / "images"
    lists = output_dir / "colmap_shared" / "lists"
    db.parent.mkdir(parents=True, exist_ok=True)
    if db.exists() and not args.skip_existing:
        db.unlink()
    for band in args.active_bands:
        list_path = lists / f"{band}_images.txt"
        if not list_path.exists() or not list_path.read_text().strip():
            continue
        run([
            colmap, "feature_extractor",
            "--database_path", db,
            "--image_path", images,
            "--image_list_path", list_path,
            "--ImageReader.single_camera", "1",
            "--ImageReader.camera_model", args.camera_model,
            "--ImageReader.default_focal_length_factor", str(args.default_focal_length_factor),
            "--SiftExtraction.use_gpu", "1",
            "--SiftExtraction.gpu_index", args.gpu_index,
            "--SiftExtraction.max_image_size", str(args.max_image_size),
            "--SiftExtraction.max_num_features", str(args.max_num_features),
            "--SiftExtraction.estimate_affine_shape", str(args.estimate_affine_shape),
            "--SiftExtraction.domain_size_pooling", str(args.domain_size_pooling),
            "--SiftExtraction.num_threads", str(args.sift_num_threads),
        ], dry_run=args.dry_run)


def run_matching(args: argparse.Namespace, colmap: Path, output_dir: Path) -> None:
    db = output_dir / "colmap_shared" / "database.db"
    pairs = output_dir / "colmap_shared" / "lists" / "match_pairs.txt"
    run([
        colmap, "matches_importer",
        "--database_path", db,
        "--match_list_path", pairs,
        "--match_type", "pairs",
        "--SiftMatching.use_gpu", str(args.matching_use_gpu),
        "--SiftMatching.num_threads", str(args.matching_num_threads),
        "--SiftMatching.gpu_index", args.gpu_index,
        "--SiftMatching.max_num_matches", str(args.max_num_matches),
        "--SiftMatching.guided_matching", str(args.guided_matching),
        "--TwoViewGeometry.min_num_inliers", str(args.min_num_inliers),
    ], dry_run=args.dry_run)



def parse_rgb_variants(args: argparse.Namespace) -> list[tuple[str, str]]:
    raw = args.rgb_variants.strip()
    if not raw:
        return [(args.matcher_type, args.camera_model)]
    variants: list[tuple[str, str]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            matcher, camera_model = [part.strip() for part in item.split(":", 1)]
        else:
            matcher, camera_model = item, args.camera_model
        if matcher not in {"exhaustive", "sequential_loop"}:
            raise ValueError(f"RGB variant matcher must be exhaustive or sequential_loop, got {matcher}")
        variants.append((matcher, camera_model.upper()))
    if not variants:
        raise ValueError("No RGB COLMAP variants requested")
    return variants


def rgb_variant_name(matcher_type: str, camera_model: str) -> str:
    matcher_part = "seqloop" if matcher_type == "sequential_loop" else matcher_type
    return f"colmap_rgb_{matcher_part}_{camera_model.lower()}"


def run_feature_extraction_for_list(
    args: argparse.Namespace,
    colmap: Path,
    *,
    db: Path,
    images: Path,
    image_list: Path,
    camera_model: str,
) -> None:
    if not image_list.exists() or not image_list.read_text().strip():
        return
    db.parent.mkdir(parents=True, exist_ok=True)
    run([
        colmap, "feature_extractor",
        "--database_path", db,
        "--image_path", images,
        "--image_list_path", image_list,
        "--ImageReader.single_camera", "1",
        "--ImageReader.camera_model", camera_model,
        "--ImageReader.default_focal_length_factor", str(args.default_focal_length_factor),
        "--SiftExtraction.use_gpu", "1",
        "--SiftExtraction.gpu_index", args.gpu_index,
        "--SiftExtraction.max_image_size", str(args.max_image_size),
        "--SiftExtraction.max_num_features", str(args.max_num_features),
        "--SiftExtraction.estimate_affine_shape", str(args.estimate_affine_shape),
        "--SiftExtraction.domain_size_pooling", str(args.domain_size_pooling),
        "--SiftExtraction.num_threads", str(args.sift_num_threads),
    ], dry_run=args.dry_run)


def run_rgb_matching_variant(args: argparse.Namespace, colmap: Path, *, db: Path, matcher_type: str) -> None:
    base = [
        colmap,
        "exhaustive_matcher" if matcher_type == "exhaustive" else "sequential_matcher",
        "--database_path", db,
        "--SiftMatching.use_gpu", str(args.matching_use_gpu),
            "--SiftMatching.num_threads", str(args.matching_num_threads),
        "--SiftMatching.gpu_index", args.gpu_index,
        "--SiftMatching.max_num_matches", str(args.max_num_matches),
        "--SiftMatching.guided_matching", str(args.guided_matching),
    ]
    if matcher_type == "sequential_loop":
        base.extend([
            "--SequentialMatching.overlap", str(args.sequential_overlap),
            "--SequentialMatching.loop_detection", "0",
        ])
    run(base, dry_run=args.dry_run)


def run_rgb_mapper_variant(args: argparse.Namespace, colmap: Path, *, db: Path, images: Path, lists: Path, sparse_root: Path) -> Path:
    if sparse_root.exists() and not args.skip_existing:
        shutil.rmtree(sparse_root)
    sparse_root.mkdir(parents=True, exist_ok=True)
    run([
        colmap, "mapper",
        "--database_path", db,
        "--image_path", images,
        "--image_list_path", lists / "rgb_images.txt",
        "--output_path", sparse_root,
        "--Mapper.multiple_models", "1",
        "--Mapper.min_model_size", str(args.mapper_min_model_size),
        "--Mapper.min_num_matches", str(args.min_num_matches),
        "--Mapper.ba_refine_focal_length", "1",
        "--Mapper.ba_refine_principal_point", "0",
        "--Mapper.ba_refine_extra_params", "1",
        "--Mapper.ba_global_images_freq", "100000000",
        "--Mapper.ba_global_points_freq", "100000000",
    ], dry_run=args.dry_run)
    if args.dry_run:
        return sparse_root / "0"
    return largest_model_dir(sparse_root)


def run_rgb_colmap_variant(args: argparse.Namespace, colmap: Path, output_dir: Path, matcher_type: str, camera_model: str) -> dict[str, Any]:
    name = rgb_variant_name(matcher_type, camera_model)
    root = output_dir / name
    db = root / "database.db"
    sparse_root = root / "sparse"
    images = output_dir / "images"
    lists = output_dir / "colmap_shared" / "lists"
    root.mkdir(parents=True, exist_ok=True)
    if db.exists() and not args.skip_existing:
        db.unlink()

    if stage_enabled(args, "features"):
        run_feature_extraction_for_list(args, colmap, db=db, images=images, image_list=lists / "rgb_images.txt", camera_model=camera_model)
    if stage_enabled(args, "match"):
        run_rgb_matching_variant(args, colmap, db=db, matcher_type=matcher_type)

    model = sparse_root / "0"
    if stage_enabled(args, "map"):
        model = run_rgb_mapper_variant(args, colmap, db=db, images=images, lists=lists, sparse_root=sparse_root)
    elif sparse_root.exists() and not args.dry_run:
        model = largest_model_dir(sparse_root)

    expected_rgb = len(image_names(images, "rgb"))
    diagnostic = {"variant": name, "matcher_type": matcher_type, "camera_model": camera_model, "root": str(root), "database": str(db), "model": str(model)}
    if (stage_enabled(args, "diagnose") or stage_enabled(args, "finalize") or args.stage == "all") and not args.dry_run and model.exists():
        diagnostic.update(write_colmap_diagnostics(args, colmap, root, sparse_root, model, db, expected_rgb))
    (root / "variant_config.json").write_text(json.dumps(diagnostic, indent=2))
    return diagnostic


def qvec_to_rotmat(q: list[float]) -> list[list[float]]:
    w, x, y, z = q
    return [
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
    ]


def camera_center(qvec: list[float], tvec: list[float]) -> list[float]:
    rot = qvec_to_rotmat(qvec)
    return [
        -sum(rot[row][col] * tvec[row] for row in range(3))
        for col in range(3)
    ]


def median(values: list[float]) -> float | None:
    if not values:
        return None
    vals = sorted(values)
    mid = len(vals) // 2
    if len(vals) % 2:
        return float(vals[mid])
    return float((vals[mid - 1] + vals[mid]) / 2.0)


def mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def read_images_txt_stats(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    lines = path.read_text().splitlines()
    out: list[dict[str, Any]] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("#"):
            i += 1
            continue
        parts = line.split()
        if len(parts) < 10:
            i += 1
            continue
        points_line = lines[i + 1].strip() if i + 1 < len(lines) else ""
        triples = points_line.split()
        point_ids = []
        for j in range(2, len(triples), 3):
            try:
                pid = int(triples[j])
            except ValueError:
                continue
            if pid != -1:
                point_ids.append(pid)
        qvec = [float(v) for v in parts[1:5]]
        tvec = [float(v) for v in parts[5:8]]
        out.append({
            "image_id": int(parts[0]),
            "camera_id": int(parts[8]),
            "name": parts[9],
            "frame": frame_number_from_name(parts[9]),
            "center": camera_center(qvec, tvec),
            "num_observations": len(point_ids),
        })
        i += 2
    return out


def read_points_errors(path: Path) -> list[float]:
    if not path.exists():
        return []
    errors = []
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 8:
            errors.append(float(parts[7]))
    return errors


def frame_number_from_name(name: str) -> int | None:
    match = re.search(r"(\d+)(?=\.[^.]+$|$)", Path(name).stem)
    return int(match.group(1)) if match else None


def pair_id_to_image_ids(pair_id: int) -> tuple[int, int]:
    image_id2 = pair_id % COLMAP_PAIR_ID_MULTIPLIER
    image_id1 = (pair_id - image_id2) // COLMAP_PAIR_ID_MULTIPLIER
    return int(image_id1), int(image_id2)


def database_pair_stats(db: Path) -> dict[int, dict[str, int]]:
    stats: dict[int, dict[str, int]] = {}
    if not db.exists():
        return stats
    try:
        conn = sqlite3.connect(str(db))
    except sqlite3.Error:
        return stats
    try:
        for pair_id, rows in conn.execute("SELECT pair_id, rows FROM matches"):
            a, b = pair_id_to_image_ids(int(pair_id))
            for image_id in (a, b):
                item = stats.setdefault(image_id, {"num_match_pairs": 0, "num_matches": 0, "num_inliers": 0})
                item["num_match_pairs"] += 1
                item["num_matches"] += int(rows or 0)
        for pair_id, rows in conn.execute("SELECT pair_id, rows FROM two_view_geometries"):
            a, b = pair_id_to_image_ids(int(pair_id))
            for image_id in (a, b):
                item = stats.setdefault(image_id, {"num_match_pairs": 0, "num_matches": 0, "num_inliers": 0})
                item["num_inliers"] += int(rows or 0)
    except sqlite3.Error:
        pass
    finally:
        conn.close()
    return stats


def pca_top_view(points: list[list[float]]) -> list[list[float]]:
    if not points:
        return []
    try:
        import numpy as np
    except ImportError:
        return [[p[0], p[2]] for p in points]
    arr = np.asarray(points, dtype=float)
    arr = arr - arr.mean(axis=0, keepdims=True)
    if len(arr) < 2:
        return arr[:, [0, 2]].tolist()
    _, _, vt = np.linalg.svd(arr, full_matrices=False)
    coords = arr @ vt[:2].T
    return coords.tolist()


def fitted_line_angle_deg(coords: list[list[float]]) -> float | None:
    if len(coords) < 20:
        return None
    try:
        import numpy as np
    except ImportError:
        return None
    arr = np.asarray(coords, dtype=float)
    deltas = np.linalg.norm(np.diff(arr, axis=0), axis=1)
    if len(deltas) == 0:
        return None
    split = int(np.argmax(deltas)) + 1
    if split < 5 or len(arr) - split < 5:
        split = len(arr) // 2
    dirs = []
    for segment in (arr[:split], arr[split:]):
        centered = segment - segment.mean(axis=0, keepdims=True)
        if len(centered) < 2:
            return None
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        dirs.append(vt[0])
    dot = abs(float(np.dot(dirs[0], dirs[1]) / max(1e-12, np.linalg.norm(dirs[0]) * np.linalg.norm(dirs[1]))))
    dot = max(-1.0, min(1.0, dot))
    return float(math.degrees(math.acos(dot)))


def turn_frame_bounds(args: argparse.Namespace, images: list[dict[str, Any]], coords: list[list[float]]) -> tuple[int | None, int | None]:
    frames = [img["frame"] for img in images if img.get("frame") is not None]
    if not frames:
        return None, None
    if args.turn_frame_range.strip():
        lo, hi, _ = parse_frame_range_spec(args.turn_frame_range, max(frames) + 1, 0)
        return lo, hi
    if len(coords) < 3:
        mid = frames[len(frames) // 2]
    else:
        try:
            import numpy as np
            arr = np.asarray(coords, dtype=float)
            v1 = arr[1:-1] - arr[:-2]
            v2 = arr[2:] - arr[1:-1]
            denom = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)
            valid = denom > 1e-9
            angles = np.zeros(len(v1), dtype=float)
            dots = (v1 * v2).sum(axis=1)
            angles[valid] = np.arccos(np.clip(dots[valid] / denom[valid], -1.0, 1.0))
            mid = images[int(np.argmax(angles)) + 1].get("frame") or frames[len(frames) // 2]
        except Exception:
            mid = frames[len(frames) // 2]
    return int(mid) - args.turn_weak_window, int(mid) + args.turn_weak_window


def write_trajectory_plot(path: Path, images: list[dict[str, Any]], coords: list[list[float]]) -> None:
    csv_path = path.with_suffix(".csv")
    csv_lines = ["name,frame,x,y,num_observations,num_matches,num_inliers"]
    for img, xy in zip(images, coords):
        csv_lines.append(
            f"{img['name']},{img.get('frame') or ''},{xy[0]:.8f},{xy[1]:.8f},{img.get('num_observations', 0)},{img.get('num_matches', 0)},{img.get('num_inliers', 0)}"
        )
    csv_path.write_text("\n".join(csv_lines) + "\n")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    if not coords:
        return
    xs = [xy[0] for xy in coords]
    ys = [xy[1] for xy in coords]
    frames = [img.get("frame") or idx for idx, img in enumerate(images)]
    plt.figure(figsize=(8, 6))
    plt.scatter(xs, ys, c=frames, s=14, cmap="viridis")
    plt.plot(xs, ys, linewidth=0.8, alpha=0.65)
    if xs:
        plt.scatter([xs[0]], [ys[0]], c="lime", s=60, edgecolors="black", label="start")
        plt.scatter([xs[-1]], [ys[-1]], c="red", s=60, edgecolors="black", label="end")
    plt.axis("equal")
    plt.colorbar(label="RGB frame")
    plt.legend(loc="best")
    plt.title("COLMAP RGB Camera Trajectory (Top View PCA)")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def write_colmap_diagnostics(args: argparse.Namespace, colmap: Path, variant_root: Path, sparse_root: Path, model: Path, db: Path, expected_rgb: int) -> dict[str, Any]:
    txt_model = variant_root / "model_txt"
    convert_model(args, colmap, model, txt_model)
    images = read_images_txt_stats(txt_model / "images.txt")
    images.sort(key=lambda item: (item.get("frame") is None, item.get("frame") or 0, item["name"]))
    point_errors = read_points_errors(txt_model / "points3D.txt")
    pair_stats = database_pair_stats(db)
    for img in images:
        img.update(pair_stats.get(img["image_id"], {"num_match_pairs": 0, "num_matches": 0, "num_inliers": 0}))
    obs = [float(img["num_observations"]) for img in images]
    reproj = [float(v) for v in point_errors]
    coords = pca_top_view([img["center"] for img in images])
    line_angle = fitted_line_angle_deg(coords)
    turn_lo, turn_hi = turn_frame_bounds(args, images, coords)
    obs_med = median(obs) or 0.0
    inlier_values = [float(img.get("num_inliers", 0)) for img in images]
    inlier_med = median(inlier_values) or 0.0
    weak = []
    for img in images:
        frame = img.get("frame")
        near_turn = frame is not None and turn_lo is not None and turn_hi is not None and turn_lo <= frame <= turn_hi
        if near_turn and (img["num_observations"] <= obs_med or img.get("num_inliers", 0) <= inlier_med):
            weak.append({
                "name": img["name"],
                "frame": frame,
                "num_observations": img["num_observations"],
                "num_matches": img.get("num_matches", 0),
                "num_inliers": img.get("num_inliers", 0),
            })
    weak.sort(key=lambda item: (item["num_observations"], item["num_inliers"], item["frame"] or 0))
    sparse_models = len([p for p in sparse_root.iterdir() if p.is_dir()]) if sparse_root.exists() else 0
    registered = len(images)
    registered_fraction = registered / expected_rgb if expected_rgb else 0.0
    folded = line_angle is not None and line_angle > args.bad_trajectory_line_angle
    quality_reasons = []
    if expected_rgb and registered_fraction < 0.80:
        quality_reasons.append(f"low_registered_fraction={registered_fraction:.3f}")
    if reproj and (mean(reproj) or 0.0) > 5.0:
        quality_reasons.append(f"high_mean_reprojection_error={mean(reproj):.3f}")
    if args.fail_on_folded_trajectory and folded:
        quality_reasons.append(f"folded_trajectory_line_angle={line_angle:.1f}")
    quality_ok = not quality_reasons
    plot_path = variant_root / "trajectory_topview.png"
    write_trajectory_plot(plot_path, images, coords)
    diagnostics = {
        "registered_images": registered,
        "expected_rgb_images": expected_rgb,
        "registered_fraction": registered_fraction,
        "sparse_models": sparse_models,
        "mean_observations_per_registered_image": mean(obs),
        "median_observations_per_registered_image": median(obs),
        "mean_reprojection_error": mean(reproj),
        "median_reprojection_error": median(reproj),
        "trajectory_line_angle_degrees": line_angle,
        "folded_trajectory_warning": folded,
        "turn_frame_range": [turn_lo, turn_hi],
        "weakly_registered_images_near_turn": weak[:40],
        "per_image_match_stats": [
            {
                "name": img["name"],
                "frame": img.get("frame"),
                "num_observations": img.get("num_observations", 0),
                "num_match_pairs": img.get("num_match_pairs", 0),
                "num_matches": img.get("num_matches", 0),
                "num_inliers": img.get("num_inliers", 0),
            }
            for img in images
        ],
        "trajectory_plot": str(plot_path),
        "trajectory_csv": str(plot_path.with_suffix(".csv")),
        "quality_ok": quality_ok,
        "quality_reasons": quality_reasons,
    }
    (variant_root / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
    print(f"Diagnostics {variant_root.name}: registered {registered}/{expected_rgb}, reproj mean={diagnostics['mean_reprojection_error']}, quality_ok={quality_ok}", flush=True)
    if weak:
        print(f"  Weak near-turn RGB images: {', '.join(item['name'] for item in weak[:8])}", flush=True)
    return diagnostics


def select_rgb_variant(args: argparse.Namespace, output_dir: Path, diagnostics: list[dict[str, Any]]) -> dict[str, Any]:
    if args.selected_rgb_variant != "auto":
        for diag in diagnostics:
            if Path(diag.get("root", "")).name == args.selected_rgb_variant or diag.get("variant") == args.selected_rgb_variant:
                return diag
        root = output_dir / args.selected_rgb_variant
        if not root.exists():
            raise FileNotFoundError(f"Selected RGB variant not found: {args.selected_rgb_variant}")
        data = load_json(root / "variant_config.json", {"variant": args.selected_rgb_variant, "root": str(root), "model": str(root / "sparse" / "0"), "database": str(root / "database.db")})
        return data
    if not diagnostics:
        raise RuntimeError("No RGB variant diagnostics available for auto selection")
    def score(diag: dict[str, Any]) -> tuple[float, float, float, float]:
        return (
            1.0 if diag.get("quality_ok") else 0.0,
            float(diag.get("registered_images") or 0),
            float(diag.get("median_observations_per_registered_image") or 0.0),
            -float(diag.get("mean_reprojection_error") or 9999.0),
        )
    return max(diagnostics, key=score)


def write_rgb_variants_summary(output_dir: Path, diagnostics: list[dict[str, Any]], selected: dict[str, Any]) -> None:
    summary = {
        "selected_variant": selected.get("variant") or Path(selected.get("root", "")).name,
        "selected_root": selected.get("root"),
        "selected_model": selected.get("model"),
        "selected_database": selected.get("database"),
        "variants": diagnostics,
    }
    (output_dir / "colmap_rgb_variants_summary.json").write_text(json.dumps(summary, indent=2))


def run_rgb_mapping(args: argparse.Namespace, colmap: Path, output_dir: Path) -> Path:
    db = output_dir / "colmap_shared" / "database.db"
    images = output_dir / "images"
    lists = output_dir / "colmap_shared" / "lists"
    sparse_rgb = output_dir / "colmap_shared" / "sparse_rgb"
    if sparse_rgb.exists() and not args.skip_existing:
        shutil.rmtree(sparse_rgb)
    sparse_rgb.mkdir(parents=True, exist_ok=True)
    run([
        colmap, "mapper",
        "--database_path", db,
        "--image_path", images,
        "--image_list_path", lists / "rgb_images.txt",
        "--output_path", sparse_rgb,
        "--Mapper.multiple_models", "1",
        "--Mapper.min_model_size", str(args.mapper_min_model_size),
        "--Mapper.min_num_matches", str(args.min_num_matches),
        "--Mapper.ba_refine_principal_point", "0",
        "--Mapper.ba_global_images_freq", "100000000",
        "--Mapper.ba_global_points_freq", "100000000",
    ], dry_run=args.dry_run)
    if args.dry_run:
        return sparse_rgb / "0"
    return largest_model_dir(sparse_rgb)


def run_direct_mapping(args: argparse.Namespace, colmap: Path, output_dir: Path) -> Path:
    db = output_dir / "colmap_shared" / "database.db"
    images = output_dir / "images"
    lists = output_dir / "colmap_shared" / "lists"
    sparse_direct = output_dir / "colmap_shared" / "sparse_direct"
    if sparse_direct.exists() and not args.skip_existing:
        shutil.rmtree(sparse_direct)
    sparse_direct.mkdir(parents=True, exist_ok=True)
    run([
        colmap, "mapper",
        "--database_path", db,
        "--image_path", images,
        "--image_list_path", lists / "all_images.txt",
        "--output_path", sparse_direct,
        "--Mapper.multiple_models", "1",
        "--Mapper.min_model_size", str(args.mapper_min_model_size),
        "--Mapper.min_num_matches", str(args.min_num_matches),
        "--Mapper.ba_refine_principal_point", "0",
        "--Mapper.ba_global_images_freq", "100000000",
        "--Mapper.ba_global_points_freq", "100000000",
    ], dry_run=args.dry_run)
    if args.dry_run:
        return sparse_direct / "0"
    return largest_model_dir(sparse_direct)


def largest_model_dir(root: Path) -> Path:
    candidates = [p for p in root.iterdir() if p.is_dir()] if root.exists() else []
    if not candidates:
        raise FileNotFoundError(f"No COLMAP model directories found in {root}")
    best = None
    best_count = -1
    for path in candidates:
        count = count_registered_images(path)
        if count > best_count:
            best = path
            best_count = count
    assert best is not None
    return best


def count_registered_images(model_dir: Path) -> int:
    txt = model_dir / "images.txt"
    if txt.exists():
        return sum(1 for line in txt.read_text().splitlines() if line and not line.startswith("#")) // 2
    bin_file = model_dir / "images.bin"
    if bin_file.exists():
        with bin_file.open("rb") as f:
            data = f.read(8)
        return int.from_bytes(data, "little", signed=False) if len(data) == 8 else 0
    return 0



def prepare_selected_database_for_registration(args: argparse.Namespace, colmap: Path, output_dir: Path, selected: dict[str, Any]) -> Path:
    db = Path(selected["database"])
    images = output_dir / "images"
    lists = output_dir / "colmap_shared" / "lists"
    camera_model = str(selected.get("camera_model") or args.camera_model).upper()
    if len(args.active_bands) <= 1:
        return db
    for band in args.active_bands:
        if band == "rgb":
            continue
        if stage_enabled(args, "features"):
            run_feature_extraction_for_list(args, colmap, db=db, images=images, image_list=lists / f"{band}_images.txt", camera_model=camera_model)
    if stage_enabled(args, "match"):
        pairs = lists / "match_pairs.txt"
        run([
            colmap, "matches_importer",
            "--database_path", db,
            "--match_list_path", pairs,
            "--match_type", "pairs",
            "--SiftMatching.use_gpu", str(args.matching_use_gpu),
            "--SiftMatching.num_threads", str(args.matching_num_threads),
            "--SiftMatching.gpu_index", args.gpu_index,
            "--SiftMatching.max_num_matches", str(args.max_num_matches),
            "--SiftMatching.guided_matching", str(args.guided_matching),
            "--TwoViewGeometry.min_num_inliers", str(args.min_num_inliers),
        ], dry_run=args.dry_run)
    return db


def run_registration(args: argparse.Namespace, colmap: Path, output_dir: Path, rgb_model: Path, database_path: Path | None = None) -> Path:
    db = database_path or (output_dir / "colmap_shared" / "database.db")
    registered = output_dir / "colmap_shared" / "sparse_registered"
    if registered.exists() and not args.skip_existing:
        shutil.rmtree(registered)
    registered.mkdir(parents=True, exist_ok=True)
    out = registered / "0"
    if out.exists() and not out.is_dir():
        out.unlink()
    out.mkdir(parents=True, exist_ok=True)
    run([
        colmap, "image_registrator",
        "--database_path", db,
        "--input_path", rgb_model,
        "--output_path", out,
        "--Mapper.fix_existing_images", "1",
        "--Mapper.ba_refine_focal_length", "0",
        "--Mapper.ba_refine_principal_point", "0",
        "--Mapper.ba_refine_extra_params", "0",
        "--Mapper.ba_global_images_freq", "100000000",
        "--Mapper.ba_global_points_freq", "100000000",
        "--Mapper.abs_pose_min_num_inliers", str(args.abs_pose_min_num_inliers),
        "--Mapper.abs_pose_max_error", str(args.abs_pose_max_error),
        "--Mapper.min_num_matches", str(args.min_num_matches),
    ], dry_run=args.dry_run)
    return out


def convert_model(args: argparse.Namespace, colmap: Path, input_model: Path, output_model: Path) -> None:
    if output_model.exists() and not args.skip_existing:
        shutil.rmtree(output_model)
    output_model.mkdir(parents=True, exist_ok=True)
    run([
        colmap, "model_converter",
        "--input_path", input_model,
        "--output_path", output_model,
        "--output_type", "TXT",
    ], dry_run=args.dry_run)


def write_scene_metadata(args: argparse.Namespace, output_dir: Path, videos: list[VideoItem], extraction: list[dict[str, Any]]) -> None:
    images_dir = output_dir / "images"
    final_images = {
        name
        for band in args.active_bands
        for name in image_names(images_dir, band)
    }
    active_channels = {Path(name).stem: args.active_band_channels[Path(name).stem.split("_", 1)[0]] for name in final_images}
    (output_dir / "band_info.json").write_text(json.dumps(active_channels, indent=2))

    frame_info = {}
    for name in sorted(final_images):
        stem = Path(name).stem
        band = stem.split("_", 1)[0]
        frame_info[stem] = {
            "band_key": "RGB" if band == "rgb" else band,
            "band_name": band,
            "channels": args.active_band_channels[band],
        }
    (output_dir / "frame_info.json").write_text(json.dumps(frame_info, indent=2))

    summary = {
        "source": "prepare_vineyard_video_colmap.py",
        "video_dir": str(args.video_dir),
        "output_dir": str(output_dir),
        "frames_per_video": args.frames_per_video,
        "videos_per_band": {band: sum(1 for item in videos if item.band == band) for band in args.active_bands},
        "frames_per_band_target": {band: frames_requested_for_band(args, band) * sum(1 for item in videos if item.band == band) for band in args.active_bands},
        "rgb_frames_per_video": args.rgb_frames_per_video,
        "rgb_dense_ranges": args.rgb_dense_ranges,
        "bands": args.active_bands,
        "channels": args.active_band_channels,
        "extraction": extraction,
    }
    (output_dir / "partial_channels_summary.json").write_text(json.dumps(summary, indent=2))


def parse_registered_names(images_txt: Path) -> list[str]:
    if not images_txt.exists():
        return []
    out = []
    for line in images_txt.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 10 and parts[9].lower().endswith((".png", ".jpg", ".jpeg")):
            out.append(parts[9])
    return out


def write_registration_summary(args: argparse.Namespace, output_dir: Path) -> None:
    names = parse_registered_names(output_dir / "sparse" / "0" / "images.txt")
    by_band: dict[str, int] = {band: 0 for band in args.active_bands}
    for name in names:
        band = Path(name).stem.split("_", 1)[0]
        by_band[band] = by_band.get(band, 0) + 1
    summary = {
        "registration_mode": args.registration_mode,
        "registered_images": len(names),
        "registered_per_band": by_band,
        "registered_names": names,
    }
    path = output_dir / "colmap_shared" / "registration_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["registered_per_band"], indent=2), flush=True)


def write_register_config(args: argparse.Namespace, output_dir: Path) -> None:
    config = {
        "registration_mode": args.registration_mode,
        "frames_per_video": args.frames_per_video,
        "rgb_frames_per_video": args.rgb_frames_per_video,
        "rgb_dense_ranges": args.rgb_dense_ranges,
        "rgb_variants": args.rgb_variants,
        "selected_rgb_variant": args.selected_rgb_variant,
        "matcher_type": args.matcher_type,
        "sequential_overlap": args.sequential_overlap,
        "camera_model": args.camera_model,
        "rgb_contains": args.rgb_contains,
        "active_bands": args.active_bands,
        "channels": args.active_band_channels,
        "direct_intra_band_radius": args.direct_intra_band_radius,
        "direct_cross_band_radius": args.direct_cross_band_radius,
        "sequential_window": args.sequential_window,
        "cross_band_window": args.cross_band_window,
        "max_num_features": args.max_num_features,
        "estimate_affine_shape": args.estimate_affine_shape,
        "domain_size_pooling": args.domain_size_pooling,
        "max_num_matches": args.max_num_matches,
        "min_num_matches": args.min_num_matches,
        "min_num_inliers": args.min_num_inliers,
        "guided_matching": args.guided_matching,
        "matching_use_gpu": args.matching_use_gpu,
        "matching_num_threads": args.matching_num_threads,
        "use_gpu": 1,
        "gpu_index": str(args.gpu_index),
        "grayscale_colmap_bands": args.grayscale_colmap_bands,
        "sift_num_threads": args.sift_num_threads,
        "sift_max_image_size": args.max_image_size,
        "global_ba_images_freq": 100000000,
        "global_ba_points_freq": 100000000,
    }
    path = output_dir / "colmap_shared" / "register_config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2))


def main() -> None:
    args = parse_args()
    args.active_bands = resolve_active_bands(args)
    args.active_band_channels = make_active_band_channels(args.active_bands)
    output_dir = args.output_dir.resolve()
    raw_dir = output_dir / "frames_raw"
    images_dir = output_dir / "images"
    images_rgb_dir = output_dir / "images_rgb"
    colmap = resolve_colmap(args.colmap_bin)

    output_dir.mkdir(parents=True, exist_ok=True)
    videos = discover_videos(args, args.video_dir, args.active_bands)
    (output_dir / "videos_manifest.json").write_text(json.dumps([
        {
            "band": item.band,
            "segment_index": item.segment_index,
            "start_output_index": item.start_output_index,
            "path": str(item.path),
        }
        for item in videos
    ], indent=2))
    write_register_config(args, output_dir)

    extraction = load_json(output_dir / "extraction_report.json", [])
    if stage_enabled(args, "extract"):
        extraction = []
        for item in videos:
            print(f"Extracting {frames_requested_for_band(args, item.band)} frames: {item.band} <- {item.path.name}", flush=True)
            extraction.append(extract_video_frames(args, item, raw_dir, images_dir, images_rgb_dir))
        (output_dir / "extraction_report.json").write_text(json.dumps(extraction, indent=2))
        write_scene_metadata(args, output_dir, videos, extraction)

    write_image_lists(output_dir, args.active_bands)
    pair_stats = make_pairs(args, images_dir, output_dir / "colmap_shared" / "lists")
    (output_dir / "colmap_shared" / "pairing_summary.json").write_text(json.dumps(pair_stats, indent=2))

    selected_quality: dict[str, Any] | None = None
    if args.registration_mode in {"rgb_register", "rgb_only"}:
        variant_diagnostics = []
        for matcher_type, camera_model in parse_rgb_variants(args):
            print(f"Running RGB COLMAP variant: {matcher_type} / {camera_model}", flush=True)
            variant_diagnostics.append(run_rgb_colmap_variant(args, colmap, output_dir, matcher_type, camera_model))
        selected = select_rgb_variant(args, output_dir, variant_diagnostics)
        write_rgb_variants_summary(output_dir, variant_diagnostics, selected)
        selected_quality = selected
        if not selected.get("quality_ok", True) and not args.allow_bad_colmap:
            raise RuntimeError(
                "Selected RGB COLMAP reconstruction failed quality checks; refusing to finalize/train from it. "
                f"Reasons: {selected.get('quality_reasons', [])}. "
                "Inspect trajectory_topview.png under the variant folder or pass --allow_bad_colmap to override."
            )
        model = Path(selected["model"])
        if args.registration_mode == "rgb_only" or len(args.active_bands) == 1:
            registered = model
        elif stage_enabled(args, "register"):
            selected_db = prepare_selected_database_for_registration(args, colmap, output_dir, selected)
            registered = run_registration(args, colmap, output_dir, model, selected_db)
        else:
            registered = output_dir / "colmap_shared" / "sparse_registered" / "0"
    else:
        if stage_enabled(args, "features"):
            run_feature_extraction(args, colmap, output_dir)
        if stage_enabled(args, "match"):
            run_matching(args, colmap, output_dir)
        model = output_dir / "colmap_shared" / "sparse_direct" / "0"
        if stage_enabled(args, "map"):
            model = run_direct_mapping(args, colmap, output_dir)
        registered = model

    if stage_enabled(args, "finalize"):
        convert_model(args, colmap, registered, output_dir / "sparse" / "0")
        write_registration_summary(args, output_dir)
        if selected_quality is not None:
            (output_dir / "colmap_quality.json").write_text(json.dumps(selected_quality, indent=2))

    print(f"Prepared scene root: {output_dir}", flush=True)


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open() as f:
        return json.load(f)


if __name__ == "__main__":
    main()
