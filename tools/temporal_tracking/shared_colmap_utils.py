"""Shared helpers for temporal COLMAP pose preparation scripts."""

from __future__ import annotations

import importlib.util
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
COLMAP_CAMERA_FILES = ("cameras.bin", "cameras.txt")
COLMAP_IMAGE_FILES = ("images.bin", "images.txt")
COLMAP_POINT_FILES = ("points3D.ply", "points3D.bin", "points3D.txt")


@dataclass(frozen=True)
class TemporalScene:
    key: str
    date_prefix: str
    source_path: Path
    output_path: Path


DEFAULT_SCENES = (
    TemporalScene(
        key="vinyes_20260321",
        date_prefix="20260321",
        source_path=Path("data/vinyes_20260321"),
        output_path=Path("output/vinyes_20260321"),
    ),
    TemporalScene(
        key="vinyes_20260418_rgb_colmap_shared",
        date_prefix="20260418",
        source_path=Path("data/vinyes_20260418_rgb_colmap_shared"),
        output_path=Path("output/vinyes_20260418_rgb_colmap_shared"),
    ),
    TemporalScene(
        key="vinyes_20260509",
        date_prefix="20260509",
        source_path=Path("data/vinyes_20260509_pinhole"),
        output_path=Path("output/vinyes_20260509"),
    ),
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_repo_path(path: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path
    return (repo_root() / path).resolve()


def default_scenes() -> list[TemporalScene]:
    return [
        TemporalScene(
            key=scene.key,
            date_prefix=scene.date_prefix,
            source_path=resolve_repo_path(scene.source_path),
            output_path=resolve_repo_path(scene.output_path),
        )
        for scene in DEFAULT_SCENES
    ]


def iter_image_files(image_dir: Path) -> list[Path]:
    if not image_dir.is_dir():
        return []
    return sorted(
        [p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES],
        key=lambda p: p.name,
    )


def candidate_image_dirs(scene_root: Path) -> list[Path]:
    candidates = [
        scene_root / "images_rgb",
        scene_root / "frames_raw" / "rgb",
        scene_root / "frames_raw" / "RGB",
        scene_root / "images",
        scene_root / "input",
    ]
    return [path for path in candidates if path.is_dir()]


def choose_rgb_image_dir(scene_root: Path) -> Path | None:
    for image_dir in candidate_image_dirs(scene_root):
        if iter_image_files(image_dir):
            return image_dir
    return None


def has_colmap_cameras(path: Path) -> bool:
    return any((path / name).exists() for name in COLMAP_CAMERA_FILES)


def has_colmap_images(path: Path) -> bool:
    return any((path / name).exists() for name in COLMAP_IMAGE_FILES)


def has_colmap_points(path: Path) -> bool:
    return any((path / name).exists() for name in COLMAP_POINT_FILES)


def find_sparse_models(scene_root: Path, max_depth: int = 5) -> list[Path]:
    if not scene_root.exists():
        return []
    models = []
    for candidate in (scene_root / "sparse" / "0", scene_root / "distorted" / "sparse" / "0"):
        if candidate.is_dir() and (
            has_colmap_cameras(candidate) or has_colmap_images(candidate) or has_colmap_points(candidate)
        ):
            models.append(candidate)
    for path in scene_root.rglob("*"):
        if not path.is_dir():
            continue
        try:
            rel_depth = len(path.relative_to(scene_root).parts)
        except ValueError:
            continue
        if rel_depth > max_depth:
            continue
        if has_colmap_cameras(path) or has_colmap_images(path) or has_colmap_points(path):
            models.append(path)
    return sorted(set(models), key=lambda p: str(p))


def normalize_sparse_model(path: Path) -> Path:
    if has_colmap_cameras(path) or has_colmap_images(path):
        return path
    sparse0 = path / "sparse" / "0"
    if sparse0.is_dir():
        return sparse0
    return path


def import_colmap_loader():
    loader_path = repo_root() / "scene" / "colmap_loader.py"
    spec = importlib.util.spec_from_file_location("temporal_shared_colmap_loader", loader_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load COLMAP loader from {loader_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_cameras(model_dir: Path) -> dict:
    loader = import_colmap_loader()
    if (model_dir / "cameras.bin").exists():
        return loader.read_intrinsics_binary(str(model_dir / "cameras.bin"))
    if (model_dir / "cameras.txt").exists():
        return loader.read_intrinsics_text(str(model_dir / "cameras.txt"))
    raise FileNotFoundError(f"No cameras.bin/txt found in {model_dir}")


def read_images(model_dir: Path) -> dict:
    loader = import_colmap_loader()
    if (model_dir / "images.bin").exists():
        return loader.read_extrinsics_binary(str(model_dir / "images.bin"))
    if (model_dir / "images.txt").exists():
        return loader.read_extrinsics_text(str(model_dir / "images.txt"))
    raise FileNotFoundError(f"No images.bin/txt found in {model_dir}")


def count_registered_images(model_dir: Path) -> int:
    try:
        return len(read_images(model_dir))
    except FileNotFoundError:
        return 0


def format_float(value: float) -> str:
    return f"{float(value):.17g}"


def write_cameras_text(path: Path, cameras: dict) -> None:
    with path.open("w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {len(cameras)}\n")
        for camera_id in sorted(cameras):
            cam = cameras[camera_id]
            params = " ".join(format_float(v) for v in cam.params)
            f.write(f"{cam.id} {cam.model} {cam.width} {cam.height} {params}\n")


def write_images_text(path: Path, images: Iterable, name_overrides: dict[int, str] | None = None) -> None:
    images = list(images)
    name_overrides = name_overrides or {}
    with path.open("w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, IMAGE_NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(images)}\n")
        for image in sorted(images, key=lambda item: item.id):
            qvec = " ".join(format_float(v) for v in image.qvec)
            tvec = " ".join(format_float(v) for v in image.tvec)
            name = name_overrides.get(image.id, image.name)
            f.write(f"{image.id} {qvec} {tvec} {image.camera_id} {name}\n")
            triples = []
            for xy, point3d_id in zip(image.xys, image.point3D_ids):
                triples.extend((format_float(xy[0]), format_float(xy[1]), str(int(point3d_id))))
            f.write(" ".join(triples) + "\n")


def copy_best_points_file(source_model: Path, dest_model: Path) -> Path | None:
    for name in COLMAP_POINT_FILES:
        src = source_model / name
        if src.exists():
            dst = dest_model / name
            if src.resolve() != dst.resolve():
                shutil.copy2(src, dst)
            return dst
    empty = dest_model / "points3D.txt"
    empty.write_text("# Empty points3D file; shared reconstruction did not expose points.\n")
    return empty


def prefixed_image_name(date_prefix: str, image_name: str) -> str:
    return f"{date_prefix}__{image_name}"


def split_prefixed_image_name(image_name: str) -> tuple[str | None, str]:
    if "__" not in image_name:
        return None, image_name
    prefix, original = image_name.split("__", 1)
    return prefix, original
