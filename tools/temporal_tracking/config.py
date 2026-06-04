"""Configuration helpers for temporal vine tracking scripts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


@dataclass(frozen=True)
class SceneConfig:
    name: str
    source_path: Path
    colmap_path: Path
    selected_frame: str | None
    mask_path: Path | None
    model_path: Path | None = None
    gaussian_ply_path: Path | None = None

    @property
    def image_dir(self) -> Path:
        for dirname in ("images_rgb", "images"):
            candidate = self.source_path / dirname
            if candidate.is_dir():
                return candidate
        raise FileNotFoundError(
            f"{self.name}: no image folder found under {self.source_path}. "
            "Expected images_rgb/ or images/. Set source_path to a prepared scene root."
        )


@dataclass(frozen=True)
class TrackingConfig:
    reference_scene: str
    scenes: dict[str, SceneConfig]
    output_dir: Path

    @property
    def reference(self) -> SceneConfig:
        try:
            return self.scenes[self.reference_scene]
        except KeyError as exc:
            raise ValueError(
                f"reference_scene '{self.reference_scene}' is not one of: "
                f"{', '.join(sorted(self.scenes))}"
            ) from exc


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required to read temporal tracking YAML configs. "
            "Install pyyaml or provide a YAML-capable environment."
        ) from exc
    with path.open("r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} did not parse to a YAML mapping.")
    return data


def _path(value: str | None, config_dir: Path) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = (config_dir / path).resolve()
    return path


def normalize_colmap_path(path: Path) -> Path:
    if (path / "cameras.bin").exists() or (path / "cameras.txt").exists():
        return path
    sparse0 = path / "sparse" / "0"
    if sparse0.is_dir():
        return sparse0
    raise FileNotFoundError(
        f"COLMAP path {path} is neither a sparse/0 folder nor a scene root with sparse/0."
    )


def validate_colmap_path(path: Path, scene_name: str) -> None:
    cameras = (path / "cameras.bin").exists() or (path / "cameras.txt").exists()
    images = (path / "images.bin").exists() or (path / "images.txt").exists()
    points = (
        (path / "points3D.bin").exists()
        or (path / "points3D.txt").exists()
        or (path / "points3D.ply").exists()
    )
    missing = []
    if not cameras:
        missing.append("cameras.bin/txt")
    if not images:
        missing.append("images.bin/txt")
    if not points:
        missing.append("points3D.bin/txt/ply")
    if missing:
        raise FileNotFoundError(f"{scene_name}: missing COLMAP files in {path}: {missing}")


def load_config(config_path: str | Path, validate: bool = True) -> TrackingConfig:
    config_path = Path(config_path).expanduser().resolve()
    data = load_yaml(config_path)
    config_dir = config_path.parent

    reference_scene = data.get("reference_scene")
    if not reference_scene:
        raise ValueError("Config must define reference_scene.")

    output_dir = _path(data.get("output_dir"), config_dir)
    if output_dir is None:
        raise ValueError("Config must define output_dir.")

    raw_scenes = data.get("scenes")
    if not isinstance(raw_scenes, dict) or not raw_scenes:
        raise ValueError("Config must define a non-empty scenes mapping.")

    scenes: dict[str, SceneConfig] = {}
    for scene_name, raw_scene in raw_scenes.items():
        if not isinstance(raw_scene, dict):
            raise ValueError(f"{scene_name}: scene entry must be a mapping.")
        source_path = _path(raw_scene.get("source_path"), config_dir)
        colmap_path = _path(raw_scene.get("colmap_path"), config_dir)
        if source_path is None:
            raise ValueError(f"{scene_name}: source_path is required.")
        if colmap_path is None:
            raise ValueError(f"{scene_name}: colmap_path is required.")

        scene = SceneConfig(
            name=scene_name,
            source_path=source_path,
            colmap_path=normalize_colmap_path(colmap_path),
            selected_frame=raw_scene.get("selected_frame"),
            mask_path=_path(raw_scene.get("mask_path"), config_dir),
            model_path=_path(raw_scene.get("model_path"), config_dir),
            gaussian_ply_path=_path(raw_scene.get("gaussian_ply_path"), config_dir),
        )
        scenes[scene_name] = scene

    cfg = TrackingConfig(
        reference_scene=str(reference_scene),
        scenes=scenes,
        output_dir=output_dir,
    )

    if validate:
        validate_config(cfg)
    return cfg


def validate_config(cfg: TrackingConfig) -> None:
    _ = cfg.reference
    for scene in cfg.scenes.values():
        if not scene.source_path.is_dir():
            raise FileNotFoundError(f"{scene.name}: source_path does not exist: {scene.source_path}")
        image_dir = scene.image_dir
        if not any(p.suffix.lower() in IMAGE_SUFFIXES for p in image_dir.iterdir()):
            raise FileNotFoundError(f"{scene.name}: no image files found in {image_dir}")
        validate_colmap_path(scene.colmap_path, scene.name)
        if scene.mask_path is not None and not scene.mask_path.is_dir():
            raise FileNotFoundError(f"{scene.name}: mask_path does not exist: {scene.mask_path}")


def iter_image_files(image_dir: Path) -> list[Path]:
    return sorted(
        [p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES],
        key=lambda p: p.name,
    )


def resolve_selected_image(scene: SceneConfig) -> Path:
    image_dir = scene.image_dir
    images = iter_image_files(image_dir)
    if not images:
        raise FileNotFoundError(f"{scene.name}: no images found in {image_dir}")
    if scene.selected_frame is None:
        return images[0]

    requested = Path(scene.selected_frame)
    if requested.is_absolute() and requested.exists():
        return requested
    direct = image_dir / scene.selected_frame
    if direct.exists():
        return direct
    matches = [p for p in images if p.stem == requested.stem]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            f"{scene.name}: selected_frame '{scene.selected_frame}' was not found in {image_dir}."
        )
    raise ValueError(f"{scene.name}: selected_frame '{scene.selected_frame}' is ambiguous: {matches}")


def resolve_mask(scene: SceneConfig, image_path: Path) -> Path:
    if scene.mask_path is None:
        raise FileNotFoundError(
            f"{scene.name}: mask_path is not set. Expected a folder of 2D integer PNG masks "
            "with 0 as background and positive local object IDs."
        )
    candidate = scene.mask_path / f"{image_path.stem}.png"
    if candidate.exists():
        return candidate
    alternatives = sorted(scene.mask_path.glob(f"{image_path.stem}.*"))
    if alternatives:
        return alternatives[0]
    raise FileNotFoundError(
        f"{scene.name}: no mask found for {image_path.name}. Checked {candidate}. "
        "Expected mask format: 2D integer image/array, 0 background, positive IDs. "
        "Set mask_path in the config to the folder containing matching mask stems."
    )
