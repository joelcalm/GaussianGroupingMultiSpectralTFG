#!/usr/bin/env python3
"""Exploratory RGB/multispectral representation separability analysis.

Modes:
  sam: uses SAM/object masks on RGB views as labels.
  predicted_pixel: uses model-predicted rendered object labels as pixel labels.
  predicted_gaussian: uses model-predicted object labels directly on Gaussians.

The predicted modes are representation/class-coherence analyses, not ground-truth
segmentation accuracy measurements.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import sys
import warnings
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


SCRIPT_PATH = Path(__file__).resolve()
if SCRIPT_PATH.parents[1].name == "jcalm":
    JCALM_ROOT = SCRIPT_PATH.parents[1]
    WORKSPACE_ROOT = JCALM_ROOT.parent
else:
    WORKSPACE_ROOT = SCRIPT_PATH.parents[1]
    JCALM_ROOT = WORKSPACE_ROOT / "jcalm"

DEFAULT_MODEL_PATH = JCALM_ROOT / "output" / "vinyes_sam3_vineid_200"
DEFAULT_SOURCE_PATH = WORKSPACE_ROOT / "vineyard_posematch" / "vinyes_sam3_vineid_200"
DEFAULT_OUTPUT_DIR = JCALM_ROOT / "outputs" / "multispectral_separability"
CHANNEL_NAMES = ["R", "G", "B", "b470", "b505", "b525", "b590", "b635", "b660", "b850"]
RGB_PREFIXES = ("rgb", "rgbp", "RGB")
FEATURE_SET_ALIASES = {
    "rgb": "RGB",
    "ms": "MS",
    "rgbms": "RGB_MS",
}


def add_jcalm_to_path() -> None:
    paths = [
        JCALM_ROOT,
        JCALM_ROOT / "submodules" / "diff-gaussian-rasterization",
        JCALM_ROOT / "submodules" / "simple-knn",
    ]
    for path in paths:
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--source_path", type=Path, default=DEFAULT_SOURCE_PATH)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--iteration", type=int, default=-1, help="-1 uses the largest saved iteration.")
    parser.add_argument("--resolution", type=int, default=None, help="Override cfg_args resolution.")
    parser.add_argument("--label_source", choices=["sam", "predicted_pixel", "predicted_gaussian"], default="sam")
    parser.add_argument(
        "--label_mode",
        choices=["semantic", "instance"],
        default="semantic",
        help="Use semantic classes or instance ids. Predicted modes remap instance-trained predictions to semantic classes by default.",
    )
    parser.add_argument("--view_split", choices=["train", "test", "all"], default="train", help="Camera split to render for pixel modes.")
    parser.add_argument("--train_split", action="store_true", help="When --view_split uses eval mode, split by images_train/ if it exists instead of LLFF holdout.")
    parser.add_argument("--llffhold", type=int, default=8, help="Index stride for eval test views when --train_split is not available.")
    parser.add_argument("--max_views", type=int, default=40, help="Evenly sample this many views from the selected split. <=0 uses all.")
    parser.add_argument("--samples_per_view_class", type=int, default=300, help="Pixels sampled per view/class. <=0 keeps all pixels for that view/class.")
    parser.add_argument("--max_samples_per_class", type=int, default=4000, help="Final per-label sample cap. <=0 disables this cap.")
    parser.add_argument("--max_gaussians", type=int, default=0, help="Total Gaussian sample cap after filtering/per-class cap. <=0 disables this cap.")
    parser.add_argument("--min_pred_confidence", type=float, default=0.0, help="Minimum classifier softmax confidence for predicted_gaussian labels.")
    parser.add_argument("--min_opacity", type=float, default=0.0, help="Minimum activated Gaussian opacity for predicted_gaussian samples.")
    parser.add_argument("--max_scale_percentile", type=float, default=0.0, help="If >0 and <100, drop Gaussians with max scale above this percentile after other filters.")
    parser.add_argument("--metric_max_samples", type=int, default=10000)
    parser.add_argument("--plot_max_samples", type=int, default=8000)
    parser.add_argument("--max-samples", type=int, default=None, help="Stratified sample cap for LDA/t-SNE/UMAP plots and projected-space metrics. Defaults to --plot_max_samples.")
    parser.add_argument("--test_size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--random-state", type=int, default=None, help="Alias for --seed used by the extra analyses.")
    parser.add_argument("--run-viewdep", dest="run_viewdep", action="store_true", default=True, help="Write the decoder view-dependence report. Enabled by default for compatibility.")
    parser.add_argument("--run-lda", action="store_true", help="Generate LDA plots and projected-space clustering metrics.")
    parser.add_argument("--run-qda", action="store_true", help="Run a QDA auxiliary classifier and save extra classifier metrics.")
    parser.add_argument("--run-tsne", action="store_true", help="Generate t-SNE plots and projected-space clustering metrics.")
    parser.add_argument("--run-umap", action="store_true", help="Generate UMAP plots and projected-space clustering metrics. Skips cleanly if umap-learn is unavailable.")
    parser.add_argument("--level", choices=["gaussian", "pixel", "all"], default="all", help="Level label used for extra output names. Current --label_source determines the collected samples.")
    parser.add_argument("--feature-set", choices=["rgb", "ms", "rgbms", "all"], default="all", help="Feature set filter for analysis outputs.")
    parser.add_argument("--ignore_labels", type=int, nargs="*", default=[0])
    knn_group = parser.add_mutually_exclusive_group()
    knn_group.add_argument("--include_knn", dest="include_knn", action="store_true")
    knn_group.add_argument("--no_include_knn", dest="include_knn", action="store_false")
    parser.set_defaults(include_knn=False)
    parser.add_argument("--signature_max_classes", type=int, default=12)
    args = parser.parse_args()
    if args.random_state is not None:
        args.seed = int(args.random_state)
    if args.max_samples is None:
        args.max_samples = args.plot_max_samples
    return args


def load_cfg_args(model_path: Path) -> argparse.Namespace:
    cfg_path = model_path / "cfg_args"
    if not cfg_path.exists():
        return argparse.Namespace()
    text = cfg_path.read_text()
    return eval(text, {"Namespace": argparse.Namespace})


def find_iteration(model_path: Path, requested: int) -> int:
    point_cloud_root = model_path / "point_cloud"
    if requested > 0:
        return requested
    iterations = []
    for path in point_cloud_root.glob("iteration_*"):
        try:
            iterations.append(int(path.name.split("_")[-1]))
        except ValueError:
            pass
    if not iterations:
        raise FileNotFoundError(f"No point_cloud/iteration_* directories found under {model_path}")
    return max(iterations)


def load_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text())


def label_names(source_path: Path, label_mode: str, num_classes: int, label_source: str) -> dict[int, str]:
    metadata = source_path / "metadata"
    class_map = load_json(metadata / "class_map.json", {})
    class_names = {int(v): str(k) for k, v in class_map.items()}
    instance_map = load_json(metadata / "instance_label_map.json", {})
    instance_names = {}
    for key, row in instance_map.items():
        label = row.get("label") or row.get("class_name") or str(key)
        instance_names[int(key)] = str(label)

    if label_mode == "semantic":
        return class_names

    # Predicted labels come from the trained classifier. For instance-trained
    # vineyard runs, num_classes is much larger than the semantic class map.
    if instance_names and max(instance_names.keys(), default=-1) >= num_classes - 1:
        return instance_names
    return class_names


def semantic_label_remap(source_path: Path, label_mode: str) -> dict[int, int] | None:
    if label_mode != "semantic":
        return None
    metadata = source_path / "metadata"
    instance_map = load_json(metadata / "instance_label_map.json", {})
    remap = {}
    for key, row in instance_map.items():
        if "class_id" in row:
            remap[int(key)] = int(row["class_id"])
    return remap or None


def remap_label_array(labels: np.ndarray, label_remap: dict[int, int] | None) -> np.ndarray:
    if not label_remap:
        return labels
    remapped = labels.copy()
    for src, dst in label_remap.items():
        remapped[labels == src] = dst
    return remapped


def make_load_args(user_args: argparse.Namespace, cfg: argparse.Namespace, label_mode: str) -> argparse.Namespace:
    source_path = str(user_args.source_path.resolve())
    object_path = "semantic_mask" if label_mode == "semantic" else "object_mask"
    resolution = user_args.resolution if user_args.resolution is not None else getattr(cfg, "resolution", 4)
    return argparse.Namespace(
        source_path=source_path,
        images=getattr(cfg, "images", "images"),
        eval=user_args.view_split in {"test", "all"},
        object_path=object_path,
        n_views=100,
        random_init=False,
        train_split=bool(user_args.train_split),
        llffhold=int(user_args.llffhold),
        white_background=getattr(cfg, "white_background", False),
        resolution=resolution,
        data_device=getattr(cfg, "data_device", "cuda"),
        num_channels=int(getattr(cfg, "num_channels", 10)),
        num_classes=int(getattr(cfg, "num_classes", 200)),
    )


def read_scene_infos(source_path: Path, load_args: argparse.Namespace):
    from scene.dataset_readers import readColmapSceneInfo

    return readColmapSceneInfo(
        str(source_path),
        load_args.images,
        load_args.eval,
        load_args.object_path,
        llffhold=load_args.llffhold,
        n_views=load_args.n_views,
        random_init=False,
        train_split=load_args.train_split,
    )


def evenly_limit(items: list, max_items: int) -> list:
    items = sorted(items, key=lambda cam: cam.image_name)
    if max_items and max_items > 0 and len(items) > max_items:
        indices = np.linspace(0, len(items) - 1, max_items).round().astype(int)
        items = [items[int(i)] for i in indices]
    return items


def split_cam_infos(scene_info, view_split: str) -> list:
    if view_split == "train":
        return list(scene_info.train_cameras)
    if view_split == "test":
        return list(scene_info.test_cameras)
    if view_split == "all":
        return list(scene_info.train_cameras) + list(scene_info.test_cameras)
    raise ValueError(f"Unsupported view_split: {view_split}")


def select_sam_cam_infos(scene_info, view_split: str, max_views: int):
    rgb_infos = [
        cam
        for cam in split_cam_infos(scene_info, view_split)
        if cam.image_name.startswith(RGB_PREFIXES) and cam.objects is not None
    ]
    return evenly_limit(rgb_infos, max_views)


def select_predicted_cam_infos(scene_info, view_split: str, max_views: int):
    # Includes RGB and registered multispectral band frames. Labels will come
    # from the model, not from cam.objects.
    return evenly_limit(split_cam_infos(scene_info, view_split), max_views)


def build_cameras(cam_infos, load_args: argparse.Namespace):
    from utils.camera_utils import cameraList_from_camInfos

    return cameraList_from_camInfos(
        cam_infos,
        resolution_scale=1.0,
        args=load_args,
        single_channel_mode=False,
        num_channels=load_args.num_channels,
        object_id_mapping=None,
    )


def load_model(model_path: Path, cfg: argparse.Namespace, iteration: int):
    from gaussian_renderer import GaussianModel
    from utils.color_decoder import ColorDecoder

    num_channels = int(getattr(cfg, "num_channels", 10))
    color_embed_dim = int(getattr(cfg, "color_embed_dim", 32))
    hidden_dim = int(getattr(cfg, "color_decoder_hidden_dim", 128))
    hidden_layers = int(getattr(cfg, "color_decoder_num_hidden_layers", 3))
    num_objects = int(getattr(cfg, "num_objects", 16))
    num_classes = int(getattr(cfg, "num_classes", 200))
    sh_degree = int(getattr(cfg, "sh_degree", 0))

    iter_dir = model_path / "point_cloud" / f"iteration_{iteration}"
    ply_path = iter_dir / "point_cloud.ply"
    decoder_path = iter_dir / "color_decoder.pth"
    classifier_path = iter_dir / "classifier.pth"
    if not ply_path.exists():
        raise FileNotFoundError(f"Missing Gaussian PLY: {ply_path}")
    if not decoder_path.exists():
        raise FileNotFoundError(f"Missing ColorDecoder weights: {decoder_path}")
    if not classifier_path.exists():
        raise FileNotFoundError(f"Missing classifier weights: {classifier_path}")

    gaussians = GaussianModel(
        sh_degree,
        num_objects=num_objects,
        use_color_embed=True,
        color_embed_dim=color_embed_dim,
    )
    gaussians.load_ply(str(ply_path))

    color_decoder = ColorDecoder(
        input_dim=color_embed_dim,
        hidden_dim=hidden_dim,
        output_dim=num_channels,
        num_hidden_layers=hidden_layers,
    ).cuda()
    color_decoder.load_state_dict(torch.load(decoder_path, map_location="cuda"))
    color_decoder.eval()

    classifier = torch.nn.Conv2d(num_objects, num_classes, kernel_size=1).cuda()
    classifier.load_state_dict(torch.load(classifier_path, map_location="cuda"))
    classifier.eval()
    return gaussians, color_decoder, classifier


def decoder_view_dependence_report(color_decoder, gaussians) -> dict:
    """Describe whether Gaussian color decoding can depend on view direction."""
    forward_sig = inspect.signature(color_decoder.forward)
    forward_inputs = [
        name
        for name, param in forward_sig.parameters.items()
        if name != "self"
        and param.kind
        in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }
    ]
    view_like_inputs = [
        name
        for name in forward_inputs
        if any(token in name.lower() for token in ("view", "dir", "ray", "camera", "campos", "center"))
    ]
    view_dependent = bool(view_like_inputs)

    sample_count = int(min(1024, gaussians.get_color_embedding.shape[0]))
    with torch.no_grad():
        sample = gaussians.get_color_embedding[:sample_count]
        decoded_a = color_decoder(sample)
        decoded_b = color_decoder(sample)
        repeat_delta = torch.abs(decoded_a - decoded_b)

    return {
        "decoder_class": color_decoder.__class__.__name__,
        "decoder_forward_signature": str(forward_sig),
        "decoder_forward_inputs": forward_inputs,
        "view_like_forward_inputs": view_like_inputs,
        "decoder_is_view_dependent": view_dependent,
        "existing_pg_analysis_view_direction": None,
        "existing_pg_analysis_view_direction_description": (
            "None. Gaussian-level pg_ features are decoded as color_decoder(gaussians.get_color_embedding); "
            "no camera center, ray, or view direction is passed to the MLP."
        ),
        "viewpoint_change_check": {
            "method": (
                "Architecture/input check plus repeated decode on the same embeddings. Because the current "
                "decoder has no view-direction input, changing viewpoint cannot change decoded RGB/MS values."
            ),
            "num_sampled_gaussians": sample_count,
            "max_abs_change_across_viewpoints": 0.0,
            "mean_abs_change_across_viewpoints": 0.0,
            "max_abs_repeat_decode_delta": float(repeat_delta.max().detach().cpu().item()) if sample_count else 0.0,
            "mean_abs_repeat_decode_delta": float(repeat_delta.mean().detach().cpu().item()) if sample_count else 0.0,
            "changes_significantly_with_viewpoint": False,
        },
        "renderer_note": (
            "The renderer also uses color_decoder(pc.get_color_embedding) when use_color_embed is enabled. "
            "View direction is only used in the SH branch, where directions are computed from each Gaussian "
            "position to the active camera center."
        ),
    }


def save_decoder_view_dependence_report(out_dir: Path, report: dict) -> None:
    json_path = out_dir / "decoder_view_dependence.json"
    txt_path = out_dir / "decoder_view_dependence.txt"
    json_path.write_text(json.dumps(report, indent=2))
    lines = [
        "Decoder view-dependence report",
        "==============================",
        f"decoder_class: {report['decoder_class']}",
        f"decoder_forward_signature: {report['decoder_forward_signature']}",
        f"decoder_is_view_dependent: {report['decoder_is_view_dependent']}",
        f"existing_pg_analysis_view_direction: {report['existing_pg_analysis_view_direction_description']}",
        (
            "viewpoint_change_check: "
            f"max_abs_change={report['viewpoint_change_check']['max_abs_change_across_viewpoints']}, "
            f"mean_abs_change={report['viewpoint_change_check']['mean_abs_change_across_viewpoints']}, "
            f"significant={report['viewpoint_change_check']['changes_significantly_with_viewpoint']}"
        ),
        f"renderer_note: {report['renderer_note']}",
    ]
    txt_path.write_text("\n".join(lines) + "\n")


def cap_per_class(x: np.ndarray, y: np.ndarray, groups: np.ndarray | None, max_samples_per_class: int, seed: int):
    rng = np.random.default_rng(seed)
    keep_indices = []
    for class_id in np.unique(y):
        idx = np.flatnonzero(y == class_id)
        if max_samples_per_class and max_samples_per_class > 0 and idx.size > max_samples_per_class:
            idx = rng.choice(idx, size=max_samples_per_class, replace=False)
        keep_indices.append(idx)
    keep = np.concatenate(keep_indices)
    rng.shuffle(keep)
    if groups is None:
        return x[keep], y[keep], None
    return x[keep], y[keep], groups[keep]


def cap_total(samples: dict[str, np.ndarray], y: np.ndarray, max_count: int, seed: int):
    if not max_count or max_count <= 0 or y.shape[0] <= max_count:
        return samples, y
    rng = np.random.default_rng(seed)
    keep = rng.choice(y.shape[0], size=max_count, replace=False)
    return {name: values[keep] for name, values in samples.items()}, y[keep]


def subset_samples(samples: dict[str, np.ndarray], keep: np.ndarray, groups: np.ndarray | None):
    subset = {name: values[keep] for name, values in samples.items()}
    if groups is None:
        return subset, None
    return subset, groups[keep]


def filter_split_ready_samples(
    samples: dict[str, np.ndarray],
    y: np.ndarray,
    groups: np.ndarray | None,
    test_size: float,
):
    notes = {}
    dropped = []
    keep = np.ones(y.shape[0], dtype=bool)

    while True:
        yy = y[keep]
        counts = Counter(yy.tolist())
        sparse = [label for label, count in counts.items() if count < 2]
        if sparse:
            dropped.extend(sparse)
            keep &= ~np.isin(y, sparse)
            continue
        n = int(yy.shape[0])
        n_classes = len(counts)
        if n_classes < 2:
            raise RuntimeError("Need at least two labels with enough samples for separability analysis.")
        test_count = max(int(math.ceil(n * test_size)), n_classes)
        train_count = n - test_count
        if train_count >= n_classes:
            break
        # Stratified splitting needs at least one train and one test sample per class.
        label_to_drop = min(counts.items(), key=lambda item: (item[1], item[0]))[0]
        dropped.append(label_to_drop)
        keep &= y != label_to_drop

    if dropped:
        notes["dropped_labels_for_probe_split"] = sorted({int(label) for label in dropped})
    samples, groups = subset_samples(samples, keep, groups)
    return samples, y[keep], groups, notes


def stratified_test_count(y: np.ndarray, test_size: float):
    n = int(y.shape[0])
    n_classes = int(np.unique(y).size)
    if n_classes <= 1 or min(Counter(y).values()) < 2:
        return None
    test_count = max(int(math.ceil(n * test_size)), n_classes)
    if n - test_count < n_classes:
        return None
    return test_count


def pick_coords(coords: np.ndarray, samples_per_view_class: int, rng: np.random.Generator) -> np.ndarray:
    if samples_per_view_class and samples_per_view_class > 0:
        take = min(samples_per_view_class, coords.shape[0])
        return coords[rng.choice(coords.shape[0], size=take, replace=False)]
    return coords


def collect_sam_pixel_samples(
    cameras,
    gaussians,
    color_decoder,
    num_channels: int,
    samples_per_view_class: int,
    max_samples_per_class: int,
    ignore_labels: set[int],
    seed: int,
):
    from gaussian_renderer import render

    rng = np.random.default_rng(seed)
    background = torch.zeros(num_channels, dtype=torch.float32, device="cuda")
    pipe = SimpleNamespace(debug=False, convert_SHs_python=False, compute_cov3D_python=False)
    xs, ys, groups = [], [], []

    for view_idx, view in enumerate(cameras):
        if view.objects is None:
            continue
        with torch.no_grad():
            pkg = render(view, gaussians, pipe, background, color_decoder=color_decoder)
            rendered = torch.clamp(pkg["render"][:num_channels], 0.0, 1.0).detach().cpu().numpy()

        labels = view.objects.detach().cpu().numpy().astype(np.int64)
        features = np.moveaxis(rendered, 0, -1)
        valid = ~np.isin(labels, list(ignore_labels))
        for class_id in np.unique(labels[valid]):
            coords = np.argwhere(valid & (labels == class_id))
            if coords.size == 0:
                continue
            picked = pick_coords(coords, samples_per_view_class, rng)
            take = picked.shape[0]
            xs.append(features[picked[:, 0], picked[:, 1], :])
            ys.append(np.full(take, int(class_id), dtype=np.int64))
            groups.append(np.full(take, view_idx, dtype=np.int64))

    if not xs:
        raise RuntimeError("No SAM-labeled samples were collected. Check RGB masks and label_mode.")
    x = np.concatenate(xs, axis=0).astype(np.float32)
    y = np.concatenate(ys, axis=0)
    group = np.concatenate(groups, axis=0)
    x, y, group = cap_per_class(x, y, group, max_samples_per_class, seed)
    return {"SPECTRAL10": x}, y, group, {}


def collect_predicted_pixel_samples(
    cameras,
    gaussians,
    color_decoder,
    classifier,
    num_channels: int,
    samples_per_view_class: int,
    max_samples_per_class: int,
    ignore_labels: set[int],
    seed: int,
    label_remap: dict[int, int] | None = None,
):
    from gaussian_renderer import render

    rng = np.random.default_rng(seed)
    background = torch.zeros(num_channels, dtype=torch.float32, device="cuda")
    pipe = SimpleNamespace(debug=False, convert_SHs_python=False, compute_cov3D_python=False)
    xs, ys, groups = [], [], []

    for view_idx, view in enumerate(cameras):
        print(f"[render] view {view_idx + 1}/{len(cameras)}: {view.image_name}", flush=True)
        with torch.no_grad():
            pkg = render(view, gaussians, pipe, background, color_decoder=color_decoder)
            rendered = torch.clamp(pkg["render"][:num_channels], 0.0, 1.0)
            logits = classifier(pkg["render_object"])
            labels = torch.argmax(logits, dim=0).detach().cpu().numpy().astype(np.int64)
            labels = remap_label_array(labels, label_remap)
            features = np.moveaxis(rendered.detach().cpu().numpy(), 0, -1)

        valid = ~np.isin(labels, list(ignore_labels))
        for class_id in np.unique(labels[valid]):
            coords = np.argwhere(valid & (labels == class_id))
            if coords.size == 0:
                continue
            picked = pick_coords(coords, samples_per_view_class, rng)
            take = picked.shape[0]
            xs.append(features[picked[:, 0], picked[:, 1], :])
            ys.append(np.full(take, int(class_id), dtype=np.int64))
            groups.append(np.full(take, view_idx, dtype=np.int64))

    if not xs:
        raise RuntimeError("No predicted-pixel samples were collected. Check ignore_labels and rendered views.")
    x = np.concatenate(xs, axis=0).astype(np.float32)
    y = np.concatenate(ys, axis=0)
    group = np.concatenate(groups, axis=0)
    x, y, group = cap_per_class(x, y, group, max_samples_per_class, seed)
    notes = {
        "pixel_embedding_features": "COLOR_EMBEDDING is skipped in predicted_pixel mode; the current renderer only rasterizes colors and object features directly.",
    }
    return {"SPECTRAL10": x}, y, group, notes


def gaussian_object_logits(gaussians, classifier):
    objects = gaussians.get_objects
    if objects.dim() == 3:
        if objects.shape[1] == 1:
            classifier_input = objects.permute(2, 0, 1)  # [num_objects, N, 1]
        elif objects.shape[2] == 1:
            classifier_input = objects.permute(1, 0, 2)  # [num_objects, N, 1]
        else:
            raise ValueError(f"Unsupported object tensor shape: {tuple(objects.shape)}")
    elif objects.dim() == 2:
        classifier_input = objects.t().unsqueeze(-1)
    else:
        raise ValueError(f"Unsupported object tensor rank: {objects.dim()}")
    logits = classifier(classifier_input)
    if logits.dim() == 3:
        logits = logits.squeeze(-1).transpose(0, 1)  # [N, num_classes]
    elif logits.dim() == 2:
        logits = logits.transpose(0, 1)
    else:
        raise ValueError(f"Unsupported classifier output shape: {tuple(logits.shape)}")
    return logits


def collect_predicted_gaussian_samples(
    gaussians,
    color_decoder,
    classifier,
    max_samples_per_class: int,
    max_gaussians: int,
    min_pred_confidence: float,
    min_opacity: float,
    max_scale_percentile: float,
    ignore_labels: set[int],
    seed: int,
    label_remap: dict[int, int] | None = None,
):
    with torch.no_grad():
        embedding = gaussians.get_color_embedding
        spectral10 = color_decoder(embedding)
        logits = gaussian_object_logits(gaussians, classifier)
        probs = torch.softmax(logits, dim=1)
        confidence, labels = torch.max(probs, dim=1)
        opacity = gaussians.get_opacity.squeeze(-1)
        max_scale = gaussians.get_scaling.max(dim=1).values

    y = labels.detach().cpu().numpy().astype(np.int64)
    y = remap_label_array(y, label_remap)
    conf = confidence.detach().cpu().numpy().astype(np.float32)
    opacity_np = opacity.detach().cpu().numpy().astype(np.float32)
    max_scale_np = max_scale.detach().cpu().numpy().astype(np.float32)
    valid = ~np.isin(y, list(ignore_labels))
    if min_pred_confidence > 0:
        valid &= conf >= float(min_pred_confidence)
    if min_opacity > 0:
        valid &= opacity_np >= float(min_opacity)
    scale_threshold = None
    if 0.0 < max_scale_percentile < 100.0 and np.any(valid):
        scale_threshold = float(np.percentile(max_scale_np[valid], max_scale_percentile))
        valid &= max_scale_np <= scale_threshold
    if not np.any(valid):
        raise RuntimeError("No predicted-Gaussian samples remain after label/confidence/opacity/scale filtering.")

    samples = {
        "SPECTRAL10": spectral10.detach().cpu().numpy().astype(np.float32)[valid],
        "COLOR_EMBEDDING": embedding.detach().cpu().numpy().astype(np.float32)[valid],
    }
    y = y[valid]

    stacked = np.concatenate([samples["SPECTRAL10"], samples["COLOR_EMBEDDING"]], axis=1)
    stacked, y, _ = cap_per_class(stacked, y, None, max_samples_per_class, seed)
    n0 = samples["SPECTRAL10"].shape[1]
    samples = {
        "SPECTRAL10": stacked[:, :n0],
        "COLOR_EMBEDDING": stacked[:, n0:],
    }
    samples, y = cap_total(samples, y, max_gaussians, seed)
    notes = {
        "min_pred_confidence": min_pred_confidence,
        "min_opacity": min_opacity,
        "max_scale_percentile": max_scale_percentile,
        "max_scale_threshold": scale_threshold,
        "n_gaussians_before_filter": int(labels.shape[0]),
        "n_gaussians_after_filter_before_caps": int(np.count_nonzero(valid)),
    }
    return samples, y, None, notes



def build_feature_sets(samples: dict[str, np.ndarray], label_source: str) -> dict[str, np.ndarray]:
    x10 = samples["SPECTRAL10"]
    sets = {
        "RGB": x10[:, 0:3],
        "MS": x10[:, 3:10],
        "RGB_MS": x10[:, 0:10],
    }
    if label_source == "predicted_gaussian":
        for name in ["COLOR_EMBEDDING"]:
            if name in samples:
                sets[name] = samples[name]
    return sets




def feature_slug(name: str) -> str:
    return name.lower().replace("_", "")


def selected_feature_sets(feature_sets: dict[str, np.ndarray], feature_set: str, *, include_non_spectral: bool) -> dict[str, np.ndarray]:
    if feature_set == "all":
        if include_non_spectral:
            return dict(feature_sets)
        return {name: x for name, x in feature_sets.items() if name in FEATURE_SET_ALIASES.values()}
    wanted = FEATURE_SET_ALIASES[feature_set]
    return {wanted: feature_sets[wanted]} if wanted in feature_sets else {}


def level_prefix_for_run(label_source: str, requested_level: str) -> str:
    actual_level = "gaussian" if label_source == "predicted_gaussian" else "pixel"
    if requested_level != "all" and requested_level != actual_level:
        warnings.warn(
            f"--level={requested_level} was requested, but --label_source={label_source} collected {actual_level}-level samples; "
            f"using the {actual_level} prefix for outputs.",
            RuntimeWarning,
        )
    return "pg" if actual_level == "gaussian" else "pp"


def grouped_split(y: np.ndarray, groups: np.ndarray, test_size: float, seed: int):
    from sklearn.model_selection import GroupShuffleSplit, train_test_split

    unique_groups = np.unique(groups)
    if unique_groups.size >= 2:
        splitter = GroupShuffleSplit(n_splits=50, test_size=test_size, random_state=seed)
        all_classes = set(np.unique(y).tolist())
        best = None
        best_score = -1
        for train_idx, test_idx in splitter.split(np.zeros_like(y), y, groups):
            train_classes = set(np.unique(y[train_idx]).tolist())
            test_classes = set(np.unique(y[test_idx]).tolist())
            score = len(train_classes & test_classes)
            if train_classes == all_classes and score > best_score:
                best = (train_idx, test_idx)
                best_score = score
            if train_classes == all_classes and test_classes == all_classes:
                return train_idx, test_idx, "group_by_view"
        if best is not None:
            return best[0], best[1], "group_by_view_partial_classes"

    test_count = stratified_test_count(y, test_size)
    train_idx, test_idx = train_test_split(
        np.arange(y.shape[0]),
        test_size=test_count if test_count is not None else test_size,
        random_state=seed,
        stratify=y if test_count is not None else None,
    )
    split_name = "random_pixel_stratified_fallback" if test_count is not None else "random_pixel_fallback"
    return train_idx, test_idx, split_name


def random_gaussian_split(y: np.ndarray, test_size: float, seed: int):
    from sklearn.model_selection import train_test_split

    test_count = stratified_test_count(y, test_size)
    train_idx, test_idx = train_test_split(
        np.arange(y.shape[0]),
        test_size=test_count if test_count is not None else test_size,
        random_state=seed,
        stratify=y if test_count is not None else None,
    )
    return train_idx, test_idx, "random_gaussian_stratified" if test_count is not None else "random_gaussian_fallback"


def sample_indices(n: int, max_n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if max_n <= 0 or n <= max_n:
        return np.arange(n)
    return rng.choice(n, size=max_n, replace=False)


def separability_metrics(x: np.ndarray, y: np.ndarray, max_samples: int, seed: int) -> dict[str, float]:
    from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
    from sklearn.preprocessing import StandardScaler

    idx = sample_indices(x.shape[0], max_samples, seed)
    xs = StandardScaler().fit_transform(x[idx])
    ys = y[idx]
    if np.unique(ys).size < 2 or xs.shape[0] <= np.unique(ys).size:
        return {"silhouette": math.nan, "davies_bouldin": math.nan, "calinski_harabasz": math.nan}
    return {
        "silhouette": float(silhouette_score(xs, ys)),
        "davies_bouldin": float(davies_bouldin_score(xs, ys)),
        "calinski_harabasz": float(calinski_harabasz_score(xs, ys)),
    }




def stratified_sample_indices(y: np.ndarray, max_samples: int, seed: int) -> np.ndarray:
    if max_samples <= 0 or y.shape[0] <= max_samples:
        return np.arange(y.shape[0])

    rng = np.random.default_rng(seed)
    labels, counts = np.unique(y, return_counts=True)
    target_total = max(int(max_samples), int(labels.size))
    raw = counts.astype(np.float64) * (target_total / float(y.shape[0]))
    takes = np.minimum(counts, np.maximum(1, np.floor(raw).astype(int)))

    while takes.sum() > target_total:
        candidates = np.flatnonzero(takes > 1)
        if candidates.size == 0:
            break
        fractions = raw[candidates] - np.floor(raw[candidates])
        drop_pos = candidates[np.argmin(fractions)]
        takes[drop_pos] -= 1

    while takes.sum() < target_total:
        candidates = np.flatnonzero(takes < counts)
        if candidates.size == 0:
            break
        deficits = raw[candidates] - takes[candidates]
        add_pos = candidates[np.argmax(deficits)]
        takes[add_pos] += 1

    keep = []
    for label, take in zip(labels, takes):
        idx = np.flatnonzero(y == label)
        keep.append(rng.choice(idx, size=int(take), replace=False))
    keep = np.concatenate(keep)
    rng.shuffle(keep)
    return keep


def projected_space_metrics(coords: np.ndarray, y: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

    labels = np.unique(y)
    if labels.size < 2 or coords.shape[0] <= labels.size:
        return {"silhouette": math.nan, "davies_bouldin": math.nan, "calinski_harabasz": math.nan}
    try:
        return {
            "silhouette": float(silhouette_score(coords, y)),
            "davies_bouldin": float(davies_bouldin_score(coords, y)),
            "calinski_harabasz": float(calinski_harabasz_score(coords, y)),
        }
    except ValueError as exc:
        warnings.warn(f"Skipping projected-space clustering metrics: {exc}", RuntimeWarning)
        return {"silhouette": math.nan, "davies_bouldin": math.nan, "calinski_harabasz": math.nan}


def lda_projection(x: np.ndarray, y: np.ndarray, max_samples: int, seed: int):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.preprocessing import StandardScaler

    idx = stratified_sample_indices(y, max_samples, seed)
    xs = StandardScaler().fit_transform(x[idx])
    ys = y[idx]
    n_classes = np.unique(ys).size
    n_components = min(2, xs.shape[1], n_classes - 1)
    if n_components < 1:
        return None, idx, "LDA requires at least two classes."
    try:
        coords = LinearDiscriminantAnalysis(n_components=n_components).fit_transform(xs, ys)
    except ValueError as exc:
        return None, idx, str(exc)
    if coords.shape[1] == 1:
        coords = np.concatenate([coords, np.zeros((coords.shape[0], 1), dtype=coords.dtype)], axis=1)
    return coords, idx, ""


def tsne_projection(x: np.ndarray, y: np.ndarray, max_samples: int, seed: int):
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    idx = stratified_sample_indices(y, max_samples, seed)
    xs = StandardScaler().fit_transform(x[idx])
    if xs.shape[0] < 4:
        return None, idx, "t-SNE requires at least four samples."
    perplexity = min(30.0, max(2.0, (xs.shape[0] - 1) / 3.0))
    kwargs = {
        "n_components": 2,
        "perplexity": perplexity,
        "init": "pca" if xs.shape[1] >= 2 else "random",
        "learning_rate": 200.0,
        "random_state": seed,
    }
    if "max_iter" in inspect.signature(TSNE).parameters:
        kwargs["max_iter"] = 1000
    else:
        kwargs["n_iter"] = 1000
    try:
        coords = TSNE(**kwargs).fit_transform(xs)
    except (TypeError, ValueError) as exc:
        return None, idx, str(exc)
    return coords, idx, ""


def umap_projection(x: np.ndarray, y: np.ndarray, max_samples: int, seed: int):
    from sklearn.preprocessing import StandardScaler

    try:
        import umap
    except ImportError:
        message = "umap-learn is not installed; skipping UMAP."
        warnings.warn(message, RuntimeWarning)
        return None, np.array([], dtype=np.int64), message

    idx = stratified_sample_indices(y, max_samples, seed)
    xs = StandardScaler().fit_transform(x[idx])
    if xs.shape[0] < 3:
        return None, idx, "UMAP requires at least three samples."
    n_neighbors = min(15, xs.shape[0] - 1)
    try:
        coords = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=0.1,
            metric="euclidean",
            random_state=seed,
        ).fit_transform(xs)
    except ValueError as exc:
        return None, idx, str(exc)
    return coords, idx, ""


def save_projection_plot(
    out_dir: Path,
    filename: str,
    title: str,
    coords: np.ndarray,
    y: np.ndarray,
    names: dict[int, str],
) -> None:
    import matplotlib.pyplot as plt

    labels = np.array(sorted(np.unique(y)))
    cmap = plt.get_cmap("tab20", max(len(labels), 1))
    fig, ax = plt.subplots(figsize=(8, 6))
    for pos, label in enumerate(labels):
        mask = y == label
        ax.scatter(coords[mask, 0], coords[mask, 1], s=5, alpha=0.45, color=cmap(pos), label=names.get(int(label), str(label)))
    ax.set_title(title)
    ax.set_xlabel("component 1")
    ax.set_ylabel("component 2")
    if len(labels) <= 16:
        ax.legend(markerscale=3, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=180)
    plt.close(fig)


def qda_probe(x: np.ndarray, y: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray, seed: int) -> dict[str, float | str]:
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    train_counts = Counter(y[train_idx].tolist())
    if min(train_counts.values(), default=0) < 2:
        return {
            "accuracy": math.nan,
            "macro_f1": math.nan,
            "weighted_f1": math.nan,
            "note": "QDA skipped because at least one train class has fewer than two samples.",
        }

    clf = make_pipeline(StandardScaler(), QuadraticDiscriminantAnalysis(reg_param=0.01))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(x[train_idx], y[train_idx])
        pred = clf.predict(x[test_idx])
    except ValueError as exc:
        return {"accuracy": math.nan, "macro_f1": math.nan, "weighted_f1": math.nan, "note": str(exc)}
    labels = np.array(sorted(np.unique(y)))
    return {
        "accuracy": float(accuracy_score(y[test_idx], pred)),
        "macro_f1": float(f1_score(y[test_idx], pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y[test_idx], pred, labels=labels, average="weighted", zero_division=0)),
        "note": "",
    }


def classifier_probe(x: np.ndarray, y: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray, seed: int):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
            n_jobs=None,
            random_state=seed,
        ),
    )
    clf.fit(x[train_idx], y[train_idx])
    pred = clf.predict(x[test_idx])
    labels = np.array(sorted(np.unique(y)))
    return {
        "accuracy": float(accuracy_score(y[test_idx], pred)),
        "balanced_accuracy": float(recall_score(y[test_idx], pred, labels=labels, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y[test_idx], pred, labels=labels, average="macro", zero_division=0)),
        "confusion": confusion_matrix(y[test_idx], pred, labels=labels),
        "labels": labels,
    }


def knn_probe(x: np.ndarray, y: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray):
    from sklearn.metrics import accuracy_score, f1_score, recall_score
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    clf = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7, weights="distance"))
    clf.fit(x[train_idx], y[train_idx])
    pred = clf.predict(x[test_idx])
    labels = np.array(sorted(np.unique(y)))
    return {
        "knn_accuracy": float(accuracy_score(y[test_idx], pred)),
        "knn_balanced_accuracy": float(recall_score(y[test_idx], pred, labels=labels, average="macro", zero_division=0)),
        "knn_macro_f1": float(f1_score(y[test_idx], pred, labels=labels, average="macro", zero_division=0)),
    }


def save_class_counts(out_dir: Path, y: np.ndarray, names: dict[int, str]) -> None:
    counts = Counter(y.tolist())
    with (out_dir / "class_counts.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label_id", "label_name", "count"])
        for label, count in sorted(counts.items()):
            writer.writerow([int(label), names.get(int(label), str(label)), int(count)])


def save_confusion_matrix(out_dir: Path, name: str, cm: np.ndarray, labels: np.ndarray, names: dict[int, str]) -> None:
    import matplotlib.pyplot as plt

    csv_path = out_dir / f"confusion_{name}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        header = ["true\\pred"] + [names.get(int(label), str(label)) for label in labels]
        writer.writerow(header)
        for label, row in zip(labels, cm):
            writer.writerow([names.get(int(label), str(label)), *row.tolist()])

    row_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, np.maximum(row_sum, 1), where=row_sum >= 0)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_norm, cmap="viridis", vmin=0.0, vmax=1.0)
    tick_names = [names.get(int(label), str(label)) for label in labels]
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(tick_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(tick_names, fontsize=8)
    ax.set_xlabel("Predicted by probe")
    ax.set_ylabel("Reference label")
    ax.set_title(f"{name} confusion matrix (row-normalized)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_dir / f"confusion_{name}.png", dpi=180)
    plt.close(fig)


def save_pca_plot(out_dir: Path, name: str, x: np.ndarray, y: np.ndarray, names: dict[int, str], max_samples: int, seed: int) -> None:
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    idx = sample_indices(x.shape[0], max_samples, seed)
    xs = StandardScaler().fit_transform(x[idx])
    ys = y[idx]
    coords = PCA(n_components=2, random_state=seed).fit_transform(xs)

    labels = np.array(sorted(np.unique(ys)))
    cmap = plt.get_cmap("tab20", max(len(labels), 1))
    fig, ax = plt.subplots(figsize=(8, 6))
    for pos, label in enumerate(labels):
        mask = ys == label
        ax.scatter(coords[mask, 0], coords[mask, 1], s=5, alpha=0.45, color=cmap(pos), label=names.get(int(label), str(label)))
    ax.set_title(f"{name} PCA")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    if len(labels) <= 16:
        ax.legend(markerscale=3, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / f"pca_{name}.png", dpi=180)
    plt.close(fig)


def save_spectral_signatures(out_dir: Path, x10: np.ndarray, y: np.ndarray, names: dict[int, str], max_classes: int) -> None:
    import matplotlib.pyplot as plt

    counts = Counter(y.tolist())
    labels = [label for label, _ in counts.most_common(max_classes)]

    with (out_dir / "spectral_signatures.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label_id", "label_name", "count", *[f"mean_{c}" for c in CHANNEL_NAMES], *[f"std_{c}" for c in CHANNEL_NAMES]])
        for label in sorted(np.unique(y)):
            vals = x10[y == label]
            writer.writerow([
                int(label),
                names.get(int(label), str(label)),
                int(vals.shape[0]),
                *vals.mean(axis=0).tolist(),
                *vals.std(axis=0).tolist(),
            ])

    fig, ax = plt.subplots(figsize=(9, 5))
    xs = np.arange(len(CHANNEL_NAMES))
    for label in labels:
        vals = x10[y == label]
        ax.plot(xs, vals.mean(axis=0), marker="o", linewidth=1.5, label=names.get(int(label), str(label)))
    ax.set_xticks(xs)
    ax.set_xticklabels(CHANNEL_NAMES, rotation=45, ha="right")
    ax.set_ylabel("Predicted channel value")
    ax.set_title("Per-class average predicted spectral signature")
    ax.legend(fontsize=8, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "spectral_signatures.png", dpi=180)
    plt.close(fig)


def write_results(out_dir: Path, rows: list[dict]) -> None:
    fieldnames = [
        "feature_set",
        "n_samples",
        "n_classes",
        "split",
        "silhouette",
        "davies_bouldin",
        "calinski_harabasz",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "knn_accuracy",
        "knn_balanced_accuracy",
        "knn_macro_f1",
    ]
    with (out_dir / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})




def write_extra_projection_metrics(out_dir: Path, level_prefix: str, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = [
        "level",
        "feature_set",
        "projection",
        "n_samples",
        "n_classes",
        "silhouette",
        "davies_bouldin",
        "calinski_harabasz",
        "note",
    ]
    with (out_dir / f"separability_extra_metrics_{level_prefix}.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_extra_classifier_metrics(out_dir: Path, level_prefix: str, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = [
        "level",
        "feature_set",
        "classifier",
        "n_train",
        "n_test",
        "n_classes",
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "note",
    ]
    with (out_dir / f"classifier_extra_metrics_{level_prefix}.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def run_extra_analyses(
    out_dir: Path,
    feature_sets: dict[str, np.ndarray],
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    names: dict[int, str],
    args: argparse.Namespace,
    level_prefix: str,
) -> dict[str, list[dict]]:
    spectral_sets = selected_feature_sets(feature_sets, args.feature_set, include_non_spectral=False)
    projection_rows = []
    classifier_rows = []
    projection_fns = []
    if args.run_lda:
        projection_fns.append(("lda", lda_projection))
    if args.run_tsne:
        projection_fns.append(("tsne", tsne_projection))
    if args.run_umap:
        projection_fns.append(("umap", umap_projection))

    for set_name, x in spectral_sets.items():
        slug = feature_slug(set_name)
        for method_name, projection_fn in projection_fns:
            print(f"[extra] {level_prefix}_{slug}_{method_name}: max_samples={args.max_samples}")
            coords, idx, note = projection_fn(x, y, args.max_samples, args.seed)
            row = {
                "level": level_prefix,
                "feature_set": set_name,
                "projection": method_name,
                "note": note,
            }
            if coords is None:
                row.update({
                    "n_samples": 0,
                    "n_classes": 0,
                    "silhouette": math.nan,
                    "davies_bouldin": math.nan,
                    "calinski_harabasz": math.nan,
                })
            else:
                ys = y[idx]
                save_projection_plot(
                    out_dir,
                    f"{level_prefix}_{slug}_{method_name}.png",
                    f"{level_prefix} {set_name} {method_name.upper()}",
                    coords,
                    ys,
                    names,
                )
                row.update({
                    "n_samples": int(coords.shape[0]),
                    "n_classes": int(np.unique(ys).size),
                    **projected_space_metrics(coords, ys),
                })
            projection_rows.append(row)

        if args.run_qda:
            print(f"[extra] {level_prefix}_{slug}_qda")
            qda = qda_probe(x, y, train_idx, test_idx, args.seed)
            classifier_rows.append({
                "level": level_prefix,
                "feature_set": set_name,
                "classifier": "qda",
                "n_train": int(train_idx.size),
                "n_test": int(test_idx.size),
                "n_classes": int(np.unique(y).size),
                **qda,
            })

    write_extra_projection_metrics(out_dir, level_prefix, projection_rows)
    write_extra_classifier_metrics(out_dir, level_prefix, classifier_rows)
    return {"projection_rows": projection_rows, "classifier_rows": classifier_rows}


def evaluate_feature_sets(
    out_dir: Path,
    feature_sets: dict[str, np.ndarray],
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    split_name: str,
    names: dict[int, str],
    args: argparse.Namespace,
) -> list[dict]:
    rows = []
    save_class_counts(out_dir, y, names)
    for set_name, x in feature_sets.items():
        print(f"[feature] {set_name}: dim={x.shape[1]}")
        row = {
            "feature_set": set_name,
            "n_samples": int(x.shape[0]),
            "n_classes": int(np.unique(y).size),
            "split": split_name,
        }
        row.update(separability_metrics(x, y, args.metric_max_samples, args.seed))
        probe = classifier_probe(x, y, train_idx, test_idx, args.seed)
        row.update({k: probe[k] for k in ["accuracy", "balanced_accuracy", "macro_f1"]})
        if args.include_knn:
            row.update(knn_probe(x, y, train_idx, test_idx))
        save_confusion_matrix(out_dir, set_name, probe["confusion"], probe["labels"], names)
        save_pca_plot(out_dir, set_name, x, y, names, args.plot_max_samples, args.seed)
        rows.append(row)
    write_results(out_dir, rows)
    return rows


def mode_output_dir(base_dir: Path, label_source: str) -> Path:
    base_dir = base_dir.resolve()
    return base_dir if base_dir.name == label_source else base_dir / label_source


def main() -> None:
    args = parse_args()
    add_jcalm_to_path()

    if not torch.cuda.is_available():
        raise RuntimeError("This analysis uses the jcalm CUDA rasterizer, but CUDA is not available.")

    model_path = args.model_path.resolve()
    source_path = args.source_path.resolve()
    out_dir = mode_output_dir(args.output_dir, args.label_source)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_cfg_args(model_path)
    iteration = find_iteration(model_path, args.iteration)
    num_channels = int(getattr(cfg, "num_channels", 10))
    num_classes = int(getattr(cfg, "num_classes", 200))
    effective_label_mode = args.label_mode
    load_args = make_load_args(args, cfg, effective_label_mode)
    names = label_names(source_path, effective_label_mode, num_classes, args.label_source)
    label_remap = semantic_label_remap(source_path, effective_label_mode)

    print(f"[setup] model_path={model_path}")
    print(f"[setup] source_path={source_path}")
    print(f"[setup] iteration={iteration}")
    print(f"[setup] label_source={args.label_source}, label_mode={effective_label_mode}, object_path={load_args.object_path}")

    scene_info = read_scene_infos(source_path, load_args)
    gaussians, color_decoder, classifier = load_model(model_path, cfg, iteration)
    view_dependence = None
    if args.run_viewdep:
        view_dependence = decoder_view_dependence_report(color_decoder, gaussians)
        save_decoder_view_dependence_report(out_dir, view_dependence)
        print(f"[view] decoder_is_view_dependent={view_dependence['decoder_is_view_dependent']}")
        print(f"[view] pg_view_direction={view_dependence['existing_pg_analysis_view_direction_description']}")
    notes = {}

    if args.label_source == "sam":
        cam_infos = select_sam_cam_infos(scene_info, args.view_split, args.max_views)
        if not cam_infos:
            raise RuntimeError(f"No RGB camera infos with SAM labels found in {source_path}")
        print(f"[views] using {len(cam_infos)} RGB SAM-labeled views")
        cameras = build_cameras(cam_infos, load_args)
        samples, y, groups, notes = collect_sam_pixel_samples(
            cameras,
            gaussians,
            color_decoder,
            num_channels=num_channels,
            samples_per_view_class=args.samples_per_view_class,
            max_samples_per_class=args.max_samples_per_class,
            ignore_labels=set(args.ignore_labels),
            seed=args.seed,
        )
        samples, y, groups, split_notes = filter_split_ready_samples(samples, y, groups, args.test_size)
        notes.update(split_notes)
        train_idx, test_idx, split_name = grouped_split(y, groups, args.test_size, args.seed)
    elif args.label_source == "predicted_pixel":
        cam_infos = select_predicted_cam_infos(scene_info, args.view_split, args.max_views)
        if not cam_infos:
            raise RuntimeError(f"No registered camera infos found in {source_path}")
        print(f"[views] using {len(cam_infos)} registered views with model-predicted pixel labels")
        cameras = build_cameras(cam_infos, load_args)
        samples, y, groups, notes = collect_predicted_pixel_samples(
            cameras,
            gaussians,
            color_decoder,
            classifier,
            num_channels=num_channels,
            samples_per_view_class=args.samples_per_view_class,
            max_samples_per_class=args.max_samples_per_class,
            ignore_labels=set(args.ignore_labels),
            seed=args.seed,
            label_remap=label_remap,
        )
        samples, y, groups, split_notes = filter_split_ready_samples(samples, y, groups, args.test_size)
        notes.update(split_notes)
        train_idx, test_idx, split_name = grouped_split(y, groups, args.test_size, args.seed)
    else:
        samples, y, groups, notes = collect_predicted_gaussian_samples(
            gaussians,
            color_decoder,
            classifier,
            max_samples_per_class=args.max_samples_per_class,
            max_gaussians=args.max_gaussians,
            min_pred_confidence=args.min_pred_confidence,
            min_opacity=args.min_opacity,
            max_scale_percentile=args.max_scale_percentile,
            ignore_labels=set(args.ignore_labels),
            seed=args.seed,
            label_remap=label_remap,
        )
        samples, y, groups, split_notes = filter_split_ready_samples(samples, y, groups, args.test_size)
        notes.update(split_notes)
        train_idx, test_idx, split_name = random_gaussian_split(y, args.test_size, args.seed)
        cam_infos = []

    print(f"[samples] collected {y.shape[0]} samples across {np.unique(y).size} labels")
    print(f"[split] {split_name}: train={train_idx.size}, test={test_idx.size}")

    save_spectral_signatures(out_dir, samples["SPECTRAL10"], y, names, args.signature_max_classes)
    all_features = build_feature_sets(samples, args.label_source)
    features = selected_feature_sets(all_features, args.feature_set, include_non_spectral=True)
    if not features:
        raise RuntimeError(f"No feature sets selected by --feature-set={args.feature_set}")
    evaluate_feature_sets(out_dir, features, y, train_idx, test_idx, split_name, names, args)

    level_prefix = level_prefix_for_run(args.label_source, args.level)
    extra_results = {"projection_rows": [], "classifier_rows": []}
    if args.run_lda or args.run_tsne or args.run_umap or args.run_qda:
        extra_results = run_extra_analyses(out_dir, features, y, train_idx, test_idx, names, args, level_prefix)

    summary = {
        "model_path": str(model_path),
        "source_path": str(source_path),
        "iteration": iteration,
        "label_source": args.label_source,
        "label_mode": effective_label_mode,
        "view_split": args.view_split,
        "num_views": len(cam_infos),
        "num_samples": int(y.shape[0]),
        "num_labels": int(np.unique(y).size),
        "split": split_name,
        "channel_names": CHANNEL_NAMES,
        "feature_sets": list(features.keys()),
        "feature_set_filter": args.feature_set,
        "extra_level_prefix": level_prefix,
        "extra_methods": {
            "lda": bool(args.run_lda),
            "qda": bool(args.run_qda),
            "tsne": bool(args.run_tsne),
            "umap": bool(args.run_umap),
            "max_samples": int(args.max_samples),
            "random_state": int(args.seed),
        },
        "extra_projection_rows": extra_results["projection_rows"],
        "extra_classifier_rows": extra_results["classifier_rows"],
        "interpretation_note": "Predicted label modes are representation/class-coherence analyses using the model's own labels, not ground-truth segmentation accuracy.",
        "decoder_view_dependence": view_dependence,
        "notes": notes,
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[done] wrote results to {out_dir}")


if __name__ == "__main__":
    main()
