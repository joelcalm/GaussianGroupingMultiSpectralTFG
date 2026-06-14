#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from pathlib import Path
import os
from PIL import Image
import numpy as np
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim, l1_loss
try:
    import lpips as lpips_module
except ImportError:
    lpips_module = None
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace

_lpips_fn = None

def _get_lpips_fn():
    global _lpips_fn
    if lpips_module is None:
        raise ImportError("lpips is required for photometric LPIPS; rerun with --object_only to skip it")
    if _lpips_fn is None:
        _lpips_fn = lpips_module.LPIPS(net='vgg').cuda()
    return _lpips_fn

def compute_lpips(img1, img2):
    fn = _get_lpips_fn()
    return fn(img1 * 2 - 1, img2 * 2 - 1).item()

def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    npy_files = sorted(f for f in os.listdir(renders_dir) if f.endswith('.npy'))
    if npy_files:
        for fname in npy_files:
            render = np.load(renders_dir / fname)
            gt = np.load(gt_dir / fname)
            renders.append(torch.from_numpy(render).unsqueeze(0).cuda())
            gts.append(torch.from_numpy(gt).unsqueeze(0).cuda())
            image_names.append(fname)
        return renders, gts, image_names
    for fname in sorted(os.listdir(renders_dir)):
        if not fname.endswith('.png'):
            continue
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names

def read_frames_index(method_dir):
    index_path = method_dir / "frames_index.json"
    if not index_path.exists():
        return {}
    with open(index_path) as f:
        rows = json.load(f)
    active = {}
    for row in rows:
        stem = row.get("file_stem", f"{int(row['index']):05d}")
        channels = row.get("active_channels")
        if channels:
            active[f"{stem}.npy"] = [int(c) for c in channels]
            active[f"{stem}.png"] = [int(c) for c in channels]
    return active

def select_active_channels(render, gt, image_name, active_by_name):
    channels = active_by_name.get(image_name)
    if not channels:
        return render, gt, list(range(render.shape[1]))
    valid = [ch for ch in channels if 0 <= ch < render.shape[1] and ch < gt.shape[1]]
    if not valid:
        return render, gt, list(range(render.shape[1]))
    return render[:, valid, :, :], gt[:, valid, :, :], valid

def lpips_inputs(render, gt):
    if render.shape[1] == 1:
        return render.expand(-1, 3, -1, -1), gt.expand(-1, 3, -1, -1)
    if render.shape[1] == 2:
        return render[:, :1].expand(-1, 3, -1, -1), gt[:, :1].expand(-1, 3, -1, -1)
    if render.shape[1] > 3:
        vis_ch = [0, 3, 6] if render.shape[1] >= 7 else list(range(3))
        vis_ch = [ch for ch in vis_ch if ch < render.shape[1]]
        if len(vis_ch) < 3:
            vis_ch = list(range(3))
        return render[:, vis_ch, :, :], gt[:, vis_ch, :, :]
    return render, gt

def read_cfg_args(scene_dir):
    """Read saved cfg_args to detect single_channel_mode."""
    cfg_path = os.path.join(scene_dir, "cfg_args")
    if not os.path.exists(cfg_path):
        return {}
    with open(cfg_path) as f:
        cfg_ns = eval(f.read())
    return vars(cfg_ns)

def id2rgb(idx):
    if idx <= 0:
        return (0, 0, 0)
    h = (idx * 1.6180339887) % 1
    s = 0.5 + (idx % 2) * 0.5
    l = 0.5

    import colorsys

    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return (int(r * 255), int(g * 255), int(b * 255))

def color_decode_table(max_label_id):
    return {id2rgb(idx): int(idx) for idx in range(int(max_label_id) + 1)}

def decode_colorized_labels(arr, decode_table):
    out = np.zeros(arr.shape[:2], dtype=np.int64)
    flat_rgb = arr.reshape(-1, arr.shape[-1])
    flat_out = out.reshape(-1)
    unknown_pixels = 0
    for rgb in np.unique(flat_rgb, axis=0):
        key = tuple(int(v) for v in rgb[:3])
        mask = np.all(flat_rgb[:, :3] == rgb[:3], axis=1)
        label = decode_table.get(key)
        if label is None:
            unknown_pixels += int(mask.sum())
            label = 0
        flat_out[mask] = label
    return out, unknown_pixels

def read_label_image(path, decode_table=None):
    arr = np.array(Image.open(path))
    if arr.ndim == 2:
        return arr.astype(np.int64), 0
    if arr.ndim == 3 and arr.shape[-1] == 1:
        return arr[..., 0].astype(np.int64), 0
    if decode_table is None:
        raise ValueError(f"{path} is colorized, but no object color decode table was provided")
    return decode_colorized_labels(arr[..., :3].astype(np.uint8), decode_table)

def resize_label_nearest(arr, shape):
    if arr.shape == shape:
        return arr
    im = Image.fromarray(arr.astype(np.uint16))
    im = im.resize((shape[1], shape[0]), Image.NEAREST)
    return np.array(im).astype(np.int64)

def object_label_metrics(gt, pred, labels):
    rows = {}
    ious = []
    dices = []
    for lab in labels:
        gt_mask = gt == lab
        pred_mask = pred == lab
        gt_pixels = int(gt_mask.sum())
        pred_pixels = int(pred_mask.sum())
        intersection = int(np.logical_and(gt_mask, pred_mask).sum())
        union = int(np.logical_or(gt_mask, pred_mask).sum())
        iou = float(intersection / union) if union > 0 else None
        dice = float((2 * intersection) / (gt_pixels + pred_pixels)) if (gt_pixels + pred_pixels) > 0 else None
        rows[int(lab)] = {
            "gt_pixels": gt_pixels,
            "pred_pixels": pred_pixels,
            "intersection": intersection,
            "union": union,
            "iou": iou,
            "dice_f1": dice,
        }
        if gt_pixels > 0:
            ious.append(0.0 if iou is None else iou)
            dices.append(0.0 if dice is None else dice)
    return rows, {
        "mIoU": float(np.mean(ious)) if ious else None,
        "Dice_F1": float(np.mean(dices)) if dices else None,
    }

def read_method_frames(method_dir):
    frames_path = method_dir / "frames_index.json"
    if frames_path.exists():
        with open(frames_path) as f:
            return json.load(f)
    pred_dir = method_dir / "objects_pred_index"
    gt_dir = method_dir / "gt_objects_index"
    if not pred_dir.is_dir():
        pred_dir = method_dir / "objects_pred"
    if not gt_dir.is_dir():
        gt_dir = method_dir / "gt_objects_color"
    if not pred_dir.is_dir() or not gt_dir.is_dir():
        return []
    rows = []
    for idx, pred_path in enumerate(sorted(pred_dir.glob("*.png"))):
        if (gt_dir / pred_path.name).exists():
            rows.append({
                "index": idx,
                "file_stem": pred_path.stem,
                "image_name": pred_path.stem,
                "has_object_mask": True,
            })
    return rows

def compact_label_mapping(raw_to_key):
    keys = sorted({key for key in raw_to_key.values() if key is not None})
    key_to_label = {key: idx + 1 for idx, key in enumerate(keys)}
    return {int(raw): key_to_label[key] for raw, key in raw_to_key.items() if key in key_to_label}

def latest_vine_merge_map(scene_dir):
    merge_paths = sorted((Path(scene_dir) / "vine_tracklet_merges").glob("iteration_*/vine_tracklet_merge_map.json"))
    if not merge_paths:
        return {}
    with open(merge_paths[-1]) as f:
        merge = json.load(f)
    raw_to_physical = {}
    for row in merge.get("physical_vines", []):
        physical_id = int(row["physical_vine_id"])
        for member_id in row.get("member_instance_ids", []):
            raw_to_physical[int(member_id)] = f"vine:physical_{physical_id:04d}"
    return raw_to_physical

def load_instance_label_map(scene_dir):
    path = Path(scene_dir) / "instance_label_map.json"
    if not path.exists():
        return {}
    with open(path) as f:
        rows = json.load(f)
    return {int(k): v for k, v in rows.items()}

def build_metric_label_mappings(scene_dir):
    label_map = load_instance_label_map(scene_dir)
    vine_physical = latest_vine_merge_map(scene_dir)
    semantic_keys = {}
    instance_keys = {}

    for raw_label, meta in label_map.items():
        if raw_label <= 0:
            continue
        class_id = int(meta.get("class_id", raw_label))
        if class_id > 0:
            semantic_keys[raw_label] = f"semantic:{class_id}"

        object_type = str(meta.get("object_type", "")).lower()
        instance_id = meta.get("instance_id")
        source = str(meta.get("source", ""))
        class_name = str(meta.get("class_name", "object"))

        if raw_label in vine_physical:
            instance_keys[raw_label] = vine_physical[raw_label]
        elif object_type and object_type != "background" and instance_id not in {None, "", "background"}:
            instance_keys[raw_label] = f"{object_type}:{instance_id}"
        elif source == "hierarchical_composite" and instance_id not in {None, "", "background"}:
            instance_keys[raw_label] = f"{class_name}:{instance_id}"
        elif source == "sam3_video_tracker":
            track_id = meta.get("sam3_track_id", raw_label)
            instance_keys[raw_label] = f"{class_name}:track_{track_id}"

    if not semantic_keys:
        semantic_keys = {idx: f"semantic:{idx}" for idx in range(1, 256)}
    if not instance_keys:
        instance_keys = {idx: f"instance:{idx}" for idx in range(1, 256)}

    return {
        "Semantic": compact_label_mapping(semantic_keys),
        "Instance": compact_label_mapping(instance_keys),
    }

def remap_labels(arr, mapping):
    if not mapping:
        return np.zeros(arr.shape, dtype=np.int64)
    max_arr = int(arr.max()) if arr.size else 0
    max_key = max(mapping.keys()) if mapping else 0
    lut = np.zeros(max(max_arr, max_key) + 1, dtype=np.int64)
    for raw, mapped in mapping.items():
        if raw >= 0 and raw < lut.shape[0]:
            lut[int(raw)] = int(mapped)
    safe = arr.astype(np.int64, copy=False)
    out = np.zeros(safe.shape, dtype=np.int64)
    in_range = safe < lut.shape[0]
    out[in_range] = lut[safe[in_range]]
    return out

def object_mask_dirs(method_dir):
    pred_dir = method_dir / "objects_pred_index"
    gt_dir = method_dir / "gt_objects_index"
    using_colorized_fallback = False
    if not pred_dir.is_dir():
        pred_dir = method_dir / "objects_pred"
        using_colorized_fallback = True
    if not gt_dir.is_dir():
        gt_dir = method_dir / "gt_objects_color"
        using_colorized_fallback = True
    return pred_dir, gt_dir, using_colorized_fallback

def collect_object_pairs(method_dir, pred_dir, gt_dir):
    if not pred_dir.is_dir() or not gt_dir.is_dir():
        return [], "missing_object_prediction_or_gt_dirs"
    pairs = []
    for row in read_method_frames(method_dir):
        if not row.get("has_object_mask", True):
            continue
        stem = row.get("file_stem", f"{int(row['index']):05d}")
        pred_path = pred_dir / f"{stem}.png"
        gt_path = gt_dir / f"{stem}.png"
        if pred_path.exists() and gt_path.exists():
            pairs.append((stem, pred_path, gt_path))
    if not pairs:
        return [], "no_frames_with_object_masks"
    return pairs, None

def compute_label_mode_metrics(method_dir, max_label_id, label_mapping, prefix):
    pred_dir, gt_dir, using_colorized_fallback = object_mask_dirs(method_dir)
    pairs, skip_reason = collect_object_pairs(method_dir, pred_dir, gt_dir)
    if skip_reason:
        return None, {}, {"reason": skip_reason}

    decode_table = color_decode_table(max_label_id) if using_colorized_fallback else None
    gt_all = []
    pred_all = []
    labels = set()
    per_view = {f"{prefix}_mIoU": {}, f"{prefix}_Dice_F1": {}}
    unknown_pixels = 0
    evaluated_frames = 0
    valid_pixels_total = 0

    for stem, pred_path, gt_path in pairs:
        pred_raw, pred_unknown = read_label_image(pred_path, decode_table)
        gt_raw, gt_unknown = read_label_image(gt_path, decode_table)
        unknown_pixels += pred_unknown + gt_unknown
        pred_raw = resize_label_nearest(pred_raw, gt_raw.shape)
        pred = remap_labels(pred_raw, label_mapping)
        gt = remap_labels(gt_raw, label_mapping)
        valid = gt > 0
        if not np.any(valid):
            continue

        gt_valid = gt[valid]
        pred_valid = pred[valid]
        frame_labels = sorted(int(v) for v in np.unique(gt_valid) if int(v) > 0)
        _, frame_macro = object_label_metrics(gt_valid, pred_valid, frame_labels)
        per_view[f"{prefix}_mIoU"][f"{stem}.png"] = frame_macro["mIoU"]
        per_view[f"{prefix}_Dice_F1"][f"{stem}.png"] = frame_macro["Dice_F1"]

        labels.update(frame_labels)
        gt_all.append(gt_valid)
        pred_all.append(pred_valid)
        evaluated_frames += 1
        valid_pixels_total += int(valid.sum())

    if not gt_all:
        return None, per_view, {"reason": f"no_{prefix.lower()}_foreground_pixels"}

    labels = sorted(labels)
    gt_cat = np.concatenate(gt_all)
    pred_cat = np.concatenate(pred_all)
    _, macro = object_label_metrics(gt_cat, pred_cat, labels)
    pixel_accuracy = float(np.mean(gt_cat == pred_cat)) if gt_cat.size else None

    return {
        f"{prefix}_mIoU": macro["mIoU"],
        f"{prefix}_Dice_F1": macro["Dice_F1"],
        f"{prefix}_frames_evaluated": evaluated_frames,
        f"{prefix}_valid_pixels": valid_pixels_total,
        f"{prefix}_label_count": len(labels),
        f"{prefix}_pixel_accuracy": pixel_accuracy,
        f"{prefix}_unknown_color_pixels": unknown_pixels,
        f"{prefix}_used_colorized_mask_fallback": using_colorized_fallback,
    }, per_view, None

def clear_old_object_metric_keys(method_result, per_view_result):
    prefixes = ("Object_", "Semantic_", "Instance_")
    for key in list(method_result.keys()):
        if key in {"mIoU", "Dice_F1", "Dice_Coefficient_F1"} or key.startswith(prefixes):
            method_result.pop(key, None)
    for key in list(per_view_result.keys()):
        if key in {"mIoU", "Dice_F1"} or key.startswith(("Semantic_", "Instance_")):
            per_view_result.pop(key, None)

def update_object_instance_results(full_dict, per_view_dict, scene_dir, method, method_dir, num_classes):
    method_result = full_dict[scene_dir][method]
    per_view_result = per_view_dict[scene_dir][method]
    clear_old_object_metric_keys(method_result, per_view_result)

    mappings = build_metric_label_mappings(scene_dir)
    summaries = {}
    skip_reasons = {}
    for prefix in ["Semantic", "Instance"]:
        summary, mode_per_view, skip = compute_label_mode_metrics(
            method_dir,
            max(0, int(num_classes) - 1),
            mappings[prefix],
            prefix,
        )
        if summary is None:
            skip_reasons[prefix] = skip["reason"] if skip else "unknown"
            method_result[f"{prefix}_mIoU"] = None
            method_result[f"{prefix}_Dice_F1"] = None
            method_result[f"{prefix}_frames_evaluated"] = 0
            method_result[f"{prefix}_metric_skip_reason"] = skip_reasons[prefix]
        else:
            summaries[prefix] = summary
            method_result.update(summary)
            per_view_result.update(mode_per_view)

    primary = summaries.get("Instance") or summaries.get("Semantic")
    if primary:
        primary_prefix = "Instance" if "Instance" in summaries else "Semantic"
        method_result["mIoU"] = primary[f"{primary_prefix}_mIoU"]
        method_result["Dice_F1"] = primary[f"{primary_prefix}_Dice_F1"]
        method_result["Dice_Coefficient_F1"] = primary[f"{primary_prefix}_Dice_F1"]
        method_result["Object_metric_primary"] = primary_prefix
    else:
        method_result["mIoU"] = None
        method_result["Dice_F1"] = None
        method_result["Dice_Coefficient_F1"] = None
        method_result["Object_metric_skip_reason"] = "; ".join(f"{k}: {v}" for k, v in skip_reasons.items())
        print(f"  [Object metrics] skipped: {method_result['Object_metric_skip_reason']}")
        return

    if "Semantic" in summaries:
        print("  [Semantic labels]")
        print("    mIoU   : {:>12.7f}".format(method_result["Semantic_mIoU"]))
        print("    Dice/F1: {:>12.7f}".format(method_result["Semantic_Dice_F1"]))
        print("    Labels : {:>12d}".format(method_result["Semantic_label_count"]))
    else:
        print(f"  [Semantic labels] skipped: {skip_reasons.get('Semantic', 'unknown')}")

    if "Instance" in summaries:
        print("  [Instance/object labels]")
        print("    mIoU   : {:>12.7f}".format(method_result["Instance_mIoU"]))
        print("    Dice/F1: {:>12.7f}".format(method_result["Instance_Dice_F1"]))
        print("    Labels : {:>12d}".format(method_result["Instance_label_count"]))
        print("    Frames : {:>12d}".format(method_result["Instance_frames_evaluated"]))
    else:
        print(f"  [Instance/object labels] skipped: {skip_reasons.get('Instance', 'unknown')}")

def evaluate(model_paths, object_only=False):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            results_path = Path(scene_dir) / "results.json"
            per_view_path = Path(scene_dir) / "per_view.json"
            if results_path.exists():
                with open(results_path) as f:
                    full_dict[scene_dir] = json.load(f)
            else:
                full_dict[scene_dir] = {}
            if per_view_path.exists():
                with open(per_view_path) as f:
                    per_view_dict[scene_dir] = json.load(f)
            else:
                per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            cfg = read_cfg_args(scene_dir)
            single_channel_mode = cfg.get('single_channel_mode', False)
            num_channels = cfg.get('num_channels', 3)
            num_classes = cfg.get('num_classes', cfg.get('num_objects', 256))
            if single_channel_mode:
                channel_names = {0: 'R', 1: 'G', 2: 'B'} if num_channels == 3 else {i: f'B{i}' for i in range(num_channels)}
                print(f"  Single channel mode detected ({num_channels} channels)")

            test_dir = Path(scene_dir) / "test"

            for method in os.listdir(test_dir):
                print("Method:", method)

                full_dict[scene_dir].setdefault(method, {})
                per_view_dict[scene_dir].setdefault(method, {})
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                method_dir = test_dir / method
                if object_only:
                    update_object_instance_results(full_dict, per_view_dict, scene_dir, method, method_dir, num_classes)
                    print("")
                    continue

                gt_dir = method_dir/ "gt"
                renders_dir = method_dir / "renders"
                renders, gts, image_names = readImages(renders_dir, gt_dir)
                active_by_name = read_frames_index(method_dir)

                # --- Full metrics (always computed) ---
                ssims = []
                psnrs = []
                lpipss = []
                l1s = []

                img_channels = renders[0].shape[1] if len(renders) > 0 else 3
                eval_num_ch = min(num_channels, img_channels)
                vis_ch = [0, 3, 6] if img_channels >= 7 else list(range(min(3, img_channels)))

                if single_channel_mode:
                    ch_ssims = {ch: [] for ch in range(eval_num_ch)}
                    ch_psnrs = {ch: [] for ch in range(eval_num_ch)}
                    ch_lpipss = {ch: [] for ch in range(eval_num_ch)}
                    ch_l1s = {ch: [] for ch in range(eval_num_ch)}

                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    render_eval, gt_eval, active_channels = select_active_channels(renders[idx], gts[idx], image_names[idx], active_by_name)
                    ssims.append(ssim(render_eval, gt_eval))
                    psnrs.append(psnr(render_eval, gt_eval))
                    l1s.append(l1_loss(render_eval, gt_eval).item())
                    r_lpips, g_lpips = lpips_inputs(render_eval, gt_eval)
                    lpipss.append(compute_lpips(r_lpips, g_lpips))

                    if single_channel_mode:
                        for ch in active_channels:
                            if ch >= eval_num_ch:
                                continue
                            r_ch = renders[idx][:, ch:ch+1, :, :]
                            g_ch = gts[idx][:, ch:ch+1, :, :]
                            ch_ssims[ch].append(ssim(r_ch, g_ch))
                            ch_psnrs[ch].append(psnr(r_ch, g_ch))
                            r_ch_3 = r_ch.expand(-1, 3, -1, -1)
                            g_ch_3 = g_ch.expand(-1, 3, -1, -1)
                            ch_lpipss[ch].append(compute_lpips(r_ch_3, g_ch_3))
                            ch_l1s[ch].append(l1_loss(r_ch, g_ch).item())

                full_label = f"Full ({img_channels}-ch)" if img_channels > 3 else "Full RGB"
                print(f"  [{full_label}]")
                print("    SSIM : {:>12.7f}".format(torch.tensor(ssims).mean()))
                print("    PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean()))
                print("    LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean()))
                print("    L1   : {:>12.7f}".format(torch.tensor(l1s).mean()))

                full_dict[scene_dir][method].update({
                    "SSIM": torch.tensor(ssims).mean().item(),
                    "PSNR": torch.tensor(psnrs).mean().item(),
                    "LPIPS": torch.tensor(lpipss).mean().item(),
                    "L1": torch.tensor(l1s).mean().item(),
                })
                per_view_dict[scene_dir][method].update({
                    "SSIM": {name: s for s, name in zip(torch.tensor(ssims).tolist(), image_names)},
                    "PSNR": {name: p for p, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                    "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
                    "L1": {name: v for v, name in zip(l1s, image_names)},
                })

                # --- Print & store per-channel results ---
                if single_channel_mode:
                    print(f"  [Per-channel] ({eval_num_ch} channels)")
                    macro_ssim, macro_psnr, macro_lpips, macro_l1 = 0.0, 0.0, 0.0, 0.0
                    valid_macro_channels = []
                    for ch in range(eval_num_ch):
                        if len(ch_ssims[ch]) == 0:
                            continue
                        cn = channel_names.get(ch, f'B{ch}')
                        s = torch.tensor(ch_ssims[ch]).mean().item()
                        p = torch.tensor(ch_psnrs[ch]).mean().item()
                        lp = torch.tensor(ch_lpipss[ch]).mean().item()
                        l = torch.tensor(ch_l1s[ch]).mean().item()
                        macro_ssim += s; macro_psnr += p; macro_lpips += lp; macro_l1 += l
                        valid_macro_channels.append(ch)
                        print(f"    {cn}: n={len(ch_ssims[ch])}  SSIM={s:.7f}  PSNR={p:.7f}  LPIPS={lp:.7f}  L1={l:.7f}")

                        full_dict[scene_dir][method][f"ch_{cn}_SSIM"] = s
                        full_dict[scene_dir][method][f"ch_{cn}_PSNR"] = p
                        full_dict[scene_dir][method][f"ch_{cn}_LPIPS"] = lp
                        full_dict[scene_dir][method][f"ch_{cn}_L1"] = l
                        per_view_dict[scene_dir][method][f"ch_{cn}_SSIM"] = {name: v for v, name in zip(torch.tensor(ch_ssims[ch]).tolist(), image_names)}
                        per_view_dict[scene_dir][method][f"ch_{cn}_PSNR"] = {name: v for v, name in zip(torch.tensor(ch_psnrs[ch]).tolist(), image_names)}
                        per_view_dict[scene_dir][method][f"ch_{cn}_LPIPS"] = {name: v for v, name in zip(torch.tensor(ch_lpipss[ch]).tolist(), image_names)}
                        per_view_dict[scene_dir][method][f"ch_{cn}_L1"] = {name: v for v, name in zip(ch_l1s[ch], image_names)}

                    macro_denom = max(1, len(valid_macro_channels))
                    macro_ssim /= macro_denom; macro_psnr /= macro_denom
                    macro_lpips /= macro_denom; macro_l1 /= macro_denom
                    print(f"    Macro-avg: SSIM={macro_ssim:.7f}  PSNR={macro_psnr:.7f}  LPIPS={macro_lpips:.7f}  L1={macro_l1:.7f}")
                    full_dict[scene_dir][method]["macro_SSIM"] = macro_ssim
                    full_dict[scene_dir][method]["macro_PSNR"] = macro_psnr
                    full_dict[scene_dir][method]["macro_LPIPS"] = macro_lpips
                    full_dict[scene_dir][method]["macro_L1"] = macro_l1

                update_object_instance_results(full_dict, per_view_dict, scene_dir, method, method_dir, num_classes)
                print("")

            with open(scene_dir + "/results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
        except Exception as e:
            print("Unable to compute metrics for model", scene_dir, ":", e)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    parser.add_argument('--object_only', action='store_true', help="Only compute object/instance mIoU and Dice/F1, preserving existing photometric metrics.")
    args = parser.parse_args()

    if not args.object_only:
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    evaluate(args.model_paths, object_only=args.object_only)
