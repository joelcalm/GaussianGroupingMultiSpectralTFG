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
import lpips as lpips_module
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace

_lpips_fn = None

def _get_lpips_fn():
    global _lpips_fn
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

def evaluate(model_paths):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            cfg = read_cfg_args(scene_dir)
            single_channel_mode = cfg.get('single_channel_mode', False)
            num_channels = cfg.get('num_channels', 3)
            if single_channel_mode:
                channel_names = {0: 'R', 1: 'G', 2: 'B'} if num_channels == 3 else {i: f'B{i}' for i in range(num_channels)}
                print(f"  Single channel mode detected ({num_channels} channels)")

            test_dir = Path(scene_dir) / "test"

            for method in os.listdir(test_dir):
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                method_dir = test_dir / method
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
                print("")

            with open(scene_dir + "/results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
        except Exception as e:
            print("Unable to compute metrics for model", scene_dir, ":", e)

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    args = parser.parse_args()
    evaluate(args.model_paths)
