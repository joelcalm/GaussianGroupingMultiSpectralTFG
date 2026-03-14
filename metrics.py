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
    for fname in sorted(os.listdir(renders_dir)):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names

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

                # --- Full-RGB metrics (always computed) ---
                ssims = []
                psnrs = []
                lpipss = []
                l1s = []

                # --- Per-channel accumulators ---
                if single_channel_mode:
                    ch_ssims = {ch: [] for ch in range(num_channels)}
                    ch_psnrs = {ch: [] for ch in range(num_channels)}
                    ch_lpipss = {ch: [] for ch in range(num_channels)}
                    ch_l1s = {ch: [] for ch in range(num_channels)}

                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    ssims.append(ssim(renders[idx], gts[idx]))
                    psnrs.append(psnr(renders[idx], gts[idx]))
                    lpipss.append(compute_lpips(renders[idx], gts[idx]))
                    l1s.append(l1_loss(renders[idx], gts[idx]).item())

                    if single_channel_mode:
                        for ch in range(num_channels):
                            r_ch = renders[idx][:, ch:ch+1, :, :]
                            g_ch = gts[idx][:, ch:ch+1, :, :]
                            ch_ssims[ch].append(ssim(r_ch, g_ch))
                            ch_psnrs[ch].append(psnr(r_ch, g_ch))
                            r_ch_3 = r_ch.expand(-1, 3, -1, -1)
                            g_ch_3 = g_ch.expand(-1, 3, -1, -1)
                            ch_lpipss[ch].append(compute_lpips(r_ch_3, g_ch_3))
                            ch_l1s[ch].append(l1_loss(r_ch, g_ch).item())

                # --- Print full-RGB results ---
                print("  [Full RGB]")
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
                    print("  [Per-channel]")
                    macro_ssim, macro_psnr, macro_lpips, macro_l1 = 0.0, 0.0, 0.0, 0.0
                    for ch in range(num_channels):
                        cn = channel_names[ch]
                        s = torch.tensor(ch_ssims[ch]).mean().item()
                        p = torch.tensor(ch_psnrs[ch]).mean().item()
                        lp = torch.tensor(ch_lpipss[ch]).mean().item()
                        l = torch.tensor(ch_l1s[ch]).mean().item()
                        macro_ssim += s; macro_psnr += p; macro_lpips += lp; macro_l1 += l
                        print(f"    {cn}: SSIM={s:.7f}  PSNR={p:.7f}  LPIPS={lp:.7f}  L1={l:.7f}")

                        full_dict[scene_dir][method][f"ch_{cn}_SSIM"] = s
                        full_dict[scene_dir][method][f"ch_{cn}_PSNR"] = p
                        full_dict[scene_dir][method][f"ch_{cn}_LPIPS"] = lp
                        full_dict[scene_dir][method][f"ch_{cn}_L1"] = l
                        per_view_dict[scene_dir][method][f"ch_{cn}_SSIM"] = {name: v for v, name in zip(torch.tensor(ch_ssims[ch]).tolist(), image_names)}
                        per_view_dict[scene_dir][method][f"ch_{cn}_PSNR"] = {name: v for v, name in zip(torch.tensor(ch_psnrs[ch]).tolist(), image_names)}
                        per_view_dict[scene_dir][method][f"ch_{cn}_LPIPS"] = {name: v for v, name in zip(torch.tensor(ch_lpipss[ch]).tolist(), image_names)}
                        per_view_dict[scene_dir][method][f"ch_{cn}_L1"] = {name: v for v, name in zip(ch_l1s[ch], image_names)}

                    macro_ssim /= num_channels; macro_psnr /= num_channels
                    macro_lpips /= num_channels; macro_l1 /= num_channels
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
