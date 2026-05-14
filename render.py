# Copyright (C) 2023, Gaussian-Grouping
# Gaussian-Grouping research group, https://github.com/lkeab/gaussian-grouping
# All rights reserved.
#
# ------------------------------------------------------------------------
# Modified from codes in Gaussian-Splatting 
# GRAPHDECO research group, https://team.inria.fr/graphdeco

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
from PIL import Image
import colorsys
import cv2
from sklearn.decomposition import PCA
from utils.color_decoder import ColorDecoder
import json

def clear_pngs(path):
    if not os.path.isdir(path):
        return
    for name in os.listdir(path):
        if name.endswith(".png"):
            os.remove(os.path.join(path, name))

def feature_to_rgb(features):
    # Input features shape: (16, H, W)
    
    # Reshape features for PCA
    H, W = features.shape[1], features.shape[2]
    features_reshaped = features.view(features.shape[0], -1).T

    # Apply PCA and get the first 3 components
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(features_reshaped.cpu().numpy())

    # Reshape back to (H, W, 3)
    pca_result = pca_result.reshape(H, W, 3)

    # Normalize to [0, 255]
    pca_normalized = 255 * (pca_result - pca_result.min()) / (pca_result.max() - pca_result.min())

    rgb_array = pca_normalized.astype('uint8')

    return rgb_array

def id2rgb(id, max_num_obj=65535):
    if not 0 <= id <= max_num_obj:
        raise ValueError("ID should be in range(0, max_num_obj)")

    # Convert the ID into a hue value
    golden_ratio = 1.6180339887
    h = ((id * golden_ratio) % 1)           # Ensure value is between 0 and 1
    s = 0.5 + (id % 2) * 0.5       # Alternate between 0.5 and 1.0
    l = 0.5

    
    # Use colorsys to convert HSL to RGB
    rgb = np.zeros((3, ), dtype=np.uint8)
    if id==0:   #invalid region
        return rgb
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    rgb[0], rgb[1], rgb[2] = int(r*255), int(g*255), int(b*255)

    return rgb

def visualize_obj(objects):
    rgb_mask = np.zeros((*objects.shape[-2:], 3), dtype=np.uint8)
    all_obj_ids = np.unique(objects)
    for id in all_obj_ids:
        colored_mask = id2rgb(id)
        rgb_mask[objects == id] = colored_mask
    return rgb_mask


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, classifier, color_decoder=None, single_channel_mode=False, num_channels=3):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    colormask_path = os.path.join(model_path, name, "ours_{}".format(iteration), "objects_feature16")
    gt_colormask_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_objects_color")
    pred_obj_path = os.path.join(model_path, name, "ours_{}".format(iteration), "objects_pred")
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(colormask_path, exist_ok=True)
    makedirs(gt_colormask_path, exist_ok=True)
    makedirs(pred_obj_path, exist_ok=True)

    channel_paths = {}
    if single_channel_mode or num_channels > 3:
        channel_names = {0: 'R', 1: 'G', 2: 'B'} if num_channels == 3 else {i: f'B{i}' for i in range(num_channels)}
        for ch_id, ch_name in channel_names.items():
            p = os.path.join(model_path, name, "ours_{}".format(iteration), f"channel_{ch_name}")
            makedirs(p, exist_ok=True)
            clear_pngs(p)
            channel_paths[ch_id] = p

    frames_index = []
    channel_frames_index = {int(ch_id): [] for ch_id in channel_paths}
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        results = render(view, gaussians, pipeline, background, color_decoder=color_decoder)
        rendering = results["render"]
        rendering_obj = results["render_object"]
        
        logits = classifier(rendering_obj)
        pred_obj = torch.argmax(logits,dim=0)
        pred_obj_mask = visualize_obj(pred_obj.cpu().numpy().astype(np.uint32))
        
        gt_objects = view.objects
        if gt_objects is not None:
            gt_rgb_mask = visualize_obj(gt_objects.cpu().numpy().astype(np.uint32))
        else:
            gt_rgb_mask = np.zeros((view.image_height, view.image_width, 3), dtype=np.uint8)

        rgb_mask = feature_to_rgb(rendering_obj)
        Image.fromarray(rgb_mask).save(os.path.join(colormask_path, '{0:05d}'.format(idx) + ".png"))
        Image.fromarray(gt_rgb_mask).save(os.path.join(gt_colormask_path, '{0:05d}'.format(idx) + ".png"))
        Image.fromarray(pred_obj_mask).save(os.path.join(pred_obj_path, '{0:05d}'.format(idx) + ".png"))
        gt = view.original_image[:num_channels, :, :]
        active_channels = getattr(view, "active_channels", None)
        active_channels = active_channels.tolist() if active_channels is not None else list(range(gt.shape[0]))
        active_channels = sorted({int(c) for c in active_channels if 0 <= int(c) < gt.shape[0]})
        active_channel_set = set(active_channels)
        frames_index.append({
            "index": idx,
            "file_stem": "{0:05d}".format(idx),
            "image_name": view.image_name,
            "active_channels": [int(c) for c in active_channels],
            "has_object_mask": gt_objects is not None,
        })

        if num_channels > 3:
            vis_ch = [0, 1, 2] if num_channels >= 3 else list(range(min(3, num_channels)))
            render_vis = rendering[vis_ch]
            gt_vis = gt[vis_ch]
            torchvision.utils.save_image(render_vis, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(gt_vis, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
            np.save(os.path.join(render_path, '{0:05d}'.format(idx) + ".npy"), torch.clamp(rendering, 0.0, 1.0).cpu().numpy())
            np.save(os.path.join(gts_path, '{0:05d}'.format(idx) + ".npy"), gt.cpu().numpy())
        else:
            torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

        if single_channel_mode or num_channels > 3:
            for ch_id, ch_path in channel_paths.items():
                if ch_id not in active_channel_set:
                    continue
                ch_render = rendering[ch_id:ch_id+1].expand(3, -1, -1)
                ch_gt = gt[ch_id:ch_id+1].expand(3, -1, -1)
                ch_combined = torch.cat([ch_render, ch_gt], dim=2)
                torchvision.utils.save_image(ch_combined, os.path.join(ch_path, '{0:05d}'.format(idx) + ".png"))
                channel_frames_index[int(ch_id)].append({
                    "index": idx,
                    "file_stem": "{0:05d}".format(idx),
                    "image_name": view.image_name,
                })

    with open(os.path.join(model_path, name, "ours_{}".format(iteration), "frames_index.json"), "w") as f:
        json.dump(frames_index, f, indent=2)
    if channel_paths:
        with open(os.path.join(model_path, name, "ours_{}".format(iteration), "channel_frames_index.json"), "w") as f:
            json.dump(channel_frames_index, f, indent=2)

    out_path = os.path.join(render_path[:-8],'concat')
    makedirs(out_path,exist_ok=True)

    gt_files = sorted(f for f in os.listdir(gts_path) if f.endswith(".png"))
    if len(gt_files) == 0:
        return

    sample_gt = np.array(Image.open(os.path.join(gts_path, gt_files[0])))
    fourcc = cv2.VideoWriter.fourcc(*'DIVX')
    size = (sample_gt.shape[1]*5, sample_gt.shape[0])
    fps = float(5) if 'train' in out_path else float(1)
    writer = cv2.VideoWriter(os.path.join(out_path,'result.mp4'), fourcc, fps, size)

    for file_name in gt_files:
        gt_img = np.array(Image.open(os.path.join(gts_path,file_name)))
        rgb = np.array(Image.open(os.path.join(render_path,file_name)))
        gt_obj = np.array(Image.open(os.path.join(gt_colormask_path,file_name)))
        render_obj = np.array(Image.open(os.path.join(colormask_path,file_name)))
        pred_obj = np.array(Image.open(os.path.join(pred_obj_path,file_name)))

        result = np.hstack([gt_img,rgb,gt_obj,pred_obj,render_obj])
        result = result.astype('uint8')

        Image.fromarray(result).save(os.path.join(out_path,file_name))
        writer.write(result[:,:,::-1])

    writer.release()


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        use_color_embed = dataset.use_color_embed if hasattr(dataset, 'use_color_embed') else False
        color_embed_dim = dataset.color_embed_dim if hasattr(dataset, 'color_embed_dim') else 16
        color_decoder_hidden_dim = dataset.color_decoder_hidden_dim if hasattr(dataset, 'color_decoder_hidden_dim') else 32
        color_decoder_num_hidden_layers = dataset.color_decoder_num_hidden_layers if hasattr(dataset, 'color_decoder_num_hidden_layers') else 2
        single_channel_mode = getattr(dataset, 'single_channel_mode', False)
        num_channels = getattr(dataset, 'num_channels', 3)
        num_objects = getattr(dataset, 'num_objects', 16)
        gaussians = GaussianModel(dataset.sh_degree, num_objects=num_objects, use_color_embed=use_color_embed, color_embed_dim=color_embed_dim)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        
        num_classes = dataset.num_classes
        print("Num classes: ",num_classes)
        print("Single channel mode: ", single_channel_mode)

        classifier = torch.nn.Conv2d(gaussians.num_objects, num_classes, kernel_size=1)
        classifier.cuda()
        classifier.load_state_dict(torch.load(os.path.join(dataset.model_path,"point_cloud","iteration_"+str(scene.loaded_iter),"classifier.pth")))

        color_decoder = None
        if use_color_embed:
            color_decoder = ColorDecoder(
                input_dim=color_embed_dim,
                hidden_dim=color_decoder_hidden_dim,
                output_dim=num_channels,
                num_hidden_layers=color_decoder_num_hidden_layers,
            )
            color_decoder.cuda()
            decoder_path = os.path.join(dataset.model_path,"point_cloud","iteration_"+str(scene.loaded_iter),"color_decoder.pth")
            if os.path.exists(decoder_path):
                color_decoder.load_state_dict(torch.load(decoder_path))
            color_decoder.eval()

        bg_color = [1] * num_channels if dataset.white_background else [0] * num_channels
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, classifier, color_decoder,
                        single_channel_mode=single_channel_mode, num_channels=num_channels)

        if (not skip_test) and (len(scene.getTestCameras()) > 0):
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, classifier, color_decoder,
                        single_channel_mode=single_channel_mode, num_channels=num_channels)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)
