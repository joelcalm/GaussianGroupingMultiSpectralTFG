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

from scene.cameras import Camera
import numpy as np
from PIL import Image
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal
import torch
from scipy.ndimage import zoom as ndimage_zoom

WARNED = False


def _pack_rgb_ids(mask_rgb):
    mask_rgb = mask_rgb.astype(np.int64)
    return mask_rgb[..., 0] + (mask_rgb[..., 1] << 8) + (mask_rgb[..., 2] << 16)


def _build_rgb_object_id_mapping(cam_infos, max_classes):
    rgb_ids = set()
    saw_rgb_mask = False

    for cam_info in cam_infos:
        if cam_info.objects is None:
            continue

        obj_np = np.array(cam_info.objects)
        if obj_np.ndim != 3 or obj_np.shape[-1] < 3:
            continue

        rgb = obj_np[..., :3]
        if np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(rgb[..., 0], rgb[..., 2]):
            continue

        saw_rgb_mask = True
        rgb_ids.update(np.unique(_pack_rgb_ids(rgb)).tolist())

    if not saw_rgb_mask:
        return None

    max_classes = int(max_classes) if max_classes is not None else 256
    max_classes = max(max_classes, 1)

    sorted_ids = sorted(int(v) for v in rgb_ids)
    mapping = {}
    next_class = 0

    if 0 in rgb_ids:
        mapping[0] = 0
        next_class = 1

    fg_ids = [v for v in sorted_ids if v != 0]
    for rgb_id in fg_ids:
        if next_class >= max_classes:
            break
        mapping[rgb_id] = next_class
        next_class += 1

    dropped = len(fg_ids) - max(0, next_class - (1 if 0 in mapping else 0))
    print(f"[camera_utils] Decoding RGB object masks with {len(mapping)} mapped IDs (max_classes={max_classes}).")
    if dropped > 0:
        print(f"[camera_utils] Warning: dropped {dropped} RGB IDs because they exceed num_classes={max_classes}. They map to background (0).")

    return mapping


def _convert_object_mask_to_indices(mask_np, object_id_mapping):
    if mask_np.ndim == 2:
        return mask_np.astype(np.int64)

    if mask_np.ndim == 3 and mask_np.shape[-1] == 1:
        return mask_np[..., 0].astype(np.int64)

    if mask_np.ndim == 3 and mask_np.shape[-1] >= 3:
        rgb = mask_np[..., :3]
        if np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(rgb[..., 0], rgb[..., 2]):
            return rgb[..., 0].astype(np.int64)

        packed = _pack_rgb_ids(rgb)
        if object_id_mapping is None:
            # Fallback to deterministic per-image remap when no scene-level mapping is available.
            remapped = np.zeros_like(packed, dtype=np.int64)
            uniq = np.sort(np.unique(packed))
            next_class = 0
            if uniq.size > 0 and uniq[0] == 0:
                next_class = 1
            for rgb_id in uniq.tolist():
                if rgb_id == 0:
                    remapped[packed == rgb_id] = 0
                else:
                    remapped[packed == rgb_id] = next_class
                    next_class += 1
            return remapped

        keys = np.array(sorted(object_id_mapping.keys()), dtype=np.int64)
        vals = np.array([object_id_mapping[k] for k in keys], dtype=np.int64)
        idx = np.searchsorted(keys, packed)
        valid = (idx < keys.size) & (keys[idx] == packed)
        remapped = np.zeros_like(packed, dtype=np.int64)
        remapped[valid] = vals[idx[valid]]
        return remapped

    raise ValueError(f"Unsupported object mask shape: {mask_np.shape}")

def _resize_multispectral(ms_array, resolution):
    """Resize a [H, W, C] float32 numpy array to (target_w, target_h)."""
    target_w, target_h = resolution
    h, w, c = ms_array.shape
    if h == target_h and w == target_w:
        return ms_array
    scale_h = target_h / h
    scale_w = target_w / w
    return ndimage_zoom(ms_array, (scale_h, scale_w, 1), order=1)


def loadCam(args, id, cam_info, resolution_scale, **kwargs):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    ms_path = getattr(cam_info, 'multispectral_path', None)
    if ms_path is not None:
        ms_array = np.load(ms_path)  # [H, W, C] float32 in [0, 1]
        ms_array = _resize_multispectral(ms_array, resolution)
        gt_image = torch.from_numpy(ms_array).permute(2, 0, 1).float()  # [C, H, W]
        loaded_mask = None
    else:
        resized_image_rgb = PILtoTorch(cam_info.image, resolution)
        gt_image = resized_image_rgb[:3, ...]
        loaded_mask = None
        if resized_image_rgb.shape[0] == 4:
            loaded_mask = resized_image_rgb[3:4, ...]

    objects = cam_info.objects
    objects_tensor = None
    if objects is not None:
        objects = objects.resize(resolution, Image.NEAREST)
        object_id_mapping = kwargs.get('object_id_mapping', None)
        objects_np = _convert_object_mask_to_indices(np.array(objects), object_id_mapping)
        objects_tensor = torch.from_numpy(objects_np.astype(np.int64))
    channel_idx = kwargs.get('channel_idx', -1)
    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device,
                  objects=objects_tensor,
                  channel_idx=channel_idx)

def cameraList_from_camInfos(cam_infos, resolution_scale, args, single_channel_mode=False, num_channels=3, object_id_mapping=None):
    camera_list = []

    if object_id_mapping is None:
        max_classes = getattr(args, 'num_classes', 256)
        object_id_mapping = _build_rgb_object_id_mapping(cam_infos, max_classes)

    for id, c in enumerate(cam_infos):
        channel_idx = id % num_channels if single_channel_mode else -1
        camera_list.append(loadCam(args, id, c, resolution_scale, channel_idx=channel_idx, object_id_mapping=object_id_mapping))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry
