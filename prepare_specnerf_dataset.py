"""
Prepare a Spec-NeRF scene for Gaussian Grouping training.

Converts raw per-pose, per-channel TIFF images into COLMAP-format scene with
20-channel .npy multispectral images + pseudo-RGB PNGs for mask generation.

Usage:
    python prepare_specnerf_dataset.py --scene xjhdesk
    python prepare_specnerf_dataset.py --scene xjhdesk --filter filter19

Expected input layout:
    data/Spec-NeRF/RAW/multi-view-MSI/<filter>/<scene>/
        sparse/0/cameras.bin, images.bin, points3D.bin
        pose0img/images/img_XXXX.tiff  (20 single-channel uint16 TIFFs)
        pose1img/images/...
        ...
        jpegs/img_XXXX.jpeg

Output layout (COLMAP-compatible):
    data/Spec-NeRF/<scene>/
        sparse/0/cameras.txt, images.txt, points3D.txt (PINHOLE model)
        sparse/0/points3D.ply  (converted from .bin)
        images/img_1000.png    (pseudo-RGB for mask generation)
        images_multispectral/img_1000.npy  (20-channel float32, [0,1])
"""

import argparse
import os
import struct
import sys

import numpy as np
from PIL import Image


def read_colmap_cameras_bin(path):
    cameras = {}
    with open(path, "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num):
            cam_id = struct.unpack("<I", f.read(4))[0]
            model_id = struct.unpack("<I", f.read(4))[0]
            width = struct.unpack("<Q", f.read(8))[0]
            height = struct.unpack("<Q", f.read(8))[0]
            n_params = {0: 3, 1: 3, 2: 4, 3: 4, 4: 8}.get(model_id, 4)
            params = struct.unpack(f"<{n_params}d", f.read(8 * n_params))
            cameras[cam_id] = {
                "model_id": model_id,
                "width": width,
                "height": height,
                "params": params,
            }
    return cameras


def read_colmap_images_bin(path):
    images = {}
    with open(path, "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num):
            img_id = struct.unpack("<I", f.read(4))[0]
            qw, qx, qy, qz = struct.unpack("<dddd", f.read(32))
            tx, ty, tz = struct.unpack("<ddd", f.read(24))
            camera_id = struct.unpack("<I", f.read(4))[0]
            name = b""
            while True:
                ch = f.read(1)
                if ch == b"\x00":
                    break
                name += ch
            num_pts2d = struct.unpack("<Q", f.read(8))[0]
            f.read(num_pts2d * 24)
            images[img_id] = {
                "qvec": (qw, qx, qy, qz),
                "tvec": (tx, ty, tz),
                "camera_id": camera_id,
                "name": name.decode(),
            }
    return images


def read_colmap_points3d_bin(path):
    """Read points3D.bin and return (xyz, rgb) arrays."""
    pts_xyz = []
    pts_rgb = []
    with open(path, "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num):
            _pid = struct.unpack("<Q", f.read(8))[0]
            x, y, z = struct.unpack("<ddd", f.read(24))
            r, g, b = struct.unpack("<BBB", f.read(3))
            _err = struct.unpack("<d", f.read(8))[0]
            track_len = struct.unpack("<Q", f.read(8))[0]
            f.read(track_len * 8)
            pts_xyz.append([x, y, z])
            pts_rgb.append([r, g, b])
    return np.array(pts_xyz), np.array(pts_rgb, dtype=np.uint8)


def write_cameras_txt(path, cam_id, width, height, fx, fy, cx, cy):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: 1\n")
        f.write(f"{cam_id} PINHOLE {width} {height} {fx:.10f} {fy:.10f} {cx:.10f} {cy:.10f}\n")


def write_images_txt(path, images_data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(images_data)}\n")
        for img_id, qvec, tvec, cam_id, name in images_data:
            qstr = " ".join(f"{q:.10f}" for q in qvec)
            tstr = " ".join(f"{t:.10f}" for t in tvec)
            f.write(f"{img_id} {qstr} {tstr} {cam_id} {name}\n")
            f.write("\n")


def write_points3d_txt(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        f.write("# Number of points: 0\n")


def write_points3d_ply(path, xyz, rgb):
    from plyfile import PlyData, PlyElement

    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ]
    elements = np.empty(len(xyz), dtype=dtype)
    elements["x"] = xyz[:, 0]
    elements["y"] = xyz[:, 1]
    elements["z"] = xyz[:, 2]
    elements["nx"] = 0
    elements["ny"] = 0
    elements["nz"] = 0
    elements["red"] = rgb[:, 0]
    elements["green"] = rgb[:, 1]
    elements["blue"] = rgb[:, 2]

    el = PlyElement.describe(elements, "vertex")
    PlyData([el]).write(path)


def undistort_radial(img_arr, fx, cx, cy, k1):
    """Remove radial distortion (single k1 coefficient, shared focal length)."""
    h, w = img_arr.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    x_norm = (xs - cx) / fx
    y_norm = (ys - cy) / fx
    r2 = x_norm ** 2 + y_norm ** 2
    radial = 1.0 + k1 * r2
    x_undist = x_norm * radial
    y_undist = y_norm * radial
    map_x = (x_undist * fx + cx).astype(np.float32)
    map_y = (y_undist * fy + cy).astype(np.float32) if False else (y_undist * fx + cy).astype(np.float32)

    try:
        import cv2
        if img_arr.ndim == 2:
            return cv2.remap(img_arr, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        else:
            return cv2.remap(img_arr, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    except ImportError:
        from scipy.ndimage import map_coordinates
        if img_arr.ndim == 2:
            return map_coordinates(img_arr, [map_y, map_x], order=1, mode="nearest").astype(img_arr.dtype)
        else:
            out = np.empty_like(img_arr)
            for c in range(img_arr.shape[2]):
                out[:, :, c] = map_coordinates(img_arr[:, :, c], [map_y, map_x], order=1, mode="nearest")
            return out


def main():
    parser = argparse.ArgumentParser(description="Prepare Spec-NeRF scene for Gaussian Grouping")
    parser.add_argument("--scene", type=str, default="xjhdesk", help="Scene name")
    parser.add_argument("--filter", type=str, default="filter19", help="Filter directory name")
    parser.add_argument("--data_root", type=str, default="data/Spec-NeRF",
                        help="Root of Spec-NeRF data")
    parser.add_argument("--num_channels", type=int, default=20)
    parser.add_argument("--rgb_channels", type=int, nargs=3, default=[0, 7, 14],
                        help="Channel indices for pseudo-RGB visualization")
    parser.add_argument("--undistort", action="store_true", default=True,
                        help="Undistort images (RADIAL -> PINHOLE)")
    parser.add_argument("--no_undistort", action="store_true",
                        help="Skip undistortion (just drop k1)")
    args = parser.parse_args()

    raw_scene_dir = os.path.join(args.data_root, "RAW", "multi-view-MSI", args.filter, args.scene)
    out_dir = os.path.join(args.data_root, args.scene)

    if not os.path.isdir(raw_scene_dir):
        print(f"ERROR: Raw scene directory not found: {raw_scene_dir}")
        sys.exit(1)

    # --- Read existing COLMAP binary data ---
    print("Reading COLMAP binary data...")
    cameras = read_colmap_cameras_bin(os.path.join(raw_scene_dir, "sparse/0/cameras.bin"))
    images = read_colmap_images_bin(os.path.join(raw_scene_dir, "sparse/0/images.bin"))

    assert len(cameras) == 1, f"Expected 1 camera, got {len(cameras)}"
    cam_id = list(cameras.keys())[0]
    cam = cameras[cam_id]
    print(f"  Camera: model_id={cam['model_id']} {cam['width']}x{cam['height']}")
    print(f"  Images: {len(images)} views")

    # RADIAL model: params = (f, cx, cy, k1)
    if cam["model_id"] == 2:
        f_shared, cx, cy, k1 = cam["params"]
        fx, fy = f_shared, f_shared
        print(f"  RADIAL params: f={f_shared:.2f} cx={cx:.1f} cy={cy:.1f} k1={k1:.6f}")
        do_undistort = args.undistort and not args.no_undistort
        if do_undistort:
            print(f"  Will undistort images (k1={k1:.6f})")
        else:
            print(f"  Skipping undistortion, dropping k1={k1:.6f}")
    elif cam["model_id"] == 3:
        fx, fy, cx, cy = cam["params"]
        k1 = 0
        do_undistort = False
    else:
        fx = cam["params"][0]
        fy = fx
        cx, cy = cam["params"][1], cam["params"][2]
        k1 = 0
        do_undistort = False

    width, height = cam["width"], cam["height"]

    # --- Write COLMAP text files (PINHOLE model) ---
    sparse_dir = os.path.join(out_dir, "sparse", "0")
    write_cameras_txt(os.path.join(sparse_dir, "cameras.txt"), cam_id, width, height, fx, fy, cx, cy)
    print(f"  cameras.txt written (PINHOLE)")

    # Sort images by id for consistent ordering
    sorted_img_ids = sorted(images.keys())
    images_data = []
    for img_id in sorted_img_ids:
        im = images[img_id]
        png_name = im["name"].replace(".jpeg", ".png").replace(".jpg", ".png")
        images_data.append((img_id, im["qvec"], im["tvec"], im["camera_id"], png_name))

    write_images_txt(os.path.join(sparse_dir, "images.txt"), images_data)
    print(f"  images.txt written ({len(images_data)} images)")

    write_points3d_txt(os.path.join(sparse_dir, "points3D.txt"))

    # Convert points3D.bin -> points3D.ply
    pts_bin = os.path.join(raw_scene_dir, "sparse/0/points3D.bin")
    if os.path.exists(pts_bin):
        xyz, rgb = read_colmap_points3d_bin(pts_bin)
        if len(xyz) > 0:
            ply_path = os.path.join(sparse_dir, "points3D.ply")
            write_points3d_ply(ply_path, xyz, rgb)
            print(f"  points3D.ply written ({len(xyz)} points)")
        else:
            print("  WARNING: points3D.bin has 0 points")
    else:
        print("  WARNING: points3D.bin not found; will need --random_init")

    # --- Discover pose directories and build channel stacks ---
    pose_dirs = sorted([
        d for d in os.listdir(raw_scene_dir)
        if d.startswith("pose") and d.endswith("img") and os.path.isdir(os.path.join(raw_scene_dir, d))
    ], key=lambda d: int("".join(c for c in d if c.isdigit())))

    num_poses = len(pose_dirs)
    print(f"\nFound {num_poses} pose directories: {pose_dirs}")
    assert num_poses == len(images), \
        f"Mismatch: {num_poses} pose dirs but {len(images)} COLMAP images"

    images_dir = os.path.join(out_dir, "images")
    ms_dir = os.path.join(out_dir, "images_multispectral")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(ms_dir, exist_ok=True)

    rgb_ch = args.rgb_channels
    print(f"Pseudo-RGB channels: {rgb_ch}")
    print(f"Processing {num_poses} poses x {args.num_channels} channels...")

    for pose_idx, (img_id, img_data) in enumerate(zip(sorted_img_ids, [images[k] for k in sorted_img_ids])):
        pose_dir = pose_dirs[pose_idx]
        tiff_dir = os.path.join(raw_scene_dir, pose_dir, "images")

        tiffs = sorted(os.listdir(tiff_dir))
        assert len(tiffs) == args.num_channels, \
            f"Expected {args.num_channels} TIFFs in {tiff_dir}, got {len(tiffs)}"

        channels = []
        for tiff_name in tiffs:
            img = Image.open(os.path.join(tiff_dir, tiff_name))
            arr = np.array(img)  # uint16, (H, W)
            channels.append(arr)

        stack = np.stack(channels, axis=-1)  # (H, W, 20), uint16

        if do_undistort:
            stack_f = stack.astype(np.float32)
            undistorted = undistort_radial(stack_f, fx, cx, cy, k1)
            ms_float = undistorted / 65535.0
        else:
            ms_float = stack.astype(np.float32) / 65535.0

        stem = img_data["name"].replace(".jpeg", "").replace(".jpg", "")

        np.save(os.path.join(ms_dir, f"{stem}.npy"), ms_float.astype(np.float32))

        pseudo_rgb = (ms_float[:, :, rgb_ch] * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(pseudo_rgb).save(os.path.join(images_dir, f"{stem}.png"))

        print(f"  [{pose_idx+1}/{num_poses}] {pose_dir} -> {stem} "
              f"(stack: {stack.shape}, range: [{ms_float.min():.3f}, {ms_float.max():.3f}])")

    print(f"\nDataset prepared at: {out_dir}")
    print(f"  sparse/0/             - COLMAP PINHOLE cameras + poses (text format)")
    print(f"  images/               - pseudo-RGB PNGs ({num_poses} frames)")
    print(f"  images_multispectral/ - {args.num_channels}-channel float32 .npy ({num_poses} frames)")
    print(f"\nNext steps:")
    print(f"  1. Generate masks:  bash script/prepare_pseudo_label.sh Spec-NeRF/{args.scene} 1")
    print(f"  2. Train:           bash script/train_mms.sh ...")


if __name__ == "__main__":
    main()
