"""
Prepare an X-NeRF scene for Gaussian Grouping training.

Converts raw .npy arrays (ms_imgs, rgb_poses) into COLMAP-format scene with
per-view multispectral .npy files + pseudo-RGB PNGs for mask generation.

Usage:
    python prepare_xnerf_dataset.py --scene penguin
    python prepare_xnerf_dataset.py --scene penguin --focal 510 --rgb_channels 0 4 9

Expected input layout:
    data/xnerf/<scene>/
        ms_imgs.npy    (N, H, W, C) float32 [0,1]
        rgb_poses.npy  (N, 4, 4) float32 camera-to-world

Output layout (COLMAP-compatible):
    data/xnerf/<scene>/
        sparse/0/cameras.txt, images.txt, points3D.txt (PINHOLE model)
        images/<NNNN>.png              (pseudo-RGB for mask generation)
        images_multispectral/<NNNN>.npy (C-channel float32, [0,1])
"""

import argparse
import os
import sys

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as Rot


def c2w_to_colmap(c2w_4x4):
    """Convert a 4x4 camera-to-world matrix (NeRF / OpenGL convention:
    Y up, -Z forward) to COLMAP quaternion + translation (OpenCV: Y down, Z forward)."""
    c2w = c2w_4x4.copy()
    c2w[:3, 1:3] *= -1  # flip Y and Z axes

    w2c = np.linalg.inv(c2w)
    R_w2c = w2c[:3, :3]
    t_w2c = w2c[:3, 3]

    quat = Rot.from_matrix(R_w2c).as_quat()  # [x, y, z, w]
    qvec = np.array([quat[3], quat[0], quat[1], quat[2]])  # COLMAP: [w, x, y, z]
    return qvec, t_w2c


def write_cameras_txt(path, camera_id, width, height, fx, fy, cx, cy):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: 1\n")
        f.write(f"{camera_id} PINHOLE {width} {height} {fx:.10f} {fy:.10f} {cx:.10f} {cy:.10f}\n")


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


def main():
    parser = argparse.ArgumentParser(description="Prepare X-NeRF scene for Gaussian Grouping")
    parser.add_argument("--scene", type=str, default="penguin", help="Scene name")
    parser.add_argument("--data_root", type=str, default="data/xnerf",
                        help="Root of X-NeRF data")
    parser.add_argument("--focal", type=float, default=None,
                        help="Focal length in pixels (default: image width)")
    parser.add_argument("--fov_deg", type=float, default=None,
                        help="Horizontal FOV in degrees (alternative to --focal)")
    parser.add_argument("--rgb_channels", type=int, nargs=3, default=[0, 4, 9],
                        help="Channel indices for pseudo-RGB visualization")
    args = parser.parse_args()

    scene_dir = os.path.join(args.data_root, args.scene)

    ms_path = os.path.join(scene_dir, "ms_imgs.npy")
    poses_path = os.path.join(scene_dir, "rgb_poses.npy")

    if not os.path.exists(ms_path):
        print(f"ERROR: {ms_path} not found")
        sys.exit(1)
    if not os.path.exists(poses_path):
        print(f"ERROR: {poses_path} not found")
        sys.exit(1)

    ms_imgs = np.load(ms_path)
    poses = np.load(poses_path)

    num_views, height, width, num_channels = ms_imgs.shape
    print(f"Scene: {args.scene}")
    print(f"  ms_imgs: {ms_imgs.shape} (dtype={ms_imgs.dtype}, range=[{ms_imgs.min():.3f}, {ms_imgs.max():.3f}])")
    print(f"  poses:   {poses.shape}")
    assert poses.shape[0] == num_views, \
        f"Mismatch: {poses.shape[0]} poses but {num_views} images"

    if args.focal is not None:
        fx = fy = args.focal
    elif args.fov_deg is not None:
        fx = fy = width / (2.0 * np.tan(np.radians(args.fov_deg) / 2.0))
    else:
        fx = fy = float(width)
    cx = width / 2.0
    cy = height / 2.0
    print(f"  Intrinsics: fx={fx:.2f} fy={fy:.2f} cx={cx:.1f} cy={cy:.1f}")

    # --- Write COLMAP text files ---
    sparse_dir = os.path.join(scene_dir, "sparse", "0")
    write_cameras_txt(os.path.join(sparse_dir, "cameras.txt"),
                      camera_id=1, width=width, height=height,
                      fx=fx, fy=fy, cx=cx, cy=cy)
    print(f"  cameras.txt written (PINHOLE)")

    images_data = []
    for i in range(num_views):
        c2w = poses[i]
        qvec, tvec = c2w_to_colmap(c2w)
        img_name = f"{i:04d}.png"
        images_data.append((i + 1, qvec, tvec, 1, img_name))

    write_images_txt(os.path.join(sparse_dir, "images.txt"), images_data)
    print(f"  images.txt written ({num_views} images)")

    write_points3d_txt(os.path.join(sparse_dir, "points3D.txt"))
    print(f"  points3D.txt written (empty — use --random_init for training)")

    # --- Create images/ and images_multispectral/ ---
    images_dir = os.path.join(scene_dir, "images")
    ms_out_dir = os.path.join(scene_dir, "images_multispectral")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(ms_out_dir, exist_ok=True)

    rgb_ch = args.rgb_channels
    for ch in rgb_ch:
        assert ch < num_channels, f"rgb_channel {ch} >= num_channels {num_channels}"
    print(f"  Pseudo-RGB channels: {rgb_ch}")
    print(f"  Processing {num_views} views x {num_channels} channels...")

    for i in range(num_views):
        stem = f"{i:04d}"
        ms_float = ms_imgs[i]  # (H, W, C) float32

        np.save(os.path.join(ms_out_dir, f"{stem}.npy"), ms_float)

        pseudo_rgb = (ms_float[:, :, rgb_ch] * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(pseudo_rgb).save(os.path.join(images_dir, f"{stem}.png"))

        if (i + 1) % 10 == 0 or i == 0 or i == num_views - 1:
            print(f"    [{i+1}/{num_views}] {stem} "
                  f"(range: [{ms_float.min():.3f}, {ms_float.max():.3f}])")

    print(f"\nDataset prepared at: {scene_dir}")
    print(f"  sparse/0/             - COLMAP PINHOLE cameras + poses (text format)")
    print(f"  images/               - pseudo-RGB PNGs ({num_views} frames)")
    print(f"  images_multispectral/ - {num_channels}-channel float32 .npy ({num_views} frames)")
    print(f"\nNext steps:")
    print(f"  1. Generate masks:  bash script/prepare_pseudo_label.sh xnerf/{args.scene} 1")
    print(f"  2. Train:           python train.py -s data/xnerf/{args.scene} --config_file config/phase3/xnerf_baseline.json --random_init")


if __name__ == "__main__":
    main()
