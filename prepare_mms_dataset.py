"""
Prepare an MMS-DATA scene for Gaussian Grouping training.

Converts MMS meta_data.json (multispectral modality) into COLMAP-format files
and creates 9-channel .npy training images + pseudo-RGB PNGs for mask generation.

Usage:
    cd data/multi-modal-studio
    python ../../prepare_mms_dataset.py --scene birdhouse

Expected input layout (from MMS-DATA sample):
    mms-data_birdhouse/mms-data_demosaicked_undistorted/scenes/birdhouse/
        meta_data.json
        modalities/multispectral/  (0000.npy ... 0049.npy)
    mms-data_birdhouse/mms-data_demosaicked_undistorted/scenes/birdhouse/
        pointcloud.ply   (or in mms-data_raw)

Output layout (COLMAP-compatible):
    birdhouse/
        sparse/0/cameras.txt
        sparse/0/images.txt
        sparse/0/points3D.txt  (or .ply)
        images/0000.png         (pseudo-RGB for mask gen / fallback)
        images_multispectral/0000.npy  (9-channel float32, [0,1])
        input/                  (already created by mms.py)
"""

import argparse
import json
import os
import struct
import sys

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as Rot


def camtoworld_to_colmap(camtoworld_3x4):
    """Convert a 3x4 camera-to-world matrix (OpenCV convention) to COLMAP
    quaternion + translation (world-to-camera)."""
    c2w = np.eye(4)
    c2w[:3, :] = np.array(camtoworld_3x4)

    w2c = np.linalg.inv(c2w)
    R_w2c = w2c[:3, :3]
    t_w2c = w2c[:3, 3]

    quat = Rot.from_matrix(R_w2c).as_quat()  # [x, y, z, w]
    qvec = np.array([quat[3], quat[0], quat[1], quat[2]])  # COLMAP: [w, x, y, z]

    return qvec, t_w2c


def write_cameras_txt(path, camera_id, model, width, height, params):
    """Write COLMAP cameras.txt."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: 1\n")
        params_str = " ".join(f"{p:.10f}" for p in params)
        f.write(f"{camera_id} {model} {width} {height} {params_str}\n")


def write_images_txt(path, images_data):
    """Write COLMAP images.txt.
    images_data: list of (image_id, qvec, tvec, camera_id, name)
    """
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
            f.write("\n")  # empty POINTS2D line


def write_points3d_txt(path):
    """Write an empty COLMAP points3D.txt (SfM will init from random or PLY)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        f.write("# Number of points: 0\n")


def convert_pointcloud(src_ply, dst_ply):
    """Copy point cloud PLY, converting to COLMAP-expected format if needed."""
    from plyfile import PlyData, PlyElement

    plydata = PlyData.read(src_ply)
    verts = plydata["vertex"]
    x = np.asarray(verts["x"])
    y = np.asarray(verts["y"])
    z = np.asarray(verts["z"])

    has_color = "red" in verts.data.dtype.names
    if has_color:
        r = np.asarray(verts["red"])
        g = np.asarray(verts["green"])
        b = np.asarray(verts["blue"])
    else:
        r = np.full_like(x, 128, dtype=np.uint8)
        g = np.full_like(x, 128, dtype=np.uint8)
        b = np.full_like(x, 128, dtype=np.uint8)

    has_normals = "nx" in verts.data.dtype.names
    if has_normals:
        nx = np.asarray(verts["nx"])
        ny = np.asarray(verts["ny"])
        nz = np.asarray(verts["nz"])
    else:
        nx = np.zeros_like(x)
        ny = np.zeros_like(x)
        nz = np.zeros_like(x)

    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ]
    elements = np.empty(len(x), dtype=dtype)
    elements["x"] = x
    elements["y"] = y
    elements["z"] = z
    elements["nx"] = nx
    elements["ny"] = ny
    elements["nz"] = nz
    elements["red"] = r.astype(np.uint8)
    elements["green"] = g.astype(np.uint8)
    elements["blue"] = b.astype(np.uint8)

    os.makedirs(os.path.dirname(dst_ply), exist_ok=True)
    el = PlyElement.describe(elements, "vertex")
    PlyData([el]).write(dst_ply)
    print(f"  Point cloud: {len(x)} points -> {dst_ply}")


def main():
    parser = argparse.ArgumentParser(description="Prepare MMS-DATA scene for Gaussian Grouping")
    parser.add_argument("--scene", type=str, default="birdhouse", help="Scene name")
    parser.add_argument("--mms_root", type=str, default=".",
                        help="Root of multi-modal-studio data dir")
    parser.add_argument("--modality", type=str, default="multispectral",
                        help="Which MMS modality to use")
    parser.add_argument("--num_channels", type=int, default=9)
    parser.add_argument("--rgb_channels", type=int, nargs=3, default=[0, 3, 6],
                        help="Channel indices for pseudo-RGB visualization")
    args = parser.parse_args()

    scene = args.scene
    mms_root = args.mms_root

    # Paths
    meta_path = os.path.join(
        mms_root, f"mms-data_{scene}",
        "mms-data_demosaicked_undistorted", "scenes", scene, "meta_data.json"
    )
    ms_dir = os.path.join(
        mms_root, f"mms-data_{scene}",
        "mms-data_demosaicked_undistorted", "scenes", scene,
        "modalities", args.modality,
    )
    pointcloud_paths = [
        os.path.join(mms_root, f"mms-data_{scene}", "mms-data_raw", "scenes", scene, "pointcloud.ply"),
        os.path.join(mms_root, f"mms-data_{scene}", "mms-data_demosaicked_undistorted", "scenes", scene, "pointcloud.ply"),
    ]

    out_dir = os.path.join(mms_root, scene)

    if not os.path.exists(meta_path):
        print(f"ERROR: meta_data.json not found at {meta_path}")
        sys.exit(1)

    with open(meta_path) as f:
        meta = json.load(f)

    modality_data = meta["modalities"][args.modality]
    frames = modality_data["frames"]

    print(f"Scene: {scene}")
    print(f"Modality: {args.modality}")
    print(f"Frames: {len(frames)}")
    print(f"Resolution: {modality_data['width']}x{modality_data['height']}")
    print(f"Camera model: {modality_data['camera_model']}")

    # --- 1. Write COLMAP cameras.txt ---
    sparse_dir = os.path.join(out_dir, "sparse", "0")
    cam_model = modality_data["camera_model"]
    width = modality_data["width"]
    height = modality_data["height"]
    fx = modality_data["fx"]
    fy = modality_data["fy"]
    cx = modality_data["cx"]
    cy = modality_data["cy"]

    write_cameras_txt(
        os.path.join(sparse_dir, "cameras.txt"),
        camera_id=1,
        model="PINHOLE",
        width=width, height=height,
        params=[fx, fy, cx, cy],
    )
    print(f"  cameras.txt written")

    # --- 2. Write COLMAP images.txt ---
    sorted_frames = sorted(frames, key=lambda f: f["frame_id"])
    images_data = []
    for frame in sorted_frames:
        fid = frame["frame_id"]
        fname = frame["file_name"]
        c2w = frame["camtoworld"]
        qvec, tvec = camtoworld_to_colmap(c2w)

        img_name = fname.replace(".npy", ".png")
        images_data.append((fid + 1, qvec, tvec, 1, img_name))

    write_images_txt(os.path.join(sparse_dir, "images.txt"), images_data)
    print(f"  images.txt written ({len(images_data)} images)")

    # --- 3. Write empty points3D.txt (we'll use the PLY directly) ---
    write_points3d_txt(os.path.join(sparse_dir, "points3D.txt"))

    # --- 4. Convert point cloud ---
    pc_src = None
    for p in pointcloud_paths:
        if os.path.exists(p):
            pc_src = p
            break

    if pc_src:
        convert_pointcloud(pc_src, os.path.join(sparse_dir, "points3D.ply"))
    else:
        print("  WARNING: No point cloud found. Will need random_init for training.")

    # --- 5. Create images/ (pseudo-RGB) and images_multispectral/ (9-ch .npy) ---
    images_dir = os.path.join(out_dir, "images")
    ms_out_dir = os.path.join(out_dir, "images_multispectral")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(ms_out_dir, exist_ok=True)

    max_value = 2**16
    rgb_ch = args.rgb_channels

    for frame in sorted_frames:
        fid = frame["frame_id"]
        src_name = frame["file_name"]
        stem = src_name.replace(".npy", "")
        src_path = os.path.join(ms_dir, src_name)

        if not os.path.exists(src_path):
            print(f"  WARNING: {src_path} not found, skipping")
            continue

        ms_img = np.load(src_path)  # [H, W, 9], uint16
        assert ms_img.shape[-1] == args.num_channels, \
            f"Expected {args.num_channels} channels, got {ms_img.shape[-1]}"

        # Normalize to [0, 1] float32
        ms_float = (ms_img.astype(np.float32) / max_value)
        np.save(os.path.join(ms_out_dir, f"{stem}.npy"), ms_float)

        # Pseudo-RGB PNG for visualization / mask generation
        rgb_img = (ms_float[:, :, rgb_ch] * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(rgb_img).save(os.path.join(images_dir, f"{stem}.png"))

    print(f"  images/ and images_multispectral/ written")
    print(f"\nDataset prepared at: {out_dir}")
    print(f"  sparse/0/  - COLMAP format cameras + poses")
    print(f"  images/    - pseudo-RGB PNGs ({len(sorted_frames)} frames)")
    print(f"  images_multispectral/ - 9-channel float32 .npy ({len(sorted_frames)} frames)")
    print(f"\nNext steps:")
    print(f"  1. Generate object masks: bash script/prepare_pseudo_label.sh multi-modal-studio/{scene} 1")
    print(f"  2. Train: python train.py -s data/multi-modal-studio/{scene} --config_file config/gaussian_dataset/train_mms.json ...")


if __name__ == "__main__":
    main()
