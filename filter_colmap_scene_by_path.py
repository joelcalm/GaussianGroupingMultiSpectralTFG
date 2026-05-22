#!/usr/bin/env python3
"""Filter a prepared COLMAP vineyard scene by camera-path consistency.

The intended use is for shared RGB/multispectral vineyard scenes where RGB is the
reference trajectory and registered band frames are accepted only if their camera
center lies close to that trajectory. The script rewrites sparse/0 as a valid
COLMAP TXT model, filters point tracks to kept images, and builds a training scene
with symlinked images/masks plus filtered metadata.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

IMAGE_EXTS = ('.png', '.jpg', '.jpeg')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source_scene', type=Path, required=True)
    p.add_argument('--output_scene', type=Path, required=True)
    p.add_argument('--input_sparse', type=Path, default=None, help='Defaults to source_scene/sparse/0')
    p.add_argument('--reference_prefix', default='rgb')
    p.add_argument('--max_nearest_ref_distance', type=float, default=0.75)
    p.add_argument('--min_track_observations', type=int, default=2)
    p.add_argument('--copy_images', action='store_true')
    p.add_argument('--overwrite', action='store_true')
    return p.parse_args()


def qvec2rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y],
    ], dtype=float)


def frame_number(name: str) -> int:
    stem = Path(name).stem
    return int(stem.split('_')[-1])


def band_name(name: str) -> str:
    return Path(name).stem.split('_', 1)[0]


def read_cameras(path: Path):
    comments, rows = [], []
    for line in path.read_text().splitlines():
        if line.startswith('#'):
            comments.append(line)
        elif line.strip():
            rows.append(line)
    return comments, rows


def read_images(path: Path):
    comments, images = [], []
    lines = path.read_text().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line.startswith('#'):
            comments.append(line)
            i += 1
            continue
        parts = line.split()
        if len(parts) < 10 or not parts[9].lower().endswith(IMAGE_EXTS):
            i += 1
            continue
        points_line = lines[i + 1] if i + 1 < len(lines) else ''
        q = np.array(list(map(float, parts[1:5])))
        t = np.array(list(map(float, parts[5:8])))
        center = -qvec2rotmat(q).T @ t
        images.append({
            'id': int(parts[0]),
            'camera_id': int(parts[8]),
            'name': parts[9],
            'band': band_name(parts[9]),
            'frame': frame_number(parts[9]),
            'center': center,
            'line': line,
            'points_line': points_line,
        })
        i += 2
    return comments, images


def parse_points2d(line: str):
    vals = line.split()
    triples = []
    for i in range(0, len(vals), 3):
        if i + 2 >= len(vals):
            break
        triples.append([vals[i], vals[i + 1], int(vals[i + 2])])
    return triples


def format_points2d(triples):
    return ' '.join(f'{x} {y} {pid}' for x, y, pid in triples)


def read_points3d(path: Path):
    comments, points = [], []
    for line in path.read_text().splitlines():
        if not line:
            continue
        if line.startswith('#'):
            comments.append(line)
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        track = []
        tail = parts[8:]
        for i in range(0, len(tail), 2):
            if i + 1 >= len(tail):
                break
            track.append((int(tail[i]), int(tail[i + 1])))
        points.append({'id': int(parts[0]), 'head': parts[:8], 'track': track})
    return comments, points


def link_or_copy(src: Path, dst: Path, copy: bool):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        os.symlink(str(src.resolve()), dst)


def load_json(path: Path, default: Any):
    if not path.exists():
        return default
    with path.open() as f:
        return json.load(f)


def main():
    args = parse_args()
    src = args.source_scene.resolve()
    out = args.output_scene.resolve()
    sparse = (args.input_sparse or (src / 'sparse' / '0')).resolve()
    if out.exists():
        if not args.overwrite:
            raise FileExistsError(f'{out} exists; pass --overwrite')
        shutil.rmtree(out)
    out.mkdir(parents=True)

    cam_comments, camera_rows = read_cameras(sparse / 'cameras.txt')
    img_comments, images = read_images(sparse / 'images.txt')
    point_comments, points = read_points3d(sparse / 'points3D.txt')

    ref = sorted([im for im in images if im['band'] == args.reference_prefix], key=lambda im: im['frame'])
    if not ref:
        raise RuntimeError(f'No reference images with prefix {args.reference_prefix!r}')
    ref_centers = np.stack([im['center'] for im in ref])

    keep_ids = set()
    image_report = []
    for im in images:
        nearest = float(np.min(np.linalg.norm(ref_centers - im['center'], axis=1)))
        keep = im['band'] == args.reference_prefix or nearest <= args.max_nearest_ref_distance
        if keep:
            keep_ids.add(im['id'])
        image_report.append({
            'image_id': im['id'],
            'name': im['name'],
            'band': im['band'],
            'frame': im['frame'],
            'nearest_reference_distance': nearest,
            'kept': keep,
            'center': [float(v) for v in im['center']],
        })

    # Filter points first, then clear image references to dropped points.
    kept_point_ids = set()
    filtered_points = []
    for pt in points:
        track = [(iid, pidx) for iid, pidx in pt['track'] if iid in keep_ids]
        if len(track) >= args.min_track_observations:
            kept_point_ids.add(pt['id'])
            filtered_points.append({**pt, 'track': track})

    kept_images = []
    for im in images:
        if im['id'] not in keep_ids:
            continue
        triples = parse_points2d(im['points_line'])
        for tri in triples:
            if tri[2] not in kept_point_ids:
                tri[2] = -1
        kept = dict(im)
        kept['points_line'] = format_points2d(triples)
        kept_images.append(kept)

    used_camera_ids = {im['camera_id'] for im in kept_images}
    kept_camera_rows = [row for row in camera_rows if int(row.split()[0]) in used_camera_ids]

    sparse_out = out / 'sparse' / '0'
    sparse_out.mkdir(parents=True)
    (sparse_out / 'cameras.txt').write_text('\n'.join(cam_comments + kept_camera_rows) + '\n')
    with (sparse_out / 'images.txt').open('w') as f:
        f.write('# Image list with two lines of data per image:\n')
        f.write('#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n')
        f.write('#   POINTS2D[] as (X, Y, POINT3D_ID)\n')
        f.write(f'# Number of images: {len(kept_images)}\n')
        for im in kept_images:
            f.write(im['line'] + '\n')
            f.write(im['points_line'] + '\n')
    with (sparse_out / 'points3D.txt').open('w') as f:
        f.write('# 3D point list with one line of data per point:\n')
        f.write('#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n')
        f.write(f'# Number of points: {len(filtered_points)}\n')
        for pt in filtered_points:
            track = ' '.join(f'{iid} {pidx}' for iid, pidx in pt['track'])
            f.write(' '.join(pt['head'] + ([track] if track else [])) + '\n')

    kept_names = {im['name'] for im in kept_images}
    for im in kept_images:
        src_img = src / 'images' / im['name']
        if src_img.exists():
            link_or_copy(src_img, out / 'images' / im['name'], args.copy_images)
        src_mask = src / 'object_mask' / im['name']
        if src_mask.exists():
            link_or_copy(src_mask, out / 'object_mask' / im['name'], args.copy_images)
        if im['band'] == args.reference_prefix:
            src_rgb = src / 'images_rgb' / im['name']
            if src_rgb.exists():
                link_or_copy(src_rgb, out / 'images_rgb' / im['name'], args.copy_images)

    for name in ['videos_manifest.json', 'extraction_report.json', 'partial_channels_summary.json']:
        if (src / name).exists():
            shutil.copy2(src / name, out / name)
    if (src / 'metadata').is_dir():
        shutil.copytree(src / 'metadata', out / 'metadata')

    for meta_name in ['band_info.json', 'frame_info.json']:
        data = load_json(src / meta_name, {})
        filtered = {}
        for key, val in data.items():
            stem = Path(key).stem
            png = f'{stem}.png'
            if png in kept_names or key in kept_names or stem in {Path(n).stem for n in kept_names}:
                filtered[key] = val
        (out / meta_name).write_text(json.dumps(filtered, indent=2))

    counts = Counter(im['band'] for im in kept_images)
    source_counts = Counter(im['band'] for im in images)
    report = {
        'source_scene': str(src),
        'input_sparse': str(sparse),
        'output_scene': str(out),
        'reference_prefix': args.reference_prefix,
        'max_nearest_ref_distance': args.max_nearest_ref_distance,
        'min_track_observations': args.min_track_observations,
        'source_counts': dict(sorted(source_counts.items())),
        'kept_counts': dict(sorted(counts.items())),
        'dropped_counts': dict(sorted((band, source_counts[band] - counts.get(band, 0)) for band in source_counts)),
        'source_num_points3D': len(points),
        'kept_num_points3D': len(filtered_points),
        'images': image_report,
    }
    audit_dir = out / 'colmap_path_filter'
    audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / 'filter_report.json').write_text(json.dumps(report, indent=2))
    with (audit_dir / 'filter_report.csv').open('w') as f:
        f.write('kept,band,frame,name,nearest_reference_distance,x,y,z\n')
        for row in image_report:
            c = row['center']
            f.write(f"{int(row['kept'])},{row['band']},{row['frame']},{row['name']},{row['nearest_reference_distance']},{c[0]},{c[1]},{c[2]}\n")

    summary = load_json(out / 'metadata' / 'registered_images_summary.json', {})
    summary.update({
        'source_scene_before_path_filter': str(src),
        'path_filter_reference_prefix': args.reference_prefix,
        'path_filter_max_nearest_ref_distance': args.max_nearest_ref_distance,
        'path_filter_kept_images': len(kept_images),
        'path_filter_kept_per_band': dict(sorted(counts.items())),
    })
    (out / 'metadata').mkdir(parents=True, exist_ok=True)
    (out / 'metadata' / 'registered_images_summary.json').write_text(json.dumps(summary, indent=2))

    print(json.dumps({k: report[k] for k in ['source_counts', 'kept_counts', 'dropped_counts', 'source_num_points3D', 'kept_num_points3D']}, indent=2))
    print(f'Wrote filtered scene: {out}')


if __name__ == '__main__':
    main()
