import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData, PlyElement


C0 = 0.28209479177387814


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def id_to_rgb(idx):
    if idx <= 0:
        return np.array([0, 0, 0], dtype=np.uint8)
    golden_ratio = 1.6180339887
    h = (idx * golden_ratio) % 1.0
    s = 0.55 + (idx % 2) * 0.35
    l = 0.52

    def hue_to_rgb(p, q, t):
        if t < 0:
            t += 1
        if t > 1:
            t -= 1
        if t < 1 / 6:
            return p + (q - p) * 6 * t
        if t < 1 / 2:
            return q
        if t < 2 / 3:
            return p + (q - p) * (2 / 3 - t) * 6
        return p

    q = l * (1 + s) if l < 0.5 else l + s - l * s
    p = 2 * l - q
    rgb = [hue_to_rgb(p, q, h + 1 / 3), hue_to_rgb(p, q, h), hue_to_rgb(p, q, h - 1 / 3)]
    return np.array([int(v * 255) for v in rgb], dtype=np.uint8)


def sanitize_name(value):
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return value or "selection"


def split_values(values):
    out = []
    for value in values or []:
        out.extend(part.strip() for part in str(value).split(",") if part.strip())
    return out


def parse_int_values(values):
    out = []
    for value in split_values(values):
        out.append(int(value))
    return out


def find_iteration(model_path, iteration):
    point_root = Path(model_path) / "point_cloud"
    if iteration != -1:
        return int(iteration)
    candidates = []
    for path in point_root.glob("iteration_*"):
        try:
            candidates.append(int(path.name.split("_")[-1]))
        except ValueError:
            pass
    if not candidates:
        raise FileNotFoundError(f"No iteration_* directories found in {point_root}")
    return max(candidates)


def load_json(path, default):
    path = Path(path)
    if not path.exists():
        return default
    with open(path) as f:
        return json.load(f)


def load_metadata(model_path):
    model_path = Path(model_path)
    class_map_raw = load_json(model_path / "class_map.json", {})
    instance_map_raw = load_json(model_path / "instance_label_map.json", {})

    class_name_to_id = {}
    class_id_to_name = {}
    for name, idx in class_map_raw.items():
        class_name_to_id[str(name).lower()] = int(idx)
        class_id_to_name[int(idx)] = str(name)

    instance_map = {}
    for key, row in instance_map_raw.items():
        inst_id = int(key)
        class_id = int(row.get("class_id", inst_id))
        class_name = str(row.get("class_name", class_id_to_name.get(class_id, class_id)))
        label = str(row.get("label", f"{class_name}_{inst_id:04d}"))
        instance_map[inst_id] = {
            "class_id": class_id,
            "class_name": class_name,
            "label": label,
        }
        class_name_to_id.setdefault(class_name.lower(), class_id)
        class_id_to_name.setdefault(class_id, class_name)

    return class_name_to_id, class_id_to_name, instance_map


def find_default_merge_map(model_path, iteration):
    iteration = find_iteration(model_path, iteration)
    path = Path(model_path) / "vine_tracklet_merges" / f"iteration_{iteration}" / "vine_tracklet_merge_map.json"
    return path if path.exists() else None


def load_physical_vine_map(model_path, iteration, merge_map_path=None):
    path = Path(merge_map_path) if merge_map_path else find_default_merge_map(model_path, iteration)
    if path is None or not path.exists():
        return {}, None
    data = load_json(path, {})
    out = {}
    for row in data.get("physical_vines", []):
        physical_id = int(row["physical_vine_id"])
        out[physical_id] = {
            "physical_vine_id": physical_id,
            "label": f"physical_vine_{physical_id:04d}",
            "member_instance_ids": [int(v) for v in row.get("member_instance_ids", [])],
            "num_tracklets": int(row.get("num_tracklets", 0)),
            "points": int(row.get("points", 0)),
        }
    return out, path


def load_scene_data(model_path, iteration):
    iteration = find_iteration(model_path, iteration)
    iter_dir = Path(model_path) / "point_cloud" / f"iteration_{iteration}"
    ply_path = iter_dir / "point_cloud.ply"
    classifier_path = iter_dir / "classifier.pth"
    if not ply_path.exists():
        raise FileNotFoundError(ply_path)
    if not classifier_path.exists():
        raise FileNotFoundError(classifier_path)

    ply = PlyData.read(str(ply_path))
    vertices = ply["vertex"].data
    obj_names = sorted(
        [p.name for p in ply["vertex"].properties if p.name.startswith("obj_dc_")],
        key=lambda name: int(name.split("_")[-1]),
    )
    if not obj_names:
        raise ValueError(f"{ply_path} has no obj_dc_* fields")
    objects = np.stack([np.asarray(vertices[name]) for name in obj_names], axis=1).astype(np.float32)

    state = torch.load(str(classifier_path), map_location="cpu")
    weight = state["weight"].detach().cpu().numpy().reshape(state["weight"].shape[0], -1)
    bias = state["bias"].detach().cpu().numpy() if "bias" in state else np.zeros(weight.shape[0], dtype=np.float32)
    logits = objects @ weight.T + bias[None, :]
    pred_instance = np.argmax(logits, axis=1).astype(np.int32)
    pred_score = torch.softmax(torch.from_numpy(logits), dim=1).max(dim=1).values.numpy().astype(np.float32)

    xyz = np.stack([np.asarray(vertices["x"]), np.asarray(vertices["y"]), np.asarray(vertices["z"])], axis=1).astype(np.float32)
    opacity = sigmoid(np.asarray(vertices["opacity"], dtype=np.float32))

    f_dc_names = [f"f_dc_{i}" for i in range(3)]
    if all(name in vertices.dtype.names for name in f_dc_names):
        rgb = np.stack([np.asarray(vertices[name]) for name in f_dc_names], axis=1) * C0 + 0.5
        rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    else:
        rgb = np.repeat(np.array([[180, 180, 180]], dtype=np.uint8), xyz.shape[0], axis=0)

    return {
        "iteration": iteration,
        "iter_dir": iter_dir,
        "ply": ply,
        "vertices": vertices,
        "xyz": xyz,
        "rgb": rgb,
        "opacity": opacity.astype(np.float32),
        "pred_instance": pred_instance,
        "pred_score": pred_score,
    }


def normalize_class_name(name):
    lowered = str(name).strip().lower()
    aliases = {
        "vine": "vine_plant",
        "vines": "vine_plant",
        "plant": "vine_plant",
        "plants": "vine_plant",
        "post": "wooden_post",
        "posts": "wooden_post",
        "wall": "stone_wall",
        "vegetation": "shrub_or_other_vegetation",
        "shrub": "shrub_or_other_vegetation",
        "shrubs": "shrub_or_other_vegetation",
    }
    return aliases.get(lowered, lowered)


def selection_from_args(args, class_name_to_id, instance_map, pred_instance, physical_vine_map=None):
    physical_vine_map = physical_vine_map or {}
    class_names = [normalize_class_name(v) for v in split_values(args.class_name)]
    class_ids = set(parse_int_values(args.class_id))
    labels = {v.lower() for v in split_values(args.label)}
    instance_ids = set(parse_int_values(args.instance_id) + parse_int_values(args.object_id))
    object_indexes = parse_int_values(args.object_index)
    physical_vine_ids = parse_int_values(getattr(args, "physical_vine_id", []))

    query = " ".join(split_values(args.query)).lower()
    if query:
        query_has_vine = re.search(r"\bvines?\b|vine_plant|physical_vine", query) is not None
        numbers = [int(token) for token in re.findall(r"\b\d+\b", query)]
        if query_has_vine:
            class_names.append("vine_plant")
            if physical_vine_map and numbers:
                physical_vine_ids.extend(numbers)
            else:
                object_indexes.extend(numbers)
        else:
            instance_ids.update(numbers)

    for name in class_names:
        if name not in class_name_to_id:
            valid = ", ".join(sorted(class_name_to_id))
            raise ValueError(f"Unknown class '{name}'. Valid classes: {valid}")
        class_ids.add(class_name_to_id[name])

    candidate_instances = []
    for inst_id, row in sorted(instance_map.items()):
        if class_ids and row["class_id"] not in class_ids:
            continue
        if labels and row["label"].lower() not in labels:
            continue
        candidate_instances.append(inst_id)

    selected_instances = set(instance_ids)
    selected_physical_vines = []
    for physical_id in physical_vine_ids:
        if physical_id not in physical_vine_map:
            valid = ", ".join(str(v) for v in sorted(physical_vine_map))
            raise ValueError(f"Unknown physical vine '{physical_id}'. Valid physical vine IDs: {valid}")
        selected_instances.update(physical_vine_map[physical_id]["member_instance_ids"])
        selected_physical_vines.append(physical_vine_map[physical_id])

    if object_indexes:
        if not candidate_instances:
            candidate_instances = sorted(int(v) for v in np.unique(pred_instance))
        for object_index in object_indexes:
            if object_index < 1 or object_index > len(candidate_instances):
                raise ValueError(f"object-index {object_index} is outside 1..{len(candidate_instances)}")
            selected_instances.add(candidate_instances[object_index - 1])
    elif (class_ids or labels) and not selected_instances:
        selected_instances.update(candidate_instances)

    if not selected_instances:
        selected_instances.update(int(v) for v in np.unique(pred_instance))

    return selected_instances, selected_physical_vines


def summarize_instances(pred_instance, instance_map):
    rows = []
    unique, counts = np.unique(pred_instance, return_counts=True)
    count_by_id = {int(k): int(v) for k, v in zip(unique, counts)}
    class_counts = {}
    vine_index = 0
    for inst_id in sorted(instance_map):
        row = instance_map[inst_id]
        class_name = row["class_name"]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
        display_index = class_counts[class_name]
        if class_name == "vine_plant":
            vine_index = display_index
        else:
            vine_index = None
        rows.append({
            "instance_id": inst_id,
            "class_id": row["class_id"],
            "class_name": class_name,
            "label": row["label"],
            "class_index": display_index,
            "vine_index": vine_index,
            "points": count_by_id.get(inst_id, 0),
        })
    return rows


def print_listing(rows):
    by_class = {}
    for row in rows:
        by_class.setdefault(row["class_name"], []).append(row)
    for class_name in sorted(by_class):
        items = by_class[class_name]
        total_points = sum(row["points"] for row in items)
        print(f"{class_name}: {len(items)} instances, {total_points} predicted points")
        for row in items:
            print(
                f"  #{row['class_index']:03d}  instance_id={row['instance_id']:03d}  "
                f"points={row['points']:7d}  label={row['label']}"
            )


def print_physical_listing(physical_vine_map):
    if not physical_vine_map:
        return
    print("physical_vines:")
    for physical_id, row in sorted(physical_vine_map.items()):
        members = ",".join(str(v) for v in row["member_instance_ids"])
        print(
            f"  vine {physical_id:03d}  tracklets={row['num_tracklets']:3d}  "
            f"points={row['points']:7d}  instance_ids={members}"
        )


def colors_for_points(pred_instance, instance_map, mode, rgb):
    if mode == "rgb":
        return rgb
    colors = np.zeros((pred_instance.shape[0], 3), dtype=np.uint8)
    for inst_id in np.unique(pred_instance):
        inst_id = int(inst_id)
        if mode == "class":
            color_id = instance_map.get(inst_id, {}).get("class_id", inst_id)
        else:
            color_id = inst_id
        colors[pred_instance == inst_id] = id_to_rgb(color_id)
    return colors


def write_point_cloud(path, xyz, colors, opacity, pred_instance, instance_map):
    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ("opacity", "f4"), ("pred_instance_id", "i4"), ("class_id", "i4"),
    ]
    elements = np.empty(xyz.shape[0], dtype=dtype)
    class_ids = np.array([instance_map.get(int(i), {}).get("class_id", -1) for i in pred_instance], dtype=np.int32)
    elements["x"], elements["y"], elements["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    elements["red"], elements["green"], elements["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]
    elements["opacity"] = opacity
    elements["pred_instance_id"] = pred_instance
    elements["class_id"] = class_ids
    PlyData([PlyElement.describe(elements, "vertex")]).write(str(path))


def write_gaussian_ply(path, source_ply, vertices):
    vertex = PlyElement.describe(vertices, "vertex")
    PlyData([vertex], text=source_ply.text, byte_order=source_ply.byte_order).write(str(path))


def write_html(path, xyz, colors, max_points):
    if xyz.shape[0] > max_points:
        rng = np.random.default_rng(7)
        keep = np.sort(rng.choice(xyz.shape[0], max_points, replace=False))
        xyz = xyz[keep]
        colors = colors[keep]

    center = xyz.mean(axis=0) if xyz.size else np.zeros(3)
    scale = float(np.linalg.norm(xyz.max(axis=0) - xyz.min(axis=0))) if xyz.size else 1.0
    if scale <= 0:
        scale = 1.0
    pts = ((xyz - center) / scale).astype(np.float32)
    pos_list = ",".join(f"{v:.6g}" for v in pts.reshape(-1))
    color_list = ",".join(str(int(v)) for v in colors.reshape(-1))

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Selected Gaussians</title>
  <style>
    html, body {{ margin: 0; height: 100%; background: #111; color: #ddd; font: 13px system-ui, sans-serif; }}
    #hud {{ position: fixed; left: 12px; top: 10px; background: rgba(0,0,0,.55); padding: 8px 10px; border-radius: 6px; }}
    canvas {{ width: 100vw; height: 100vh; display: block; }}
  </style>
</head>
<body>
<canvas id="c"></canvas>
<div id="hud">{xyz.shape[0]} points. Drag to rotate, wheel to zoom.</div>
<script>
const positions = new Float32Array([{pos_list}]);
const colors = new Uint8Array([{color_list}]);
const canvas = document.getElementById('c');
const gl = canvas.getContext('webgl');
if (!gl) throw new Error('WebGL unavailable');
const vs = `
attribute vec3 p; attribute vec3 c; uniform mat4 m; varying vec3 vc;
void main() {{ gl_Position = m * vec4(p, 1.0); gl_PointSize = 2.0; vc = c / 255.0; }}
`;
const fs = `precision mediump float; varying vec3 vc; void main() {{ gl_FragColor = vec4(vc, 1.0); }}`;
function shader(type, src) {{ const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s); return s; }}
const prog = gl.createProgram(); gl.attachShader(prog, shader(gl.VERTEX_SHADER, vs)); gl.attachShader(prog, shader(gl.FRAGMENT_SHADER, fs)); gl.linkProgram(prog); gl.useProgram(prog);
function buffer(data, attr, size, type) {{ const b = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, b); gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW); const loc = gl.getAttribLocation(prog, attr); gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, size, type, false, 0, 0); }}
buffer(positions, 'p', 3, gl.FLOAT); buffer(colors, 'c', 3, gl.UNSIGNED_BYTE);
const mLoc = gl.getUniformLocation(prog, 'm');
let rx = -0.8, ry = 0.0, zoom = 2.4, dragging = false, lx = 0, ly = 0;
canvas.onmousedown = e => {{ dragging = true; lx = e.clientX; ly = e.clientY; }};
canvas.onmouseup = () => dragging = false;
canvas.onmousemove = e => {{ if (!dragging) return; ry += (e.clientX - lx) * .01; rx += (e.clientY - ly) * .01; lx = e.clientX; ly = e.clientY; draw(); }};
canvas.onwheel = e => {{ zoom *= Math.exp(e.deltaY * .001); draw(); e.preventDefault(); }};
function mat() {{
  const cx=Math.cos(rx), sx=Math.sin(rx), cy=Math.cos(ry), sy=Math.sin(ry), z=zoom;
  return new Float32Array([cy, sx*sy, cx*sy, 0, 0, cx, -sx, 0, -sy, sx*cy, cx*cy, 0, 0, 0, -z, 1]);
}}
function draw() {{
  canvas.width = innerWidth * devicePixelRatio; canvas.height = innerHeight * devicePixelRatio;
  gl.viewport(0, 0, canvas.width, canvas.height); gl.clearColor(.07, .07, .07, 1); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  gl.uniformMatrix4fv(mLoc, false, mat()); gl.drawArrays(gl.POINTS, 0, positions.length / 3);
}}
onresize = draw; draw();
</script>
</body>
</html>
"""
    with open(path, "w") as f:
        f.write(html)


def export_selection(args):
    class_name_to_id, class_id_to_name, instance_map = load_metadata(args.model_path)
    data = load_scene_data(args.model_path, args.iteration)
    physical_vine_map, physical_merge_path = load_physical_vine_map(
        args.model_path,
        data["iteration"],
        getattr(args, "merge_map", None),
    )
    rows = summarize_instances(data["pred_instance"], instance_map)

    if args.list:
        print_listing(rows)
        print_physical_listing(physical_vine_map)
        if not args.export:
            return None

    selected_instances, selected_physical_vines = selection_from_args(
        args,
        class_name_to_id,
        instance_map,
        data["pred_instance"],
        physical_vine_map,
    )
    mask = np.isin(data["pred_instance"], np.array(sorted(selected_instances), dtype=np.int32))
    if args.min_opacity > 0:
        mask &= data["opacity"] >= args.min_opacity

    color_mode = args.color_by
    colors = colors_for_points(data["pred_instance"], instance_map, color_mode, data["rgb"])

    selected_rows = [row for row in rows if row["instance_id"] in selected_instances]
    if args.output_name:
        name = sanitize_name(args.output_name)
    elif len(selected_physical_vines) == 1:
        name = sanitize_name(selected_physical_vines[0]["label"])
    elif selected_rows:
        classes = sorted({row["class_name"] for row in selected_rows})
        name = sanitize_name("_".join(classes[:3]))
        if len(selected_instances) == 1:
            only = selected_rows[0]
            name = sanitize_name(only["label"])
    else:
        name = "selection"

    out_dir = Path(args.output_dir or Path(args.model_path) / "selections" / f"iteration_{data['iteration']}")
    out_dir.mkdir(parents=True, exist_ok=True)
    gaussian_path = out_dir / f"{name}_gaussians.ply"
    point_path = out_dir / f"{name}_points.ply"
    summary_path = out_dir / f"{name}_summary.json"
    html_path = out_dir / f"{name}.html"

    write_gaussian_ply(gaussian_path, data["ply"], data["vertices"][mask])
    write_point_cloud(
        point_path,
        data["xyz"][mask],
        colors[mask],
        data["opacity"][mask],
        data["pred_instance"][mask],
        instance_map,
    )
    if args.html:
        write_html(html_path, data["xyz"][mask], colors[mask], args.max_html_points)

    summary = {
        "model_path": str(args.model_path),
        "iteration": data["iteration"],
        "selected_instances": sorted(int(v) for v in selected_instances),
        "selected_physical_vines": selected_physical_vines,
        "physical_merge_map": str(physical_merge_path) if physical_merge_path else None,
        "selected_points": int(mask.sum()),
        "total_points": int(mask.shape[0]),
        "min_opacity": args.min_opacity,
        "color_by": color_mode,
        "outputs": {
            "gaussians_ply": str(gaussian_path),
            "points_ply": str(point_path),
            "html": str(html_path) if args.html else None,
        },
        "instances": selected_rows,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Selected {summary['selected_points']} / {summary['total_points']} Gaussians")
    print(f"Wrote full Gaussian PLY: {gaussian_path}")
    print(f"Wrote point-cloud PLY:   {point_path}")
    if args.html:
        print(f"Wrote HTML viewer:      {html_path}")
    print(f"Wrote summary:          {summary_path}")
    return summary


def launch_app(args):
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError("Gradio is not installed. Use the CLI export mode, or install gradio to run --app.") from exc

    class_name_to_id, class_id_to_name, instance_map = load_metadata(args.model_path)
    data = load_scene_data(args.model_path, args.iteration)
    physical_vine_map, physical_merge_path = load_physical_vine_map(args.model_path, data["iteration"], args.merge_map)
    rows = summarize_instances(data["pred_instance"], instance_map)
    class_names = sorted({row["class_name"] for row in rows})
    labels = [row["label"] for row in rows]
    physical_labels = [f"vine {pid}" for pid in sorted(physical_vine_map)]

    def run(class_name, label, physical_vines, instance_ids, object_indexes, color_by, min_opacity):
        local_args = argparse.Namespace(**vars(args))
        local_args.class_name = class_name or []
        local_args.label = label or []
        local_args.physical_vine_id = [v.split()[-1] for v in (physical_vines or [])]
        local_args.instance_id = [instance_ids] if instance_ids else []
        local_args.object_id = []
        local_args.object_index = [object_indexes] if object_indexes else []
        local_args.color_by = color_by
        local_args.min_opacity = float(min_opacity or 0.0)
        local_args.html = True
        local_args.list = False
        local_args.export = True
        summary = export_selection(local_args)
        return json.dumps(summary, indent=2), summary["outputs"]["html"]

    with gr.Blocks(title="Gaussian Selector") as demo:
        gr.Markdown("# Gaussian Selector")
        with gr.Row():
            class_name = gr.Dropdown(class_names, multiselect=True, label="Classes")
            label = gr.Dropdown(labels, multiselect=True, label="Instance labels")
            physical_vines = gr.Dropdown(physical_labels, multiselect=True, label="Physical vines")
        with gr.Row():
            instance_ids = gr.Textbox(label="Instance IDs, comma-separated")
            object_indexes = gr.Textbox(label="Class-local object indexes, comma-separated")
        with gr.Row():
            color_by = gr.Radio(["instance", "class", "rgb"], value=args.color_by, label="Color by")
            min_opacity = gr.Number(value=args.min_opacity, label="Min opacity")
        button = gr.Button("Export selection")
        summary = gr.Code(label="Summary", language="json")
        html_file = gr.File(label="HTML viewer")
        button.click(run, [class_name, label, physical_vines, instance_ids, object_indexes, color_by, min_opacity], [summary, html_file])

    demo.launch(server_name=args.host, server_port=args.port)


def build_parser():
    parser = argparse.ArgumentParser(description="Select and export Gaussians by predicted class or instance.")
    parser.add_argument("-m", "--model_path", required=True)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--class-name", nargs="*", default=[], help="Class names, e.g. vine_plant or vines")
    parser.add_argument("--class-id", nargs="*", default=[])
    parser.add_argument("--label", nargs="*", default=[], help="Instance labels, e.g. vine_plant_0013")
    parser.add_argument("--instance-id", nargs="*", default=[], help="Predicted instance IDs")
    parser.add_argument("--object-id", nargs="*", default=[], help="Alias for --instance-id")
    parser.add_argument("--physical-vine-id", nargs="*", default=[], help="Physical vine IDs from vine_tracklet_merge_map.json")
    parser.add_argument("--merge-map", default=None, help="Physical vine merge map. Defaults to model_path/vine_tracklet_merges/iteration_*/vine_tracklet_merge_map.json")
    parser.add_argument("--object-index", nargs="*", default=[], help="1-based index within the selected class, e.g. --class-name vine_plant --object-index 3")
    parser.add_argument("--query", nargs="*", default=[], help='Simple text query, e.g. "only show vines" or "vine 3"')
    parser.add_argument("--min-opacity", type=float, default=0.0)
    parser.add_argument("--color-by", choices=["instance", "class", "rgb"], default="instance")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--html", action="store_true")
    parser.add_argument("--max-html-points", type=int, default=200000)
    parser.add_argument("--list", action="store_true", help="List classes/instances and their predicted point counts")
    parser.add_argument("--export", action="store_true", help="Export even when --list is used")
    parser.add_argument("--app", action="store_true", help="Launch a small Gradio selection app")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.app:
        launch_app(args)
    else:
        export_selection(args)
