#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

FIELDS = [
    "frame",
    "group",
    "target_vine_a_leaf_ids",
    "target_vine_a_trunk_ids",
    "target_vine_b_leaf_ids",
    "target_vine_b_trunk_ids",
    "reference_wooden_post_ids",
    "notes",
]
EDIT_FIELDS = FIELDS[2:7]

HTML = r'''<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Target Vine ID Editor</title>
<style>
:root{color-scheme:dark;--bg:#111;--panel:#1b1b1b;--line:#333;--text:#eee;--muted:#aaa;--accent:#77c7ff}
*{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--text);font-family:Arial,sans-serif} header{position:sticky;top:0;background:#0d0d0d;border-bottom:1px solid var(--line);padding:12px 18px;z-index:2;display:flex;gap:16px;align-items:center;flex-wrap:wrap} button{background:#26394a;color:#fff;border:1px solid #45647f;border-radius:4px;padding:7px 10px;cursor:pointer} button:hover{background:#31506a}.danger{background:#4a2626;border-color:#804545}.ok{background:#21492a;border-color:#3d7d49}.wrap{display:grid;grid-template-columns:minmax(520px,1fr) 520px;gap:14px;padding:14px}.frame{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:10px}.frame h2{font-size:16px;margin:0 0 8px}.imgbox{max-height:76vh;overflow:auto;border:1px solid var(--line);background:#000}.imgbox img{width:100%;display:block}.side{position:sticky;top:66px;align-self:start;background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:12px;max-height:calc(100vh - 80px);overflow:auto}.field{display:grid;grid-template-columns:170px 1fr;gap:8px;align-items:center;margin:8px 0}.field input{width:100%;background:#101010;color:#fff;border:1px solid #444;border-radius:4px;padding:7px}.field input.active{outline:2px solid var(--accent)}.candidates{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:12px}.cand{border:1px solid #444;background:#171717;border-radius:5px;padding:7px}.cand strong{font-size:15px}.cand small{display:block;color:var(--muted);font-size:11px;margin-top:3px}.cand button{margin-top:6px;width:100%;padding:5px}.nav{display:flex;gap:8px;align-items:center}.muted{color:var(--muted)}.toast{color:#9ee6a7}.warn{color:#ffca7a}.links a{color:var(--accent);margin-right:10px} select{background:#111;color:#fff;border:1px solid #444;border-radius:4px;padding:6px}
@media(max-width:1100px){.wrap{grid-template-columns:1fr}.side{position:static;max-height:none}}
</style>
</head>
<body>
<header>
  <strong>Target Vine ID Editor</strong>
  <span id="count" class="muted"></span>
  <div class="nav"><button onclick="prevFrame()">Prev</button><select id="frameSelect" onchange="showFrame(this.value)"></select><button onclick="nextFrame()">Next</button></div>
  <button class="ok" onclick="saveCsv()">Save CSV</button>
  <span id="status" class="toast"></span>
</header>
<div class="wrap">
  <main class="frame">
    <h2 id="title"></h2>
    <div class="links"><a id="overlayLink" target="_blank">Open overlay</a><a id="rgbLink" target="_blank">Open RGB</a><a id="maskLink" target="_blank">Open color mask</a></div>
    <p class="muted">Click a field, then click candidate IDs below to append them. Use commas for split masks.</p>
    <div class="imgbox"><img id="overlay"></div>
  </main>
  <aside class="side">
    <h2>Mapping</h2>
    <div id="fields"></div>
    <h2>Candidate source IDs</h2>
    <p class="muted">These are the source leaf/trunk/post labels detected in this frame.</p>
    <div id="candidates" class="candidates"></div>
  </aside>
</div>
<script>
let data=null, current=0, activeField='target_vine_a_leaf_ids';
const editFields=['target_vine_a_leaf_ids','target_vine_a_trunk_ids','target_vine_b_leaf_ids','target_vine_b_trunk_ids','reference_wooden_post_ids'];
const labels={target_vine_a_leaf_ids:'vine A leaf',target_vine_a_trunk_ids:'vine A trunk',target_vine_b_leaf_ids:'vine B leaf',target_vine_b_trunk_ids:'vine B trunk',reference_wooden_post_ids:'reference post',notes:'notes'};
async function init(){data=await (await fetch('/api/data')).json(); const sel=document.getElementById('frameSelect'); data.rows.forEach((r,i)=>{let o=document.createElement('option');o.value=i;o.textContent=`${i+1}. ${r.frame} ${r.group}`;sel.appendChild(o)}); document.getElementById('count').textContent=`${data.rows.length} frames`; showFrame(0)}
function showFrame(i){current=Number(i); document.getElementById('frameSelect').value=current; const r=data.rows[current]; const stem=r.frame.replace('.png','.jpg'); document.getElementById('title').textContent=`${current+1}/${data.rows.length} ${r.frame} | ${r.group}`; document.getElementById('overlay').src=`/file/annotated_overlays/${stem}`; document.getElementById('overlayLink').href=`/file/annotated_overlays/${stem}`; document.getElementById('rgbLink').href=`/file/rgb/${r.frame}`; document.getElementById('maskLink').href=`/file/color_masks/${r.frame}`; renderFields(r); renderCandidates(r.frame); window.scrollTo(0,0)}
function renderFields(r){const box=document.getElementById('fields'); box.innerHTML=''; [...editFields,'notes'].forEach(f=>{const div=document.createElement('div'); div.className='field'; const lab=document.createElement('label'); lab.textContent=labels[f]; const inp=document.createElement('input'); inp.value=r[f]||''; inp.dataset.field=f; inp.onfocus=()=>{activeField=f; document.querySelectorAll('input').forEach(x=>x.classList.remove('active')); inp.classList.add('active')}; inp.oninput=()=>{r[f]=inp.value}; if(f===activeField) inp.classList.add('active'); div.append(lab,inp); box.append(div)});}
function renderCandidates(frame){const box=document.getElementById('candidates'); box.innerHTML=''; (data.candidates[frame]||[]).forEach(c=>{const d=document.createElement('div'); d.className='cand'; d.innerHTML=`<strong>${c.label_id}</strong><small>${c.class_name}</small><small>area ${c.area_px}</small><small>cx ${c.cx}, cy ${c.cy}</small>`; const b=document.createElement('button'); b.textContent='Add'; b.onclick=()=>addId(c.label_id); d.appendChild(b); box.appendChild(d)});}
function addId(id){const r=data.rows[current]; const val=(r[activeField]||'').trim(); const ids=val?val.split(',').map(x=>x.trim()).filter(Boolean):[]; if(!ids.includes(String(id))) ids.push(String(id)); r[activeField]=ids.join(','); renderFields(r);}
function prevFrame(){showFrame(Math.max(0,current-1))}
function nextFrame(){showFrame(Math.min(data.rows.length-1,current+1))}
async function saveCsv(){document.getElementById('status').textContent='Saving...'; const res=await fetch('/api/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({rows:data.rows})}); const out=await res.json(); document.getElementById('status').textContent=out.ok?'Saved target_id_mapping_template.csv':'Save failed'; if(!out.ok) document.getElementById('status').className='warn';}
init();
</script>
</body>
</html>'''


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})


class Handler(BaseHTTPRequestHandler):
    subset: Path
    mapping_csv: Path

    def log_message(self, fmt: str, *args):
        print(fmt % args)

    def send_bytes(self, body: bytes, content_type: str, code: int = 200):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/" or parsed.path == "/index.html":
            self.send_bytes(HTML.encode(), "text/html; charset=utf-8")
            return
        if parsed.path == "/api/data":
            rows = read_csv(self.mapping_csv)
            bbox_rows = read_csv(self.subset / "label_bboxes.csv")
            candidates: dict[str, list[dict[str, str]]] = {}
            for row in bbox_rows:
                if row.get("class_name") not in {"vine_leaf", "vine_trunk", "wooden_post"}:
                    continue
                candidates.setdefault(row["frame"], []).append(row)
            for vals in candidates.values():
                vals.sort(key=lambda r: (int(r.get("cx", 0)), int(r.get("cy", 0))))
            self.send_bytes(json.dumps({"rows": rows, "candidates": candidates}).encode(), "application/json")
            return
        if parsed.path.startswith("/file/"):
            rel = Path(unquote(parsed.path[len("/file/"):]))
            path = (self.subset / rel).resolve()
            if not str(path).startswith(str(self.subset.resolve())) or not path.exists():
                self.send_error(404)
                return
            ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            self.send_bytes(path.read_bytes(), ctype)
            return
        self.send_error(404)

    def do_POST(self):
        if urlparse(self.path).path != "/api/save":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length) or b"{}")
        rows = payload.get("rows", [])
        write_csv(self.mapping_csv, rows)
        self.send_bytes(json.dumps({"ok": True}).encode(), "application/json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Browser editor for target-vine mapping CSV.")
    parser.add_argument("--subset_dir", type=Path, required=True)
    parser.add_argument("--mapping_csv", type=Path, default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    Handler.subset = args.subset_dir.resolve()
    Handler.mapping_csv = (args.mapping_csv or args.subset_dir / "target_id_mapping_template.csv").resolve()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Serving {Handler.subset}")
    print(f"Open http://{args.host}:{args.port}/")
    server.serve_forever()


if __name__ == "__main__":
    main()
