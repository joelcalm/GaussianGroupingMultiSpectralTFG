# Vineyard SAM3 Masks, 3D Gaussian Object IDs, and Selection Workflow

This note explains the current end-to-end vineyard workflow: how masks are made, how SAM3 tracklets become object IDs, what the Gaussian Splatting model learns, how tracklets are merged into physical-vine candidates, and how `select_gaussians.py` lets us ask for things like `only vines` or `vine 3`.

The concrete scene described here is:

```text
scene:  vinyes_sam3_vineid_200
model:  jcalm/output/vinyes_sam3_vineid_200
SAM3:   vineyard_posematch/sam3_video_vine_semantic_instances_1344_cap96
```

## 1. Big Picture

There are three separate ideas that are easy to mix together:

1. **2D masks**: per-frame pixel labels produced from SAM3 on the RGB frames.
2. **3D Gaussian object features**: a learnable object embedding attached to each Gaussian.
3. **Classifier output IDs**: a small trained classifier that maps those object features to semantic or instance labels.

The Gaussian model does not start with perfect object IDs. During training it learns object features for each Gaussian because, when those Gaussians are rendered into labeled RGB views, their rendered object features are penalized if they do not classify to the ground-truth mask label at each pixel.

So yes: the trained GS model learns a 3D object/instance representation. More precisely:

- Each Gaussian has a learned object feature vector, stored in the PLY as `obj_dc_*`.
- Rendering projects/blends those object features into a per-pixel feature map called `render_object`.
- A `1x1` classifier predicts a label for each pixel from `render_object`.
- The label can mean a class like `vine_plant`, or an instance/tracklet like `vine_plant_track_0002`, depending on the masks used for training.

## 2. Where the Ground Truth Comes From

The object ground truth comes from SAM3-generated masks, not from manual labels.

The SAM3 script is:

```text
vineyard_posematch/sam3_vine_video.py
```

It runs SAM3 video prediction over the RGB frame sequence with prompts such as:

```text
vine_plant: grapevine trunk and branches
wooden_post: wooden vineyard post
ground: bare soil ground in a vineyard
sky: blue sky
tree: tree canopy or tree trunk
stone_wall: stone wall made of rocks
shrub_or_other_vegetation: shrub, bush, or other vegetation
building: building wall or building facade
```

SAM3 returns masks and, in the video predictor, per-object track IDs. The important implementation detail is that the Ultralytics/SAM3 result boxes contain the video track ID:

```text
[x1, y1, x2, y2, track_id, score, cls]
```

Our script preserves those IDs instead of simply merging all masks of a class together.

SAM3 outputs include:

```text
semantic_index_masks/       per-pixel semantic class ID
semantic_instance_masks/    per-pixel global instance/tracklet ID
semantic_color_masks/       color preview of semantic classes
semantic_overlays/          semantic preview over RGB images
vine_binary_masks/          binary vine masks
metadata/class_map.json
metadata/instance_label_map.json
metadata/sam3_track_id_map.json
metadata/instance_tracking_report.json
```

The key training labels are the indexed PNGs, not the color previews.

## 3. Semantic Classes vs Object/Instance IDs

There are two useful label modes:

### Semantic Mode

In semantic mode, every vine pixel has the same label:

```text
vine_plant = 1
wooden_post = 2
ground = 3
...
```

This is good for queries like:

```text
only show vines
only show wooden posts
```

But it cannot distinguish vine 1 from vine 2.

### Instance Mode

In instance mode, object-like things can get their own IDs.

For this scene, raw SAM3 produced many tracklets:

```text
raw SAM3 instances: 1316
vine tracklets:     353
wooden post tracks: 550
tree tracks:        269
...
```

Training on all 1316 IDs would make the classifier large and noisy. So the prepared scene uses a compact hybrid:

```text
semantic classes remain as normal:
  background = 0
  vine_plant = 1
  wooden_post = 2
  ground = 3
  ...

vine tracklets get individual IDs:
  vine_plant_track_0000 = 9
  vine_plant_track_0001 = 10
  vine_plant_track_0002 = 11
  ...
```

Other tracked classes are folded back to their semantic class. This gives us individual vine IDs without exploding the classifier with hundreds of posts, trees, and small fragments.

For `vinyes_sam3_vineid_200`, the compact config has:

```text
num_classes = 362
label_mode  = instance
```

The label map is:

```text
jcalm/output/vinyes_sam3_vineid_200/instance_label_map.json
```

## 4. Scene Preparation

The prep script is:

```text
jcalm/prepare_vinyes_sam3_200.py
```

It takes the source COLMAP/multispectral scene and the SAM3 masks, then creates a training scene:

```text
vineyard_posematch/vinyes_sam3_vineid_200
```

Important folders/files:

```text
images/                         registered RGB/multispectral images
object_mask/                    masks used as training object GT
semantic_mask/                  semantic-only masks
sam3_instance_mask/             raw aligned SAM3 instance masks
metadata/active_channels.json   which bands are valid for each image
metadata/class_map.json
metadata/instance_label_map.json
metadata/instance_tracking_report.json
```

Only RGB frames have SAM3 masks. The other bands still participate in photometric training, but they usually do not have object-mask supervision unless there is a matching `object_mask/<image_name>.png`.

This is why the dataset loader checks, for each registered image:

```text
object_mask/<image_name>.png
```

If that mask exists, the view has object supervision. If it does not, that view trains only photometric reconstruction.

## 5. What Training Learns

Training happens in:

```text
jcalm/train.py
```

For every sampled training camera:

1. The model renders an image.
2. It also renders `render_object`, a per-pixel object-feature image.
3. If the camera has a GT object mask, the classifier predicts labels from `render_object`.
4. The predicted labels are compared to the mask with cross-entropy.
5. The image itself is trained with L1/SSIM photometric loss.
6. A 3D regularization term encourages nearby Gaussians to have compatible object predictions.

Conceptually:

```text
Gaussian object features
        |
        v
rendered object feature image
        |
        v
1x1 classifier
        |
        v
per-pixel label prediction
        |
        v
cross-entropy against SAM3 mask label
```

The total loss is roughly:

```text
photometric image loss
+ object mask classification loss
+ 3D object consistency loss
```

The important point: the model is not directly handed a permanent object ID for each Gaussian. It learns object features because the same Gaussians must explain many views, and the rendered object-feature image must match the 2D object masks in those views.

After training, the saved model contains:

```text
point_cloud/iteration_30000/point_cloud.ply    Gaussian geometry/color/object features
point_cloud/iteration_30000/classifier.pth     object-feature-to-label classifier
```

## 6. Rendering Predictions

Rendering happens in:

```text
jcalm/render.py
```

It renders:

```text
renders/             RGB preview PNGs plus full multispectral .npy tensors
gt/                  GT preview PNGs plus full multispectral .npy tensors
objects/             predicted object masks
gt_objects/          GT object masks where available
color_masks/         object feature visualization
channel_B*/          per-channel render/GT side-by-side images
```

The main `renders/*.png` are now visual RGB channels `[0, 1, 2]`, not false-color multispectral composites.

For metrics, the full `.npy` tensors are used when present, so changing the preview PNG colors does not change the actual multispectral metrics.

## 7. Why Tracklet Merging Is Needed

SAM3 video tracking gives **tracklets**, not guaranteed biological-vine identities.

A real vine can be split into multiple SAM3 track IDs because:

- the trunk is partially occluded,
- the vine appears/disappears across frames,
- SAM3 splits branches/trunk into separate tracks,
- camera motion changes visibility,
- masks are imperfect.

So a label like:

```text
vine_plant_track_0125
```

means “SAM3 tracked this as a consistent video object fragment”, not necessarily “this is one complete physical vine”.

We therefore added a second stage that proposes physical-vine groups.

## 8. Candidate Tracklet Merging

The merge script is:

```text
jcalm/merge_vine_tracklets.py
```

It uses two sources of evidence:

### 3D Evidence

From the trained Gaussian model:

- centroid of each vine tracklet point cluster,
- 3D bounding box,
- nearest-neighbor distance between sampled Gaussian points,
- number of predicted Gaussians per tracklet.

If two tracklets occupy nearby/overlapping 3D space, they are candidate parts of the same physical vine.

### 2D Mask Evidence

From the compact `object_mask/*.png` labels:

- whether two tracklets are adjacent in the same frame,
- whether they co-occur in the same frame,
- whether one disappears and another appears nearby shortly after,
- center distance in image space.

This helps distinguish true splits from unrelated nearby 3D fragments.

The script writes:

```text
vine_tracklet_stats.json
vine_tracklet_merge_candidates.json
vine_tracklet_merge_candidates.csv
vine_tracklet_merge_map.json
vine_tracklet_merge_report.txt
```

The map is conservative. It creates automatic components only when an edge has a high score and strong evidence, such as repeated 2D adjacency or very close 3D proximity.

For the current run:

```text
supported vine tracklets:      154
ranked candidate edges:        280
physical-vine candidates:      135
multi-tracklet groups:         9
```

The output folder is:

```text
jcalm/output/vinyes_sam3_vineid_200/vine_tracklet_merges/iteration_30000/
```

## 9. Why Physical Vine PNGs Look Black

The files in:

```text
physical_vine_mask/
```

are 16-bit label masks. Pixel value `3` means physical vine ID 3. Normal image viewers display values like `1`, `2`, and `3` as almost black because those are tiny intensity values.

They are not empty.

For visual inspection, use:

```text
physical_vine_mask_color/
```

Those are RGB preview masks where each physical vine ID gets a visible color.

Use the 16-bit masks for data/training. Use the color masks only for looking.

## 10. Class and Object Identification

There are now three levels of selection:

### Class Selection

Class selection asks for all Gaussians predicted as a semantic class.

Examples:

```bash
python select_gaussians.py \
  -m output/vinyes_sam3_vineid_200 \
  --query vines
```

This exports all predicted vine Gaussians.

You can also use explicit class names:

```bash
python select_gaussians.py \
  -m output/vinyes_sam3_vineid_200 \
  --class-name vine_plant
```

### Raw Tracklet / Instance Selection

Raw instance selection asks for a specific trained label ID, for example:

```bash
python select_gaussians.py \
  -m output/vinyes_sam3_vineid_200 \
  --instance-id 11
```

This selects the Gaussian points classified as instance label `11`, such as one SAM3 vine tracklet.

This is useful for debugging, but it may be too fragmented for “one biological vine”.

### Physical Vine Selection

Physical vine selection uses:

```text
vine_tracklet_merge_map.json
```

If that map exists, `select_gaussians.py` understands a query like:

```bash
python select_gaussians.py \
  -m output/vinyes_sam3_vineid_200 \
  --query "vine 3"
```

This selects physical vine candidate `3`, not raw tracklet index 3.

For the current output, `vine 3` maps to:

```text
physical_vine_0003
member raw instance IDs: [11, 44]
```

So the exported PLY contains Gaussians predicted as either tracklet `11` or tracklet `44`.

## 11. `select_gaussians.py` Outputs

The selector writes two PLYs:

```text
*_gaussians.ply
*_points.ply
```

The difference:

- `*_gaussians.ply`: preserves the full Gaussian PLY fields for the selected Gaussians.
- `*_points.ply`: simple point cloud with XYZ/RGB/opacity/predicted IDs, easier to inspect in many viewers.

It also writes:

```text
*_summary.json
```

That summary records:

- which class/instances were selected,
- whether a physical-vine merge map was used,
- how many Gaussians were exported,
- the output file paths.

Examples already generated:

```text
output/vinyes_sam3_vineid_200/selections/iteration_30000/all_vines_gaussians.ply
output/vinyes_sam3_vineid_200/selections/iteration_30000/all_vines_points.ply

output/vinyes_sam3_vineid_200/selections/iteration_30000/physical_vine_0003_gaussians.ply
output/vinyes_sam3_vineid_200/selections/iteration_30000/physical_vine_0003_points.ply
```

## 12. Practical Commands

Generate/update merge candidates and physical-vine masks:

```bash
cd /home/msiau/workspace/jcalm
source activate_env.sh

python merge_vine_tracklets.py \
  -m output/vinyes_sam3_vineid_200 \
  --iteration 30000 \
  --write-remapped-masks
```

List all raw instances and physical-vine candidates:

```bash
python select_gaussians.py \
  -m output/vinyes_sam3_vineid_200 \
  --list
```

Export all vines:

```bash
python select_gaussians.py \
  -m output/vinyes_sam3_vineid_200 \
  --query vines \
  --output-name all_vines \
  --color-by rgb
```

Export physical vine 3:

```bash
python select_gaussians.py \
  -m output/vinyes_sam3_vineid_200 \
  --query "vine 3" \
  --output-name physical_vine_0003 \
  --color-by rgb
```

Export a raw SAM3 tracklet by instance ID:

```bash
python select_gaussians.py \
  -m output/vinyes_sam3_vineid_200 \
  --instance-id 11 \
  --output-name raw_tracklet_11 \
  --color-by rgb
```

Launch the small selector UI:

```bash
python select_gaussians.py \
  -m output/vinyes_sam3_vineid_200 \
  --app
```

## 13. Mental Model

A compact way to think about the pipeline:

```text
RGB frames
  -> SAM3 semantic + tracked instance masks
  -> prepared object_mask/*.png
  -> train Gaussian model with image loss + object mask loss
  -> each Gaussian learns an object feature
  -> classifier predicts class/tracklet ID from that feature
  -> tracklet merge map groups raw vine fragments into physical-vine candidates
  -> select_gaussians exports all vines, one raw tracklet, or one physical vine
```

So:

- `only vines` = class-level selection.
- `instance-id 11` = raw learned SAM3 tracklet label.
- `vine 3` = merged physical-vine candidate 3, using the merge map.

The current system is strongest for “show all vines” and “show this candidate vine group”. The biological vine identity still depends on the quality of SAM3 masks and the merge map, so ambiguous groups should be reviewed before using them as final ground truth for a new training run.
