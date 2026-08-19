# BevTesla — Agent Guide

Single-file Python app (`main.py`) that reads 1–4 dashcam feeds + optional telemetry CSV, runs YOLOv8 detection, ByteTrack+Kalman tracking, and renders a Tesla FSD-style BEV video.

## Commands

```bash
# Activate venv (WSL/Linux)
source .venv/bin/activate

pip install -r requirements.txt   # opencv, numpy, ultralytics, scipy, transformers, torch, torchvision
python main.py                    # outputs tesla_bev_fsd_clone.mp4 + tesla_map.mp4
```

No lint, typecheck, test, or build tooling is configured.

## Architecture

Everything is in `main.py` (~690 lines). Configuration constants live at lines 12–139.

- **Input**: `VIDEOS` dict — only `front` is required; `back`/`left`/`right` optional (set `None` to disable). Telemetry CSV optional and **disabled by default** (`TELEMETRY_CSV = None`); set a path to enable it. Current front video: `videos/rapidsave.com__-9iv42du4doih1.mp4`; output is `tesla_bev_fsd_clone_9iv42du4doih1.mp4` (`OUT_VIDEO`).
- **Depth (optional)**: **Depth Anything V2 Small** via `transformers` (auto-downloaded from HF, `DEPTH_MODEL = "depth_anything_v2"`; `None` disables). Per-frame relative depth is converted to meters by **flat-ground calibration**: for road pixels below the horizon, geometric distance `dist = CAM_HEIGHT * FOCAL_DEPTH / (py - horizon)` is known, so `scale = median(dist_geo * pred)`; front-camera detections sample depth at the bbox bottom-center instead of the geometric focal formula. Do NOT anchor scale to the horizon max — it stretched close objects far away. Also renders a turbo colormap overlay on the front view (`DEPTH_OVERLAY`).
- **Detection**: YOLOv8 nano (`yolov8n.pt`, auto-downloaded) at `conf=0.15`, classes = `DIMS` keys (car/truck/bus/motorcycle/person). One pass per enabled camera; without depth, distance uses the **front focal** (`FOCAL_DEPTH`). **Hood filter**: front-camera boxes with bottom in the bottom 16% of frame, centered within ±22% of image width, and very wide/flat (w/h > 2.2) are dropped — the ego hood is often misclassified as a car (real close cars have w/h < 1.5, so they're unaffected). **Lateral alignment**: object ego_x uses the SAME `FOCAL_DEPTH` as the lane flat-ground projection (NOT `FOCAL_HORIZ`), so detected vehicles align with lane geometry in the BEV.
- **Lane detection (optional)**: **UFLDv2** (Tusimple ResNet18, `UFLD_MODEL = "ufldv2_tusimple_res18.pth"`; `None` disables). The `parsingNet` is reimplemented standalone in `_load_ufld_model()` (ResNet18 → Conv1x1 → MLP), so no repo/DALI deps. Output lane points in pixels are projected to ego frame via the same flat-ground formula (`dist = CAM_HEIGHT*FOCAL/(py-horizon)`, `x=(px-CAM_W/2)*dist/FOCAL`), fit to x(y) quadratics, and smoothed with EMA (`UFLD_EMA`). Lane 1/2 are the ego-lane dashed markers, 0/3 the outer solid lines. When active, detected lanes replace `draw_clean_lanes_animated` and the blue path gradient follows the real lane center; the front RGB view gets a translucent blue lane overlay (`draw_lanes_overlay_rgb`); the HUD shows lateral offset, lane width, curvature, object count per class, and closest-vehicle distance. The speedometer was removed from the BEV (speed is 0 without telemetry CSV); the center-top line now shows `OBJ: N` counts and `FRENTE: X.X m` (closest tracked object ahead of `CAR_FRONT_TIP`). Rendering order stays: lanes → path → objects → ego → text.
- **Tracking**: `BEVByteTracker` — two-tier Hungarian matching (high>=0.5/low>=0.1), 8m match radius, max 15 frame coast, output only when age≥2 (or re-matched this frame).
- **Cross-camera NMS**: `_nms_ego()` deduplicates in ego-space with 2.2m radius.
- **Dynamic BEV camera**: `cam_y` fixed at −10m; chase-cam style — height 14→22m and look-ahead 26→36m as speed→130 km/h, keeping the whole ego car in view and pushing the car lower (closer to viewer) with the horizon out to ~100m.
- **Detection render range**: `MAX_DIST = 70m` — detections beyond it are not drawn in the BEV (`render_tesla_ui` filter).
- **Outputs**: dynamic camera grid (1 col per camera if ≤3, else 3 cols) + BEV canvas → `tesla_bev_fsd_clone.mp4`; OSM tile map → `tesla_map.mp4` (only when ≥2 valid GPS points). Both writers use **`mp4v`** codec, not H.264.

## Coordinate Systems

| Space | Origin | Axes |
|---|---|---|
| Camera pixel | Top-left | +X right, +Y down |
| Ego-centric (world) | Ego center | +X right, +Y forward, +Z up |
| BEV image | `project_bev()` output | 2D canvas pixels |

## Gotchas

- **Rendering order**: `draw_clean_lanes_animated` → `draw_fsd_path_gradient` → `draw_solid_3d_vehicle` (objects) → `draw_solid_3d_vehicle` (ego) → `render_tesla_ui` text. Changing it breaks occlusion.
- **`global_offset_y`** accumulates `speed * dt` each frame. Resetting it causes a visual jump in lane/path animations.
- **Telemetry fallback**: missing, `na`, or `0.0` values reuse the last valid reading. If the CSV is shorter than the video, out-of-range frames get a zero-state placeholder (`{"speed_mps": 0, "heading": 0}`) — NOT a repeated last row.
- **Telemetry CSV column names** (differ from README): `speed_mps`, `acceleration_x_mps2`, `acceleration_y_mps2`, `heading_deg`, `GPS_latitude_deg`, `GPS_longitude_deg`.
- **Focal lengths**: no camera-view lane overlay anymore (functions removed). Detection distance always uses the front focal.
- **Camera yaw**: `CAM_YAW` dict drives ego-coordinate transforms; add a yaw + `CAM_OFFSETS` entry when adding a new camera.
- **Code comments are in Portuguese (Brazilian)** — keep new comments consistent.
- **Model files**: only `yolov8n.pt` and `ufldv2_tusimple_res18.pth` exist in root (both gitignored; ultralytics auto-downloads `yolov8n.pt` if missing, UFLD weights must be downloaded manually from the UFLDv2 repo). Depth Anything V2 Small is cached by HF (`transformers`, now in `requirements.txt`).
- **1–3 sec/frame on CPU** (750 frames = 15–30 min). GPU cuts it to seconds total.
- The main loop breaks early if any enabled video ends (`cap.read()` returns False).
