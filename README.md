# BevTesla

A Tesla FSD-style Bird's Eye View (BEV) visualization system built in Python. It reads a front dashcam feed (plus optional rear/left/right and telemetry), runs YOLOv8 object detection, estimates monocular depth, detects lane geometry, and renders an animated top-down view where every detected vehicle is positioned in real-world (x, y) coordinates — guided by the detected lanes.

## Demo

| Main Example |
|---|
| [tesla_bev_fsd_clone_9iv42du4doih1.mp4](tesla_bev_fsd_clone_9iv42du4doih1.mp4) |

The BEV view auto-zooms with speed (wider FOV at highway speeds, tighter in the city) and positions detected vehicles using depth + flat-ground lane geometry, so objects appear exactly where they are on the road.

## Data & Acknowledgements

The dashcam footage powering this project was captured with a **[NATIX VX360](https://natix.network)** — a multi-camera capture device for vehicles (currently Tesla, more OEMs coming).

NATIX is building the infrastructure layer for real-world AI perception data. Contributors install VX360 devices, capture multi-camera driving data, and earn $NATIX tokens for sharing it to a global, crowd-sourced network. The resulting dataset is used by autonomous driving teams, map makers, and AI labs training perception stacks and world models.

Learn more at [natix.network](https://natix.network).

## Features

- **BEV rendering** — Tesla FSD-style top-down canvas with ego vehicle, detected objects and dynamic chase camera (`cam_y` fixed at −10 m; height 14→22 m and look-ahead 26→36 m as speed increases)
- **YOLOv8 detection** — cars, trucks, buses, motorcycles, people; confidence threshold 0.15; **hood filter** drops the ego hood when it is misclassified as a vehicle
- **Monocular depth** — Depth Anything V2 Small converts relative depth to real meters via **flat-ground calibration** (`dist = CAM_HEIGHT · FOCAL / (py − horizon)`), sampled at each bbox bottom-center
- **Lane detection** — UFLDv2 (Tusimple ResNet18) reads the road geometry; detected lanes replace the synthetic ones, the blue path follows the real lane center, and a translucent overlay is drawn on the front RGB view
- **Lane-guided positioning** — objects use the same focal length as the lane flat-ground projection, so vehicles align with the real road geometry in the BEV
- **ByteTrack + Kalman tracking** — two-tier Hungarian matching (0.5 / 0.1), 8 m match radius, 15-frame coast, ghost-track suppression
- **Cross-camera NMS** — ego-space deduplication (2.2 m radius) prevents double-counting the same vehicle seen by multiple cameras
- **HUD** — lateral lane offset, lane width, curvature, per-class object counts, closest-object distance, heading/accel/GPS

## Requirements

- Python 3.9+
- Windows / Linux / macOS
- GPU optional (YOLOv8 runs on CPU; GPU cuts inference time ~5×)

## Setup

```bash
# 1. Clone
git clone https://github.com/Ga0512/Tesla-BevCam-Clone.git
cd BevTesla

# 2. Create virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the UFLDv2 lane weights (manual — not auto-downloaded)
#    Place ufldv2_tusimple_res18.pth in the project root.
#    Get it from the UFLDv2 repo (Google Drive link in the README).
```

`yolov8n.pt` (detection) and the Depth Anything V2 Small model (transformers) are auto-downloaded on first run.

## Input Files

Place your dashcam videos and telemetry CSV under `videos/`:

```
videos/
├── front.mp4                 ← required
├── back.mp4                  ← optional (set None to disable)
├── left.mp4                  ← optional
├── right.mp4                 ← optional
└── telemetry.csv             ← optional (set TELEMETRY_CSV to a path)
```

Edit the `VIDEOS` and `TELEMETRY_CSV` paths at the top of `main.py` to match your filenames.

### Telemetry CSV format

| Column | Description |
|---|---|
| `speed_mps` | Ego speed in m/s |
| `acceleration_x_mps2` | Lateral acceleration (m/s²) |
| `acceleration_y_mps2` | Longitudinal acceleration (m/s²) |
| `heading_deg` | Compass heading in degrees (0 = north) |
| `GPS_latitude_deg` | GPS latitude (decimal degrees) |
| `GPS_longitude_deg` | GPS longitude (decimal degrees) |

Missing, `na`, or `0.0` values reuse the last valid reading. Without telemetry, speed/heading default to 0 (the speedometer is intentionally not shown).

## Usage

```bash
python main.py
```

Outputs (written with the `mp4v` codec):

- `tesla_bev_fsd_clone.mp4` — camera grid + BEV canvas
- `tesla_map.mp4` — OSM map with ego GPS trail (only when ≥2 valid GPS points)

Processing time: ~1–3 seconds per frame on CPU (750 frames = ~15–30 min). A CUDA-capable GPU reduces this to seconds total.

## Configuration

All tunable parameters live at the top of `main.py`:

| Parameter | Default | Description |
|---|---|---|
| `MAX_FRAMES` | `750` | Frames to process (750 = 25 s at 30 FPS) |
| `CAM_W / CAM_H` | `640 × 360` | Camera resolution |
| `BEV_W / BEV_H` | `800 × 720` | BEV canvas size |
| `MAX_DIST` | `70 m` | Max detection range shown in the BEV |
| `FOV_BEV` | `75°` | BEV virtual camera field of view |
| `DEPTH_MODEL` | `"depth_anything_v2"` | Depth Anything V2 Small (`None` disables depth) |
| `UFLD_MODEL` | `"ufldv2_tusimple_res18.pth"` | UFLDv2 lane weights (`None` disables lanes) |

## Architecture

```
main.py
├── Data loading        load_telemetry(), cv2.VideoCapture
├── Detection           YOLOv8n → depth-sampled distances, hood filter
├── Depth               compute_depth_map() — flat-ground calibrated
├── Lanes               detect_lanes() (UFLDv2) → _fit_lane_polys()
├── NMS                 _nms_ego() — cross-camera deduplication
├── Tracking            KalmanTrack + BEVByteTracker
├── BEV rendering       project_bev(), draw_detected_lanes(),
│                       draw_fsd_path_gradient_real(), draw_solid_3d_vehicle()
└── HUD                 render_tesla_ui()
```

### Coordinate systems

| Space | Origin | Axes |
|---|---|---|
| Camera pixel | Top-left of frame | +X right, +Y down |
| Ego-centric | Ego vehicle center | +X right, +Y forward, +Z up |
| BEV image | `project_bev()` output | 2D canvas pixels |

## License

MIT

[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-support-yellow)](https://buymeacoffee.com/gabrielcicotoste)
