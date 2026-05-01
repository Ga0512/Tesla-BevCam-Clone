# BevTesla

A Tesla FSD-style Bird's Eye View (BEV) visualization system built in Python. Reads 4 synchronized dashcam feeds (front, rear, left, right) plus a telemetry CSV, runs YOLOv8 object detection, tracks every detected vehicle with a ByteTrack + Kalman filter, and renders an animated 360° top-down output video — plus a second output that overlays the GPS trajectory on a live OpenStreetMap tile map.

## Demo

| BEV Output | OSM Map Output |
|---|---|
| `tesla_bev_fsd_clone.mp4` | `tesla_map.mp4` |

The BEV view auto-zooms based on speed (wider FOV at highway speeds, tighter in city) and shows full 360° coverage including rear-camera detections.

## Data & Acknowledgements

The 4-camera dashcam footage and synchronized telemetry powering this project were captured with a **[NATIX VX360](https://natix.network)** — a multi-camera capture device for vehicles (currently Tesla, more OEMs coming).

NATIX is building the infrastructure layer for real-world AI perception data. Contributors install VX360 devices, capture multi-camera driving data, and earn $NATIX tokens for sharing it to a global, crowd-sourced network. The resulting dataset is used by autonomous driving teams, map makers, and AI labs training perception stacks and world models.

Learn more at [natix.network](https://natix.network).

## Features

- **360° BEV** — front, rear, left and right cameras all contribute detections to a single top-down canvas
- **YOLOv8 detection** — cars, trucks, buses, motorcycles, people; confidence threshold 0.15
- **ByteTrack + Kalman tracking** — two-tier Hungarian matching, ghost-track suppression, per-class color trails
- **Dynamic FOV** — BEV camera height and look-ahead distance interpolate with ego speed (0 → 130 km/h)
- **Cross-camera NMS** — ego-space deduplication prevents double-counting the same vehicle seen by two cameras
- **OSM map output** — downloads and stitches OpenStreetMap tiles; draws ego trail and per-vehicle GPS trails
- **Telemetry HUD** — speedometer, rotating compass, GPS coordinates, lateral/longitudinal acceleration panels

## Requirements

- Python 3.9+
- Windows / Linux / macOS
- GPU optional (YOLOv8 runs on CPU; GPU cuts inference time ~5×)

## Setup

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/BevTesla.git
cd BevTesla

# 2. Create virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

## Input Files

Place your dashcam videos and telemetry CSV under `videos/`:

```
videos/
├── FRONT_<timestamp>.mp4
├── REAR_<timestamp>.mp4
├── LEFT_<timestamp>.mp4
├── RIGHT_<timestamp>.mp4
└── FRONT_<timestamp>.csv   ← telemetry (speed, GPS, heading, acceleration)
```

Edit the `VIDEOS` and `TELEMETRY_CSV` paths at the top of `main.py` to match your filenames.

### Telemetry CSV format

The CSV must contain at minimum these columns (extra columns are ignored):

| Column | Description |
|---|---|
| `speed_mps` | Ego speed in m/s |
| `latitude` | GPS latitude (decimal degrees) |
| `longitude` | GPS longitude (decimal degrees) |
| `heading` | Compass heading in degrees (0 = north) |
| `accel_x` | Lateral acceleration (m/s²) |
| `accel_y` | Longitudinal acceleration (m/s²) |

## Usage

```bash
python main.py
```

Outputs:

- `tesla_bev_fsd_clone.mp4` — side-by-side 4-camera grid + 360° BEV canvas
- `tesla_map.mp4` — OSM map with ego and vehicle GPS trails

Processing time: ~1–3 seconds per frame on CPU (750 frames = ~15–30 min). A CUDA-capable GPU reduces this to seconds total.

## Configuration

All tunable parameters live at the top of `main.py` (lines 8–80):

| Parameter | Default | Description |
|---|---|---|
| `MAX_FRAMES` | `750` | Frames to process (750 = 25 s at 30 FPS) |
| `CAM_W / CAM_H` | `640 × 360` | Camera resolution |
| `BEV_W / BEV_H` | `800 × 720` | BEV canvas size |
| `MAX_DIST` | `80 m` | Max detection range (front) |
| `REAR_DIST` | `20 m` | Rear detection range |
| `FOV_BEV` | `75°` | BEV virtual camera field of view |
| `MAP_ZOOM` | `17` | OSM tile zoom level |

## Architecture

```
main.py
├── Data loading        load_telemetry(), cv2.VideoCapture ×4
├── Detection           YOLOv8n → pixel_to_ego() per camera
├── NMS                 _nms_ego() — cross-camera deduplication
├── Tracking            KalmanTrack + BEVByteTracker
├── BEV rendering       project_bev(), draw_fsd_path(), draw_vehicle_3d()
├── Telemetry HUD       draw_telemetry()
├── OSM map             build_osm_map(), ego_centric_to_gps()
└── Output              H.264 MP4 via cv2.VideoWriter
```

### Coordinate systems

| Space | Origin | Axes |
|---|---|---|
| Camera pixel | Top-left of frame | +X right, +Y down |
| Ego-centric | Ego vehicle center | +X right, +Y forward, +Z up |
| BEV image | `project_bev()` output | 2D canvas pixels |

## License

MIT
