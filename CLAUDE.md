# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BevTesla is a Tesla FSD-clone Bird's Eye View (BEV) visualization system. It reads 4 synchronized dashcam video feeds (front, rear, left, right) and a telemetry CSV, runs YOLOv8 object detection on each frame, tracks objects with a custom ByteTrack+Kalman implementation, then renders a Tesla FSD-style animated output video.

## Commands

```bash
# Activate virtual environment (Windows)
venv\Scripts\activate

# Install dependencies (no requirements.txt — install manually)
pip install opencv-python numpy ultralytics scipy

# Run the application (outputs tesla_bev_fsd_clone.mp4)
python main.py
```

No linting, formatting, or test suite is configured.

## Configuration

All parameters live at the top of `main.py` (lines 8–72):

- `VIDEOS` — input video file paths (front/rear/left/right)
- `TELEMETRY_CSV` — path to telemetry CSV
- `MAX_FRAMES` — frames to process (default 750 = 25 seconds at 30 FPS)
- `CAM_W / CAM_H` — camera resolution (640×360)
- `BEV_W / BEV_H` — BEV canvas size (800×720)
- `DIMS` — real-world object dimensions per class (meters)
- Focal lengths (`FL_*`) — camera calibration constants

## Architecture

The entire application is a single file (`main.py`) with five logical layers:

### 1. Data Loading
- `load_telemetry()` — parses the CSV into a list of dicts; missing or zero values fall back to the last valid reading to prevent telemetry jumps.
- Four `cv2.VideoCapture` objects read frames in lockstep.

### 2. Object Detection & 3D Projection
- YOLOv8 nano (`yolov8n.pt`) runs on each camera frame with confidence threshold 0.15, filtering only: `car`, `truck`, `bus`, `motorcycle`, `person`.
- `pixel_to_ego()` — converts a 2D bounding box to ego-centric 3D coordinates (vehicle is origin, +Y forward, +X right) using the known real-world object height and camera focal length.
- Each camera has fixed offsets (`CAM_OFFSETS`) and a heading angle added to the ego-centric conversion.

### 3. Tracking (ByteTrack + Kalman)
- `KalmanTrack` — state vector `[x, y, vx, vy]`, constant-velocity model, tuned Q/R matrices.
- `BEVByteTracker` — two-tier Hungarian matching: high-confidence detections assigned first, then low-confidence detections matched only to already-existing tracks. Tracks require age ≥ 2 frames before output; deleted after 15 frames without a match.

### 4. BEV Rendering
- `project_bev()` — projects 3D world points to 2D BEV image via a virtual overhead camera at (0, −22, 16) with 55° vertical FOV.
- `draw_fsd_path()` — animated blue gradient "warp" tunnel with flow lines.
- `draw_lane_markings()` — dashed center + solid edge lines; dash offset animates based on `global_offset_y` (accumulated distance = speed × frame_time).
- `draw_vehicle_3d()` — face-based polygon rendering with brightness variation for pseudo-3D shading.
- `draw_telemetry()` — speedometer, rotating compass, GPS/acceleration panel.

### 5. Output Composition
- 2×2 camera grid (front/right top row, left/rear bottom row) combined with the BEV canvas into a single frame.
- Written to `tesla_bev_fsd_clone.mp4` using H.264 at 30 FPS.

## Coordinate Systems

Understanding these three spaces is essential for any changes to detection or rendering:

| Space | Origin | Axes |
|---|---|---|
| Camera pixel | Top-left of frame | +X right, +Y down |
| Ego-centric (world) | Ego vehicle center | +X right, +Y forward, +Z up |
| BEV image | `project_bev()` output | 2D canvas pixels |

## Key Non-Obvious Details

- **Focal length per camera**: front and rear use different focal lengths with an aspect-ratio correction factor; mismatches cause distance estimation errors.
- **Telemetry sync**: frame index maps directly to CSV row index; if the CSV is shorter than the video, the last valid row is reused.
- **Animation offset**: `global_offset_y` accumulates across frames — resetting it mid-run will cause a visual jump in lane/flow animations.
- **Two-tier tracking threshold**: the split between "high" and "low" confidence detections is 0.4. Lowering it causes more ghost tracks; raising it drops slow-moving objects.
- **Rendering order matters**: path → lanes → tracked vehicles → telemetry UI; drawing out of order changes occlusion behavior.
