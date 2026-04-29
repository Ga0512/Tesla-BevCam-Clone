import cv2
import numpy as np
import csv
import math
import os
import time
import urllib.request
from collections import deque
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment

# ─── CAMINHOS E ARQUIVOS ─────────────────────────────────
VIDEOS = {
    "front": "./videos/FRONT_2025-10-11_06-01-36.mp4",
    "back":  "./videos/REAR_2025-10-11_06-01-36.mp4",
    "left":  "./videos/LEFT_2025-10-11_06-01-36.mp4",
    "right": "./videos/RIGHT_2025-10-11_06-01-36.mp4",
}
TELEMETRY_CSV = "./videos/FRONT_2025-10-11_06-01-36.csv"
OUT_VIDEO     = "tesla_bev_fsd_clone.mp4"
OUT_MAP       = "tesla_map.mp4"

FPS        = 30
MAX_FRAMES = FPS * 25

# ─── RESOLUÇÃO ───────────────────────────────────────────
CAM_W, CAM_H = 640, 360
BEV_W, BEV_H = 800, 720

# ─── PALETA TESLA FSD CLONE (BGR) ────────────────────────
BEV_BG         = (242, 242, 242)
LANE_LINE      = (200, 200, 200)
FSD_BLUE_CORE  = (255, 120, 20)
FSD_FLOW_COLOR = (255, 180, 100)
EGO_COLOR      = (50, 50, 50)
OBJ_COLOR      = (100, 100, 100)
SHADOW_COLOR   = (210, 210, 210)
TEXT_COLOR     = (40, 40, 40)
UI_ACCENT      = (180, 180, 180)

# Cores dos rastros por classe (BGR)
TRAIL_COLORS = {
    "car":        (210,  80,  80),   # azul-escuro
    "truck":      ( 60, 180,  60),   # verde
    "bus":        (200, 130,   0),   # laranja
    "motorcycle": ( 60, 200, 200),   # ciano
    "person":     (160,   0, 200),   # roxo
}

# ─── FÍSICA DO MUNDO E CONSTANTES ────────────────────────
LANE_WIDTH    = 4
LANE_HALF     = LANE_WIDTH / 2
MAX_DIST      = 80.0
REAR_DIST     = 20.0
CAR_FRONT_TIP = 2.1

DIMS = {
    "car":        (1.9, 1.5, 4.5),
    "truck":      (2.6, 3.2, 10.0),
    "bus":        (2.8, 3.5, 12.0),
    "motorcycle": (0.8, 1.2, 2.0),
    "person":     (0.6, 1.7,  0.4),
}
H_REAL = {k: v[1] for k, v in DIMS.items()}

CAM_OFFSETS = {
    "front": ( 0.0,  1.8),
    "back":  ( 0.0, -2.0),
    "left":  (-1.0,  0.5),
    "right": ( 1.0,  0.5),
}

# ─── PARÂMETROS DE CÂMERA PROJECT ────────────────────────
FOCAL_DEPTH = 1250.0 / (1280 / CAM_W)
FOCAL_HORIZ = 1150.0 / (1280 / CAM_W)
CAM_HEIGHT  =  1.2
FOCAL_VERT  = FOCAL_DEPTH
FOCAL_REAR_HORIZ = FOCAL_HORIZ * 0.55
FOCAL_REAR_VERT  = FOCAL_VERT * 0.55
REAR_HORIZON_OFFSET = -35

# ─── CÂMERA BEV DINÂMICA 360° ────────────────────────────
#
#  Para o BEV traseiro ser visível a câmera DEVE ficar em
#  Y < -REAR_DIST (i.e. atrás do limite traseiro de detecção).
#  cam_y é fixo em -32 m; apenas a altura (cam_z) varia com a
#  velocidade para dar o zoom-out automático igual ao Tesla real.
#
#  FOV 75° (vertical) — amplo o suficiente para frente e trás
#  ficarem dentro do canvas sem distorção excessiva.
#
#  Posições verificadas analiticamente:
#    Repouso  → cobre  ~40 m frente,  ~18 m atrás do ego
#    130 km/h → cobre ~200 m frente,  ~15 m atrás do ego

BEV_CAM_Y = -32.0                                        # fixo, atrás de REAR_DIST
FOV_BEV   = np.deg2rad(75)
FOCAL_BEV = BEV_H / (2 * np.tan(FOV_BEV / 2))          # ≈ 469 px

def v_norm(v): return v / (np.linalg.norm(v) + 1e-9)

# Estado mutável da câmera BEV (atualizado a cada frame)
_bev_pos          = np.array([0.0, BEV_CAM_Y, 20.0], float)
_bev_fwd          = np.zeros(3, float)
_bev_right        = np.zeros(3, float)
_bev_up           = np.zeros(3, float)
_bev_smooth_speed = 0.0

def _rebuild_bev_basis(look_y=10.0):
    global _bev_fwd, _bev_right, _bev_up
    cam_at     = np.array([0.0, look_y, 0.0], float)
    _bev_fwd   = v_norm(cam_at - _bev_pos)
    _bev_right = v_norm(np.cross(_bev_fwd, [0, 0, 1]))
    _bev_up    = np.cross(_bev_right, _bev_fwd)

def update_bev_camera(speed_mps):
    """Recalcula câmera BEV a cada frame: zoom-out automático com a velocidade."""
    global _bev_pos, _bev_smooth_speed
    _bev_smooth_speed += 0.06 * (speed_mps - _bev_smooth_speed)   # τ ≈ 0.5 s
    t      = min(_bev_smooth_speed / 36.0, 1.0)                   # satura em 130 km/h
    cam_z  = 20.0 + t * 30.0   # altura: 20 m → 50 m
    look_y = 10.0 + t * 30.0   # mira:   10 m → 40 m à frente do ego
    _bev_pos = np.array([0.0, BEV_CAM_Y, cam_z], float)
    _rebuild_bev_basis(look_y)

_rebuild_bev_basis(10.0)   # inicializa com velocidade zero

def project_bev(P):
    d = P - _bev_pos
    z = float(np.dot(d, _bev_fwd))
    if z < 0.05: return None
    x = float(np.dot(d, _bev_right))
    y = float(np.dot(d, _bev_up))
    px = int(BEV_W / 2 + x / z * FOCAL_BEV)
    py = int(BEV_H / 2 - y / z * FOCAL_BEV)
    return px, py

# ═══════════════════════════════════════════════════════════
#  TRACKER: BYTETRACK (BEV) + KALMAN
# ═══════════════════════════════════════════════════════════
class KalmanTrack:
    def __init__(self, track_id, x, y, cls):
        self.id = track_id; self.cls = cls; self.age = 0; self.time_since_update = 0
        self.state = np.array([x, y, 0.0, 0.0], dtype=np.float32)
        self.P = np.eye(4, dtype=np.float32) * 5.0
        dt = 1.0 / FPS
        self.F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        self.Q = np.diag([0.1, 0.1, 0.001, 0.5])
        self.R = np.diag([1.5, 1.5])

    def predict(self):
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.time_since_update += 1
        return self.state[:2]

    def update(self, x, y):
        self.time_since_update = 0; self.age += 1
        y_res = np.array([x, y], dtype=np.float32) - (self.H @ self.state)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y_res
        self.P = (np.eye(4) - K @ self.H) @ self.P

class BEVByteTracker:
    def __init__(self, max_age=15, high_thresh=0.5, low_thresh=0.1, match_thresh=8.0):
        self.tracks = []; self.next_id = 0
        self.max_age = max_age; self.high_thresh = high_thresh
        self.low_thresh = low_thresh; self.match_thresh = match_thresh

    def _match(self, dets, tracks, thresh):
        if not dets or not tracks: return [], list(range(len(dets))), list(range(len(tracks)))
        cost = np.zeros((len(dets), len(tracks)), dtype=np.float32)
        for i, d in enumerate(dets):
            for j, t in enumerate(tracks):
                cost[i, j] = np.linalg.norm(t.state[:2] - np.array([d["ego_x"], d["ego_y"]]))
        r_ind, c_ind = linear_sum_assignment(cost)
        matches, un_d, un_t = [], [], []
        matched_d, matched_t = set(), set()
        for r, c in zip(r_ind, c_ind):
            if cost[r, c] < thresh:
                matches.append((r, c)); matched_d.add(r); matched_t.add(c)
        un_d = [i for i in range(len(dets)) if i not in matched_d]
        un_t = [j for j in range(len(tracks)) if j not in matched_t]
        return matches, un_d, un_t

    def update(self, detections):
        d_high = [d for d in detections if d["conf"] >= self.high_thresh]
        d_low  = [d for d in detections if self.low_thresh <= d["conf"] < self.high_thresh]
        for t in self.tracks: t.predict()
        m_high, u_d_high, u_t_high = self._match(d_high, self.tracks, self.match_thresh)
        for d_i, t_i in m_high: self.tracks[t_i].update(d_high[d_i]["ego_x"], d_high[d_i]["ego_y"])
        rem_t = [self.tracks[i] for i in u_t_high]
        m_low, _, _ = self._match(d_low, rem_t, self.match_thresh)
        for d_i, t_i in m_low: self.tracks[u_t_high[t_i]].update(d_low[d_i]["ego_x"], d_low[d_i]["ego_y"])
        for d_i in u_d_high:
            self.tracks.append(KalmanTrack(self.next_id, d_high[d_i]["ego_x"], d_high[d_i]["ego_y"], d_high[d_i]["cls"]))
            self.next_id += 1
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        return [{"id": t.id, "ego_x": float(t.state[0]), "ego_y": float(t.state[1]), "cls": t.cls}
                for t in self.tracks if t.age >= 2 or t.time_since_update == 0]

# ═══════════════════════════════════════════════════════════
#  NMS EGO-CÊNTRICO
# ═══════════════════════════════════════════════════════════

def _nms_ego(dets, min_dist=2.2):
    """Remove detecções sobrepostas no espaço ego-cêntrico.
    Mantém a de maior confiança quando duas estão a menos de min_dist metros."""
    kept = []
    for d in sorted(dets, key=lambda x: -x["conf"]):
        if not any(math.hypot(d["ego_x"] - k["ego_x"],
                              d["ego_y"] - k["ego_y"]) < min_dist
                   for k in kept):
            kept.append(d)
    return kept

# ═══════════════════════════════════════════════════════════
#  MAPA OSM
# ═══════════════════════════════════════════════════════════
TILE_SIZE  = 256
MAP_ZOOM   = 17
TILE_CACHE = "./tile_cache"

def _lat_lon_to_tile_frac(lat, lon, zoom):
    n = 2 ** zoom
    x = (lon + 180.0) / 360.0 * n
    lat_r = math.radians(lat)
    y = (1.0 - math.log(math.tan(lat_r) + 1.0 / math.cos(lat_r)) / math.pi) / 2.0 * n
    return x, y

def _download_tile(z, tx, ty):
    os.makedirs(TILE_CACHE, exist_ok=True)
    path = os.path.join(TILE_CACHE, f"{z}_{tx}_{ty}.png")
    if not os.path.exists(path):
        url = f"https://tile.openstreetmap.org/{z}/{tx}/{ty}.png"
        req = urllib.request.Request(url, headers={"User-Agent": "BevTesla/1.0 educational"})
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                with open(path, "wb") as f:
                    f.write(resp.read())
            time.sleep(0.05)   # respeita rate limit OSM
        except Exception as e:
            print(f"  Tile {z}/{tx}/{ty} falhou: {e}")
            return None
    return cv2.imread(path)

def build_osm_map(gps_trail, zoom=MAP_ZOOM):
    """Baixa e costura tiles OSM para o bounding box do trajeto."""
    lats = [p[0] for p in gps_trail]
    lons = [p[1] for p in gps_trail]
    pad  = 0.0008   # margem em graus
    lat_max = max(lats) + pad;  lat_min = min(lats) - pad
    lon_min = min(lons) - pad;  lon_max = max(lons) + pad

    fx0, fy0 = _lat_lon_to_tile_frac(lat_max, lon_min, zoom)
    fx1, fy1 = _lat_lon_to_tile_frac(lat_min, lon_max, zoom)

    tx_min, ty_min = int(fx0),     int(fy0)
    tx_max, ty_max = int(fx1) + 1, int(fy1) + 1

    n_x = tx_max - tx_min
    n_y = ty_max - ty_min
    print(f"  Baixando {n_x * n_y} tile(s) OSM (zoom {zoom})...")

    stitched = np.full((n_y * TILE_SIZE, n_x * TILE_SIZE, 3), 210, np.uint8)
    for ty in range(ty_min, ty_max):
        for tx in range(tx_min, tx_max):
            tile = _download_tile(zoom, tx, ty)
            if tile is not None:
                py = (ty - ty_min) * TILE_SIZE
                px = (tx - tx_min) * TILE_SIZE
                stitched[py:py + TILE_SIZE, px:px + TILE_SIZE] = tile[:TILE_SIZE, :TILE_SIZE]

    return stitched, tx_min, ty_min, zoom

def gps_to_map_px(lat, lon, tx_min, ty_min, zoom):
    fx, fy = _lat_lon_to_tile_frac(lat, lon, zoom)
    return int((fx - tx_min) * TILE_SIZE), int((fy - ty_min) * TILE_SIZE)

def ego_centric_to_gps(ego_x, ego_y, ego_lat, ego_lon, heading_deg):
    """Coordenada ego-cêntrica (x=direita, y=frente) em metros → GPS absoluto."""
    h     = math.radians(heading_deg)
    north = ego_y * math.cos(h) - ego_x * math.sin(h)
    east  = ego_y * math.sin(h) + ego_x * math.cos(h)
    dlat  = north / 111320.0
    dlon  = east  / (111320.0 * math.cos(math.radians(ego_lat)))
    return ego_lat + dlat, ego_lon + dlon

def render_map_frame(base_map, gps_trail, veh_gps_trails,
                     ego_lat, ego_lon, tx_min, ty_min, zoom):
    frame = base_map.copy()
    h_map, w_map = frame.shape[:2]

    # ── Trajeto do ego ──
    n = len(gps_trail)
    for i in range(1, n):
        alpha = 0.25 + 0.75 * (i / n)
        color = tuple(int(c * alpha) for c in FSD_BLUE_CORE)
        p1 = gps_to_map_px(gps_trail[i-1][0], gps_trail[i-1][1], tx_min, ty_min, zoom)
        p2 = gps_to_map_px(gps_trail[i  ][0], gps_trail[i  ][1], tx_min, ty_min, zoom)
        cv2.line(frame, p1, p2, color, 4, cv2.LINE_AA)

    # ── Rastros dos veículos detectados ──
    for tid, trail in veh_gps_trails.items():
        pts = list(trail)
        if len(pts) < 2:
            continue
        cls    = pts[0][2]
        base_c = TRAIL_COLORS.get(cls, (180, 180, 180))
        m = len(pts)
        for i in range(1, m):
            alpha = 0.2 + 0.8 * (i / m)
            color = tuple(int(c * alpha) for c in base_c)
            p1 = gps_to_map_px(pts[i-1][0], pts[i-1][1], tx_min, ty_min, zoom)
            p2 = gps_to_map_px(pts[i  ][0], pts[i  ][1], tx_min, ty_min, zoom)
            cv2.line(frame, p1, p2, color, 2, cv2.LINE_AA)
        # Ponto atual do veículo
        lp = gps_to_map_px(pts[-1][0], pts[-1][1], tx_min, ty_min, zoom)
        cv2.circle(frame, lp, 6, base_c, -1, cv2.LINE_AA)
        cv2.circle(frame, lp, 7, (255, 255, 255), 1, cv2.LINE_AA)

    # ── Posição atual do ego ──
    ego_px = gps_to_map_px(ego_lat, ego_lon, tx_min, ty_min, zoom)
    cv2.circle(frame, ego_px, 10, FSD_BLUE_CORE, -1, cv2.LINE_AA)
    cv2.circle(frame, ego_px, 12, (255, 255, 255), 2, cv2.LINE_AA)

    # ── Crop centralizado no ego ──
    cx, cy = ego_px
    x0 = max(0, min(cx - BEV_W // 2, w_map - BEV_W))
    y0 = max(0, min(cy - BEV_H // 2, h_map - BEV_H))
    crop = frame[y0:y0 + BEV_H, x0:x0 + BEV_W]

    if crop.shape[:2] != (BEV_H, BEV_W):
        padded = np.full((BEV_H, BEV_W, 3), 210, np.uint8)
        padded[:crop.shape[0], :crop.shape[1]] = crop
        crop = padded

    # Legenda de classes
    legend_y = 20
    for cls, color in TRAIL_COLORS.items():
        cv2.circle(crop, (14, legend_y), 5, color, -1, cv2.LINE_AA)
        cv2.putText(crop, cls, (24, legend_y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (30, 30, 30), 1, cv2.LINE_AA)
        legend_y += 18
    # Ego na legenda
    cv2.circle(crop, (14, legend_y), 5, FSD_BLUE_CORE, -1, cv2.LINE_AA)
    cv2.putText(crop, "ego", (24, legend_y + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (30, 30, 30), 1, cv2.LINE_AA)

    cv2.putText(crop, "\xa9 OpenStreetMap contributors",
                (BEV_W - 210, BEV_H - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.28, (80, 80, 80), 1, cv2.LINE_AA)

    return crop

# ═══════════════════════════════════════════════════════════
#  RENDERIZAÇÃO BEV (original)
# ═══════════════════════════════════════════════════════════

def draw_fsd_path_gradient(img, offset_y):
    path_width = LANE_HALF * 0.85
    y_pts = np.linspace(CAR_FRONT_TIP, MAX_DIST, 50)
    for i in range(len(y_pts) - 1):
        y1, y2 = y_pts[i], y_pts[i+1]
        progress = (y1 - CAR_FRONT_TIP) / (MAX_DIST - CAR_FRONT_TIP)
        alpha = max(0.0, 1.0 - (progress ** 1.2))
        color = (
            int(BEV_BG[0] * (1 - alpha) + FSD_BLUE_CORE[0] * alpha),
            int(BEV_BG[1] * (1 - alpha) + FSD_BLUE_CORE[1] * alpha),
            int(BEV_BG[2] * (1 - alpha) + FSD_BLUE_CORE[2] * alpha)
        )
        pl1, pr1 = project_bev(np.array([-path_width, y1, 0.01])), project_bev(np.array([path_width, y1, 0.01]))
        pl2, pr2 = project_bev(np.array([-path_width, y2, 0.01])), project_bev(np.array([path_width, y2, 0.01]))
        if pl1 and pr1 and pl2 and pr2:
            cv2.fillPoly(img, [np.array([pl1, pr1, pr2, pl2])], color)

    flow_spacing = 8.0
    shift = offset_y % flow_spacing
    curr_y = CAR_FRONT_TIP - shift
    while curr_y < MAX_DIST:
        if curr_y > CAR_FRONT_TIP + 1.0:
            progress = (curr_y - CAR_FRONT_TIP) / (MAX_DIST - CAR_FRONT_TIP)
            alpha = max(0.0, 1.0 - (progress ** 1.5))
            flow_color = (
                int(FSD_BLUE_CORE[0] * (1 - alpha) + FSD_FLOW_COLOR[0] * alpha),
                int(FSD_BLUE_CORE[1] * (1 - alpha) + FSD_FLOW_COLOR[1] * alpha),
                int(FSD_BLUE_CORE[2] * (1 - alpha) + FSD_FLOW_COLOR[2] * alpha)
            )
            p_center1 = project_bev(np.array([0, curr_y, 0.02]))
            p_center2 = project_bev(np.array([0, curr_y + 1.5, 0.02]))
            if p_center1 and p_center2:
                cv2.line(img, p_center1, p_center2, flow_color, 2, cv2.LINE_AA)
        curr_y += flow_spacing

def draw_clean_lanes_animated(img, offset_y):
    dash_length = 3.0
    gap_length = 5.0
    cycle = dash_length + gap_length
    shift = offset_y % cycle
    y_vals = np.linspace(-REAR_DIST, MAX_DIST, 40)
    for side in (-1, 1):
        pts = [p for p in (project_bev(np.array([side * (LANE_HALF+LANE_WIDTH), y, 0])) for y in y_vals) if p]
        if len(pts) > 1: cv2.polylines(img, [np.array(pts, np.int32)], False, LANE_LINE, 2, cv2.LINE_AA)
    curr_y = -REAR_DIST - shift
    while curr_y < MAX_DIST:
        for side in (-1, 1):
            p1 = project_bev(np.array([side * LANE_HALF, curr_y, 0]))
            p2 = project_bev(np.array([side * LANE_HALF, curr_y + dash_length, 0]))
            if p1 and p2:
                cv2.line(img, p1, p2, LANE_LINE, 2, cv2.LINE_AA)
        curr_y += cycle

def draw_solid_3d_vehicle(img, center, dims, is_ego=False):
    w, h_dim, d_val = dims
    x0, y0, z0 = center
    corners = np.array([
        [x0 - w/2, y0 - d_val/2, z0], [x0 + w/2, y0 - d_val/2, z0],
        [x0 + w/2, y0 + d_val/2, z0], [x0 - w/2, y0 + d_val/2, z0],
        [x0 - w/2, y0 - d_val/2, z0 + h_dim], [x0 + w/2, y0 - d_val/2, z0 + h_dim],
        [x0 + w/2, y0 + d_val/2, z0 + h_dim], [x0 - w/2, y0 + d_val/2, z0 + h_dim],
    ])
    proj = [project_bev(c) for c in corners]
    if not all(proj): return
    base_color = EGO_COLOR if is_ego else OBJ_COLOR
    top_color  = tuple(min(255, int(c * 1.4)) for c in base_color)
    side_color = tuple(int(c * 0.9) for c in base_color)
    back_color = tuple(int(c * 0.7) for c in base_color)
    cv2.fillPoly(img, [np.array([proj[0], proj[1], proj[2], proj[3]])], SHADOW_COLOR)
    for face_idx, color in [([0,1,5,4], back_color), ([1,2,6,5], side_color),
                             ([0,3,7,4], side_color), ([2,3,7,6], base_color),
                             ([4,5,6,7], top_color)]:
        cv2.fillPoly(img, [np.array([proj[i] for i in face_idx], np.int32)], color)
    cv2.polylines(img, [np.array([proj[4], proj[5], proj[6], proj[7]], np.int32)], True, top_color, 1, cv2.LINE_AA)

def render_tesla_ui(tracked, ego_state, offset_y):
    bev = np.full((BEV_H, BEV_W, 3), BEV_BG, np.uint8)
    draw_clean_lanes_animated(bev, offset_y)
    draw_fsd_path_gradient(bev, offset_y)
    for obj in tracked:
        if -30 < obj["ego_x"] < 30 and -REAR_DIST < obj["ego_y"] < MAX_DIST:
            draw_solid_3d_vehicle(bev, (obj["ego_x"], obj["ego_y"], 0), DIMS.get(obj["cls"], DIMS["car"]))
    draw_solid_3d_vehicle(bev, (0, 0, 0), DIMS["car"], is_ego=True)

    speed_kph = int(ego_state.get('speed_mps', 0) * 3.6)
    accel_x   = ego_state.get('accel_x', 0)
    heading   = ego_state.get('heading', 0)
    lat       = ego_state.get('lat', 0.0)
    lon       = ego_state.get('lon', 0.0)

    cv2.putText(bev, "PRND", (40, 50), cv2.FONT_HERSHEY_DUPLEX, 0.6, UI_ACCENT, 1, cv2.LINE_AA)
    cv2.putText(bev, f"{speed_kph}", (BEV_W // 2 - 35, 60), cv2.FONT_HERSHEY_DUPLEX, 1.8, TEXT_COLOR, 2, cv2.LINE_AA)
    cv2.putText(bev, "km/h", (BEV_W // 2 + 35, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, UI_ACCENT, 1, cv2.LINE_AA)
    bottom_y = BEV_H - 30
    cv2.putText(bev, f"HDG: {heading:05.1f} deg", (30, bottom_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_COLOR, 1, cv2.LINE_AA)
    cv2.putText(bev, f"ACC: {accel_x:+.2f} m/s^2", (200, bottom_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_COLOR, 1, cv2.LINE_AA)
    cv2.putText(bev, f"GPS: {lat:.5f}, {lon:.5f}", (BEV_W - 250, bottom_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_COLOR, 1, cv2.LINE_AA)
    compass_x = BEV_W // 2
    compass_y = 85
    cv2.line(bev, (compass_x - 20, compass_y), (compass_x + 20, compass_y), UI_ACCENT, 1, cv2.LINE_AA)
    dx = int(15 * math.sin(math.radians(heading)))
    dy = int(15 * math.cos(math.radians(heading)))
    cv2.circle(bev, (compass_x + dx, compass_y - dy), 3, FSD_BLUE_CORE, -1, cv2.LINE_AA)
    return bev

# ─── FUNÇÕES AUXILIARES ─────────────────────────────────
def project_front_cam(wx, wy):
    if wy < 0.5: return None
    return int(CAM_W/2 + (wx*FOCAL_HORIZ)/wy), int(CAM_H/2 + (CAM_HEIGHT*FOCAL_VERT)/wy)

def project_rear_cam(wx, wy):
    if wy > -0.5: return None
    return int(CAM_W/2 + (-wx*FOCAL_REAR_HORIZ)/-wy), int(CAM_H/2 + (CAM_HEIGHT*FOCAL_REAR_VERT)/-wy) + REAR_HORIZON_OFFSET

def draw_cam_lanes_minimal(frame, is_front=True):
    overlay = frame.copy()
    y_range = np.linspace(3.0, 60, 40) if is_front else np.linspace(-2, -40, 30)
    for side in (-LANE_HALF, LANE_HALF):
        pts = [p for p in (project_front_cam(side, y) if is_front else project_rear_cam(side, y) for y in y_range) if p]
        if len(pts) > 1: cv2.polylines(overlay, [np.array(pts)], False, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    return frame

def load_telemetry(csv_path):
    data = []
    last_valid = {"speed_mps": 0.0, "accel_x": 0.0, "accel_y": 0.0,
                  "heading": 0.0, "lat": 0.0, "lon": 0.0}

    def parse_value(val_str, key):
        try:
            if val_str is None or str(val_str).strip().lower() == 'na':
                return last_valid[key]
            val_float = float(val_str)
            if val_float == 0.0:
                return last_valid[key]
            last_valid[key] = val_float
            return val_float
        except ValueError:
            return last_valid[key]

    try:
        with open(csv_path, newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                data.append({
                    "speed_mps": parse_value(row.get("speed_mps"),            "speed_mps"),
                    "accel_x":   parse_value(row.get("acceleration_x_mps2"), "accel_x"),
                    "accel_y":   parse_value(row.get("acceleration_y_mps2"), "accel_y"),
                    "heading":   parse_value(row.get("heading_deg"),          "heading"),
                    "lat":       parse_value(row.get("GPS_latitude_deg"),     "lat"),
                    "lon":       parse_value(row.get("GPS_longitude_deg"),    "lon"),
                })
    except Exception as e:
        print(f"Aviso: Erro ao ler telemetria ({e})")
    return data

# ═══════════════════════════════════════════════════════════
#  MAIN LOOP
# ═══════════════════════════════════════════════════════════

def main():
    model          = YOLO("yolov8n.pt")
    tracker        = BEVByteTracker()
    caps           = {k: cv2.VideoCapture(v) for k, v in VIDEOS.items()}
    telemetry_data = load_telemetry(TELEMETRY_CSV)

    # ── Prepara mapa OSM ──────────────────────────────────
    gps_all = [(r["lat"], r["lon"]) for r in telemetry_data
               if r["lat"] != 0.0 and r["lon"] != 0.0]
    has_map = len(gps_all) >= 2
    base_map = tx_min = ty_min = map_zoom = None
    out_map  = None
    if has_map:
        print("Construindo mapa OSM...")
        base_map, tx_min, ty_min, map_zoom = build_osm_map(gps_all)
        out_map = cv2.VideoWriter(OUT_MAP, cv2.VideoWriter_fourcc(*"mp4v"),
                                  FPS, (BEV_W, BEV_H))
    else:
        print("GPS insuficiente — mapa desativado.")

    out = cv2.VideoWriter(OUT_VIDEO, cv2.VideoWriter_fourcc(*"mp4v"),
                          FPS, (CAM_W * 2 + BEV_W, BEV_H))
    print("Gerando UI FSD Clone Dinâmica com Telemetria e Movimento...")

    global_offset_y  = 0.0
    gps_trail        = []                  # [(lat, lon)]  trajeto ego
    veh_gps_trails   = {}                  # {id: deque[(lat, lon, cls)]}

    for frame_idx in range(MAX_FRAMES):
        frames, ok = {}, True
        for cam, cap in caps.items():
            ret, f = cap.read()
            if not ret: ok = False; break
            frames[cam] = cv2.resize(f, (CAM_W, CAM_H))
        if not ok: break

        ego_state = (telemetry_data[frame_idx]
                     if frame_idx < len(telemetry_data)
                     else {"speed_mps": 0, "heading": 0})

        dt = 1.0 / FPS
        speed_now = ego_state.get("speed_mps", 0)
        global_offset_y += speed_now * dt
        update_bev_camera(speed_now)

        global_dets = []
        cam_views   = {}

        for cam, frame in frames.items():
            view = frame.copy()
            for b in model(frame, conf=0.15, verbose=False)[0].boxes:
                cls = model.names[int(b.cls[0])]
                if cls not in DIMS: continue
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                dist = (H_REAL[cls] * FOCAL_DEPTH) / max(y2 - y1, 1)
                lat  = (((x1+x2)/2 - CAM_W/2) * dist) / FOCAL_HORIZ
                if cam == "front": ex, ey = lat,  dist
                elif cam == "back": ex, ey = -lat, -dist
                elif cam == "left": ex, ey = -dist, lat
                else:               ex, ey =  dist, -lat
                ox, oy = CAM_OFFSETS[cam]
                global_dets.append({"ego_x": ex+ox, "ego_y": ey+oy,
                                     "cls": cls, "conf": float(b.conf[0])})
                cv2.rectangle(view, (x1, y1), (x2, y2), (200, 200, 200), 1)
            cam_views[cam] = (draw_cam_lanes_minimal(view, cam == "front")
                              if cam in ["front", "back"] else view)

        global_dets = _nms_ego(global_dets)
        tracked = tracker.update(global_dets)

        # ── Atualiza trajetos GPS ─────────────────────────
        ego_lat  = ego_state.get("lat", 0.0)
        ego_lon  = ego_state.get("lon", 0.0)
        ego_hdg  = ego_state.get("heading", 0.0)

        if ego_lat != 0.0 and ego_lon != 0.0:
            if not gps_trail or (ego_lat, ego_lon) != gps_trail[-1]:
                gps_trail.append((ego_lat, ego_lon))

        active_ids = set()
        for obj in tracked:
            tid = obj["id"]
            active_ids.add(tid)
            if ego_lat != 0.0 and ego_lon != 0.0:
                veh_lat, veh_lon = ego_centric_to_gps(
                    obj["ego_x"], obj["ego_y"], ego_lat, ego_lon, ego_hdg)
                if tid not in veh_gps_trails:
                    veh_gps_trails[tid] = deque(maxlen=300)
                veh_gps_trails[tid].append((veh_lat, veh_lon, obj["cls"]))
        for old_id in list(veh_gps_trails):
            if old_id not in active_ids:
                del veh_gps_trails[old_id]

        # ── BEV original ─────────────────────────────────
        bev = render_tesla_ui(tracked, ego_state, global_offset_y)
        grid = np.vstack([np.hstack([cam_views["front"], cam_views["right"]]),
                          np.hstack([cam_views["left"],  cam_views["back"]])])
        out.write(np.hstack([grid, bev]))

        # ── Mapa OSM ─────────────────────────────────────
        if has_map and out_map and ego_lat != 0.0 and len(gps_trail) >= 2:
            map_frame = render_map_frame(base_map, gps_trail, veh_gps_trails,
                                         ego_lat, ego_lon,
                                         tx_min, ty_min, map_zoom)
            out_map.write(map_frame)

        if frame_idx % FPS == 0:
            print(f"{frame_idx // FPS}s processados... "
                  f"(Velocidade: {int(ego_state.get('speed_mps', 0)*3.6)} km/h"
                  f" | tracks: {len(tracked)})")

    for cap in caps.values(): cap.release()
    out.release()
    if out_map: out_map.release()
    print("Sucesso!")
    print("  BEV:", OUT_VIDEO)
    if has_map: print("  Mapa:", OUT_MAP)

if __name__ == "__main__":
    main()
