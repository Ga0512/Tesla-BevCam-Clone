import cv2
import numpy as np
import csv
import math
import os
import time
import urllib.request
from collections import deque
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment# ─── CAMINHOS E ARQUIVOS ─────────────────────────────────
# Only "front" is required. Set optional cameras to None to disable.
VIDEOS = {
    "front": "/mnt/c/Users/gabriel.cicotoste/HDSeagate/Projects/BevTesla/videos/rapidsave.com__-9iv42du4doih1.mp4",
    "back":  None,
    "left":  None,
    "right": None,
}
TELEMETRY_CSV = None   # opcional: defina o caminho do CSV para ativar a telemetria
DEPTH_MODEL   = "depth_anything_v2"   # None para desativar; "depth_anything_v2" = V2 Small (leve)
DEPTH_MAX_M   = 100.0   # pixel mais distante (sem céu) considerado como 100 m
DEPTH_OVERLAY = True    # mostra colormap de profundidade na view frontal
UFLD_MODEL    = "ufldv2_tusimple_res18.pth"   # UFLDv2 (Tusimple) para detecção de faixa; None desativa
UFLD_EMA      = 0.65    # fator de suavização temporal dos polinômios de faixa (0=nenhuma, 1=máx)
OUT_VIDEO     = "tesla_bev_fsd_clone_9iv42du4doih1.mp4"
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
MAX_DIST      = 70.0
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
    "front":      ( 0.0,  1.8),
    "back":       ( 0.0, -2.0),
    "left":       (-1.0,  0.5),
    "right":      ( 1.0,  0.5),
}
CAM_YAW = {
    "front":       0.0,
    "back":        math.pi,
    "left":        math.pi / 2,
    "right":      -math.pi / 2,
}

# ─── PARÂMETROS DE CÂMERA PROJECT ────────────────────────
FOCAL_DEPTH = 1250.0 / (1280 / CAM_W)
FOCAL_HORIZ = 1150.0 / (1280 / CAM_W)
CAM_HEIGHT  = 1.2     # altura da câmera em relação ao chão (m)

# ─── CÂMERA BEV DINÂMICA 360° ────────────────────────────
#
#  Para o BEV traseiro ser visível a câmera DEVE ficar em
#  Y < -REAR_DIST (i.e. atrás do limite traseiro de detecção).
#  cam_y é fixo em -10 m (mais perto do ego → menos rua atrás);
#  altura (cam_z) e mira (look_y) variam com a velocidade.
#
#  Câmera baixa e logo atrás do ego → vista de "chase cam":
#  um pouco acima do carro, enxergando o carro ego inteiro.
#  Mira longa (look_y) → carro desce no canvas (mais perto de
#  nós) e o horizonte fica visível até ~100 m.
#
#  FOV 75° (vertical) — amplo o suficiente para frente e trás
#  ficarem dentro do canvas sem distorção excessiva.

BEV_CAM_Y = -10.0                                        # fixo, atrás do ego
FOV_BEV   = np.deg2rad(75)
FOCAL_BEV = BEV_H / (2 * np.tan(FOV_BEV / 2))          # ≈ 469 px

def v_norm(v): return v / (np.linalg.norm(v) + 1e-9)

# Estado mutável da câmera BEV (atualizado a cada frame)
_bev_pos          = np.array([0.0, BEV_CAM_Y, 14.0], float)
_bev_fwd          = np.zeros(3, float)
_bev_right        = np.zeros(3, float)
_bev_up           = np.zeros(3, float)
_bev_smooth_speed = 0.0

def _rebuild_bev_basis(look_y=26.0):
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
    cam_z  = 14.0 + t * 8.0    # altura: 14 m → 22 m (mantém o carro em vista)
    look_y = 26.0 + t * 10.0   # mira:   26 m → 36 m à frente do ego
    _bev_pos = np.array([0.0, BEV_CAM_Y, cam_z], float)
    _rebuild_bev_basis(look_y)

_rebuild_bev_basis(26.0)   # inicializa com velocidade zero

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

def render_tesla_ui(tracked, ego_state, offset_y, lane_polys=None):
    bev = np.full((BEV_H, BEV_W, 3), BEV_BG, np.uint8)
    if lane_polys:
        draw_detected_lanes(bev, offset_y, lane_polys)
        draw_fsd_path_gradient_real(bev, lane_polys)
    else:
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

    # ── Contagem de objetos por classe ──
    cls_counts = {}
    for obj in tracked:
        cls_counts[obj["cls"]] = cls_counts.get(obj["cls"], 0) + 1
    counts_str = "  ".join(f"{k}:{v}" for k, v in sorted(cls_counts.items())) or "nenhum"

    # ── Objeto mais próximo à frente (para ACC/TTC) ──
    front_objs = [o for o in tracked if o["ego_y"] > CAR_FRONT_TIP]
    closest = min(front_objs, key=lambda o: o["ego_y"]) if front_objs else None

    cv2.putText(bev, "PRND", (40, 50), cv2.FONT_HERSHEY_DUPLEX, 0.6, UI_ACCENT, 1, cv2.LINE_AA)
    cv2.putText(bev, f"OBJ: {len(tracked)}  [{counts_str}]", (BEV_W // 2 - 160, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)
    if closest:
        cv2.putText(bev, f"FRENTE: {closest['ego_y']:.1f} m", (BEV_W - 220, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, FSD_BLUE_CORE, 1, cv2.LINE_AA)
    bottom_y = BEV_H - 30
    cv2.putText(bev, f"HDG: {heading:05.1f} deg", (30, bottom_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_COLOR, 1, cv2.LINE_AA)
    cv2.putText(bev, f"ACC: {accel_x:+.2f} m/s^2", (200, bottom_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_COLOR, 1, cv2.LINE_AA)
    cv2.putText(bev, f"GPS: {lat:.5f}, {lon:.5f}", (BEV_W - 250, bottom_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_COLOR, 1, cv2.LINE_AA)
    if lane_polys and (1 in lane_polys) and (2 in lane_polys):
        x_l = _poly_x(lane_polys[1], 0.0); x_r = _poly_x(lane_polys[2], 0.0)
        lane_offset = (x_l + x_r) / 2.0
        lane_width  = x_r - x_l
        # curvatura da pista no ponto 2m à frente (κ ≈ 2c / (1+b²)^1.5)
        b = 0.5 * (lane_polys[1][1] + lane_polys[2][1])
        c = 0.5 * (lane_polys[1][0] + lane_polys[2][0])
        curv = abs(2.0 * c) / (1.0 + b * b) ** 1.5 if abs(b) > 1e-6 else abs(2.0 * c)
        cv2.putText(bev, f"LANE: {lane_offset:+.2f} m", (30, bottom_y - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, FSD_BLUE_CORE, 1, cv2.LINE_AA)
        cv2.putText(bev, f"W: {lane_width:.2f} m", (200, bottom_y - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, FSD_BLUE_CORE, 1, cv2.LINE_AA)
        cv2.putText(bev, f"CURV: {curv:.4f} 1/m", (330, bottom_y - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, FSD_BLUE_CORE, 1, cv2.LINE_AA)
    compass_x = BEV_W // 2
    compass_y = 85
    cv2.line(bev, (compass_x - 20, compass_y), (compass_x + 20, compass_y), UI_ACCENT, 1, cv2.LINE_AA)
    dx = int(15 * math.sin(math.radians(heading)))
    dy = int(15 * math.cos(math.radians(heading)))
    cv2.circle(bev, (compass_x + dx, compass_y - dy), 3, FSD_BLUE_CORE, -1, cv2.LINE_AA)
    return bev

# ─── FUNÇÕES AUXILIARES ─────────────────────────────────
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
#  DEPTH ESTIMATION (Depth Anything V2 Small)
# ═══════════════════════════════════════════════════════════

_depth_model = None
_depth_processor = None

def _load_depth_model():
    """Carrega Depth Anything V2 Small via transformers (GPU se disponível)."""
    global _depth_model, _depth_processor
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        _depth_processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
        _depth_model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
        if torch.cuda.is_available():
            _depth_model.to("cuda")
        _depth_model.eval()
        print("  Depth: Depth Anything V2 Small ativo.")
        return True
    except Exception as e:
        print(f"  Aviso: depth desativado ({e})")
        _depth_model = None
        return False

def compute_depth_map(frame):
    """Mapa de profundidade em metros, calibrado pela geometria do chão.

    O depth relativo é convertido para metros ajustando uma escala única:
    para pixels do chão abaixo do horizonte, a distância geométrica é
    conhecida (flat-ground: dist = CAM_HEIGHT * FOCAL_DEPTH / (py - horizon)),
    então escala = mediana(dist_geo * pred) nessa faixa. Isso ancora o
    relativo na geometria da câmera, sem depender do horizonte (que era
    o bug: tudo ficava esticado/afastado)."""
    import torch
    import numpy as np
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    inputs = _depth_processor(images=rgb, return_tensors="pt")
    device = next(_depth_model.parameters()).device
    with torch.no_grad():
        pred = _depth_model(**inputs.to(device)).predicted_depth
    pred = torch.nn.functional.interpolate(
        pred.unsqueeze(1), size=(CAM_H, CAM_W),
        mode="bicubic", align_corners=False).squeeze().cpu().numpy()
    horizon_row = CAM_H // 2
    rows = np.arange(horizon_row + 5, CAM_H)
    dist_geo = CAM_HEIGHT * FOCAL_DEPTH / (rows - horizon_row)
    ratios = dist_geo[:, None] * np.maximum(pred[rows, :], 1e-3)
    scale = float(np.median(ratios))
    return np.clip(scale / np.maximum(pred, 1e-3), 0.0, DEPTH_MAX_M * 1.5)

# ═══════════════════════════════════════════════════════════
#  LANE DETECTION (UFLDv2 — Tusimple ResNet18)
# ═══════════════════════════════════════════════════════════
#
#  A entrada esperada é 3×320×800 (Tensor de 800 de largura).
#  Preprocess: Resize para 400×800 + crop das 320 linhas inferiores.
#  Saída: até 4 faixas, cada uma como lista de pontos (x, y) em
#  pixels no espaço do frame de entrada.
#
#  Projeção flat-ground idêntica ao depth: para cada ponto da faixa,
#      dist = CAM_HEIGHT * FOCAL_DEPTH / (py - horizon)
#      ego_x = (px - CAM_W/2) * dist / FOCAL_DEPTH
#      ego_y = dist
#  Em seguida ajusta-se um polinômio x(y) por faixa, suavizado com
#  EMA (UFLD_EMA) — igual ao que ADAS reais fazem antes de renderizar.

_ufld_net    = None
_ufld_anchors = {}
_ufld_smoothed = {}   # {lane_idx: poly(x = a + b*y + c*y^2)}

def _load_ufld_model():
    """Carrega UFLDv2 (Tusimple ResNet18) se o peso existir."""
    global _ufld_net, _ufld_anchors
    try:
        import torch
        import torchvision
        if not os.path.exists(UFLD_MODEL):
            print(f"  UFLD: peso '{UFLD_MODEL}' ausente — lane desativado.")
            return False

        # ── parsingNet reproduzido standalone (arquitetura do repo) ──
        num_grid_row, num_cls_row = 100, 56
        num_grid_col, num_cls_col = 100, 41
        num_lanes = 4
        input_h, input_w = 320, 800
        mlp_mid = 2048
        input_dim = input_h // 32 * input_w // 32 * 8
        dim1 = num_grid_row * num_cls_row * num_lanes
        dim2 = num_grid_col * num_cls_col * num_lanes
        dim3 = 2 * num_cls_row * num_lanes
        dim4 = 2 * num_cls_col * num_lanes
        total_dim = dim1 + dim2 + dim3 + dim4

        res = torchvision.models.resnet18(pretrained=False)
        layers = [res.conv1, res.bn1, res.relu, res.maxpool,
                  res.layer1, res.layer2, res.layer3, res.layer4]

        class UFLDBackbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1, self.bn1, self.relu, self.maxpool = layers[:4]
                self.layer1, self.layer2, self.layer3, self.layer4 = layers[4:]
            def forward(self, x):
                x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
                x2 = self.layer1(x)
                x3 = self.layer2(x2)
                x4 = self.layer3(x3)
                fea = self.layer4(x4)     # 512 canais → pool
                return x2, x3, x4, fea

        class UFLLaneNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = UFLDBackbone()
                self.pool = torch.nn.Conv2d(512, 8, 1)
                self.cls = torch.nn.Sequential(
                    torch.nn.Identity(),          # cls.0 (fc_norm=False)
                    torch.nn.Linear(input_dim, mlp_mid),
                    torch.nn.ReLU(),
                    torch.nn.Linear(mlp_mid, total_dim),
                )
            def forward(self, x):
                x2, x3, x4, fea = self.backbone(x)
                fea = self.pool(fea).view(-1, input_dim)
                out = self.cls(fea)
                return {
                    "loc_row":   out[:, :dim1].view(-1, num_grid_row, num_cls_row, num_lanes),
                    "loc_col":   out[:, dim1:dim1+dim2].view(-1, num_grid_col, num_cls_col, num_lanes),
                    "exist_row": out[:, dim1+dim2:dim1+dim2+dim3].view(-1, 2, num_cls_row, num_lanes),
                    "exist_col": out[:, dim1+dim2+dim3:].view(-1, 2, num_cls_col, num_lanes),
                }

        net = UFLLaneNet()
        sd = torch.load(UFLD_MODEL, map_location="cpu")["model"]
        sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
        # renomeia 'model.' → 'backbone.'
        sd = {("backbone." + k[6:]) if k.startswith("model.") else k: v
              for k, v in sd.items()}
        net.load_state_dict(sd, strict=False)
        if torch.cuda.is_available():
            net.to("cuda")
        net.eval()
        _ufld_net = net
        _ufld_anchors = {
            "row": torch.linspace(160, 710, num_cls_row) / 720.0,
            "col": torch.linspace(0, 1, num_cls_col),
        }
        _ufld_smoothed.clear()
        print("  UFLD: UFLDv2 Tusimple ResNet18 ativo.")
        return True
    except Exception as e:
        print(f"  Aviso: UFLD desativado ({e})")
        _ufld_net = None
        return False

def _ufld_preprocess(frame):
    import torch
    import torchvision
    from PIL import Image
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    t = torchvision.transforms.Compose([
        torchvision.transforms.Resize((400, 800)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406),
                                         (0.229, 0.224, 0.225)),
    ])
    img = t(img)[:, -320:, :].unsqueeze(0)
    return img

def _ufld_pixels(pred):
    """Decodifica a saída da rede em pontos (x, y) por faixa, no espaço
    do frame (640×360). Faixas 1/2 = margens da pista do ego (row anchors),
    faixas 0/3 = faixas externas (col anchors)."""
    import torch
    lr = pred["loc_row"][0]; lc = pred["loc_col"][0]
    er = pred["exist_row"][0].argmax(0); ec = pred["exist_col"][0].argmax(0)
    ngr, ncr, nlr = lr.shape
    ngc, ncc, nlc = lc.shape
    mir = lr.argmax(0); mic = lc.argmax(0)
    row_anchor = _ufld_anchors["row"]
    col_anchor = _ufld_anchors["col"]
    lanes = {}

    # Faixas detectadas via row anchors (1, 2): x varia, y fixo
    for i in (1, 2):
        pts = []
        if er[:, i].sum() > ncr / 2:
            for k in range(ncr):
                if er[k, i]:
                    a = torch.arange(max(0, mir[k, i] - 1),
                                     min(ngr - 1, mir[k, i] + 1) + 1,
                                     device=lr.device)
                    x = (lr[a, k, i].softmax(0) * a.float()).sum().item() + 0.5
                    x = x / (ngr - 1) * CAM_W
                    y = float(row_anchor[k] * CAM_H)
                    pts.append((int(x), int(y)))
        lanes[i] = pts

    # Faixas externas via col anchors (0, 3): y varia, x fixo
    for i in (0, 3):
        pts = []
        if ec[:, i].sum() > ncc / 4:
            for k in range(ncc):
                if ec[k, i]:
                    a = torch.arange(max(0, mic[k, i] - 1),
                                     min(ngc - 1, mic[k, i] + 1) + 1,
                                     device=lc.device)
                    y = (lc[a, k, i].softmax(0) * a.float()).sum().item() + 0.5
                    y = y / (ngc - 1) * CAM_H
                    x = float(col_anchor[k] * CAM_W)
                    pts.append((int(x), int(y)))
        lanes[i] = pts
    return lanes

def detect_lanes(frame):
    """Retorna {lane_idx: lista de (x, y) em pixels do frame}."""
    global _ufld_net
    if _ufld_net is None:
        return {}
    import torch
    dev = next(_ufld_net.parameters()).device
    x = _ufld_preprocess(frame).to(dev)
    with torch.no_grad():
        pred = _ufld_net(x)
    return _ufld_pixels(pred)

def _fit_lane_polys(lanes_px):
    """Converte pontos de faixa para o frame do ego e ajusta um polinômio
    x(y) = a + b*y + c*y^2 por faixa. Suaviza com EMA entre frames."""
    import torch
    horizon = CAM_H // 2
    polys = {}
    for idx, pts in lanes_px.items():
        if len(pts) < 4:
            continue
        ego = []
        for (px, py) in pts:
            if py <= horizon:
                continue
            dist = CAM_HEIGHT * FOCAL_DEPTH / (py - horizon)
            ex = (px - CAM_W / 2) * dist / FOCAL_DEPTH
            if 0.5 < dist < MAX_DIST + 15:
                ego.append((ex, dist))
        if len(ego) < 4:
            continue
        ego = np.array(ego)
        y_vals = ego[:, 1]; x_vals = ego[:, 0]
        order = np.argsort(y_vals)
        y_vals = y_vals[order]; x_vals = x_vals[order]
        # polyfit x = f(y), grau 2 — curvas suaves
        coeffs = np.polyfit(y_vals, x_vals, 2)
        # EMA sobre os coeficientes
        if idx in _ufld_smoothed:
            coeffs = UFLD_EMA * _ufld_smoothed[idx] + (1 - UFLD_EMA) * coeffs
        _ufld_smoothed[idx] = coeffs
        polys[idx] = coeffs
    # remove faixas que não apareceram neste frame
    for idx in list(_ufld_smoothed):
        if idx not in lanes_px:
            del _ufld_smoothed[idx]
    return polys

def _poly_x(coeffs, y):
    return coeffs[0] * y * y + coeffs[1] * y + coeffs[2]

def draw_lanes_overlay_rgb(view, lanes_px, alpha=0.35):
    """Overlay azul translúcido sobre a view RGB: preenche a pista do ego
    (entre faixas 1 e 2) e desenha as linhas das faixas detectadas."""
    overlay = view.copy()
    lane_color = (255, 130, 30)     # azul translúcido (BGR)
    if 1 in lanes_px and 2 in lanes_px and len(lanes_px[1]) > 4 and len(lanes_px[2]) > 4:
        l1 = np.array(sorted(lanes_px[1], key=lambda p: p[1]), np.int32)
        l2 = np.array(sorted(lanes_px[2], key=lambda p: p[1]), np.int32)
        n = min(len(l1), len(l2))
        if n > 4:
            hull = np.vstack([l1[:n], l2[:n][::-1]])
            cv2.fillPoly(overlay, [hull], lane_color)
    for idx, pts in lanes_px.items():
        if len(pts) < 2:
            continue
        ordered = np.array(sorted(pts, key=lambda p: p[1]), np.int32)
        cv2.polylines(overlay, [ordered], False, lane_color, 3, cv2.LINE_AA)
    return cv2.addWeighted(overlay, alpha, view, 1 - alpha, 0)

def draw_detected_lanes(img, offset_y, lane_polys):
    """Desenha as faixas detectadas no BEV: 1/2 tracejadas (pista do ego),
    0/3 contínuas (externas). Substitui as lanes animadas falsas."""
    dash_len = 3.0; gap_len = 5.0; cycle = dash_len + gap_len
    shift = offset_y % cycle
    y_pts = np.linspace(CAR_FRONT_TIP, MAX_DIST, 60)
    for idx, coeffs in lane_polys.items():
        pts = []
        for y in y_pts:
            p = project_bev(np.array([_poly_x(coeffs, y), y, 0.0]))
            if p: pts.append(p)
        if len(pts) < 2: continue
        if idx in (0, 3):
            cv2.polylines(img, [np.array(pts, np.int32)], False,
                          (120, 120, 120), 3, cv2.LINE_AA)
        else:
            dash = (idx == 1)
            side = 1 if idx == 2 else -1
            cy = -shift
            prev = None
            for y in y_pts:
                if cy >= CAR_FRONT_TIP:
                    p = project_bev(np.array([_poly_x(coeffs, y), y, 0.0]))
                    if p:
                        if prev and y - prev[1] <= dash_len:
                            cv2.line(img, prev[0], p, (120, 120, 120), 3, cv2.LINE_AA)
                        prev = (p, y)
                cy += cycle
    return img

def draw_fsd_path_gradient_real(img, lane_polys):
    """Path azul (gradiente) seguindo o centro da pista detectada."""
    if (1 not in lane_polys) or (2 not in lane_polys):
        return draw_fsd_path_gradient(img, 0.0)
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
        x_l1 = _poly_x(lane_polys[1], y1); x_r1 = _poly_x(lane_polys[2], y1)
        x_l2 = _poly_x(lane_polys[1], y2); x_r2 = _poly_x(lane_polys[2], y2)
        pl1 = project_bev(np.array([x_l1 + 0.15, y1, 0.01]))
        pr1 = project_bev(np.array([x_r1 - 0.15, y1, 0.01]))
        pl2 = project_bev(np.array([x_l2 + 0.15, y2, 0.01]))
        pr2 = project_bev(np.array([x_r2 - 0.15, y2, 0.01]))
        if pl1 and pr1 and pl2 and pr2:
            cv2.fillPoly(img, [np.array([pl1, pr1, pr2, pl2])], color)
    return img

# ═══════════════════════════════════════════════════════════
#  MAIN LOOP
# ═══════════════════════════════════════════════════════════

def main():
    if "front" not in VIDEOS:
        raise ValueError("VIDEOS must contain at least a 'front' camera entry.")

    model          = YOLO("yolov8n.pt")
    tracker        = BEVByteTracker()
    caps           = {k: cv2.VideoCapture(v) for k, v in VIDEOS.items() if v is not None}

    depth_active = False
    if DEPTH_MODEL:
        depth_active = _load_depth_model()

    ufld_active = False
    if UFLD_MODEL:
        ufld_active = _load_ufld_model()

    telemetry_data = []
    if TELEMETRY_CSV and os.path.exists(TELEMETRY_CSV):
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

    # ── Grid dinâmico de câmeras ───────────────────────────
    cam_order = [k for k, v in VIDEOS.items() if v is not None]
    n_cams = len(cam_order)
    n_cols = n_cams if n_cams <= 3 else 3
    n_rows = (n_cams + n_cols - 1) // n_cols
    grid_w = n_cols * CAM_W
    grid_h = n_rows * CAM_H
    out_w = grid_w + BEV_W
    out_h = max(grid_h, BEV_H)

    out = cv2.VideoWriter(OUT_VIDEO, cv2.VideoWriter_fourcc(*"mp4v"),
                          FPS, (out_w, out_h))
    print("Gerando UI FSD Clone Dinâmica com Telemetria e Movimento...")

    global_offset_y  = 0.0
    gps_trail        = []
    veh_gps_trails   = {}

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

        depth_map = None
        if depth_active and "front" in frames:
            depth_map = compute_depth_map(frames["front"])

        lane_polys = {}
        lanes_px = {}
        if ufld_active and "front" in frames:
            lanes_px = detect_lanes(frames["front"])
            lane_polys = _fit_lane_polys(lanes_px)

        for cam, frame in frames.items():
            view = frame.copy()
            yaw = CAM_YAW.get(cam, 0.0)
            cos_t = math.cos(yaw)
            sin_t = math.sin(yaw)
            for b in model(frame, conf=0.15, verbose=False)[0].boxes:
                cls = model.names[int(b.cls[0])]
                if cls not in DIMS: continue
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                # Filtra o capô do carro ego: aparece fixo na parte inferior
                # central do frame frontal (região do hood), muito largo e
                # baixo (w/h ~3-7) — o YOLO às vezes o classifica como veículo.
                # Carros reais próximos têm w/h < 1.5, então não são afetados.
                if cam == "front":
                    w_bb, h_bb = x2 - x1, y2 - y1
                    cx_b, cy_b = (x1 + x2) / 2, y2
                    hood = (cy_b > CAM_H * 0.84
                            and abs(cx_b - CAM_W / 2) < CAM_W * 0.22
                            and w_bb / max(h_bb, 1) > 2.2)
                    if hood:
                        continue
                dist = (H_REAL[cls] * FOCAL_DEPTH) / max(y2 - y1, 1)
                if depth_map is not None and cam == "front":
                    d = float(depth_map[min(y2, CAM_H - 1), int(min((x1 + x2) / 2, CAM_W - 1))])
                    if d > 0:
                        dist = d
                # Mesma focal da lane (FOCAL_DEPTH): alinha o lateral do
                # objeto com a geometria flat-ground usada nas faixas.
                lat  = (((x1+x2)/2 - CAM_W/2) * dist) / FOCAL_DEPTH
                ex = lat * cos_t - dist * sin_t
                ey = lat * sin_t + dist * cos_t
                ox, oy = CAM_OFFSETS.get(cam, (0.0, 0.0))
                global_dets.append({"ego_x": ex+ox, "ego_y": ey+oy,
                                     "cls": cls, "conf": float(b.conf[0])})
                cv2.rectangle(view, (x1, y1), (x2, y2), (200, 200, 200), 1)
            cam_views[cam] = view
            if ufld_active and cam == "front" and lanes_px:
                cam_views[cam] = draw_lanes_overlay_rgb(cam_views[cam], lanes_px)
            if DEPTH_OVERLAY and depth_map is not None and cam == "front":
                depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                cam_views[cam] = cv2.addWeighted(
                    cam_views[cam], 0.7,
                    cv2.applyColorMap(depth_norm, cv2.COLORMAP_TURBO), 0.3, 0)

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
        bev = render_tesla_ui(tracked, ego_state, global_offset_y, lane_polys)

        # ── Grid dinâmico ────────────────────────────────
        grid_rows = []
        for r in range(n_rows):
            row_cams = cam_order[r * n_cols : (r + 1) * n_cols]
            row_frames = []
            for c in row_cams:
                row_frames.append(cam_views.get(c, np.zeros((CAM_H, CAM_W, 3), np.uint8)))
            while len(row_frames) < n_cols:
                row_frames.append(np.zeros((CAM_H, CAM_W, 3), np.uint8))
            grid_rows.append(np.hstack(row_frames))
        grid = np.vstack(grid_rows)
        if grid.shape[0] < out_h:
            pad = np.full((out_h - grid.shape[0], grid.shape[1], 3), 0, np.uint8)
            grid = np.vstack([grid, pad])
        canvas = np.hstack([grid, bev])
        out.write(canvas)

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
