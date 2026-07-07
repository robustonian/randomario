"""Frame-based world-state estimation.

From the last few raw frames (no RAM peeking) this module estimates:

- camera scroll per frame (strip correlation between consecutive frames),
- moving objects via scroll-compensated frame differencing + connected
  components, tracked over time to get position / velocity / acceleration,
- which track is Mario (the one matching his known kinematics), giving a
  camera->world calibration for every other entity,
- gaps (pits) in the ground ahead, by scanning the bottom rows for columns
  that show only background color.

The planner consumes ``hints()`` to time gap jumps and enemy stomps; the
death analyzer consumes ``entities_near()`` to attribute a death to a
specific nearby entity.
"""
from collections import deque
from typing import Dict, List, Optional

import cv2
import numpy as np

SCREEN_W = 256
SCREEN_H = 240
HUD_ROWS = 32          # score/time area: ignore for motion
DIFF_THRESHOLD = 40    # per-pixel abs-gray difference considered "moving"
MIN_BLOB_AREA = 14
MAX_BLOB_SIZE = 80
MATCH_RADIUS = 22.0
SCROLL_MAX = 8         # NES scrolls at most a few px per frame
GAP_MIN_WIDTH = 12     # px; narrower "gaps" are sprite noise
GROUND_BAND = (208, 236)
SKY_BAND = (40, 96)


class Track:
    _next_id = 1

    def __init__(self, sx: float, sy: float, w: int, h: int):
        self.id = Track._next_id
        Track._next_id += 1
        self.sx = sx
        self.sy = sy
        self.w = w
        self.h = h
        self.vx = 0.0   # screen-space velocity (px/frame)
        self.vy = 0.0
        self.ax = 0.0
        self.ay = 0.0
        self.wx = None  # world x (needs camera calibration)
        self.wvx = 0.0  # world-space velocity
        self.age = 0
        self.missed = 0

    def advance(self, sx: float, sy: float, w: int, h: int):
        vx = sx - self.sx
        vy = sy - self.sy
        self.ax = 0.6 * self.ax + 0.4 * (vx - self.vx)
        self.ay = 0.6 * self.ay + 0.4 * (vy - self.vy)
        self.vx = 0.5 * self.vx + 0.5 * vx
        self.vy = 0.5 * self.vy + 0.5 * vy
        self.sx, self.sy, self.w, self.h = sx, sy, w, h
        self.age += 1
        self.missed = 0


class VisionTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.prev_gray: Optional[np.ndarray] = None
        self.camera_x: Optional[float] = None
        self.tracks: List[Track] = []
        self.mario_id: Optional[int] = None
        self.mario_vx = 0.0
        self.mario_ax = 0.0
        self._last_info_x = None
        self._gaps_world: List[int] = []
        self._edge_world: Optional[int] = None  # right edge of current platform
        self._scroll = 0
        self._frames = 0

    # ---------------------------------------------------------------- update

    def update(self, frame: np.ndarray, info: dict):
        gray = cv2.cvtColor(np.ascontiguousarray(frame), cv2.COLOR_RGB2GRAY).astype(np.int16)
        info_x = int(info.get('x_pos', 0))
        info_y = int(info.get('y_pos', 0))
        self._frames += 1

        if self.prev_gray is None:
            self.prev_gray = gray
            self._last_info_x = info_x
            return

        scroll = self._estimate_scroll(self.prev_gray, gray)
        self._scroll = scroll
        if self.camera_x is None:
            self.camera_x = float(info_x - 128)
        else:
            self.camera_x += scroll

        mask = self._motion_mask(self.prev_gray, gray, scroll)
        # Scene cut (area change / death fade): motion everywhere -> reset tracks
        if mask.mean() > 0.25:
            self.tracks = []
            self.mario_id = None
            self.prev_gray = gray
            self._last_info_x = info_x
            return

        blobs = self._find_blobs(mask)
        self._match_tracks(blobs)
        self._identify_mario(info_x, info_y, scroll)
        self._update_world_coords(scroll)
        if self._frames % 4 == 0:
            self._gaps_world = self._detect_gaps(frame)
            self._edge_world = self._detect_platform_edge(frame)

        self.prev_gray = gray
        self._last_info_x = info_x

    # ------------------------------------------------------------ internals

    def _estimate_scroll(self, prev: np.ndarray, cur: np.ndarray) -> int:
        """Best horizontal shift of a mid-level strip (background-dominated)."""
        band_p = prev[140:200]
        band_c = cur[140:200]
        best_s, best_err = 0, None
        for s in range(0, SCROLL_MAX + 1):
            if s:
                err = np.mean(np.abs(band_c[:, :-s] - band_p[:, s:]))
            else:
                err = np.mean(np.abs(band_c - band_p))
            if best_err is None or err < best_err:
                best_err, best_s = err, s
        return best_s

    def _motion_mask(self, prev: np.ndarray, cur: np.ndarray, scroll: int) -> np.ndarray:
        if scroll:
            diff = np.abs(cur[:, :-scroll] - prev[:, scroll:])
            diff = np.pad(diff, ((0, 0), (0, scroll)), constant_values=0)
        else:
            diff = np.abs(cur - prev)
        mask = (diff > DIFF_THRESHOLD).astype(np.uint8)
        mask[:HUD_ROWS] = 0
        return mask

    def _find_blobs(self, mask: np.ndarray) -> List[tuple]:
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        blobs = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area < MIN_BLOB_AREA or w > MAX_BLOB_SIZE or h > MAX_BLOB_SIZE:
                continue
            cx, cy = centroids[i]
            # Scroll-compensation seams create phantom blobs at the borders.
            if cx < 8 or cx > SCREEN_W - 8:
                continue
            blobs.append((float(cx), float(cy), int(w), int(h)))
        return blobs

    def _match_tracks(self, blobs: List[tuple]):
        unused = list(blobs)
        for tr in self.tracks:
            px, py = tr.sx + tr.vx, tr.sy + tr.vy
            best, best_d = None, MATCH_RADIUS
            for b in unused:
                d = ((b[0] - px) ** 2 + (b[1] - py) ** 2) ** 0.5
                if d < best_d:
                    best, best_d = b, d
            if best is not None:
                unused.remove(best)
                tr.advance(*best)
            else:
                tr.missed += 1
                tr.sx += tr.vx  # coast
                tr.sy += tr.vy
        self.tracks = [t for t in self.tracks if t.missed <= 6]
        for b in unused:
            self.tracks.append(Track(*b))

    def _identify_mario(self, info_x: int, info_y: int, scroll: int):
        """Mario's screen y is derivable from info; his screen dx must equal
        (world dx - scroll). Pick the track that fits both best."""
        mario_sy = max(0, 255 - info_y)
        dx_world = info_x - (self._last_info_x if self._last_info_x is not None else info_x)
        expect_dx = dx_world - scroll
        best, best_cost = None, 26.0
        for tr in self.tracks:
            if tr.age < 2:
                continue
            cost = abs((tr.sy + tr.h / 2) - mario_sy) * 0.5 + abs(tr.vx - expect_dx) * 3.0
            if tr.id == self.mario_id:
                cost -= 6.0  # stickiness
            if cost < best_cost:
                best, best_cost = tr, cost
        if best is not None:
            self.mario_id = best.id
            self.camera_x = float(info_x - best.sx)  # calibrate camera->world
            self.mario_ax = 0.7 * self.mario_ax + 0.3 * (dx_world - self.mario_vx)
        self.mario_vx = 0.6 * self.mario_vx + 0.4 * dx_world

    def _update_world_coords(self, scroll: int):
        if self.camera_x is None:
            return
        for tr in self.tracks:
            wx = self.camera_x + tr.sx
            if tr.wx is not None:
                wvx = wx - tr.wx
                if abs(wvx) < 8:
                    tr.wvx = 0.6 * tr.wvx + 0.4 * wvx
            tr.wx = wx

    def _detect_gaps(self, frame: np.ndarray) -> List[int]:
        """World-x of gap (pit) left edges ahead of the camera."""
        sky = frame[SKY_BAND[0]:SKY_BAND[1]]
        # Dominant background color from the sky band (works for every theme)
        pixels = sky.reshape(-1, 3)
        sample = pixels[:: max(1, len(pixels) // 512)]
        colors, counts = np.unique(sample, axis=0, return_counts=True)
        bg = colors[counts.argmax()].astype(np.int16)

        band = frame[GROUND_BAND[0]:GROUND_BAND[1]].astype(np.int16)
        is_bg = (np.abs(band - bg).sum(axis=2) < 40).all(axis=0)

        gaps = []
        if self.camera_x is None:
            return gaps
        j = 0
        while j < SCREEN_W:
            if is_bg[j]:
                k = j
                while k < SCREEN_W and is_bg[k]:
                    k += 1
                if k - j >= GAP_MIN_WIDTH:
                    gaps.append(int(self.camera_x + j))
                j = k
            else:
                j += 1
        return gaps

    def _detect_platform_edge(self, frame: np.ndarray) -> Optional[int]:
        """World-x where the platform under Mario's feet ends (background
        starts), scanned at foot level — works on elevated platforms where
        the bottom-band gap scan is meaningless."""
        mario = self.mario_track()
        if mario is None or self.camera_x is None:
            return None
        foot = int(mario.sy + mario.h) + 2
        if foot >= SCREEN_H - 6 or foot < HUD_ROWS:
            return None
        sky = frame[SKY_BAND[0]:SKY_BAND[1]]
        pixels = sky.reshape(-1, 3)
        sample = pixels[:: max(1, len(pixels) // 256)]
        colors, counts = np.unique(sample, axis=0, return_counts=True)
        bg = colors[counts.argmax()].astype(np.int16)

        band = frame[foot:foot + 4].astype(np.int16)
        is_bg = (np.abs(band - bg).sum(axis=2) < 40).all(axis=0)
        start = min(SCREEN_W - 1, max(0, int(mario.sx)))
        run = 0
        for j in range(start, SCREEN_W):
            if is_bg[j]:
                run += 1
                if run >= 6:
                    return int(self.camera_x + j - run + 1)
            else:
                run = 0
        return None

    # ------------------------------------------------------------- consumers

    def mario_track(self) -> Optional[Track]:
        for tr in self.tracks:
            if tr.id == self.mario_id:
                return tr
        return None

    def entities(self) -> List[Dict]:
        out = []
        for tr in self.tracks:
            if tr.id == self.mario_id or tr.age < 3 or tr.wx is None:
                continue
            out.append({'x': int(tr.wx), 'y': int(tr.sy), 'vx': round(tr.wvx, 2),
                        'ax': round(tr.ax, 2), 'w': tr.w, 'h': tr.h, 'id': tr.id})
        return out

    def entities_near(self, world_x: int, radius: int = 40) -> List[Dict]:
        return [e for e in self.entities() if abs(e['x'] - world_x) <= radius]

    def hints(self) -> dict:
        mario = self.mario_track()
        mario_x = None
        if mario is not None and mario.wx is not None:
            mario_x = int(mario.wx)
        gaps_ahead = [g for g in self._gaps_world
                      if mario_x is None or g > mario_x - 8]
        enemies = sorted(self.entities(), key=lambda e: e['x'])
        if mario_x is not None:
            enemies = [e for e in enemies if e['x'] > mario_x - 16]
        edge = self._edge_world
        if edge is not None and mario_x is not None and edge < mario_x - 8:
            edge = None
        return {
            'mario_vx': max(0.5, self.mario_vx),
            'mario_ax': self.mario_ax,
            'mario_x': mario_x,
            'gaps': gaps_ahead,
            'enemies': enemies,
            'edge': edge,
            'scroll': self._scroll,
        }
