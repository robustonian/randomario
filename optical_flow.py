import argparse
import os
import pickle
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import numpy as np
import pygame
import cv2  # Added for optical flow
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT, SIMPLE_MOVEMENT, RIGHT_ONLY
import urllib.request
import urllib.error

# Action sets from gym_super_mario_bros
ACTION_SETS = {
    'right_only': RIGHT_ONLY,
    'simple': SIMPLE_MOVEMENT,
    'complex': COMPLEX_MOVEMENT,
}

def get_action_indices(actions: List[List[str]]) -> dict:
    """Get action indices for the given action set."""
    indices = {}
    for i, action in enumerate(actions):
        if action == ['NOOP']:
            indices['NOOP'] = i
        elif action == ['right']:
            indices['RIGHT'] = i
        elif action == ['right', 'A']:
            indices['RIGHT_JUMP'] = i
        elif action == ['right', 'B']:
            indices['RIGHT_DASH'] = i
        elif action == ['right', 'A', 'B']:
            indices['RIGHT_DASH_JUMP'] = i
        elif action == ['A']:
            indices['JUMP'] = i
        elif action == ['left']:
            indices['LEFT'] = i
        elif action == ['left', 'A']:
            indices['LEFT_JUMP'] = i
        elif action == ['left', 'B']:
            indices['LEFT_DASH'] = i
        elif action == ['left', 'A', 'B']:
            indices['LEFT_DASH_JUMP'] = i
        elif action == ['down']:
            indices['DOWN'] = i
        elif action == ['down', 'A']:
            indices['DOWN_JUMP'] = i
        elif action == ['up']:
            indices['UP'] = i
    return indices

# Environment utilities
def make_env(stage: str, actions: List[List[str]]):
    env_name = f"SuperMarioBros-{stage}-v0"
    try:
        env = gym_super_mario_bros.make(env_name, render_mode='rgb_array', apply_api_compatibility=True)
    except TypeError:
        env = gym_super_mario_bros.make(env_name, render_mode='rgb_array')
    env = JoypadSpace(env, actions)
    return env

def reset_env(env):
    try:
        obs, info = env.reset()
    except ValueError:
        obs = env.reset()
        info = {}
    return obs, info

def step_env(env, action_idx: int):
    try:
        obs, reward, terminated, truncated, info = env.step(action_idx)
        done = terminated or truncated
    except ValueError:
        obs, reward, done, info = env.step(action_idx)
    return obs, reward, done, info

# Cell (discrete state) definition and archive
@dataclass
class Cell:
    cell_id: Tuple[int, int, str]  # (x_bin, y_bin, status)
    path: List[int] = field(default_factory=list)  # Action sequence from start to reach this cell
    max_x: int = 0  # Maximum x reached when reaching this cell
    visits: int = 0  # Number of times used as starting cell
    created_at: float = field(default_factory=time.time)

# Optical Flow Tracker Class
class FlowBasedTracker:
    """
    Tracks camera movement and object motion using optical flow analysis.
    
    Features:
    - Camera scroll estimation from background flow
    - Mario detection and tracking via residual flow
    - Other moving object detection
    - Velocity and acceleration estimation
    """
    
    def __init__(self, fps=60, hud_height=40):
        self.prev_gray = None
        self.fps = fps
        self.dt = 1.0 / max(1, fps)
        self.hud_height = hud_height
        
        # Camera tracking
        self.cam_vx_ema = 0.0
        self.cam_vy_ema = 0.0
        self.ema_alpha = 0.3  # Exponential moving average for stability
        self.cam_x = 0.0
        self.cam_y = 0.0
        
        # Mario tracking
        self.mario_bbox = None
        self.mario_vx = 0.0
        self.mario_vy = 0.0
        self.mario_ax = 0.0
        self.mario_ay = 0.0
        self.prev_mario_v = (0.0, 0.0)
        
        # Other objects
        self.objects = []  # List of {bbox, v_screen, v_world}
        
        # Flow parameters (enhanced for better moving object detection)
        self.flow_params = {
            'pyr_scale': 0.5,
            'levels': 4,           # Increased for better slow motion capture
            'winsize': 21,         # Larger window for small/slow movements
            'iterations': 5,       # More iterations for accuracy
            'poly_n': 7,          # Higher polynomial degree
            'poly_sigma': 1.5,    # Larger gaussian window
            'flags': cv2.OPTFLOW_FARNEBACK_GAUSSIAN
        }
        
        # Motion detection thresholds (improved sensitivity)
        self.motion_threshold_percentile = 85  # More sensitive to slow motion
        self.motion_threshold_multiplier = 0.7
        self.min_component_area = 12           # Allow smaller objects
        self.min_component_size = 3
        self.max_component_size = 80
        
        # Camera tracking with improved stability
        self.ema_alpha = 0.15  # Reduced for more stability
        
        # Mario detection parameters
        self.mario_sprite_size = (18, 28)  # Approximate Mario sprite size
        self.mario_initial_x_bias = 100  # Prefer left side of screen initially
        
    def _estimate_camera_flow(self, flow, mario_bbox=None):
        """Estimate camera movement from background flow, excluding HUD area and Mario vicinity."""
        H, W, _ = flow.shape
        
        # Create mask to exclude HUD (top portion of screen)
        mask = np.zeros((H, W), dtype=np.uint8)
        mask[self.hud_height:, :] = 1
        
        # Exclude Mario vicinity to avoid contamination
        if mario_bbox is not None:
            mx, my, mw, mh = mario_bbox
            margin = 12
            x0 = max(0, mx - margin)
            x1 = min(W, mx + mw + margin)
            y0 = max(0, my - margin)
            y1 = min(H, my + mh + margin)
            mask[y0:y1, x0:x1] = 0
        
        # Extract flow vectors from valid regions
        fx = flow[..., 0][mask > 0]
        fy = flow[..., 1][mask > 0]
        
        if fx.size < 100:  # Not enough flow vectors
            return 0.0, 0.0
            
        # Use median for robust estimation (resistant to outliers)
        vx = np.median(fx)
        vy = np.median(fy)
        
        # Clamp vertical velocity for horizontal scrolling games
        vy = np.clip(vy, -0.2, 0.2)
        
        # Apply exponential moving average for temporal stability
        self.cam_vx_ema = (1 - self.ema_alpha) * self.cam_vx_ema + self.ema_alpha * vx
        self.cam_vy_ema = (1 - self.ema_alpha) * self.cam_vy_ema + self.ema_alpha * vy
        
        return self.cam_vx_ema, self.cam_vy_ema
    
    def _segment_moving_objects(self, flow, cam_vx, cam_vy):
        """Segment moving objects by removing camera motion from flow."""
        # Calculate residual flow (flow - camera motion)
        residual = flow.copy()
        residual[..., 0] -= cam_vx
        residual[..., 1] -= cam_vy
        
        # Calculate magnitude of residual flow
        magnitude = np.linalg.norm(residual, axis=2)
        
        # Use MAD (Median Absolute Deviation) for robust adaptive threshold
        med = np.median(magnitude)
        mad = 1.4826 * np.median(np.abs(magnitude - med))  # Scale factor for normal distribution
        threshold = max(0.4, med + 2.5 * mad)  # Lower minimum threshold, robust outlier detection
        
        # Create binary mask for moving regions
        mask = (magnitude > threshold).astype(np.uint8) * 255
        
        # Exclude HUD area
        mask[:self.hud_height, :] = 0
        
        # Morphological operations to clean up the mask (reduced opening for small objects)
        kernel = np.ones((3, 3), np.uint8)
        # Skip MORPH_OPEN to preserve small moving objects
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=2)
        
        return mask, residual
    
    def _find_connected_components(self, mask):
        """Find connected components in the motion mask."""
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        components = []
        for i in range(1, num_labels):  # Skip background (label 0)
            x, y, w, h, area = stats[i]
            
            # Filter components by size
            if (area < self.min_component_area or 
                w < self.min_component_size or h < self.min_component_size or
                w > self.max_component_size or h > self.max_component_size):
                continue
                
            components.append((x, y, w, h, area))
        
        return components
    
    def _calculate_mean_flow_in_bbox(self, flow, bbox):
        """Calculate mean flow within a bounding box."""
        x, y, w, h = bbox
        roi = flow[y:y+h, x:x+w]
        
        if roi.size == 0:
            return 0.0, 0.0
            
        vx = np.mean(roi[..., 0])
        vy = np.mean(roi[..., 1])
        
        return vx, vy
    
    def _identify_mario(self, components, prev_mario_bbox):
        """Identify Mario from detected components."""
        if not components:
            return None
            
        if prev_mario_bbox is None:
            # First frame: use heuristics based on size and position
            scored_components = []
            for (x, y, w, h, area) in components:
                # Size score (prefer Mario-like sprite size)
                size_score = -abs(h - self.mario_sprite_size[1]) - abs(w - self.mario_sprite_size[0])
                
                # Position score (prefer left side of screen)
                pos_score = -(x - self.mario_initial_x_bias) ** 2 * 1e-3
                
                total_score = size_score + pos_score
                scored_components.append((total_score, (x, y, w, h)))
            
            # Return highest scoring component
            scored_components.sort(reverse=True, key=lambda t: t[0])
            return scored_components[0][1]
        
        else:
            # Subsequent frames: use nearest neighbor to previous Mario position
            prev_x, prev_y, prev_w, prev_h = prev_mario_bbox
            prev_center_x = prev_x + prev_w / 2
            prev_center_y = prev_y + prev_h / 2
            
            best_component = None
            min_distance_sq = float('inf')
            
            for (x, y, w, h, area) in components:
                center_x = x + w / 2
                center_y = y + h / 2
                
                distance_sq = (center_x - prev_center_x) ** 2 + (center_y - prev_center_y) ** 2
                
                if distance_sq < min_distance_sq:
                    min_distance_sq = distance_sq
                    best_component = (x, y, w, h)
            
            return best_component
    
    def _static_wall_ahead(self, gray, mario_bbox):
        """Detect static walls/obstacles ahead of Mario using edge density analysis."""
        if mario_bbox is None:
            return False
        
        H, W = gray.shape
        x, y, w, h = mario_bbox
        
        # Define region ahead of Mario at foot level
        rx0 = min(W-1, x + w + 8)      # Start 8px ahead of Mario
        rx1 = min(W, x + w + 56)       # Extend 56px forward
        ry0 = max(0, y + h - 10)       # 10px above Mario's bottom
        ry1 = min(H, y + h + 4)        # 4px below Mario's bottom
        
        roi = gray[ry0:ry1, rx0:rx1]
        if roi.size == 0:
            return False
            
        # Apply Gaussian blur to reduce noise, then edge detection
        roi_blur = cv2.GaussianBlur(roi, (3, 3), 0)
        edges = cv2.Canny(roi_blur, 40, 120)
        
        # Check for strong vertical edges (walls, pipes)
        col_sums = edges.sum(axis=0) / 255.0
        max_vertical_edge = col_sums.max() if col_sums.size > 0 else 0
        
        return max_vertical_edge >= 6  # Strong vertical edge indicates wall
    
    def _gap_ahead(self, gray, mario_bbox):
        """Detect gaps/holes ahead of Mario using texture/edge density analysis."""
        if mario_bbox is None:
            return False
            
        H, W = gray.shape
        x, y, w, h = mario_bbox
        
        # Define region below Mario's feet, looking ahead
        rx0 = min(W-1, x + w + 8)      # Start 8px ahead of Mario
        rx1 = min(W, x + w + 56)       # Extend 56px forward
        by0 = min(H-1, y + h + 2)      # Just below Mario's feet
        by1 = min(H, y + h + 14)       # Look down 14px
        
        roi = gray[by0:by1, rx0:rx1]
        if roi.size == 0:
            return False
            
        # Apply blur and edge detection
        roi_blur = cv2.GaussianBlur(roi, (3, 3), 0)
        edges = cv2.Canny(roi_blur, 30, 100)
        
        # Calculate edge density - low density suggests empty space (gap)
        edge_density = edges.sum() / 255.0 / max(1, roi.size)
        
        return edge_density < 0.02  # Very low edge density indicates gap
    
    def update(self, frame_rgb):
        """
        Process a new frame and return tracking results.
        
        Args:
            frame_rgb: RGB frame from the game (H, W, 3)
            
        Returns:
            Dictionary containing:
            - camera_v: Camera velocity (vx, vy)
            - camera_xy: Cumulative camera position (x, y)
            - mario: Mario tracking info or None
            - objects: List of other detected objects
        """
        # Convert to grayscale for optical flow
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        
        # Initialize result structure
        result = {
            "camera_v": (0.0, 0.0),
            "camera_xy": (self.cam_x, self.cam_y),
            "mario": None,
            "objects": [],
        }
        
        # Need previous frame for optical flow
        if self.prev_gray is None:
            self.prev_gray = gray
            return result
        
        # Calculate optical flow using Farneback method
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None, **self.flow_params
        )
        
        # Estimate camera movement (pass previous Mario bbox to avoid contamination)
        cam_vx, cam_vy = self._estimate_camera_flow(flow, self.mario_bbox)
        
        # Update cumulative camera position
        self.cam_x += cam_vx
        self.cam_y += cam_vy
        
        # Segment moving objects
        motion_mask, residual_flow = self._segment_moving_objects(flow, cam_vx, cam_vy)
        
        # Find connected components
        components = self._find_connected_components(motion_mask)
        
        # Identify Mario
        mario_bbox = self._identify_mario(components, self.mario_bbox)
        
        if mario_bbox is not None:
            # Calculate Mario's screen velocity
            mario_vx_screen, mario_vy_screen = self._calculate_mean_flow_in_bbox(flow, mario_bbox)
            
            # Calculate world velocity (relative to stationary background)
            mario_vx_world = mario_vx_screen + cam_vx
            mario_vy_world = mario_vy_screen + cam_vy
            
            # Calculate acceleration
            mario_ax = (mario_vx_world - self.prev_mario_v[0]) / self.dt
            mario_ay = (mario_vy_world - self.prev_mario_v[1]) / self.dt
            
            # Update tracking state
            self.prev_mario_v = (mario_vx_world, mario_vy_world)
            self.mario_bbox = mario_bbox
            
            # Store Mario info
            result["mario"] = {
                "bbox": mario_bbox,
                "v_screen": (mario_vx_screen, mario_vy_screen),
                "v_world": (mario_vx_world, mario_vy_world),
                "a": (mario_ax, mario_ay),
            }
        
        # Process other objects (excluding Mario's region)
        other_objects = []
        for (x, y, w, h, area) in components:
            # Skip if this component overlaps with Mario
            if mario_bbox is not None:
                mx, my, mw, mh = mario_bbox
                if (x < mx + mw and x + w > mx and y < my + mh and y + h > my):
                    continue
            
            # Calculate object velocities
            obj_vx_screen, obj_vy_screen = self._calculate_mean_flow_in_bbox(flow, (x, y, w, h))
            obj_vx_world = obj_vx_screen + cam_vx
            obj_vy_world = obj_vy_screen + cam_vy
            
            other_objects.append({
                "bbox": (x, y, w, h),
                "v_screen": (obj_vx_screen, obj_vy_screen),
                "v_world": (obj_vx_world, obj_vy_world)
            })
        
        # Limit number of tracked objects (keep largest by area)
        other_objects.sort(key=lambda obj: obj["bbox"][2] * obj["bbox"][3], reverse=True)
        result["objects"] = other_objects[:8]
        
        # Update result with camera info
        result["camera_v"] = (cam_vx, cam_vy)
        result["camera_xy"] = (self.cam_x, self.cam_y)
        
        # Add static obstacle detection
        result["static_wall_ahead"] = self._static_wall_ahead(gray, mario_bbox)
        result["gap_ahead"] = self._gap_ahead(gray, mario_bbox)
        
        # Store current frame for next iteration
        self.prev_gray = gray
        
        return result

class FlowHeuristicPolicy:
    """
    Optical Flowベースの簡易ヒューリスティック戦略。
    既定は右ダッシュ、障害物/減速/落下兆候でジャンプを開始し、保持時間を調整します。
    アクションセットに応じて可能な最良の組み合わせを選びます。
    """
    def __init__(self, actions, action_indices, target_fps=60):
        self.actions = actions
        self.a = action_indices
        self.dt = 1.0 / max(1, target_fps)
        # 内部状態
        self.jump_hold = 0
        self.ground_frames = 0
        self.last_bottom_y = None
        self.stuck_frames = 0

        # 閾値
        self.vx_slow = 0.5       # v_world_x がこの値未満だと減速/詰まり傾向
        self.fall_vy = 0.6       # v_world_y がこの値より大きい（下向き）と落下中
        self.obstacle_dx = (10, 48)  # マリオの右端から前方 dx 範囲を障害物候補
        self.obstacle_dy_tol = 10    # 足元高さの重なり許容
        self.min_jump_hold = 6
        self.max_jump_hold = 14
        self.stuck_limit = 120    # 連続フレーム数（~2.0秒）で詰まり判定

    def reset(self):
        self.jump_hold = 0
        self.ground_frames = 0
        self.last_bottom_y = None
        self.stuck_frames = 0

    def _best(self, labels):
        # 与えた候補ラベル列から、利用可能な最初のものを返す
        for lab in labels:
            if lab in self.a:
                return self.a[lab]
        # 最終フォールバック
        return self.a.get('RIGHT', self.a.get('NOOP', 0))

    def _right(self):
        return self._best(['RIGHT_DASH', 'RIGHT'])

    def _right_jump(self):
        return self._best(['RIGHT_DASH_JUMP', 'RIGHT_JUMP', 'JUMP', 'RIGHT'])

    def _left(self):
        return self._best(['LEFT', 'RIGHT'])  # 無ければ右

    def _left_jump(self):
        return self._best(['LEFT_JUMP', 'JUMP', 'LEFT', 'RIGHT'])

    def _noop(self):
        return self._best(['NOOP'])

    def _on_ground_update(self, mario):
        # 簡易な接地判定：ボトムYが安定かつ垂直速度が小さい
        if not mario:
            self.ground_frames = 0
            self.last_bottom_y = None
            return False
        x, y, w, h = mario['bbox']
        bottom_y = y + h
        vy = mario['v_world'][1]
        if self.last_bottom_y is not None and abs(bottom_y - self.last_bottom_y) <= 1 and abs(vy) < 0.3:
            self.ground_frames += 1
        else:
            self.ground_frames = 0
        self.last_bottom_y = bottom_y
        return self.ground_frames >= 2

    def _obstacle_ahead(self, mario, objects, flow_estimates):
        """Detect obstacles ahead using both moving objects and static detection."""
        moving_obstacle = False
        
        if mario:
            # Check for moving objects near Mario
            mx, my, mw, mh = mario['bbox']
            right_edge = mx + mw
            for obj in objects:
                ox, oy, ow, oh = obj['bbox']
                ocx = ox + ow / 2
                dx = ocx - right_edge
                if dx < self.obstacle_dx[0] or dx > self.obstacle_dx[1]:
                    continue
                # 足元付近に重なるもの（地上の敵・ファイアバー等の一部）
                if not (oy < my + mh and oy + oh > my + mh - self.obstacle_dy_tol):
                    continue
                moving_obstacle = True
                break
        
        # Check for static obstacles (walls, pipes, gaps)
        static_wall = bool(flow_estimates.get("static_wall_ahead", False))
        gap = bool(flow_estimates.get("gap_ahead", False))
        
        return moving_obstacle or static_wall or gap

    def decide(self, flow_estimates):
        # Marioが見つからない場合は、右ダッシュ前進
        mario = flow_estimates.get('mario')
        objects = flow_estimates.get('objects', [])
        # 既にジャンプ保持中なら継続
        if self.jump_hold > 0:
            self.jump_hold -= 1
            return self._right_jump()

        if not mario:
            return self._right()

        vx, vy = mario['v_world']
        on_ground = self._on_ground_update(mario)

        # 詰まり検出（ほぼ前進していない）
        if abs(vx) < 0.2 and on_ground:
            self.stuck_frames += 1
        else:
            self.stuck_frames = 0

        # 落下中ならジャンプボタンを追加保持（滞空延長）
        if not on_ground and vy > self.fall_vy:
            self.jump_hold = max(self.jump_hold, 4)
            return self._right_jump()

        # 障害物や減速兆候でジャンプ開始
        obstacle = self._obstacle_ahead(mario, objects, flow_estimates)
        if on_ground and (obstacle or vx < self.vx_slow or self.stuck_frames > self.stuck_limit):
            # 速度に応じて保持時間を調整（速いほど長めに）
            hold = int(self.min_jump_hold + (self.max_jump_hold - self.min_jump_hold) * max(0.0, min(1.0, vx / 2.5)))
            self.jump_hold = max(self.jump_hold, hold)
            self.stuck_frames = 0
            return self._right_jump()

        # たまに段差で引っかかる場合の微調整（空中で右ジャンプ継続、接地で稀に左微調整）
        if on_ground and vx < 0.25 and not obstacle:
            # ほんの少し左に戻って体勢立て直し（アクションが無い場合は単に右）
            return self._left() if 'LEFT' in self.a else self._right()

        # 既定は右ダッシュで前進
        return self._right()

# UI constants
SCREEN_WIDTH = 1024
SCREEN_HEIGHT = 640

GAME_SCREEN_BASE_WIDTH = 256
GAME_SCREEN_BASE_HEIGHT = 240
GAME_SCALE_FACTOR = 2
GAME_SCREEN_WIDTH = GAME_SCREEN_BASE_WIDTH * GAME_SCALE_FACTOR
GAME_SCREEN_HEIGHT = GAME_SCREEN_BASE_HEIGHT * GAME_SCALE_FACTOR
GAME_SCREEN_X = 30
GAME_SCREEN_Y = 50
GAME_SCREEN_RECT = pygame.Rect(GAME_SCREEN_X, GAME_SCREEN_Y, GAME_SCREEN_WIDTH, GAME_SCREEN_HEIGHT)

BACKGROUND_COLOR = (30, 30, 30)
HIGHLIGHT_COLOR = pygame.Color("yellow")
TEXT_COLOR = (255, 255, 255)
ACCENT_COLOR = (255, 200, 0)
MARIO_TRACK_COLOR = (0, 255, 0)  # Green for Mario
OBJECT_TRACK_COLOR = (255, 0, 0)  # Red for other objects
FLOW_INFO_COLOR = (0, 200, 255)  # Cyan for flow info

FPS = 60

# Controller image (optional)
CONTROLLER_IMAGE_PATH = 'fig/famicon01_01.png'
CONTROLLER_DISPLAY_WIDTH = 360
CONTROLLER_X = GAME_SCREEN_X + GAME_SCREEN_WIDTH + 30
CONTROLLER_Y = GAME_SCREEN_Y
CONTROLLER_RECT = None  # Set if image exists

# Button areas (original image coordinate system: 800x800 as reference)
ORIGINAL_IMG_WIDTH_FOR_COORDS = 800
ORIGINAL_IMG_HEIGHT_FOR_COORDS = 800
DPAD_UP_RECT_ORIG = pygame.Rect(208 - 46, 340 + 22, 30, 40)
DPAD_DOWN_RECT_ORIG = pygame.Rect(208 - 46, 436 + 22, 30, 40)
DPAD_LEFT_RECT_ORIG = pygame.Rect(155 - 46, 393 + 22, 40, 30)
DPAD_RIGHT_RECT_ORIG = pygame.Rect(250 - 46, 393 + 22, 40, 30)
BUTTON_B_CENTER_ORIG = (549, 461)
BUTTON_B_RADIUS_ORIG = 28
BUTTON_A_CENTER_ORIG = (644, 461)
BUTTON_A_RADIUS_ORIG = 28

BUTTON_GEOMETRIES_ORIG = {
    'dpad_up': {'type': 'rect', 'geom': DPAD_UP_RECT_ORIG},
    'dpad_down': {'type': 'rect', 'geom': DPAD_DOWN_RECT_ORIG},
    'dpad_left': {'type': 'rect', 'geom': DPAD_LEFT_RECT_ORIG},
    'dpad_right': {'type': 'rect', 'geom': DPAD_RIGHT_RECT_ORIG},
    'button_a': {'type': 'circle', 'geom': (BUTTON_A_CENTER_ORIG, BUTTON_A_RADIUS_ORIG)},
    'button_b': {'type': 'circle', 'geom': (BUTTON_B_CENTER_ORIG, BUTTON_B_RADIUS_ORIG)},
}

COMMAND_TO_KEY = {
    'right': 'dpad_right', 'left': 'dpad_left', 'up': 'dpad_up', 'down': 'dpad_down',
    'A': 'button_a', 'B': 'button_b', 'NOOP': 'noop'
}

def convert_frame_to_surface(frame_np: np.ndarray) -> pygame.Surface:
    # gym obs is (H, W, 3). pygame.surfarray.make_surface expects (W, H, 3)
    frame_swapped = np.transpose(frame_np, (1, 0, 2))
    return pygame.surfarray.make_surface(frame_swapped)

def try_load_controller_image() -> Tuple[Optional[pygame.Surface], Optional[pygame.Rect], Dict]:
    # Download image if it doesn't exist
    if not os.path.exists(CONTROLLER_IMAGE_PATH):
        try:
            os.makedirs(os.path.dirname(CONTROLLER_IMAGE_PATH), exist_ok=True)
            url = 'https://stockmaterial.net/wp/wp-content/uploads/img/famicon01_01.png'
            urllib.request.urlretrieve(url, CONTROLLER_IMAGE_PATH)
        except Exception:
            return None, None, {}

    try:
        base = pygame.image.load(CONTROLLER_IMAGE_PATH).convert_alpha()
    except Exception:
        return None, None, {}

    original_w = base.get_width()
    original_h = base.get_height()
    if original_w == 0:
        return None, None, {}

    aspect = original_h / original_w
    display_h = int(CONTROLLER_DISPLAY_WIDTH * aspect)
    scaled = pygame.transform.smoothscale(base, (CONTROLLER_DISPLAY_WIDTH, display_h))
    rect = pygame.Rect(CONTROLLER_X, CONTROLLER_Y, CONTROLLER_DISPLAY_WIDTH, display_h)

    # Scale button coordinates
    scale_x = CONTROLLER_DISPLAY_WIDTH / ORIGINAL_IMG_WIDTH_FOR_COORDS
    scale_y = display_h / ORIGINAL_IMG_HEIGHT_FOR_COORDS
    scaled_geoms = {}

    for key, data in BUTTON_GEOMETRIES_ORIG.items():
        if data['type'] == 'rect':
            r = data['geom']
            scaled_rect = pygame.Rect(
                int(r.left * scale_x), int(r.top * scale_y),
                max(1, int(r.width * scale_x)), max(1, int(r.height * scale_y))
            )
            scaled_geoms[key] = {'type': 'rect', 'geom': scaled_rect}
        else:
            (cx, cy), radius = data['geom']
            sx = cx * scale_x
            sy = cy * scale_y
            sr = max(1, int(radius * min(scale_x, scale_y)))
            scaled_geoms[key] = {'type': 'circle', 'geom': ((sx, sy), sr)}

    return scaled, rect, scaled_geoms

def draw_controller(surface, base_img, base_rect, scaled_geoms, pressed_keys: List[str]):
    if base_img is None or base_rect is None:
        return
    surface.blit(base_img, base_rect.topleft)
    for key in pressed_keys:
        if key not in scaled_geoms:
            continue
        info = scaled_geoms[key]
        if info['type'] == 'rect':
            highlight = info['geom'].move(base_rect.left, base_rect.top)
            pygame.draw.rect(surface, HIGHLIGHT_COLOR, highlight, width=3, border_radius=4)
        else:
            (cx, cy), r = info['geom']
            pygame.draw.circle(surface, HIGHLIGHT_COLOR, (int(base_rect.left + cx), int(base_rect.top + cy)), r, width=3)

def action_to_pressed_keys(action_idx: int, actions: List[List[str]]) -> List[str]:
    keys = set()
    for token in actions[action_idx]:
        if token in COMMAND_TO_KEY:
            mapped = COMMAND_TO_KEY[token]
            if mapped != 'noop':
                keys.add(mapped)
    return sorted(list(keys))

def scale_bbox_to_screen(bbox, game_screen_x, game_screen_y, scale_factor):
    """Scale a bounding box from game coordinates to screen coordinates."""
    x, y, w, h = bbox
    screen_x = int(game_screen_x + x * scale_factor)
    screen_y = int(game_screen_y + y * scale_factor)
    screen_w = int(w * scale_factor)
    screen_h = int(h * scale_factor)
    return pygame.Rect(screen_x, screen_y, screen_w, screen_h)

# Go-Explore style agent with UI integration and optical flow tracking
class GoExploreUIRunner:
    def __init__(
        self,
        stage: str,
        archive_path: str,
        actions: List[List[str]],
        seed: Optional[int] = None,
        cell_size_x: int = 50,
        cell_size_y: int = 40,
        max_steps_per_episode: int = 6000,
        explore_steps_after_return: int = 1200,
        stuck_frame_window: int = 60,
        save_every_episodes: int = 10,
        down_press_prob: float = 0.02,
        left_adjust_prob: float = 0.05,
        jump_start_prob: float = 0.12,
        jump_min_hold: int = 6,
        jump_max_hold: int = 14,
        target_fps: int = 60,
    ):
        self.stage = stage
        self.archive_path = archive_path
        self.actions = actions
        self.action_indices = get_action_indices(actions)
        self.cell_size_x = cell_size_x
        self.cell_size_y = cell_size_y
        self.max_steps_per_episode = max_steps_per_episode
        self.explore_steps_after_return = explore_steps_after_return
        self.stuck_frame_window = stuck_frame_window
        self.save_every_episodes = save_every_episodes

        self.down_press_prob = down_press_prob
        self.left_adjust_prob = left_adjust_prob
        self.jump_start_prob = jump_start_prob
        self.jump_min_hold = jump_min_hold
        self.jump_max_hold = jump_max_hold

        self.target_fps = target_fps

        # Environment
        self.env = make_env(stage, actions)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            try:
                self.env.reset(seed=seed)
            except TypeError:
                pass

        # Archive and stats
        self.archive: Dict[Tuple[int, int, str], Cell] = {}
        self.best_overall_x: int = 0
        self.best_overall_path: List[int] = []
        self.episodes_done: int = 0
        self.first_clear_episode: Optional[int] = None  # Episode number of first stage clear

        # Replay/Explore control
        self.current_plan: List[int] = []  # Current replay plan (return to cell)
        self.plan_ptr: int = 0
        self.mode: str = "replay"  # "replay" or "explore"
        self.explore_steps_left: int = self.explore_steps_after_return
        self.current_path: List[int] = []  # Action sequence executed in current episode
        self.ep_max_x: int = 0
        self.last_info: dict = {}
        self.prev_x_deque = deque(maxlen=self.stuck_frame_window)

        # Heuristic holds
        self.jump_hold_frames_remaining = 0
        self.left_hold_frames_remaining = 0
        self.down_hold_frames_remaining = 0

        # Optical flow tracker
        self.flow_tracker = FlowBasedTracker(fps=target_fps, hud_height=40)
        self.last_flow_estimates = {}
        # Flow-based policy
        self.flow_policy = FlowHeuristicPolicy(self.actions, self.action_indices, target_fps)

        # UI
        self._init_pygame()
        self.controller_img, self.controller_rect, self.scaled_geoms = try_load_controller_image()

        # Load archive if exists
        self._load_archive_if_exists()

        # Start first episode
        self._start_new_episode()

    def _init_pygame(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption(f"Go-Explore Mario with Optical Flow - {self.stage}")
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_medium = pygame.font.Font(None, 34)

    def _draw_optical_flow_overlay(self, frame: np.ndarray):
        """Draw optical flow tracking visualization on the game screen."""
        if not self.last_flow_estimates:
            return
            
        # Draw Mario tracking
        mario_info = self.last_flow_estimates.get("mario")
        if mario_info:
            bbox = mario_info["bbox"]
            screen_rect = scale_bbox_to_screen(bbox, GAME_SCREEN_X, GAME_SCREEN_Y, GAME_SCALE_FACTOR)
            pygame.draw.rect(self.screen, MARIO_TRACK_COLOR, screen_rect, 2)
            
            # Draw velocity vector for Mario
            vx, vy = mario_info["v_world"]
            if abs(vx) > 0.1 or abs(vy) > 0.1:  # Only draw if significant velocity
                center_x = screen_rect.centerx
                center_y = screen_rect.centery
                end_x = center_x + int(vx * 10)  # Scale for visibility
                end_y = center_y + int(vy * 10)
                pygame.draw.line(self.screen, MARIO_TRACK_COLOR, 
                               (center_x, center_y), (end_x, end_y), 2)
                # Arrow head
                pygame.draw.circle(self.screen, MARIO_TRACK_COLOR, (end_x, end_y), 3)
        
        # Draw other objects
        for obj in self.last_flow_estimates.get("objects", []):
            bbox = obj["bbox"]
            screen_rect = scale_bbox_to_screen(bbox, GAME_SCREEN_X, GAME_SCREEN_Y, GAME_SCALE_FACTOR)
            pygame.draw.rect(self.screen, OBJECT_TRACK_COLOR, screen_rect, 1)

    def _draw_ui(self, frame: np.ndarray, action_idx: int):
        self.screen.fill(BACKGROUND_COLOR)

        if frame is not None:
            surf = convert_frame_to_surface(frame)
            surf_scaled = pygame.transform.scale(surf, GAME_SCREEN_RECT.size)
            self.screen.blit(surf_scaled, GAME_SCREEN_RECT.topleft)
            
            # Draw optical flow overlay
            self._draw_optical_flow_overlay(frame)

        # Controller
        pressed_keys = action_to_pressed_keys(action_idx, self.actions)
        draw_controller(self.screen, self.controller_img, self.controller_rect, self.scaled_geoms, pressed_keys)

        # Text Info
        x = int(self.last_info.get('x_pos', 0))
        time_left = int(self.last_info.get('time', 0))
        flag = bool(self.last_info.get('flag_get', False))
        mode_str = "REPLAY" if self.mode == "replay" else "EXPLORE"

        clear_status = f"FirstClear: {self.first_clear_episode}" if self.first_clear_episode else "FirstClear: None"
        
        # Basic game info
        info_lines = [
            f"Stage: {self.stage} | Mode: {mode_str} | {clear_status}",
            f"Ep: {self.episodes_done} | BestX: {self.best_overall_x} | Cells: {len(self.archive)}",
            f"EpMaxX: {self.ep_max_x} | CurrX: {x} | Time: {time_left} | Flag: {flag}",
            f"PlanLen: {len(self.current_plan)} | PlanPtr: {self.plan_ptr} | ExploreLeft: {self.explore_steps_left}",
            f"Action: {'+'.join(self.actions[action_idx])}",
        ]
        
        # Optical flow info
        if self.last_flow_estimates:
            cam_vx, cam_vy = self.last_flow_estimates.get("camera_v", (0.0, 0.0))
            cam_x, cam_y = self.last_flow_estimates.get("camera_xy", (0.0, 0.0))
            
            mario_info = self.last_flow_estimates.get("mario")
            if mario_info:
                mario_vx, mario_vy = mario_info["v_world"]
                mario_ax, mario_ay = mario_info["a"]
                info_lines.extend([
                    f"Camera V: ({cam_vx:.2f}, {cam_vy:.2f}) px/f | Pos: ({cam_x:.1f}, {cam_y:.1f})",
                    f"Mario V_world: ({mario_vx:.2f}, {mario_vy:.2f}) | A: ({mario_ax:.2f}, {mario_ay:.2f})",
                ])
            else:
                info_lines.append(f"Camera V: ({cam_vx:.2f}, {cam_vy:.2f}) px/f | Mario: Not detected")
                
            num_objects = len(self.last_flow_estimates.get("objects", []))
            info_lines.append(f"Other objects detected: {num_objects}")
        else:
            info_lines.append("Optical flow: Initializing...")
            
        info_lines.append("Keys: R=Reset episode / S=Save archive / L=Load archive / ESC=Quit / P=Pause")
        
        # Draw all info lines
        y = GAME_SCREEN_RECT.bottom + 10
        for i, line in enumerate(info_lines):
            # Use different color for optical flow info
            color = FLOW_INFO_COLOR if "Camera V:" in line or "Mario V_world:" in line or "objects detected:" in line else TEXT_COLOR
            text = self.font_small.render(line, True, color)
            self.screen.blit(text, (GAME_SCREEN_X, y))
            y += 20

        pygame.display.flip()

    def _load_archive_if_exists(self):
        if self.archive_path and os.path.exists(self.archive_path):
            try:
                with open(self.archive_path, "rb") as f:
                    data = pickle.load(f)
                self.archive = data.get("archive", {})
                self.best_overall_x = data.get("best_overall_x", 0)
                self.best_overall_path = data.get("best_overall_path", [])
                self.episodes_done = data.get("episodes_done", 0)
                self.first_clear_episode = data.get("first_clear_episode", None)
                
                clear_info = f", first_clear_ep={self.first_clear_episode}" if self.first_clear_episode else ", not_cleared_yet"
                print(f"[INFO] Loaded archive: {len(self.archive)} cells, best_x={self.best_overall_x}, episodes_done={self.episodes_done}{clear_info} from {self.archive_path}")
            except Exception as e:
                print(f"[WARN] Failed to load archive from {self.archive_path}: {e}")

    def _save_archive(self):
        if not self.archive_path:
            return
        data = {
            "archive": self.archive,
            "best_overall_x": self.best_overall_x,
            "best_overall_path": self.best_overall_path,
            "episodes_done": self.episodes_done,
            "first_clear_episode": self.first_clear_episode,
            "stage": self.stage,
            "updated_at": time.time(),
        }
        tmp_path = self.archive_path + ".tmp"
        try:
            with open(tmp_path, "wb") as f:
                pickle.dump(data, f)
            os.replace(tmp_path, self.archive_path)
            clear_info = f", first_clear_ep={self.first_clear_episode}" if self.first_clear_episode else ""
            print(f"[INFO] Archive saved to {self.archive_path} (cells={len(self.archive)}, best_x={self.best_overall_x}, episodes_done={self.episodes_done}{clear_info})")
        except Exception as e:
            print(f"[WARN] Failed to save archive: {e}")

    def _info_to_cell_id(self, info: dict) -> Tuple[int, int, str]:
        x = int(info.get('x_pos', 0))
        y = int(info.get('y_pos', 0))
        status = str(info.get('status', 'small'))
        x_bin = x // self.cell_size_x
        y_bin = max(0, y // self.cell_size_y)
        return (int(x_bin), int(y_bin), status)

    def _maybe_add_or_update_cell(self, cell_id: Tuple[int, int, str], path: List[int], current_x: int):
        cell = self.archive.get(cell_id)
        if cell is None:
            self.archive[cell_id] = Cell(cell_id=cell_id, path=list(path), max_x=current_x)
        else:
            updated = False
            if current_x > cell.max_x:
                cell.max_x = current_x
                updated = True
            if len(path) < len(cell.path):
                cell.path = list(path)
                updated = True

    def _select_start_cell_path(self) -> List[int]:
        if not self.archive:
            return []
        cells = list(self.archive.values())
        cells.sort(key=lambda c: c.max_x, reverse=True)
        top_k = cells[: min(60, len(cells))]
        best_x = max(1, self.best_overall_x)
        weights = []
        for c in top_k:
            w = (1.0 / (c.visits + 1.0)) * (0.5 + 0.5 * (c.max_x / best_x))
            weights.append(max(1e-8, w))
        choice = random.choices(top_k, weights=weights, k=1)[0]
        choice.visits += 1
        return list(choice.path)

    def _sample_explore_action(self) -> int:
        # Get action indices, using fallback values if actions don't exist
        idx_down = self.action_indices.get('DOWN', self.action_indices.get('NOOP', 0))
        idx_down_jump = self.action_indices.get('DOWN_JUMP', idx_down)
        idx_left = self.action_indices.get('LEFT', self.action_indices.get('NOOP', 0))
        idx_left_jump = self.action_indices.get('LEFT_JUMP', idx_left)
        idx_right_dash_jump = self.action_indices.get('RIGHT_DASH_JUMP', self.action_indices.get('RIGHT_JUMP', self.action_indices.get('RIGHT', 1)))
        idx_right_dash = self.action_indices.get('RIGHT_DASH', self.action_indices.get('RIGHT', 1))

        if self.down_hold_frames_remaining > 0:
            self.down_hold_frames_remaining -= 1
            return idx_down if random.random() < 0.6 else idx_down_jump

        if self.left_hold_frames_remaining > 0:
            self.left_hold_frames_remaining -= 1
            return idx_left if random.random() < 0.6 else idx_left_jump

        if self.jump_hold_frames_remaining > 0:
            self.jump_hold_frames_remaining -= 1
            return idx_right_dash_jump

        r = random.random()
        if r < self.down_press_prob:
            self.down_hold_frames_remaining = random.randint(10, 30)
            return idx_down
        elif r < self.down_press_prob + self.left_adjust_prob:
            self.left_hold_frames_remaining = random.randint(5, 20)
            return idx_left
        elif r < self.down_press_prob + self.left_adjust_prob + self.jump_start_prob:
            self.jump_hold_frames_remaining = random.randint(self.jump_min_hold, self.jump_max_hold)
            return idx_right_dash_jump

        return idx_right_dash

    def _start_new_episode(self):
        self.episodes_done += 1
        self.current_path = []
        self.ep_max_x = 0
        self.prev_x_deque.clear()

        # Plan (return to cell)
        self.current_plan = self._select_start_cell_path()
        self.plan_ptr = 0
        self.mode = "replay"
        self.explore_steps_left = self.explore_steps_after_return

        # Reset env
        obs, info = reset_env(self.env)
        self.last_frame = obs
        self.last_info = info
        self.ep_max_x = max(self.ep_max_x, int(info.get('x_pos', 0)))
        
        # Reset policy state
        self.flow_policy.reset()

    def _end_episode(self, reason: str):
        print(f"[EP] End: {reason} | ep={self.episodes_done} | ep_max_x={self.ep_max_x} | best_x={self.best_overall_x} | cells={len(self.archive)}")
        if (self.episodes_done % self.save_every_episodes) == 0:
            self._save_archive()
        self._start_new_episode()

    def run(self, max_episodes: int):
        running = True
        paused = False

        while running and self.episodes_done <= max_episodes:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        print("[USER] Reset episode requested.")
                        self._start_new_episode()
                    elif event.key == pygame.K_s:
                        print("[USER] Save archive.")
                        self._save_archive()
                    elif event.key == pygame.K_l:
                        print("[USER] Load archive.")
                        self._load_archive_if_exists()
                    elif event.key == pygame.K_p:
                        paused = not paused
                        print(f"[USER] Pause toggled: {paused}")

            if paused:
                self._draw_ui(self.last_frame, self.action_indices.get('NOOP', 0))
                self.clock.tick(self.target_fps)
                continue

            # Decide action
            if self.mode == "replay" and self.plan_ptr < len(self.current_plan):
                action_idx = self.current_plan[self.plan_ptr]
            elif self.mode == "replay" and self.plan_ptr >= len(self.current_plan):
                # Replay finished -> switch to explore
                self.mode = "explore"
                # 直後はポリシーで決定
                action_idx = self.flow_policy.decide(self.last_flow_estimates or {})
            else:
                # 光フローポリシーで探索
                action_idx = self.flow_policy.decide(self.last_flow_estimates or {})

            # Step
            next_obs, reward, done, info = step_env(self.env, action_idx)
            self.last_frame = next_obs
            self.last_info = info
            
            # Update optical flow tracking
            self.last_flow_estimates = self.flow_tracker.update(next_obs)

            # Bookkeeping
            self.current_path.append(action_idx)
            if self.mode == "replay":
                self.plan_ptr += 1
            else:
                self.explore_steps_left = max(0, self.explore_steps_left - 1)

            x = int(info.get('x_pos', 0))
            self.ep_max_x = max(self.ep_max_x, x)
            self.prev_x_deque.append(x)

            # Cell update
            cell_id = self._info_to_cell_id(info)
            self._maybe_add_or_update_cell(cell_id, self.current_path, x)

            # Best update
            if self.ep_max_x > self.best_overall_x:
                self.best_overall_x = self.ep_max_x
                self.best_overall_path = list(self.current_path)
                print(f"[PROGRESS] New best x: {self.best_overall_x} (cells={len(self.archive)})")

            # Episode termination conditions
            reason = None
            if done:
                cleared = bool(info.get('flag_get', False))
                if cleared:
                    reason = "STAGE CLEAR"
                    # Record first clear episode if not already recorded
                    if self.first_clear_episode is None:
                        self.first_clear_episode = self.episodes_done
                        print(f"[MILESTONE] First stage clear achieved at episode {self.first_clear_episode}!")
                elif int(info.get('time', 400)) <= 1:
                    reason = "TIME UP"
                else:
                    reason = "DEAD"
            else:
                # Stuck (x barely changes)
                if len(self.prev_x_deque) == self.prev_x_deque.maxlen:
                    if max(self.prev_x_deque) - min(self.prev_x_deque) < 2:
                        reason = "STUCK"
                # Finished exploring
                if reason is None and self.mode == "explore" and self.explore_steps_left <= 0:
                    reason = "EXPLORE_DONE"

            # Replay too long (safety)
            if reason is None and self.mode == "replay" and self.plan_ptr > self.max_steps_per_episode:
                reason = "REPLAY_TOO_LONG"

            # Total steps exceeded limit (safety)
            if reason is None and len(self.current_path) >= self.max_steps_per_episode:
                reason = "MAX_STEPS"

            # UI draw
            self._draw_ui(self.last_frame, action_idx)

            # End episode if reason decided
            if reason is not None:
                self._end_episode(reason)

            self.clock.tick(self.target_fps)

        self._save_archive()
        try:
            self.env.close()
        except Exception:
            pass
        pygame.quit()

def main():
    parser = argparse.ArgumentParser(description="Go-Explore style Mario with Optical Flow tracking and Pygame UI")
    parser.add_argument("--stage", "-s", type=str, default="1-1", help="Stage like 1-1, 2-1, 4-2, 8-4, etc.")
    parser.add_argument("--episodes", "-e", type=int, default=1000, help="Max episodes to run")
    parser.add_argument("--archive", "-a", type=str, default=None, help="Path to save/load archive (default: pkl/optical_flow_archive_{stage}_{actions}.pkl)")
    parser.add_argument("--actions", type=str, default="right_only", choices=["right_only", "simple", "complex"], help="Action set to use")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--fps", type=int, default=60, help="Target FPS for UI")
    parser.add_argument("--max-steps", type=int, default=6000, help="Max steps per episode")
    parser.add_argument("--explore-steps", type=int, default=1200, help="Explore steps after replay")
    args = parser.parse_args()

    # Create pkl directory if it doesn't exist
    pkl_dir = "pkl"
    os.makedirs(pkl_dir, exist_ok=True)
    
    archive_path = args.archive or f"{pkl_dir}/optical_flow_archive_{args.stage}_{args.actions}.pkl"
    actions = ACTION_SETS[args.actions]

    print(f"Using action set '{args.actions}' with {len(actions)} actions:")
    for i, action in enumerate(actions):
        print(f"  {i}: {'+'.join(action)}")
    
    print("\nOptical Flow Features:")
    print("  - Real-time camera motion estimation")
    print("  - Mario detection and velocity tracking")
    print("  - Other moving object detection")
    print("  - Visual overlay showing tracked objects")

    runner = GoExploreUIRunner(
        stage=args.stage,
        archive_path=archive_path,
        actions=actions,
        seed=args.seed,
        max_steps_per_episode=args.max_steps,
        explore_steps_after_return=args.explore_steps,
        target_fps=args.fps,
    )
    runner.run(max_episodes=args.episodes)

if __name__ == "__main__":
    main()