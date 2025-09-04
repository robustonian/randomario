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

# Go-Explore style agent with UI integration
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
        pygame.display.set_caption(f"Go-Explore Mario - {self.stage}")
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 24)
        self.font_medium = pygame.font.Font(None, 34)

    def _draw_ui(self, frame: np.ndarray, action_idx: int):
        self.screen.fill(BACKGROUND_COLOR)

        if frame is not None:
            surf = convert_frame_to_surface(frame)
            surf_scaled = pygame.transform.scale(surf, GAME_SCREEN_RECT.size)
            self.screen.blit(surf_scaled, GAME_SCREEN_RECT.topleft)

        # Controller
        pressed_keys = action_to_pressed_keys(action_idx, self.actions)
        draw_controller(self.screen, self.controller_img, self.controller_rect, self.scaled_geoms, pressed_keys)

        # Text Info
        x = int(self.last_info.get('x_pos', 0))
        time_left = int(self.last_info.get('time', 0))
        flag = bool(self.last_info.get('flag_get', False))
        mode_str = "REPLAY" if self.mode == "replay" else "EXPLORE"

        info_lines = [
            f"Stage: {self.stage} | Mode: {mode_str}",
            f"Ep: {self.episodes_done} | BestX: {self.best_overall_x} | Cells: {len(self.archive)}",
            f"EpMaxX: {self.ep_max_x} | CurrX: {x} | Time: {time_left} | Flag: {flag}",
            f"PlanLen: {len(self.current_plan)} | PlanPtr: {self.plan_ptr} | ExploreLeft: {self.explore_steps_left}",
            f"Action: {'+'.join(self.actions[action_idx])}",
            "Keys: R=Reset episode / S=Save archive / L=Load archive / ESC=Quit / P=Pause",
        ]
        y = GAME_SCREEN_RECT.bottom + 10
        for line in info_lines:
            text = self.font_small.render(line, True, TEXT_COLOR)
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
                print(f"[INFO] Loaded archive: {len(self.archive)} cells, best_x={self.best_overall_x} from {self.archive_path}")
            except Exception as e:
                print(f"[WARN] Failed to load archive from {self.archive_path}: {e}")

    def _save_archive(self):
        if not self.archive_path:
            return
        data = {
            "archive": self.archive,
            "best_overall_x": self.best_overall_x,
            "best_overall_path": self.best_overall_path,
            "stage": self.stage,
            "updated_at": time.time(),
        }
        tmp_path = self.archive_path + ".tmp"
        try:
            with open(tmp_path, "wb") as f:
                pickle.dump(data, f)
            os.replace(tmp_path, self.archive_path)
            print(f"[INFO] Archive saved to {self.archive_path} (cells={len(self.archive)}, best_x={self.best_overall_x})")
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
                action_idx = self._sample_explore_action()
            else:
                action_idx = self._sample_explore_action()

            # Step
            next_obs, reward, done, info = step_env(self.env, action_idx)
            self.last_frame = next_obs
            self.last_info = info

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
    parser = argparse.ArgumentParser(description="Go-Explore style Mario with Pygame UI")
    parser.add_argument("--stage", "-s", type=str, default="1-1", help="Stage like 1-1, 2-1, 4-2, 8-4, etc.")
    parser.add_argument("--episodes", "-e", type=int, default=1000, help="Max episodes to run")
    parser.add_argument("--archive", "-a", type=str, default=None, help="Path to save/load archive (e.g., ./mario_ge_ui_1-1.pkl)")
    parser.add_argument("--actions", type=str, default="right_only", choices=["right_only", "simple", "complex"], help="Action set to use")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--fps", type=int, default=60, help="Target FPS for UI")
    parser.add_argument("--max-steps", type=int, default=6000, help="Max steps per episode")
    parser.add_argument("--explore-steps", type=int, default=1200, help="Explore steps after replay")
    args = parser.parse_args()

    archive_path = args.archive or f"./go_explore_archive_{args.stage}_{args.actions}.pkl"
    actions = ACTION_SETS[args.actions]

    print(f"Using action set '{args.actions}' with {len(actions)} actions:")
    for i, action in enumerate(actions):
        print(f"  {i}: {'+'.join(action)}")

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