"""Go-Explore style random agent (ported from the original test.py).

Kept as the baseline to compare against the model-based planner agent.
Archives stay in the original ``pkl/`` pickle format so existing data and
the dashboard keep working.
"""
import os
import pickle
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .actions import ACTION_SETS, get_action_indices
from .env_utils import MarioSession


@dataclass
class Cell:
    cell_id: Tuple[int, int, str]  # (x_bin, y_bin, status)
    path: List[int] = field(default_factory=list)
    max_x: int = 0
    visits: int = 0
    created_at: float = field(default_factory=time.time)


def default_archive_path(stage: str, actions_name: str) -> str:
    return os.path.join('pkl', f'go_explore_archive_{stage}_{actions_name}.pkl')


class CellUnpickler(pickle.Unpickler):
    """Old archives were written by ``test.py`` run as a script, so their
    Cell class is recorded as ``__main__.Cell`` (or ``test.Cell``).  Redirect
    any class literally named Cell to ours."""

    def find_class(self, module, name):
        if name == 'Cell':
            return Cell
        return super().find_class(module, name)


def load_archive(path: str) -> dict:
    with open(path, 'rb') as f:
        return CellUnpickler(f).load()


class GoExploreAgent:
    def __init__(self, stage: str, archive_path: Optional[str] = None,
                 actions_name: str = 'right_only', ui=None, seed: Optional[int] = None,
                 cell_size_x: int = 50, cell_size_y: int = 40,
                 max_steps_per_episode: int = 6000, explore_steps_after_return: int = 1200,
                 stuck_frame_window: int = 60, save_every_episodes: int = 10,
                 down_press_prob: float = 0.02, left_adjust_prob: float = 0.05,
                 jump_start_prob: float = 0.12, jump_min_hold: int = 6,
                 jump_max_hold: int = 14, log=print):
        self.stage = stage
        self.actions_name = actions_name
        self.actions = ACTION_SETS[actions_name]
        self.action_indices = get_action_indices(self.actions)
        self.archive_path = archive_path or default_archive_path(stage, actions_name)
        self.ui = ui
        self.log = log

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

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.session = MarioSession(stage, self.actions)
        self.vision = None  # keeps the shared UI protocol happy

        self.archive: Dict[Tuple[int, int, str], Cell] = {}
        self.best_x = 0
        self.best_overall_path: List[int] = []
        self.episode = 0
        self.first_clear_episode: Optional[int] = None

        self.mode = 'replay'
        self.current_plan: List[int] = []
        self.plan_ptr = 0
        self.explore_steps_left = 0
        self.current_path: List[int] = []
        self.ep_max_x = 0
        self.last_info: dict = {}
        self.last_frame = None
        self.prev_x_deque = deque(maxlen=stuck_frame_window)
        self._jump_hold = 0
        self._left_hold = 0
        self._down_hold = 0

        self._load_archive_if_exists()

    # ------------------------------------------------------------- archive

    def _load_archive_if_exists(self):
        if not (self.archive_path and os.path.exists(self.archive_path)):
            return
        try:
            data = load_archive(self.archive_path)
            self.archive = data.get('archive', {})
            self.best_x = data.get('best_overall_x', 0)
            self.best_overall_path = data.get('best_overall_path', [])
            self.episode = data.get('episodes_done', 0)
            self.first_clear_episode = data.get('first_clear_episode', None)
            self.log(f"[INFO] Loaded archive: {len(self.archive)} cells, "
                     f"best_x={self.best_x}, episodes={self.episode}")
        except Exception as e:
            self.log(f"[WARN] Failed to load archive {self.archive_path}: {e}")

    def save_archive(self):
        if not self.archive_path:
            return
        data = {
            'archive': self.archive,
            'best_overall_x': self.best_x,
            'best_overall_path': self.best_overall_path,
            'episodes_done': self.episode,
            'first_clear_episode': self.first_clear_episode,
            'stage': self.stage,
            'updated_at': time.time(),
        }
        tmp = self.archive_path + '.tmp'
        try:
            os.makedirs(os.path.dirname(self.archive_path) or '.', exist_ok=True)
            with open(tmp, 'wb') as f:
                pickle.dump(data, f)
            os.replace(tmp, self.archive_path)
            self.log(f"[INFO] Archive saved ({len(self.archive)} cells, best_x={self.best_x})")
        except Exception as e:
            self.log(f"[WARN] Failed to save archive: {e}")

    # ------------------------------------------------------------ go-explore

    def _cell_id(self, info: dict) -> Tuple[int, int, str]:
        x = int(info.get('x_pos', 0))
        y = int(info.get('y_pos', 0))
        return (x // self.cell_size_x, max(0, y // self.cell_size_y),
                str(info.get('status', 'small')))

    def _update_cell(self, cell_id, path: List[int], x: int):
        cell = self.archive.get(cell_id)
        if cell is None:
            self.archive[cell_id] = Cell(cell_id=cell_id, path=list(path), max_x=x)
        else:
            if x > cell.max_x:
                cell.max_x = x
            if len(path) < len(cell.path):
                cell.path = list(path)

    def _select_start_path(self) -> List[int]:
        if not self.archive:
            return []
        cells = sorted(self.archive.values(), key=lambda c: c.max_x, reverse=True)
        top = cells[:min(60, len(cells))]
        best = max(1, self.best_x)
        weights = [max(1e-8, (1.0 / (c.visits + 1.0)) * (0.5 + 0.5 * c.max_x / best))
                   for c in top]
        choice = random.choices(top, weights=weights, k=1)[0]
        choice.visits += 1
        return list(choice.path)

    def _sample_action(self) -> int:
        idx = self.action_indices
        i_down = idx.get('DOWN', idx.get('NOOP', 0))
        i_down_j = idx.get('DOWN_JUMP', i_down)
        i_left = idx.get('LEFT', idx.get('NOOP', 0))
        i_left_j = idx.get('LEFT_JUMP', i_left)
        i_rdj = idx.get('RIGHT_DASH_JUMP', idx.get('RIGHT_JUMP', idx.get('RIGHT', 1)))
        i_rd = idx.get('RIGHT_DASH', idx.get('RIGHT', 1))

        if self._down_hold > 0:
            self._down_hold -= 1
            return i_down if random.random() < 0.6 else i_down_j
        if self._left_hold > 0:
            self._left_hold -= 1
            return i_left if random.random() < 0.6 else i_left_j
        if self._jump_hold > 0:
            self._jump_hold -= 1
            return i_rdj

        r = random.random()
        if r < self.down_press_prob:
            self._down_hold = random.randint(10, 30)
            return i_down
        if r < self.down_press_prob + self.left_adjust_prob:
            self._left_hold = random.randint(5, 20)
            return i_left
        if r < self.down_press_prob + self.left_adjust_prob + self.jump_start_prob:
            self._jump_hold = random.randint(self.jump_min_hold, self.jump_max_hold)
            return i_rdj
        return i_rd

    # -------------------------------------------------------------- episode

    def _start_episode(self):
        self.episode += 1
        self.current_path = []
        self.ep_max_x = 0
        self.prev_x_deque.clear()
        self.current_plan = self._select_start_path()
        self.plan_ptr = 0
        self.mode = 'replay'
        self.explore_steps_left = self.explore_steps_after_return
        obs, info = self.session.reset()
        self.last_frame = obs
        self.last_info = info
        self.ep_max_x = int(info.get('x_pos', 0))

    def run(self, max_episodes: int = 1000) -> dict:
        self._start_episode()
        running = True
        while running and self.episode <= max_episodes:
            if self.mode == 'replay' and self.plan_ptr < len(self.current_plan):
                action = self.current_plan[self.plan_ptr]
            else:
                if self.mode == 'replay':
                    self.mode = 'explore'
                action = self._sample_action()

            obs, reward, done, info = self.session.step(action)
            self.last_frame = obs
            self.last_info = info
            self.current_path.append(action)
            if self.mode == 'replay':
                self.plan_ptr += 1
            else:
                self.explore_steps_left = max(0, self.explore_steps_left - 1)

            x = int(info.get('x_pos', 0))
            self.ep_max_x = max(self.ep_max_x, x)
            self.prev_x_deque.append(x)
            self._update_cell(self._cell_id(info), self.current_path, x)
            if self.ep_max_x > self.best_x:
                self.best_x = self.ep_max_x
                self.best_overall_path = list(self.current_path)

            reason = None
            if done:
                if info.get('flag_get'):
                    reason = 'STAGE CLEAR'
                    if self.first_clear_episode is None:
                        self.first_clear_episode = self.episode
                        self.log(f"[MILESTONE] First clear at episode {self.episode}!")
                elif int(info.get('time', 400)) <= 1:
                    reason = 'TIME UP'
                else:
                    reason = 'DEAD'
            elif len(self.prev_x_deque) == self.prev_x_deque.maxlen \
                    and max(self.prev_x_deque) - min(self.prev_x_deque) < 2:
                reason = 'STUCK'
            elif self.mode == 'explore' and self.explore_steps_left <= 0:
                reason = 'EXPLORE_DONE'
            elif len(self.current_path) >= self.max_steps_per_episode:
                reason = 'MAX_STEPS'

            if self.ui is not None:
                cmd = self.ui.on_frame(self, action)
                if cmd == 'quit':
                    running = False
                elif cmd == 'reset':
                    reason = 'USER_RESET'

            if reason is not None:
                self.log(f"[EP] {reason} | ep={self.episode} ep_max_x={self.ep_max_x} "
                         f"best_x={self.best_x} cells={len(self.archive)}")
                if self.episode % self.save_every_episodes == 0:
                    self.save_archive()
                self._start_episode()

        self.save_archive()
        self.session.close()
        return {
            'stage': self.stage,
            'cleared': self.first_clear_episode is not None,
            'first_clear_episode': self.first_clear_episode,
            'episodes': self.episode,
            'best_x': self.best_x,
        }

    # ------------------------------------------------------------------- UI

    def ui_rows(self):
        return [
            ('Mode', self.mode.upper(), None),
            ('Cells', len(self.archive), None),
            ('Plan len', len(self.current_plan), None),
            ('Plan ptr', self.plan_ptr, None),
            ('Explore left', self.explore_steps_left, None),
            ('Ep max X', self.ep_max_x, None),
        ]
