"""Shared pygame UI for all agents.

Design goals (vs. the old per-file UIs):

- one modern dark-theme layout used by both the planner and Go-Explore agent,
- runtime-variable speed that never touches emulation accuracy: number keys
  pick 1x/2x/4x/8x/MAX/SMOOTH, where speed only changes frame pacing and how
  often we *draw* (every frame is still emulated exactly once),
- SMOOTH mode: emulation and planning run unthrottled while every frame is
  recorded into a playback buffer that the window drains at a steady rate —
  the planner's "thinking" pauses are hidden behind buffered playback
  (like video streaming, at the cost of a few seconds of display latency),
- live vision overlay (detected entities / Mario / gaps) and planner state,
- famicom controller visualization kept from the original project.
"""
import os
import time as _time
import urllib.request
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import pygame

# ------------------------------------------------------------------- theme

BG = (16, 18, 24)
PANEL = (26, 30, 40)
PANEL_BORDER = (52, 60, 78)
TEXT = (226, 232, 240)
TEXT_DIM = (130, 140, 158)
ACCENT = (255, 184, 28)
GOOD = (74, 222, 128)
BAD = (248, 113, 113)
INFO = (96, 165, 250)
MARIO_BOX = (74, 222, 128)
ENTITY_BOX = (248, 113, 113)
GAP_MARK = (255, 184, 28)

WINDOW_W, WINDOW_H = 1180, 720
GAME_X, GAME_Y = 24, 64
GAME_SCALE = 2
GAME_W, GAME_H = 256 * GAME_SCALE, 240 * GAME_SCALE
SIDE_X = GAME_X + GAME_W + 24
SIDE_W = WINDOW_W - SIDE_X - 24

# name -> target fps for the *play* segments (0 = unthrottled, -1 = SMOOTH)
SPEED_MODES: List[Tuple[str, int]] = [
    ('1x', 60), ('2x', 120), ('4x', 240), ('8x', 480), ('MAX', 0), ('SMOOTH', -1)]

# SMOOTH playback buffer: bounds display latency (~4s at 60fps)
SMOOTH_MAX_BUFFER = 240

# ------------------------------------------------- controller visualization

CONTROLLER_IMAGE_PATH = os.path.join('fig', 'famicon01_01.png')
CONTROLLER_IMAGE_URL = 'https://stockmaterial.net/wp/wp-content/uploads/img/famicon01_01.png'
_ORIG_SIZE = 800
_BUTTON_GEOMS = {
    'dpad_up': ('rect', (162, 362, 30, 40)),
    'dpad_down': ('rect', (162, 458, 30, 40)),
    'dpad_left': ('rect', (109, 415, 40, 30)),
    'dpad_right': ('rect', (204, 415, 40, 30)),
    'button_a': ('circle', (644, 461, 28)),
    'button_b': ('circle', (549, 461, 28)),
}
_COMMAND_TO_KEY = {'right': 'dpad_right', 'left': 'dpad_left', 'up': 'dpad_up',
                   'down': 'dpad_down', 'A': 'button_a', 'B': 'button_b'}


def _load_controller(width: int):
    if not os.path.exists(CONTROLLER_IMAGE_PATH):
        try:
            os.makedirs(os.path.dirname(CONTROLLER_IMAGE_PATH), exist_ok=True)
            urllib.request.urlretrieve(CONTROLLER_IMAGE_URL, CONTROLLER_IMAGE_PATH)
        except Exception:
            return None, {}
    try:
        base = pygame.image.load(CONTROLLER_IMAGE_PATH).convert_alpha()
    except Exception:
        return None, {}
    h = int(width * base.get_height() / base.get_width())
    img = pygame.transform.smoothscale(base, (width, h))
    sx = width / _ORIG_SIZE
    sy = h / _ORIG_SIZE
    geoms = {}
    for key, (kind, g) in _BUTTON_GEOMS.items():
        if kind == 'rect':
            geoms[key] = ('rect', pygame.Rect(int(g[0] * sx), int(g[1] * sy),
                                              max(1, int(g[2] * sx)), max(1, int(g[3] * sy))))
        else:
            geoms[key] = ('circle', (g[0] * sx, g[1] * sy, max(1, int(g[2] * min(sx, sy)))))
    return img, geoms


class GameUI:
    """UI protocol: agents call ``ui.on_frame(agent, action_idx)`` once per
    emulated frame; the return value is None or a command ('quit'/'reset').
    Long searches keep the window alive (and, in SMOOTH mode, keep playback
    running) by calling ``ui.pump()`` between rollouts."""

    def __init__(self, stage: str, actions: List[List[str]], agent_name: str,
                 speed: str = '1x', show_vision: bool = True):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        pygame.display.set_caption(f"RandoMario — {agent_name} — {stage}")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('dejavusansmono,consolas,monospace', 15)
        self.font_small = pygame.font.SysFont('dejavusansmono,consolas,monospace', 13)
        self.font_big = pygame.font.SysFont('dejavusansmono,consolas,monospace', 22, bold=True)
        self.font_huge = pygame.font.SysFont('dejavusansmono,consolas,monospace', 44, bold=True)
        self.stage = stage
        self.actions = actions
        self.agent_name = agent_name
        self.speed_idx = next(
            (i for i, (n, _) in enumerate(SPEED_MODES) if n.lower() == speed.lower()), 0)
        self.show_vision = show_vision
        self.paused = False
        self.frame_count = 0
        self._last_draw = 0.0
        self._pending_cmd: Optional[str] = None
        self.controller_img, self.controller_geoms = _load_controller(300)
        self.game_surf = pygame.Surface((256, 240))

        # SMOOTH-mode playback buffer of pre-rendered state snapshots
        self.buffer: deque = deque()
        self._last_state: Optional[dict] = None
        self._next_play = 0.0

    # ------------------------------------------------------------- snapshot

    def _snapshot_state(self, agent, action_idx: int) -> dict:
        """Freeze everything a frame needs to be drawn later (SMOOTH mode
        draws with a delay, so live agent state can't be used)."""
        frame = getattr(agent, 'last_frame', None)
        surf = None
        if frame is not None:
            surf = pygame.Surface((256, 240))
            pygame.surfarray.blit_array(surf, np.transpose(frame, (1, 0, 2)))
        vision = getattr(agent, 'vision', None)
        vis = None
        if vision is not None and vision.camera_x is not None:
            mario = vision.mario_track()
            vis = {
                'camera': vision.camera_x,
                'mario': (mario.sx, mario.sy, mario.w, mario.h) if mario else None,
                'entities': vision.entities(),
                'gaps': list(vision._gaps_world),
            }
        info = getattr(agent, 'last_info', {}) or {}
        return {
            'surf': surf,
            'action': action_idx,
            'info': dict(info),
            'episode': getattr(agent, 'episode', getattr(agent, 'episodes_done', 0)),
            'best_x': getattr(agent, 'best_x', getattr(agent, 'best_overall_x', 0)),
            'first_clear': getattr(agent, 'first_clear_episode', None),
            'rows': list(agent.ui_rows()),
            'vision': vis,
        }

    # ---------------------------------------------------------------- events

    def _poll(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._pending_cmd = 'quit'
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self._pending_cmd = 'quit'
                elif event.key == pygame.K_r:
                    self._pending_cmd = 'reset'
                elif event.key == pygame.K_p:
                    self.paused = not self.paused
                elif event.key == pygame.K_v:
                    self.show_vision = not self.show_vision
                elif pygame.K_1 <= event.key <= pygame.K_6:
                    self.speed_idx = event.key - pygame.K_1
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    self.speed_idx = min(self.speed_idx + 1, len(SPEED_MODES) - 1)
                elif event.key == pygame.K_MINUS:
                    self.speed_idx = max(self.speed_idx - 1, 0)

    def _take_cmd(self) -> Optional[str]:
        cmd, self._pending_cmd = self._pending_cmd, None
        return cmd

    # ------------------------------------------------------ smooth playback

    def _play_rate(self) -> float:
        """Adaptive playback fps: drain fast when the buffer is fat, slow
        down (instead of stalling) when production lags behind 60fps."""
        n = len(self.buffer)
        if n > 150:
            return 80.0
        if n > 60:
            return 60.0
        if n > 20:
            return 45.0
        return 30.0

    def _consume_due(self, thinking: bool = False):
        """Play buffered frames whose display time has come."""
        now = _time.time()
        drew = 0
        while self.buffer and now >= self._next_play and drew < 5:
            state = self.buffer.popleft()
            self._last_state = state
            self._draw_state(state, thinking=False)
            drew += 1
            # Fixed cadence with bounded catch-up (avoids both drift and
            # long jump-cut bursts after a slow rollout batch).
            self._next_play = max(self._next_play, now - 0.15) + 1.0 / self._play_rate()
            now = _time.time()
        if not drew and thinking and not self.buffer and self._last_state \
                and now - self._last_draw > 0.2:
            # Buffer ran dry while the planner thinks: show why we're waiting.
            self._draw_state(self._last_state, thinking=True)

    def drain(self, timeout: float = 15.0):
        """Play out whatever is still in the SMOOTH buffer before the window
        goes away (otherwise the last few seconds — e.g. the actual clear —
        would never be shown). ESC skips."""
        if SPEED_MODES[self.speed_idx][1] != -1:
            return
        t0 = _time.time()
        while self.buffer and _time.time() - t0 < timeout:
            self._poll()
            if self._pending_cmd == 'quit':
                self._pending_cmd = None
                self.buffer.clear()
                break
            self._consume_due()
            _time.sleep(0.004)

    def celebrate(self, agent, seconds: float = 4.0):
        """Called by agents on a stage clear: finish the buffered playback,
        then hold a celebration banner for a few seconds (any key to skip)."""
        self.drain()
        state = dict(self._last_state) if self._last_state \
            else self._snapshot_state(agent, 0)
        # The buffered snapshot predates the clear bookkeeping: refresh the
        # fields the banner shows (episode / first clear).
        state['episode'] = getattr(agent, 'episode', state.get('episode', 0))
        state['first_clear'] = getattr(agent, 'first_clear_episode', None)
        t0 = _time.time()
        while _time.time() - t0 < seconds:
            self._poll()
            if self._pending_cmd:
                break
            self._draw_state(state, celebrate=True)
            self.clock.tick(30)

    def _wait_paused(self):
        while self.paused:
            self._poll()
            if self._pending_cmd in ('quit', 'reset'):
                return
            if self._last_state:
                self._draw_state(self._last_state, paused=True)
            self.clock.tick(30)

    # ----------------------------------------------------------------- frame

    def on_frame(self, agent, action_idx: int) -> Optional[str]:
        self.frame_count += 1
        self._poll()
        if self.paused:
            self._wait_paused()
        cmd = self._take_cmd()
        if cmd:
            return cmd

        name, fps = SPEED_MODES[self.speed_idx]

        if fps == -1:  # SMOOTH: record now, display later at a steady rate
            self.buffer.append(self._snapshot_state(agent, action_idx))
            self._consume_due()
            # Bound latency: if emulation outruns playback, pace it down.
            while len(self.buffer) > SMOOTH_MAX_BUFFER:
                self._poll()
                if self._pending_cmd:
                    return self._take_cmd()
                self._consume_due()
                _time.sleep(0.002)
            return None

        state = self._snapshot_state(agent, action_idx)
        self._last_state = state
        now = _time.time()
        # MAX: draw at most ~30 wall-fps so rendering never limits emulation.
        if fps > 0 or (now - self._last_draw) >= (1.0 / 30.0):
            self._draw_state(state)
        if fps > 0:
            self.clock.tick(fps)
        return None

    def pump(self):
        """Called by the planner between rollouts: keeps the window alive and,
        in SMOOTH mode, keeps eating the playback buffer so the game appears
        to run while the agent is thinking."""
        self._poll()
        if self.paused:
            self._wait_paused()
        if SPEED_MODES[self.speed_idx][1] == -1:
            self._consume_due(thinking=True)

    # ------------------------------------------------------------- rendering

    def _panel(self, rect: pygame.Rect, title: str = None) -> int:
        pygame.draw.rect(self.screen, PANEL, rect, border_radius=8)
        pygame.draw.rect(self.screen, PANEL_BORDER, rect, width=1, border_radius=8)
        y = rect.y + 10
        if title:
            self.screen.blit(self.font.render(title, True, TEXT_DIM), (rect.x + 14, y))
            y += 24
        return y

    def _rows(self, rect: pygame.Rect, y: int, rows: List[Tuple[str, str, tuple]]):
        for label, value, color in rows:
            self.screen.blit(self.font_small.render(label, True, TEXT_DIM), (rect.x + 14, y))
            surf = self.font.render(str(value), True, color or TEXT)
            self.screen.blit(surf, (rect.right - 14 - surf.get_width(), y - 1))
            y += 22
        return y

    def _draw_state(self, state: dict, paused: bool = False, thinking: bool = False,
                    celebrate: bool = False):
        self.screen.fill(BG)
        info = state['info']

        # Header
        title = self.font_big.render(f"RandoMario  ·  {self.agent_name}  ·  Stage {self.stage}",
                                     True, TEXT)
        self.screen.blit(title, (GAME_X, 18))
        speed_name, _ = SPEED_MODES[self.speed_idx]
        status = f"SPEED {speed_name}"
        if speed_name == 'SMOOTH':
            status += f"  buf {len(self.buffer):3d}"
        if thinking:
            status += "   ⚙ PLANNING…"
        if paused:
            status += "   ⏸ PAUSED"
        badge = self.font.render(status, True, BAD if paused else ACCENT)
        self.screen.blit(badge, (WINDOW_W - 24 - badge.get_width(), 24))

        # Game screen
        if state['surf'] is not None:
            scaled = pygame.transform.scale(state['surf'], (GAME_W, GAME_H))
            self.screen.blit(scaled, (GAME_X, GAME_Y))
            if self.show_vision and state['vision'] is not None:
                self._draw_vision_overlay(state['vision'])
        pygame.draw.rect(self.screen, PANEL_BORDER,
                         pygame.Rect(GAME_X - 1, GAME_Y - 1, GAME_W + 2, GAME_H + 2), 1)
        # Clear banner: shows automatically the moment the flag frame is
        # displayed (works with SMOOTH's delayed playback), and is held by
        # celebrate() after the episode ends.
        if celebrate or info.get('flag_get'):
            self._draw_clear_banner(state)

        # Side: controller (left) + RUN panel (right of it)
        y = GAME_Y
        ctrl_h = 0
        if self.controller_img is not None:
            self.screen.blit(self.controller_img, (SIDE_X, y))
            ctrl_h = self.controller_img.get_height()
            pressed = [_COMMAND_TO_KEY[t] for t in self.actions[state['action']]
                       if t in _COMMAND_TO_KEY]
            for key in pressed:
                kind, g = self.controller_geoms[key]
                if kind == 'rect':
                    pygame.draw.rect(self.screen, ACCENT, g.move(SIDE_X, y), 3, border_radius=4)
                else:
                    pygame.draw.circle(self.screen, ACCENT,
                                       (int(SIDE_X + g[0]), int(y + g[1])), g[2], 3)

        run_x = SIDE_X + 312
        rect = pygame.Rect(run_x, y, WINDOW_W - run_x - 24, 168)
        py = self._panel(rect, 'RUN')
        rows = [
            ('Episode', state['episode'], TEXT),
            ('X', int(info.get('x_pos', 0)), TEXT),
            ('Best X', state['best_x'], INFO),
            ('Time', int(info.get('time', 0)), TEXT),
            ('First clear', state['first_clear'] or '—',
             GOOD if state['first_clear'] else TEXT_DIM),
        ]
        if info.get('flag_get'):
            rows.append(('FLAG', 'GET!', GOOD))
        self._rows(rect, py, rows)
        y = max(rect.bottom, GAME_Y + ctrl_h) + 14

        # Side: agent panel (full side width)
        rect = pygame.Rect(SIDE_X, y, SIDE_W, WINDOW_H - y - 52)
        py = self._panel(rect, 'AGENT')
        self._rows(rect, py, [(k, v, c) for (k, v, c) in state['rows']])

        # Footer
        keys = "1-6 speed (6=SMOOTH) · +/- speed · P pause · V vision · R reset · ESC quit"
        self.screen.blit(self.font_small.render(keys, True, TEXT_DIM),
                         (GAME_X, WINDOW_H - 32))
        pygame.display.flip()
        self._last_draw = _time.time()

    def _draw_clear_banner(self, state: dict):
        overlay = pygame.Surface((GAME_W, GAME_H), pygame.SRCALPHA)
        overlay.fill((10, 12, 16, 140))
        self.screen.blit(overlay, (GAME_X, GAME_Y))
        cx = GAME_X + GAME_W // 2
        cy = GAME_Y + GAME_H // 2
        line1 = self.font_huge.render('STAGE CLEAR!', True, ACCENT)
        sub = f"Stage {self.stage}  ·  Episode {state['episode']}"
        if state['first_clear'] == state['episode']:
            sub += "  ·  FIRST CLEAR"
        line2 = self.font_big.render(sub, True, GOOD)
        if line2.get_width() > GAME_W - 68:
            line2 = self.font.render(sub, True, GOOD)
        width = max(440, line1.get_width() + 60, line2.get_width() + 60)
        box = pygame.Rect(0, 0, min(width, GAME_W - 8), 150)
        box.center = (cx, cy)
        pygame.draw.rect(self.screen, PANEL, box, border_radius=14)
        pygame.draw.rect(self.screen, GOOD, box, width=3, border_radius=14)
        self.screen.blit(line1, (cx - line1.get_width() // 2, box.y + 24))
        self.screen.blit(line2, (cx - line2.get_width() // 2, box.y + 90))

    def _draw_vision_overlay(self, vis: dict):
        cam = vis['camera']
        if vis['mario'] is not None:
            sx, sy, w, h = vis['mario']
            x, ytop = GAME_X + int(sx * GAME_SCALE), GAME_Y + int(sy * GAME_SCALE)
            pygame.draw.rect(self.screen, MARIO_BOX,
                             pygame.Rect(x - w, ytop - h, w * 2 + 4, h * 2 + 4), 2)
        for e in vis['entities']:
            px = GAME_X + int((e['x'] - cam) * GAME_SCALE)
            py = GAME_Y + int(e['y'] * GAME_SCALE)
            r = pygame.Rect(px - e['w'], py - e['h'], e['w'] * 2 + 4, e['h'] * 2 + 4)
            pygame.draw.rect(self.screen, ENTITY_BOX, r, 2)
            vel = self.font_small.render(f"{e['vx']:+.1f}", True, ENTITY_BOX)
            self.screen.blit(vel, (r.x, r.y - 14))
        for g in vis['gaps']:
            px = GAME_X + int((g - cam) * GAME_SCALE)
            if GAME_X <= px <= GAME_X + GAME_W:
                pygame.draw.line(self.screen, GAP_MARK, (px, GAME_Y + GAME_H - 60),
                                 (px, GAME_Y + GAME_H), 3)

    def close(self):
        try:
            pygame.quit()
        except Exception:
            pass
