"""Shared pygame UI for all agents.

Design goals (vs. the old per-file UIs):

- one modern dark-theme layout used by both the planner and Go-Explore agent,
- runtime-variable speed that never touches emulation accuracy: number keys
  pick 1x/2x/4x/8x/MAX, where speed only changes frame pacing and how often
  we *draw* (every frame is still emulated exactly once),
- live vision overlay (detected entities / Mario / gaps) and planner state,
- famicom controller visualization kept from the original project.
"""
import os
import urllib.request
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

SPEED_MODES: List[Tuple[str, int]] = [
    ('1x', 60), ('2x', 120), ('4x', 240), ('8x', 480), ('MAX', 0)]

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
    emulated frame; the return value is None or a command ('quit'/'reset')."""

    def __init__(self, stage: str, actions: List[List[str]], agent_name: str,
                 speed: str = '1x', show_vision: bool = True):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        pygame.display.set_caption(f"RandoMario — {agent_name} — {stage}")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('dejavusansmono,consolas,monospace', 15)
        self.font_small = pygame.font.SysFont('dejavusansmono,consolas,monospace', 13)
        self.font_big = pygame.font.SysFont('dejavusansmono,consolas,monospace', 22, bold=True)
        self.stage = stage
        self.actions = actions
        self.agent_name = agent_name
        self.speed_idx = next((i for i, (n, _) in enumerate(SPEED_MODES) if n == speed),
                              len(SPEED_MODES) - 1 if speed.upper() == 'MAX' else 0)
        self.show_vision = show_vision
        self.paused = False
        self.frame_count = 0
        self._last_draw = 0.0
        self.controller_img, self.controller_geoms = _load_controller(300)
        self.game_surf = pygame.Surface((256, 240))

    def pump(self):
        """Keep the window responsive during long planner searches."""
        pygame.event.pump()

    # ---------------------------------------------------------------- events

    def _poll(self) -> Optional[str]:
        cmd = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                cmd = 'quit'
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    cmd = 'quit'
                elif event.key == pygame.K_r:
                    cmd = 'reset'
                elif event.key == pygame.K_p:
                    self.paused = not self.paused
                elif event.key == pygame.K_v:
                    self.show_vision = not self.show_vision
                elif pygame.K_1 <= event.key <= pygame.K_5:
                    self.speed_idx = event.key - pygame.K_1
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    self.speed_idx = min(self.speed_idx + 1, len(SPEED_MODES) - 1)
                elif event.key == pygame.K_MINUS:
                    self.speed_idx = max(self.speed_idx - 1, 0)
        return cmd

    # ----------------------------------------------------------------- frame

    def on_frame(self, agent, action_idx: int) -> Optional[str]:
        self.frame_count += 1
        cmd = self._poll()
        if cmd:
            return cmd

        name, fps = SPEED_MODES[self.speed_idx]
        import time as _time
        now = _time.time()
        # MAX: draw at most ~30 wall-fps so rendering never limits emulation.
        draw = fps > 0 or (now - self._last_draw) >= (1.0 / 30.0)
        if draw:
            self._draw(agent, action_idx)
            self._last_draw = now

        while self.paused:
            self._draw(agent, action_idx, paused=True)
            sub = self._poll()
            if sub in ('quit', 'reset'):
                return sub
            if sub is None and not self.paused:
                break
            self.clock.tick(30)

        if fps > 0:
            self.clock.tick(fps)
        return None

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

    def _draw(self, agent, action_idx: int, paused: bool = False):
        self.screen.fill(BG)
        info = getattr(agent, 'last_info', {}) or {}
        frame = getattr(agent, 'last_frame', None)

        # Header
        title = self.font_big.render(f"RandoMario  ·  {self.agent_name}  ·  Stage {self.stage}",
                                     True, TEXT)
        self.screen.blit(title, (GAME_X, 18))
        speed_name, _ = SPEED_MODES[self.speed_idx]
        badge = self.font.render(f"SPEED {speed_name}" + ("   ⏸ PAUSED" if paused else ""),
                                 True, ACCENT if not paused else BAD)
        self.screen.blit(badge, (WINDOW_W - 24 - badge.get_width(), 24))

        # Game screen
        if frame is not None:
            pygame.surfarray.blit_array(self.game_surf, np.transpose(frame, (1, 0, 2)))
            scaled = pygame.transform.scale(self.game_surf, (GAME_W, GAME_H))
            self.screen.blit(scaled, (GAME_X, GAME_Y))
            if self.show_vision:
                self._draw_vision_overlay(agent)
        pygame.draw.rect(self.screen, PANEL_BORDER,
                         pygame.Rect(GAME_X - 1, GAME_Y - 1, GAME_W + 2, GAME_H + 2), 1)

        # Side: controller (left) + RUN panel (right of it)
        y = GAME_Y
        ctrl_h = 0
        if self.controller_img is not None:
            self.screen.blit(self.controller_img, (SIDE_X, y))
            ctrl_h = self.controller_img.get_height()
            pressed = [_COMMAND_TO_KEY[t] for t in self.actions[action_idx]
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
        flag = bool(info.get('flag_get', False))
        rows = [
            ('Episode', getattr(agent, 'episode', getattr(agent, 'episodes_done', 0)), TEXT),
            ('X', int(info.get('x_pos', 0)), TEXT),
            ('Best X', getattr(agent, 'best_x', getattr(agent, 'best_overall_x', 0)), INFO),
            ('Time', int(info.get('time', 0)), TEXT),
            ('First clear', getattr(agent, 'first_clear_episode', None) or '—',
             GOOD if getattr(agent, 'first_clear_episode', None) else TEXT_DIM),
        ]
        if flag:
            rows.append(('FLAG', 'GET!', GOOD))
        self._rows(rect, py, rows)
        y = max(rect.bottom, GAME_Y + ctrl_h) + 14

        # Side: agent panel (full side width)
        rect = pygame.Rect(SIDE_X, y, SIDE_W, WINDOW_H - y - 52)
        py = self._panel(rect, 'AGENT')
        self._rows(rect, py, [(k, v, c) for (k, v, c) in agent.ui_rows()])

        # Footer
        keys = "1-5 speed · +/- speed · P pause · V vision · R reset · ESC quit"
        self.screen.blit(self.font_small.render(keys, True, TEXT_DIM),
                         (GAME_X, WINDOW_H - 32))
        pygame.display.flip()

    def _draw_vision_overlay(self, agent):
        vision = getattr(agent, 'vision', None)
        if vision is None or vision.camera_x is None:
            return
        cam = vision.camera_x

        def to_screen(wx, sy):
            return (GAME_X + int((wx - cam) * GAME_SCALE), GAME_Y + int(sy * GAME_SCALE))

        mario = vision.mario_track()
        if mario is not None:
            x, ytop = GAME_X + int(mario.sx * GAME_SCALE), GAME_Y + int(mario.sy * GAME_SCALE)
            pygame.draw.rect(self.screen, MARIO_BOX,
                             pygame.Rect(x - mario.w, ytop - mario.h,
                                         mario.w * 2 + 4, mario.h * 2 + 4), 2)
        for e in vision.entities():
            px, py = to_screen(e['x'], e['y'])
            r = pygame.Rect(px - e['w'], py - e['h'], e['w'] * 2 + 4, e['h'] * 2 + 4)
            pygame.draw.rect(self.screen, ENTITY_BOX, r, 2)
            vel = self.font_small.render(f"{e['vx']:+.1f}", True, ENTITY_BOX)
            self.screen.blit(vel, (r.x, r.y - 14))
        for g in vision._gaps_world:
            px = GAME_X + int((g - cam) * GAME_SCALE)
            if GAME_X <= px <= GAME_X + GAME_W:
                pygame.draw.line(self.screen, GAP_MARK, (px, GAME_Y + GAME_H - 60),
                                 (px, GAME_Y + GAME_H), 3)

    def close(self):
        try:
            pygame.quit()
        except Exception:
            pass
