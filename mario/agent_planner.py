"""Episode loop for the model-based planner agent.

Responsibilities beyond the per-replan search (see planner.py):

- executes chosen plans on the real environment,
- watches for stalls / warp-zone exits / section loops,
- classifies every real death (pit / enemy / hazard / timeout) and stores it
  in the hazard memory so later episodes plan around it,
- records results to the planner results DB (separate from the random agent).
"""
import time
from collections import deque
from typing import Optional

import numpy as np

from .env_utils import MarioSession
from .actions import ACTION_SETS
from .memory import PlannerMemory
from .planner import Planner

STUCK_ESCALATE_FRAMES = 240
STUCK_ABORT_FRAMES = 900
LOOP_JUMP_BACK = 150


class PlannerAgent:
    def __init__(self, stage: str, memory: Optional[PlannerMemory] = None,
                 ui=None, vision=None, max_frames_per_episode: int = 14000,
                 log=print):
        self.stage = stage
        self.session = MarioSession(stage, ACTION_SETS['planner'])
        # Planning happens on a SHADOW emulator. nes-py's savestate restore is
        # imperfect once frames advance between backup and restore (PPU phase
        # leaks, shifting lag frames), so rollouts slowly corrupt the emulator
        # they run on. Keeping the play emulator rollout-free makes the real
        # trajectory a pure function of its inputs — i.e. always replayable.
        self.shadow = MarioSession(stage, ACTION_SETS['planner'])
        self.memory = memory if memory is not None else PlannerMemory()
        self.planner = Planner(self.shadow, self.memory)
        self.ui = ui
        if ui is not None and hasattr(ui, 'pump'):
            self.planner.pump = ui.pump
        self.vision = vision
        self.max_frames_per_episode = max_frames_per_episode
        self.log = log
        self.run_id = time.strftime('%Y%m%d-%H%M%S')

        # Live state exposed to the UI
        self.episode = 0
        self.best_x = 0
        self.last_plan = None
        self.last_info = {}
        self.last_frame = None
        self.first_clear_episode = None
        # Fair-benchmark bookkeeping (see benchmark.py)
        self.real_frames = 0     # frames actually played
        self.deaths = 0
        self.wall_total = 0.0
        self.clear_stats = None  # metrics frozen at the first clear

    @property
    def total_emulator_frames(self) -> int:
        """Every frame emulated anywhere: real play + shadow mirroring +
        rollouts + resyncs + replay validation."""
        return self.session.frames_emulated + self.shadow.frames_emulated

    def _resync_shadow(self, episode_actions) -> None:
        """Rebuild the planning emulator from the play emulator's input
        history (the play emulator is pure, so replaying its inputs
        reproduces it exactly)."""
        self.shadow.reset()
        for a in episode_actions:
            if self.shadow.smb.done:
                break
            self.shadow.step(a)

    def _validate_replay(self, actions) -> bool:
        """Replay the recorded inputs from a fresh reset and confirm they
        reach the flag again. Guarantees that stored 'clear' recordings are
        actually reproducible before they ever land in the DB."""
        session = self.session
        obs, info = session.reset()
        for a in actions:
            obs, reward, done, info = session.step(a)
            if done:
                return bool(info.get('flag_get'))
        return False

    # ------------------------------------------------------- death analysis

    def _classify_death(self, info: dict) -> (str, dict):
        smb = self.session.smb
        detail = {}
        if smb.ram[0x00b5] > 1:
            cause = 'pit'
        elif int(info.get('time', 400)) <= 1:
            cause = 'timeout'
        else:
            cause = 'contact'
            if self.vision is not None:
                near = self.vision.entities_near(int(info.get('x_pos', 0)), radius=40)
                if near:
                    cause = 'enemy'
                    detail['entities'] = near[:3]
        return cause, detail

    # -------------------------------------------------------------- episode

    def run_episode(self) -> dict:
        session = self.session
        self.episode += 1
        rollout_frames_start = self.planner.rollout_frames_total
        t0 = time.time()
        obs, info = session.reset()
        self.shadow.reset()  # canonical reset: both emulators now identical
        resyncs = 0
        if self.vision is not None:
            self.vision.reset()
            self.vision.update(obs, info)
        self.last_info = info
        self.last_frame = obs

        max_x = int(info.get('x_pos', 0))
        frames = 0
        prev_x = max_x
        stuck_frames = 0
        stuck_base_y = int(info.get('y_pos', 0))
        loops = 0
        force_wave = 1
        last_action = None
        result, cause, detail = None, '', {}
        area = int(session.smb.ram[0x0760])
        episode_plans = []  # (x0, sig, frame) history for blame assignment
        episode_actions = []  # every executed action, for deterministic replay
        xy_history = deque(maxlen=700)  # (x, y) per frame, for lane profiles
        self.planner.recent_xy = xy_history

        while result is None:
            hints = self.vision.hints() if self.vision is not None else None
            plan = self.planner.plan(hints=hints, last_action=last_action,
                                     force_wave=force_wave)
            self.last_plan = plan
            episode_plans.append((plan.x0, plan.sig, frames))
            force_wave = 1

            for i in range(min(plan.exec_frames, len(plan.actions))):
                action = plan.actions[i]
                obs, reward, done, info = session.step(action)
                frames += 1
                last_action = action
                episode_actions.append(action)
                # Mirror the step on the planning emulator; repair it from the
                # input history whenever rollout leakage made it drift.
                if not done:
                    if not self.shadow.smb.done:
                        try:
                            self.shadow.step(action)
                        except ValueError:
                            pass
                    if self.shadow.smb.done or not np.array_equal(
                            session.smb.ram, self.shadow.smb.ram):
                        resyncs += 1
                        self._resync_shadow(episode_actions)
                self.last_info = info
                self.last_frame = obs
                if self.vision is not None:
                    self.vision.update(obs, info)

                if self.ui is not None:
                    cmd = self.ui.on_frame(self, action)
                    if cmd == 'quit':
                        result, cause = 'abort', 'user'
                        break
                    if cmd == 'reset':
                        result, cause = 'reset', 'user'
                        break

                x = int(info.get('x_pos', 0))
                new_area = int(session.smb.ram[0x0760])
                if new_area != area:
                    # Legit section change (pipe / flag area): rebase trackers.
                    area = new_area
                    prev_x = x
                    stuck_frames = 0
                elif x < prev_x - LOOP_JUMP_BACK and not done:
                    loops += 1
                    self.memory.record_death(self.stage, prev_x, 'loop',
                                             {'to_x': x, 'loops': loops})
                    # A loop warp means the lane we took was wrong — remember
                    # the failed y-profile and blame the approach plans.
                    from .planner import lane_profile
                    prof = lane_profile(xy_history, prev_x - 800, prev_x + 8)
                    if prof:
                        self.planner.failed_lanes.append(prof)
                    for px0, psig, pframe in episode_plans:
                        if pframe >= frames - 200:
                            self.memory.blacklist_plan(self.stage, px0, psig)
                    self.log(f"[{self.stage}] loop detected at x={prev_x} -> {x} "
                             f"(failed lanes: {len(self.planner.failed_lanes)})")
                prev_x = x
                y = int(info.get('y_pos', 0))
                xy_history.append((x, y))
                if x > max_x:
                    max_x = x
                    if x > self.best_x:
                        self.best_x = x
                    stuck_frames = 0
                    stuck_base_y = y
                elif abs(y - stuck_base_y) > 88:
                    # Big net vertical displacement (elevator/vine ride, tall
                    # climb) is progress too, even while x stands still.
                    stuck_frames = 0
                    stuck_base_y = y
                else:
                    stuck_frames += 1

                if done:
                    if info.get('flag_get'):
                        result, cause = 'clear', 'flag'
                    else:
                        cause, detail = self._classify_death(info)
                        result = 'death'
                    break
                if frames >= self.max_frames_per_episode:
                    result, cause = 'maxframes', 'limit'
                    break
                if stuck_frames >= STUCK_ABORT_FRAMES:
                    result, cause = 'stuck', 'stuck'
                    self.memory.record_death(self.stage, x, 'stuck', {})
                    break

            if result is None and stuck_frames >= STUCK_ESCALATE_FRAMES:
                force_wave = 2 if stuck_frames < 2 * STUCK_ESCALATE_FRAMES else 3

        wall = time.time() - t0
        x_final = int(self.last_info.get('x_pos', 0))
        if result == 'death':
            self.memory.record_death(self.stage, x_final, cause, detail)
            # Blame assignment: the plan that doomed us is often not the one
            # running when we died (a jump into a dead-end a few plans back),
            # but blaming too far back poisons the good plans that got us
            # here — only blacklist plans started shortly before the death.
            for px0, psig, pframe in episode_plans:
                if pframe >= frames - 160:
                    self.memory.blacklist_plan(self.stage, px0, psig)
        if result == 'clear' and self.first_clear_episode is None:
            self.first_clear_episode = self.episode
        validated = True
        if result not in ('abort', 'reset'):
            actions_to_store = episode_actions
            if result == 'clear' and not self._validate_replay(episode_actions):
                self.log(f"[{self.stage}] WARNING: clear recording failed replay "
                         f"validation — inputs not stored")
                actions_to_store = None
                validated = False
            self.memory.record_episode(
                self.stage, self.run_id, self.episode, result, cause, max_x,
                frames, int(self.last_info.get('time', 0)), wall,
                actions=actions_to_store,
                rollout_frames=self.planner.rollout_frames_total - rollout_frames_start)
        self.log(f"[{self.stage}] ep{self.episode}: {result}({cause}) "
                 f"max_x={max_x} frames={frames} wall={wall:.1f}s "
                 f"plans={self.planner.plans_total} resyncs={resyncs}")
        if result == 'clear' and self.ui is not None and hasattr(self.ui, 'celebrate'):
            self.ui.celebrate(self)
        self.real_frames += frames
        self.wall_total += wall
        if result == 'death':
            self.deaths += 1
        if result == 'clear' and self.clear_stats is None:
            self.clear_stats = {
                'clear_ep': self.episode,
                'real_frames': self.real_frames,
                'rollout_frames': self.planner.rollout_frames_total,
                'total_frames': self.total_emulator_frames,
                'wall_sec': self.wall_total,
                'deaths': self.deaths,
                'validated': validated,
            }
        return {'result': result, 'cause': cause, 'max_x': max_x,
                'frames': frames, 'wall': wall, 'validated': validated}

    # ------------------------------------------------------------------- UI

    def ui_rows(self):
        p = self.last_plan
        rows = [
            ('Plan', p.name if p else '—', None),
            ('Wave', p.wave if p else '—', None),
            ('Score', f"{p.score:.0f}" if p else '—', None),
            ('Candidates', p.n_candidates if p else '—', None),
            ('Plans total', self.planner.plans_total, None),
            ('Rollout frames', self.planner.rollout_frames_total, None),
            ('Hazards known', len(self.memory.hazards(self.stage)), None),
        ]
        if self.vision is not None:
            h = self.vision.hints()
            rows += [
                ('Mario vx (vision)', f"{h['mario_vx']:.2f}", None),
                ('Entities', len(h['enemies']), None),
                ('Gaps ahead', len(h['gaps']), None),
            ]
        return rows

    # ------------------------------------------------------------------ run

    def run(self, max_episodes: int = 50, stop_on_clear: bool = True,
            frame_budget: Optional[int] = None) -> dict:
        cleared = False
        aborted = False
        try:
            while self.episode < max_episodes:
                out = self.run_episode()
                if out['result'] == 'abort':
                    aborted = True
                    break
                if out['result'] == 'clear':
                    cleared = True
                    # Keep playing until we hold a *reproducible* recording.
                    if stop_on_clear and out.get('validated', True):
                        break
                if frame_budget is not None and self.total_emulator_frames >= frame_budget:
                    self.log(f"[{self.stage}] frame budget exhausted "
                             f"({self.total_emulator_frames}/{frame_budget})")
                    break
            # Flush any frames still waiting in the SMOOTH playback buffer so
            # the run's final moments are actually shown before closing.
            if not aborted and self.ui is not None and hasattr(self.ui, 'drain'):
                self.ui.drain()
        finally:
            self.session.close()
            self.shadow.close()
        return {
            'stage': self.stage,
            'cleared': cleared,
            'first_clear_episode': self.first_clear_episode,
            'episodes': self.episode,
            'best_x': self.best_x,
        }
