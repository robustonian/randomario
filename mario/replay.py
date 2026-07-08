"""Replay agent: re-run a recorded planner run without any planning.

The planner records every executed input per episode (the NES is
deterministic, so replaying the inputs from a hard reset reproduces the
episode exactly). This agent plays episode 1 through the first clear of the
most recent successful run at a steady frame rate — speed keys work as
usual — and ends with the clear celebration.
"""
from typing import List, Optional

from .actions import ACTION_SETS
from .env_utils import MarioSession
from .memory import PlannerMemory


class ReplayAgent:
    def __init__(self, stage: str, memory: Optional[PlannerMemory] = None,
                 ui=None, run_id: Optional[str] = None, video=None, log=print):
        self.stage = stage
        self.memory = memory if memory is not None else PlannerMemory()
        self.ui = ui
        self.video = video  # optional mario.video.VideoRecorder
        self.log = log
        self.episodes: List[dict] = self.memory.replay_episodes(stage, run_id)
        self.session = MarioSession(stage, ACTION_SETS['planner'])
        self.vision = None  # no overlay in replay

        # Live state for the shared UI
        self.episode = 0
        self.best_x = 0
        self.last_info: dict = {}
        self.last_frame = None
        self.first_clear_episode = next(
            (e['ep'] for e in self.episodes if e['result'] == 'clear'), None)
        self._current: Optional[dict] = None
        self._frame_idx = 0

    def ui_rows(self):
        cur = self._current or {}
        total = len(cur.get('actions', []))
        return [
            ('Mode', 'REPLAY', None),
            ('Run', cur.get('run_id', '—'), None),
            ('Episode result', f"{cur.get('result', '—')}({cur.get('cause', '')})", None),
            ('Frame', f"{self._frame_idx}/{total}", None),
            ('Episodes in run', len(self.episodes), None),
            ('Clear episode', self.first_clear_episode or '—', None),
        ]

    def run(self, max_episodes: int = 0) -> dict:
        if not self.episodes:
            self.log(f"[{self.stage}] No recorded run with a clear found in the "
                     f"planner DB. Run the planner first: "
                     f"uv run play.py --agent planner --stage {self.stage}")
            return {'stage': self.stage, 'cleared': False,
                    'first_clear_episode': None, 'episodes': 0, 'best_x': 0}

        quit_requested = False
        for epi in self.episodes:
            if quit_requested:
                break
            self._current = epi
            self.episode = epi['ep']
            self._frame_idx = 0
            obs, info = self.session.reset()
            self.last_frame, self.last_info = obs, info
            self.log(f"[{self.stage}] replaying ep{epi['ep']} "
                     f"({epi['result']}, {len(epi['actions'])} frames)")

            flag_seen = False
            for action in epi['actions']:
                obs, reward, done, info = self.session.step(action)
                self._frame_idx += 1
                self.last_frame, self.last_info = obs, info
                self.best_x = max(self.best_x, int(info.get('x_pos', 0)))
                flag_seen = flag_seen or bool(info.get('flag_get'))
                if self.video is not None:
                    self.video.add(obs)
                if self.ui is not None:
                    cmd = self.ui.on_frame(self, action)
                    if cmd == 'quit':
                        quit_requested = True
                        break
                    if cmd == 'reset':  # skip to the next episode
                        break
                if done:
                    break

            if self.video is not None and not quit_requested:
                if flag_seen:
                    # Keep the emulator rolling past `done` so the video gets
                    # the flag slide, the walk into the castle and fireworks.
                    smb = self.session.smb
                    for _ in range(420):
                        smb._frame_advance(0)
                        self.video.add(smb.screen)
                elif self.last_frame is not None:
                    # Hold the death frame briefly between episodes.
                    self.video.add(self.last_frame, repeat=45)

            if epi['result'] == 'clear' and not flag_seen and not quit_requested:
                # Should not happen for recordings made after replay
                # validation was added; be loud rather than celebrate a lie.
                self.log(f"[{self.stage}] WARNING: recorded clear (ep{epi['ep']}) "
                         f"did not reproduce on replay")
            if not quit_requested and flag_seen \
                    and self.ui is not None and hasattr(self.ui, 'celebrate'):
                self.ui.celebrate(self, seconds=5.0)

        if self.video is not None:
            self.video.close()
            self.log(f"[{self.stage}] video saved: {self.video.path} "
                     f"({self.video.frames} frames, "
                     f"{self.video.frames / self.video.fps:.1f}s)")
        self.session.close()
        return {
            'stage': self.stage,
            'cleared': self.first_clear_episode is not None,
            'first_clear_episode': self.first_clear_episode,
            'episodes': len(self.episodes),
            'best_x': self.best_x,
        }
