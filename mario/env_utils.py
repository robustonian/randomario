"""Environment helpers: direct env construction, savestates and fast rollouts.

The stock ``gym.make`` path stacks TimeLimit / api-compatibility / checker
wrappers that slow every frame down and hide the underlying
``SuperMarioBrosEnv``.  We instantiate the env directly and expose:

- ``hard_reset``   : full power-cycle reset that does not depend on the
                     single nes-py backup slot (the planner reuses that slot
                     for lookahead rollouts).
- ``snapshot``     : save the full emulator state + python-side mirrors.
- ``rollout_step`` : advance one frame without gym bookkeeping and read the
                     outcome straight from RAM (~2x faster than env.step and
                     side-effect free for the gym layer).
"""
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np

warnings.filterwarnings('ignore', category=UserWarning, module='gym')
warnings.filterwarnings('ignore', category=DeprecationWarning)

from gym_super_mario_bros import SuperMarioBrosEnv
from nes_py.wrappers import JoypadSpace
import nes_py.nes_env as _nes


def parse_stage(stage: str) -> Tuple[int, int]:
    world, num = stage.split('-')
    return int(world), int(num)


ALL_STAGES = [f"{w}-{s}" for w in range(1, 9) for s in range(1, 5)]
# Stages containing wrap-around maze checkpoints, excluded from the planner
# verification scope. (These are the true loop stages of SMB; 6-4 is a normal
# castle and clears fine. 7-4 is attempted best-effort with lane learning.)
LOOP_STAGES = ['4-4', '7-4', '8-4']


class RolloutState:
    """Lightweight per-frame readout used inside lookahead rollouts."""

    __slots__ = ('x', 'y', 'dead', 'flag', 'fell', 'world_changed')

    def __init__(self, x, y, dead, flag, fell, world_changed):
        self.x = x
        self.y = y
        self.dead = dead
        self.flag = flag
        self.fell = fell
        self.world_changed = world_changed


class MarioSession:
    """Owns the raw env + joypad mapping and the savestate machinery."""

    def __init__(self, stage: str, actions: List[List[str]]):
        self.stage = stage
        self.world, self.stage_num = parse_stage(stage)
        self.smb = SuperMarioBrosEnv(target=(self.world, self.stage_num))
        self.env = JoypadSpace(self.smb, actions)
        self.actions = actions
        # Pre-resolved controller bytes for the fast rollout path.
        self.action_bytes = [self.env._action_map[i] for i in range(len(actions))]
        self._saved_mirrors = None
        # Every emulated frame this session ever ran (real steps AND rollout
        # steps) — the honest "compute spent" unit for fair benchmarking.
        self.frames_emulated = 0

    # ------------------------------------------------------------------ reset

    def reset(self) -> Tuple[np.ndarray, dict]:
        """Full reset to the stage start, independent of the backup slot."""
        smb = self.smb
        smb._has_backup = False
        smb._will_reset()
        _nes._LIB.Reset(smb._env)
        # A console reset keeps RAM contents. Leftover state (in-game timer,
        # SMB's PRNG registers, demo timers) would make the start-screen skip
        # take a history-dependent number of frames, so episodes would not be
        # reproducible in isolation (broke replay on 8-1). Zero all RAM for a
        # canonical cold-boot start every episode.
        smb.ram[:] = 0
        smb._did_reset()
        smb._skip_start_screen()
        smb._did_reset()
        smb.done = False
        return smb.screen, smb._get_info()

    # ------------------------------------------------------------- real steps

    def step(self, action_idx: int):
        """Regular gym step (dying shortcut, area skips, info dict)."""
        obs, reward, done, info = self.env.step(action_idx)
        self.frames_emulated += 1
        return obs, reward, done, info

    # -------------------------------------------------------------- snapshots

    def snapshot(self):
        smb = self.smb
        self._saved_mirrors = (smb.done, smb._time_last, smb._x_position_last)
        smb._backup()

    def restore(self):
        smb = self.smb
        smb._restore()
        smb.done, smb._time_last, smb._x_position_last = self._saved_mirrors

    # ----------------------------------------------------------- fast rollout

    def rollout_step(self, action_idx: int) -> RolloutState:
        smb = self.smb
        smb._frame_advance(self.action_bytes[action_idx])
        self.frames_emulated += 1
        ram = smb.ram
        player_state = ram[0x000e]
        y_viewport = ram[0x00b5]
        dead = player_state == 0x06 or player_state == 0x0b or y_viewport > 1
        world_changed = (ram[0x075f] + 1 != self.world) or (ram[0x075c] + 1 != self.stage_num)
        return RolloutState(
            x=int(ram[0x6d]) * 0x100 + int(ram[0x86]),
            y=smb._y_position,
            dead=bool(dead),
            flag=bool(smb._flag_get),
            fell=bool(y_viewport > 1),
            world_changed=bool(world_changed),
        )

    # ------------------------------------------------------------- inspection

    @property
    def x_pos(self) -> int:
        return int(self.smb._x_position)

    @property
    def y_pos(self) -> int:
        return int(self.smb._y_position)

    @property
    def time_left(self) -> int:
        return int(self.smb._time)

    @property
    def is_swimming(self) -> bool:
        # SMB RAM $0704: 1 while Mario is in water physics.
        return bool(self.smb.ram[0x0704] == 1)

    @property
    def frame(self) -> np.ndarray:
        return self.smb.screen

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass
