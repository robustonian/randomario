# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RandoMario is a Super Mario Bros. automation project built on `gym-super-mario-bros` (nes-py emulator). It contains two agents whose data stores are deliberately separate so they can be compared:

1. **Planner agent** (`mario/agent_planner.py`) — model-based: estimates world state from recent frames (vision), generates candidate input plans for the next few dozen frames, evaluates them with savestate-based lookahead rollouts on the emulator, and executes the best. Deaths are analyzed (pit / enemy / contact / timeout / loop / stuck) and stored per stage in a hazard memory so the same failure is not repeated. **No stage-specific hardcoding** — the same logic clears unseen stages. Results/learning live in `db/planner.sqlite`.
2. **Go-Explore agent** (`mario/agent_explore.py`) — the random baseline (formerly `test.py`): cell-archive based return-then-explore. Archives live in `pkl/go_explore_archive_{stage}_{actions}.pkl`.

Loop stages **4-4, 7-4, 8-4** are excluded from planner verification scope (SMB's wrap-around maze checkpoints; 6-4 is a normal castle and clears fine). 7-4 is still attempted best-effort via loop detection + lane-profile learning.

## Architecture

```
mario/
├── env_utils.py      # MarioSession: direct SuperMarioBrosEnv construction (no gym.make
│                     # wrapper overhead), hard reset independent of the nes-py backup slot,
│                     # snapshot()/restore() savestates, rollout_step() fast frame advance
├── actions.py        # Action sets incl. PLANNER_MOVEMENT superset; get_action_indices()
├── vision.py         # VisionTracker: scroll-compensated frame differencing, blob tracking
│                     # (position/velocity/acceleration), Mario identification, gap detection
├── planner.py        # Planner: candidate waves (1=fast, 2=fine sweep, 3=long waits/retreat),
│                     # swim candidates in water, vision-seeded gap/stomp candidates,
│                     # rollout scoring (progress + flag bonus - death/warp penalty)
├── memory.py         # PlannerMemory: SQLite (WAL) hazards, plan blacklist, episodes, clears
├── agent_planner.py  # PlannerAgent: episode loop, stall/loop/warp detection, death analysis
├── agent_explore.py  # GoExploreAgent + Cell + CellUnpickler (loads legacy __main__.Cell pickles)
└── ui.py             # GameUI: shared pygame UI, runtime speed control (1x..8x/MAX),
                      # vision overlay, controller visualization
play.py               # unified CLI entry point (--agent planner|explore)
verify.py             # parallel headless verification across stages (multiprocessing)
progress_viewer.py    # comparison dashboard (tkinter, stdlib-only: run with system python3)
```

### Key technical points

- **Savestates**: nes-py has a single C++ backup slot (`_backup`/`_restore`). The planner snapshots at each replan point, rolls candidates out, restores. Because the slot gets reused, `MarioSession.reset()` performs a **hard reset** (console reset + `_skip_start_screen`) instead of relying on the slot. After a console reset the NES RAM persists, so the stale timer must be cleared (`ram[0x07f8:0x07fb] = 0`) or the stage-select writes get skipped and 1-1 loads regardless of target.
- **Fast rollouts** use `smb._frame_advance()` + direct RAM reads (`RolloutState`), skipping gym bookkeeping — ~950fps/core.
- **Replan cadence**: a plan that survives its full horizon executes `EXEC_CLEAN` (35) frames before replanning; risky plans replan every 10 frames. Near known hazards the horizon is extended (70 → 120/170 frames) and finer candidate waves engage immediately.
- **Speed options never change emulation** — only frame pacing and draw frequency (keys 1-5, +/-; MAX draws at most ~30fps wall-clock).
- **Water detection** via RAM `$0704` swim flag switches the planner to stroke-tap candidates.
- **Warp/loop guards**: rollouts treat a world/stage change as a heavy penalty (avoids warp zones); real-play x jump-backs (>150px, same area) are recorded as 'loop' hazards.

## Development Commands

```bash
uv sync

# Planner (model-based) agent
uv run play.py --agent planner --stage 1-1              # UI, 1x
uv run play.py --agent planner --stage 8-1 --speed max  # UI, fast-forward
uv run play.py --agent planner --stage 2-2 --headless   # no UI, max speed
uv run play.py --agent planner --stage 1-1 --no-vision  # rollout-only planning

# Go-Explore (random baseline)
uv run play.py --agent explore --stage 1-1 --actions right_only
uv run test.py --stage 1-1                              # legacy CLI (thin wrapper)

# Verify all target stages in parallel (headless)
uv run verify.py                                        # 29 stages, logs/ + db/planner.sqlite
uv run verify.py --stages 8-1 7-4 --episodes 60 --workers 8

# Comparison dashboard (random pkl vs planner sqlite; stdlib only)
python3 progress_viewer.py
```

Note: the uv-managed CPython 3.8 has a broken Tcl/Tk setup on this machine — run the dashboard with **system** `python3` (it is dependency-free on purpose).

### Runtime keys (pygame UI)

`1-5` speed presets · `+/-` speed step · `P` pause · `V` vision overlay · `R` reset episode · `ESC` quit

## Testing / Validation

```bash
uv run python -c "import mario.planner, mario.vision, mario.ui; print('ok')"   # import check
timeout 60 uv run play.py --agent planner --stage 1-1 --headless --episodes 1  # 1-1 clears in ~30s
uv run verify.py --stages 1-1 1-4 2-2 --episodes 5 --workers 3                 # smoke multi-type
```

There is no formal test framework; validation is via headless runs. 1-1, 1-2, 1-4 (castle), 2-2 (water) are known to clear on episode 1.

## Data / Storage

- `db/planner.sqlite` (gitignored): tables `hazards`, `blacklist`, `episodes`, `clears`. WAL mode for parallel verification workers.
- `pkl/*.pkl`: Go-Explore archives. Legacy files pickled the Cell class as `__main__.Cell` / `test.Cell`; always load through `mario.agent_explore.load_archive()` (CellUnpickler) — plain `pickle.load` breaks.
- `logs/verify_{stage}.log`: per-stage verification logs.

## Gotchas

- `env.step` raises `ValueError: cannot step in a done environment` — always check `done` and reset via `MarioSession.reset()` (never `env.reset()`, which would restore the reused backup slot mid-level).
- A fresh jump requires releasing A first; the planner inserts a release frame automatically (`_release_guard`) when consecutive plans both hold A.
- gym 0.26 API-compat wrappers are intentionally bypassed; `MarioSession.step` returns the classic 4-tuple.
- Frames returned by the env are views into the emulator screen buffer — `.copy()` before storing (vision stores grayscale copies).
