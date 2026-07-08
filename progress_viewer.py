#!/usr/bin/env python3
"""Mario progress dashboard — Random (Go-Explore) vs Planner (model-based).

Shows, for every stage 1-1 … 8-4, how each agent is doing side by side:

    ✅ EP3          cleared, first clear at episode 3
    🔄 x1520/EP12   still trying: best x so far / episodes spent
    ⚫ —            not attempted

Data sources (kept deliberately separate so the approaches are comparable):
- Random agent : pkl/go_explore_archive_{stage}_{actions}.pkl
- Planner agent: db/planner.sqlite

The window watches both for changes and refreshes automatically.
"""
import os
import pickle
import sqlite3
import time
import tkinter as tk
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Deliberately self-contained (no mario/ imports): runs on plain system
# python with tkinter, no gym/numpy needed.
ALL_STAGES = [f"{w}-{s}" for w in range(1, 9) for s in range(1, 5)]
LOOP_STAGES = ['4-4', '7-4', '8-4']  # SMB's wrap-around maze stages
PLANNER_DB_PATH = os.path.join('db', 'planner.sqlite')
ACTION_SET_FALLBACKS = ['right_only', 'complex', 'simple', 'planner']


@dataclass
class Cell:
    cell_id: Tuple[int, int, str]
    path: List[int] = field(default_factory=list)
    max_x: int = 0
    visits: int = 0
    created_at: float = field(default_factory=time.time)


class _CellUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'Cell':
            return Cell
        return super().find_class(module, name)


def load_archive(path: str) -> dict:
    with open(path, 'rb') as f:
        return _CellUnpickler(f).load()

# ---------------------------------------------------------------- theme
BG = '#101218'
PANEL = '#1a1e28'
PANEL_LIGHT = '#232838'
TEXT = '#e2e8f0'
TEXT_DIM = '#828ca0'
GREEN = '#4ade80'
ORANGE = '#fbbf24'
GRAY = '#4b5563'
BLUE = '#60a5fa'
FONT = ('Segoe UI', 10)
FONT_SMALL = ('Segoe UI', 9)
FONT_BOLD = ('Segoe UI', 11, 'bold')
FONT_TITLE = ('Segoe UI', 15, 'bold')


class StageStatus:
    def __init__(self, cleared: bool = False, first_clear_ep: Optional[int] = None,
                 episodes: int = 0, best_x: int = 0, attempted: bool = False):
        self.cleared = cleared
        self.first_clear_ep = first_clear_ep
        self.episodes = episodes
        self.best_x = best_x
        self.attempted = attempted

    @property
    def text(self) -> str:
        if self.cleared:
            return f"✅ EP{self.first_clear_ep}"
        if self.attempted:
            return f"🔄 x{self.best_x}/EP{self.episodes}"
        return "—"

    @property
    def color(self) -> str:
        if self.cleared:
            return GREEN
        if self.attempted:
            return ORANGE
        return TEXT_DIM


_random_cache: Dict[str, tuple] = {}  # path -> (mtime, StageStatus)


def read_random_status(stage: str) -> StageStatus:
    for actions in ACTION_SET_FALLBACKS:
        path = os.path.join('pkl', f'go_explore_archive_{stage}_{actions}.pkl')
        if not os.path.exists(path):
            continue
        mtime = os.path.getmtime(path)
        cached = _random_cache.get(path)
        if cached and cached[0] == mtime:
            return cached[1]
        try:
            d = load_archive(path)
            first = d.get('first_clear_episode')
            status = StageStatus(cleared=first is not None, first_clear_ep=first,
                                 episodes=d.get('episodes_done', 0),
                                 best_x=d.get('best_overall_x', 0), attempted=True)
        except Exception:
            status = StageStatus(attempted=True)
        _random_cache[path] = (mtime, status)
        return status
    return StageStatus()


def read_planner_statuses(db_path: str = PLANNER_DB_PATH) -> Dict[str, StageStatus]:
    out: Dict[str, StageStatus] = {}
    if not os.path.exists(db_path):
        return out
    try:
        db = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
        eps = dict()
        for stage, n, best in db.execute(
                "SELECT stage, COUNT(*), MAX(max_x) FROM episodes GROUP BY stage"):
            eps[stage] = (n, best or 0)
        clears = dict()
        for stage, ep in db.execute("SELECT stage, MIN(ep) FROM clears GROUP BY stage"):
            clears[stage] = ep
        db.close()
        for stage, (n, best) in eps.items():
            first = clears.get(stage)
            out[stage] = StageStatus(cleared=first is not None, first_clear_ep=first,
                                     episodes=n, best_x=best, attempted=True)
        for stage, ep in clears.items():
            if stage not in out:
                out[stage] = StageStatus(cleared=True, first_clear_ep=ep, attempted=True)
    except Exception:
        pass
    return out


def read_replay_statuses(db_path: str = PLANNER_DB_PATH) -> Dict[str, int]:
    """Stage -> clear episode of the recorded (replayable) run: the run that
    `play.py --agent replay` will show, i.e. episodes 1..N ending in a clear
    with verified input recordings."""
    out: Dict[str, int] = {}
    if not os.path.exists(db_path):
        return out
    try:
        db = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
        # SQLite bare-column + MAX(): ep comes from the newest matching row.
        for stage, ep, _ in db.execute(
                "SELECT stage, ep, MAX(created) FROM episodes"
                " WHERE result='clear' AND actions IS NOT NULL GROUP BY stage"):
            out[stage] = ep
        db.close()
    except Exception:
        pass
    return out


class Dashboard:
    REFRESH_MS = 2000

    def __init__(self):
        self.root = tk.Tk()
        self.root.title('RandoMario — Random vs Planner Dashboard')
        self.root.configure(bg=BG)
        self.root.geometry('1240x600')
        self.cells: Dict[str, dict] = {}
        self.summary_var = tk.StringVar()
        self._build()
        self._schedule()

    def _build(self):
        tk.Label(self.root, text='🍄 Super Mario Bros. — Random vs Planner',
                 font=FONT_TITLE, bg=BG, fg=TEXT).pack(pady=(14, 2))
        tk.Label(self.root,
                 text='RND = Go-Explore(random)   PLN = planner first clear   '
                      'RPL = recorded replayable run (episodes to clear, --agent replay)   '
                      '✅ cleared   🔄 best x / episodes   loop stages excluded for PLN',
                 font=FONT_SMALL, bg=BG, fg=TEXT_DIM).pack(pady=(0, 10))

        grid = tk.Frame(self.root, bg=BG)
        grid.pack(expand=True, fill='both', padx=14)

        for w in range(8):
            grid.columnconfigure(w, weight=1, uniform='world')
        for w in range(1, 9):
            wf = tk.Frame(grid, bg=PANEL, highlightbackground=PANEL_LIGHT,
                          highlightthickness=1)
            wf.grid(row=0, column=w - 1, padx=3, pady=2, sticky='nsew')
            tk.Label(wf, text=f'WORLD {w}', font=FONT_BOLD, bg=PANEL,
                     fg=BLUE).pack(pady=(6, 2))
            for s in range(1, 5):
                stage = f'{w}-{s}'
                cf = tk.Frame(wf, bg=PANEL_LIGHT)
                cf.pack(fill='x', padx=5, pady=3)
                head = stage + ('  (loop)' if stage in LOOP_STAGES else '')
                tk.Label(cf, text=head, font=FONT_BOLD, bg=PANEL_LIGHT,
                         fg=TEXT).pack(anchor='w', padx=6, pady=(3, 0))
                rows = {}
                for key, label in (('random', 'RND'), ('planner', 'PLN'),
                                   ('replay', 'RPL')):
                    rf = tk.Frame(cf, bg=PANEL_LIGHT)
                    rf.pack(fill='x', padx=6)
                    tk.Label(rf, text=label, font=FONT_SMALL, bg=PANEL_LIGHT,
                             fg=TEXT_DIM, width=4, anchor='w').pack(side='left')
                    val = tk.Label(rf, text='—', font=FONT_SMALL, bg=PANEL_LIGHT,
                                   fg=TEXT_DIM, anchor='e')
                    val.pack(side='right')
                    rows[key] = val
                tk.Frame(cf, bg=PANEL_LIGHT, height=3).pack()
                self.cells[stage] = rows

        bottom = tk.Frame(self.root, bg=BG)
        bottom.pack(fill='x', padx=14, pady=(6, 12))
        tk.Button(bottom, text='↻ Refresh', font=FONT, bg=PANEL_LIGHT, fg=TEXT,
                  relief='flat', padx=12, pady=2,
                  activebackground=BLUE, command=self.refresh).pack(side='left')
        tk.Label(bottom, textvariable=self.summary_var, font=FONT, bg=BG,
                 fg=TEXT_DIM, anchor='e').pack(side='right')

    def refresh(self):
        planner = read_planner_statuses()
        replay = read_replay_statuses()
        rnd_cleared = pln_cleared = rpl_ready = 0
        target = [s for s in ALL_STAGES if s not in LOOP_STAGES]
        for stage in ALL_STAGES:
            rnd = read_random_status(stage)
            pln = planner.get(stage, StageStatus())
            rpl_ep = replay.get(stage)
            self.cells[stage]['random'].config(text=rnd.text, fg=rnd.color)
            self.cells[stage]['planner'].config(text=pln.text, fg=pln.color)
            if rpl_ep is not None:
                self.cells[stage]['replay'].config(text=f'🎬 EP{rpl_ep}', fg=BLUE)
            else:
                self.cells[stage]['replay'].config(text='—', fg=TEXT_DIM)
            if stage in target:
                rnd_cleared += rnd.cleared
                pln_cleared += pln.cleared
                rpl_ready += rpl_ep is not None
        self.summary_var.set(
            f'target stages ({len(target)}): random {rnd_cleared}/{len(target)} · '
            f'planner {pln_cleared}/{len(target)} · '
            f'replayable {rpl_ready}/{len(target)} · updated {time.strftime("%H:%M:%S")}')

    def _schedule(self):
        self.refresh()
        self.root.after(self.REFRESH_MS, self._schedule)

    def run(self):
        self.root.mainloop()


if __name__ == '__main__':
    Dashboard().run()
