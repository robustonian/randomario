"""Death analysis memory and result storage for the planner agent.

Everything lives in ``db/planner.sqlite`` — deliberately separate from the
Go-Explore (random) agent's ``pkl/`` archives so the two approaches can be
compared side by side.
"""
import json
import os
import sqlite3
import time
from typing import Dict, List, Optional

PLANNER_DB_PATH = os.path.join('db', 'planner.sqlite')

_SCHEMA = """
CREATE TABLE IF NOT EXISTS hazards (
    id INTEGER PRIMARY KEY,
    stage TEXT NOT NULL,
    x INTEGER NOT NULL,
    cause TEXT NOT NULL,
    detail TEXT DEFAULT '',
    hits INTEGER DEFAULT 1,
    created REAL,
    updated REAL
);
CREATE INDEX IF NOT EXISTS idx_hazards_stage_x ON hazards(stage, x);
CREATE TABLE IF NOT EXISTS blacklist (
    id INTEGER PRIMARY KEY,
    stage TEXT NOT NULL,
    x_bin INTEGER NOT NULL,
    sig TEXT NOT NULL,
    created REAL,
    UNIQUE(stage, x_bin, sig)
);
CREATE TABLE IF NOT EXISTS episodes (
    id INTEGER PRIMARY KEY,
    stage TEXT NOT NULL,
    run_id TEXT,
    ep INTEGER,
    result TEXT,
    cause TEXT,
    max_x INTEGER,
    frames INTEGER,
    game_time_left INTEGER,
    wall_sec REAL,
    created REAL
);
CREATE TABLE IF NOT EXISTS clears (
    id INTEGER PRIMARY KEY,
    stage TEXT NOT NULL,
    run_id TEXT,
    ep INTEGER,
    frames INTEGER,
    wall_sec REAL,
    game_time_left INTEGER,
    created REAL
);
"""

# How close (in world px) two deaths must be to count as the same hazard.
HAZARD_MERGE_RADIUS = 32
BLACKLIST_X_BIN = 24


class PlannerMemory:
    """Per-stage hazard knowledge + episode results, SQLite backed."""

    def __init__(self, db_path: str = PLANNER_DB_PATH):
        os.makedirs(os.path.dirname(db_path) or '.', exist_ok=True)
        self.db = sqlite3.connect(db_path, timeout=30.0)
        # WAL lets the parallel verification workers share one DB file.
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA busy_timeout=30000")
        self.db.executescript(_SCHEMA)
        self.db.commit()
        self._hazard_cache: Dict[str, List[dict]] = {}
        self._blacklist_cache: Dict[str, set] = {}

    # ---------------------------------------------------------------- hazards

    def record_death(self, stage: str, x: int, cause: str, detail: dict = None):
        detail_json = json.dumps(detail or {}, ensure_ascii=False)
        now = time.time()
        row = self.db.execute(
            "SELECT id, hits FROM hazards WHERE stage=? AND cause=? AND ABS(x-?)<=? "
            "ORDER BY ABS(x-?) LIMIT 1",
            (stage, cause, x, HAZARD_MERGE_RADIUS, x)).fetchone()
        if row:
            self.db.execute("UPDATE hazards SET hits=?, detail=?, updated=? WHERE id=?",
                            (row[1] + 1, detail_json, now, row[0]))
        else:
            self.db.execute(
                "INSERT INTO hazards(stage, x, cause, detail, created, updated) VALUES (?,?,?,?,?,?)",
                (stage, x, cause, detail_json, now, now))
        self.db.commit()
        self._hazard_cache.pop(stage, None)

    def hazards(self, stage: str) -> List[dict]:
        if stage not in self._hazard_cache:
            rows = self.db.execute(
                "SELECT x, cause, hits FROM hazards WHERE stage=?", (stage,)).fetchall()
            self._hazard_cache[stage] = [
                {'x': r[0], 'cause': r[1], 'hits': r[2]} for r in rows]
        return self._hazard_cache[stage]

    def hazards_in(self, stage: str, x_from: int, x_to: int) -> List[dict]:
        return [h for h in self.hazards(stage) if x_from <= h['x'] <= x_to]

    # -------------------------------------------------------------- blacklist

    def blacklist_plan(self, stage: str, x: int, sig: str):
        try:
            self.db.execute(
                "INSERT OR IGNORE INTO blacklist(stage, x_bin, sig, created) VALUES (?,?,?,?)",
                (stage, x // BLACKLIST_X_BIN, sig, time.time()))
            self.db.commit()
        finally:
            self._blacklist_cache.pop(stage, None)

    def is_blacklisted(self, stage: str, x: int, sig: str) -> bool:
        if stage not in self._blacklist_cache:
            rows = self.db.execute(
                "SELECT x_bin, sig FROM blacklist WHERE stage=?", (stage,)).fetchall()
            self._blacklist_cache[stage] = {(r[0], r[1]) for r in rows}
        return (x // BLACKLIST_X_BIN, sig) in self._blacklist_cache[stage]

    # ---------------------------------------------------------------- results

    def record_episode(self, stage: str, run_id: str, ep: int, result: str, cause: str,
                       max_x: int, frames: int, game_time_left: int, wall_sec: float):
        self.db.execute(
            "INSERT INTO episodes(stage, run_id, ep, result, cause, max_x, frames,"
            " game_time_left, wall_sec, created) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (stage, run_id, ep, result, cause, max_x, frames, game_time_left,
             wall_sec, time.time()))
        if result == 'clear':
            self.db.execute(
                "INSERT INTO clears(stage, run_id, ep, frames, wall_sec, game_time_left, created)"
                " VALUES (?,?,?,?,?,?,?)",
                (stage, run_id, ep, frames, wall_sec, game_time_left, time.time()))
        self.db.commit()

    def stage_summary(self, stage: str) -> dict:
        ep_row = self.db.execute(
            "SELECT COUNT(*), MAX(max_x) FROM episodes WHERE stage=?", (stage,)).fetchone()
        clear_row = self.db.execute(
            "SELECT MIN(ep), COUNT(*), MIN(frames) FROM clears WHERE stage=?", (stage,)).fetchone()
        return {
            'episodes': ep_row[0] or 0,
            'best_x': ep_row[1] or 0,
            'first_clear_ep': clear_row[0],
            'clears': clear_row[1] or 0,
            'best_frames': clear_row[2],
        }

    def close(self):
        try:
            self.db.close()
        except Exception:
            pass
