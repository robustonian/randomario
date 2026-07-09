#!/usr/bin/env python3
"""Fair PLN-vs-RND benchmark.

Both agents start from a completely fresh state (empty hazard DB / empty
archive) and get the same *emulator-frame* budget — not the same episode
count, because a planner "episode" hides thousands of rollout frames.

    uv run benchmark.py --stages 1-1 1-4 2-2 --reps 3 --budget 3000000
    uv run benchmark.py --mode cumulative --stages 1-1   # with learned DB

Per run we record:
    cleared, clear_ep, deaths_to_clear,
    real_frames / rollout_frames / total_emulator_frames to clear,
    wall_sec_to_clear, replayable (recording reproduces on a fresh replay),
    and the fresh/cumulative condition.

Results land in db/benchmark.sqlite — view them with:
    uv run bench_server.py    ->  http://localhost:8765
"""
import argparse
import multiprocessing as mp
import os
import sqlite3
import tempfile
import time

BENCH_DB = os.path.join('db', 'benchmark.sqlite')

_SCHEMA = """
CREATE TABLE IF NOT EXISTS bench_runs (
    id INTEGER PRIMARY KEY,
    agent TEXT NOT NULL,            -- 'planner' | 'random'
    stage TEXT NOT NULL,
    mode TEXT NOT NULL,             -- 'fresh' | 'cumulative'
    rep INTEGER,
    budget INTEGER,
    cleared INTEGER,
    clear_ep INTEGER,
    deaths_to_clear INTEGER,
    real_frames INTEGER,
    rollout_frames INTEGER,
    total_frames INTEGER,
    wall_sec REAL,
    replayable INTEGER,
    episodes_total INTEGER,
    created REAL
);
"""


def _open_db(path=BENCH_DB):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    db = sqlite3.connect(path, timeout=60)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA busy_timeout=60000")
    db.executescript(_SCHEMA)
    db.commit()
    return db


def _record(row: dict, db_path=BENCH_DB):
    db = _open_db(db_path)
    db.execute(
        "INSERT INTO bench_runs(agent, stage, mode, rep, budget, cleared, clear_ep,"
        " deaths_to_clear, real_frames, rollout_frames, total_frames, wall_sec,"
        " replayable, episodes_total, created)"
        " VALUES (:agent,:stage,:mode,:rep,:budget,:cleared,:clear_ep,:deaths,"
        ":real_frames,:rollout_frames,:total_frames,:wall_sec,:replayable,"
        ":episodes_total,:created)", row)
    db.commit()
    db.close()


def _replayable_random(stage: str, actions_name: str, path) -> int:
    from mario.env_utils import MarioSession
    from mario.actions import ACTION_SETS
    s = MarioSession(stage, ACTION_SETS[actions_name])
    s.reset()
    flag = False
    for a in path:
        obs, r, done, info = s.step(a)
        if done:
            flag = bool(info.get('flag_get'))
            break
    s.close()
    return int(flag)


def run_job(job):
    agent_name, stage, mode, rep, budget, db_path = job
    os.makedirs('logs', exist_ok=True)
    log_path = os.path.join('logs', f'bench_{agent_name}_{stage}_{mode}_r{rep}.log')
    t0 = time.time()
    with open(log_path, 'a', buffering=1) as logf:
        def log(msg):
            logf.write(msg + '\n')

        row = dict(agent=agent_name, stage=stage, mode=mode, rep=rep, budget=budget,
                   cleared=0, clear_ep=None, deaths=None, real_frames=None,
                   rollout_frames=None, total_frames=None, wall_sec=None,
                   replayable=None, episodes_total=0, created=time.time())

        if agent_name == 'planner':
            from mario.agent_planner import PlannerAgent
            from mario.memory import PlannerMemory, PLANNER_DB_PATH
            from mario.vision import VisionTracker
            if mode == 'fresh':
                fd, mem_path = tempfile.mkstemp(suffix='.sqlite', prefix='bench_pln_')
                os.close(fd)
            else:
                mem_path = PLANNER_DB_PATH
            agent = PlannerAgent(stage, memory=PlannerMemory(mem_path),
                                 vision=VisionTracker(), log=log)
            agent.run(max_episodes=100000, stop_on_clear=True, frame_budget=budget)
            cs = agent.clear_stats
            row['episodes_total'] = agent.episode
            if cs:
                row.update(cleared=1, clear_ep=cs['clear_ep'], deaths=cs['deaths'],
                           real_frames=cs['real_frames'],
                           rollout_frames=cs['rollout_frames'],
                           total_frames=cs['total_frames'], wall_sec=cs['wall_sec'],
                           replayable=int(bool(cs['validated'])))
            else:
                row.update(deaths=agent.deaths, real_frames=agent.real_frames,
                           rollout_frames=agent.planner.rollout_frames_total,
                           total_frames=agent.total_emulator_frames,
                           wall_sec=time.time() - t0)
            if mode == 'fresh':
                for suf in ('', '-wal', '-shm'):
                    try:
                        os.remove(mem_path + suf)
                    except OSError:
                        pass
        else:
            from mario.agent_explore import GoExploreAgent
            actions_name = 'right_only'
            if mode == 'fresh':
                fd, arch_path = tempfile.mkstemp(suffix='.pkl', prefix='bench_rnd_')
                os.close(fd)
                os.remove(arch_path)  # agent expects "missing" for a fresh start
            else:
                from mario.agent_explore import default_archive_path
                arch_path = default_archive_path(stage, actions_name)
            agent = GoExploreAgent(stage, archive_path=arch_path,
                                   actions_name=actions_name, log=log)
            agent.run(max_episodes=1000000, frame_budget=budget, stop_on_clear=True)
            cs = agent.clear_stats
            row['episodes_total'] = agent.episode
            if cs:
                row.update(cleared=1, clear_ep=cs['clear_ep'], deaths=cs['deaths'],
                           real_frames=cs['real_frames'], rollout_frames=0,
                           total_frames=cs['total_frames'], wall_sec=cs['wall_sec'],
                           replayable=_replayable_random(
                               stage, actions_name, agent.best_overall_path))
            else:
                row.update(deaths=agent.deaths,
                           real_frames=agent.session.frames_emulated,
                           rollout_frames=0,
                           total_frames=agent.total_emulator_frames,
                           wall_sec=time.time() - t0)
            if mode == 'fresh':
                for p in (arch_path, arch_path + '.tmp'):
                    try:
                        os.remove(p)
                    except OSError:
                        pass

        _record(row, db_path)
        log(f"BENCH RESULT: {row}")
    return row


def main():
    p = argparse.ArgumentParser(description='Fair planner-vs-random benchmark')
    p.add_argument('--stages', nargs='+', default=['1-1'])
    p.add_argument('--reps', type=int, default=3)
    p.add_argument('--budget', type=int, default=3_000_000,
                   help='Emulator-frame budget per run (default 3M ≈ 50min game time)')
    p.add_argument('--mode', choices=['fresh', 'cumulative'], default='fresh')
    p.add_argument('--agents', nargs='+', choices=['planner', 'random'],
                   default=['planner', 'random'])
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--db', default=BENCH_DB)
    args = p.parse_args()

    jobs = [(a, s, args.mode, r, args.budget, args.db)
            for s in args.stages for r in range(1, args.reps + 1)
            for a in args.agents]
    print(f"{len(jobs)} runs ({len(args.stages)} stages x {args.reps} reps x "
          f"{len(args.agents)} agents, mode={args.mode}, budget={args.budget:,})")
    t0 = time.time()
    ctx = mp.get_context('spawn')
    with ctx.Pool(args.workers) as pool:
        results = pool.map(run_job, jobs)

    ok = sum(r['cleared'] for r in results)
    print(f"\nDone: {ok}/{len(results)} runs cleared in {(time.time()-t0)/60:.1f} min")
    print(f"View: uv run bench_server.py  ->  http://localhost:8765")


if __name__ == '__main__':
    main()
