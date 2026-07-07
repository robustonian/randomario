#!/usr/bin/env python3
"""Parallel headless verification of the planner agent across stages.

Runs the model-based planner on every (non-loop) stage in worker processes
until each stage is cleared or the episode budget runs out, then prints a
summary table. All results land in db/planner.sqlite (WAL) so the dashboard
can watch progress live:

    uv run verify.py                     # all 29 target stages
    uv run verify.py --stages 8-1 7-4    # specific stages
    uv run verify.py --episodes 60 --workers 16
"""
import argparse
import multiprocessing as mp
import os
import time

from mario.env_utils import ALL_STAGES, LOOP_STAGES


def target_stages():
    return [s for s in ALL_STAGES if s not in LOOP_STAGES]


def run_stage(args):
    stage, episodes, db_path, max_frames = args
    os.makedirs('logs', exist_ok=True)
    log_path = os.path.join('logs', f'verify_{stage}.log')
    t0 = time.time()
    with open(log_path, 'a', buffering=1) as logf:
        def log(msg):
            logf.write(msg + '\n')

        from mario.agent_planner import PlannerAgent
        from mario.memory import PlannerMemory
        from mario.vision import VisionTracker
        agent = PlannerAgent(stage, memory=PlannerMemory(db_path),
                             vision=VisionTracker(),
                             max_frames_per_episode=max_frames, log=log)
        out = agent.run(max_episodes=episodes, stop_on_clear=True)
        out['wall_sec'] = time.time() - t0
        log(f"SUMMARY: {out}")
    return out


def main():
    p = argparse.ArgumentParser(description='Verify planner agent on all stages')
    p.add_argument('--stages', nargs='*', default=None,
                   help='Stages to verify (default: all except loop stages)')
    p.add_argument('--episodes', type=int, default=40, help='Episode budget per stage')
    p.add_argument('--workers', type=int, default=12)
    p.add_argument('--db', default=os.path.join('db', 'planner.sqlite'))
    p.add_argument('--max-frames', type=int, default=14000)
    args = p.parse_args()

    stages = args.stages or target_stages()
    print(f"Verifying {len(stages)} stages with {args.workers} workers "
          f"(budget {args.episodes} episodes/stage)\n")
    t0 = time.time()

    jobs = [(s, args.episodes, args.db, args.max_frames) for s in stages]
    ctx = mp.get_context('spawn')
    with ctx.Pool(args.workers) as pool:
        results = pool.map(run_stage, jobs)

    results.sort(key=lambda r: (int(r['stage'].split('-')[0]), int(r['stage'].split('-')[1])))
    cleared = sum(1 for r in results if r['cleared'])
    print(f"\n{'stage':>6} {'clear':>6} {'1st-ep':>7} {'episodes':>9} {'best_x':>7} {'wall':>8}")
    for r in results:
        print(f"{r['stage']:>6} {'YES' if r['cleared'] else 'no':>6} "
              f"{str(r['first_clear_episode'] or '-'):>7} {r['episodes']:>9} "
              f"{r['best_x']:>7} {r['wall_sec']:>7.0f}s")
    print(f"\nCleared {cleared}/{len(stages)} stages in {(time.time()-t0)/60:.1f} min")
    return 0 if cleared == len(stages) else 1


if __name__ == '__main__':
    raise SystemExit(main())
