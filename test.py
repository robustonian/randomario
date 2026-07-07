#!/usr/bin/env python3
"""Backward-compatible wrapper around the Go-Explore agent.

The implementation moved to ``mario/agent_explore.py``; this file keeps the
old ``uv run test.py`` workflow and the old pickle class path (``test.Cell``)
working. Prefer ``uv run play.py --agent explore`` going forward.
"""
import argparse
import sys

from mario.agent_explore import Cell, GoExploreAgent, default_archive_path  # noqa: F401


def main():
    parser = argparse.ArgumentParser(description='Go-Explore style Mario with Pygame UI')
    parser.add_argument('--stage', '-s', default='1-1')
    parser.add_argument('--episodes', '-e', type=int, default=1000)
    parser.add_argument('--archive', '-a', default=None)
    parser.add_argument('--actions', default='right_only',
                        choices=['right_only', 'simple', 'complex'])
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--fps', type=int, default=60, help='(kept for compatibility)')
    parser.add_argument('--max-steps', type=int, default=6000)
    parser.add_argument('--explore-steps', type=int, default=1200)
    parser.add_argument('--headless', action='store_true')
    args = parser.parse_args()

    ui = None
    if not args.headless:
        from mario.actions import ACTION_SETS
        from mario.ui import GameUI
        ui = GameUI(args.stage, ACTION_SETS[args.actions], 'Go-Explore (random)')

    agent = GoExploreAgent(
        args.stage, archive_path=args.archive, actions_name=args.actions, ui=ui,
        seed=args.seed, max_steps_per_episode=args.max_steps,
        explore_steps_after_return=args.explore_steps)
    try:
        out = agent.run(max_episodes=args.episodes)
    finally:
        if ui is not None:
            ui.close()
    print(f"stage={out['stage']} cleared={out['cleared']} episodes={out['episodes']}")


if __name__ == '__main__':
    sys.exit(main())
