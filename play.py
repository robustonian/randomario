#!/usr/bin/env python3
"""RandoMario unified entry point.

Examples:
    uv run play.py --agent planner --stage 1-1                 # model-based agent, UI
    uv run play.py --agent planner --stage 8-1 --speed max     # fast-forward
    uv run play.py --agent planner --stage 2-2 --headless      # no UI, max speed
    uv run play.py --agent explore --stage 1-1 --actions right_only
"""
import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='RandoMario — Super Mario Bros. agents')
    p.add_argument('--agent', choices=['planner', 'explore'], default='planner',
                   help='planner: model-based lookahead agent / explore: Go-Explore random baseline')
    p.add_argument('--stage', '-s', default='1-1', help='Stage like 1-1 ... 8-4')
    p.add_argument('--episodes', '-e', type=int, default=None,
                   help='Max episodes (default: planner 50 / explore 1000)')
    p.add_argument('--headless', action='store_true', help='No UI, run at max speed')
    p.add_argument('--speed', default='1x', choices=['1x', '2x', '4x', '8x', 'max'],
                   help='Initial speed (changeable at runtime with keys 1-5 / +/-)')
    p.add_argument('--no-vision', action='store_true',
                   help='Disable frame-based vision (planner only)')
    p.add_argument('--no-stop-on-clear', action='store_true',
                   help='Keep playing after the first clear (planner only)')
    p.add_argument('--db', default=None, help='Planner DB path (default: db/planner.sqlite)')
    p.add_argument('--max-frames', type=int, default=14000,
                   help='Max frames per episode (planner only)')
    # explore-specific
    p.add_argument('--actions', default='right_only',
                   choices=['right_only', 'simple', 'complex'],
                   help='Action set for the explore agent')
    p.add_argument('--archive', '-a', default=None, help='Archive pkl path (explore agent)')
    p.add_argument('--seed', type=int, default=None, help='Random seed (explore agent)')
    p.add_argument('--max-steps', type=int, default=6000,
                   help='Max steps per episode (explore agent)')
    p.add_argument('--explore-steps', type=int, default=1200,
                   help='Explore steps after replay (explore agent)')
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    speed = args.speed.upper() if args.speed == 'max' else args.speed

    ui = None
    if not args.headless:
        from mario.ui import GameUI
        from mario.actions import ACTION_SETS
        actions = ACTION_SETS['planner' if args.agent == 'planner' else args.actions]
        ui = GameUI(args.stage, actions,
                    'Planner (model-based)' if args.agent == 'planner' else 'Go-Explore (random)',
                    speed=speed)

    try:
        if args.agent == 'planner':
            from mario.agent_planner import PlannerAgent
            from mario.memory import PlannerMemory, PLANNER_DB_PATH
            from mario.vision import VisionTracker
            agent = PlannerAgent(
                args.stage,
                memory=PlannerMemory(args.db or PLANNER_DB_PATH),
                ui=ui,
                vision=None if args.no_vision else VisionTracker(),
                max_frames_per_episode=args.max_frames)
            out = agent.run(max_episodes=args.episodes or 50,
                            stop_on_clear=not args.no_stop_on_clear)
        else:
            from mario.agent_explore import GoExploreAgent
            agent = GoExploreAgent(
                args.stage, archive_path=args.archive, actions_name=args.actions,
                ui=ui, seed=args.seed, max_steps_per_episode=args.max_steps,
                explore_steps_after_return=args.explore_steps)
            out = agent.run(max_episodes=args.episodes or 1000)
    finally:
        if ui is not None:
            ui.close()

    print(f"\n=== Summary ===\n"
          f"stage={out['stage']} cleared={out['cleared']} "
          f"first_clear_episode={out['first_clear_episode']} "
          f"episodes={out['episodes']} best_x={out['best_x']}")
    return 0 if out['cleared'] else 1


if __name__ == '__main__':
    sys.exit(main())
