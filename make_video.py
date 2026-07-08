#!/usr/bin/env python3
"""Render clear runs to video — random, planner, or a side-by-side comparison.

    uv run make_video.py --stage 1-1 --mode compare        # random | planner
    uv run make_video.py --stage 1-1 --mode random
    uv run make_video.py --stage 1-1 --mode planner        # clear episode only

Sources (both are deterministic input recordings):
- random : best_overall_path from pkl/go_explore_archive_{stage}_*.pkl
- planner: the validated clear episode from db/planner.sqlite

Quality: nearest-neighbor 4x upscale + near-lossless x264 (see mario/video.py).
"""
import argparse
import os
import time

import cv2
import numpy as np

from mario.actions import ACTION_SETS
from mario.agent_explore import default_archive_path, load_archive
from mario.env_utils import MarioSession
from mario.memory import PlannerMemory, PLANNER_DB_PATH
from mario.video import VideoRecorder

SCALE = 4
PANE_W, PANE_H = 256 * SCALE, 240 * SCALE
HEADER_H = 72
ENDING_FRAMES = 420   # keep emulating past the flag (slide, castle, fireworks)
FINAL_HOLD = 90

BG = (16, 18, 24)
COL_RANDOM = (36, 191, 251)    # RGB amber-ish
COL_PLANNER = (128, 222, 74)   # RGB green


def find_random_recording(stage: str):
    for actions_name in ('right_only', 'complex', 'simple'):
        path = default_archive_path(stage, actions_name)
        if not os.path.exists(path):
            continue
        d = load_archive(path)
        if d.get('first_clear_episode') is None:
            continue
        return actions_name, list(d['best_overall_path']), d['first_clear_episode']
    raise SystemExit(f"No cleared random archive found for {stage} in pkl/")


def find_planner_recording(stage: str, db_path: str = PLANNER_DB_PATH):
    eps = PlannerMemory(db_path).replay_episodes(stage)
    clear = next((e for e in eps if e['result'] == 'clear'), None)
    if clear is None:
        raise SystemExit(
            f"No replayable planner clear for {stage} — run "
            f"`uv run play.py --agent planner --stage {stage} --headless` first")
    return 'planner', clear['actions'], clear['ep']


class ClearPlayback:
    """Steps one recorded clear run frame by frame, then the post-flag ending,
    then freezes on the last frame."""

    def __init__(self, stage: str, actions_name: str, path):
        self.session = MarioSession(stage, ACTION_SETS[actions_name])
        self.path = path
        self.idx = 0
        self.ending_left = ENDING_FRAMES
        self.done = False
        self.flag = False
        obs, info = self.session.reset()
        self.frame = np.array(obs, copy=True)

    def tick(self) -> bool:
        """Advance one frame; returns True while still producing new frames."""
        smb = self.session.smb
        if not self.done and self.idx < len(self.path):
            obs, r, done, info = self.session.step(self.path[self.idx])
            self.idx += 1
            self.frame = np.array(obs, copy=True)
            if done:
                self.done = True
                self.flag = bool(info.get('flag_get'))
            return True
        if self.done and self.flag and self.ending_left > 0:
            smb._frame_advance(0)
            self.ending_left -= 1
            self.frame = np.array(smb.screen, copy=True)
            return True
        self.done = True
        return False

    def close(self):
        self.session.close()


def label_bar(width: int, text_left: str, text_right: str = None) -> np.ndarray:
    bar = np.zeros((HEADER_H, width, 3), np.uint8)
    bar[:] = BG
    font = cv2.FONT_HERSHEY_SIMPLEX

    def put(text, cx, color):
        (tw, th), _ = cv2.getTextSize(text, font, 1.1, 2)
        cv2.putText(bar, text, (cx - tw // 2, (HEADER_H + th) // 2),
                    font, 1.1, color, 2, cv2.LINE_AA)

    if text_right is None:
        put(text_left, width // 2, (226, 232, 240))
    else:
        put(text_left, width // 4, COL_RANDOM)
        put(text_right, 3 * width // 4, COL_PLANNER)
        bar[:, width // 2 - 2: width // 2 + 2] = (46, 54, 72)
    return bar


def upscale(frame: np.ndarray) -> np.ndarray:
    return cv2.resize(frame, (PANE_W, PANE_H), interpolation=cv2.INTER_NEAREST)


def render_single(stage: str, mode: str, out: str):
    if mode == 'random':
        actions_name, path, ep = find_random_recording(stage)
        label = f"RANDOM (Go-Explore)  -  Stage {stage}  -  first clear EP{ep}"
    else:
        actions_name, path, ep = find_planner_recording(stage)
        label = f"PLANNER (model-based)  -  Stage {stage}  -  clear episode EP{ep}"
    print(f"{mode}: {len(path)} input frames (clear at EP{ep})")

    pb = ClearPlayback(stage, actions_name, path)
    rec = VideoRecorder(out, size=(PANE_W, PANE_H + HEADER_H))
    bar = label_bar(PANE_W, label)
    while pb.tick():
        rec.add(np.vstack([bar, upscale(pb.frame)]))
    rec.add(np.vstack([bar, upscale(pb.frame)]), repeat=FINAL_HOLD)
    if not pb.flag:
        print(f"WARNING: {mode} recording did not reach the flag on replay")
    rec.close()
    pb.close()
    print(f"saved: {out} ({rec.frames} frames, {rec.frames / 60:.1f}s)")


def render_compare(stage: str, out: str):
    r_name, r_path, r_ep = find_random_recording(stage)
    p_name, p_path, p_ep = find_planner_recording(stage)
    print(f"random: {len(r_path)} frames (first clear EP{r_ep}) | "
          f"planner: {len(p_path)} frames (clear episode EP{p_ep})")

    left = ClearPlayback(stage, r_name, r_path)
    right = ClearPlayback(stage, p_name, p_path)
    width = PANE_W * 2
    rec = VideoRecorder(out, size=(width, PANE_H + HEADER_H))
    bar = label_bar(width,
                    f"RANDOM  first clear EP{r_ep}",
                    f"PLANNER  first clear EP{p_ep}")

    def compose():
        return np.vstack([bar, np.hstack([upscale(left.frame),
                                          upscale(right.frame)])])

    alive = True
    while alive:
        a = left.tick()
        b = right.tick()
        alive = a or b
        if alive:
            rec.add(compose())
    rec.add(compose(), repeat=FINAL_HOLD)
    for pb, name in ((left, 'random'), (right, 'planner')):
        if not pb.flag:
            print(f"WARNING: {name} recording did not reach the flag on replay")
        pb.close()
    rec.close()
    print(f"saved: {out} ({rec.frames} frames, {rec.frames / 60:.1f}s)")


def main():
    p = argparse.ArgumentParser(description='Render clear runs to video')
    p.add_argument('--stage', '-s', default='1-1')
    p.add_argument('--mode', choices=['random', 'planner', 'compare'],
                   default='compare')
    p.add_argument('--out', default=None, help='Output mp4 path')
    args = p.parse_args()

    out = args.out or f"videos/{args.mode}_{args.stage}_{time.strftime('%Y%m%d-%H%M%S')}.mp4"
    if args.mode == 'compare':
        render_compare(args.stage, out)
    else:
        render_single(args.stage, args.mode, out)


if __name__ == '__main__':
    main()
