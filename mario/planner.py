"""Model-predictive planner.

Every replan step the planner:

1. builds candidate input sequences (macro moves) for the next few dozen
   frames — seeded by vision hints (gaps / approaching enemies) and by the
   hazard memory of previous deaths,
2. evaluates each candidate by actually rolling the emulator forward from a
   savestate (the model is exact, the search is what matters),
3. picks the best-scoring plan; if everything dies it escalates through
   wider/finer candidate waves with a longer horizon before giving up.

The same planner is used unmodified on every stage: nothing here is
stage-specific, per-stage knowledge only ever enters through the hazard
memory learned from that stage's own deaths.
"""
import hashlib
import random
from typing import Dict, List, Optional

from .actions import get_action_indices
from .env_utils import MarioSession

BASE_HORIZON = 70
HAZARD_HORIZON = 120
WAVE3_HORIZON = 170
LOOP_HORIZON = 260  # long enough to cross a maze checkpoint zone and see
                    # the warp-back within the rollout itself

# Score shaping
FLAG_BONUS = 5000.0
DEATH_PENALTY = 1200.0
WARP_PENALTY = 3000.0

# With little game-clock left, waiting is no longer safe: charge candidates
# per frame so forward progress dominates (dash earns ~2.5px/frame ≫ cost).
URGENT_TIME_LEFT = 90
URGENT_FRAME_COST = 0.6

# Rollouts are exact, so a plan that survived its horizon (and tail) can be
# executed for a good chunk of that horizon before replanning.
EXEC_CLEAN = 50
EXEC_TAIL_OK = 32
EXEC_CAUTIOUS = 12
EXEC_DEADLOCK = 30      # nothing progresses: replan less often, not more
CLEAN_PROGRESS_MIN = 8

# Early accept: in open terrain a candidate running at (near) full speed with
# no death and a grounded ending needs no further search.
FAST_ACCEPT_SPEED = 2.0  # px/frame (max run speed is ~2.5)

# After picking the best candidates, roll a bit further (dash continuation)
# to catch dooms hiding just beyond the horizon (e.g. sinking platforms,
# jumps that land in a pit after the plan ends).
TAIL_FRAMES = 90
TAIL_TOP_K = 4
TAIL_DEATH_PENALTY = 700.0


class PlanResult:
    __slots__ = ('actions', 'name', 'score', 'died', 'flag', 'flag_frame', 'warp',
                 'end_x', 'max_x', 'frames', 'wave', 'clean', 'sig', 'n_candidates',
                 'tail_ok', 'progress', 'x0', 'first_commit')

    def __init__(self, actions, name, score, died, flag, flag_frame, warp,
                 end_x, max_x, frames, wave, clean, sig, n_candidates,
                 tail_ok=False, progress=0, x0=0, first_commit=None):
        self.actions = actions
        self.name = name
        self.score = score
        self.died = died
        self.flag = flag
        self.flag_frame = flag_frame
        self.warp = warp
        self.end_x = end_x
        self.max_x = max_x
        self.frames = frames
        self.wave = wave
        self.clean = clean
        self.sig = sig
        self.n_candidates = n_candidates
        self.tail_ok = tail_ok
        self.progress = progress
        self.x0 = x0
        self.first_commit = first_commit

    @property
    def exec_frames(self) -> int:
        if self.flag:
            return len(self.actions)
        if self.died or self.warp:
            # Least-bad plan: execute only a sliver, then rethink.
            return EXEC_CAUTIOUS
        if self.progress < 4:
            # Deadlock (waiting): replan at a slower cadence.
            return EXEC_DEADLOCK
        # Rollouts are exact, so a surviving plan is safe for all the frames
        # it was simulated for; keep a small margin so the next plan starts
        # with lookahead to spare. Executing less would cause receding-horizon
        # procrastination (delayed jumps that never happen).
        base = max(EXEC_CAUTIOUS, self.frames - 16)
        if not self.tail_ok and self.first_commit is not None and self.first_commit > 4:
            # The tail after this plan died: walk up to the takeoff point but
            # don't jump yet — replan there with full lookahead instead.
            base = min(base, self.first_commit)
        return base


def plan_signature(actions: List[int]) -> str:
    return hashlib.sha1(bytes(actions[:48])).hexdigest()[:16]


def lane_profile(points, x_from: int, x_to: int) -> dict:
    """Coarse y-lane taken across an x-range: {x//32: y//32}."""
    prof = {}
    for x, y in points:
        if x_from <= x <= x_to:
            prof[x // 32] = y // 32
    return prof


def lane_matches(prof: dict, failed: dict, min_overlap: int = 4) -> bool:
    common = set(prof) & set(failed)
    if len(common) < min_overlap:
        return False
    return all(prof[b] == failed[b] for b in common)


class EdgeJumpPolicy:
    """Closed-loop candidate: (optionally) back off, take a running start and
    jump exactly when x reaches the platform edge.  Evaluated inside the
    rollout (where x is known each frame); the emitted actions are recorded
    and replayed verbatim on the real environment."""

    def __init__(self, idx: dict, start_x: int, edge_x: int,
                 back_px: int, margin: int, hold: int):
        self.idx = idx
        self.start_x = start_x
        self.edge_x = edge_x
        self.back_px = back_px
        self.margin = margin
        self.hold = hold
        self.phase = 0  # 0=retreat 1=settle 2=run 3=jump 4=cruise
        self.counter = 0

    def step(self, x: int, i: int) -> int:
        idx = self.idx
        if self.phase == 0:
            if self.back_px <= 0 or x <= self.start_x - self.back_px or i > 150:
                self.phase = 1
                self.counter = 6 if self.back_px > 0 else 0
            else:
                # Hop-retreat: plain walking gets pinned on staircase steps,
                # so alternate jump/walk to climb backwards over them.
                return idx['LEFT_JUMP'] if (i % 20) < 10 else idx['LEFT']
        if self.phase == 1:
            if self.counter > 0:
                self.counter -= 1
                return idx['NOOP']
            self.phase = 2
        if self.phase == 2:
            if x >= self.edge_x - self.margin:
                self.phase = 3
                self.counter = self.hold
            else:
                return idx['RIGHT_DASH']
        if self.phase == 3:
            if self.counter > 0:
                self.counter -= 1
                return idx['RIGHT_DASH_JUMP']
            self.phase = 4
        return idx['RIGHT_DASH']


class Planner:
    def __init__(self, session: MarioSession, memory=None):
        self.session = session
        self.idx = get_action_indices(session.actions)
        self.memory = memory
        self.pump = None  # optional UI callback, called between rollouts
        self.failed_lanes = []  # y-profiles that warped back in a loop maze
        self.recent_xy = None   # real-play (x, y) history shared by the agent
        self.rollout_frames_total = 0
        self.plans_total = 0

    # -------------------------------------------------------- macro helpers

    def _seq(self, *parts) -> List[int]:
        """parts: (action_name, frames) pairs -> flat action index list."""
        out: List[int] = []
        for name, n in parts:
            out.extend([self.idx[name]] * n)
        return out

    def _pad(self, seq: List[int], horizon: int, filler: str = 'RIGHT_DASH') -> List[int]:
        if len(seq) < horizon:
            seq = seq + [self.idx[filler]] * (horizon - len(seq))
        return seq[:horizon]

    def _release_guard(self, seq: List[int], last_action: Optional[int]) -> List[int]:
        """A new jump needs a fresh A press: insert one release frame if the
        previously executed action already held A."""
        if last_action is None or not seq:
            return seq
        a_actions = {self.idx.get(k) for k in
                     ('RIGHT_DASH_JUMP', 'RIGHT_JUMP', 'JUMP', 'LEFT_JUMP')}
        if last_action in a_actions and seq[0] in a_actions:
            release = {
                self.idx.get('RIGHT_DASH_JUMP'): self.idx['RIGHT_DASH'],
                self.idx.get('RIGHT_JUMP'): self.idx['RIGHT'],
                self.idx.get('JUMP'): self.idx['NOOP'],
                self.idx.get('LEFT_JUMP'): self.idx['LEFT'],
            }.get(seq[0], self.idx['NOOP'])
            return [release] + seq
        return seq

    # ------------------------------------------------------ candidate waves

    def _ground_wave1(self, h: int) -> List:
        c = [
            ('dash', self._seq(('RIGHT_DASH', h))),
            ('walk', self._seq(('RIGHT', h))),
        ]
        for hold in (6, 12, 20, 28):
            c.append((f'jump{hold}',
                      self._seq(('RIGHT_DASH_JUMP', hold), ('RIGHT_DASH', h))))
        c.append(('djump10-20',
                  self._seq(('RIGHT_DASH', 10), ('RIGHT_DASH_JUMP', 20), ('RIGHT_DASH', h))))
        c.append(('smalljump',
                  self._seq(('RIGHT_JUMP', 8), ('RIGHT', h))))
        c.append(('upjump',
                  self._seq(('JUMP', 14), ('RIGHT_DASH', h))))
        c.append(('wait16', self._seq(('NOOP', 16), ('RIGHT_DASH', h))))
        return c

    def _ground_wave2(self, h: int, fine: bool = True) -> List:
        c = []
        for delay in range(0, 25, 4 if fine else 8):
            for hold in (10, 18, 26):
                c.append((f'd{delay}j{hold}',
                          self._seq(('RIGHT_DASH', delay), ('RIGHT_DASH_JUMP', hold),
                                    ('RIGHT_DASH', h))))
        # Jump then stop mid-air: lands on the next platform instead of
        # overshooting it (tree/scale/mushroom stages).
        for hold in (10, 16, 24):
            c.append((f'jumpstop{hold}',
                      self._seq(('RIGHT_DASH_JUMP', hold), ('NOOP', h))))
        c.append(('walkjumpstop',
                  self._seq(('RIGHT_JUMP', 12), ('NOOP', h))))
        for delay in (4, 10, 16):
            c.append((f'walkd{delay}j8',
                      self._seq(('RIGHT', delay), ('RIGHT_JUMP', 8), ('RIGHT', h))))
        for hold in (22, 32):
            c.append((f'upjump{hold}',
                      self._seq(('JUMP', hold), ('RIGHT_DASH', h))))
        c += self._random_wave(h, 4)
        for w in (24, 48):
            c.append((f'wait{w}', self._seq(('NOOP', w), ('RIGHT_DASH', h))))
            c.append((f'wait{w}jump',
                      self._seq(('NOOP', w), ('RIGHT_DASH_JUMP', 20), ('RIGHT_DASH', h))))
        c.append(('retreat',
                  self._seq(('LEFT', 12), ('NOOP', 4), ('RIGHT_DASH_JUMP', 24),
                            ('RIGHT_DASH', h))))
        return c

    def _ground_wave3(self, h: int) -> List:
        c = []
        for w in (30, 60, 90, 120):
            c.append((f'longwait{w}', self._seq(('NOOP', w), ('RIGHT_DASH', h))))
            c.append((f'longwait{w}jump',
                      self._seq(('NOOP', w), ('RIGHT_DASH_JUMP', 22), ('RIGHT_DASH', h))))
        for back in (24, 48, 96):
            c.append((f'retreat{back}',
                      self._seq(('LEFT', back), ('NOOP', 8), ('RIGHT_DASH', h))))
            c.append((f'retreat{back}jump',
                      self._seq(('LEFT', back), ('NOOP', 8), ('RIGHT_DASH', 16),
                                ('RIGHT_DASH_JUMP', 26), ('RIGHT_DASH', h))))
        c.append(('backtrack',
                  self._seq(('LEFT_DASH', 60), ('NOOP', 8), ('RIGHT_DASH', h))))
        # Hop onto a (moving) platform, ride it, hop off — scales and lifts.
        for h1 in (16, 26):
            for ride in (40, 80):
                c.append((f'hop{h1}ride{ride}',
                          self._seq(('RIGHT_DASH_JUMP', h1), ('NOOP', ride),
                                    ('RIGHT_DASH_JUMP', 28), ('RIGHT_DASH', h))))
        for delay in range(0, 31, 6):
            c.append((f'w3d{delay}j24',
                      self._seq(('RIGHT_DASH', delay), ('RIGHT_DASH_JUMP', 24),
                                ('RIGHT_DASH', h))))
        c.append(('walkonly', self._seq(('RIGHT', h))))
        c.append(('idle', self._seq(('NOOP', h))))
        # Tall wall right ahead (pipe / staircase pocket): back off far and
        # take a running full jump, sweeping the takeoff distance.
        x_here = self.session.x_pos
        for back, margin in ((90, 24), (90, 40), (130, 55), (130, 70),
                             (130, 85), (160, 100)):
            pol = EdgeJumpPolicy(self.idx, x_here, x_here + 8, back, margin, 32)
            pol.min_horizon = 180 + back
            c.append((f'walljump-b{back}m{margin}', pol))
        for hold in (22, 32):
            c.append((f'upjump{hold}',
                      self._seq(('JUMP', hold), ('RIGHT_DASH', h))))
        c += self._random_wave(h, 8)
        return c

    _RANDOM_MOVES = [('RIGHT_DASH', 5), ('RIGHT_DASH_JUMP', 4), ('RIGHT', 2),
                     ('RIGHT_JUMP', 2), ('JUMP', 2), ('NOOP', 2),
                     ('LEFT', 1), ('LEFT_JUMP', 1), ('DOWN', 1)]

    def _random_wave(self, h: int, n: int) -> List:
        """Random macro sequences, safety-checked by the rollout like any
        other candidate. The deterministic waves can systematically miss
        certain solutions (springboard timing, odd wall climbs); these
        eventually find them and break stuck-loops."""
        names = [m[0] for m in self._RANDOM_MOVES]
        weights = [m[1] for m in self._RANDOM_MOVES]
        cands = []
        for k in range(n):
            seq: List[int] = []
            while len(seq) < h:
                move = random.choices(names, weights=weights, k=1)[0]
                seq.extend(self._seq((move, random.randint(4, 26))))
            cands.append((f'rand{k}', seq[:h]))
        return cands

    def _swim_wave(self, h: int, wave: int) -> List:
        """Water physics: A must be tapped (press+release) to stroke."""
        c = []
        periods = (6, 10, 16, 24) if wave == 1 else (5, 8, 12, 20)
        for p in periods:
            cycle = self._seq(('RIGHT_JUMP', 2), ('RIGHT', max(1, p - 2)))
            c.append((f'swim{p}', (cycle * (h // len(cycle) + 1))[:h]))
        # Rise vertically / hold position / sink
        up = self._seq(('JUMP', 2), ('NOOP', 6))
        c.append(('rise', (up * (h // 8 + 1))[:h]))
        c.append(('sink', self._seq(('RIGHT', h))))
        if wave >= 2:
            back = self._seq(('LEFT_JUMP', 2), ('LEFT', 8))
            c.append(('swimback', (back * (h // 10 + 1))[:h]))
            for d in (6, 12, 20):
                cyc = self._seq(('RIGHT_JUMP', 2), ('RIGHT', 8))
                c.append((f'swimwait{d}',
                          (self._seq(('NOOP', d)) + cyc * (h // 10 + 1))[:h]))
        if wave >= 3:
            for w in (30, 60):
                cyc = self._seq(('RIGHT_JUMP', 2), ('RIGHT', 8))
                hold = self._seq(('JUMP', 2), ('NOOP', 8)) * (w // 10 + 1)
                c.append((f'tread{w}', (hold[:w] + cyc * (h // 10 + 1))[:h]))
        return c

    def _edge_candidates(self, x0: int, hints: Optional[dict]) -> List:
        """Closed-loop running-start jumps at platform/pit edges seen by
        vision. These crack wide gaps that fixed-timing candidates miss."""
        if not hints:
            return []
        edges = []
        e = hints.get('edge')
        if e is not None and 0 < e - x0 < 220:
            edges.append(e)
        for g in hints.get('gaps', [])[:2]:
            if 0 < g - x0 < 220 and all(abs(g - e2) > 30 for e2 in edges):
                edges.append(g)
        cands = []
        for e in edges[:2]:
            for back, margin, hold in ((0, 6, 28), (0, 14, 28), (0, 2, 32),
                                       (48, 8, 28), (96, 8, 32), (140, 8, 32)):
                pol = EdgeJumpPolicy(self.idx, x0, e, back, margin, hold)
                pol.min_horizon = 130 + back + hold
                cands.append((f'edge+{e - x0}b{back}m{margin}', pol))
        return cands

    def _vision_candidates(self, h: int, hints: Optional[dict]) -> List:
        """Turn estimated world state (from frames) into targeted plans."""
        if not hints:
            return []
        c = []
        vx = max(1.0, float(hints.get('mario_vx', 1.5)))
        for gap in hints.get('gaps', [])[:2]:
            dist = gap - self.session.x_pos
            if 0 < dist < 260:
                takeoff = max(0, int((dist - 8) / vx))
                for dt in (-4, 0, 4):
                    d = max(0, takeoff + dt)
                    if d < h - 20:
                        c.append((f'gapjump@{d}',
                                  self._seq(('RIGHT_DASH', d), ('RIGHT_DASH_JUMP', 28),
                                            ('RIGHT_DASH', h))))
        for ent in hints.get('enemies', [])[:3]:
            dist = ent['x'] - self.session.x_pos
            closing = vx - min(0.0, ent.get('vx', 0.0))
            if 0 < dist < 160 and closing > 0.2:
                meet = max(0, int(dist / closing) - 12)
                if meet < h - 16:
                    c.append((f'stomp@{meet}',
                              self._seq(('RIGHT_DASH', meet), ('RIGHT_DASH_JUMP', 12),
                                        ('RIGHT_DASH', h))))
        return c

    # ------------------------------------------------------------ evaluation

    def _evaluate(self, actions, horizon: int, x0: int, y0: int = 0,
                  record_traj: bool = False, urgent: bool = False) -> dict:
        """actions: list of action indices, or a closed-loop policy object
        with .step(x, i). For policies the emitted actions are recorded."""
        session = self.session
        policy = None if isinstance(actions, list) else actions
        recorded = [] if policy is not None else None
        traj = [] if record_traj else None
        max_x = x0
        end_x = x0
        x = x0
        died = flag = warp = False
        flag_frame = None
        frames = 0
        y_tail = []
        n = horizon if policy is not None else min(len(actions), horizon)
        for i in range(n):
            a = policy.step(x, i) if policy is not None else actions[i]
            if recorded is not None:
                recorded.append(a)
            rs = session.rollout_step(a)
            frames = i + 1
            end_x = rs.x
            x = rs.x
            if traj is not None:
                traj.append((rs.x, rs.y))
            y_tail.append(rs.y)
            if len(y_tail) > 6:
                y_tail.pop(0)
            if rs.x > max_x:
                max_x = rs.x
            if rs.flag:
                flag = True
                flag_frame = frames
                break
            if rs.dead:
                died = True
                break
            if rs.world_changed:
                warp = True
                break
        airborne = bool(y_tail) and (max(y_tail) - min(y_tail)) > 3
        self.rollout_frames_total += frames
        score = (max_x - x0) + 0.5 * (end_x - x0)
        if (max_x - x0) < 8 and not died and not warp and not airborne and y_tail:
            # x is blocked: reward gaining height on something solid
            # (elevators, vines, climbing to an upper route).
            score += 0.4 * max(0, y_tail[-1] - y0)
        if urgent:
            score -= URGENT_FRAME_COST * frames
        if flag:
            score += FLAG_BONUS - frames
        if died:
            score -= DEATH_PENALTY - 2.0 * frames
        if warp:
            score -= WARP_PENALTY
        return dict(score=score, died=died, flag=flag, flag_frame=flag_frame,
                    warp=warp, end_x=end_x, max_x=max_x, frames=frames,
                    airborne=airborne, recorded=recorded, traj=traj)

    # ----------------------------------------------------------------- plan

    def plan(self, hints: Optional[dict] = None,
             last_action: Optional[int] = None,
             force_wave: int = 1) -> PlanResult:
        session = self.session
        x0 = session.x_pos
        stage = session.stage
        swimming = session.is_swimming

        urgent = self.session.time_left < URGENT_TIME_LEFT
        hazard_near = False
        hazard_hot = False
        loop_zone = False
        loop_xs = []
        if self.memory is not None:
            # Timeout hazards must NOT trigger the cautious (slow) machinery —
            # the cure for running out of time is speed, not longer waits.
            hs = [h for h in self.memory.hazards_in(stage, x0, x0 + 280)
                  if h['cause'] != 'timeout']
            hazard_near = bool(hs)
            # Died here repeatedly: skip straight to the fine search.
            hazard_hot = any(h['hits'] >= 2 for h in hs)
            loop_xs = [h['x'] for h in self.memory.hazards_in(stage, x0, x0 + 420)
                       if h['cause'] == 'loop']
            loop_zone = bool(loop_xs)
        loop_x = min(loop_xs) if loop_zone else None
        gap_near = bool(hints) and any(
            0 < g - x0 < 250 for g in hints.get('gaps', []))

        y0 = session.y_pos
        session.snapshot()
        best = None
        best_meta = None
        wave = max(1, force_wave)
        if (hazard_hot or loop_zone) and not swimming:
            wave = max(wave, 2)
        n_candidates = 0
        evaluated = []
        tail_ok = False
        try:
            while wave <= 3:
                if swimming:
                    horizon = HAZARD_HORIZON if (hazard_near or wave > 1) else BASE_HORIZON
                    cands = self._swim_wave(horizon, wave)
                else:
                    if wave == 1:
                        horizon = HAZARD_HORIZON if hazard_near else BASE_HORIZON
                        cands = self._ground_wave1(horizon)
                    elif wave == 2:
                        horizon = LOOP_HORIZON if loop_zone else HAZARD_HORIZON
                        cands = self._ground_wave2(horizon)
                    else:
                        horizon = LOOP_HORIZON if loop_zone else WAVE3_HORIZON
                        cands = self._ground_wave3(horizon)
                cands = self._vision_candidates(horizon, hints) + cands
                if (gap_near or hazard_near) and not swimming:
                    cands = self._edge_candidates(x0, hints) + cands

                for name, seq in cands:
                    if self.pump is not None:
                        self.pump()
                    is_policy = not isinstance(seq, list)
                    if is_policy:
                        h_eff = max(horizon, getattr(seq, 'min_horizon', 0))
                    else:
                        h_eff = horizon
                        seq = self._release_guard(self._pad(seq, horizon), last_action)
                        sig = plan_signature(seq)
                        if self.memory is not None and self.memory.is_blacklisted(stage, x0, sig):
                            continue
                    n_candidates += 1
                    meta = self._evaluate(seq, h_eff, x0, y0,
                                          record_traj=loop_zone, urgent=urgent)
                    session.restore()
                    if loop_zone:
                        # Warping back is still worth more than waiting forever
                        # (each warp teaches us a failed lane) ...
                        drop = meta['max_x'] - meta['end_x']
                        if drop > 150 and not meta['died']:
                            meta['score'] += 0.35 * drop
                        # ... but a lane that already warped us back is out.
                        # Prepend the real approach so profiles are complete
                        # even when replanning right before the checkpoint.
                        pts = list(self.recent_xy or []) + meta['traj']
                        # SMB maze checkpoints come in chains of ~3 screens.
                        prof = lane_profile(pts, loop_x - 800, loop_x + 8)
                        if any(lane_matches(prof, f) for f in self.failed_lanes):
                            meta['score'] -= 900.0
                    if is_policy:
                        seq = meta['recorded']
                        sig = plan_signature(seq)
                        if self.memory is not None and self.memory.is_blacklisted(stage, x0, sig):
                            continue
                    evaluated.append((name, seq, sig, meta))
                    if best is None or meta['score'] > best_meta['score']:
                        best = (name, seq, sig)
                        best_meta = meta
                    if meta['flag']:
                        break
                    # Open terrain at full speed with a grounded ending —
                    # no need to search (or tail-check) further.
                    if (wave == 1 and not hazard_near and not gap_near
                            and not swimming and not meta['died'] and not meta['warp']
                            and not meta['airborne']
                            and (meta['max_x'] - x0) >= FAST_ACCEPT_SPEED * horizon):
                        best = (name, seq, sig)
                        best_meta = meta
                        break

                if best_meta is not None and best_meta['flag']:
                    break
                # Escalate when the best plan still dies or makes no progress.
                if best_meta is None or best_meta['died'] or best_meta['warp'] \
                        or (best_meta['max_x'] - x0) < CLEAN_PROGRESS_MIN:
                    wave += 1
                else:
                    break

            # Tail validation: re-rank the top surviving plans by also rolling
            # ~1.5s past the horizon, so a doom hiding just beyond it (pit
            # landing, sinking platform) demotes the plan before we commit.
            # Only worth the extra rollouts when something risky is around:
            # plan ends mid-air, a gap is visible ahead, or deaths happened here.
            risky = (best is None or best_meta['flag'] or hazard_near or gap_near
                     or best_meta['airborne'])
            tail_ok = not risky  # nothing risky around == implicitly validated
            need_tail = (best is not None and not best_meta['flag']
                         and (hazard_near or gap_near or best_meta['airborne']))
            if need_tail:
                survivors = [c for c in evaluated
                             if not c[3]['died'] and not c[3]['warp']]
                survivors.sort(key=lambda c: c[3]['score'], reverse=True)
                top = survivors[:TAIL_TOP_K]
                if top:
                    if swimming:
                        stroke = self._seq(('RIGHT_JUMP', 2), ('RIGHT', 8))
                        filler = (stroke * (TAIL_FRAMES // len(stroke) + 1))[:TAIL_FRAMES]
                    else:
                        filler = [self.idx['RIGHT_DASH']] * TAIL_FRAMES
                    best_adj = None
                    for name, seq, sig, meta in top:
                        if self.pump is not None:
                            self.pump()
                        ext_seq = list(seq) + filler
                        ext = self._evaluate(ext_seq, len(ext_seq), x0, y0,
                                             urgent=urgent)
                        session.restore()
                        ok = not ext['died'] and not ext['warp']
                        # Penalize by how soon after the plan the doom hits;
                        # dying at the very end of the tail is nearly free so
                        # stalling never beats progress.
                        tail_survived = max(0, ext['frames'] - len(seq))
                        adj = meta['score'] - (
                            0.0 if ok else
                            TAIL_DEATH_PENALTY * max(0.0, 1.0 - tail_survived / TAIL_FRAMES))
                        if best_adj is None or adj > best_adj[0]:
                            best_adj = (adj, name, seq, sig, meta, ok)
                    _, b_name, b_seq, b_sig, b_meta, tail_ok = best_adj
                    best = (b_name, b_seq, b_sig)
                    best_meta = b_meta
        finally:
            session.restore()

        self.plans_total += 1
        if best is None:
            # Everything was blacklisted — fall back to plain dash.
            seq = self._seq(('RIGHT_DASH', BASE_HORIZON))
            best = ('dash', seq, plan_signature(seq))
            session.snapshot()
            best_meta = self._evaluate(seq, BASE_HORIZON, x0, y0)
            session.restore()
        name, seq, sig = best
        clean = (not best_meta['died'] and not best_meta['warp'] and not hazard_near
                 and (best_meta['max_x'] - x0) >= CLEAN_PROGRESS_MIN and wave == 1
                 and tail_ok)
        a_set = {self.idx.get(k) for k in
                 ('RIGHT_DASH_JUMP', 'RIGHT_JUMP', 'JUMP', 'LEFT_JUMP')}
        first_commit = next((i for i, a in enumerate(seq) if a in a_set), None)
        return PlanResult(
            actions=seq, name=name, score=best_meta['score'], died=best_meta['died'],
            flag=best_meta['flag'], flag_frame=best_meta['flag_frame'],
            warp=best_meta['warp'], end_x=best_meta['end_x'], max_x=best_meta['max_x'],
            frames=best_meta['frames'], wave=min(wave, 3), clean=clean, sig=sig,
            n_candidates=n_candidates, tail_ok=tail_ok,
            progress=best_meta['max_x'] - x0, x0=x0, first_commit=first_commit)
