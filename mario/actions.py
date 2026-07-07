"""Action set definitions shared by all agents."""
from typing import Dict, List

from gym_super_mario_bros.actions import COMPLEX_MOVEMENT, RIGHT_ONLY, SIMPLE_MOVEMENT

# Planner uses a purpose-built superset so it can brake, retreat, duck and swim.
PLANNER_MOVEMENT = [
    ['NOOP'],
    ['right'],
    ['right', 'B'],
    ['right', 'A', 'B'],
    ['right', 'A'],
    ['A'],
    ['left'],
    ['left', 'A'],
    ['left', 'B'],
    ['down'],
    ['B'],
]

ACTION_SETS: Dict[str, List[List[str]]] = {
    'right_only': RIGHT_ONLY,
    'simple': SIMPLE_MOVEMENT,
    'complex': COMPLEX_MOVEMENT,
    'planner': PLANNER_MOVEMENT,
}

# Canonical names used by agents to look up indices in whatever action set is active.
_NAME_TO_BUTTONS = {
    'NOOP': ['NOOP'],
    'RIGHT': ['right'],
    'RIGHT_DASH': ['right', 'B'],
    'RIGHT_DASH_JUMP': ['right', 'A', 'B'],
    'RIGHT_JUMP': ['right', 'A'],
    'JUMP': ['A'],
    'LEFT': ['left'],
    'LEFT_JUMP': ['left', 'A'],
    'LEFT_DASH': ['left', 'B'],
    'DOWN': ['down'],
    'DOWN_JUMP': ['down', 'A'],
    'UP': ['up'],
    'FIRE': ['B'],
}


def get_action_indices(actions: List[List[str]]) -> Dict[str, int]:
    """Map canonical action names to indices of the given action set."""
    indices: Dict[str, int] = {}
    for name, buttons in _NAME_TO_BUTTONS.items():
        for i, action in enumerate(actions):
            if action == buttons:
                indices[name] = i
                break
    return indices
