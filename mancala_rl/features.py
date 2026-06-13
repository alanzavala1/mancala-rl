"""Board encoding for the learning agent.

The network always sees the position from the perspective of the player to move
("my pits, my store, opponent's pits, opponent's store"), normalized to [0, 1].
Perspective-canonical encoding means the network learns one representation
regardless of which physical side is to move, and the 6 action slots always
refer to "my" pits. Normalizing fixes one of v1's problems (raw seed counts).
"""

from . import engine

NUM_FEATURES = 14
NUM_ACTIONS = 6

# All seeds in play sum to this, so dividing by it keeps features in [0, 1]
# and makes the whole vector sum to 1.0.
_TOTAL_SEEDS = 2 * len(engine.P1_PITS) * engine.STARTING_SEEDS  # 48


def encode(state):
    """Return a 14-float list: [my 6 pits, my store, opp 6 pits, opp store]."""
    b = state.board
    if state.current_player == 1:
        mine, opp = b[0:7], b[7:14]   # A-F + store1, then G-L + store2
    else:
        mine, opp = b[7:14], b[0:7]
    return [x / _TOTAL_SEEDS for x in (*mine, *opp)]
