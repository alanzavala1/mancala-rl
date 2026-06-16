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

# Counts are scaled by the starting seeds per pit, so 1.0 == a full starting pit
# and the network sees seed counts at near-integer resolution. Dividing by the
# 48-seed total instead (the old choice) squashed every count into [0, ~0.5], so
# the 1-vs-2-seed distinctions that decide captures and extra turns differed by
# only ~0.02 -- this keeps that resolution.
_SCALE = engine.STARTING_SEEDS  # 4

# Value target = the FINAL store margin from the mover's view, squashed to
# [-1, 1]. Unlike a plain win/loss label this keeps a learning signal alive in a
# first-player-win game (the margin still varies once every game is a P1 win) and
# rewards decisive, safe conversion over knife-edge wins. The margin is the final
# score only -- never a per-move reward, which was v1's mistake (it just imitated
# greedy). MARGIN_SCALE controls resolution: a margin of this size maps to 1.0,
# so close games (where conversion actually matters) keep full resolution.
MARGIN_SCALE = 16


def margin_value(margin):
    """Squash a final store margin (store1 - store2, mover-relative) to [-1, 1]."""
    v = margin / MARGIN_SCALE
    return 1.0 if v > 1.0 else (-1.0 if v < -1.0 else v)


def encode(state):
    """Return a 14-float list: [my 6 pits, my store, opp 6 pits, opp store]."""
    b = state.board
    if state.current_player == 1:
        mine, opp = b[0:7], b[7:14]   # A-F + store1, then G-L + store2
    else:
        mine, opp = b[7:14], b[0:7]
    return [x / _SCALE for x in (*mine, *opp)]
