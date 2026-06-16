"""Board encoding for the network.

The network always sees the board from the perspective of the player to move
("my pits, my store, opponent's pits, opponent's store"). I encode it this way
so the network learns a single representation no matter which physical side is to
move, and so the six action slots always mean "my" pits. The counts are scaled
(see _SCALE) rather than fed raw -- feeding raw seed counts was one of v1's
mistakes.
"""

from . import engine

NUM_FEATURES = 14
NUM_ACTIONS = 6

# Scale each count by the starting seeds per pit, so a full starting pit reads as
# 1.0 and the network sees counts at near-integer resolution. The obvious
# alternative, dividing by the 48-seed total, squashes everything into [0, ~0.5]:
# the gap between 1 and 2 seeds -- which decides captures and extra turns --
# shrinks to about 0.02, too small for the network to use.
_SCALE = engine.STARTING_SEEDS  # 4

# Value target: the final store margin, from the mover's view, squashed to
# [-1, 1]. I train on the margin rather than a plain win/loss bit because in a
# first-player-win game every decent agent wins, so a win/loss target saturates
# and stops teaching; the margin keeps varying, so there is always a gradient,
# and it pushes the agent to convert a won game decisively instead of by one
# seed. This is the *final* score only, never a per-move reward -- rewarding the
# per-move store difference is exactly what made v1 imitate greedy.
# MARGIN_SCALE sets the scale: a final margin of MARGIN_SCALE maps to 1.0.
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
