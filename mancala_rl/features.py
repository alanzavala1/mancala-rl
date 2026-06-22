"""Board encoding for the network.

The network always sees the board from the perspective of the player to move
("my pits, my store, opponent's pits, opponent's store"). I encode it this way
so the network learns a single representation no matter which physical side is to
move, and so the six action slots always mean "my" pits. The counts are scaled
(see _SCALE) rather than fed raw -- feeding raw seed counts was one of v1's
mistakes.
"""

import math

from . import engine

NUM_FEATURES = 14
NUM_ACTIONS = 6
ACTION_FEATURES = 10
ACTION_AWARE_FEATURES = NUM_FEATURES + NUM_ACTIONS * ACTION_FEATURES

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
    return value_target(margin)


def value_target(margin, mode="clip", scale=MARGIN_SCALE):
    """Convert a mover-relative final margin into a value target.

    mode="clip" is the original target. mode="tanh" keeps large margins
    bounded while preserving a smoother gradient around the scale boundary.
    """
    if scale <= 0:
        raise ValueError("value target scale must be positive")
    if mode == "clip":
        v = margin / scale
        return 1.0 if v > 1.0 else (-1.0 if v < -1.0 else v)
    if mode == "tanh":
        return math.tanh(margin / scale)
    raise ValueError(f"unknown value target mode: {mode}")


def encode(state):
    """Return a 14-float list: [my 6 pits, my store, opp 6 pits, opp store]."""
    b = state.board
    if state.current_player == 1:
        mine, opp = b[0:7], b[7:14]   # A-F + store1, then G-L + store2
    else:
        mine, opp = b[7:14], b[0:7]
    return [x / _SCALE for x in (*mine, *opp)]


def _player_pit_totals(state, player):
    b = state.board
    if player == 1:
        return sum(b[0:6]), sum(b[7:13])
    return sum(b[7:13]), sum(b[0:6])


def _store_margin(state, player):
    s1, s2 = engine.stores(state)
    return (s1 - s2) if player == 1 else (s2 - s1)


def action_features(state):
    """Return exact per-action tactical features for the player to move.

    These are generated from the engine state rather than inferred from the
    perspective-canonical board vector. That keeps side-specific sowing,
    captures, extra turns, and terminal sweeps aligned with the real rules.
    """
    player = state.current_player
    legal = set(engine.legal_moves(state))
    before_store = engine.stores(state)[player - 1]
    own_before, opp_before = _player_pit_totals(state, player)
    remaining_before = own_before + opp_before
    rows = []

    for action in range(NUM_ACTIONS):
        if action not in legal:
            rows.append([0.0] * ACTION_FEATURES)
            continue

        next_state, _, done, info = engine.step(state, action)
        after_store = engine.stores(next_state)[player - 1]
        own_after, opp_after = _player_pit_totals(next_state, player)
        remaining_after = own_after + opp_after
        store_gain = after_store - before_store
        pit_seeds = state.board[action if player == 1 else action + 7]
        margin_after = _store_margin(next_state, player)
        remaining_drop = remaining_before - remaining_after

        rows.append([
            1.0,                                        # legal
            pit_seeds / 12.0,
            store_gain / 24.0,
            1.0 if info["extra_turn"] else 0.0,
            1.0 if done else 0.0,
            value_target(margin_after, mode="clip", scale=24.0),
            remaining_after / 48.0,
            own_after / 48.0,
            opp_after / 48.0,
            remaining_drop / 24.0,
        ])
    return rows


def encode_action_aware(state):
    """Return board features plus flattened per-action consequence features."""
    return encode(state) + [x for row in action_features(state) for x in row]


def encode_for_model(state, architecture="mlp"):
    """Encode a state for the requested network architecture."""
    if architecture == "action_aware":
        return encode_action_aware(state)
    if architecture == "mlp":
        return encode(state)
    raise ValueError(f"unknown architecture: {architecture}")
