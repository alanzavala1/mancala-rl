"""Correctness tests for the engine port.

Runnable two ways:
    python tests/test_engine.py        # plain, no dependencies
    pytest tests/test_engine.py        # if pytest is installed
"""

import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from mancala_rl import engine
from mancala_rl.engine import State, BOARD_ORDER


def _state(counts, player):
    """Build a State from a {pit: count} dict, defaulting unset pits to 0."""
    board = tuple(counts.get(k, 0) for k in BOARD_ORDER)
    return State(board, player)


def test_reset_is_standard_opening():
    s = engine.reset()
    assert s.current_player == 1
    assert engine.stores(s) == (0, 0)
    assert sum(s.board) == 48
    assert all(s.board[i] == 4 for i in range(6))         # A-F
    assert all(s.board[i] == 4 for i in range(7, 13))     # G-L


def test_all_moves_legal_at_start():
    assert engine.legal_moves(engine.reset()) == [0, 1, 2, 3, 4, 5]


def test_landing_in_own_store_grants_extra_turn():
    # Player 1: a single seed in F (right before store 1). Extra seeds on the
    # board keep both sides non-empty so the game does not end on this move.
    s = _state({'F': 1, 'A': 2, 'G': 3}, player=1)
    ns, reward, done, info = engine.step(s, 5)  # action 5 == pit F
    assert info['extra_turn'] is True
    assert ns.current_player == 1        # same player acts again
    assert engine.stores(ns)[0] == 1     # the seed is in store 1
    assert reward == 0.0 and done is False


def test_capture_into_own_empty_pit():
    # Player 1 plays B (1 seed) -> lands in empty C; opposite of C is I (3).
    # G=2 keeps player 2 non-empty so the capture does not also end the game.
    s = _state({'B': 1, 'C': 0, 'I': 3, 'A': 1, 'G': 2}, player=1)
    ns, _, _, info = engine.step(s, 1)  # action 1 == pit B
    d = dict(zip(BOARD_ORDER, ns.board))
    assert d['1'] == 4    # captured 1 (own) + 3 (opposite) into store 1
    assert d['C'] == 0
    assert d['I'] == 0


def test_opponent_store_is_skipped():
    # A long sow from A must never deposit into player 2's store.
    s = _state({'A': 20}, player=1)
    ns, _, _, _ = engine.step(s, 0)
    assert engine.stores(ns)[1] == 0


def test_terminal_sweep_and_reward():
    # Player 1 has only F=1; playing it empties P1, so P2's seeds get swept.
    s = _state({'F': 1, 'G': 3}, player=1)
    ns, reward, done, info = engine.step(s, 5)
    assert done is True
    assert info['winner'] == 2
    assert engine.stores(ns) == (1, 3)   # 1 in store1, 3 swept into store2
    assert reward == -1.0                # from the mover's (player 1) view


def test_step_does_not_mutate_input_state():
    s = engine.reset()
    before = s.board
    engine.step(s, 0)
    assert s.board == before


def test_random_game_conserves_seeds_and_terminates():
    rng = random.Random(0)
    state = engine.reset()
    for _ in range(1000):
        state, _, done, _ = engine.step(state, rng.choice(engine.legal_moves(state)))
        if done:
            break
    else:
        raise AssertionError("game did not terminate within 1000 moves")
    assert sum(state.board) == 48


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return failures


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
