"""Tests for the board encoding.

    python tests/test_features.py
    pytest tests/test_features.py
"""

import sys

from mancala_rl import engine, features
from mancala_rl.engine import State


def test_shape_and_opening_encoding():
    vec = features.encode(engine.reset())
    assert len(vec) == features.NUM_FEATURES == 14
    assert all(x >= 0.0 for x in vec)
    # 1.0 == a full starting pit; the opening is twelve full pits, empty stores.
    assert vec == [1.0] * 6 + [0.0] + [1.0] * 6 + [0.0]


def test_opening_is_symmetric():
    # In the opening both sides are identical, so own-half == opponent-half.
    vec = features.encode(engine.reset())
    assert vec[0:7] == vec[7:14]


def test_perspective_is_canonical():
    # A position seen by player 1 must encode identically to the mirror
    # position (sides swapped) seen by player 2.
    board = (1, 0, 3, 0, 2, 0, 5,    # A-F, store1
             0, 4, 0, 1, 0, 2, 3)    # G-L, store2
    s1 = State(board, 1)
    mirrored = board[7:14] + board[0:7]   # swap the two halves
    s2 = State(mirrored, 2)
    assert features.encode(s1) == features.encode(s2)


def test_to_move_half_comes_first():
    board = (1, 0, 3, 0, 2, 0, 5,
             0, 4, 0, 1, 0, 2, 3)
    # Player 2 to move: "my" half should be G-L + store2.
    vec = features.encode(State(board, 2))
    expected_mine = [x / features._SCALE for x in board[7:14]]
    assert vec[0:7] == expected_mine


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
