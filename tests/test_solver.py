"""Correctness tests for the solver.

Two load-bearing checks:
  1. The solver's fast board core never diverges from the proven engine.
  2. The optimized alpha-beta+TT solver returns exactly the same values as a
     dead-simple plain minimax, across many positions.

Runnable two ways:
    python tests/test_solver.py
    pytest tests/test_solver.py
"""

import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from mancala_rl import engine, solver
from mancala_rl.solver import apply_fast, legal_actions, minimax_plain, Solver


def test_fast_core_matches_engine():
    # Drive the engine and the solver's fast core through identical random
    # games; their boards, turns, and outcomes must agree at every step.
    rng = random.Random(123)
    for _ in range(300):
        first = rng.choice([1, 2])
        state = engine.reset(first_player=first)
        board, player = state.board, state.current_player
        for _ in range(500):
            assert engine.legal_moves(state) == legal_actions(board, player)
            a = rng.choice(engine.legal_moves(state))

            state, _, done, info = engine.step(state, a)
            fboard, fplayer, fdone, fmargin = apply_fast(board, player, a)

            assert tuple(state.board) == fboard
            assert done == fdone
            if done:
                winner = info["winner"]
                expected = 1 if fmargin > 0 else (2 if fmargin < 0 else 0)
                assert winner == expected
                break
            assert state.current_player == fplayer
            board, player = fboard, fplayer


def _small_position(rng, max_seeds=6):
    """A random non-terminal position with few seeds in play (cheap to solve)."""
    pits = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
    while True:
        b = [0] * 14
        for _ in range(rng.randint(2, max_seeds)):
            b[rng.choice(pits)] += 1
        if sum(b[0:6]) > 0 and sum(b[7:13]) > 0:   # both sides non-empty
            b[6] = rng.randint(0, 3)
            b[13] = rng.randint(0, 3)
            return tuple(b)


def test_alphabeta_equals_plain_minimax():
    rng = random.Random(7)
    for _ in range(400):
        board = _small_position(rng)
        player = rng.choice([1, 2])
        expected = minimax_plain(board, player)
        got = Solver().value(board, player)
        assert got == expected, f"{board} p{player}: {got} != {expected}"


def test_solve_returns_all_optimal_moves():
    # On a small position, every move the solver flags as optimal must actually
    # achieve the optimal value, and no other move may match it.
    rng = random.Random(99)
    for _ in range(100):
        board = _small_position(rng)
        player = rng.choice([1, 2])
        state = engine.State(board, player)
        value, best = solver.Solver().solve(state)
        for a in legal_actions(board, player):
            child, cp, done, margin = apply_fast(board, player, a)
            v = margin if done else Solver().value(child, cp)
            if a in best:
                assert v == value
            else:
                assert v != value


def test_forced_terminal_margin():
    # Player 1 has only F=1; playing it lands in store 1, empties P1, and the
    # five P2 seeds get swept -> final 1 vs 5, margin -4.
    board = tuple(0 if i not in (5, 7) else (1 if i == 5 else 5) for i in range(14))
    assert Solver().value(board, 1) == -4
    assert minimax_plain(board, 1) == -4


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
