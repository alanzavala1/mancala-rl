"""Cross-language tests for the C solver.

Requires the DLL to be built first (csolver/build.ps1). Two checks:
  1. The C game rules never diverge from the trusted Python engine.
  2. The C alpha-beta agrees with the Python exact solver where both can see
     the whole game (small endgame positions).

    python tests/test_csolver.py
    pytest tests/test_csolver.py
"""

import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from mancala_rl import engine
from mancala_rl.solver import Solver, legal_actions
from mancala_rl import csolver


def test_c_rules_match_engine():
    rng = random.Random(2024)
    for _ in range(300):
        state = engine.reset(first_player=rng.choice([1, 2]))
        board, player = state.board, state.current_player
        for _ in range(500):
            a = rng.choice(engine.legal_moves(state))
            state, _, done, info = engine.step(state, a)
            c_board, c_next = csolver.apply_one(board, player, a)
            assert tuple(state.board) == c_board
            if done:
                assert c_next == 0
                break
            assert c_next == state.current_player
            board, player = c_board, c_next


def _small_position(rng, max_seeds=6):
    pits = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
    while True:
        b = [0] * 14
        for _ in range(rng.randint(2, max_seeds)):
            b[rng.choice(pits)] += 1
        if sum(b[0:6]) > 0 and sum(b[7:13]) > 0:
            b[6] = rng.randint(0, 3)
            b[13] = rng.randint(0, 3)
            return tuple(b)


def test_c_search_matches_exact_solver_in_endgames():
    # On small positions the whole game fits in the search budget, so the C
    # depth-limited search must return the same value as the exact solver.
    rng = random.Random(11)
    for _ in range(200):
        board = _small_position(rng)
        player = rng.choice([1, 2])
        exact = Solver().value(board, player)
        state = engine.State(board, player)
        values = csolver.move_values(state, depth=40)
        legal = legal_actions(board, player)
        best = max(values[a] for a in legal) if player == 1 \
            else min(values[a] for a in legal)
        assert best == exact, f"{board} p{player}: C {best} != exact {exact}"


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
