"""Tests for MCTS (needs torch).

The key correctness check: even with an UNtrained (random) network, MCTS with
enough simulations must find a move that wins immediately, because the search
reaches the terminal position and reads its exact +1 result. We cross-check the
winning move with the exact solver.

    .venv\\Scripts\\python tests/test_mcts.py
"""

import random
import sys

import torch

from mancala_rl import engine
from mancala_rl.engine import State
from mancala_rl.network import MancalaNet
from mancala_rl.mcts import MCTS, best_action, visit_counts
from mancala_rl import reference


# Player 1 to move. Playing B (action 1) sows into the empty C, captures the 3
# seeds opposite in I, which empties player 2 -> game ends with player 1 ahead.
# Playing A (action 0) does not end the game. So B is a forced win-in-one.
_FORCED_WIN = State((1, 1, 0, 0, 0, 0, 0,
                     0, 0, 3, 0, 0, 0, 0), 1)


def _net():
    torch.manual_seed(0)            # deterministic random weights
    return MancalaNet().eval()


def test_reference_confirms_the_win():
    # Sanity: action 1 really is (uniquely) optimal here.
    value = reference.minimax(_FORCED_WIN)
    best = reference.best_moves(_FORCED_WIN)
    assert value > 0 and best == [1]


def test_mcts_finds_forced_win():
    mcts = MCTS(_net(), torch.device("cpu"), n_simulations=300)
    root = mcts.search(_FORCED_WIN)
    assert best_action(root, random.Random(0)) == 1


def test_visit_counts_are_well_formed():
    mcts = MCTS(_net(), torch.device("cpu"), n_simulations=200)
    root = mcts.search(engine.reset())
    counts = visit_counts(root)
    assert set(counts) == set(engine.legal_moves(engine.reset()))
    assert sum(counts.values()) == 200      # one root edge visited per simulation


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
