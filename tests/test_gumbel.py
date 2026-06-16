"""Tests for the Gumbel AlphaZero search (needs torch).

    .venv\\Scripts\\python tests/test_gumbel.py
"""

import random
import sys

import torch

from mancala_rl import engine, features
from mancala_rl.engine import State
from mancala_rl.network import MancalaNet
from mancala_rl.gumbel import GumbelMCTS


def _net():
    torch.manual_seed(0)
    return MancalaNet().eval()


# Player 1 to move; B (action 1) lands in empty C, captures opposite, ends the
# game with player 1 ahead -> a forced win in one. Same position as test_mcts.
_FORCED_WIN = State((1, 1, 0, 0, 0, 0, 0,
                     0, 0, 3, 0, 0, 0, 0), 1)


def test_gumbel_finds_forced_win():
    g = GumbelMCTS(_net(), torch.device("cpu"), n_simulations=64, m=6,
                   rng=random.Random(0))
    action, _, _ = g.search(_FORCED_WIN)
    assert action == 1


def test_gumbel_policy_is_well_formed():
    g = GumbelMCTS(_net(), torch.device("cpu"), n_simulations=48, m=16,
                   rng=random.Random(1))
    action, policy, value = g.search(engine.reset())
    legal = engine.legal_moves(engine.reset())
    assert action in legal
    assert len(policy) == features.NUM_ACTIONS
    assert abs(sum(policy) - 1.0) < 1e-6
    assert all(0.0 <= p <= 1.0 for p in policy)
    assert all(policy[a] == 0.0 for a in range(6) if a not in legal)
    assert -1.0 <= value <= 1.0


def test_gumbel_single_legal_move():
    # Only one legal move -> must return it with an all-on-it policy.
    s = State((0, 0, 0, 0, 0, 2, 0,
               3, 0, 0, 0, 0, 0, 0), 1)          # only pit F (action 5) is legal
    g = GumbelMCTS(_net(), torch.device("cpu"), n_simulations=16, rng=random.Random(2))
    action, policy, _ = g.search(s)
    assert action == 5
    assert policy[5] == 1.0


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
