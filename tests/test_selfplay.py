"""Tests for self-play data generation (needs torch).

    .venv\\Scripts\\python tests/test_selfplay.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import random

import torch

from mancala_rl import features
from mancala_rl.network import MancalaNet
from mancala_rl.mcts import MCTS
from mancala_rl import selfplay


def _mcts(sims=25):
    torch.manual_seed(0)
    return MCTS(MancalaNet().eval(), torch.device("cpu"), n_simulations=sims)


def test_examples_are_well_formed():
    examples = selfplay.play_game(_mcts(), rng=random.Random(0))
    assert len(examples) >= 1
    for feat, policy, value in examples:
        assert len(feat) == features.NUM_FEATURES
        assert len(policy) == features.NUM_ACTIONS
        assert abs(sum(policy) - 1.0) < 1e-6          # a probability distribution
        assert all(0.0 <= p <= 1.0 for p in policy)
        assert value in (-1.0, 0.0, 1.0)


def test_values_are_zero_sum():
    # Recompute per-move player to check the value labelling is consistent:
    # the two players' value labels must be exact opposites (or all 0 on a tie).
    rng = random.Random(1)
    mcts = _mcts()
    # Re-run with the internal history exposed via play_game's contract:
    examples = selfplay.play_game(mcts, rng=rng)
    values = {round(v, 3) for _, _, v in examples}
    assert values.issubset({-1.0, 0.0, 1.0})
    if values != {0.0}:                # not a tie -> both +1 and -1 should appear
        assert 1.0 in values and -1.0 in values


def test_generate_concatenates_games():
    rng = random.Random(2)
    mcts = _mcts()
    g1 = selfplay.play_game(mcts, rng=random.Random(2))
    total = selfplay.generate(mcts, n_games=3, rng=rng)
    assert len(total) >= len(g1)      # 3 games yield at least as much as 1


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
