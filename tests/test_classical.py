"""Tests for the classical (no-network) MCTS ablation agent.

    python tests/test_classical.py
"""

import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from mancala_rl import engine
from mancala_rl.bots import RandomBot
from mancala_rl.classical import ClassicalMCTSBot
from mancala_rl.evaluate import evaluate


def test_returns_legal_moves():
    bot = ClassicalMCTSBot(n_simulations=30, rollout="random")
    state = engine.reset()
    a = bot.act(state, random.Random(0))
    assert a in engine.legal_moves(state)


def test_classical_mcts_beats_random():
    # A working search with rollouts must crush a random mover.
    bot = ClassicalMCTSBot(n_simulations=150, rollout="random")
    result = evaluate(bot, RandomBot(), n_games=40, seed=0)
    assert result.score > 0.75, result.score


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
