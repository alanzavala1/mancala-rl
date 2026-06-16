"""Measure the C solver's strength vs the baselines, by search depth.

Shows the depth -> strength dial and the wall-clock cost. vs Greedy is reported
with its confidence interval (the interesting matchup); vs Random with score.

Usage:
    python scripts/measure_csolver.py
"""

import time

from mancala_rl.bots import RandomBot, GreedyBot
from mancala_rl.csolver import CSolverBot
from mancala_rl.evaluate import evaluate

N = 200
DEPTHS = (2, 4, 6, 8)

print(f"\nC solver vs baselines  (n={N} per matchup, seed=0)\n", flush=True)
for depth in DEPTHS:
    t0 = time.perf_counter()
    vs_random = evaluate(CSolverBot(depth), RandomBot(), n_games=N, seed=0)
    vs_greedy = evaluate(CSolverBot(depth), GreedyBot(), n_games=N, seed=0)
    dt = time.perf_counter() - t0
    lo, hi = vs_greedy.ci
    print(f"  depth {depth:2d}:  vs Random score {vs_random.score:6.1%}   "
          f"vs Greedy {vs_greedy.win_rate:6.1%} "
          f"[{lo:.0%}, {hi:.0%}]   ({dt:.1f}s)", flush=True)
