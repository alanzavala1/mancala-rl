"""Run the baseline matchups and print a reference table.

Usage (from the repo root):

    python scripts/run_baselines.py
    python scripts/run_baselines.py --games 5000 --seed 1

Two of the matchups are mirror matches (Random vs Random, Greedy vs Greedy).
With seat-swapping these should land near 50%, which is the harness's own
sanity check. Greedy vs Random is the reference line the learning agent will
have to clear.
"""

import argparse

from mancala_rl.bots import RandomBot, GreedyBot
from mancala_rl.evaluate import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=2000,
                        help="games per matchup (default 2000)")
    parser.add_argument("--seed", type=int, default=0,
                        help="master RNG seed (default 0)")
    args = parser.parse_args()

    matchups = [
        ("sanity: should be ~50%", RandomBot(), RandomBot()),
        ("sanity: should be ~50%", GreedyBot(), GreedyBot()),
        ("reference line", GreedyBot(), RandomBot()),
    ]

    print(f"\nBaseline matchups  (games={args.games}, seed={args.seed})\n")
    for note, a, b in matchups:
        result = evaluate(a, b, n_games=args.games, seed=args.seed)
        print(f"  {result.line()}")
        print(f"      {note}\n")


if __name__ == "__main__":
    main()
