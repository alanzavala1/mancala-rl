"""Measure a trained champion vs the solver across search depths, split by seat.

Kalah(6,4) is a first-player win, so a fair aggregate vs a strong opponent sits
near 50%. The informative view is split by who moves first:
  - "agent first": how deep a solver can the agent beat WITH the first-move edge
    (its offense / ability to convert a won game).
  - "agent second": how deep a solver can it beat FROM the losing seat
    (its defense).

    .venv\\Scripts\\python scripts/depth_sweep.py
    .venv\\Scripts\\python scripts/depth_sweep.py --games 40 --depths 4 6 8 10 12
"""

import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import random
import torch

from mancala_rl.network import load_net
from mancala_rl.mcts import MCTSBot
from mancala_rl.csolver import CSolverBot
from mancala_rl.evaluate import play_game


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--champion", default="runs/champion.pt")
    p.add_argument("--games", type=int, default=40, help="games per seat per depth")
    p.add_argument("--sims", type=int, default=100)
    p.add_argument("--depths", type=int, nargs="+", default=[4, 6, 8, 10, 12])
    p.add_argument("--opening-plies", type=int, default=6,
                   help="random opening moves per game, to vary otherwise-identical matches")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    net = load_net(args.champion, device)
    agent = MCTSBot(net, device, n_simulations=args.sims)

    n = args.games
    print(f"\nagent ({args.sims} sims) vs Solver, {n} games per seat per depth\n", flush=True)
    print(f"  {'depth':>5} {'agent first':>14} {'agent second':>14} {'time':>8}", flush=True)
    for d in args.depths:
        solver = CSolverBot(depth=d)
        t0 = time.perf_counter()
        op = args.opening_plies
        rng = random.Random(args.seed)
        first = sum(1 for _ in range(n) if play_game(agent, solver, rng, opening_plies=op) == 1)
        rng = random.Random(args.seed + 1)
        second = sum(1 for _ in range(n) if play_game(solver, agent, rng, opening_plies=op) == 2)
        dt = time.perf_counter() - t0
        print(f"  {d:>5} {f'{first}/{n}':>14} {f'{second}/{n}':>14} {f'{dt:.0f}s':>8}",
              flush=True)


if __name__ == "__main__":
    main()
