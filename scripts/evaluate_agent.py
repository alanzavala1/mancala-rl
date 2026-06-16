"""Evaluate a trained champion against the baselines and the solver.

Loads a saved network, wraps it in MCTS, and plays high-game-count matches vs
RandomBot, GreedyBot, and the C solver -- each reported as a win rate with a
95% confidence interval. The solver match is the real test of strength.

    .venv\\Scripts\\python scripts/evaluate_agent.py
    .venv\\Scripts\\python scripts/evaluate_agent.py --games 1000 --sims 100 --solver-depth 8
"""

import argparse

import torch

from mancala_rl.network import load_net
from mancala_rl.mcts import MCTSBot
from mancala_rl.bots import RandomBot, GreedyBot
from mancala_rl.csolver import CSolverBot
from mancala_rl.evaluate import evaluate


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--champion", default="runs/best.pt")
    p.add_argument("--games", type=int, default=400)
    p.add_argument("--sims", type=int, default=100, help="agent MCTS simulations")
    p.add_argument("--solver-depth", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    net = load_net(args.champion, device)
    agent = MCTSBot(net, device, n_simulations=args.sims)

    opponents = [
        RandomBot(),
        GreedyBot(),
        CSolverBot(depth=args.solver_depth),
    ]

    print(f"\nchampion = {args.champion}   (agent MCTS sims = {args.sims})")
    print(f"games per matchup = {args.games}, seed = {args.seed}\n")
    for opp in opponents:
        result = evaluate(agent, opp, n_games=args.games, seed=args.seed)
        print("  " + result.line())


if __name__ == "__main__":
    main()
