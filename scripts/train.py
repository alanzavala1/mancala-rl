"""Train the AlphaZero-style agent by self-play ("Option B": gate-less + diversified).

Designed from what the depth sweep revealed about the first agent (strong
offense, brittle, no defense):

  - GATE-LESS: one network, continuously trained and used for self-play. The
    earlier gated loop froze the champion early; keeping the latest network keeps
    the self-play data fresh, so learning doesn't stall.
  - DIVERSIFIED STARTS: each self-play game begins with a few random, unrecorded
    opening moves (--random-opening), forcing the agent through varied and
    *disadvantaged* positions -- the cure for brittleness and for not defending.

One iteration: self-play -> replay buffer -> train -> measure vs the baselines
(and, every --solver-every iterations, a seat-split vs the solver) -> checkpoint.

Run from the repo root with the venv:
    .venv\\Scripts\\python scripts/train.py
    .venv\\Scripts\\python scripts/train.py --iterations 150 --games 30 --sims 100

Outputs (under runs/, gitignored): training_log.csv and champion.pt.
"""

import argparse
import csv
import pathlib
import random
import sys
import time
from collections import deque

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from mancala_rl import selfplay
from mancala_rl.network import MancalaNet
from mancala_rl.mcts import MCTS, MCTSBot
from mancala_rl.training import train_step
from mancala_rl.evaluate import evaluate, play_game
from mancala_rl.bots import RandomBot, GreedyBot
from mancala_rl.csolver import CSolverBot


def solver_seatsplit(agent, depth, n, seed):
    """Return (agent wins as first player, agent wins as second player) vs solver."""
    solver = CSolverBot(depth=depth)
    rng = random.Random(seed)
    first = sum(1 for _ in range(n) if play_game(agent, solver, rng) == 1)
    rng = random.Random(seed + 1)
    second = sum(1 for _ in range(n) if play_game(solver, agent, rng) == 2)
    return first, second


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--iterations", type=int, default=80)
    p.add_argument("--games", type=int, default=30, help="self-play games per iteration")
    p.add_argument("--sims", type=int, default=80, help="MCTS simulations per move")
    p.add_argument("--eval-games", type=int, default=30)
    p.add_argument("--train-steps", type=int, default=300)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--buffer", type=int, default=50000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--random-opening", type=int, default=8,
                   help="max random unrecorded opening moves per self-play game")
    p.add_argument("--temp-moves", type=int, default=12,
                   help="plies sampled by visit count before playing greedily")
    p.add_argument("--dirichlet", type=float, default=0.3, help="root exploration noise")
    p.add_argument("--solver-every", type=int, default=10,
                   help="measure a seat-split vs the solver every N iterations (0=never)")
    p.add_argument("--solver-depth", type=int, default=8)
    p.add_argument("--solver-games", type=int, default=20, help="games per seat in the solver check")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="runs")
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "training_log.csv"
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["iteration", "buffer", "loss", "vs_random", "vs_greedy",
             "solver_first", "solver_second", "seconds"])

    net = MancalaNet().to(device)                      # one network, trained in place
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    buffer = deque(maxlen=args.buffer)

    print(f"device={device}  iterations={args.iterations}  games/iter={args.games}  "
          f"sims={args.sims}  random-opening={args.random_opening}\n", flush=True)

    for it in range(1, args.iterations + 1):
        t0 = time.perf_counter()

        # SELF-PLAY with the current network (noise + random openings for variety)
        net.eval()
        sp_mcts = MCTS(net, device, n_simulations=args.sims, dirichlet_alpha=args.dirichlet)
        buffer.extend(selfplay.generate(
            sp_mcts, n_games=args.games, rng=random.Random(args.seed + it),
            temperature_moves=args.temp_moves, random_opening=args.random_opening))

        # TRAIN the same network on replay samples (no gate; keep the latest)
        net.train()
        buf_list = list(buffer)
        losses = []
        for _ in range(args.train_steps):
            batch = random.sample(buf_list, min(len(buf_list), args.batch))
            losses.append(train_step(net, opt, batch, device)[0])
        avg_loss = sum(losses) / len(losses)

        # MEASURE vs the baselines
        net.eval()
        agent = MCTSBot(net, device, args.sims)
        vs_random = evaluate(agent, RandomBot(), n_games=args.eval_games,
                             seed=args.seed + 2000 + it).score
        vs_greedy = evaluate(agent, GreedyBot(), n_games=args.eval_games,
                             seed=args.seed + 3000 + it).score

        sf = ss = None
        if args.solver_every and it % args.solver_every == 0:
            sf, ss = solver_seatsplit(agent, args.solver_depth,
                                      args.solver_games, args.seed + 4000 + it)

        dt = time.perf_counter() - t0
        torch.save(net.state_dict(), out / "champion.pt")
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [it, len(buffer), f"{avg_loss:.4f}", f"{vs_random:.3f}", f"{vs_greedy:.3f}",
                 "" if sf is None else sf, "" if ss is None else ss, f"{dt:.1f}"])

        line = (f"iter {it:3d} | loss {avg_loss:5.3f} | "
                f"vs Random {vs_random:5.1%}  vs Greedy {vs_greedy:5.1%}")
        if sf is not None:
            n = args.solver_games
            line += (f" | solver(d{args.solver_depth}) "
                     f"first {sf}/{n} second {ss}/{n}")
        line += f" | {dt:4.0f}s"
        print(line, flush=True)

    print(f"\ndone. champion -> {out/'champion.pt'} ; log -> {csv_path}")


if __name__ == "__main__":
    main()
