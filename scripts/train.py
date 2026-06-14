"""Train the AlphaZero-style agent by self-play.

One ITERATION is the whole AlphaZero cycle:
  1. the champion network plays games against itself -> training examples
  2. a copy ("challenger") trains on those examples
  3. the challenger must beat the champion to be promoted (the "gate")
  4. measure the champion vs the baselines, log it, save a checkpoint
Repeat. Over iterations the "vs Greedy" score should climb -- that's learning.

Run from the repo root with the project venv:

    .venv\\Scripts\\python scripts/train.py                  # sensible defaults
    .venv\\Scripts\\python scripts/train.py --iterations 40 --games 25 --sims 80
    .venv\\Scripts\\python scripts/train.py --help           # all options

Outputs (under runs/, which is gitignored):
    runs/training_log.csv   one row per iteration (for the training-curve plot)
    runs/champion.pt        the current best network's weights
"""

import argparse
import copy
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
from mancala_rl.evaluate import evaluate
from mancala_rl.bots import RandomBot, GreedyBot


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--games", type=int, default=15, help="self-play games per iteration")
    p.add_argument("--sims", type=int, default=40, help="MCTS simulations per move")
    p.add_argument("--eval-games", type=int, default=20, help="games per evaluation")
    p.add_argument("--train-steps", type=int, default=150, help="gradient steps per iteration")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--buffer", type=int, default=20000, help="replay buffer capacity")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--promote", type=float, default=0.55,
                   help="challenger score vs champion needed to promote")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu", help="cpu (faster here) or cuda")
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
            ["iteration", "buffer", "loss", "challenger_vs_champion",
             "promoted", "vs_random", "vs_greedy", "seconds"])

    champion = MancalaNet().to(device).eval()
    buffer = deque(maxlen=args.buffer)

    print(f"device={device}  iterations={args.iterations}  "
          f"games/iter={args.games}  sims={args.sims}\n", flush=True)

    for it in range(1, args.iterations + 1):
        t0 = time.perf_counter()

        # 1) SELF-PLAY with the champion (Dirichlet noise -> extra exploration)
        sp_mcts = MCTS(champion, device, n_simulations=args.sims, dirichlet_alpha=0.3)
        buffer.extend(selfplay.generate(
            sp_mcts, n_games=args.games, rng=random.Random(args.seed + it)))

        # 2) TRAIN a challenger (a copy of the champion) on replay samples
        challenger = copy.deepcopy(champion).train()
        opt = torch.optim.Adam(challenger.parameters(), lr=args.lr, weight_decay=1e-4)
        buf_list = list(buffer)                     # cheap random.sample source
        losses = []
        for _ in range(args.train_steps):
            batch = random.sample(buf_list, min(len(buf_list), args.batch))
            losses.append(train_step(challenger, opt, batch, device)[0])
        challenger.eval()
        avg_loss = sum(losses) / len(losses)

        # 3) GATE: promote the challenger only if it clearly beats the champion
        gate = evaluate(MCTSBot(challenger, device, args.sims),
                        MCTSBot(champion, device, args.sims),
                        n_games=args.eval_games, seed=args.seed + 1000 + it)
        promoted = gate.score > args.promote
        if promoted:
            champion = challenger

        # 4) MEASURE the champion vs the baselines (this is the training curve)
        vs_random = evaluate(MCTSBot(champion, device, args.sims), RandomBot(),
                             n_games=args.eval_games, seed=args.seed + 2000 + it)
        vs_greedy = evaluate(MCTSBot(champion, device, args.sims), GreedyBot(),
                             n_games=args.eval_games, seed=args.seed + 3000 + it)

        dt = time.perf_counter() - t0
        torch.save(champion.state_dict(), out / "champion.pt")
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [it, len(buffer), f"{avg_loss:.4f}", f"{gate.score:.3f}",
                 int(promoted), f"{vs_random.score:.3f}", f"{vs_greedy.score:.3f}",
                 f"{dt:.1f}"])

        print(f"iter {it:3d} | loss {avg_loss:5.3f} | "
              f"challenger vs champion {gate.score:5.1%} "
              f"{'PROMOTED' if promoted else 'kept':8s} | "
              f"champion vs Random {vs_random.score:5.1%}  vs Greedy {vs_greedy.score:5.1%} "
              f"| {dt:4.0f}s", flush=True)

    print(f"\ndone. champion -> {out/'champion.pt'} ; log -> {csv_path}")


if __name__ == "__main__":
    main()
