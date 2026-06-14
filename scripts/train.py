"""Train the AlphaZero-style agent by self-play (pure self-play, multi-core, save-best).

Pure self-play -- the network only ever learns from games against itself, no
external opponents in the training data. Speed comes from running self-play
across CPU cores (--workers); the GPU doesn't help a network this small.

Each iteration: self-play (parallel) -> replay buffer -> train. Every
--eval-every iterations: measure the real signal (a seat-split vs the solver,
plus the saturated baselines for the curve) and save best.pt whenever the solver
score improves, so gate-less drift can't discard a good network.

Run:
    .venv\\Scripts\\python scripts/train.py --iterations 400 --sims 256 --workers 8 --out runs3

Outputs (under --out): champion.pt (latest), best.pt (best vs solver), training_log.csv.
"""

import argparse
import csv
import multiprocessing as mp
import os
import pathlib
import random
import sys
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor

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


def solver_seatsplit(agent, depth, n, seed, opening_plies=6):
    """(agent wins as first player, agent wins as second player) vs solver.

    opening_plies random opening moves vary each game so the metric isn't a
    near-binary replay of one deterministic line.
    """
    solver = CSolverBot(depth=depth)
    rng = random.Random(seed)
    first = sum(1 for _ in range(n) if play_game(agent, solver, rng, opening_plies=opening_plies) == 1)
    rng = random.Random(seed + 1)
    second = sum(1 for _ in range(n) if play_game(solver, agent, rng, opening_plies=opening_plies) == 2)
    return first, second


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--iterations", type=int, default=300)
    p.add_argument("--games", type=int, default=32)
    p.add_argument("--sims", type=int, default=256)
    p.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 8),
                   help="self-play worker processes (1 = serial)")
    p.add_argument("--train-steps", type=int, default=300)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--buffer", type=int, default=60000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--random-opening", type=int, default=8)
    p.add_argument("--temp-moves", type=int, default=12)
    p.add_argument("--dirichlet", type=float, default=0.3)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--eval-games", type=int, default=30)
    p.add_argument("--solver-depth", type=int, default=8)
    p.add_argument("--solver-games", type=int, default=40)
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
             "solver_first", "solver_second", "best", "seconds"])

    net = MancalaNet().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    buffer = deque(maxlen=args.buffer)
    best_score = -1

    use_pool = args.workers and args.workers > 1
    executor = ProcessPoolExecutor(
        max_workers=args.workers, mp_context=mp.get_context("spawn")) if use_pool else None

    print(f"device={device}  iterations={args.iterations}  games/iter={args.games}  "
          f"sims={args.sims}  workers={args.workers if use_pool else 1}\n", flush=True)

    try:
        for it in range(1, args.iterations + 1):
            t0 = time.perf_counter()

            net.eval()
            if use_pool:
                state_dict = {k: v.detach().cpu() for k, v in net.state_dict().items()}
                examples = selfplay.generate_parallel(
                    executor, state_dict, args.games, args.workers, args.sims,
                    args.dirichlet, args.temp_moves, args.random_opening, args.seed + it)
            else:
                sp_mcts = MCTS(net, device, n_simulations=args.sims,
                               dirichlet_alpha=args.dirichlet)
                examples = selfplay.generate(
                    sp_mcts, n_games=args.games, rng=random.Random(args.seed + it),
                    temperature_moves=args.temp_moves, random_opening=args.random_opening)
            buffer.extend(examples)

            net.train()
            buf_list = list(buffer)
            losses = []
            for _ in range(args.train_steps):
                batch = random.sample(buf_list, min(len(buf_list), args.batch))
                losses.append(train_step(net, opt, batch, device)[0])
            avg_loss = sum(losses) / len(losses)
            net.eval()
            torch.save(net.state_dict(), out / "champion.pt")

            do_eval = (it % args.eval_every == 0) or (it == args.iterations)
            if do_eval:
                agent = MCTSBot(net, device, args.sims)
                vr = evaluate(agent, RandomBot(), n_games=args.eval_games,
                              seed=args.seed + 2000 + it).score
                vg = evaluate(agent, GreedyBot(), n_games=args.eval_games,
                              seed=args.seed + 3000 + it).score
                sf, ss = solver_seatsplit(agent, args.solver_depth, args.solver_games,
                                          args.seed + 4000 + it)
                score = sf + ss
                new_best = score > best_score
                if new_best:
                    best_score = score
                    torch.save(net.state_dict(), out / "best.pt")
                dt = time.perf_counter() - t0
                n = args.solver_games
                print(f"iter {it:3d} | loss {avg_loss:5.3f} | vs R {vr:4.0%} G {vg:4.0%} | "
                      f"solver(d{args.solver_depth}) first {sf:2d}/{n} second {ss:2d}/{n}"
                      f"{'  << NEW BEST' if new_best else ''} | {dt:4.0f}s", flush=True)
                with open(csv_path, "a", newline="") as f:
                    csv.writer(f).writerow(
                        [it, len(buffer), f"{avg_loss:.4f}", f"{vr:.3f}", f"{vg:.3f}",
                         sf, ss, int(new_best), f"{dt:.1f}"])
            else:
                dt = time.perf_counter() - t0
                print(f"iter {it:3d} | loss {avg_loss:5.3f} | {dt:4.0f}s", flush=True)
                with open(csv_path, "a", newline="") as f:
                    csv.writer(f).writerow(
                        [it, len(buffer), f"{avg_loss:.4f}", "", "", "", "", "", f"{dt:.1f}"])
    finally:
        if executor is not None:
            executor.shutdown()

    print(f"\ndone. latest -> {out/'champion.pt'} ; best vs solver -> {out/'best.pt'} "
          f"(score {best_score}/{2*args.solver_games}) ; log -> {csv_path}", flush=True)


if __name__ == "__main__":
    main()
