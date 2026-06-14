"""Map the strength-vs-search frontier for a trained network.

For a fixed network, measure playing strength (win rate vs the baselines) and
per-move latency across MCTS simulation budgets -- from 0 (raw policy, no search)
up to full search. Shows how much strength each extra bit of compute buys: the
core "efficient / lightweight" result.

    .venv\\Scripts\\python scripts/strength_vs_search.py --champion runs_final/best.pt
"""

import argparse
import pathlib
import random
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mancala_rl.network import MancalaNet, PolicyBot
from mancala_rl.mcts import MCTSBot
from mancala_rl.bots import RandomBot, GreedyBot
from mancala_rl.evaluate import evaluate
from mancala_rl import engine


def latency_us(bot, reps):
    """Average per-move latency in microseconds, from the opening position."""
    state = engine.reset()
    rng = random.Random(0)
    for _ in range(3):
        bot.act(state, rng)                       # warmup
    t0 = time.perf_counter()
    for _ in range(reps):
        bot.act(state, rng)
    return (time.perf_counter() - t0) / reps * 1e6


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--champion", default="runs_final/best.pt")
    p.add_argument("--games", type=int, default=200)
    p.add_argument("--sims", type=int, nargs="+",
                   default=[0, 1, 2, 4, 8, 16, 32, 64, 128, 256])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="assets/strength_vs_search.png")
    args = p.parse_args()

    device = torch.device("cpu")
    net = MancalaNet().to(device)
    net.load_state_dict(torch.load(args.champion, map_location=device))
    net.eval()

    sims, vs_g, vs_r, lat = [], [], [], []
    print(f"\n{'sims':>5} {'vs Greedy':>10} {'vs Random':>10} {'latency':>12}")
    for s in args.sims:
        bot = PolicyBot(net, device) if s == 0 else MCTSBot(net, device, n_simulations=s)
        g = evaluate(bot, GreedyBot(), n_games=args.games, seed=args.seed).win_rate * 100
        r = evaluate(bot, RandomBot(), n_games=args.games, seed=args.seed).win_rate * 100
        reps = 30 if s >= 64 else (100 if s >= 8 else 1000)
        ms = latency_us(bot, reps)
        sims.append(s); vs_g.append(g); vs_r.append(r); lat.append(ms)
        label = "raw net" if s == 0 else f"{s}"
        print(f"{label:>5} {g:9.1f}% {r:9.1f}% {ms:9.0f} us", flush=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    xs = list(range(len(sims)))
    ax1.plot(xs, vs_g, "-o", label="vs Greedy")
    ax1.plot(xs, vs_r, "-o", label="vs Random")
    ax1.set_xticks(xs)
    ax1.set_xticklabels(["raw" if s == 0 else str(s) for s in sims])
    ax1.set_xlabel("MCTS simulations per move")
    ax1.set_ylabel("win rate (%)")
    ax1.set_ylim(0, 105)
    ax1.set_title("Strength vs search budget")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(lat, vs_g, "-o", label="vs Greedy")
    ax2.plot(lat, vs_r, "-o", label="vs Random")
    ax2.set_xscale("log")
    ax2.set_xlabel("latency (us / move, log scale)")
    ax2.set_ylabel("win rate (%)")
    ax2.set_ylim(0, 105)
    ax2.set_title("Efficiency frontier (strength vs cost)")
    ax2.legend()
    ax2.grid(alpha=0.3, which="both")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
