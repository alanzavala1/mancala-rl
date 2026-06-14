"""Raw-network strength vs model size, across trained checkpoints.

For each checkpoint, evaluate the RAW policy (no search) vs the baselines and
report parameter count / size. Plots strength vs size -- "how small can the
network be and still be good?" Strength here is the network's own skill,
isolated from search (that's why we use PolicyBot, not MCTS).

    .venv\\Scripts\\python scripts/size_sweep.py --champions runs_h16/best.pt runs_h32/best.pt runs_h64/best.pt runs_final/best.pt
"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mancala_rl.network import load_net, PolicyBot
from mancala_rl.bots import RandomBot, GreedyBot
from mancala_rl.evaluate import evaluate


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--champions", nargs="+", required=True)
    p.add_argument("--games", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="assets/size_sweep.png")
    args = p.parse_args()

    rows = []
    print(f"\n{'shape':>10} {'params':>8} {'KB':>7} {'vs Greedy':>10} {'vs Random':>10}")
    for path in args.champions:
        net = load_net(path)
        params = sum(t.numel() for t in net.parameters())
        kb = params * 4 / 1024                      # float32
        bot = PolicyBot(net)
        g = evaluate(bot, GreedyBot(), args.games, args.seed).win_rate * 100
        r = evaluate(bot, RandomBot(), args.games, args.seed).win_rate * 100
        rows.append((net.hidden, params, kb, g, r))
        print(f"{f'{net.hidden}x{net.layers}':>10} {params:8,} {kb:7.1f} "
              f"{g:9.1f}% {r:9.1f}%", flush=True)

    rows.sort(key=lambda x: x[1])
    kbs = [x[2] for x in rows]
    G = [x[3] for x in rows]
    R = [x[4] for x in rows]
    widths = [x[0] for x in rows]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(kbs, G, "-o", label="vs Greedy")
    ax.plot(kbs, R, "-o", label="vs Random")
    for x, y, w in zip(kbs, G, widths):
        ax.annotate(f"{w}w", (x, y), textcoords="offset points", xytext=(0, 7), fontsize=8)
    ax.axhline(50, color="gray", ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("model size (KB, float32, log scale)")
    ax.set_ylabel("raw-net win rate (%)")
    ax.set_ylim(0, 105)
    ax.set_title("How small can the network be? (raw policy, no search)")
    ax.legend()
    ax.grid(alpha=0.3, which="both")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
