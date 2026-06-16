"""Plot the training curve from a training_log.csv.

Two panels: training loss, and strength vs the depth-8 solver (win rate as first
and second player, varied openings) over iterations. Saved as a PNG for the README.

    .venv\\Scripts\\python scripts/plot_training.py --log runs_final/training_log.csv
"""

import argparse
import csv
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", default="runs/training_log.csv")
    p.add_argument("--out", default="assets/training_curve.png")
    p.add_argument("--solver-games", type=int, default=40)
    p.add_argument("--solver-depth", type=int, default=8)
    args = p.parse_args()

    iters, loss = [], []
    eval_iters, sfirst, ssecond = [], [], []
    with open(args.log, newline="") as f:
        for row in csv.DictReader(f):
            iters.append(int(row["iteration"]))
            loss.append(float(row["loss"]))
            if row["solver_first"]:
                eval_iters.append(int(row["iteration"]))
                sfirst.append(100 * int(row["solver_first"]) / args.solver_games)
                ssecond.append(100 * int(row["solver_second"]) / args.solver_games)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(iters, loss, color="tab:blue")
    ax1.set_title("Training loss")
    ax1.set_xlabel("iteration")
    ax1.set_ylabel("loss")
    ax1.grid(alpha=0.3)

    ax2.plot(eval_iters, sfirst, "-o", ms=3, label="as 1st player")
    ax2.plot(eval_iters, ssecond, "-o", ms=3, label="as 2nd player")
    ax2.axhline(50, color="gray", ls="--", lw=1)
    ax2.set_title(f"Win rate vs depth-{args.solver_depth} solver (varied openings)")
    ax2.set_xlabel("iteration")
    ax2.set_ylabel("win rate (%)")
    ax2.set_ylim(0, 100)
    ax2.legend()
    ax2.grid(alpha=0.3)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print("saved", out)


if __name__ == "__main__":
    main()
