"""Tournament across a pool of agents, with Elo and a seat split.

By default this runs a full round-robin: every agent plays every other agent,
seat-swapped, with varied openings. For fast model iteration, --schedule focused
only plays the matchups involving the main network-guided agent. That preserves
the headline evidence while skipping low-value baseline-vs-baseline games.

The pool spans genuinely different algorithms so the number means something
relative to a spread of strategies:

  - Random            -- the floor
  - Greedy            -- a 1-ply store-differential heuristic
  - AB-d{N}           -- alpha-beta search at depth N (a different algorithm)
  - Net-raw           -- our policy network, no search (one forward pass)
  - Net-mcts{S}       -- our agent: MCTS + the network at S simulations

Every pairing is split by seat, so you can see each agent's score as player 1
vs as player 2. Three matrices plus the ratings are written to --out as CSV.

    .venv\\Scripts\\python scripts/tournament.py --champion runs/best.pt
    .venv\\Scripts\\python scripts/tournament.py --champion runs/best.pt --games 50 --classical
    .venv\\Scripts\\python scripts/tournament.py --champion runs/best.pt --schedule focused --games 100
"""

import argparse
import csv
import itertools
import math
import pathlib
import random

import torch

from mancala_rl.bots import RandomBot, GreedyBot
from mancala_rl.csolver import CSolverBot
from mancala_rl.mcts import MCTSBot
from mancala_rl.network import load_net, PolicyBot
from mancala_rl.evaluate import play_game


def build_pool(net, device, sims, depths, classical=False):
    pool = [("Random", RandomBot()), ("Greedy", GreedyBot())]
    for d in depths:
        pool.append((f"AB-d{d}", CSolverBot(depth=d)))
    if classical:                          # no-network MCTS at the same sim budget
        from mancala_rl.classical import ClassicalMCTSBot
        pool.append((f"MCTSrand{sims}", ClassicalMCTSBot(sims, rollout="random")))
        pool.append((f"MCTSgrdy{sims}", ClassicalMCTSBot(sims, rollout="greedy")))
    pool.append(("Net-raw", PolicyBot(net, device)))
    pool.append((f"Net-mcts{sims}", MCTSBot(net, device, n_simulations=sims)))
    return pool


def seat_score(p1, p2, games, opening_plies, rng):
    """Score rate of the player-1 agent over `games` games (draws = 0.5)."""
    s = 0.0
    for _ in range(games):
        w = play_game(p1, p2, rng, opening_plies=opening_plies)
        s += 1.0 if w == 1 else (0.5 if w == 0 else 0.0)
    return s / games


def fit_elo(score, games, anchor=1500.0, iters=5000, prior=0.5):
    """Bradley-Terry MLE (minorization-maximization), converted to Elo."""
    n = len(score)
    p = [1.0] * n
    wins = [
        sum(score[i][j] + prior for j in range(n) if games[i][j] > 0)
        for i in range(n)
    ]
    for _ in range(iters):
        nxt = list(p)
        for i in range(n):
            denom = sum((games[i][j] + 2 * prior) / (p[i] + p[j])
                        for j in range(n) if j != i and games[i][j] > 0)
            if denom > 0:
                nxt[i] = wins[i] / denom
        nxt = [max(x, 1e-12) for x in nxt]
        logmean = sum(math.log(x) for x in nxt) / n     # keep the scale anchored
        p = [x / math.exp(logmean) for x in nxt]
    return [anchor + 400.0 * math.log10(pi) for pi in p]


def resolve_focus(names, requested, sims):
    if requested:
        if requested in names:
            return names.index(requested)
        raise SystemExit(f"--focus-agent must be one of: {', '.join(names)}")
    return names.index(f"Net-mcts{sims}")


def scheduled_pairs(names, schedule, focus_idx):
    if schedule == "full":
        return list(itertools.combinations(range(len(names)), 2))
    if schedule == "focused":
        return [(min(focus_idx, j), max(focus_idx, j))
                for j in range(len(names)) if j != focus_idx]
    raise ValueError(f"unknown schedule: {schedule}")


def games_for_pair(name_a, name_b, default_games, baseline_games):
    if baseline_games is None:
        return default_games
    if "Random" in (name_a, name_b) or "Greedy" in (name_a, name_b):
        return min(default_games, baseline_games)
    return default_games


def weighted_mean_score(values, counts):
    total = sum(counts)
    if total == 0:
        return None
    return sum(v * c for v, c in zip(values, counts)) / total


def write_matrix_csv(path, names, matrix):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + names)
        for i, name in enumerate(names):
            row = [name]
            for j in range(len(names)):
                row.append("" if matrix[i][j] is None else f"{matrix[i][j]:.3f}")
            w.writerow(row)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--champion", default="runs/best.pt")
    ap.add_argument("--games", type=int, default=30, help="games per seat per pairing")
    ap.add_argument("--sims", type=int, default=800, help="MCTS simulations for our agent")
    ap.add_argument("--depths", type=int, nargs="+", default=[4, 6, 8, 10],
                    help="alpha-beta opponent depths to include")
    ap.add_argument("--opening-plies", type=int, default=8,
                    help="random opening moves per game (variety/de-noising)")
    ap.add_argument("--classical", action="store_true",
                    help="add no-network MCTS (random + greedy rollouts) to ablate the net")
    ap.add_argument("--schedule", choices=["full", "focused"], default="full",
                    help="full round-robin or only matchups involving the focus agent")
    ap.add_argument("--focus-agent", default=None,
                    help="agent name used by --schedule focused; default is Net-mcts{sims}")
    ap.add_argument("--baseline-games", type=int, default=None,
                    help="games per seat for pairs containing Random or Greedy")
    ap.add_argument("--elo-prior", type=float, default=0.5,
                    help="small paired-game prior to keep sparse/lopsided Elo finite")
    ap.add_argument("--out", default="runs_tournament", help="directory for the CSV results")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    net = load_net(args.champion, device)
    pool = build_pool(net, device, args.sims, args.depths, classical=args.classical)
    names = [nm for nm, _ in pool]
    n = len(pool)
    focus_idx = resolve_focus(names, args.focus_agent, args.sims)
    pairs = scheduled_pairs(names, args.schedule, focus_idx)
    rng = random.Random(args.seed)

    # fp[i][j] = i's score rate when i is PLAYER 1 against j (as player 2).
    fp = [[None] * n for _ in range(n)]
    seat_games = [[0] * n for _ in range(n)]
    print(f"\n{args.schedule} tournament: {n} agents, {len(pairs)} pairings, "
          f"{args.games} games/seat/pair, {args.opening_plies} opening plies, "
          f"agent sims={args.sims}\n", flush=True)
    for i, j in pairs:
        g = games_for_pair(names[i], names[j], args.games, args.baseline_games)
        fp[i][j] = seat_score(pool[i][1], pool[j][1], g, args.opening_plies, rng)
        seat_games[i][j] = g
        fp[j][i] = seat_score(pool[j][1], pool[i][1], g, args.opening_plies, rng)
        seat_games[j][i] = g
        game_note = f", {g}/seat" if g != args.games else ""
        print(f"  {names[i]:>13} vs {names[j]:<13}  "
              f"P1 {fp[i][j]:5.1%}   (opp P1 {fp[j][i]:5.1%})"
              f"{game_note}", flush=True)

    # Derive the overall (seat-averaged) matrix and the Elo inputs.
    overall = [[None] * n for _ in range(n)]
    score_total = [[0.0] * n for _ in range(n)]
    played = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j or fp[i][j] is None or fp[j][i] is None:
                continue
            gij = seat_games[i][j]
            gji = seat_games[j][i]
            i_as_p1 = fp[i][j] * gij
            i_as_p2 = (1.0 - fp[j][i]) * gji       # i as player 2 when j led
            score_total[i][j] = i_as_p1 + i_as_p2
            played[i][j] = gij + gji
            overall[i][j] = score_total[i][j] / played[i][j]

    elos = fit_elo(score_total, played, prior=args.elo_prior)
    p1_rate = []
    p2_rate = []
    overall_rate = []
    for i in range(n):
        p1_values = [fp[i][j] for j in range(n) if fp[i][j] is not None]
        p1_counts = [seat_games[i][j] for j in range(n) if fp[i][j] is not None]
        p2_values = [1.0 - fp[j][i] for j in range(n) if fp[j][i] is not None]
        p2_counts = [seat_games[j][i] for j in range(n) if fp[j][i] is not None]
        p1 = weighted_mean_score(p1_values, p1_counts)
        p2 = weighted_mean_score(p2_values, p2_counts)
        both_values = p1_values + p2_values
        both_counts = p1_counts + p2_counts
        p1_rate.append(p1 if p1 is not None else 0.0)
        p2_rate.append(p2 if p2 is not None else 0.0)
        overall_rate.append(weighted_mean_score(both_values, both_counts) or 0.0)
    first_scores = [fp[i][j] for i in range(n) for j in range(n)
                    if fp[i][j] is not None]
    first_counts = [seat_games[i][j] for i in range(n) for j in range(n)
                    if fp[i][j] is not None]
    first_adv = weighted_mean_score(first_scores, first_counts) or 0.0

    def show(title, mat):
        print(f"\n{title}\n")
        print(" " * 14 + "".join(f"{nm[:7]:>8}" for nm in names))
        for i in range(n):
            row = f"{names[i]:>13} "
            for j in range(n):
                row += f"{'--':>8}" if mat[i][j] is None else f"{mat[i][j]:>8.0%}"
            print(row)

    show("overall score matrix (row's score % vs column):", overall)
    show("first-player matrix (row as PLAYER 1, score % vs column as player 2):", fp)

    order = sorted(range(n), key=lambda i: -elos[i])
    print("\nratings + seat split (sorted by Elo):\n")
    print(f"  {'agent':>13} {'Elo':>6} {'as P1':>7} {'as P2':>7} {'overall':>8}")
    for i in order:
        print(f"  {names[i]:>13} {elos[i]:6.0f} {p1_rate[i]:7.1%} "
              f"{p2_rate[i]:7.1%} {overall_rate[i]:8.1%}")
    print(f"\n  pool-wide first-player advantage: {first_adv:.1%} "
          f"(50% = no edge; openings are randomized {args.opening_plies} plies)")

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_matrix_csv(out / "overall_matrix.csv", names, overall)
    write_matrix_csv(out / "first_player_matrix.csv", names, fp)
    with open(out / "ratings.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["agent", "elo", "as_p1", "as_p2", "overall"])
        for i in order:
            w.writerow([names[i], f"{elos[i]:.1f}", f"{p1_rate[i]:.4f}",
                        f"{p2_rate[i]:.4f}", f"{overall_rate[i]:.4f}"])
    print(f"\nsaved: {out/'overall_matrix.csv'}, {out/'first_player_matrix.csv'}, "
          f"{out/'ratings.csv'}", flush=True)


if __name__ == "__main__":
    main()
