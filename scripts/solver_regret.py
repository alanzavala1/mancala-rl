"""Measure agent move quality against the C solver.

This is a diagnostic, not a training script. It samples non-terminal positions,
asks the C solver to score every legal move, and reports how often each agent
chooses a solver-best move plus how much score margin it gives up when it does
not.

Examples:
    .venv\\Scripts\\python scripts/solver_regret.py --champion runs_v6/best.pt
    .venv\\Scripts\\python scripts/solver_regret.py --champion runs_v6/best.pt --mcts-sims 800
"""

import argparse
import csv
import pathlib
import random
from collections import defaultdict

import torch

from mancala_rl import csolver, engine
from mancala_rl.bots import GreedyBot, RandomBot
from mancala_rl.mcts import MCTSBot
from mancala_rl.network import PolicyBot, load_net


PITS = tuple(range(0, 6)) + tuple(range(7, 13))


def phase_of(state):
    """Bucket by how many seeds are still in pits, not stores."""
    remaining = sum(state.board[i] for i in PITS)
    if remaining >= 32:
        return "opening"
    if remaining >= 16:
        return "midgame"
    return "endgame"


def final_margin(state):
    s1, s2 = engine.stores(state)
    return s1 - s2


def solver_move_values(state, depth, exact_max_seeds):
    """Return legal action values in the player-1 margin frame.

    For small endgames, solve each child exactly. Otherwise use the C solver's
    depth-limited action values, which are much cheaper and still useful for
    ranking tactical choices.
    """
    legal = engine.legal_moves(state)
    remaining = sum(state.board[i] for i in PITS)
    if exact_max_seeds >= 0 and remaining <= exact_max_seeds:
        values = {}
        for action in legal:
            child, _, done, _ = engine.step(state, action)
            values[action] = final_margin(child) if done else csolver.solve_exact(child)
        return values, "exact"

    raw = csolver.move_values(state, depth)
    return {action: raw[action] for action in legal}, f"depth{depth}"


def mover_value(value, player):
    return value if player == 1 else -value


def best_actions(values, player):
    mover_values = {a: mover_value(v, player) for a, v in values.items()}
    best = max(mover_values.values())
    return [a for a, v in mover_values.items() if v == best], best


def choose_source_action(source, state, rng, policy_bot, greedy_bot, random_bot):
    if source == "random":
        return random_bot.act(state, rng)
    if source == "greedy":
        return greedy_bot.act(state, rng)
    if source == "policy":
        return policy_bot.act(state, rng)
    if source == "mixed":
        roll = rng.random()
        if roll < 0.20:
            return random_bot.act(state, rng)
        if roll < 0.55:
            return greedy_bot.act(state, rng)
        return policy_bot.act(state, rng)
    raise ValueError(f"unknown sample source: {source}")


def collect_candidate_positions(n_target, rng, source, policy_bot, max_plies):
    """Collect a broad set of positions, then sample evenly across phases."""
    greedy_bot = GreedyBot()
    random_bot = RandomBot()
    candidates = []
    min_candidates = max(n_target * 4, n_target + 30)
    while len(candidates) < min_candidates:
        state = engine.reset()
        for ply in range(max_plies):
            candidates.append((state, ply, phase_of(state)))
            action = choose_source_action(
                source, state, rng, policy_bot, greedy_bot, random_bot)
            state, _, done, _ = engine.step(state, action)
            if done:
                break

    by_phase = defaultdict(list)
    for item in candidates:
        by_phase[item[2]].append(item)

    selected = []
    phases = ["opening", "midgame", "endgame"]
    per_phase = n_target // len(phases)
    for phase in phases:
        pool = by_phase[phase]
        take = min(per_phase, len(pool))
        selected.extend(rng.sample(pool, take))

    remaining = n_target - len(selected)
    if remaining > 0:
        chosen_ids = {id(item) for item in selected}
        pool = [item for item in candidates if id(item) not in chosen_ids]
        selected.extend(rng.sample(pool, min(remaining, len(pool))))

    rng.shuffle(selected)
    return selected[:n_target]


def make_agents(names, net, device, mcts_sims):
    agents = {}
    for name in names:
        if name == "raw":
            agents[name] = PolicyBot(net, device)
        elif name == "mcts":
            agents[name] = MCTSBot(net, device, n_simulations=mcts_sims)
        elif name == "greedy":
            agents[name] = GreedyBot()
        elif name == "random":
            agents[name] = RandomBot()
        else:
            raise ValueError(f"unknown agent: {name}")
    return agents


def empty_stats():
    return {"n": 0, "optimal": 0, "regret_sum": 0.0, "max_regret": 0.0}


def update_stats(bucket, optimal, regret):
    bucket["n"] += 1
    bucket["optimal"] += int(optimal)
    bucket["regret_sum"] += regret
    bucket["max_regret"] = max(bucket["max_regret"], regret)


def summarize(stats):
    rows = []
    for agent in sorted(stats):
        for phase in ["all", "opening", "midgame", "endgame"]:
            bucket = stats[agent].get(phase)
            if not bucket or bucket["n"] == 0:
                continue
            n = bucket["n"]
            rows.append({
                "agent": agent,
                "phase": phase,
                "n": n,
                "optimal_rate": bucket["optimal"] / n,
                "avg_regret": bucket["regret_sum"] / n,
                "max_regret": bucket["max_regret"],
            })
    return rows


def write_summary(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["agent", "phase", "n", "optimal_rate",
                        "avg_regret", "max_regret"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "agent": row["agent"],
                "phase": row["phase"],
                "n": row["n"],
                "optimal_rate": f"{row['optimal_rate']:.4f}",
                "avg_regret": f"{row['avg_regret']:.4f}",
                "max_regret": f"{row['max_regret']:.4f}",
            })


def print_summary(rows):
    print("\nsolver-regret summary\n")
    print(f"{'agent':>8} {'phase':>8} {'n':>5} {'optimal':>9} {'avg regret':>11} {'max':>7}")
    for row in rows:
        print(
            f"{row['agent']:>8} {row['phase']:>8} {row['n']:5d} "
            f"{row['optimal_rate']:8.1%} {row['avg_regret']:11.3f} "
            f"{row['max_regret']:7.1f}"
        )


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--champion", default="runs/best.pt")
    p.add_argument("--samples", type=int, default=180)
    p.add_argument("--sample-source", choices=["mixed", "policy", "greedy", "random"],
                   default="mixed")
    p.add_argument("--max-plies", type=int, default=220)
    p.add_argument("--solver-depth", type=int, default=12)
    p.add_argument("--exact-max-seeds", type=int, default=12,
                   help="solve child moves exactly when pit seeds are at or below this; -1 disables")
    p.add_argument("--agents", nargs="+", default=["raw", "mcts", "greedy"],
                   choices=["raw", "mcts", "greedy", "random"])
    p.add_argument("--mcts-sims", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="results/solver_regret_summary.csv")
    args = p.parse_args()

    rng = random.Random(args.seed)
    device = torch.device(args.device)
    net = load_net(args.champion, device)
    policy_bot = PolicyBot(net, device)
    agents = make_agents(args.agents, net, device, args.mcts_sims)

    positions = collect_candidate_positions(
        args.samples, rng, args.sample_source, policy_bot, args.max_plies)

    stats = defaultdict(lambda: defaultdict(empty_stats))
    source_counts = defaultdict(int)
    for i, (state, _, phase) in enumerate(positions, start=1):
        values, source = solver_move_values(
            state, args.solver_depth, args.exact_max_seeds)
        source_counts[source] += 1
        best, best_value = best_actions(values, state.current_player)
        for name, agent in agents.items():
            action = agent.act(state, rng)
            chosen = mover_value(values[action], state.current_player)
            regret = best_value - chosen
            optimal = action in best
            update_stats(stats[name]["all"], optimal, regret)
            update_stats(stats[name][phase], optimal, regret)
        if i % 25 == 0:
            print(f"scored {i}/{len(positions)} positions", flush=True)

    rows = summarize(stats)
    print_summary(rows)
    print("\nsolver labels:", ", ".join(f"{k}={v}" for k, v in sorted(source_counts.items())))

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_summary(out, rows)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
