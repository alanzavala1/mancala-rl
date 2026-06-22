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

Outputs (under --out): champion.pt (latest), best.pt (best vs solver),
checkpoint.pt (exact resume state), training_log.csv.
"""

import argparse
import csv
import math
import multiprocessing as mp
import os
import pathlib
import random
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from mancala_rl import selfplay
from mancala_rl.network import MancalaNet, load_net, save_net
from mancala_rl.mcts import MCTS, MCTSBot
from mancala_rl.training import ReplaySampler, describe_examples, train_step
from mancala_rl.evaluate import evaluate, play_game
from mancala_rl.bots import RandomBot, GreedyBot
from mancala_rl.csolver import CSolverBot


LOG_HEADER = [
    "iteration", "buffer", "loss", "vs_random", "vs_greedy",
    "solver_first", "solver_second", "best", "seconds",
    "new_opening", "new_midgame", "new_endgame",
    "buffer_opening", "buffer_midgame", "buffer_endgame",
    "buffer_behind", "buffer_close", "buffer_ahead",
    "target_entropy", "abs_value", "phase_balanced_frac",
    "diverse_frac", "priority_frac", "priority_phase",
    "random_opening_used", "value_mode", "value_scale",
    "residual", "layer_norm", "architecture", "c_puct",
    "gate_score",
]


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


def match_score(agent_a, agent_b, n, seed, opening_plies=8):
    """Seat-split score for agent_a, with randomized opening plies."""
    rng = random.Random(seed)
    score = 0.0
    for i in range(n):
        if i % 2 == 0:
            winner = play_game(agent_a, agent_b, rng, opening_plies=opening_plies)
            agent_a_player = 1
        else:
            winner = play_game(agent_b, agent_a, rng, opening_plies=opening_plies)
            agent_a_player = 2
        if winner == 0:
            score += 0.5
        elif winner == agent_a_player:
            score += 1.0
    return score / n


def scheduled_random_opening(args, iteration):
    """Opening randomization cap for this iteration.

    Default behavior is the original fixed --random-opening. If
    --random-opening-final is set, linearly ramp from --random-opening to that
    final value over the run.
    """
    if args.random_opening_final is None or args.iterations <= 1:
        return args.random_opening
    frac = (iteration - 1) / (args.iterations - 1)
    start = args.random_opening
    end = args.random_opening_final
    return int(round(start + frac * (end - start)))


def _torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # older torch
        return torch.load(path, map_location=device)


def _write_log_header(csv_path, args):
    if args.resume:
        if not csv_path.exists():
            with open(csv_path, "w", newline="") as f:
                csv.writer(f).writerow(LOG_HEADER)
        return
    if csv_path.exists() and not args.overwrite and not args.append_log:
        raise SystemExit(
            f"{csv_path} already exists. Use --resume for exact continuation, "
            "--append-log to add rows, --overwrite to replace it, or a new --out."
        )
    mode = "a" if args.append_log and csv_path.exists() else "w"
    with open(csv_path, mode, newline="") as f:
        if mode == "w":
            csv.writer(f).writerow(LOG_HEADER)


def _infer_best_score(csv_path):
    if not csv_path.exists():
        return -1
    best = -1
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("best") != "1":
                continue
            try:
                score = int(row["solver_first"]) + int(row["solver_second"])
            except (TypeError, ValueError):
                continue
            best = max(best, score)
    return best


def _save_checkpoint(path, net, optimizer, scheduler, buffer, best_score,
                     iteration, args, train_rng):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "iteration": iteration,
        "model": {
            "state_dict": net.state_dict(),
            "hidden": net.hidden,
            "layers": net.layers,
            "residual": net.residual,
            "layer_norm": net.layer_norm,
            "architecture": getattr(net, "architecture", "mlp"),
            "value_mode": getattr(net, "value_mode", "clip"),
            "value_scale": getattr(net, "value_scale", 16),
        },
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "buffer": list(buffer),
        "buffer_maxlen": buffer.maxlen,
        "best_score": best_score,
        "args": vars(args),
        "random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
        "train_rng_state": train_rng.getstate(),
    }, tmp)
    os.replace(tmp, path)


def _load_checkpoint(path, device):
    blob = _torch_load(path, device)
    model = blob["model"]
    net = MancalaNet(
        hidden=model["hidden"],
        layers=model["layers"],
        residual=model.get("residual", False),
        layer_norm=model.get("layer_norm", False),
        architecture=model.get("architecture", "mlp"),
    ).to(device)
    net.load_state_dict(model["state_dict"])
    net.value_mode = model.get("value_mode", "clip")
    net.value_scale = model.get("value_scale", 16)
    return blob, net


def _load_init_net(path, device):
    net = load_net(path, device)
    net.train()
    return net


def _set_cosine_scheduler_progress(optimizer, scheduler, args, completed_iterations):
    if completed_iterations <= 0:
        return
    completed = min(completed_iterations, args.iterations)
    eta_min = args.lr * 0.1
    lr = eta_min + (args.lr - eta_min) * (
        1 + math.cos(math.pi * completed / args.iterations)) / 2
    for group in optimizer.param_groups:
        group["lr"] = lr
    scheduler.last_epoch = completed
    scheduler._last_lr = [lr for _ in optimizer.param_groups]


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--iterations", type=int, default=300)
    p.add_argument("--games", type=int, default=32)
    p.add_argument("--sims", type=int, default=256)
    p.add_argument("--hidden", type=int, default=128, help="network width")
    p.add_argument("--layers", type=int, default=2, help="number of hidden layers")
    p.add_argument("--residual", action="store_true",
                   help="use residual hidden blocks after the input layer")
    p.add_argument("--layer-norm", action="store_true",
                   help="add layer normalization in the network body")
    p.add_argument("--architecture", choices=["mlp", "action-aware"],
                   default="mlp",
                   help="network policy architecture")
    p.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 8),
                   help="self-play worker processes (1 = serial)")
    p.add_argument("--train-steps", type=int, default=300)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--buffer", type=int, default=60000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--value-weight", type=float, default=1.0,
                   help="weight on the value loss relative to the policy loss")
    p.add_argument("--value-mode", choices=["clip", "tanh"], default="clip",
                   help="mapping from final score margin to value target")
    p.add_argument("--value-scale", type=float, default=16.0,
                   help="margin scale used by the value target")
    p.add_argument("--phase-balanced-frac", type=float, default=0.0,
                   help="fraction of each training batch sampled evenly across opening/midgame/endgame")
    p.add_argument("--diverse-frac", type=float, default=0.0,
                   help="fraction sampled evenly across phase x value-target buckets")
    p.add_argument("--priority-frac", type=float, default=0.0,
                   help="fraction sampled from high-entropy MCTS targets")
    p.add_argument("--priority-phase", default="midgame",
                   choices=["all", "opening", "midgame", "endgame"],
                   help="phase used for high-entropy priority sampling")
    p.add_argument("--random-opening", type=int, default=8)
    p.add_argument("--random-opening-final", type=int, default=None,
                   help="linearly ramp random-opening to this value across training")
    p.add_argument("--temp-moves", type=int, default=12)
    p.add_argument("--dirichlet", type=float, default=0.3)
    p.add_argument("--c-puct", type=float, default=1.5,
                   help="PUCT exploration constant used by search")
    p.add_argument("--gumbel", action="store_true",
                   help="use Gumbel AlphaZero self-play (sample-efficient; pair with a low --sims)")
    p.add_argument("--gumbel-m", type=int, default=16, help="Gumbel candidate actions at the root")
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--eval-games", type=int, default=30)
    p.add_argument("--solver-depth", type=int, default=8)
    p.add_argument("--solver-games", type=int, default=40)
    p.add_argument("--gate-against-best", action="store_true",
                   help="only save best.pt if a solver-improved candidate also beats the current best")
    p.add_argument("--gate-games", type=int, default=24,
                   help="head-to-head games for --gate-against-best")
    p.add_argument("--gate-threshold", type=float, default=0.53,
                   help="minimum score required against the current best")
    p.add_argument("--gate-sims", type=int, default=None,
                   help="MCTS simulations per move for checkpoint gating")
    p.add_argument("--gate-opening-plies", type=int, default=8,
                   help="random opening plies used during checkpoint gating")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="runs")
    p.add_argument("--resume", default=None,
                   help="exactly resume from a checkpoint.pt created by this script")
    p.add_argument("--init-from", default=None,
                   help="warm-start model weights from champion.pt/best.pt without replay/optimizer state")
    p.add_argument("--start-iteration", type=int, default=1,
                   help="first iteration number for warm-start runs")
    p.add_argument("--checkpoint-every", type=int, default=1,
                   help="save exact resume checkpoint every N completed iterations; 0 disables")
    p.add_argument("--checkpoint-name", default="checkpoint.pt")
    p.add_argument("--append-log", action="store_true",
                   help="append to an existing training_log.csv")
    p.add_argument("--overwrite", action="store_true",
                   help="overwrite an existing training_log.csv")
    args = p.parse_args()
    if args.resume and args.init_from:
        raise SystemExit("--resume and --init-from are mutually exclusive")
    if args.resume and args.start_iteration != 1:
        raise SystemExit("--start-iteration is only for --init-from or fresh runs")
    if args.resume and args.out == "runs":
        args.out = str(pathlib.Path(args.resume).parent)
    architecture = args.architecture.replace("-", "_")
    replay_frac = args.phase_balanced_frac + args.diverse_frac + args.priority_frac
    if replay_frac > 1.0:
        raise SystemExit(
            "--phase-balanced-frac + --diverse-frac + --priority-frac must be <= 1.0"
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_rng = random.Random(args.seed + 12345)
    device = torch.device(args.device)

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "training_log.csv"
    checkpoint_path = out / args.checkpoint_name
    _write_log_header(csv_path, args)

    checkpoint = None
    if args.resume:
        checkpoint, net = _load_checkpoint(pathlib.Path(args.resume), device)
        architecture = getattr(net, "architecture", "mlp")
        args.value_mode = getattr(net, "value_mode", args.value_mode)
        args.value_scale = getattr(net, "value_scale", args.value_scale)
    elif args.init_from:
        net = _load_init_net(args.init_from, device)
        architecture = getattr(net, "architecture", "mlp")
        args.value_mode = getattr(net, "value_mode", args.value_mode)
        args.value_scale = getattr(net, "value_scale", args.value_scale)
    else:
        net = MancalaNet(hidden=args.hidden, layers=args.layers,
                         residual=args.residual, layer_norm=args.layer_norm,
                         architecture=architecture).to(device)
        net.value_mode = args.value_mode
        net.value_scale = args.value_scale

    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    # Cosine learning-rate decay across the run: a high LR early to learn fast,
    # easing toward lr/10 so a long run settles into a better minimum instead of
    # bouncing around it (a fixed LR plateaued the loss around 1.42).
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.iterations, eta_min=args.lr * 0.1)
    buffer = deque(maxlen=args.buffer)
    best_score = -1
    start_iteration = args.start_iteration

    if checkpoint is not None:
        opt.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        buffer = deque(checkpoint["buffer"],
                       maxlen=checkpoint.get("buffer_maxlen", args.buffer))
        best_score = checkpoint.get("best_score", -1)
        start_iteration = checkpoint["iteration"] + 1
        random.setstate(checkpoint["random_state"])
        np.random.set_state(checkpoint["numpy_random_state"])
        torch.set_rng_state(checkpoint["torch_random_state"])
        train_rng.setstate(checkpoint["train_rng_state"])
    else:
        if args.append_log:
            best_score = _infer_best_score(csv_path)
        if args.init_from and args.start_iteration > 1:
            _set_cosine_scheduler_progress(
                opt, scheduler, args, args.start_iteration - 1)

    if start_iteration > args.iterations:
        print(f"checkpoint is already at iteration {start_iteration - 1}; "
              f"target --iterations is {args.iterations}", flush=True)
        return

    use_pool = args.workers and args.workers > 1
    executor = ProcessPoolExecutor(
        max_workers=args.workers, mp_context=mp.get_context("spawn")) if use_pool else None

    mode = "resume" if args.resume else ("warm-start" if args.init_from else "fresh")
    print(f"device={device}  mode={mode}  start={start_iteration}  "
          f"iterations={args.iterations}  games/iter={args.games}  "
          f"sims={args.sims}  workers={args.workers if use_pool else 1}\n", flush=True)

    try:
        for it in range(start_iteration, args.iterations + 1):
            t0 = time.perf_counter()
            random_opening_used = scheduled_random_opening(args, it)

            net.eval()
            if args.gumbel and use_pool:
                state_dict = {k: v.detach().cpu() for k, v in net.state_dict().items()}
                examples = selfplay.generate_gumbel_parallel(
                    executor, state_dict, net.hidden, net.layers, net.residual,
                    net.layer_norm, net.architecture, args.games, args.workers,
                    args.sims, args.gumbel_m, args.c_puct,
                    random_opening_used, args.value_mode, args.value_scale,
                    args.seed + it)
            elif args.gumbel:
                from mancala_rl.gumbel import GumbelMCTS
                g = GumbelMCTS(net, device, n_simulations=args.sims, m=args.gumbel_m,
                               c_puct=args.c_puct, rng=random.Random(args.seed + it),
                               value_mode=args.value_mode,
                               value_scale=args.value_scale)
                examples = selfplay.generate_gumbel(
                    g, args.games, random_opening_used, args.value_mode,
                    args.value_scale)
            elif use_pool:
                state_dict = {k: v.detach().cpu() for k, v in net.state_dict().items()}
                examples = selfplay.generate_parallel(
                    executor, state_dict, net.hidden, net.layers, net.residual,
                    net.layer_norm, net.architecture, args.games, args.workers,
                    args.sims, args.dirichlet, args.c_puct, args.temp_moves,
                    random_opening_used, args.value_mode, args.value_scale,
                    args.seed + it)
            else:
                sp_mcts = MCTS(net, device, n_simulations=args.sims,
                               c_puct=args.c_puct, dirichlet_alpha=args.dirichlet,
                               value_mode=args.value_mode,
                               value_scale=args.value_scale)
                examples = selfplay.generate(
                    sp_mcts, n_games=args.games, rng=random.Random(args.seed + it),
                    temperature_moves=args.temp_moves, random_opening=random_opening_used,
                    value_mode=args.value_mode, value_scale=args.value_scale)
            buffer.extend(examples)
            new_stats = describe_examples(examples)
            buffer_stats = describe_examples(buffer)

            net.train()
            buf_list = list(buffer)
            replay_sampler = ReplaySampler(
                buf_list, train_rng, args.phase_balanced_frac, args.diverse_frac,
                args.priority_frac, args.priority_phase)
            losses = []
            for _ in range(args.train_steps):
                batch = replay_sampler.sample(args.batch)
                losses.append(train_step(net, opt, batch, device, args.value_weight)[0])
            avg_loss = sum(losses) / len(losses)
            scheduler.step()
            net.eval()
            save_net(net, out / "champion.pt")

            do_eval = (it % args.eval_every == 0) or (it == args.iterations)
            if do_eval:
                agent = MCTSBot(net, device, args.sims, c_puct=args.c_puct,
                                value_mode=args.value_mode,
                                value_scale=args.value_scale)
                vr = evaluate(agent, RandomBot(), n_games=args.eval_games,
                              seed=args.seed + 2000 + it).score
                vg = evaluate(agent, GreedyBot(), n_games=args.eval_games,
                              seed=args.seed + 3000 + it).score
                sf, ss = solver_seatsplit(agent, args.solver_depth, args.solver_games,
                                          args.seed + 4000 + it)
                score = sf + ss
                solver_improved = score > best_score
                new_best = solver_improved
                gate_score = ""
                best_path = out / "best.pt"
                if (solver_improved and args.gate_against_best
                        and best_path.exists()):
                    gate_sims = args.gate_sims or args.sims
                    best_net = load_net(best_path, device)
                    challenger = MCTSBot(net, device, gate_sims,
                                         c_puct=args.c_puct)
                    incumbent = MCTSBot(best_net, device, gate_sims,
                                        c_puct=args.c_puct)
                    gate_score = match_score(
                        challenger, incumbent, args.gate_games,
                        seed=args.seed + 5000 + it,
                        opening_plies=args.gate_opening_plies)
                    new_best = gate_score >= args.gate_threshold
                if new_best:
                    best_score = score
                    save_net(net, best_path)
                dt = time.perf_counter() - t0
                n = args.solver_games
                gate_text = f" | gate {gate_score:4.0%}" if gate_score != "" else ""
                print(f"iter {it:3d} | loss {avg_loss:5.3f} | vs R {vr:4.0%} G {vg:4.0%} | "
                      f"solver(d{args.solver_depth}) first {sf:2d}/{n} second {ss:2d}/{n}"
                      f"{gate_text}{'  << NEW BEST' if new_best else ''} | {dt:4.0f}s",
                      flush=True)
                with open(csv_path, "a", newline="") as f:
                    csv.writer(f).writerow(
                        [it, len(buffer), f"{avg_loss:.4f}", f"{vr:.3f}", f"{vg:.3f}",
                         sf, ss, int(new_best), f"{dt:.1f}",
                         f"{new_stats['opening']:.3f}", f"{new_stats['midgame']:.3f}",
                         f"{new_stats['endgame']:.3f}", f"{buffer_stats['opening']:.3f}",
                         f"{buffer_stats['midgame']:.3f}", f"{buffer_stats['endgame']:.3f}",
                         f"{buffer_stats['behind']:.3f}", f"{buffer_stats['close']:.3f}",
                         f"{buffer_stats['ahead']:.3f}",
                         f"{new_stats['entropy']:.3f}", f"{new_stats['abs_value']:.3f}",
                         f"{args.phase_balanced_frac:.3f}", f"{args.diverse_frac:.3f}",
                         f"{args.priority_frac:.3f}", args.priority_phase,
                         random_opening_used, args.value_mode, f"{args.value_scale:.3f}",
                         int(net.residual), int(net.layer_norm), net.architecture,
                         f"{args.c_puct:.3f}",
                         "" if gate_score == "" else f"{gate_score:.3f}"])
            else:
                dt = time.perf_counter() - t0
                print(f"iter {it:3d} | loss {avg_loss:5.3f} | {dt:4.0f}s", flush=True)
                with open(csv_path, "a", newline="") as f:
                    csv.writer(f).writerow(
                        [it, len(buffer), f"{avg_loss:.4f}", "", "", "", "", "", f"{dt:.1f}",
                         f"{new_stats['opening']:.3f}", f"{new_stats['midgame']:.3f}",
                         f"{new_stats['endgame']:.3f}", f"{buffer_stats['opening']:.3f}",
                         f"{buffer_stats['midgame']:.3f}", f"{buffer_stats['endgame']:.3f}",
                         f"{buffer_stats['behind']:.3f}", f"{buffer_stats['close']:.3f}",
                         f"{buffer_stats['ahead']:.3f}",
                         f"{new_stats['entropy']:.3f}", f"{new_stats['abs_value']:.3f}",
                         f"{args.phase_balanced_frac:.3f}", f"{args.diverse_frac:.3f}",
                         f"{args.priority_frac:.3f}", args.priority_phase,
                         random_opening_used, args.value_mode, f"{args.value_scale:.3f}",
                         int(net.residual), int(net.layer_norm), net.architecture,
                         f"{args.c_puct:.3f}", ""])
            if args.checkpoint_every and (
                    it % args.checkpoint_every == 0 or it == args.iterations):
                _save_checkpoint(
                    checkpoint_path, net, opt, scheduler, buffer, best_score,
                    it, args, train_rng)
    finally:
        if executor is not None:
            executor.shutdown()

    print(f"\ndone. latest -> {out/'champion.pt'} ; best vs solver -> {out/'best.pt'} "
          f"(score {best_score}/{2*args.solver_games}) ; log -> {csv_path}", flush=True)


if __name__ == "__main__":
    main()
