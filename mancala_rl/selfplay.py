"""Self-play: the agent generates its own training data.

The current network (via MCTS) plays complete games against itself. Every
position becomes one training example:
  - the encoded board (the network's input),
  - the MCTS visit distribution over moves (a better policy than the network's
    raw hunch, because MCTS actually looked ahead -- the network learns to
    imitate it),
  - the eventual game result, from the perspective of the player who was to move
    at that position (the value to predict).

That last point is the v1 alignment fix: each position's value is the real
outcome for the player who actually had to choose there.

To keep the data varied, the first few moves of each game are sampled in
proportion to how much MCTS visited them (exploration); after that the most-
visited move is played (exploitation). For even more variety during real
training, the MCTS passed in can add Dirichlet noise at the root.
"""

import random

from . import engine, features
from .mcts import visit_counts


def play_game(mcts, rng=None, temperature_moves=10, random_opening=0, max_plies=400):
    """Play one self-play game.

    Returns a list of (features, policy_target, value) examples, one per move.
    If random_opening > 0, the game starts with up to that many random,
    *unrecorded* opening moves. This diversifies the starting distribution so the
    agent is exposed to varied and disadvantaged positions -- the cure for
    brittleness and for never learning to defend.
    """
    rng = rng or random.Random()
    state = engine.reset(first_player=1)

    for _ in range(rng.randint(0, random_opening)):
        state, _, done, _ = engine.step(state, rng.choice(engine.legal_moves(state)))
        if done:                                  # rare: just start over cleanly
            state = engine.reset(first_player=1)
            break

    history = []   # (features_vec, policy_target, player_to_move)

    winner = 0
    for ply in range(max_plies):
        root = mcts.search(state)
        counts = visit_counts(root)
        total = sum(counts.values())

        policy_target = [0.0] * features.NUM_ACTIONS
        for action, n in counts.items():
            policy_target[action] = n / total
        history.append((features.encode(state), policy_target, state.current_player))

        actions = list(counts)
        weights = [counts[a] for a in actions]
        if ply < temperature_moves:
            action = rng.choices(actions, weights=weights, k=1)[0]      # explore
        else:
            best = max(weights)
            action = rng.choice([a for a in actions if counts[a] == best])  # exploit

        state, _, done, info = engine.step(state, action)
        if done:
            winner = info["winner"]
            break
    else:
        s1, s2 = engine.stores(state)            # safety net if the cap is hit
        winner = 1 if s1 > s2 else (2 if s2 > s1 else 0)

    examples = []
    for feat, policy_target, player in history:
        if winner == 0:
            value = 0.0
        else:
            value = 1.0 if winner == player else -1.0
        examples.append((feat, policy_target, value))
    return examples


def generate(mcts, n_games, rng=None, temperature_moves=10, random_opening=0):
    """Play n_games of self-play; return all examples concatenated."""
    rng = rng or random.Random()
    examples = []
    for _ in range(n_games):
        examples.extend(play_game(mcts, rng, temperature_moves, random_opening))
    return examples


def _selfplay_worker(payload):
    """Run a chunk of self-play in a worker process. Top-level + simple args so
    it is picklable for ProcessPoolExecutor (spawn start method on Windows)."""
    import random as _random
    import numpy as _np
    import torch as _torch
    from .network import MancalaNet
    from .mcts import MCTS

    (state_dict, hidden, layers, n_games, sims, dirichlet,
     temp_moves, random_opening, seed) = payload
    _torch.set_num_threads(1)            # one thread per worker; parallelism is by process
    _random.seed(seed)
    _np.random.seed(seed % (2 ** 32 - 1))
    _torch.manual_seed(seed)

    net = MancalaNet(hidden=hidden, layers=layers)
    net.load_state_dict(state_dict)
    net.eval()
    mcts = MCTS(net, _torch.device("cpu"), n_simulations=sims, dirichlet_alpha=dirichlet)
    return generate(mcts, n_games, rng=_random.Random(seed),
                    temperature_moves=temp_moves, random_opening=random_opening)


def generate_parallel(executor, state_dict, hidden, layers, n_games, n_workers, sims,
                      dirichlet, temp_moves, random_opening, base_seed):
    """Self-play spread across worker processes via a ProcessPoolExecutor.

    state_dict must hold CPU tensors. Returns all examples concatenated.
    """
    per = [n_games // n_workers + (1 if i < n_games % n_workers else 0)
           for i in range(n_workers)]
    payloads = [(state_dict, hidden, layers, per[i], sims, dirichlet, temp_moves,
                 random_opening, base_seed * 100003 + i * 7919)
                for i in range(n_workers) if per[i] > 0]
    examples = []
    for chunk in executor.map(_selfplay_worker, payloads):
        examples.extend(chunk)
    return examples
