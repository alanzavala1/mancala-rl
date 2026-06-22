"""Self-play: the agent generates its own training data.

The current network (via MCTS) plays complete games against itself. Every
position becomes one training example:
  - the encoded board (the network's input),
  - the MCTS visit distribution over moves (a better policy than the network's
    raw hunch, because MCTS actually looked ahead -- the network learns to
    imitate it),
  - the game's final score margin, from the perspective of the player to move at
    that position (the value to predict).

Two deliberate choices in that value. It's the margin, not just win/loss -- a
scored target gives the value head more to learn (see features.margin_value).
And it's from the mover's perspective, the v1 alignment fix: each position is
labelled with the real outcome for whoever actually had to choose there.

To keep the data varied, the first few moves of each game are sampled in
proportion to how much MCTS visited them (exploration); after that the most-
visited move is played (exploitation). For even more variety during real
training, the MCTS passed in can add Dirichlet noise at the root.
"""

import random

from . import engine, features
from .mcts import visit_counts


def play_game(mcts, rng=None, temperature_moves=10, random_opening=0,
              max_plies=400, value_mode="clip", value_scale=features.MARGIN_SCALE):
    """Play one self-play game.

    Returns a list of (features, policy_target, value) examples, one per move.
    If random_opening > 0, the game starts with up to that many random,
    *unrecorded* opening moves. This diversifies the starting distribution so the
    agent is exposed to varied and disadvantaged positions -- the cure for
    brittleness and for never learning to defend.
    """
    rng = rng or random.Random()
    state = engine.reset(first_player=1)
    architecture = getattr(mcts.net, "architecture", "mlp")

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
        history.append((features.encode_for_model(state, architecture),
                        policy_target, state.current_player))

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

    s1, s2 = engine.stores(state)        # final score; the margin is the value signal
    final_margin = s1 - s2
    examples = []
    for feat, policy_target, player in history:
        mover_margin = final_margin if player == 1 else -final_margin
        examples.append((feat, policy_target,
                         features.value_target(mover_margin, value_mode, value_scale)))
    return examples


def generate(mcts, n_games, rng=None, temperature_moves=10, random_opening=0,
             value_mode="clip", value_scale=features.MARGIN_SCALE):
    """Play n_games of self-play; return all examples concatenated."""
    rng = rng or random.Random()
    examples = []
    for _ in range(n_games):
        examples.extend(play_game(mcts, rng, temperature_moves, random_opening,
                                  value_mode=value_mode, value_scale=value_scale))
    return examples


def _selfplay_worker(payload):
    """Run a chunk of self-play in a worker process. Top-level + simple args so
    it is picklable for ProcessPoolExecutor (spawn start method on Windows)."""
    import random as _random
    import numpy as _np
    import torch as _torch
    from .network import MancalaNet
    from .mcts import MCTS

    (state_dict, hidden, layers, residual, layer_norm, architecture, n_games,
     sims, c_puct, dirichlet, temp_moves, random_opening, value_mode,
     value_scale, seed) = payload
    _torch.set_num_threads(1)            # one thread per worker; parallelism is by process
    _random.seed(seed)
    _np.random.seed(seed % (2 ** 32 - 1))
    _torch.manual_seed(seed)

    net = MancalaNet(hidden=hidden, layers=layers,
                     residual=residual, layer_norm=layer_norm,
                     architecture=architecture)
    net.load_state_dict(state_dict)
    net.eval()
    mcts = MCTS(net, _torch.device("cpu"), n_simulations=sims,
                c_puct=c_puct, dirichlet_alpha=dirichlet, value_mode=value_mode,
                value_scale=value_scale)
    return generate(mcts, n_games, rng=_random.Random(seed),
                    temperature_moves=temp_moves, random_opening=random_opening,
                    value_mode=value_mode, value_scale=value_scale)


def generate_parallel(executor, state_dict, hidden, layers, residual, layer_norm,
                      architecture, n_games, n_workers, sims, dirichlet,
                      c_puct, temp_moves, random_opening, value_mode,
                      value_scale, base_seed):
    """Self-play spread across worker processes via a ProcessPoolExecutor.

    state_dict must hold CPU tensors. Returns all examples concatenated.
    """
    per = [n_games // n_workers + (1 if i < n_games % n_workers else 0)
           for i in range(n_workers)]
    payloads = [(state_dict, hidden, layers, residual, layer_norm, architecture,
                 per[i], sims, c_puct, dirichlet, temp_moves, random_opening,
                 value_mode, value_scale, base_seed * 100003 + i * 7919)
                for i in range(n_workers) if per[i] > 0]
    examples = []
    for chunk in executor.map(_selfplay_worker, payloads):
        examples.extend(chunk)
    return examples


# --- Gumbel AlphaZero self-play (isolated; uses gumbel.GumbelMCTS) -----------

def play_game_gumbel(gmcts, random_opening=0, max_plies=400,
                     value_mode="clip", value_scale=features.MARGIN_SCALE):
    """One self-play game using Gumbel search. The policy target is the search's
    improved (completed-Q) policy; the value target is the final margin."""
    rng = gmcts.rng
    state = engine.reset(first_player=1)
    architecture = getattr(gmcts.net, "architecture", "mlp")
    for _ in range(rng.randint(0, random_opening)):
        state, _, done, _ = engine.step(state, rng.choice(engine.legal_moves(state)))
        if done:
            state = engine.reset(first_player=1)
            break

    history = []
    for _ in range(max_plies):
        action, policy, _ = gmcts.search(state)
        history.append((features.encode_for_model(state, architecture),
                        policy, state.current_player))
        state, _, done, _ = engine.step(state, action)
        if done:
            break

    s1, s2 = engine.stores(state)
    final_margin = s1 - s2
    examples = []
    for feat, policy, player in history:
        mover_margin = final_margin if player == 1 else -final_margin
        examples.append((feat, policy,
                         features.value_target(mover_margin, value_mode, value_scale)))
    return examples


def generate_gumbel(gmcts, n_games, random_opening=0,
                    value_mode="clip", value_scale=features.MARGIN_SCALE):
    examples = []
    for _ in range(n_games):
        examples.extend(play_game_gumbel(gmcts, random_opening=random_opening,
                                         value_mode=value_mode, value_scale=value_scale))
    return examples


def _gumbel_worker(payload):
    import random as _random
    import numpy as _np
    import torch as _torch
    from .network import MancalaNet
    from .gumbel import GumbelMCTS

    (state_dict, hidden, layers, residual, layer_norm, architecture, n_games,
     sims, m, c_puct, random_opening, value_mode, value_scale, seed) = payload
    _torch.set_num_threads(1)
    _random.seed(seed)
    _np.random.seed(seed % (2 ** 32 - 1))
    _torch.manual_seed(seed)

    net = MancalaNet(hidden=hidden, layers=layers,
                     residual=residual, layer_norm=layer_norm,
                     architecture=architecture)
    net.load_state_dict(state_dict)
    net.eval()
    g = GumbelMCTS(net, _torch.device("cpu"), n_simulations=sims, m=m,
                   c_puct=c_puct, rng=_random.Random(seed), value_mode=value_mode,
                   value_scale=value_scale)
    return generate_gumbel(g, n_games, random_opening, value_mode, value_scale)


def generate_gumbel_parallel(executor, state_dict, hidden, layers, residual,
                             layer_norm, architecture, n_games, n_workers,
                             sims, m, c_puct, random_opening, value_mode,
                             value_scale, base_seed):
    per = [n_games // n_workers + (1 if i < n_games % n_workers else 0)
           for i in range(n_workers)]
    payloads = [(state_dict, hidden, layers, residual, layer_norm, architecture,
                 per[i], sims, m, c_puct, random_opening, value_mode, value_scale,
                 base_seed * 100003 + i * 7919)
                for i in range(n_workers) if per[i] > 0]
    examples = []
    for chunk in executor.map(_gumbel_worker, payloads):
        examples.extend(chunk)
    return examples
