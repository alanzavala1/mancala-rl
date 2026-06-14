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


def play_game(mcts, rng=None, temperature_moves=10, max_plies=400):
    """Play one self-play game.

    Returns a list of (features, policy_target, value) examples, one per move.
    """
    rng = rng or random.Random()
    state = engine.reset(first_player=1)
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


def generate(mcts, n_games, rng=None, temperature_moves=10):
    """Play n_games of self-play; return all examples concatenated."""
    rng = rng or random.Random()
    examples = []
    for _ in range(n_games):
        examples.extend(play_game(mcts, rng, temperature_moves))
    return examples
