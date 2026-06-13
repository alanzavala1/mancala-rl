"""Baseline (non-learning) agents.

Every agent implements act(state, rng) -> action index, where rng is a
random.Random the evaluation harness controls for reproducibility. Agents that
do not need randomness still accept rng and may ignore it.
"""

from . import engine


class RandomBot:
    """Plays a uniformly random legal move."""

    name = "Random"

    def act(self, state, rng):
        return rng.choice(engine.legal_moves(state))


class GreedyBot:
    """One-ply greedy on the store differential.

    Simulates each legal move and picks the one that maximizes
    (own store - opponent store) in the resulting position, breaking ties
    randomly. Because the simulation runs the real engine, this naturally
    values captures and extra turns without special-casing them. This is the
    reference heuristic the learning agent has to beat.
    """

    name = "Greedy"

    def act(self, state, rng):
        player = state.current_player
        best_score = None
        best = []
        for action in engine.legal_moves(state):
            next_state, _, _, _ = engine.step(state, action)
            s1, s2 = engine.stores(next_state)
            diff = (s1 - s2) if player == 1 else (s2 - s1)
            if best_score is None or diff > best_score:
                best_score = diff
                best = [action]
            elif diff == best_score:
                best.append(action)
        return rng.choice(best)
