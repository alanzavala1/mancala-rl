"""Classical (no-network) MCTS, for ablating the learned agent.

The learned agent is net + MCTS. These are MCTS with NO network: uniform choice
of unexpanded moves, and a plain rollout (random, or a 1-ply greedy) to estimate
each leaf -- at the same simulation budget. If the learned agent beats these
clearly at equal sims, the *learning* is doing real work; if it barely beats
them, the search is carrying it. (Alpha-beta minimax is already in the tournament
as the AB-dN agents.)

Values are kept in an absolute (player-1) frame and each node maximizes from its
own mover's view, so the extra-turn rule is handled correctly -- same convention
as mcts.py.
"""

import math

from . import engine


class _Node:
    __slots__ = ("state", "player", "parent", "pa", "children",
                 "untried", "N", "W", "terminal", "winner")

    def __init__(self, state, parent=None, pa=None):
        self.state = state
        self.player = state.current_player
        self.parent = parent
        self.pa = pa                      # action taken from parent to reach here
        self.children = []
        self.terminal = False
        self.winner = None
        self.untried = engine.legal_moves(state)
        self.N = 0
        self.W = 0.0                       # sum of +1 (P1 win) / -1 (P2 win) / 0


def _winner_to_abs(winner):
    return 1.0 if winner == 1 else (-1.0 if winner == 2 else 0.0)


def _random_rollout(state, rng):
    for _ in range(400):
        state, _, done, info = engine.step(state, rng.choice(engine.legal_moves(state)))
        if done:
            return info["winner"]
    s1, s2 = engine.stores(state)
    return 1 if s1 > s2 else (2 if s2 > s1 else 0)


def _greedy_rollout(state, rng):
    for _ in range(400):
        p = state.current_player
        best, bestv = [], None
        for a in engine.legal_moves(state):
            ns, _, _, _ = engine.step(state, a)
            s1, s2 = engine.stores(ns)
            v = (s1 - s2) if p == 1 else (s2 - s1)
            if bestv is None or v > bestv:
                bestv, best = v, [a]
            elif v == bestv:
                best.append(a)
        state, _, done, info = engine.step(state, rng.choice(best))
        if done:
            return info["winner"]
    s1, s2 = engine.stores(state)
    return 1 if s1 > s2 else (2 if s2 > s1 else 0)


class ClassicalMCTSBot:
    """UCT MCTS with no network. rollout = 'random' or 'greedy'."""

    def __init__(self, n_simulations=800, c=1.4, rollout="random"):
        self.n = n_simulations
        self.c = c
        self._rollout = _greedy_rollout if rollout == "greedy" else _random_rollout
        self.name = f"MCTS-{rollout}{n_simulations}"

    def _best_child(self, node):
        sign = 1.0 if node.player == 1 else -1.0     # node maximizes its own mover's value
        log_n = math.log(node.N + 1)
        best, best_score = None, -1e30
        for ch in node.children:
            q = sign * (ch.W / ch.N)
            u = self.c * math.sqrt(log_n / ch.N)
            if q + u > best_score:
                best_score, best = q + u, ch
        return best

    def act(self, state, rng):
        root = _Node(state)
        for _ in range(self.n):
            node = root
            while not node.untried and node.children and not node.terminal:
                node = self._best_child(node)
            if node.untried and not node.terminal:
                a = node.untried.pop(rng.randrange(len(node.untried)))
                ns, _, done, info = engine.step(node.state, a)
                child = _Node(ns, node, a)
                if done:
                    child.terminal, child.winner, child.untried = True, info["winner"], []
                node.children.append(child)
                node = child
            if node.terminal:
                w = _winner_to_abs(node.winner)
            else:
                w = _winner_to_abs(self._rollout(node.state, rng))
            while node is not None:
                node.N += 1
                node.W += w
                node = node.parent
        best = max(root.children, key=lambda ch: ch.N)
        ties = [ch.pa for ch in root.children if ch.N == best.N]
        return rng.choice(ties)
