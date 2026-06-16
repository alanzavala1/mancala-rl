"""Gumbel AlphaZero search (Danihelka et al., ICLR 2022).

A more sample-efficient planning step than vanilla PUCT MCTS: it samples a set
of candidate root actions with the Gumbel-Top-k trick, splits a small simulation
budget among them by sequential halving, and builds the training policy target
from "completed Q-values" -- which guarantees a policy *improvement* even with
very few simulations, so each self-play move teaches more per unit of compute.

This is "Gumbel at the root" with our standard PUCT for the interior descent --
the faithful, lower-risk variant. It is kept fully self-contained (its own tree)
so the proven PUCT search in mcts.py is untouched: the working agent is safe.
"""

import math
import random

import torch

from . import engine
from .features import NUM_ACTIONS, margin_value
from .network import encode_batch


def _gumbel(rng):
    u = min(max(rng.random(), 1e-12), 1.0 - 1e-12)
    return -math.log(-math.log(u))


class _Node:
    __slots__ = ("state", "player", "terminal", "winner", "legal",
                 "expanded", "P", "N", "W", "children")

    def __init__(self, state, terminal=False, winner=None):
        self.state = state
        self.player = state.current_player
        self.terminal = terminal
        self.winner = winner
        self.legal = [] if terminal else engine.legal_moves(state)
        self.expanded = False
        self.P = {}
        self.N = {}
        self.W = {}
        self.children = {}


class GumbelMCTS:
    """Returns (action_to_play, improved_policy_target, root_value) from search."""

    def __init__(self, net, device, n_simulations=64, m=16,
                 c_visit=50.0, c_scale=1.0, c_puct=1.5, rng=None):
        self.net = net
        self.device = device
        self.n = n_simulations
        self.m = m
        self.c_visit = c_visit
        self.c_scale = c_scale
        self.c_puct = c_puct
        self.rng = rng or random.Random()

    def _eval(self, node):
        """Expand node; return (raw logits over legal, value from node.player's view)."""
        x = encode_batch([node.state], self.device)
        with torch.no_grad():
            logits, value = self.net(x)
        v = float(value[0])
        lg = {a: float(logits[0][a]) for a in node.legal}
        hi = max(lg.values())
        ex = {a: math.exp(lg[a] - hi) for a in node.legal}
        z = sum(ex.values())
        node.P = {a: ex[a] / z for a in node.legal}      # softmax priors for PUCT interior
        node.N = {a: 0 for a in node.legal}
        node.W = {a: 0.0 for a in node.legal}
        node.expanded = True
        return lg, v

    def _child(self, node, a):
        c = node.children.get(a)
        if c is None:
            ns, _, done, info = engine.step(node.state, a)
            c = _Node(ns, terminal=done, winner=info["winner"] if done else None)
            node.children[a] = c
        return c

    def _select(self, node):
        total = sum(node.N.values())
        s = math.sqrt(total + 1)
        best, best_score = node.legal[0], -1e30
        for a in node.legal:
            n = node.N[a]
            q = node.W[a] / n if n > 0 else 0.0
            u = self.c_puct * node.P[a] * s / (1 + n)
            if q + u > best_score:
                best_score, best = q + u, a
        return best

    def _playout(self, root, first_action):
        """One simulation that starts by taking first_action at the root."""
        path = [(root, first_action)]
        node = self._child(root, first_action)
        while node.expanded and not node.terminal:
            a = self._select(node)
            path.append((node, a))
            node = self._child(node, a)
        if node.terminal:
            s1, s2 = engine.stores(node.state)
            v_abs = margin_value(s1 - s2)
        else:
            _, v = self._eval(node)
            v_abs = v if node.player == 1 else -v
        for nd, a in path:
            nd.N[a] += 1
            nd.W[a] += v_abs if nd.player == 1 else -v_abs

    def _q(self, root, a, fallback):
        return root.W[a] / root.N[a] if root.N[a] > 0 else fallback

    def search(self, state):
        root = _Node(state)
        logits, root_v = self._eval(root)          # root_v: value from the mover's view
        legal = root.legal
        if len(legal) == 1:
            pol = [0.0] * NUM_ACTIONS
            pol[legal[0]] = 1.0
            return legal[0], pol, root_v

        # Gumbel-Top-k: sample m candidate actions (without replacement).
        gscore = {a: logits[a] + _gumbel(self.rng) for a in legal}
        m = min(self.m, len(legal))
        survivors = sorted(legal, key=lambda a: gscore[a], reverse=True)[:m]

        # Sequential halving: split the budget across rounds, drop the worst half each round.
        n_phases = max(1, math.ceil(math.log2(m)))
        while len(survivors) > 1:
            per = max(1, (self.n // n_phases) // len(survivors))
            for a in survivors:
                for _ in range(per):
                    self._playout(root, a)
            max_n = max(root.N.values())
            survivors.sort(
                key=lambda a: gscore[a] + (self.c_visit + max_n) * self.c_scale
                * self._q(root, a, 0.0),
                reverse=True)
            survivors = survivors[:max(1, len(survivors) // 2)]
        chosen = survivors[0]

        # Improved policy target from completed Q-values (missing Q -> root value).
        max_n = max(root.N.values()) if root.N else 0
        base = {a: logits[a] + (self.c_visit + max_n) * self.c_scale
                * self._q(root, a, root_v) for a in legal}
        hi = max(base.values())
        ex = {a: math.exp(base[a] - hi) for a in legal}
        z = sum(ex.values())
        policy = [0.0] * NUM_ACTIONS
        for a in legal:
            policy[a] = ex[a] / z
        return chosen, policy, root_v


class GumbelBot:
    """Play the move Gumbel search selects (for evaluation, if wanted)."""

    name = "Gumbel"

    def __init__(self, net, device=None, n_simulations=64, m=16):
        self.search = GumbelMCTS(net, device or torch.device("cpu"),
                                 n_simulations=n_simulations, m=m).search

    def act(self, state, rng):
        return self.search(state)[0]
