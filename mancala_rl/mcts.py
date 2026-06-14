"""Monte Carlo Tree Search guided by the policy + value network (PUCT).

This is the AlphaZero-style search: instead of random rollouts, the network's
policy seeds the priors and its value replaces the rollout. Each simulation
selects down the tree by PUCT, expands one leaf (evaluating it with the net, or
reading the exact result if it's terminal), and backs the value up the path.

The one subtlety worth stating: values are stored in an *absolute*
(player-1) frame. A node interprets a value from its own player's perspective
by flipping the sign when its player isn't player 1. Because the sign depends on
*who* is to move rather than on depth parity, the extra-turn rule (same player
moves twice) is handled correctly for free -- no negamax assumption.
"""

import math

import torch

from . import engine
from .features import NUM_ACTIONS
from .network import encode_batch


class _Node:
    __slots__ = ("state", "player", "terminal", "winner",
                 "legal", "expanded", "P", "N", "W", "children")

    def __init__(self, state, terminal=False, winner=None):
        self.state = state
        self.player = state.current_player
        self.terminal = terminal
        self.winner = winner
        self.legal = [] if terminal else engine.legal_moves(state)
        self.expanded = False
        self.P = {}        # action -> prior probability
        self.N = {}        # action -> visit count
        self.W = {}        # action -> total value, from this node's perspective
        self.children = {} # action -> _Node


class MCTS:
    def __init__(self, net, device, n_simulations=100, c_puct=1.5,
                 dirichlet_alpha=None, dirichlet_frac=0.25):
        self.net = net
        self.device = device
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha   # set for self-play exploration
        self.dirichlet_frac = dirichlet_frac

    def search(self, state):
        """Run the simulations and return the root node (carries visit counts)."""
        root = _Node(state)
        self._expand(root)
        if self.dirichlet_alpha:
            self._add_dirichlet_noise(root)
        for _ in range(self.n_simulations):
            self._simulate(root)
        return root

    def _expand(self, node):
        """Evaluate node with the net, set priors, return its absolute value."""
        x = encode_batch([node.state], self.device)
        with torch.no_grad():
            logits, value = self.net(x)
        v = float(value[0])                       # from node.player's perspective
        legal = node.legal
        sub = torch.tensor([float(logits[0][a]) for a in legal])
        probs = torch.softmax(sub, dim=0).tolist()
        node.P = dict(zip(legal, probs))
        node.N = {a: 0 for a in legal}
        node.W = {a: 0.0 for a in legal}
        node.expanded = True
        return v if node.player == 1 else -v      # convert to player-1 frame

    def _simulate(self, root):
        path = []
        node = root
        while node.expanded and not node.terminal:
            a = self._select(node)
            path.append((node, a))
            child = node.children.get(a)
            if child is None:
                ns, _, done, info = engine.step(node.state, a)
                child = _Node(ns, terminal=done,
                              winner=info["winner"] if done else None)
                node.children[a] = child
            node = child

        if node.terminal:
            v_abs = 1.0 if node.winner == 1 else (-1.0 if node.winner == 2 else 0.0)
        else:
            v_abs = self._expand(node)

        for n, a in path:
            n.N[a] += 1
            n.W[a] += v_abs if n.player == 1 else -v_abs

    def _select(self, node):
        total = sum(node.N.values())
        sqrt_total = math.sqrt(total + 1)   # +1 so the prior guides the first visit too
        best_a, best_score = node.legal[0], -1e30
        for a in node.legal:
            n = node.N[a]
            q = node.W[a] / n if n > 0 else 0.0           # node's own perspective
            u = self.c_puct * node.P[a] * sqrt_total / (1 + n)
            score = q + u
            if score > best_score:
                best_score, best_a = score, a
        return best_a

    def _add_dirichlet_noise(self, root):
        import numpy as np
        legal = root.legal
        # np.random (not default_rng) so np.random.seed in training is reproducible
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(legal))
        f = self.dirichlet_frac
        for a, e in zip(legal, noise):
            root.P[a] = (1 - f) * root.P[a] + f * float(e)


def visit_counts(root):
    """Visit count per legal action -- the improved policy MCTS produces."""
    return {a: root.N[a] for a in root.legal}


def best_action(root, rng):
    """Most-visited action, ties broken with rng."""
    m = max(root.N[a] for a in root.legal)
    return rng.choice([a for a in root.legal if root.N[a] == m])


class MCTSBot:
    """Plays the most-visited move from an MCTS search. Strength scales with
    n_simulations and the quality of the network."""

    name = "MCTS"

    def __init__(self, net, device=None, n_simulations=100, c_puct=1.5):
        self.mcts = MCTS(net, device or torch.device("cpu"),
                         n_simulations, c_puct)

    def act(self, state, rng):
        return best_action(self.mcts.search(state), rng)
