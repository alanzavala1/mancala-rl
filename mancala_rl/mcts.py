"""Monte Carlo Tree Search guided by the policy + value network (PUCT).

This is the AlphaZero-style search: instead of random rollouts, the network's
policy seeds the priors and its value replaces the rollout. Each simulation
selects down the tree by PUCT, expands one leaf (evaluating it with the net, or
reading the exact result if it's terminal), and backs the value up the path.

One subtlety: values are stored in an *absolute* (player-1) frame. A node reads a
value from its own player's perspective by flipping the sign when its player
isn't player 1. Because the sign depends on *who* is to move rather than on depth
parity, the extra-turn rule (the same player moving twice) is handled correctly
for free -- no negamax assumption.

I also run a score-free MCTS-Solver (Winands & Bjornsson, 2008): terminal
positions are *proven* wins/losses/draws, and those proofs propagate up the tree
(min/max keyed on the mover, in the same absolute frame). The search then takes a
proven win when it sees one and never walks into a proven loss, so it plays the
end of the game exactly rather than only averaging playouts -- the part where the
agent used to throw away won endgames.
"""

import math

import torch

from . import engine
from .features import NUM_ACTIONS, value_target
from .network import encode_batch


class _Node:
    __slots__ = ("state", "player", "terminal", "winner",
                 "legal", "expanded", "P", "N", "W", "children", "proven")

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
        # MCTS-Solver: proven game-theoretic value in the absolute (player-1)
        # frame -- +1 a proven player-1 win, -1 a proven player-2 win, 0 a proven
        # draw, None not yet proven. Terminals are proven on creation.
        if terminal:
            self.proven = 1 if winner == 1 else (-1 if winner == 2 else 0)
        else:
            self.proven = None


class MCTS:
    def __init__(self, net, device, n_simulations=100, c_puct=1.5,
                 dirichlet_alpha=None, dirichlet_frac=0.25,
                 value_mode="clip", value_scale=16):
        self.net = net
        self.device = device
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha   # set for self-play exploration
        self.dirichlet_frac = dirichlet_frac
        self.value_mode = value_mode
        self.value_scale = value_scale
        self.architecture = getattr(net, "architecture", "mlp")

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
        x = encode_batch([node.state], self.device, self.architecture)
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
            s1, s2 = engine.stores(node.state)     # margin-based terminal value (P1 frame)
            v_abs = value_target(s1 - s2, self.value_mode, self.value_scale)
        else:
            v_abs = self._expand(node)

        for n, a in path:
            n.N[a] += 1
            n.W[a] += v_abs if n.player == 1 else -v_abs

        # MCTS-Solver: ripple proven win/loss/draw values up from the leaf.
        for n, _ in reversed(path):
            self._update_proof(n)

    def _select(self, node):
        win_val = 1 if node.player == 1 else -1   # value the mover wants (absolute frame)
        total = sum(node.N.values())
        sqrt_total = math.sqrt(total + 1)   # +1 so the prior guides the first visit too
        best_a, best_score = None, -1e30
        for a in node.legal:
            child = node.children.get(a)
            if child is not None and child.proven is not None:
                if child.proven == win_val:
                    return a                    # a proven win -- take it outright
                if child.proven == -win_val:
                    continue                    # a proven loss -- never walk into it
            n = node.N[a]
            q = node.W[a] / n if n > 0 else 0.0           # node's own perspective
            u = self.c_puct * node.P[a] * sqrt_total / (1 + n)
            score = q + u
            if score > best_score:
                best_score, best_a = score, a
        return best_a if best_a is not None else node.legal[0]

    def _update_proof(self, node):
        """Set node.proven (absolute frame) once its value is forced by its
        children: the mover has a proven win if ANY child is a proven win for it;
        a node is a proven loss/draw only once EVERY legal move is expanded and
        proven. The mover maximizes the sign if player 1, minimizes it if 2."""
        if node.proven is not None or node.terminal:
            return
        win_val = 1 if node.player == 1 else -1
        proven_children = []
        all_proven = True
        for a in node.legal:
            child = node.children.get(a)
            if child is None or child.proven is None:
                all_proven = False
                continue
            if child.proven == win_val:
                node.proven = win_val           # a forced win for the mover
                return
            proven_children.append(child.proven)
        if all_proven and proven_children:      # every move explored, none winning
            node.proven = max(proven_children) if node.player == 1 else min(proven_children)

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
    """Pick the move to play: a proven win if the solver found one, else the
    most-visited move -- and never a proven loss when something else exists."""
    win_val = 1 if root.player == 1 else -1
    proven_wins = [a for a in root.legal
                   if root.children.get(a) is not None
                   and root.children[a].proven == win_val]
    if proven_wins:
        return rng.choice(proven_wins)
    safe = [a for a in root.legal
            if not (root.children.get(a) is not None
                    and root.children[a].proven == -win_val)]
    pool = safe or root.legal
    m = max(root.N[a] for a in pool)
    return rng.choice([a for a in pool if root.N[a] == m])


class MCTSBot:
    """Plays the most-visited move from an MCTS search. Strength scales with
    n_simulations and the quality of the network."""

    name = "MCTS"

    def __init__(self, net, device=None, n_simulations=100, c_puct=1.5,
                 value_mode=None, value_scale=None):
        value_mode = value_mode or getattr(net, "value_mode", "clip")
        value_scale = value_scale or getattr(net, "value_scale", 16)
        self.mcts = MCTS(net, device or torch.device("cpu"),
                         n_simulations, c_puct, value_mode=value_mode,
                         value_scale=value_scale)

    def act(self, state, rng):
        return best_action(self.mcts.search(state), rng)
