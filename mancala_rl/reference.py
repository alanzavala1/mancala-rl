"""Brute-force exact reference -- the simplest possible correct solver.

No pruning, no transposition table: just full minimax over the game tree. It is
correct by construction and trivially readable, which is its whole point -- it
is the trusted oracle the fast C solver (mancala_rl.csolver) is tested against.
Because it has no pruning it is exponential, so only call it on small positions
(few seeds in play). The real solver for anything larger is the C one.
"""

from . import engine


def minimax(state):
    """Exact game value: the final store margin (store1 - store2) under perfect
    play. Player 1 maximizes the margin, player 2 minimizes it."""
    best = None
    for a in engine.legal_moves(state):
        ns, _, done, _ = engine.step(state, a)
        if done:
            s1, s2 = engine.stores(ns)
            v = s1 - s2
        else:
            v = minimax(ns)
        if best is None:
            best = v
        elif state.current_player == 1:
            best = max(best, v)
        else:
            best = min(best, v)
    return best


def best_moves(state):
    """Every action that achieves the minimax value (so callers can tie-break)."""
    value = minimax(state)
    out = []
    for a in engine.legal_moves(state):
        ns, _, done, _ = engine.step(state, a)
        if done:
            s1, s2 = engine.stores(ns)
            v = s1 - s2
        else:
            v = minimax(ns)
        if v == value:
            out.append(a)
    return out
