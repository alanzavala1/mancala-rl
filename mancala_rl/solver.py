"""Exact solver for the Mancala engine (alpha-beta + transposition table).

Kalah(6,4) is a solved game, so a perfect player is computable. This module
provides one. It serves three purposes for the project:

  1. A perfect-play opponent (SolverBot) for evaluation.
  2. The exact game value of any position -- in particular the opening, which
     lets us confirm the published result (first player wins by a small margin)
     on our exact rules instead of taking it on faith.
  3. An optimality oracle: solve(state) returns the optimal value and every
     optimal move, so a learning agent's move can be checked against perfect.

Two design points worth calling out:

* No negamax. Most minimax code flips the sign of the value every ply, assuming
  players alternate. This game has extra turns (land in your store -> move
  again), so players do NOT always alternate. We use explicit min/max keyed on
  whose turn it actually is: player 1 maximizes the final seed margin
  (store1 - store2), player 2 minimizes it. Consecutive same-player moves then
  fall out correctly.

* A fast internal board. The engine's dict-based step() is correct but too slow
  for the millions of nodes a solve touches, so this module sows on a flat
  14-int board directly. test_solver.py plays many random games through both
  this core and the engine and asserts they never diverge.

Board index layout (matches engine.BOARD_ORDER):
    0..5  = A..F  (player 1 pits)      6  = store 1
    7..12 = G..L  (player 2 pits)      13 = store 2
"""

from . import engine

STORE1, STORE2 = 6, 13
P1_PITS = (0, 1, 2, 3, 4, 5)
P2_PITS = (7, 8, 9, 10, 11, 12)

# Counterclockwise successor of each board index (the full ring, both stores).
NEXT_INDEX = (1, 2, 3, 4, 5, 6, 12, 13, 7, 8, 9, 10, 11, 0)

# Pit directly opposite each pit (stores have no opposite).
OPPOSITE = {0: 7, 1: 8, 2: 9, 3: 10, 4: 11, 5: 12,
            7: 0, 8: 1, 9: 2, 10: 3, 11: 4, 12: 5}

_INF = 10 ** 9
_LOWER, _EXACT, _UPPER = -1, 0, 1


def board_index(player, action):
    """Map a pit action 0..5 to its board index for the given player."""
    return action if player == 1 else 7 + action


def action_to_pit(player, action):
    """Human-readable pit letter for an action."""
    return ("ABCDEF" if player == 1 else "GHIJKL")[action]


def legal_actions(board, player):
    """Legal pit actions 0..5 for player on a flat board tuple/list."""
    base = 0 if player == 1 else 7
    return [a for a in range(6) if board[base + a] > 0]


def apply_fast(board, player, action):
    """Sow one move on a flat board.

    Returns (child_board, child_player, done, margin), where child_board is a
    14-tuple, child_player is 1/2 (None if done), done is bool, and margin is
    store1 - store2 when done else None. Faithful to engine._sow / _resolve.
    """
    b = list(board)
    idx = board_index(player, action)
    own_store = STORE1 if player == 1 else STORE2
    opp_store = STORE2 if player == 1 else STORE1
    own_pits = P1_PITS if player == 1 else P2_PITS

    seeds = b[idx]
    b[idx] = 0
    while seeds:
        idx = NEXT_INDEX[idx]
        if idx == opp_store:
            continue
        b[idx] += 1
        seeds -= 1

    extra_turn = idx == own_store
    if not extra_turn and idx in own_pits and b[idx] == 1:
        opp = OPPOSITE[idx]                       # empty-capture: opp may be 0
        b[own_store] += b[idx] + b[opp]
        b[idx] = 0
        b[opp] = 0

    p1 = b[0] + b[1] + b[2] + b[3] + b[4] + b[5]
    p2 = b[7] + b[8] + b[9] + b[10] + b[11] + b[12]
    if p1 == 0 or p2 == 0:
        if p1 == 0:
            b[STORE2] += p2
            for i in P2_PITS:
                b[i] = 0
        else:
            b[STORE1] += p1
            for i in P1_PITS:
                b[i] = 0
        return tuple(b), None, True, b[STORE1] - b[STORE2]

    next_player = player if extra_turn else (2 if player == 1 else 1)
    return tuple(b), next_player, False, None


def minimax_plain(board, player):
    """Exact minimax with no pruning and no table. Slow, obviously correct.

    This exists only as the reference the optimized Solver is tested against.
    Only call it on non-terminal boards with few seeds in play.
    """
    best = None
    for a in legal_actions(board, player):
        child, cp, done, margin = apply_fast(board, player, a)
        val = margin if done else minimax_plain(child, cp)
        if best is None:
            best = val
        elif player == 1:
            best = max(best, val)
        else:
            best = min(best, val)
    return best


class Solver:
    """Exact alpha-beta solver with a transposition table.

    The table holds absolute (depth-independent) results because every search
    runs to terminal positions, so it can be reused freely across moves and
    games. nodes counts visited positions for measurement.
    """

    def __init__(self):
        self.tt = {}
        self.nodes = 0

    def _value(self, board, player, alpha, beta):
        self.nodes += 1
        alpha_orig, beta_orig = alpha, beta

        key = (board, player)
        tt_best = None
        entry = self.tt.get(key)
        if entry is not None:
            val, flag, tt_best = entry
            if flag == _EXACT:
                return val
            if flag == _LOWER:
                if val > alpha:
                    alpha = val
            elif val < beta:
                beta = val
            if alpha >= beta:
                return val

        # Generate and order children. Extra-turn moves and bigger store gains
        # first -- good guesses prune more of the tree.
        children = []
        for a in legal_actions(board, player):
            child, cp, done, margin = apply_fast(board, player, a)
            extra = (not done) and cp == player
            gain = (child[STORE1] - board[STORE1]) if player == 1 \
                else (child[STORE2] - board[STORE2])
            children.append((a, child, cp, done, margin, extra, gain))
        children.sort(key=lambda c: (c[5], c[6]), reverse=True)
        if tt_best is not None:
            children.sort(key=lambda c: c[0] != tt_best)  # known-best first

        best_a = children[0][0]
        if player == 1:                       # maximize the margin
            value = -_INF
            for a, child, cp, done, margin, _, _ in children:
                v = margin if done else self._value(child, cp, alpha, beta)
                if v > value:
                    value, best_a = v, a
                if value > alpha:
                    alpha = value
                if alpha >= beta:
                    break
        else:                                 # minimize the margin
            value = _INF
            for a, child, cp, done, margin, _, _ in children:
                v = margin if done else self._value(child, cp, alpha, beta)
                if v < value:
                    value, best_a = v, a
                if value < beta:
                    beta = value
                if beta <= alpha:
                    break

        if value <= alpha_orig:
            flag = _UPPER
        elif value >= beta_orig:
            flag = _LOWER
        else:
            flag = _EXACT
        self.tt[key] = (value, flag, best_a)
        return value

    def value(self, board, player):
        """Exact value (store1 - store2 under perfect play) of a position."""
        return self._value(board, player, -_INF, _INF)

    def solve(self, state):
        """Return (value, best_moves) for state.

        value is the final margin under perfect play; best_moves is every
        action that achieves it (so callers can tie-break as they like).
        """
        board, player = state.board, state.current_player
        value = self._value(board, player, -_INF, _INF)
        best_moves = []
        for a in legal_actions(board, player):
            child, cp, done, margin = apply_fast(board, player, a)
            v = margin if done else self._value(child, cp, -_INF, _INF)
            if v == value:
                best_moves.append(a)
        return value, best_moves


def solve(state):
    """Convenience: solve a single position with a fresh solver."""
    return Solver().solve(state)


class SolverBot:
    """Perfect-play opponent. Keeps a transposition table across moves/games,
    so after the first full solve every later position is fast.
    """

    name = "Solver"

    def __init__(self):
        self._solver = Solver()

    def act(self, state, rng):
        _, best_moves = self._solver.solve(state)
        return rng.choice(best_moves)
