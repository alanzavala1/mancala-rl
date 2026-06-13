"""Python interface to the C solver (csolver/mancala_solver.dll).

The DLL does the game-tree search; this module just marshals the board across
the ctypes boundary and exposes a bot that fits the same act(state, rng)
interface as the other agents. Build the DLL first with csolver/build.ps1.
"""

import ctypes
import pathlib

from . import engine

_NEG_INF = -1000000

_DLL_PATH = pathlib.Path(__file__).resolve().parent.parent / "csolver" / "mancala_solver.dll"

try:
    _lib = ctypes.CDLL(str(_DLL_PATH))
except OSError as e:  # pragma: no cover - depends on build state
    raise OSError(
        f"could not load {_DLL_PATH}. Build it first: "
        f"powershell csolver/build.ps1"
    ) from e

_int_p = ctypes.POINTER(ctypes.c_int)

_lib.move_values.argtypes = [_int_p, ctypes.c_int, ctypes.c_int, _int_p]
_lib.move_values.restype = None

_lib.apply_one.argtypes = [_int_p, ctypes.c_int, ctypes.c_int, _int_p]
_lib.apply_one.restype = ctypes.c_int


def move_values(state, depth):
    """Return a list of 6 values (player-1 perspective), one per pit action,
    searched to `depth` plies. Illegal actions hold a sentinel."""
    board = (ctypes.c_int * 14)(*state.board)
    out = (ctypes.c_int * 6)()
    _lib.move_values(board, state.current_player, depth, out)
    return list(out)


def apply_one(board, player, action):
    """Apply one move via the C core. Returns (next_board, next_player_or_0).
    Used to cross-check the C rules against the Python engine."""
    inp = (ctypes.c_int * 14)(*board)
    out = (ctypes.c_int * 14)()
    nxt = _lib.apply_one(inp, player, action, out)
    return tuple(out), nxt


class CSolverBot:
    """Depth-limited alpha-beta player backed by the C solver.

    depth is the lookahead in plies; higher is stronger and slower. The search
    is exact (perfect) once the remaining game fits inside the budget.
    """

    name = "CSolver"

    def __init__(self, depth=12):
        self.depth = depth

    def act(self, state, rng):
        values = move_values(state, self.depth)
        legal = engine.legal_moves(state)
        if state.current_player == 1:
            best = max(values[a] for a in legal)
        else:
            best = min(values[a] for a in legal)
        ties = [a for a in legal if values[a] == best]
        return rng.choice(ties)
