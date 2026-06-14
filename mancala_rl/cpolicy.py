"""Python interface to the C policy-inference DLL (cnn/policy_net.dll).

The network's forward pass runs in pure C with the weights baked in, so a move
is microsecond-scale. Build it first:
    python scripts/export_weights.py
    powershell cnn/build.ps1
"""

import ctypes
import pathlib

from . import engine

_DLL = pathlib.Path(__file__).resolve().parent.parent / "cnn" / "policy_net.dll"
try:
    _lib = ctypes.CDLL(str(_DLL))
except OSError as e:  # pragma: no cover - depends on build state
    raise OSError(
        f"could not load {_DLL}. Build it: "
        f"python scripts/export_weights.py ; powershell cnn/build.ps1"
    ) from e

_int_p = ctypes.POINTER(ctypes.c_int)
_float_p = ctypes.POINTER(ctypes.c_float)
_lib.policy_best_move.argtypes = [_int_p, ctypes.c_int]
_lib.policy_best_move.restype = ctypes.c_int
_lib.policy_logits.argtypes = [_int_p, ctypes.c_int, _float_p]
_lib.policy_logits.restype = None
_lib.policy_bench.argtypes = [_int_p, ctypes.c_int, ctypes.c_int]
_lib.policy_bench.restype = ctypes.c_double


class CPolicyBot:
    """Plays the policy network's move via the pure-C forward pass."""

    name = "CPolicy"

    def act(self, state, rng):
        board = (ctypes.c_int * 14)(*state.board)
        return _lib.policy_best_move(board, state.current_player)


def logits(state):
    board = (ctypes.c_int * 14)(*state.board)
    out = (ctypes.c_float * 6)()
    _lib.policy_logits(board, state.current_player, out)
    return list(out)


def bench_us(state, reps=5_000_000):
    """Pure-C microseconds per move (no ctypes/Python overhead)."""
    board = (ctypes.c_int * 14)(*state.board)
    return _lib.policy_bench(board, state.current_player, reps)
