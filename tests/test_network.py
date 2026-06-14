"""Tests for the policy + value network (needs torch).

    .venv\\Scripts\\python tests/test_network.py
    pytest tests/test_network.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch

from mancala_rl import engine, features
from mancala_rl.network import MancalaNet, default_device, encode_batch


def _some_states():
    states = [engine.reset()]
    s = engine.reset()
    for a in engine.legal_moves(s)[:3]:
        ns, _, _, _ = engine.step(s, a)
        states.append(ns)
    return states


def test_forward_shapes():
    device = default_device()
    net = MancalaNet().to(device).eval()
    states = _some_states()
    x = encode_batch(states, device)
    with torch.no_grad():
        logits, value = net(x)
    assert logits.shape == (len(states), features.NUM_ACTIONS)
    assert value.shape == (len(states),)


def test_value_in_range():
    device = default_device()
    net = MancalaNet().to(device).eval()
    x = encode_batch(_some_states(), device)
    with torch.no_grad():
        _, value = net(x)
    assert torch.all(value >= -1.0) and torch.all(value <= 1.0)


def test_eval_is_deterministic():
    device = default_device()
    net = MancalaNet().to(device).eval()
    x = encode_batch([engine.reset()], device)
    with torch.no_grad():
        l1, v1 = net(x)
        l2, v2 = net(x)
    assert torch.allclose(l1, l2) and torch.allclose(v1, v2)


def _run_all():
    print("device:", default_device())
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return failures


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
