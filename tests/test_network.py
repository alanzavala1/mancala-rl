"""Tests for the policy + value network (needs torch).

    .venv\\Scripts\\python tests/test_network.py
    pytest tests/test_network.py
"""

import sys
import pathlib
import tempfile

import torch

from mancala_rl import engine, features
from mancala_rl.network import MancalaNet, default_device, encode_batch, load_net, save_net


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


def test_residual_layer_norm_forward_shapes():
    device = default_device()
    net = MancalaNet(hidden=64, layers=3, residual=True, layer_norm=True).to(device).eval()
    x = encode_batch(_some_states(), device)
    with torch.no_grad():
        logits, value = net(x)
    assert logits.shape == (len(_some_states()), features.NUM_ACTIONS)
    assert value.shape == (len(_some_states()),)


def test_action_aware_forward_shapes():
    device = default_device()
    net = MancalaNet(hidden=64, layers=2, architecture="action_aware").to(device).eval()
    states = _some_states()
    x = encode_batch(states, device, architecture="action_aware")
    assert x.shape == (len(states), features.ACTION_AWARE_FEATURES)
    with torch.no_grad():
        logits, value = net(x)
    assert logits.shape == (len(states), features.NUM_ACTIONS)
    assert value.shape == (len(states),)


def test_save_load_preserves_architecture_and_value_metadata():
    net = MancalaNet(hidden=64, layers=3, residual=True, layer_norm=True,
                     architecture="action_aware")
    net.value_mode = "tanh"
    net.value_scale = 12.0
    with tempfile.TemporaryDirectory() as d:
        path = pathlib.Path(d) / "net.pt"
        save_net(net, path)
        loaded = load_net(path)
    assert loaded.hidden == 64
    assert loaded.layers == 3
    assert loaded.residual is True
    assert loaded.layer_norm is True
    assert loaded.architecture == "action_aware"
    assert loaded.value_mode == "tanh"
    assert loaded.value_scale == 12.0


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
