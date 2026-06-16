"""Test that one training step actually learns (needs torch).

    .venv\\Scripts\\python tests/test_training.py
"""

import sys

import torch

from mancala_rl.network import MancalaNet
from mancala_rl.training import train_step


def _fixed_batch():
    # f2's own pits (indices 0..5) are non-empty only at 0, 2, 4, so the policy
    # target must put mass only there (matches the new legal-move masking).
    f1 = [0.08] * 6 + [0.0] + [0.08] * 6 + [0.0]
    f2 = [0.1, 0.0, 0.1, 0.0, 0.1, 0.0, 0.0, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.0]
    return [
        (f1, [0.0, 0.0, 1.0, 0.0, 0.0, 0.0], 1.0),
        (f2, [0.5, 0.0, 0.5, 0.0, 0.0, 0.0], -1.0),
    ]


def test_train_step_reduces_loss():
    torch.manual_seed(0)
    device = torch.device("cpu")
    net = MancalaNet().to(device).train()
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    batch = _fixed_batch()
    first = train_step(net, opt, batch, device)[0]
    for _ in range(150):
        last = train_step(net, opt, batch, device)[0]
    # repeatedly training on the same batch should drive the loss down a lot
    assert last < first, f"loss did not drop: {first:.3f} -> {last:.3f}"
    assert last < 0.5 * first


def _run_all():
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
