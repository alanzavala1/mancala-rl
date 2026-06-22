"""Test that one training step actually learns (needs torch).

    .venv\\Scripts\\python tests/test_training.py
"""

import sys

import torch

from mancala_rl import engine, features
from mancala_rl.network import MancalaNet
from mancala_rl.training import ReplaySampler, describe_examples, phase_of_features
from mancala_rl.training import sample_training_batch, train_step, value_bucket


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


def test_train_step_accepts_action_aware_features():
    torch.manual_seed(0)
    device = torch.device("cpu")
    net = MancalaNet(architecture="action_aware").to(device).train()
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    feat = features.encode_action_aware(engine.reset())
    batch = [(feat, [1 / 6] * 6, 0.0)]
    loss = train_step(net, opt, batch, device)[0]
    assert loss > 0.0


def _example(pit_seeds, value=0.0, policy=None):
    pit = pit_seeds / 12 / 4
    feat = [pit] * 6 + [0.0] + [pit] * 6 + [0.0]
    policy = policy or [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    return feat, policy, value


def test_phase_of_features():
    assert phase_of_features(_example(40)[0]) == "opening"
    assert phase_of_features(_example(24)[0]) == "midgame"
    assert phase_of_features(_example(8)[0]) == "endgame"


def test_value_bucket():
    assert value_bucket(-0.5) == "behind"
    assert value_bucket(0.0) == "close"
    assert value_bucket(0.5) == "ahead"


def test_describe_examples_reports_phase_mix():
    examples = [_example(40), _example(24), _example(8), _example(8)]
    stats = describe_examples(examples)
    assert stats["opening"] == 0.25
    assert stats["midgame"] == 0.25
    assert stats["endgame"] == 0.5
    assert stats["close"] == 1.0
    assert stats["entropy"] == 0.0


def test_phase_balanced_sampler_uses_all_phases():
    import random

    examples = [_example(40), _example(36), _example(24), _example(20),
                _example(8), _example(4)]
    batch = sample_training_batch(
        examples, batch_size=6, rng=random.Random(0), phase_balanced_frac=1.0)
    phases = [phase_of_features(ex[0]) for ex in batch]
    assert set(phases) == {"opening", "midgame", "endgame"}


def test_diverse_sampler_uses_phase_and_value_buckets():
    import random

    examples = []
    for seeds in (40, 24, 8):
        for value in (-0.5, 0.0, 0.5):
            examples.append(_example(seeds, value=value))
    batch = sample_training_batch(
        examples, batch_size=9, rng=random.Random(1), diverse_frac=1.0)
    buckets = {(phase_of_features(ex[0]), value_bucket(ex[2])) for ex in batch}
    assert len(buckets) == 9


def test_priority_sampler_prefers_high_entropy_targets():
    import random

    sharp = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    spread = [1 / 6] * 6
    examples = [_example(24, policy=sharp) for _ in range(9)]
    examples.extend(_example(24, policy=spread) for _ in range(3))
    batch = sample_training_batch(
        examples, batch_size=4, rng=random.Random(2),
        priority_frac=1.0, priority_phase="midgame")
    assert all(ex[1] == spread for ex in batch)


def test_replay_sampler_reuses_cached_buckets():
    import random

    examples = []
    for seeds in (40, 24, 8):
        for value in (-0.5, 0.0, 0.5):
            examples.append(_example(seeds, value=value))
    sampler = ReplaySampler(
        examples, random.Random(3), phase_balanced_frac=0.2,
        diverse_frac=0.2, priority_frac=0.1, priority_phase="midgame")
    batch = sampler.sample(9)
    assert len(batch) == 9


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
