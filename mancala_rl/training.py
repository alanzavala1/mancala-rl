"""One supervised training step on self-play data.

Given a batch of (features, policy_target, value_target) examples:
  - policy loss: cross-entropy pushing the policy head toward the MCTS visit
    distribution (the looked-ahead policy, which is better than the raw net's).
  - value loss: mean-squared error pushing the value head toward the actual
    game result (the final-margin target -- see features.margin_value).
  - total loss = policy loss + value_weight * value loss; the optimizer lowers it.

The full self-play -> train -> evaluate loop that calls this lives in
scripts/train.py.
"""

import math

import torch
import torch.nn.functional as F

from . import engine


_PIT_FEATURES = tuple(range(0, 6)) + tuple(range(7, 13))
_PHASES = ("opening", "midgame", "endgame")
_VALUE_BUCKETS = ("behind", "close", "ahead")


def phase_of_features(feat):
    """Return opening/midgame/endgame from encoded board features.

    Features are stored from the mover's perspective, but pit counts are still
    all twelve non-store counts scaled by STARTING_SEEDS. The thresholds mirror
    scripts/solver_regret.py so training diagnostics line up with solver-regret
    evaluation.
    """
    remaining = sum(feat[i] for i in _PIT_FEATURES) * engine.STARTING_SEEDS
    if remaining >= 32:
        return "opening"
    if remaining >= 16:
        return "midgame"
    return "endgame"


def policy_entropy(policy):
    """Entropy of a policy target, in nats."""
    ent = 0.0
    for p in policy:
        if p > 0:
            ent -= p * math.log(p)
    return ent


def value_bucket(value):
    """Bucket a mover-relative value target for replay diagnostics."""
    if value <= -0.33:
        return "behind"
    if value >= 0.33:
        return "ahead"
    return "close"


def describe_examples(examples):
    """Summarize replay examples for training logs."""
    n = len(examples)
    if n == 0:
        return {
            "opening": 0, "midgame": 0, "endgame": 0,
            "behind": 0, "close": 0, "ahead": 0,
            "entropy": 0.0, "abs_value": 0.0,
        }
    counts = {p: 0 for p in _PHASES}
    value_counts = {b: 0 for b in _VALUE_BUCKETS}
    entropy = 0.0
    abs_value = 0.0
    for feat, policy, value in examples:
        counts[phase_of_features(feat)] += 1
        value_counts[value_bucket(value)] += 1
        entropy += policy_entropy(policy)
        abs_value += abs(value)
    return {
        "opening": counts["opening"] / n,
        "midgame": counts["midgame"] / n,
        "endgame": counts["endgame"] / n,
        "behind": value_counts["behind"] / n,
        "close": value_counts["close"] / n,
        "ahead": value_counts["ahead"] / n,
        "entropy": entropy / n,
        "abs_value": abs_value / n,
    }


def _sample_from_buckets(batch, buckets, n, rng):
    keys = [k for k in buckets if buckets[k]]
    if not keys:
        return
    for i in range(n):
        batch.append(rng.choice(buckets[keys[i % len(keys)]]))


def _top_entropy_examples(examples, priority_phase):
    pool = examples
    if priority_phase != "all":
        pool = [ex for ex in examples if phase_of_features(ex[0]) == priority_phase]
    if not pool:
        pool = examples
    return sorted(pool, key=lambda ex: policy_entropy(ex[1]), reverse=True)


class ReplaySampler:
    """Reusable replay sampler for one frozen buffer snapshot.

    Building phase/value buckets and entropy rankings is useful, but doing it
    once per gradient step is wasteful. Training creates one ReplaySampler per
    iteration after converting the replay deque to a list, then reuses it for
    every batch in that iteration.
    """

    def __init__(self, examples, rng, phase_balanced_frac=0.0, diverse_frac=0.0,
                 priority_frac=0.0, priority_phase="midgame"):
        if not examples:
            raise ValueError("cannot sample from an empty replay buffer")
        if priority_phase not in ("all",) + _PHASES:
            raise ValueError(f"unknown priority phase: {priority_phase}")

        total_frac = phase_balanced_frac + diverse_frac + priority_frac
        if total_frac > 1.0:
            raise ValueError("replay sampling fractions must sum to <= 1.0")

        self.examples = examples
        self.rng = rng
        self.phase_balanced_frac = phase_balanced_frac
        self.diverse_frac = diverse_frac
        self.priority_frac = priority_frac
        self.priority_phase = priority_phase
        self.by_phase = {p: [] for p in _PHASES}
        self.by_phase_value = {
            (p, v): [] for p in _PHASES for v in _VALUE_BUCKETS
        }

        priority_pool = []
        for ex in examples:
            phase = phase_of_features(ex[0])
            self.by_phase[phase].append(ex)
            self.by_phase_value[(phase, value_bucket(ex[2]))].append(ex)
            if priority_phase == "all" or phase == priority_phase:
                priority_pool.append(ex)
        if not priority_pool:
            priority_pool = examples
        ranked = sorted(priority_pool, key=lambda ex: policy_entropy(ex[1]), reverse=True)
        top_n = max(1, min(len(ranked), len(ranked) // 4 or 1))
        self.priority_pool = ranked[:top_n]

    def sample(self, batch_size):
        batch_size = min(batch_size, len(self.examples))
        balanced_n = max(0, int(round(batch_size * self.phase_balanced_frac)))
        diverse_n = max(0, int(round(batch_size * self.diverse_frac)))
        priority_n = max(0, int(round(batch_size * self.priority_frac)))
        assigned = min(batch_size, balanced_n + diverse_n + priority_n)
        uniform_n = batch_size - assigned

        batch = self.rng.sample(self.examples, uniform_n) if uniform_n else []
        if balanced_n:
            _sample_from_buckets(batch, self.by_phase, balanced_n, self.rng)
        if diverse_n:
            _sample_from_buckets(batch, self.by_phase_value, diverse_n, self.rng)
        for _ in range(priority_n):
            batch.append(self.rng.choice(self.priority_pool))
        self.rng.shuffle(batch)
        return batch


def sample_training_batch(examples, batch_size, rng, phase_balanced_frac=0.0,
                          diverse_frac=0.0, priority_frac=0.0,
                          priority_phase="midgame"):
    """Sample a batch with optional diversity and priority components.

    The default is uniform replay, preserving the original behavior. When
    phase_balanced_frac > 0, that fraction of each batch is split evenly across
    opening, midgame, and endgame examples, with replacement if a bucket is
    smaller than requested. The rest remains uniform recent replay.

    diverse_frac samples evenly across phase x value buckets. priority_frac
    samples from high-entropy MCTS targets, optionally focused on one phase.
    """
    sampler = ReplaySampler(
        examples, rng, phase_balanced_frac, diverse_frac,
        priority_frac, priority_phase)
    return sampler.sample(batch_size)


def train_step(net, optimizer, batch, device, value_weight=1.0):
    """Do one gradient update. Returns (total, policy, value) losses as floats.

    value_weight scales the value loss against the policy loss (1.0 = AlphaZero's
    default; raise it if the value head -- now a margin regressor -- learns slowly)."""
    feats = torch.tensor([ex[0] for ex in batch], dtype=torch.float32, device=device)
    target_p = torch.tensor([ex[1] for ex in batch], dtype=torch.float32, device=device)
    target_v = torch.tensor([ex[2] for ex in batch], dtype=torch.float32, device=device)

    logits, value = net(feats)
    # Mask illegal moves (own pits 0..5 that are empty) before the softmax, so
    # training matches how priors are used at play time -- MCTS softmaxes over
    # legal moves only. Without this the net is trained under a different
    # normalization than it's used with.
    legal = feats[:, :6] > 0
    masked_logits = logits.masked_fill(~legal, -1e9)
    log_probs = F.log_softmax(masked_logits, dim=1)
    policy_loss = -(target_p * log_probs).sum(dim=1).mean()
    value_loss = F.mse_loss(value, target_v)
    loss = policy_loss + value_weight * value_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item(), policy_loss.item(), value_loss.item()
