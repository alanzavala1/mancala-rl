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

import torch
import torch.nn.functional as F


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
