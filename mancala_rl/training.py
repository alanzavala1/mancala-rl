"""One supervised training step on self-play data.

Given a batch of (features, policy_target, value_target) examples:
  - policy loss: cross-entropy pushing the policy head toward the MCTS visit
    distribution (the looked-ahead, better policy).
  - value loss: mean-squared error pushing the value head toward the actual
    game result.
  - total loss = policy loss + value loss; the optimizer lowers it.

The full self-play -> train -> gate loop that calls this lives in
scripts/train.py.
"""

import torch
import torch.nn.functional as F


def train_step(net, optimizer, batch, device):
    """Do one gradient update. Returns (total, policy, value) losses as floats."""
    feats = torch.tensor([ex[0] for ex in batch], dtype=torch.float32, device=device)
    target_p = torch.tensor([ex[1] for ex in batch], dtype=torch.float32, device=device)
    target_v = torch.tensor([ex[2] for ex in batch], dtype=torch.float32, device=device)

    logits, value = net(feats)
    # cross-entropy between the target distribution and the network's softmax.
    # Illegal moves have target 0, so they don't pull the loss directly (and at
    # play time MCTS only ever considers legal moves anyway).
    log_probs = F.log_softmax(logits, dim=1)
    policy_loss = -(target_p * log_probs).sum(dim=1).mean()
    value_loss = F.mse_loss(value, target_v)
    loss = policy_loss + value_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item(), policy_loss.item(), value_loss.item()
