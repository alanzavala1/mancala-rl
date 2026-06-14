"""Policy + value network for the learning agent.

A small MLP over the perspective-canonical board features (features.py), with
two heads:
  - policy: one logit per action (the 6 "my" pits). The softmax and the
    legal-move mask are applied by the caller (MCTS), not here.
  - value: a single number in [-1, 1] (tanh) -- the expected game result from
    the perspective of the player to move (+1 = that player wins, -1 = loses).

Deliberately small: the game is tiny, so two hidden layers is plenty.
"""

import torch
import torch.nn as nn

from . import features


def default_device():
    """cuda if a GPU is available, else cpu."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MancalaNet(nn.Module):
    def __init__(self, hidden=128, layers=2):
        super().__init__()
        body = [nn.Linear(features.NUM_FEATURES, hidden), nn.ReLU()]
        for _ in range(layers - 1):
            body += [nn.Linear(hidden, hidden), nn.ReLU()]
        self.body = nn.Sequential(*body)
        self.policy_head = nn.Linear(hidden, features.NUM_ACTIONS)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, x):
        """x: (batch, 14) float tensor -> (policy_logits (batch, 6),
        value (batch,))."""
        h = self.body(x)
        policy_logits = self.policy_head(h)
        value = torch.tanh(self.value_head(h)).squeeze(-1)
        return policy_logits, value


def encode_batch(states, device):
    """Encode a list of states into a (len(states), 14) float tensor."""
    rows = [features.encode(s) for s in states]
    return torch.tensor(rows, dtype=torch.float32, device=device)
