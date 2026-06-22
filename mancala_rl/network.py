"""Policy + value network for the learning agent.

A small MLP over the perspective-canonical board features (features.py), with
two heads:
  - policy: one logit per action (the 6 "my" pits). The softmax and the
    legal-move mask are applied by the caller (MCTS), not here.
  - value: a single number in [-1, 1] (tanh) -- the expected game result from
    the perspective of the player to move (+1 = that player wins, -1 = loses).

The default architecture stays as the original MLP for checkpoint compatibility.
The optional action-aware architecture keeps the same board trunk and scores
each move with exact per-action consequence features from features.py.
"""

import torch
import torch.nn as nn

from . import engine, features


def default_device():
    """cuda if a GPU is available, else cpu."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _ResidualBlock(nn.Module):
    def __init__(self, hidden, layer_norm=False):
        super().__init__()
        self.fc = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden) if layer_norm else nn.Identity()

    def forward(self, x):
        return torch.relu(x + self.norm(self.fc(x)))


class MancalaNet(nn.Module):
    def __init__(self, hidden=128, layers=2, residual=False, layer_norm=False,
                 architecture="mlp"):
        super().__init__()
        if architecture not in ("mlp", "action_aware"):
            raise ValueError(f"unknown architecture: {architecture}")
        self.hidden = hidden
        self.layers = layers
        self.residual = residual
        self.layer_norm = layer_norm
        self.architecture = architecture
        self.value_mode = "clip"
        self.value_scale = features.MARGIN_SCALE
        body = [nn.Linear(features.NUM_FEATURES, hidden)]
        if layer_norm:
            body.append(nn.LayerNorm(hidden))
        body.append(nn.ReLU())
        for _ in range(layers - 1):
            if residual:
                body.append(_ResidualBlock(hidden, layer_norm=layer_norm))
            else:
                body.append(nn.Linear(hidden, hidden))
                if layer_norm:
                    body.append(nn.LayerNorm(hidden))
                body.append(nn.ReLU())
        self.body = nn.Sequential(*body)
        if architecture == "action_aware":
            scorer_hidden = max(32, hidden // 2)
            self.policy_head = None
            self.action_scorer = nn.Sequential(
                nn.Linear(hidden + features.ACTION_FEATURES, scorer_hidden),
                nn.ReLU(),
                nn.Linear(scorer_hidden, 1),
            )
        else:
            self.policy_head = nn.Linear(hidden, features.NUM_ACTIONS)
            self.action_scorer = None
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, x):
        """Return (policy_logits (batch, 6), value (batch,))."""
        board = x[:, :features.NUM_FEATURES]
        h = self.body(board)
        if self.architecture == "action_aware":
            action = x[:, features.NUM_FEATURES:].reshape(
                x.shape[0], features.NUM_ACTIONS, features.ACTION_FEATURES)
            expanded = h.unsqueeze(1).expand(-1, features.NUM_ACTIONS, -1)
            policy_logits = self.action_scorer(
                torch.cat([expanded, action], dim=2)).squeeze(-1)
        else:
            policy_logits = self.policy_head(h)
        value = torch.tanh(self.value_head(h)).squeeze(-1)
        return policy_logits, value


def encode_batch(states, device, architecture="mlp"):
    """Encode states for the requested network architecture."""
    rows = [features.encode_for_model(s, architecture) for s in states]
    return torch.tensor(rows, dtype=torch.float32, device=device)


def save_net(net, path):
    """Save weights AND the network's shape, so it can be reloaded at any size."""
    torch.save({"state_dict": net.state_dict(), "hidden": net.hidden,
                "layers": net.layers, "residual": net.residual,
                "layer_norm": net.layer_norm,
                "architecture": getattr(net, "architecture", "mlp"),
                "value_mode": getattr(net, "value_mode", "clip"),
                "value_scale": getattr(net, "value_scale", features.MARGIN_SCALE)}, path)


def load_net(path, device="cpu"):
    """Load a network, reading its shape from the checkpoint. Falls back to the
    default 128x2 for legacy checkpoints that stored only a raw state_dict."""
    blob = torch.load(path, map_location=device)
    if isinstance(blob, dict) and "state_dict" in blob:
        net = MancalaNet(hidden=blob["hidden"], layers=blob["layers"],
                         residual=blob.get("residual", False),
                         layer_norm=blob.get("layer_norm", False),
                         architecture=blob.get("architecture", "mlp"))
        net.load_state_dict(blob["state_dict"])
        net.value_mode = blob.get("value_mode", "clip")
        net.value_scale = blob.get("value_scale", features.MARGIN_SCALE)
    else:
        net = MancalaNet()                         # legacy raw state_dict (128x2)
        net.load_state_dict(blob)
    return net.to(device).eval()


class PolicyBot:
    """Plays the network's policy head directly -- one forward pass, no search.

    The lightest possible agent. Because the policy head was trained to imitate
    the MCTS visit distribution, this measures how much of the search's
    conclusions the network internalized ("distilled") into a single forward
    pass. Strength here is strength you get for ~microseconds and a few KB.
    """

    name = "Policy"

    def __init__(self, net, device=None):
        self.net = net
        self.device = device or torch.device("cpu")

    def act(self, state, rng):
        legal = engine.legal_moves(state)
        with torch.no_grad():
            logits, _ = self.net(encode_batch(
                [state], self.device,
                getattr(self.net, "architecture", "mlp")))
        logits = logits[0]
        best = max(float(logits[a]) for a in legal)          # highest-scoring legal move
        return rng.choice([a for a in legal if float(logits[a]) == best])
