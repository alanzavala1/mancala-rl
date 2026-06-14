"""Verify the C policy inference matches PyTorch, and benchmark its latency.

    .venv\\Scripts\\python scripts/bench_cpolicy.py --champion runs_final/best.pt
"""

import argparse
import pathlib
import random
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch

from mancala_rl import engine, cpolicy
from mancala_rl.features import encode
from mancala_rl.network import MancalaNet


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--champion", default="runs_final/best.pt")
    args = p.parse_args()

    net = MancalaNet()
    net.load_state_dict(torch.load(args.champion, map_location="cpu"))
    net.eval()

    def py_logits(state):
        with torch.no_grad():
            out, _ = net(torch.tensor([encode(state)], dtype=torch.float32))
        return out[0].tolist()

    # --- correctness: C logits/move must match PyTorch across many positions ---
    rng = random.Random(0)
    max_diff, argmax_mismatch, positions = 0.0, 0, 0
    for _ in range(200):
        state = engine.reset(first_player=rng.choice([1, 2]))
        for _ in range(60):
            legal = engine.legal_moves(state)
            if not legal:
                break
            pl, cl = py_logits(state), cpolicy.logits(state)
            max_diff = max(max_diff, max(abs(pl[i] - cl[i]) for i in range(6)))
            if max(legal, key=lambda a: pl[a]) != max(legal, key=lambda a: cl[a]):
                argmax_mismatch += 1
            positions += 1
            state, _, done, _ = engine.step(state, max(legal, key=lambda a: pl[a]))
            if done:
                break

    print(f"positions checked        : {positions}")
    print(f"max |logit| diff vs torch: {max_diff:.2e}")
    print(f"argmax (move) mismatches : {argmax_mismatch}")

    # --- latency ---
    state = engine.reset()
    print(f"\npure-C latency           : {cpolicy.bench_us(state):.3f} us/move")


if __name__ == "__main__":
    main()
