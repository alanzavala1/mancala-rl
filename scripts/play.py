"""Play Kalah against the trained agent, in the terminal.

    python scripts/play.py                       # you move first; agent searches 256 sims
    python scripts/play.py --seat 2 --sims 400   # agent moves first; stronger agent

Type the letter of a pit on your side to sow it ('q' quits). Sowing goes
counter-clockwise toward your store; landing your last seed in your own store
earns another turn, and landing in one of your own empty pits captures.
"""

import argparse
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch

from mancala_rl import engine
from mancala_rl.network import load_net, encode_batch
from mancala_rl.mcts import MCTSBot

LETTERS = {1: "ABCDEF", 2: "GHIJKL"}        # pit labels for each player's side


def render(state, human_seat):
    b = state.board
    top = "".join(f"[{b[7 + i]:>2} ]" for i in range(6))      # G..L (player 2)
    bot = "".join(f"[{b[i]:>2} ]" for i in range(6))          # A..F (player 1)
    who = lambda seat: "you " if seat == human_seat else "bot "
    return "\n".join([
        "",
        "      " + "".join(f"{c:^5}" for c in "GHIJKL"),
        "    " + top + f"   P2 ({who(2)}) store {b[13]:>2}",
        "    " + bot + f"   P1 ({who(1)}) store {b[6]:>2}",
        "      " + "".join(f"{c:^5}" for c in "ABCDEF"),
        "",
    ])


def agent_favor(net, device, state, agent_seat):
    """Value-head read of the position, from the agent's fixed perspective."""
    with torch.no_grad():
        _, v = net(encode_batch([state], device))
    v = float(v[0])
    return v if state.current_player == agent_seat else -v


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--champion", default="runs/best.pt")
    p.add_argument("--sims", type=int, default=256, help="agent MCTS simulations")
    p.add_argument("--seat", type=int, choices=(1, 2), default=1, help="your seat (1 = first)")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    net = load_net(args.champion, device)
    agent = MCTSBot(net, device, n_simulations=args.sims)
    human, bot = args.seat, (2 if args.seat == 1 else 1)
    rng = random.Random()

    print(f"\nYou are P1 (you move first)." if human == 1
          else "\nYou are P2 (the agent moves first).")
    print(f"Agent: {args.champion} at {args.sims} sims.")

    state = engine.reset(first_player=1)
    info = {"winner": None}
    while True:
        print(render(state, human))
        legal = engine.legal_moves(state)
        mover = state.current_player

        if mover == human:
            choices = "/".join(LETTERS[human][a] for a in legal)
            try:
                raw = input(f"Your move ({choices}, q=quit): ").strip().upper()
            except EOFError:
                raw = "Q"
            # take the first recognized pit letter (or Q) -- robust to stray bytes
            pick = next((c for c in raw if c in LETTERS[human] or c == "Q"), "")
            if pick == "Q":
                print("\nthanks for playing.")
                return
            if pick == "" or LETTERS[human].index(pick) not in legal:
                print("  -> not a legal move; try again.")
                continue
            action = LETTERS[human].index(pick)
        else:
            action = agent.act(state, rng)
            print(f"Agent plays {LETTERS[bot][action]}.")

        state, _, done, info = engine.step(state, action)

        if mover == bot and not done:
            fav = agent_favor(net, device, state, bot)
            mood = "ahead" if fav > 0.15 else ("behind" if fav < -0.15 else "even")
            print(f"   (agent's read: {fav:+.2f} -> it thinks it's {mood})")
        if not done and state.current_player == mover:
            print(f"   ({'you get' if mover == human else 'agent gets'} an extra turn)")
        if done:
            break

    print(render(state, human))
    s1, s2 = engine.stores(state)
    mine, theirs = (s1, s2) if human == 1 else (s2, s1)
    if info["winner"] == 0:
        print(f"Draw, {mine}-{theirs}.")
    elif info["winner"] == human:
        print(f"You win, {mine}-{theirs}. Nice.")
    else:
        print(f"Agent wins, {theirs}-{mine}.")


if __name__ == "__main__":
    main()
