# mancala-rl

A reinforcement learning agent for Mancala (capture variant), built from
scratch in stages. Every performance claim in this README comes from the
evaluation harness, not from intuition.

This is a rebuild of an old school project. The game engine is kept from that
project (its rules were correct); everything else is new. See
[What I learned](#what-i-learned) for the post-mortem on the first version.

## Status

**Stage 1 complete: engine, baseline bots, and evaluation harness. No learning
agent yet.** The numbers below are the reference lines a learning agent will be
measured against.

## The game

Capture Mancala: six pits and one store per player, four seeds per pit to
start. Sow counterclockwise, skipping the opponent's store. Landing your last
seed in your own store earns another turn. Landing it in one of your own empty
pits captures that seed plus everything in the opposite pit. The game ends when
a player's pits are all empty; the other player sweeps their remaining seeds.
Higher store wins.

## Layout

```
mancala_rl/
  engine.py      immutable game state; reset / legal_moves / step
  bots.py        RandomBot, GreedyBot (the baselines)
  evaluate.py    match harness: win rate + 95% CI, seat-swapped
scripts/
  run_baselines.py   prints the reference table below
tests/
  test_engine.py     correctness tests for the engine port
```

The engine API is side-effect free. `step(state, action)` returns
`(next_state, reward, done, info)` with extra turns and captures handled
internally, and a **sparse** reward (0 every move, then +1/-1/0 at the end,
from the moving player's perspective). Actions are pit indices 0-5 for the
side whose turn it is.

## How to run

From the repo root (standard library only, nothing to install for Stage 1):

```sh
python tests/test_engine.py            # engine correctness checks
python scripts/run_baselines.py        # the reference table
python scripts/run_baselines.py --games 5000 --seed 1   # more games / new seed
```

## Results

The harness plays a fixed number of games and reports the first agent's win
rate with a 95% Wilson confidence interval. Seats are swapped every other game,
so first-move advantage cannot skew the result. `score` counts draws as half a
win, which is the right number to read for the mirror matches.

`python scripts/run_baselines.py --games 2000 --seed 0`:

| Matchup            | Win rate | 95% CI          | Score | W / L / D       |
|--------------------|---------:|-----------------|------:|-----------------|
| Random vs Random   |   48.1%  | [45.9%, 50.3%]  | 51.1% | 962 / 917 / 121 |
| Greedy vs Greedy   |   45.0%  | [42.8%, 47.2%]  | 49.5% | 900 / 920 / 180 |
| Greedy vs Random   |   93.8%  | [92.6%, 94.7%]  | 94.6% | 1875 / 92 / 33  |

Reading the table:

- The two mirror matches sit at ~50% on score, which confirms the harness is
  fair: with identical players, swapping seats leaves neither side an edge.
- **Greedy beats Random ~94% of the time.** That is the reference line. A
  learning agent that cannot clear this has not learned anything a one-ply
  heuristic doesn't already know.

## What I learned

The first version of this project trained a Deep SARSA agent that never beat
its own greedy opponent. Four reasons, each fixed by construction in this
rebuild:

1. **Misaligned transitions.** It learned from the board right after its own
   move, before the opponent replied — but the state it actually faces next is
   the one *after* the opponent moves. (Stage 2+ records transitions aligned to
   the next state the acting player sees.)
2. **The reward was the greedy heuristic.** It got ±1 per move on the store
   differential, so the best it could do was imitate greedy. This rebuild uses
   a sparse win/loss reward instead, so the agent optimizes the actual game
   outcome.
3. **One fixed opponent.** It only ever saw a single deterministic bot, so it
   only ever saw a sliver of the state space.
4. **No normalization, no replay buffer, no target network.**

The remaining sections of this README — a training curve and a results table
for the learning agent vs Random and vs Greedy — will be filled in as those
stages land.
