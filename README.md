# mancala-rl

A from-scratch AlphaZero-style agent for Kalah (capture Mancala), trained by
self-play. Every number here comes from the evaluation harness, not intuition.

It's a rebuild of a failed school project — a Deep SARSA agent that never beat
its own greedy opponent. The game engine is kept; everything else is new. The
post-mortem is at the bottom.

## Result

In a round-robin Elo tournament against a field of classical opponents, the
agent (network + MCTS) finishes first — ahead of alpha-beta search to depth 10:

| Agent | Elo |
|---|---|
| **Network + MCTS (this agent)** | **1808** |
| alpha-beta depth-10 | 1780 |
| alpha-beta depth-8 | 1679 |
| alpha-beta depth-6 | 1611 |
| alpha-beta depth-4 | 1544 |
| raw network (no search) | 1298 |
| greedy (1-ply heuristic) | 1276 |
| random | 1004 |

Head-to-head it beats greedy 99%, depth-8 search 63%, and depth-10 52%
(`scripts/tournament.py`).

**Is the learning doing the work, or just the search?** Ablation: at the same
800-simulation budget, network+MCTS beats *no-network* MCTS (plain rollouts) by
~200 Elo (77-79% head-to-head). The learned network earns most of the strength —
the search alone is not enough.

![Training curve](assets/training_curve.png)
*Loss, and win rate vs the depth-8 solver, over training.*

![Strength vs search](assets/strength_vs_search.png)
*From the opening, strength and per-move latency vs MCTS budget — the efficiency frontier.*

## The game, and ground truth

Kalah(6,4): six pits and one store per side, four seeds per pit. Sow
counter-clockwise, skip the opponent's store. Last seed in your store → move
again. Last seed in your own empty pit → capture it and the opposite pit (even
if the opposite is empty — the *empty-capture* variant).

This exact variant is **solved**: Irving, Donkers & Uiterwijk (2000) proved
Kalah(6,4) is a first-player win by 10. Our own exact C solver reproduces that
on our code — (6,1)=+2, (6,2)=+10, (6,3)=+2, and the full (6,4) opening = +10
(`scripts/verify_solution.py`). So agent strength is measured against a known
truth, not a guess.

## How it works

Self-play, no human games. A small policy+value network guides an MCTS search;
the search returns a better move than the raw network; the network is then
trained to imitate the search's move distribution and to predict the game's
final score margin. Repeat. Strength comes from the network and search together:
alone, the network sits near greedy (Elo ~1300) and no-network MCTS reaches only
~depth-6.

## What moved the needle, and what didn't

The change that broke a long plateau: training the value head on the **final
score margin**, not a win/loss bit. A scored game gives a richer signal and
rewards safe, decisive conversion.

Measured non-improvements at this scale: Gumbel AlphaZero (tied plain MCTS at
equal budget), more self-play simulations (64 ≈ 384), and a bigger network
(strength saturates). The ceiling here is the network/game, not compute.

## Layout

```
mancala_rl/        the library
  engine.py        game rules; immutable state (reset / legal_moves / step)
  features.py      board encoding (perspective-canonical)
  network.py       policy + value MLP
  mcts.py          MCTS, with terminal-proof propagation (MCTS-Solver)
  gumbel.py        Gumbel AlphaZero search (explored alternative)
  classical.py     no-network MCTS (for the ablation)
  bots.py          random and 1-ply greedy baselines
  selfplay.py      self-play data generation
  training.py      one training step
  evaluate.py      match harness (win rate + 95% CI, seat-swapped)
  csolver.py       Python wrapper for the C solver
  reference.py     brute-force minimax (test oracle)
csolver/           the exact solver in C (alpha-beta + transposition table)
scripts/           entry points: train, tournament, play, verify_solution, ...
tests/             correctness tests
```

## Run it

```sh
pip install -r requirements.txt
powershell csolver/build.ps1                  # build the C solver (needs gcc)
python scripts/train.py --out runs            # train by self-play
python scripts/tournament.py --classical      # Elo tournament + the ablation
python scripts/play.py                         # play it yourself, in the terminal
python scripts/verify_solution.py --max-n 4   # confirm the solved game value
```

## What I learned

The first version (Deep SARSA) never beat greedy. Four causes, each fixed by
construction here:

1. **Misaligned transitions** — it learned from the board right after its own
   move, not the state it actually faces next.
2. **The reward was the greedy heuristic** — ±1 per move on the store
   differential, so the best it could do was imitate greedy. This rebuild
   rewards the real game outcome.
3. **One fixed opponent** — it saw a sliver of the state space; self-play covers
   far more.
4. **No replay buffer, normalization, or search.**

And one lesson learned later, the hard way: for a *scored* game, predict the
**score**, not just win/loss — and measure against ground truth (an Elo field,
the solved value), because the right metric is what finally made the picture
clear.
