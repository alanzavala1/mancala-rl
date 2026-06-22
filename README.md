# mancala-rl

This project rebuilds a failed school reinforcement learning project into a
working self-play agent for Kalah, the common capture version of Mancala. The
final agent uses a small neural network to guide Monte Carlo Tree Search. The
search looks ahead before each move, and the network is trained to predict the
stronger move choices and final score margins that search produces.

The goal was not only to make an agent that plays well. It was to make the
strength measurable. The repo includes a tested game engine, a C alpha-beta
solver used as a correctness check and benchmark opponent, tournament scripts,
ablation experiments that remove one component at a time, and saved result
tables under `results/`.

## The Game

Kalah is played with two rows of pits and one store for each player. This project
uses Kalah(6, 4), which means each player has six pits and every pit starts with
four seeds.

```
        G    H    I    J    K    L
    [ 4 ][ 4 ][ 4 ][ 4 ][ 4 ][ 4 ]   P2 store [ 0 ]
    [ 4 ][ 4 ][ 4 ][ 4 ][ 4 ][ 4 ]   P1 store [ 0 ]
        A    B    C    D    E    F
```

On a turn, a player chooses one of their pits, picks up all seeds in it, and
sows them one at a time around the board. The opponent's store is skipped. If
the last seed lands in the player's own store, that player moves again. If the
last seed lands in one of the player's own empty pits, that seed and the seeds
directly opposite are captured into the player's store.

This repo implements the empty-capture variant used by Irving, Donkers, and
Uiterwijk in their 2000 paper, "Solving Kalah." In this variant, the capture
rule still fires when the opposite pit is empty, so the player banks the last
seed they placed. That detail matters because this exact variant has published
game values. Kalah(6, 4) is known to be a first-player win by 10 seeds under
perfect play.

The C solver in `csolver/` uses alpha-beta search, a classic game-tree search
algorithm that prunes branches that cannot change the final decision. It
reproduces the published values for the smaller Kalah(6, n) openings that are
practical to verify locally. That gives the project a ground-truth check on the
rules, rather than relying on visual inspection of the game engine.

## Where The Project Started

The first version was a Deep SARSA agent. It trained, but it still could not
reliably beat a simple greedy opponent. The failure came from several design
problems.

The old agent updated from the board after its own move instead of from the next
position where it actually had to choose. It also used a per-move store
difference as the reward, which made the reward almost identical to the greedy
heuristic it was supposed to beat. It trained against one fixed opponent, so it
saw a narrow slice of the game. Finally, it had no replay buffer, no search, and
little protection against invalid or poorly normalized states.

The rebuild keeps the original idea of learning Mancala, but changes the method.
The agent now learns from complete self-play games. Each training example stores
the board position, the move distribution found by search, and the final score
margin from the current player's point of view.

## Final Agent

The final agent is the standard path in this repo:

```text
current network -> network-guided MCTS -> self-play games -> training data -> updated network
```

The network is a small feed-forward PyTorch model with two heads. The policy
head predicts which pit to choose. The value head predicts the expected final
score margin, scaled into the range [-1, 1]. The board encoding is always from
the current player's point of view, so the same six action slots always mean
"my pits" no matter which physical side is moving.

Monte Carlo Tree Search is the lookahead step. It uses the network's policy
head to decide which branches are promising, and it uses the value head to score
positions that search has not finished. This is different from classical MCTS,
which usually estimates positions with random rollouts.

The implementation uses PUCT, a selection rule that balances two pressures:
search moves the network already thinks are promising, and search moves that
have not been explored enough yet.

The search also includes MCTS-Solver logic. When search reaches a terminal
position, it can mark that branch as a proven win, loss, or draw. Those proofs
are propagated back up the tree, so the agent takes a proven win when one is
available and avoids a proven loss when another option exists.

There is also a Gumbel AlphaZero implementation in `mancala_rl/gumbel.py`. It is
tested and can be used from the training script, but it is not the final result
reported here. In this game it matched the plain PUCT search at similar budgets
instead of improving on it, so it is kept as an explored alternative.

## Results

The main result is the foundational AlphaZero-style agent: a compact MLP policy
and value network, PUCT search, MCTS-Solver endgame proof propagation, a
score-margin value target, and solver-backed evaluation. This is the cleanest
version of the project because it is strong, small, and easy to explain.

In a full round-robin tournament, the network-guided MCTS agent finished first
against random, greedy, raw-network, and alpha-beta search opponents. Every
match was seat-swapped and used randomized openings.

| Agent | Elo | Score as P1 | Score as P2 | Overall score |
|---|---:|---:|---:|---:|
| Network + MCTS, 800 simulations | 1808 | 82.4% | 80.2% | 81.3% |
| Alpha-beta depth 10 | 1780 | 81.0% | 76.2% | 78.6% |
| Alpha-beta depth 8 | 1679 | 70.5% | 65.2% | 67.9% |
| Alpha-beta depth 6 | 1611 | 66.7% | 53.8% | 60.2% |
| Alpha-beta depth 4 | 1544 | 58.3% | 47.1% | 52.7% |
| Raw network, no search | 1298 | 31.2% | 23.8% | 27.5% |
| Greedy one-ply heuristic | 1276 | 28.6% | 22.4% | 25.5% |
| Random | 1004 | 7.6% | 5.0% | 6.3% |

The ablation asks whether the learned network actually helps, or whether MCTS
is doing nearly all the work. At the same 800-simulation budget, network-guided
MCTS stayed well ahead of no-network MCTS baselines.

| Agent | Elo | Overall score |
|---|---:|---:|
| Network + MCTS, 800 simulations | 1805 | 81.5% |
| Classical MCTS, greedy rollouts, 800 simulations | 1617 | 60.8% |
| Classical MCTS, random rollouts, 800 simulations | 1558 | 53.9% |

As a final experiment, I also tried a heavier action-aware network. It keeps the
same self-play and MCTS setup, but gives the policy head exact features for each
legal move: store gain, extra-turn status, terminal status, after-move margin,
and remaining seeds. This was a useful test, but I do not frame it as the core
of the project. It is better understood as a scale-up experiment on top of the
foundational system.

| Cost or result | Foundational MLP | Action-aware experiment | Change |
|---|---:|---:|---:|
| Parameters | 19,335 | 96,770 | 5.0x larger |
| Checkpoint size | 80 KB | 391 KB | 4.9x larger |
| Training wall time | 0.79 hours | 3.27 hours | 4.1x longer |
| MCTS800 opening move latency | 0.069 sec | 0.127 sec | 1.8x slower |
| Average score vs alpha-beta depths 4, 6, 8, 10 | 69.1% | 75.6% | +6.5 points |
| Score vs alpha-beta depth 10 | 57.8% | 65.2% | +7.4 points |

The action-aware model improved the hardest matchups, especially depth-10
alpha-beta search, and reduced MCTS average solver regret from 2.75 to 2.17.
The tradeoff is that the gain was moderate compared with the extra training time
and larger model. That makes it a useful experiment to report alongside the main
agent, not the whole story of the project.

The full run directories are too large to commit, but the compact CSV summaries
are saved under `results/`. These numbers should be read as project-scale
benchmarks, not as a claim that the agent is optimal. Kalah(6, 4) is a
first-player-win game, so seat-swapped evaluation and randomized openings are
important.

![Training curve](assets/training_curve.png)

![Strength vs search](assets/strength_vs_search.png)

## What Mattered Most

The biggest early improvement was changing the value target from win/loss to
final score margin. In a first-player-win game, many good positions eventually
become wins, so a plain win/loss target stops giving much information. A
score-margin target still distinguishes a narrow win from a decisive win.

The action-aware model was a useful final experiment, but not the foundation of
the project. The foundation is the training and evaluation pipeline: self-play,
network-guided MCTS, MCTS-Solver, score-margin value learning, randomized
openings, seat-swapped tournaments, and solver-regret diagnostics. The heavier
model showed that adding game-structured action features can improve the hardest
matchups, but also showed that the extra complexity has diminishing returns.

The other important choice was measuring against stronger references. A greedy
bot is useful as a floor, but it is not enough to prove progress. The repo uses
alpha-beta search at multiple depths, no-network MCTS baselines, a raw-network
agent, randomized openings, seat-swapped matches, and exact-solver checks.

Several ideas were implemented and then kept in perspective. Gumbel AlphaZero is
tested, but it did not become the default. A residual network and alternative
value scaling were also tried, but they were not clear wins in this setting.
That is part of the final result: the project does not just show the strongest
agent, it shows how the design was tested and narrowed down.

## Code Layout

```text
mancala_rl/
  engine.py       immutable Kalah rules and legal move handling
  features.py     board encoding and score-margin value scaling
  network.py      policy and value neural network
  mcts.py         final PUCT MCTS agent with terminal proof propagation
  gumbel.py       tested experimental Gumbel AlphaZero search
  classical.py    no-network MCTS baselines for ablation
  bots.py         random and greedy baselines
  selfplay.py     self-play data generation, including multiprocessing
  training.py     supervised policy and value update
  evaluate.py     match evaluation and confidence intervals
  csolver.py      Python wrapper around the C solver
  reference.py    small brute-force minimax oracle for tests

csolver/          C alpha-beta solver and build script
scripts/          training, evaluation, plotting, and play scripts
tests/            correctness and regression tests
assets/           generated README figures
results/          small committed CSVs for headline results
```

## Setup

Install the Python package in editable mode:

```sh
pip install -e .
```

For running the test suite with `pytest`, install the development extra:

```sh
pip install -e ".[dev]"
```

Build the C solver before running scripts or tests that import `CSolverBot`:

```sh
powershell csolver/build.ps1
```

The built DLL is intentionally ignored by git. The C source is committed, so the
solver can be rebuilt locally.

## Common Commands

Train a new agent:

```sh
python scripts/train.py --iterations 400 --sims 256 --workers 8 --out runs
```

Run the action-aware scale-up experiment:

```sh
python scripts/train.py --iterations 1000 --games 36 --sims 384 --workers 12 --hidden 192 --layers 3 --architecture action-aware --phase-balanced-frac 0.20 --diverse-frac 0.20 --priority-frac 0.10 --random-opening 2 --random-opening-final 12 --gate-against-best --out runs_action_aware_bigpush_v1
```

Resume an interrupted run:

```sh
python scripts/train.py --resume runs_action_aware_bigpush_v1/checkpoint.pt --iterations 1000
```

Run the main tournament:

```sh
python scripts/tournament.py --champion runs/best.pt --schedule focused --games 100 --baseline-games 20
```

Run the ablation with no-network MCTS baselines:

```sh
python scripts/tournament.py --champion runs/best.pt --schedule focused --games 100 --baseline-games 20 --classical
```

Verify the solved values for smaller Kalah openings:

```sh
python scripts/verify_solution.py --max-n 4
```

Run tests:

```sh
pytest
```

Play against the trained agent:

```sh
python scripts/play.py
```

## References

Irving, G., Donkers, J., and Uiterwijk, J. (2000). "Solving Kalah." ICGA Journal
23(3), 139-147.

Silver, D., et al. (2017). "Mastering the game of Go without human knowledge."
Nature 550, 354-359.

Silver, D., et al. (2018). "A general reinforcement learning algorithm that
masters chess, shogi, and Go through self-play." Science 362, 1140-1144.

Winands, M., Bjornsson, Y., and Saito, J. (2008). "Monte-Carlo Tree Search
Solver." Computers and Games.

Danihelka, I., Guez, A., Schrittwieser, J., and Silver, D. (2022). "Policy
improvement by planning with Gumbel." ICLR.

Kocsis, L., and Szepesvari, C. (2006). "Bandit based Monte-Carlo Planning."
ECML.

Browne, C., et al. (2012). "A Survey of Monte Carlo Tree Search Methods." IEEE
Transactions on Computational Intelligence and AI in Games.
