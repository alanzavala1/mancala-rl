# Results

This folder contains the small CSV artifacts used for the headline tables in
the main README.

`main_tournament_ratings.csv` is the standard full round-robin field for the
foundational MLP agent.

`ablation_tournament_ratings.csv` adds no-network MCTS baselines at the same
800-simulation budget. This is the table used to check whether the learned
network contributes strength beyond search alone.

Large checkpoints and full run directories are intentionally ignored by git.

`action_aware_experiment_pairwise.csv` compares the foundational MLP agent with
the later action-aware experiment on the matchups that matter most: alpha-beta
search opponents and no-network MCTS ablations.

`action_aware_experiment_tradeoff.csv` records the cost of the action-aware
scale-up alongside the strength gains. This is the clearest summary of what was
learned from the heavier model without overstating the improvement.

`solver_regret_action_aware_bigpush_v1_s180.csv` is the solver-regret diagnostic
for the action-aware experiment checkpoint.
