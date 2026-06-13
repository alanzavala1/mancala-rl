"""Evaluation harness.

A match is decided by playing a fixed number of games and reporting agent A's
win rate with a 95% Wilson confidence interval. Seats are swapped every other
game (A is player 1 in even games, player 2 in odd games), so neither the
first-move advantage nor the side of the board can skew the result. Every
match is driven by a single seeded RNG, so results are reproducible.
"""

import math
import random
from dataclasses import dataclass

from . import engine


@dataclass
class MatchResult:
    name_a: str
    name_b: str
    n_games: int
    a_wins: int
    b_wins: int
    draws: int

    @property
    def win_rate(self):
        """A's wins as a fraction of all games (draws count as non-wins)."""
        return self.a_wins / self.n_games

    @property
    def score(self):
        """A's score with draws counted as half."""
        return (self.a_wins + 0.5 * self.draws) / self.n_games

    @property
    def ci(self):
        """95% Wilson interval for A's win rate."""
        return wilson_interval(self.a_wins, self.n_games)

    def line(self):
        lo, hi = self.ci
        return (
            f"{self.name_a} vs {self.name_b}: "
            f"win rate {self.win_rate:6.1%}  "
            f"95% CI [{lo:5.1%}, {hi:5.1%}]  "
            f"score {self.score:5.1%}  "
            f"(W {self.a_wins} / L {self.b_wins} / D {self.draws}, "
            f"n={self.n_games})"
        )


def wilson_interval(wins, n, z=1.96):
    """95% Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = wins / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (center - margin, center + margin)


def play_game(agent_p1, agent_p2, rng, max_turns=1000):
    """Play one game. agent_p1 is player 1, agent_p2 is player 2.

    Returns the winner: 1, 2, or 0 for a tie. max_turns is a safety guard; if
    it is ever hit the game is scored by the current store totals.
    """
    state = engine.reset(first_player=1)
    agents = {1: agent_p1, 2: agent_p2}
    for _ in range(max_turns):
        action = agents[state.current_player].act(state, rng)
        state, _, done, info = engine.step(state, action)
        if done:
            return info['winner']
    s1, s2 = engine.stores(state)
    return 1 if s1 > s2 else (2 if s2 > s1 else 0)


def evaluate(agent_a, agent_b, n_games=2000, seed=0):
    """Play n_games between agent_a and agent_b with swapped seats.

    Returns a MatchResult reporting things from agent_a's perspective.
    """
    rng = random.Random(seed)
    a_wins = b_wins = draws = 0
    for i in range(n_games):
        if i % 2 == 0:
            winner = play_game(agent_a, agent_b, rng)  # A is player 1
            a_is = 1
        else:
            winner = play_game(agent_b, agent_a, rng)  # A is player 2
            a_is = 2
        if winner == 0:
            draws += 1
        elif winner == a_is:
            a_wins += 1
        else:
            b_wins += 1
    return MatchResult(
        name_a=getattr(agent_a, "name", type(agent_a).__name__),
        name_b=getattr(agent_b, "name", type(agent_b).__name__),
        n_games=n_games,
        a_wins=a_wins,
        b_wins=b_wins,
        draws=draws,
    )
