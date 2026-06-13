"""Mancala RL: a from-scratch reinforcement learning agent for Mancala."""

from . import engine
from .bots import RandomBot, GreedyBot
from .evaluate import evaluate, MatchResult
from .solver import Solver, SolverBot, solve

__all__ = [
    "engine", "RandomBot", "GreedyBot", "evaluate", "MatchResult",
    "Solver", "SolverBot", "solve",
]
