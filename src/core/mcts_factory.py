"""Pick the right MCTS implementation based on config.

``mcts_batch_size > 1`` -> :class:`BatchedMCTS` (virtual loss + batched net eval).
otherwise              -> simple recursive :class:`MCTS` (one sim at a time).

The factory lets selfplay/arena/play scripts stay agnostic to the choice;
flip a single config field to switch.
"""
from __future__ import annotations


def make_mcts(game, net, config):
    bs = getattr(config, "mcts_batch_size", 1) or 1
    if bs > 1:
        from src.core.batched_mcts import BatchedMCTS
        return BatchedMCTS(game, net, config)
    from src.core.mcts import MCTS
    return MCTS(game, net, config)
