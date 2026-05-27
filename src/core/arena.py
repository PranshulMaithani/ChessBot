"""Pit two players against each other to measure relative strength.

A "player" is a callable: ``player(canonical_board) -> action``. This keeps the
arena agnostic to whether a player is a net+MCTS, a random mover, or a perfect
solver.
"""
from __future__ import annotations

import numpy as np

from src.core.mcts import MCTS


def play_game(game, player_pos, player_neg):
    """Play one game; return the result from the +1 player's perspective."""
    players = {1: player_pos, -1: player_neg}
    board = game.get_init_state()
    player = 1
    while True:
        result = game.get_game_ended(board, player)
        if result != 0:
            return result if player == 1 else -result
        canon = game.get_canonical_form(board, player)
        action = players[player](canon)
        board, player = game.get_next_state(board, player, action)


def play_match(game, p1, p2, num_games):
    """Play ``num_games`` alternating colors. Return (p1_wins, p2_wins, draws)."""
    p1_wins = p2_wins = draws = 0
    for i in range(num_games):
        if i % 2 == 0:
            outcome = play_game(game, p1, p2)        # p1 is +1
        else:
            outcome = -play_game(game, p2, p1)       # p2 is +1, flip to p1's view
        if outcome > 0.5:
            p1_wins += 1
        elif outcome < -0.5:
            p2_wins += 1
        else:
            draws += 1
    return p1_wins, p2_wins, draws


def greedy_mcts_player(game, net, config):
    """A player that picks the most-visited move from a fresh MCTS each turn."""
    def play(canon):
        mcts = MCTS(game, net, config)
        probs = mcts.get_action_prob(canon, temp=0, add_root_noise=False)
        return int(np.argmax(probs))
    return play


def random_player(game):
    def play(canon):
        valid = game.get_valid_moves(canon, 1)
        return int(np.random.choice(np.flatnonzero(valid)))
    return play
