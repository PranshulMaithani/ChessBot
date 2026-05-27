"""Generate one self-play game and turn it into training examples.

Each move stores (canonical_board, player, search_policy). When the game ends
we backfill the outcome z (+1 win / -1 loss / ~0 draw) from each stored
position's point of view. Symmetries are expanded to multiply the data.
"""
from __future__ import annotations

import numpy as np

from src.core.mcts import MCTS


def execute_episode(game, net, config):
    mcts = MCTS(game, net, config)
    trajectory = []  # (canonical_board, player, pi)
    board = game.get_init_state()
    player = 1
    step = 0

    while True:
        step += 1
        canon = game.get_canonical_form(board, player)
        temp = 1.0 if step <= config.temp_threshold else 0.0
        pi = mcts.get_action_prob(canon, temp=temp, add_root_noise=True)

        for sym_board, sym_pi in game.get_symmetries(canon, pi):
            trajectory.append((sym_board, player, sym_pi))

        action = int(np.random.choice(len(pi), p=pi))
        board, player = game.get_next_state(board, player, action)

        result = game.get_game_ended(board, player)  # from current player's view
        if result != 0:
            # z is +result for positions where it was this player's move, else -result
            return [
                (b, pi_, result if pl == player else -result)
                for (b, pl, pi_) in trajectory
            ]
