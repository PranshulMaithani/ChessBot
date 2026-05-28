"""Generate one self-play game and turn it into training examples.

Each move stores (canonical_board, player, search_policy). When the game ends
we backfill the outcome z (+1 win / -1 loss / ~0 draw) from each stored
position's point of view. Symmetries are expanded to multiply the data.

Two failure modes that matter on chess from a cold start:
    * games that hit ``max_game_len`` are adjudicated as draws so they end,
    * if the net thinks the side to move is losing badly for N consecutive
      plies (``resign_threshold``), they resign — turns a value-uninformative
      "shuffle to the cap" into a real -1/+1 training signal. Disabled
      when ``resign_threshold`` <= -1.0.
"""
from __future__ import annotations

import numpy as np

from src.core.mcts_factory import make_mcts


def execute_episode(game, net, config):
    mcts = make_mcts(game, net, config)
    trajectory = []  # (canonical_board, player, pi)
    board = game.get_init_state()
    player = 1
    step = 0

    resign_threshold = getattr(config, "resign_threshold", -1.0)
    resign_n = getattr(config, "resign_n_consecutive", 4)
    resign_counter = 0

    while True:
        step += 1
        canon = game.get_canonical_form(board, player)
        temp = 1.0 if step <= config.temp_threshold else 0.0
        pi = mcts.get_action_prob(canon, temp=temp, add_root_noise=True)

        for sym_board, sym_pi in game.get_symmetries(canon, pi):
            trajectory.append((sym_board, player, sym_pi))

        # Resignation: net's value < threshold for N consecutive plies means
        # the side to move is convinced it's losing badly. End the game now and
        # backfill -1 for the resigning side. (Self-disabling early in training
        # because the value head starts near 0 and won't trip the threshold.)
        if resign_threshold > -1.0:
            _, value = net.predict(canon)
            if value < resign_threshold:
                resign_counter += 1
                if resign_counter >= resign_n:
                    return [
                        (b, pi_, -1.0 if pl == player else 1.0)
                        for (b, pl, pi_) in trajectory
                    ]
            else:
                resign_counter = 0

        action = int(np.random.choice(len(pi), p=pi))
        board, player = game.get_next_state(board, player, action)

        result = game.get_game_ended(board, player)  # from current player's view
        if result == 0 and config.max_game_len and step >= config.max_game_len:
            result = 1e-4  # adjudicate over-long game as a draw
        if result != 0:
            return [
                (b, pi_, result if pl == player else -result)
                for (b, pl, pi_) in trajectory
            ]
