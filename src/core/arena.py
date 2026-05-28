"""Pit two players against each other to measure relative strength.

A "player" is a callable: ``player(canonical_board) -> action``. This keeps the
arena agnostic to whether a player is a net+MCTS, a random mover, or a perfect
solver.
"""
from __future__ import annotations

import numpy as np
from tqdm import tqdm

from src.core.mcts_factory import make_mcts


def play_game(game, player_pos, player_neg, max_moves=0):
    """Play one game; return the result from the +1 player's perspective.

    ``max_moves > 0`` adjudicates over-long games as draws (0.0). Critical for
    chess, where weak players can otherwise shuffle pieces forever.
    """
    players = {1: player_pos, -1: player_neg}
    board = game.get_init_state()
    player = 1
    step = 0
    while True:
        result = game.get_game_ended(board, player)
        if result != 0:
            return result if player == 1 else -result
        if max_moves and step >= max_moves:
            return 0.0  # adjudicated draw
        canon = game.get_canonical_form(board, player)
        action = players[player](canon)
        board, player = game.get_next_state(board, player, action)
        step += 1


def play_match(game, p1, p2, num_games, max_moves=0):
    """Play ``num_games`` alternating colors. Return (p1_wins, p2_wins, draws)."""
    p1_wins = p2_wins = draws = 0
    for i in tqdm(range(num_games), desc="Arena Match", unit="game"):
        if i % 2 == 0:
            outcome = play_game(game, p1, p2, max_moves=max_moves)        # p1 is +1
        else:
            outcome = -play_game(game, p2, p1, max_moves=max_moves)       # p2 is +1, flip to p1's view
        if outcome > 0.5:
            p1_wins += 1
        elif outcome < -0.5:
            p2_wins += 1
        else:
            draws += 1
    return p1_wins, p2_wins, draws


def greedy_mcts_player(game, net, config):
    """A player that picks the most-visited move from a fresh MCTS each turn.
    Uses :func:`make_mcts` so it picks up BatchedMCTS automatically when
    ``mcts_batch_size > 1``."""
    def play(canon):
        mcts = make_mcts(game, net, config)
        probs = mcts.get_action_prob(canon, temp=0, add_root_noise=False)
        return int(np.argmax(probs))
    return play


def raw_policy_player(game, net):
    """Fast probe player: one net eval per move, pick the best legal action
    from the raw policy with no search. Useful for cheap progress checks."""
    def play(canon):
        p, _ = net.predict(canon)
        valid = game.get_valid_moves(canon, 1)
        p = p * valid
        return int(np.argmax(p))
    return play


def random_player(game):
    def play(canon):
        valid = game.get_valid_moves(canon, 1)
        return int(np.random.choice(np.flatnonzero(valid)))
    return play
