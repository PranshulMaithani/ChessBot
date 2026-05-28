"""Sanity tests for the Connect 4 game adapter."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.games.connect4 import COLS, ROWS, Connect4  # noqa: E402


def random_play(game, seed):
    rng = np.random.default_rng(seed)
    board = game.get_init_state()
    player = 1
    for _ in range(ROWS * COLS + 1):
        r = game.get_game_ended(board, player)
        if r != 0:
            return r
        valid = np.flatnonzero(game.get_valid_moves(board, player))
        assert valid.size > 0, "no legal moves but game not ended"
        action = int(rng.choice(valid))
        board, player = game.get_next_state(board, player, action)
    raise AssertionError("game did not terminate within capacity")


def test_random_games_terminate():
    game = Connect4()
    for seed in range(400):
        random_play(game, seed)


def test_horizontal_win():
    game = Connect4()
    board = game.get_init_state()
    # +1 plays cols 0..3 in the bottom row; -1 stacks cols 5/6 harmlessly.
    plays = [(1, 0), (-1, 5), (1, 1), (-1, 6), (1, 2), (-1, 5), (1, 3)]
    for player, col in plays:
        board, _ = game.get_next_state(board, player, col)
    assert game.get_game_ended(board, 1) == 1.0
    assert game.get_game_ended(board, -1) == -1.0


def test_vertical_win():
    game = Connect4()
    board = game.get_init_state()
    for i in range(4):
        board, _ = game.get_next_state(board, 1, 0)
        if i < 3:
            board, _ = game.get_next_state(board, -1, 1)
    assert game.get_game_ended(board, 1) == 1.0


def test_diagonal_win():
    game = Connect4()
    board = game.get_init_state()
    board[2, 0] = board[3, 1] = board[4, 2] = board[5, 3] = 1
    assert game.get_game_ended(board, 1) == 1.0


def test_valid_moves_mask():
    game = Connect4()
    board = game.get_init_state()
    mask = game.get_valid_moves(board, 1)
    assert mask.sum() == COLS
    # fill column 3
    for _ in range(ROWS):
        board, _ = game.get_next_state(board, 1, 3)
    mask = game.get_valid_moves(board, 1)
    assert mask[3] == 0 and mask.sum() == COLS - 1


def test_canonical_form():
    game = Connect4()
    board = game.get_init_state()
    board[5, 3] = 1
    board[5, 4] = -1
    canon = game.get_canonical_form(board, -1)
    # From -1's perspective, their own stones must read as +1.
    assert canon[5, 4] == 1 and canon[5, 3] == -1


def test_encode_shape():
    game = Connect4()
    planes = game.encode(game.get_init_state())
    assert planes.shape == (2, ROWS, COLS)
    assert planes.dtype == np.float32


def test_symmetry_mirror():
    game = Connect4()
    board = game.get_init_state()
    board[5, 1] = 1
    pi = np.arange(COLS, dtype=np.float32) / COLS
    syms = game.get_symmetries(board, pi)
    assert len(syms) == 2
    b2, p2 = syms[1]
    assert b2[5, COLS - 1 - 1] == 1, "mirror should move stone from col 1 to col 5"
    assert np.allclose(p2, pi[::-1])


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
