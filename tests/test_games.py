"""Sanity checks for Game implementations.

Runnable two ways:
    python tests/test_games.py        # plain script, prints a summary
    pytest tests/test_games.py        # if pytest is installed
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.games import TicTacToe  # noqa: E402
from src.games.base import Game  # noqa: E402


def random_playout(game: Game, seed: int):
    """Play a uniformly random legal game; return (result, last_player)."""
    rng = np.random.default_rng(seed)
    board = game.get_init_state()
    player = 1
    for _ in range(1000):  # generous safety cap
        result = game.get_game_ended(board, player)
        if result != 0:
            return result, player
        valid = game.get_valid_moves(board, player)
        actions = np.flatnonzero(valid)
        assert actions.size > 0, "no legal moves but game not ended"
        action = int(rng.choice(actions))
        board, player = game.get_next_state(board, player, action)
    raise AssertionError("game did not terminate within cap")


def test_tictactoe_all_games_terminate():
    game = TicTacToe()
    results = []
    for seed in range(500):
        result, _ = random_playout(game, seed)
        assert result != 0, "TTT game returned 'ongoing' as a result"
        results.append(result)
    # We should observe wins (|1.0|) and draws (1e-4) across many random games.
    assert any(abs(r) == 1.0 for r in results), "expected some decisive games"
    assert any(r == 1e-4 for r in results), "expected some drawn games"


def test_tictactoe_canonical_symmetry():
    game = TicTacToe()
    board = game.get_init_state()
    board, _ = game.get_next_state(board, 1, 0)   # +1 plays corner
    board, _ = game.get_next_state(board, -1, 4)  # -1 plays center
    # From -1's perspective, their own stones must read as +1.
    canon = game.get_canonical_form(board, -1)
    assert canon[1, 1] == 1, "side-to-move stones should be +1 in canonical form"
    assert canon[0, 0] == -1, "opponent stones should be -1 in canonical form"


def test_tictactoe_symmetries_consistent():
    game = TicTacToe()
    board = game.get_init_state()
    board, _ = game.get_next_state(board, 1, 0)
    pi = np.zeros(game.get_action_size(), dtype=np.float32)
    pi[1] = 1.0  # arbitrary "policy" putting all mass on action 1
    syms = game.get_symmetries(board, pi)
    assert len(syms) == 8, "3x3 board has 8 symmetries"
    for b, p in syms:
        assert b.shape == (3, 3)
        assert np.isclose(p.sum(), 1.0), "symmetry must preserve total policy mass"


def test_encode_shape():
    game = TicTacToe()
    canon = game.get_canonical_form(game.get_init_state(), 1)
    planes = game.encode(canon)
    assert planes.shape == (2, 3, 3)
    assert planes.dtype == np.float32


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
