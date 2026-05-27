"""Tic-Tac-Toe — the Stage 0 game used to validate the AlphaZero pipeline.

It is trivially small (9 cells, ~5478 reachable states), every game ends, and
optimal play is a forced draw. That makes it the perfect smoke test: if the
self-play loop is correct, the trained net + MCTS should quickly stop losing.
"""
from __future__ import annotations

import numpy as np

from .base import Game


class TicTacToe(Game):
    def __init__(self, n: int = 3):
        self.n = n

    def get_init_state(self) -> np.ndarray:
        return np.zeros((self.n, self.n), dtype=np.int8)

    def get_board_shape(self) -> tuple[int, int]:
        return (self.n, self.n)

    def get_action_size(self) -> int:
        return self.n * self.n

    def get_next_state(self, board, player, action):
        r, c = divmod(action, self.n)
        if board[r, c] != 0:
            raise ValueError(f"illegal move {action}: cell ({r},{c}) occupied")
        nxt = board.copy()
        nxt[r, c] = player
        return nxt, -player

    def get_valid_moves(self, board, player):
        return (board.reshape(-1) == 0).astype(np.int8)

    def get_game_ended(self, board, player):
        # A line of all +1 sums to +n; all -1 sums to -n.
        lines = [board[i, :].sum() for i in range(self.n)]
        lines += [board[:, j].sum() for j in range(self.n)]
        lines.append(np.trace(board))
        lines.append(np.trace(np.fliplr(board)))
        if any(s == self.n for s in lines):
            return 1.0 if player == 1 else -1.0
        if any(s == -self.n for s in lines):
            return 1.0 if player == -1 else -1.0
        if (board != 0).all():
            return 1e-4  # draw
        return 0.0

    def get_canonical_form(self, board, player):
        return (board * player).astype(np.int8)

    def encode(self, canonical_board):
        # Two planes: current player's stones, opponent's stones.
        cur = (canonical_board == 1).astype(np.float32)
        opp = (canonical_board == -1).astype(np.float32)
        return np.stack([cur, opp], axis=0)

    def string_key(self, canonical_board):
        return canonical_board.astype(np.int8).tobytes().hex()

    def get_symmetries(self, canonical_board, pi):
        # The square board has 8 symmetries (4 rotations x optional mirror).
        pi_grid = np.asarray(pi).reshape(self.n, self.n)
        out = []
        for k in range(4):
            for mirror in (False, True):
                b = np.rot90(canonical_board, k)
                p = np.rot90(pi_grid, k)
                if mirror:
                    b = np.fliplr(b)
                    p = np.fliplr(p)
                out.append((b, p.reshape(-1)))
        return out
