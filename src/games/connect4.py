"""Connect 4 — Stage 0.5 validation game.

A 6-row x 7-column grid. Players drop a piece in a column; it falls to the
lowest empty row. First to four in a row (horizontal, vertical, or either
diagonal) wins. Stage 0.5 in the roadmap because Connect 4 games are almost
always decisive, so the value head gets real signal — perfect for confirming
the AlphaZero loop *learns* at a non-trivial scale, before grinding chess.
"""
from __future__ import annotations

import numpy as np

from .base import Game

ROWS, COLS = 6, 7
_WIN = 4


class Connect4(Game):
    def get_init_state(self):
        return np.zeros((ROWS, COLS), dtype=np.int8)

    def get_board_shape(self):
        return (ROWS, COLS)

    def get_action_size(self):
        return COLS  # one action per column

    def get_next_state(self, board, player, action):
        col = int(action)
        empties = np.flatnonzero(board[:, col] == 0)
        if empties.size == 0:
            raise ValueError(f"illegal move: column {col} is full")
        row = empties.max()  # lowest empty row (row 0 = top)
        nxt = board.copy()
        nxt[row, col] = player
        return nxt, -player

    def get_valid_moves(self, board, player):
        return (board[0, :] == 0).astype(np.int8)  # column legal iff top cell empty

    def get_game_ended(self, board, player):
        # Scan every line of 4 cells; if its sum == +/-4, that player has won.
        for sign in (1, -1):
            target = sign * _WIN
            # horizontal
            for r in range(ROWS):
                for c in range(COLS - _WIN + 1):
                    if board[r, c:c + _WIN].sum() == target:
                        return 1.0 if sign == player else -1.0
            # vertical
            for c in range(COLS):
                for r in range(ROWS - _WIN + 1):
                    if board[r:r + _WIN, c].sum() == target:
                        return 1.0 if sign == player else -1.0
            # diagonal \
            for r in range(ROWS - _WIN + 1):
                for c in range(COLS - _WIN + 1):
                    if sum(board[r + i, c + i] for i in range(_WIN)) == target:
                        return 1.0 if sign == player else -1.0
            # diagonal /
            for r in range(_WIN - 1, ROWS):
                for c in range(COLS - _WIN + 1):
                    if sum(board[r - i, c + i] for i in range(_WIN)) == target:
                        return 1.0 if sign == player else -1.0
        if (board != 0).all():
            return 1e-4  # full board, no winner -> draw
        return 0.0

    def get_canonical_form(self, board, player):
        return (board * player).astype(np.int8)

    def encode(self, canonical_board):
        cur = (canonical_board == 1).astype(np.float32)
        opp = (canonical_board == -1).astype(np.float32)
        return np.stack([cur, opp], axis=0)

    def string_key(self, canonical_board):
        return canonical_board.astype(np.int8).tobytes().hex()

    def get_symmetries(self, canonical_board, pi):
        # The only Connect 4 symmetry is a left-right mirror.
        pi = np.asarray(pi)
        return [(canonical_board, pi),
                (np.fliplr(canonical_board), pi[::-1].copy())]
