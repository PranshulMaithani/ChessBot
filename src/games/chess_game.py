"""Chess as a Game for the AlphaZero pipeline, built on python-chess.

Key ideas
---------
canonical form
    The net should only ever reason "from the side-to-move's point of view".
    So the canonical board is **always white-to-move**: if it's black's turn we
    return ``board.mirror()`` (vertical flip + color swap, castling/ep adjusted).
    Because of this, every board the net/MCTS sees is white-to-move, and we only
    need to encode moves from white's perspective.

action encoding (AlphaZero's 8x8x73 = 4672 scheme)
    For each of the 64 from-squares, 73 planes describe the move:
        * 56 "queen" moves   = 8 directions x 7 distances (also covers
          queen-promotions and castling),
        * 8  knight moves,
        * 9  underpromotions = 3 file-directions x {knight, bishop, rook}.
    index = from_square * 73 + plane.

state
    The state object is a ``chess.Board``. ``player`` (+1 white / -1 black) is
    tracked by the generic loop and always matches ``board.turn``.
"""
from __future__ import annotations

import chess
import numpy as np

from .base import Game

# Queen-move directions: N, NE, E, SE, S, SW, W, NW  (as (file_delta, rank_delta) signs)
_DIRECTIONS = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
# Knight deltas (file, rank)
_KNIGHTS = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]
_UNDER_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK]

_N_QUEEN = 56          # 8 dirs * 7 distances
_N_KNIGHT = 8
_PLANES_PER_SQUARE = 73   # 56 + 8 + 9
ACTION_SIZE = 64 * _PLANES_PER_SQUARE  # 4672

# Encoded-board layout: 12 piece planes + 4 castling + 1 ep + 1 halfmove-clock
_N_INPUT_PLANES = 18


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


class ChessGame(Game):
    def get_init_state(self):
        return chess.Board()

    def get_board_shape(self):
        return (8, 8)

    def get_action_size(self):
        return ACTION_SIZE

    # --- move <-> action index -------------------------------------------------
    @staticmethod
    def move_to_index(move: chess.Move) -> int:
        ff, fr = chess.square_file(move.from_square), chess.square_rank(move.from_square)
        tf, tr = chess.square_file(move.to_square), chess.square_rank(move.to_square)
        df, dr = tf - ff, tr - fr

        if move.promotion and move.promotion != chess.QUEEN:
            dir_idx = df + 1  # df in {-1,0,1} -> {0,1,2}
            piece_idx = _UNDER_PIECES.index(move.promotion)
            plane = _N_QUEEN + _N_KNIGHT + piece_idx * 3 + dir_idx
        elif (df, dr) in _KNIGHTS:
            plane = _N_QUEEN + _KNIGHTS.index((df, dr))
        else:  # queen-like move (covers queen promotions and castling)
            dist = max(abs(df), abs(dr))
            dir_idx = _DIRECTIONS.index((_sign(df), _sign(dr)))
            plane = dir_idx * 7 + (dist - 1)
        return move.from_square * _PLANES_PER_SQUARE + plane

    def _action_to_move(self, board: chess.Board, action: int):
        # Match against legal moves (avg ~35) — no separate decoder needed.
        for move in board.legal_moves:
            if self.move_to_index(move) == action:
                return move
        raise ValueError(f"action {action} is not a legal move for this board")

    # --- dynamics --------------------------------------------------------------
    def get_next_state(self, board, player, action):
        # Canonicalize first so `action` is interpreted in the white-to-move frame
        # the policy produced it in; push on a copy so we never mutate the input.
        canon = self.get_canonical_form(board, player)
        nxt = canon.copy()
        nxt.push(self._action_to_move(canon, action))
        return nxt, -player

    def get_valid_moves(self, board, player):
        # Always called on a canonical (white-to-move) board with player=1.
        mask = np.zeros(ACTION_SIZE, dtype=np.int8)
        for move in board.legal_moves:
            mask[self.move_to_index(move)] = 1
        return mask

    def get_game_ended(self, board, player):
        outcome = board.outcome(claim_draw=True)
        if outcome is None:
            return 0.0
        if outcome.winner is None:
            return 1e-4  # draw
        # A decisive result with this side to move means this side was mated -> loss.
        return 1.0 if outcome.winner == board.turn else -1.0

    # --- perspective / encoding ------------------------------------------------
    def get_canonical_form(self, board, player):
        return board if board.turn == chess.WHITE else board.mirror()

    def encode(self, canonical_board):
        planes = np.zeros((_N_INPUT_PLANES, 8, 8), dtype=np.float32)
        b = canonical_board  # white to move: "our" = white (planes 0-5), "their" = black (6-11)
        for sq, piece in b.piece_map().items():
            r, f = chess.square_rank(sq), chess.square_file(sq)
            base = 0 if piece.color == chess.WHITE else 6
            planes[base + piece.piece_type - 1, r, f] = 1.0
        planes[12, :, :] = b.has_kingside_castling_rights(chess.WHITE)
        planes[13, :, :] = b.has_queenside_castling_rights(chess.WHITE)
        planes[14, :, :] = b.has_kingside_castling_rights(chess.BLACK)
        planes[15, :, :] = b.has_queenside_castling_rights(chess.BLACK)
        if b.ep_square is not None:
            planes[16, chess.square_rank(b.ep_square), chess.square_file(b.ep_square)] = 1.0
        planes[17, :, :] = b.halfmove_clock / 100.0
        return planes

    def string_key(self, canonical_board):
        # EPD = board + turn + castling + ep (no move counters) -> good transpositions.
        return canonical_board.epd()
