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

# Reverse-lookup dicts so move_to_index doesn't do O(n) `.index()` scans every
# call. ~138k move_to_index calls per game in the profile -> dict lookups
# turn this from ~600 ns/call into ~150 ns/call.
_KNIGHTS_IDX = {d: i for i, d in enumerate(_KNIGHTS)}
_DIRECTIONS_IDX = {d: i for i, d in enumerate(_DIRECTIONS)}
_UNDER_PIECES_IDX = {p: i for i, p in enumerate(_UNDER_PIECES)}

# move_to_index is a pure function of (from_sq, to_sq, promotion). At most
# 64*64*5 = 20480 distinct keys; in practice far fewer are ever queried.
# Caching the result turns a ~4 us computation into a ~150 ns dict lookup —
# the biggest remaining single-function speedup in the chess hot path.
_MOVE_INDEX_CACHE: dict = {}

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
        # Cache-keyed by the only fields that determine the index. Hot path:
        # one dict lookup. Cold path: original logic, then memoize.
        key = (move.from_square, move.to_square, move.promotion)
        cached = _MOVE_INDEX_CACHE.get(key)
        if cached is not None:
            return cached
        # Direct bit ops are ~2x faster than chess.square_file/rank function calls.
        from_sq, to_sq = move.from_square, move.to_square
        ff, fr = from_sq & 7, from_sq >> 3
        tf, tr = to_sq & 7, to_sq >> 3
        df, dr = tf - ff, tr - fr

        if move.promotion and move.promotion != chess.QUEEN:
            dir_idx = df + 1  # df in {-1,0,1} -> {0,1,2}
            piece_idx = _UNDER_PIECES_IDX[move.promotion]
            plane = _N_QUEEN + _N_KNIGHT + piece_idx * 3 + dir_idx
        elif (df, dr) in _KNIGHTS_IDX:
            plane = _N_QUEEN + _KNIGHTS_IDX[(df, dr)]
        else:  # queen-like move (covers queen promotions and castling)
            dist = max(abs(df), abs(dr))
            dir_idx = _DIRECTIONS_IDX[(_sign(df), _sign(dr))]
            plane = dir_idx * 7 + (dist - 1)
        result = from_sq * _PLANES_PER_SQUARE + plane
        _MOVE_INDEX_CACHE[key] = result
        return result

    def _action_to_move(self, board: chess.Board, action: int):
        # action = from_sq * 73 + plane, so from_sq is recoverable cheaply.
        # Filtering legal moves by from_sq before calling move_to_index skips
        # the ~75% of moves at other from-squares; cheap and big.
        target_from = action // _PLANES_PER_SQUARE
        for move in board.legal_moves:
            if move.from_square != target_from:
                continue
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
        # `claim_draw=True` triggers python-chess's expensive threefold-rep
        # scan over the move stack. Profiler showed it was 20% of self-play
        # CPU time. Needed for real game termination, so we keep it here.
        outcome = board.outcome(claim_draw=True)
        if outcome is None:
            return 0.0
        if outcome.winner is None:
            return 1e-4  # draw
        # A decisive result with this side to move means this side was mated -> loss.
        return 1.0 if outcome.winner == board.turn else -1.0

    def get_game_ended_no_claim(self, board, player):
        """Fast variant for use *inside* MCTS tree descent.

        Skips ``claim_draw=True`` (no threefold-rep scan). Inside the tree,
        boards are hypothetical and the per-descent `seen` set in
        :class:`BatchedMCTS._descend` already prevents infinite recursion via
        repetition, so we don't need python-chess's expensive draw-claim
        detection. The real selfplay / arena loops still call
        :meth:`get_game_ended` (with claim_draw) for actual game termination.
        """
        outcome = board.outcome(claim_draw=False)
        if outcome is None:
            return 0.0
        if outcome.winner is None:
            return 1e-4
        return 1.0 if outcome.winner == board.turn else -1.0

    # --- perspective / encoding ------------------------------------------------
    def get_canonical_form(self, board, player):
        return board if board.turn == chess.WHITE else board.mirror()

    def encode(self, canonical_board):
        # Was: Python loop over `board.piece_map().items()` doing per-square
        # numpy assignments (~150 us/call). Now: take python-chess's internal
        # bitboards and bit-unpack each one into an 8x8 plane (~50 us/call).
        # Bitboard bit `i` corresponds to chess square `i` = rank*8 + file, so
        # bytes 0..7 of the LE byte-string are ranks 0..7 — np.unpackbits with
        # bitorder='little' fills the plane in (rank, file) order.
        b = canonical_board
        planes = np.empty((_N_INPUT_PLANES, 8, 8), dtype=np.float32)
        white = b.occupied_co[chess.WHITE]
        black = b.occupied_co[chess.BLACK]
        bbs = (
            b.pawns & white,  b.knights & white, b.bishops & white,
            b.rooks & white,  b.queens & white,  b.kings & white,
            b.pawns & black,  b.knights & black, b.bishops & black,
            b.rooks & black,  b.queens & black,  b.kings & black,
        )
        for i, bb in enumerate(bbs):
            planes[i] = np.unpackbits(
                np.frombuffer(bb.to_bytes(8, "little"), dtype=np.uint8),
                bitorder="little",
            ).reshape(8, 8)
        planes[12].fill(float(b.has_kingside_castling_rights(chess.WHITE)))
        planes[13].fill(float(b.has_queenside_castling_rights(chess.WHITE)))
        planes[14].fill(float(b.has_kingside_castling_rights(chess.BLACK)))
        planes[15].fill(float(b.has_queenside_castling_rights(chess.BLACK)))
        planes[16].fill(0.0)
        if b.ep_square is not None:
            planes[16, b.ep_square >> 3, b.ep_square & 7] = 1.0
        planes[17].fill(b.halfmove_clock / 100.0)
        return planes

    def string_key(self, canonical_board):
        # Was `canonical_board.epd()` — that's a full FEN-style string build
        # (~150 us each, ~9k calls/game = 1.5 s of pure formatting). The MCTS
        # only needs a *hashable* key; a tuple of python-chess's internal
        # bitboards is functionally identical for transposition detection and
        # roughly 10x cheaper. Two positions that differ only in stale
        # castling-rights bits will get distinct keys here — that's a tiny
        # amount of duplicated search, not a correctness bug.
        b = canonical_board
        return (
            b.pawns, b.knights, b.bishops, b.rooks, b.queens, b.kings,
            b.occupied_co[chess.WHITE], b.occupied_co[chess.BLACK],
            b.turn, b.castling_rights, b.ep_square,
        )
