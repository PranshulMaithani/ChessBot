"""Convert a PGN database into supervised training examples.

Each ply of each game becomes a triple ``(planes, action_idx, z)`` where
    * ``planes``: same 18-plane white-to-move canonical encoding the net
      sees during self-play (so the net architecture / checkpoints are
      identical between pretraining and self-play),
    * ``action_idx``: the played move encoded in the AlphaZero 4672-action
      space (mirrored when the side to move is black, to match canonical),
    * ``z``: the game's outcome from *the side to move's* perspective
      (+1 they win, -1 they lose, 0 draw).

That format is what :func:`scripts/pretrain_chess.py` needs: a one-hot CE
target for the policy head and a regression target for the value head.
"""
from __future__ import annotations

from typing import Iterator, List, Tuple

import chess
import chess.pgn
import numpy as np

from src.games.chess_game import ACTION_SIZE, ChessGame


def _outcome_from_result(result: str):
    """+1 if white won, -1 if black won, 0 for draw, None if unknown."""
    if result == "1-0":
        return 1
    if result == "0-1":
        return -1
    if result == "1/2-1/2":
        return 0
    return None


def _mirror_move(move: chess.Move) -> chess.Move:
    """Mirror a Move vertically (and color-wise) — pairs with board.mirror()
    so a black-to-move move on the absolute board lands as a white-to-move
    move on the canonical (mirrored) board."""
    return chess.Move(
        chess.square_mirror(move.from_square),
        chess.square_mirror(move.to_square),
        promotion=move.promotion,
    )


def _passes_filter(game, min_elo: int) -> bool:
    if min_elo <= 0:
        return True
    try:
        welo = int(game.headers.get("WhiteElo", 0))
        belo = int(game.headers.get("BlackElo", 0))
    except (ValueError, TypeError):
        return False
    return welo >= min_elo and belo >= min_elo


def iter_pgn_examples(
    pgn_paths,
    min_elo: int = 0,
    skip_book: int = 0,
    max_positions=None,
) -> Iterator[Tuple[np.ndarray, int, int]]:
    """Yield ``(planes float32 (18,8,8), action_idx int, z int)`` per ply.

    ``skip_book`` drops the first N plies of each game (they're memorised
    opening theory, not really "learned" play).
    """
    if isinstance(pgn_paths, str):
        pgn_paths = [pgn_paths]
    adapter = ChessGame()
    yielded = 0

    for pgn_path in pgn_paths:
        with open(pgn_path, encoding="utf-8", errors="ignore") as f:
            while True:
                if max_positions is not None and yielded >= max_positions:
                    return
                try:
                    game = chess.pgn.read_game(f)
                except (ValueError, UnicodeDecodeError):
                    continue
                if game is None:
                    break

                outcome = _outcome_from_result(game.headers.get("Result", "*"))
                if outcome is None:
                    continue  # skip unfinished / abandoned games
                if not _passes_filter(game, min_elo):
                    continue

                board = game.board()
                for ply, move in enumerate(game.mainline_moves()):
                    if ply < skip_book:
                        board.push(move)
                        continue

                    # canonical: always white to move; mirror the move too
                    if board.turn == chess.WHITE:
                        canon = board
                        canon_move = move
                    else:
                        canon = board.mirror()
                        canon_move = _mirror_move(move)

                    try:
                        action_idx = adapter.move_to_index(canon_move)
                    except (KeyError, ValueError):
                        board.push(move)
                        continue
                    if not (0 <= action_idx < ACTION_SIZE):
                        board.push(move)
                        continue

                    # value target from the side-to-move's view
                    z = outcome if board.turn == chess.WHITE else -outcome
                    planes = adapter.encode(canon)
                    yield planes, action_idx, int(z)
                    yielded += 1
                    board.push(move)

                    if max_positions is not None and yielded >= max_positions:
                        return


def collect_examples(
    pgn_paths,
    max_positions: int = 500_000,
    min_elo: int = 0,
    skip_book: int = 0,
    progress_every: int = 50_000,
):
    """In-memory collection of all examples as numpy arrays.

    Returns:
        planes  : (N, 18, 8, 8) float16
        actions : (N,) int32
        values  : (N,) int8
    """
    planes_list: List[np.ndarray] = []
    actions_list: List[int] = []
    values_list: List[int] = []
    for planes, action, z in iter_pgn_examples(
        pgn_paths,
        min_elo=min_elo,
        skip_book=skip_book,
        max_positions=max_positions,
    ):
        planes_list.append(planes.astype(np.float16))
        actions_list.append(action)
        values_list.append(z)
        n = len(planes_list)
        if progress_every and n % progress_every == 0:
            print(f"  collected {n:,} positions...", flush=True)
    if not planes_list:
        raise SystemExit(
            "no training examples extracted — check the PGN path and "
            "--min-elo / --max-positions filters"
        )
    return (
        np.stack(planes_list),
        np.array(actions_list, dtype=np.int32),
        np.array(values_list, dtype=np.int8),
    )
