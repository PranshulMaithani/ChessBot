"""Tests for the PGN -> training example pipeline."""
from __future__ import annotations

import inspect
import os
import pathlib
import sys
import tempfile

import chess
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.pgn import _mirror_move, iter_pgn_examples  # noqa: E402
from src.games.chess_game import ACTION_SIZE  # noqa: E402


SMALL_PGN = """[Event "g1"]
[White "a"]
[Black "b"]
[Result "1-0"]
[WhiteElo "2500"]
[BlackElo "2500"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 1-0


[Event "g2"]
[White "c"]
[Black "d"]
[Result "0-1"]

1. d4 d5 2. c4 c6 0-1


[Event "unfinished"]
[White "e"]
[Black "f"]
[Result "*"]

1. e4 *
"""


def test_mirror_move_round_trip():
    m = chess.Move.from_uci("e2e4")
    m2 = _mirror_move(m)
    assert m2.uci() == "e7e5"
    assert _mirror_move(m2) == m
    # promotion preserved
    promo = chess.Move.from_uci("e7e8q")
    assert _mirror_move(promo).uci() == "e2e1q"


def test_iter_pgn_examples_basic(tmp_path):
    pgn_path = tmp_path / "smoke.pgn"
    pgn_path.write_text(SMALL_PGN)
    examples = list(iter_pgn_examples(str(pgn_path)))
    # game 1 has 10 plies, game 2 has 4, unfinished game is skipped
    assert len(examples) == 14
    for planes, action, z in examples:
        assert planes.shape == (18, 8, 8)
        assert planes.dtype == np.float32
        assert 0 <= action < ACTION_SIZE
        assert z in (-1, 0, 1)


def test_value_target_perspective(tmp_path):
    """In a white-won game, white-to-move positions get z=+1, black-to-move
    get z=-1 (from the side-to-move's view)."""
    pgn_path = tmp_path / "pov.pgn"
    pgn_path.write_text(SMALL_PGN)
    examples = list(iter_pgn_examples(str(pgn_path)))
    # game 1 (white wins): plies 0..9. White on even, black on odd.
    for i in range(10):
        _, _, z = examples[i]
        if i % 2 == 0:
            assert z == 1, f"ply {i}: expected +1 (white's view, white won), got {z}"
        else:
            assert z == -1
    # game 2 (black wins): plies 10..13.
    for i in range(10, 14):
        _, _, z = examples[i]
        local_ply = i - 10
        if local_ply % 2 == 0:  # white to move in a game black wins -> -1
            assert z == -1
        else:
            assert z == 1


def test_min_elo_filter_drops_unrated(tmp_path):
    """game 2 has no Elo headers, so min_elo=2000 should drop it."""
    pgn_path = tmp_path / "elo.pgn"
    pgn_path.write_text(SMALL_PGN)
    examples = list(iter_pgn_examples(str(pgn_path), min_elo=2000))
    assert len(examples) == 10   # only game 1 survives


def test_skip_book_advances_state(tmp_path):
    """skip_book=4 must drop the first 4 plies of each game; the very first
    yielded position must therefore be 'after 1. e4 e5 2. Nf3 Nc6' (move 5)."""
    pgn_path = tmp_path / "book.pgn"
    pgn_path.write_text(SMALL_PGN)
    examples = list(iter_pgn_examples(str(pgn_path), skip_book=4))
    # game 1: 10 - 4 = 6 plies kept; game 2: 4 - 4 = 0 plies kept
    assert len(examples) == 6


def test_max_positions_caps(tmp_path):
    pgn_path = tmp_path / "cap.pgn"
    pgn_path.write_text(SMALL_PGN)
    examples = list(iter_pgn_examples(str(pgn_path), max_positions=5))
    assert len(examples) == 5


def _run_all():
    failed = 0
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        try:
            with tempfile.TemporaryDirectory() as td:
                kwargs = {}
                if "tmp_path" in inspect.signature(t).parameters:
                    kwargs["tmp_path"] = pathlib.Path(td)
                t(**kwargs)
            print(f"  PASS  {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
