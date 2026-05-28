"""Correctness tests for the chess Game adapter.

These guard the two things most likely to silently corrupt training:
the 4672-move action encoding and the canonical (white-to-move) mirroring.

    python tests/test_chess.py     # or: pytest tests/test_chess.py
"""
from __future__ import annotations

import os
import sys

import chess
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.games.chess_game import ACTION_SIZE, ChessGame  # noqa: E402


def test_action_indices_unique_and_in_range():
    """Every legal move across many random positions maps to a distinct,
    in-range action index (a collision would merge two moves)."""
    game = ChessGame()
    rng = np.random.default_rng(0)
    for _ in range(300):
        board = chess.Board()
        for _ in range(int(rng.integers(0, 40))):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(moves[int(rng.integers(len(moves)))])
        if board.turn == chess.BLACK:
            board = board.mirror()  # canonical: white to move
        idxs = [game.move_to_index(m) for m in board.legal_moves]
        assert all(0 <= i < ACTION_SIZE for i in idxs)
        assert len(idxs) == len(set(idxs)), "two legal moves share an action index"


def test_valid_moves_mask_matches_legal_count():
    game = ChessGame()
    board = game.get_init_state()
    mask = game.get_valid_moves(board, 1)
    assert mask.sum() == board.legal_moves.count() == 20  # 20 opening moves


def test_action_roundtrip_via_next_state():
    """Picking each legal move by its index produces exactly that move."""
    game = ChessGame()
    board = game.get_init_state()
    for move in list(board.legal_moves):
        idx = game.move_to_index(move)
        nxt, nplayer = game.get_next_state(board, 1, idx)
        assert nplayer == -1
        assert nxt.move_stack[-1] == move


def test_canonical_is_always_white_to_move():
    game = ChessGame()
    board = game.get_init_state()
    board.push_san("e4")  # now black to move
    canon = game.get_canonical_form(board, -1)
    assert canon.turn == chess.WHITE
    # the mirrored position must have the same number of legal replies
    assert canon.legal_moves.count() == board.legal_moves.count()


def test_encode_shape_and_piece_counts():
    game = ChessGame()
    planes = game.encode(game.get_init_state())
    assert planes.shape == (18, 8, 8)
    assert planes.dtype == np.float32
    # 8 white pawns on plane 0, 8 black pawns on plane 6, 2 white rooks on plane 3
    assert planes[0].sum() == 8 and planes[6].sum() == 8
    assert planes[3].sum() == 2 and planes[9].sum() == 2


def test_underpromotion_distinct_from_queen_promotion():
    game = ChessGame()
    # White pawn on a7, black king out of the way: a7-a8 promotions.
    board = chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
    q = game.move_to_index(chess.Move.from_uci("a7a8q"))
    n = game.move_to_index(chess.Move.from_uci("a7a8n"))
    r = game.move_to_index(chess.Move.from_uci("a7a8r"))
    assert len({q, n, r}) == 3, "promotion variants must map to different indices"


def test_terminal_values():
    game = ChessGame()
    # Fool's mate: white is checkmated. Board.turn == WHITE (to move, but mated).
    fools = chess.Board()
    for mv in ["f3", "e5", "g4", "Qh4#"]:
        fools.push_san(mv)
    assert fools.is_checkmate()
    # side to move (white) is mated -> from white's perspective this is a loss
    assert game.get_game_ended(fools, 1) == -1.0
    # a fresh board is not over
    assert game.get_game_ended(game.get_init_state(), 1) == 0.0


def random_chess_game_terminates(game, seed, cap=300):
    rng = np.random.default_rng(seed)
    board = game.get_init_state()
    player = 1
    for step in range(cap):
        canon = game.get_canonical_form(board, player)
        result = game.get_game_ended(canon, 1)
        if result != 0:
            return True
        valid = np.flatnonzero(game.get_valid_moves(canon, 1))
        action = int(rng.choice(valid))
        board, player = game.get_next_state(board, player, action)
    return False  # hit the cap (legal, just a long game)


def test_selfplay_loop_consistency():
    """Drive the real generic move path (canonical -> action -> get_next_state)
    for several random games; ensures no illegal-move / mirroring errors."""
    game = ChessGame()
    for seed in range(10):
        # Should run without raising; most random games end well under the cap.
        random_chess_game_terminates(game, seed, cap=300)


def test_mcts_terminates_on_repetition_cycles():
    """Regression: King-vs-King has only shuffling moves and trivially produces
    repeated positions. The original recursive MCTS infinitely descended through
    fully-expanded position cycles and silently overflowed the C stack on the
    real chess training run. This call must return quickly."""
    from src.config import Config
    from src.core.mcts import MCTS
    from src.nn.net import NeuralNet
    game = ChessGame()
    cfg = Config(num_channels=8, num_res_blocks=1, num_sims=200, device="cpu")
    net = NeuralNet(game, cfg)
    board = chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1")  # bare kings only
    mcts = MCTS(game, net, cfg)
    probs = mcts.get_action_prob(board, temp=0)
    assert probs.shape == (ACTION_SIZE,)
    assert probs.sum() > 0


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
