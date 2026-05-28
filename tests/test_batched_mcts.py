"""Tests for BatchedMCTS — equivalence on simple games and the repetition
regression that originally killed chess."""
from __future__ import annotations

import os
import sys

import chess
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config  # noqa: E402
from src.core.batched_mcts import BatchedMCTS  # noqa: E402
from src.core.mcts_factory import make_mcts  # noqa: E402
from src.games import ChessGame, TicTacToe  # noqa: E402
from src.games.chess_game import ACTION_SIZE as CHESS_A  # noqa: E402
from src.nn.net import NeuralNet  # noqa: E402


def test_batched_mcts_returns_valid_policy_on_tictactoe():
    game = TicTacToe()
    cfg = Config(num_channels=8, num_res_blocks=1, num_sims=40,
                 mcts_batch_size=8, device="cpu")
    net = NeuralNet(game, cfg)
    mcts = BatchedMCTS(game, net, cfg)
    probs = mcts.get_action_prob(game.get_init_state(), temp=0)
    assert probs.shape == (9,)
    assert np.isclose(probs.sum(), 1.0)


def test_batched_mcts_handles_chess_repetition():
    """Same regression case as test_mcts_terminates_on_repetition_cycles —
    BatchedMCTS must also survive King-vs-King at high sim count without
    hanging or stack-overflowing."""
    game = ChessGame()
    cfg = Config(num_channels=8, num_res_blocks=1, num_sims=200,
                 mcts_batch_size=16, device="cpu")
    net = NeuralNet(game, cfg)
    board = chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
    mcts = BatchedMCTS(game, net, cfg)
    probs = mcts.get_action_prob(board, temp=0)
    assert probs.shape == (CHESS_A,)
    assert probs.sum() > 0


def test_factory_picks_batched_when_configured():
    game = TicTacToe()
    cfg_simple = Config(num_channels=8, num_res_blocks=1, num_sims=10,
                        mcts_batch_size=1, device="cpu")
    cfg_batched = Config(num_channels=8, num_res_blocks=1, num_sims=10,
                         mcts_batch_size=8, device="cpu")
    net = NeuralNet(game, cfg_simple)
    from src.core.mcts import MCTS as SimpleMCTS
    assert type(make_mcts(game, net, cfg_simple)) is SimpleMCTS
    assert type(make_mcts(game, net, cfg_batched)) is BatchedMCTS


def test_batched_mcts_concentrates_on_legal_actions():
    """The returned policy must put zero mass on illegal actions."""
    game = TicTacToe()
    cfg = Config(num_channels=8, num_res_blocks=1, num_sims=40,
                 mcts_batch_size=8, device="cpu")
    net = NeuralNet(game, cfg)
    board = game.get_init_state()
    # Fill 4 cells so several actions are illegal.
    for p, a in [(1, 0), (-1, 1), (1, 2), (-1, 3)]:
        board, _ = game.get_next_state(board, p, a)
    canon = game.get_canonical_form(board, 1)
    mcts = BatchedMCTS(game, net, cfg)
    probs = mcts.get_action_prob(canon, temp=1.0)
    valid = game.get_valid_moves(canon, 1)
    # No mass on illegal actions.
    assert float((probs * (1 - valid)).sum()) == 0.0


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
