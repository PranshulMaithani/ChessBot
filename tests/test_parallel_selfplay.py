"""Tests for parallel self-play.

multiprocessing on Windows uses 'spawn' which re-imports the test module;
the test functions live at module level (not __main__-gated) so this is fine.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config  # noqa: E402
from src.core.parallel_selfplay import parallel_selfplay  # noqa: E402
from src.games import TicTacToe  # noqa: E402
from src.nn.net import NeuralNet  # noqa: E402


def test_parallel_selfplay_tictactoe():
    """End-to-end: 2 workers, 4 games, must come back with examples whose
    shape matches the sequential format."""
    game = TicTacToe()
    cfg = Config(
        num_channels=8, num_res_blocks=1, num_sims=5,
        device="cpu", selfplay_worker_device="cpu",
        temp_threshold=3, max_game_len=0,
    )
    net = NeuralNet(game, cfg)
    examples = parallel_selfplay(game, net, cfg, num_games=4, num_workers=2)
    assert len(examples) > 0
    for board, pi, z in examples:
        assert board.shape == (3, 3)
        assert pi.shape == (9,)


def test_parallel_falls_back_when_single_worker():
    """num_workers=1 should still produce examples (uses the sequential path)."""
    game = TicTacToe()
    cfg = Config(num_channels=8, num_res_blocks=1, num_sims=5, device="cpu",
                 temp_threshold=3)
    net = NeuralNet(game, cfg)
    examples = parallel_selfplay(game, net, cfg, num_games=2, num_workers=1)
    assert len(examples) > 0


def _run_all():
    failed = 0
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
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
