"""Quick cProfile of one self-play game to see where Python time actually goes.

Runs a short game with a small net (random init) on CPU so the GPU isn't a
factor. The output table sorts by cumulative time per function; the lines
near the top tell you what to optimize."""
from __future__ import annotations

import cProfile
import io
import os
import pstats
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.setrecursionlimit(100_000)

from src.config import Config  # noqa: E402
from src.core.selfplay import execute_episode  # noqa: E402
from src.games import ChessGame  # noqa: E402
from src.nn.net import NeuralNet  # noqa: E402


def main():
    cfg = Config(
        num_channels=96, num_res_blocks=8,
        num_sims=80, mcts_batch_size=16,
        max_game_len=30,        # cap game length for a quick profile
        device="cpu",           # CPU forward isolates Python overhead
        c_puct=1.5, dirichlet_alpha=0.3, dirichlet_eps=0.0,
        temp_threshold=15, resign_threshold=-1.0,
        checkpoint_dir="models/chess",
    )
    game = ChessGame()
    net = NeuralNet(game, cfg)  # random init is fine for profiling

    pr = cProfile.Profile()
    pr.enable()
    _ = execute_episode(game, net, cfg)
    pr.disable()

    buf = io.StringIO()
    pstats.Stats(pr, stream=buf).strip_dirs().sort_stats("cumulative").print_stats(25)
    print(buf.getvalue())

    buf2 = io.StringIO()
    pstats.Stats(pr, stream=buf2).strip_dirs().sort_stats("tottime").print_stats(25)
    print("\n=== by total time (excl. subcalls) ===\n")
    print(buf2.getvalue())


if __name__ == "__main__":
    main()
