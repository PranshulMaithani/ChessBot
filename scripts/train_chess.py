"""Stage 1: train the chess engine by self-play.

Designed to run for a long time (e.g. overnight) and be safe to interrupt:
  * weights are saved every iteration to models/latest.pt (and best.pt on a
    successful arena promotion),
  * per-iteration metrics are appended to runs/chess_metrics.csv,
  * Ctrl-C anytime; resume with --resume.

Examples
--------
    # quick end-to-end smoke test (seconds, tiny everything):
    python scripts/train_chess.py --smoke

    # overnight run (sensible laptop defaults; runs until you stop it):
    python scripts/train_chess.py

    # resume from the last checkpoint:
    python scripts/train_chess.py --resume

    # push strength (slower): more search + a bigger net
    python scripts/train_chess.py --sims 160 --channels 96 --res-blocks 8
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.setrecursionlimit(100_000)  # MCTS recurses up to game length

from src.config import Config  # noqa: E402
from src.core.arena import play_match, random_player, raw_policy_player  # noqa: E402
from src.core.coach import Coach  # noqa: E402
from src.games import ChessGame  # noqa: E402
from src.nn.net import NeuralNet  # noqa: E402

METRICS_CSV = "runs/chess_metrics.csv"


def chess_config(args) -> Config:
    """Overnight-friendly defaults tuned for a single 8GB laptop GPU."""
    cfg = Config(
        num_channels=64,
        num_res_blocks=6,
        num_sims=80,
        c_puct=1.5,
        dirichlet_alpha=0.3,
        dirichlet_eps=0.25,
        num_iters=1000,        # effectively "run until stopped"
        selfplay_games=16,
        temp_threshold=15,
        max_game_len=200,
        epochs=4,
        batch_size=256,
        lr=1e-3,
        weight_decay=1e-4,
        replay_window=8,
        arena_games=12,
        update_threshold=0.55,
    )
    if args.smoke:
        cfg.num_iters, cfg.selfplay_games, cfg.num_sims = 1, 2, 8
        cfg.epochs, cfg.arena_games, cfg.max_game_len = 1, 2, 30
        cfg.num_channels, cfg.num_res_blocks = 16, 2
    for attr, val in [
        ("num_iters", args.iters), ("selfplay_games", args.games),
        ("num_sims", args.sims), ("num_channels", args.channels),
        ("num_res_blocks", args.res_blocks), ("arena_games", args.arena),
    ]:
        if val is not None:
            setattr(cfg, attr, val)
    if args.cpu:
        cfg.device = "cpu"
    cfg.seed = args.seed
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int)
    ap.add_argument("--games", type=int)
    ap.add_argument("--sims", type=int)
    ap.add_argument("--channels", type=int)
    ap.add_argument("--res-blocks", type=int)
    ap.add_argument("--arena", type=int)
    ap.add_argument("--eval-games", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cfg = chess_config(args)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.benchmark = True

    game = ChessGame()
    net = NeuralNet(game, cfg)
    print(f"device={net.device}  net={cfg.num_channels}ch x{cfg.num_res_blocks}  "
          f"sims={cfg.num_sims}  games/iter={cfg.selfplay_games}  "
          f"action_size={game.get_action_size()}")

    if args.resume and os.path.exists(os.path.join(cfg.checkpoint_dir, "latest.pt")):
        net.load_checkpoint("latest.pt")
        print("resumed from models/latest.pt")

    rnd = random_player(game)

    def eval_hook(_it, current_net):
        # Cheap probe: raw-policy (no search) vs random. A learning net should
        # beat random quickly and decisively.
        me = raw_policy_player(game, current_net)
        w, l, d = play_match(game, me, rnd, args.eval_games)
        return {"vs_random[W/L/D]": (w, l, d)}

    coach = Coach(game, net, cfg, eval_hook=eval_hook, metrics_csv=METRICS_CSV)
    try:
        coach.learn()
    except KeyboardInterrupt:
        print("\ninterrupted — latest weights are in models/latest.pt")


if __name__ == "__main__":
    main()
