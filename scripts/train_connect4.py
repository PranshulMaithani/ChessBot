"""Stage 0.5: train AlphaZero on Connect 4.

Connect 4 sits between Tic-Tac-Toe (trivial) and chess (heavy). Critically,
its games are almost always decisive, so the value head gets real training
signal from iteration 1 -- which is exactly what chess from cold start
struggles with on a laptop.

    python scripts/train_connect4.py            # full run (a few hours)
    python scripts/train_connect4.py --smoke    # quick smoke test
    python scripts/train_connect4.py --resume   # continue from latest.pt
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config  # noqa: E402
from src.core.arena import play_match, random_player, raw_policy_player  # noqa: E402
from src.core.coach import Coach  # noqa: E402
from src.games import Connect4  # noqa: E402
from src.nn.net import NeuralNet  # noqa: E402

METRICS_CSV = "runs/connect4_metrics.csv"


def connect4_config(args) -> Config:
    cfg = Config(
        num_channels=32,
        num_res_blocks=4,
        num_sims=50,
        c_puct=1.5,
        dirichlet_alpha=1.0,
        dirichlet_eps=0.25,
        num_iters=60,
        selfplay_games=25,
        temp_threshold=8,
        max_game_len=42,         # board is 6x7, max possible plies
        epochs=8,
        batch_size=128,
        lr=1e-3,
        weight_decay=1e-4,
        replay_window=15,
        arena_games=20,
        update_threshold=0.55,
        warmup_iters=2,
        checkpoint_dir="models/connect4",
    )
    if args.smoke:
        cfg.num_iters, cfg.selfplay_games, cfg.num_sims = 1, 3, 10
        cfg.epochs, cfg.arena_games, cfg.warmup_iters = 1, 2, 0
        cfg.num_channels, cfg.num_res_blocks = 16, 2
    for attr, val in [
        ("num_iters", args.iters), ("selfplay_games", args.games),
        ("num_sims", args.sims), ("arena_games", args.arena),
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
    ap.add_argument("--arena", type=int)
    ap.add_argument("--eval-games", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cfg = connect4_config(args)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.benchmark = True

    game = Connect4()
    net = NeuralNet(game, cfg)
    print(f"device={net.device}  net={cfg.num_channels}ch x{cfg.num_res_blocks}  "
          f"sims={cfg.num_sims}  games/iter={cfg.selfplay_games}  "
          f"warmup={cfg.warmup_iters}  iters={cfg.num_iters}")

    if args.resume and os.path.exists(os.path.join(cfg.checkpoint_dir, "latest.pt")):
        net.load_checkpoint("latest.pt")
        print(f"resumed from {cfg.checkpoint_dir}/latest.pt")

    rnd = random_player(game)

    def eval_hook(_it, current_net):
        # Connect 4 has decisive games, so vs-random win rate is a *real* signal:
        # a learning bot should reach near-100% wins quickly.
        me = raw_policy_player(game, current_net)
        w, l, d = play_match(game, me, rnd, args.eval_games, max_moves=cfg.max_game_len)
        return {"vs_random[W/L/D]": (w, l, d)}

    coach = Coach(game, net, cfg, eval_hook=eval_hook, metrics_csv=METRICS_CSV)
    try:
        coach.learn()
    except KeyboardInterrupt:
        print(f"\ninterrupted -- latest weights are in {cfg.checkpoint_dir}/latest.pt")


if __name__ == "__main__":
    main()
