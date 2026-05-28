"""Play Connect 4 against your trained bot.

    python scripts/play_connect4.py                 # you go first (X)
    python scripts/play_connect4.py --color second  # you go second (O)
    python scripts/play_connect4.py --sims 400      # bot thinks harder
    python scripts/play_connect4.py --checkpoint models/connect4/latest.pt
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config  # noqa: E402
from src.core.mcts import MCTS  # noqa: E402
from src.games.connect4 import COLS, ROWS, Connect4  # noqa: E402
from src.nn.net import NeuralNet  # noqa: E402


def render(board, human_first):
    you_sym = "X" if human_first else "O"
    bot_sym = "O" if human_first else "X"
    print()
    for r in range(ROWS):
        cells = []
        for v in board[r]:
            if v == 1:
                cells.append("X")
            elif v == -1:
                cells.append("O")
            else:
                cells.append(".")
        print(" " + " ".join(cells))
    print(" " + " ".join(str(c) for c in range(COLS)))
    print(f"  (you: {you_sym}, bot: {bot_sym})\n")


def bot_action(game, cfg, net, board, player):
    canon = game.get_canonical_form(board, player)
    mcts = MCTS(game, net, cfg)
    probs = mcts.get_action_prob(canon, temp=0, add_root_noise=False)
    return int(np.argmax(probs))  # column index — Connect 4 is symmetric so no translation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="models/connect4/best.pt")
    ap.add_argument("--color", choices=["first", "second"], default="first")
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--channels", type=int, default=32)
    ap.add_argument("--res-blocks", type=int, default=4)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--test", action="store_true", help="bot makes one move from start and exits")
    args = ap.parse_args()

    ckpt = args.checkpoint
    if not os.path.exists(ckpt):
        alt = os.path.join(os.path.dirname(ckpt) or "models", "latest.pt")
        if os.path.exists(alt):
            ckpt = alt
        else:
            raise SystemExit(f"no checkpoint found ({args.checkpoint}); train Connect 4 first")

    cfg = Config(
        num_channels=args.channels, num_res_blocks=args.res_blocks,
        num_sims=args.sims, device="cpu" if args.cpu else "cuda",
        checkpoint_dir=os.path.dirname(ckpt) or "models",
    )
    game = Connect4()
    net = NeuralNet(game, cfg)
    net.load_checkpoint(os.path.basename(ckpt))
    print(f"loaded {ckpt}  (sims={cfg.num_sims}, device={net.device})")

    human_first = args.color == "first"
    print("X always moves first.  You are " + ("X." if human_first else "O."))

    if args.test:
        # quick smoke from start position
        a = bot_action(game, cfg, net, game.get_init_state(), 1)
        print(f"bot suggests column {a}")
        return

    board = game.get_init_state()
    player = 1  # +1 = X = moves first

    while True:
        result = game.get_game_ended(board, player)
        if result != 0:
            render(board, human_first)
            if result < -0.5:
                winner = -player  # the side that just placed
                human_won = (winner == 1) == human_first
                print("you won! 🎉" if human_won else "bot wins.")
            else:
                print("draw.")
            return

        render(board, human_first)
        human_turn = (player == 1) == human_first

        if human_turn:
            text = input("your column (0-6, q to quit) > ").strip().lower()
            if text in ("q", "quit", "exit"):
                print("you quit.")
                return
            try:
                action = int(text)
            except ValueError:
                print("not a number, try again")
                continue
            if not (0 <= action < COLS) or not game.get_valid_moves(board, player)[action]:
                print("illegal column, try again")
                continue
        else:
            action = bot_action(game, cfg, net, board, player)
            print(f"bot plays column {action}")

        board, player = game.get_next_state(board, player, action)


if __name__ == "__main__":
    main()
