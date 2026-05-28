"""Play a game against the trained chess bot.

    python scripts/play.py                 # you are White vs models/best.pt
    python scripts/play.py --color black    # you are Black
    python scripts/play.py --sims 200       # let the bot think harder
    python scripts/play.py --checkpoint models/latest.pt

Enter moves in SAN ("Nf3", "e4", "O-O", "e8=Q") or UCI ("g1f3").
Type "quit" to resign, "board" to reprint.
"""
from __future__ import annotations

import argparse
import os
import sys

import chess
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.setrecursionlimit(100_000)

from src.config import Config  # noqa: E402
from src.core.mcts import MCTS  # noqa: E402
from src.games import ChessGame  # noqa: E402
from src.nn.net import NeuralNet  # noqa: E402


def bot_move(game, mcts_cfg, net, board):
    """Return the bot's chosen real move for the current position."""
    player = 1 if board.turn == chess.WHITE else -1
    canon = game.get_canonical_form(board, player)
    mcts = MCTS(game, net, mcts_cfg)
    probs = mcts.get_action_prob(canon, temp=0, add_root_noise=False)
    move = game._action_to_move(canon, int(np.argmax(probs)))
    if board.turn == chess.WHITE:
        return move
    # bot is Black: translate the white-frame move back onto the real board
    return chess.Move(
        chess.square_mirror(move.from_square),
        chess.square_mirror(move.to_square),
        promotion=move.promotion,
    )


def parse_human(board, text):
    for parser in (board.parse_san, board.parse_uci):
        try:
            return parser(text)
        except ValueError:
            continue
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="models/chess/best.pt")
    ap.add_argument("--color", choices=["white", "black"], default="white")
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--res-blocks", type=int, default=6)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--test", action="store_true", help="make one bot move and exit")
    args = ap.parse_args()

    ckpt = args.checkpoint
    if not os.path.exists(ckpt):
        alt = os.path.join(os.path.dirname(ckpt) or "models", "latest.pt")
        if os.path.exists(alt):
            ckpt = alt
        else:
            raise SystemExit(f"no checkpoint found ({args.checkpoint}); train first")
    cfg = Config(num_channels=args.channels, num_res_blocks=args.res_blocks,
                 num_sims=args.sims, device="cpu" if args.cpu else "cuda",
                 checkpoint_dir=os.path.dirname(ckpt) or "models")
    game = ChessGame()
    net = NeuralNet(game, cfg)
    net.load_checkpoint(os.path.basename(ckpt))
    print(f"loaded {ckpt}  (sims={cfg.num_sims}, device={net.device})")

    board = chess.Board()
    if args.test:
        mv = bot_move(game, cfg, net, board)
        print(f"bot suggests: {board.san(mv)}")
        return

    human_white = args.color == "white"
    while not board.is_game_over(claim_draw=True):
        print("\n" + str(board))
        human_turn = board.turn == (chess.WHITE if human_white else chess.BLACK)
        if human_turn:
            text = input("your move> ").strip()
            if text == "quit":
                print("you resigned.")
                return
            if text == "board":
                continue
            move = parse_human(board, text)
            if move is None or move not in board.legal_moves:
                print("illegal/unparseable move, try again")
                continue
            board.push(move)
        else:
            move = bot_move(game, cfg, net, board)
            print(f"bot plays: {board.san(move)}")
            board.push(move)

    print("\n" + str(board))
    print("result:", board.result(claim_draw=True))


if __name__ == "__main__":
    main()
