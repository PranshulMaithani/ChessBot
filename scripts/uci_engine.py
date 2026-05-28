"""UCI adapter for the Chesseng bot.

Speaks the Universal Chess Interface over stdin/stdout, so the trained net +
MCTS can be plugged into any chess GUI (Arena, Cute Chess, Banksia) or — the
intended use — into `lichess-bot` to play on Lichess.

Manual test (from the project root):
    python scripts/uci_engine.py
    > uci
    > isready
    > position startpos moves e2e4
    > go movetime 2000
    < bestmove ....
    > quit
"""
from __future__ import annotations

import argparse
import os
import sys

import chess
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.setrecursionlimit(100_000)  # MCTS recurses up to game length

from src.config import Config  # noqa: E402
from src.core.mcts_factory import make_mcts  # noqa: E402
from src.games import ChessGame  # noqa: E402
from src.nn.net import NeuralNet  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="models/chess/best.pt")
    ap.add_argument("--sims", type=int, default=400,
                    help="MCTS simulations per move (more = stronger + slower)")
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--res-blocks", type=int, default=6)
    ap.add_argument("--mcts-batch", type=int, default=32,
                    help="parallel sims per net call (>1 enables BatchedMCTS)")
    ap.add_argument("--cpu", action="store_true")
    return ap.parse_args()


def choose_move(game, cfg, net, board):
    """Run MCTS for the side to move and return a real-board chess.Move.

    The net always reasons in white-to-move canonical form. If Black is on
    move, MCTS returns a move on the mirrored board — we mirror the squares
    back so the move is legal on the real position.
    """
    player = 1 if board.turn == chess.WHITE else -1
    canon = game.get_canonical_form(board, player)
    mcts = make_mcts(game, net, cfg)
    probs = mcts.get_action_prob(canon, temp=0, add_root_noise=False)
    move = game._action_to_move(canon, int(np.argmax(probs)))
    if board.turn == chess.BLACK:
        move = chess.Move(
            chess.square_mirror(move.from_square),
            chess.square_mirror(move.to_square),
            promotion=move.promotion,
        )
    return move


def _emit(line: str):
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def _log(msg: str):
    # UCI side-channel: GUIs render "info string ..." in their log pane.
    _emit(f"info string {msg}")


def _apply_position_command(line: str) -> chess.Board:
    """Build the board described by a `position ...` UCI command."""
    tokens = line.split()
    if "startpos" in tokens:
        board = chess.Board()
        idx = tokens.index("startpos") + 1
    elif "fen" in tokens:
        fen_start = tokens.index("fen") + 1
        # A FEN has 6 space-separated fields.
        fen = " ".join(tokens[fen_start:fen_start + 6])
        board = chess.Board(fen)
        idx = fen_start + 6
    else:
        return chess.Board()
    if idx < len(tokens) and tokens[idx] == "moves":
        for uci in tokens[idx + 1:]:
            try:
                board.push_uci(uci)
            except ValueError:
                _log(f"ignored illegal move {uci}")
                break
    return board


def main():
    args = parse_args()

    # Ampere+ free speed: TF32 matmul, cudnn autotune.
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    ckpt = args.checkpoint
    if not os.path.exists(ckpt):
        alt = os.path.join(os.path.dirname(ckpt) or "models", "latest.pt")
        if os.path.exists(alt):
            ckpt = alt
        else:
            sys.stderr.write(f"no checkpoint at {args.checkpoint}\n")
            sys.exit(1)

    cfg = Config(
        num_channels=args.channels,
        num_res_blocks=args.res_blocks,
        num_sims=args.sims,
        mcts_batch_size=args.mcts_batch,
        device="cpu" if args.cpu else "cuda",
        checkpoint_dir=os.path.dirname(ckpt) or "models",
    )
    game = ChessGame()
    net = NeuralNet(game, cfg)
    net.load_checkpoint(os.path.basename(ckpt))
    _log(f"loaded {ckpt} on {net.device}, sims={cfg.num_sims}, batch={cfg.mcts_batch_size}")

    board = chess.Board()

    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        if line == "uci":
            _emit("id name Chesseng")
            _emit("id author chesseng")
            # Expose sims as a tunable option; GUIs / lichess-bot can override it.
            _emit(f"option name Sims type spin default {cfg.num_sims} min 8 max 4000")
            _emit("uciok")
        elif line == "isready":
            _emit("readyok")
        elif line.startswith("setoption"):
            parts = line.split()
            if "Sims" in parts and "value" in parts:
                try:
                    cfg.num_sims = int(parts[parts.index("value") + 1])
                    _log(f"sims set to {cfg.num_sims}")
                except (ValueError, IndexError):
                    pass
        elif line == "ucinewgame":
            board = chess.Board()
        elif line.startswith("position"):
            board = _apply_position_command(line)
        elif line.startswith("go"):
            move = choose_move(game, cfg, net, board)
            _emit(f"bestmove {move.uci()}")
        elif line == "stop":
            # Search is synchronous — nothing to interrupt. Real engines would
            # cut MCTS short here; ours just finishes the current `go`.
            pass
        elif line == "quit":
            return


if __name__ == "__main__":
    main()
