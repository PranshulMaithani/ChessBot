"""Stage 0: train the AlphaZero pipeline on Tic-Tac-Toe and prove it learns.

Run:
    python scripts/train_tictactoe.py            # full validation run
    python scripts/train_tictactoe.py --smoke    # tiny, seconds-long smoke test
    python scripts/train_tictactoe.py --iters 12 --games 30 --sims 30

Gate: a correctly-trained agent should NEVER lose — to a random mover, or to a
perfect (minimax) solver. Optimal Tic-Tac-Toe is a forced draw, so success is
"0 losses as both colors".
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config  # noqa: E402
from src.core.arena import greedy_mcts_player, play_match, random_player  # noqa: E402
from src.core.coach import Coach  # noqa: E402
from src.games import TicTacToe  # noqa: E402
from src.nn.net import NeuralNet  # noqa: E402


def perfect_ttt_player(game):
    """Exact minimax solver for Tic-Tac-Toe (memoized). Plays optimally."""
    memo = {}

    def minimax(board):  # canonical board, +1 to move
        key = game.string_key(board)
        if key in memo:
            return memo[key]
        result = game.get_game_ended(board, 1)
        if result != 0:
            memo[key] = (result, None)
            return memo[key]
        best_v, best_a = -2.0, None
        for a in np.flatnonzero(game.get_valid_moves(board, 1)):
            a = int(a)
            nb, npl = game.get_next_state(board, 1, a)
            child_v, _ = minimax(game.get_canonical_form(nb, npl))
            child_v = -child_v
            if child_v > best_v:
                best_v, best_a = child_v, a
        memo[key] = (best_v, best_a)
        return memo[key]

    return lambda canon: minimax(canon)[1]


def build_config(args):
    cfg = Config()
    if args.smoke:
        cfg.num_iters, cfg.selfplay_games, cfg.num_sims = 1, 4, 10
        cfg.epochs, cfg.arena_games = 2, 4
        args.eval_games = 4
    for attr, val in [
        ("num_iters", args.iters), ("selfplay_games", args.games),
        ("num_sims", args.sims), ("epochs", args.epochs), ("arena_games", args.arena),
    ]:
        if val is not None:
            setattr(cfg, attr, val)
    if args.cpu:
        cfg.device = "cpu"
    cfg.seed = args.seed
    cfg.checkpoint_dir = "models/tictactoe"
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int)
    ap.add_argument("--games", type=int)
    ap.add_argument("--sims", type=int)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--arena", type=int)
    ap.add_argument("--eval-games", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cfg = build_config(args)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    game = TicTacToe()
    net = NeuralNet(game, cfg)
    print(f"device={net.device}  sims={cfg.num_sims}  iters={cfg.num_iters}  "
          f"games/iter={cfg.selfplay_games}")

    rnd = random_player(game)
    perfect = perfect_ttt_player(game)

    def eval_hook(_it, current_net):
        me = greedy_mcts_player(game, current_net, cfg)
        rw, rl, rd = play_match(game, me, rnd, args.eval_games)
        pw, pl, pd = play_match(game, me, perfect, args.eval_games)
        return {"vs_random[W/L/D]": (rw, rl, rd), "vs_perfect[W/L/D]": (pw, pl, pd)}

    Coach(game, net, cfg, eval_hook=eval_hook).learn()

    print("\nStage 0 gate (want L=0 in both):")
    me = greedy_mcts_player(game, net, cfg)
    print(f"  vs random  W/L/D = {play_match(game, me, rnd, 40)}")
    print(f"  vs perfect W/L/D = {play_match(game, me, perfect, 40)}")


if __name__ == "__main__":
    main()
