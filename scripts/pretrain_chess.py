"""Supervised pretraining of the chess net on a PGN database.

The chess "cold start" problem on a laptop is that self-play with an untrained
net produces mostly drawn shuffles, so the value head never sees decisive
games. This script solves it: read a PGN of strong human games, train the
SAME net architecture with policy cross-entropy (over the played move) and
value MSE (over the game outcome). The resulting checkpoint plugs directly
into self-play via ``train_chess.py --init-from``.

    # 1) get a PGN — see docs/DATA.md (default: Lichess Elite)
    python scripts/pretrain_chess.py --pgn data/lichess_elite_2024-10.pgn

    # 2) bootstrap self-play from the pretrained net
    python scripts/train_chess.py --init-from models/chess/pretrained.pt
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config  # noqa: E402
from src.data.pgn import collect_examples  # noqa: E402
from src.games import ChessGame  # noqa: E402
from src.nn.net import NeuralNet  # noqa: E402


def evaluate(net, planes, actions, values, idx, bs):
    net.model.eval()
    pi_loss = v_loss = top1 = 0.0
    n_b = 0
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            sel = idx[i:i + bs]
            x = torch.from_numpy(planes[sel]).float().to(net.device)
            ta = torch.from_numpy(actions[sel]).long().to(net.device)
            tv = torch.from_numpy(values[sel]).float().to(net.device)
            logits, v = net.model(x)
            pi_loss += F.cross_entropy(logits, ta).item()
            v_loss += F.mse_loss(v, tv).item()
            top1 += (logits.argmax(dim=1) == ta).float().mean().item()
            n_b += 1
    return pi_loss / max(n_b, 1), v_loss / max(n_b, 1), top1 / max(n_b, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgn", required=True, action="append",
                    help="path to a PGN file (can be repeated for multiple files)")
    ap.add_argument("--max-positions", type=int, default=500_000,
                    help="cap on training positions extracted from PGN")
    ap.add_argument("--min-elo", type=int, default=2000,
                    help="ignore games where either player is below this Elo")
    ap.add_argument("--skip-book", type=int, default=8,
                    help="drop the first N plies of each game (opening theory)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--channels", type=int, default=64,
                    help="must match the net you'll continue with in self-play")
    ap.add_argument("--res-blocks", type=int, default=6)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--out", default="models/chess/pretrained.pt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"=== reading {len(args.pgn)} PGN file(s); cap "
          f"{args.max_positions:,} positions, min Elo {args.min_elo}, "
          f"skipping {args.skip_book} opening plies ===")
    t0 = time.time()
    planes, actions, values = collect_examples(
        args.pgn,
        max_positions=args.max_positions,
        min_elo=args.min_elo,
        skip_book=args.skip_book,
    )
    print(f"collected {len(planes):,} positions in {time.time() - t0:.1f}s")
    print(f"value distribution:  "
          f"+1={int((values == 1).sum()):,}   "
          f"0={int((values == 0).sum()):,}   "
          f"-1={int((values == -1).sum()):,}")

    cfg = Config(
        num_channels=args.channels, num_res_blocks=args.res_blocks,
        device="cpu" if args.cpu else "cuda",
        checkpoint_dir=os.path.dirname(args.out) or "models/chess",
    )
    game = ChessGame()
    net = NeuralNet(game, cfg)
    torch.backends.cudnn.benchmark = True
    print(f"net: {args.channels}ch x{args.res_blocks}  device={net.device}")

    n = len(planes)
    perm = np.random.permutation(n)
    n_val = max(1, int(n * args.val_frac))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    print(f"split: train={len(train_idx):,}  val={len(val_idx):,}")

    optimizer = optim.Adam(
        net.model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    bs = args.batch_size

    print("\n=== training ===")
    for epoch in range(1, args.epochs + 1):
        net.model.train()
        np.random.shuffle(train_idx)
        t_epoch = time.time()
        pi_sum = v_sum = top1_sum = 0.0
        n_b = 0
        for i in range(0, len(train_idx), bs):
            sel = train_idx[i:i + bs]
            x = torch.from_numpy(planes[sel]).float().to(net.device)
            ta = torch.from_numpy(actions[sel]).long().to(net.device)
            tv = torch.from_numpy(values[sel]).float().to(net.device)

            logits, v = net.model(x)
            l_pi = F.cross_entropy(logits, ta)
            l_v = F.mse_loss(v, tv)
            loss = l_pi + l_v
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pi_sum += l_pi.item()
            v_sum += l_v.item()
            top1_sum += (logits.argmax(dim=1) == ta).float().mean().item()
            n_b += 1

        train_pi = pi_sum / n_b
        train_v = v_sum / n_b
        train_top1 = top1_sum / n_b
        val_pi, val_v, val_top1 = evaluate(net, planes, actions, values, val_idx, bs)
        print(f"epoch {epoch}/{args.epochs}  "
              f"train pi={train_pi:.3f} v={train_v:.3f} top1={train_top1:.3f}  |  "
              f"val pi={val_pi:.3f} v={val_v:.3f} top1={val_top1:.3f}  "
              f"({time.time() - t_epoch:.1f}s)")
        net.save_checkpoint(os.path.basename(args.out))

    print(f"\nsaved {args.out}")
    print("next:  python scripts/train_chess.py --init-from " + args.out)


if __name__ == "__main__":
    main()
