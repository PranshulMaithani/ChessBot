"""The orchestrator: the AlphaZero outer loop.

For each iteration:
    1. generate self-play games with the current net,
    2. add them to a sliding replay window,
    3. train a candidate net on the window,
    4. play the candidate vs the previous net in the arena,
    5. keep the candidate only if it wins clearly (else revert).

This "only promote if better" gate is what makes the curve monotonic instead of
noisily wandering.
"""
from __future__ import annotations

import csv
import json
import os
from collections import deque

from tqdm import tqdm

from src.core.arena import greedy_mcts_player, play_match
from src.core.selfplay import execute_episode
from src.nn.net import NeuralNet


class Coach:
    def __init__(self, game, net, config, eval_hook=None, metrics_csv=None):
        self.game = game
        self.net = net
        self.config = config
        self.eval_hook = eval_hook  # optional: (iter, net) -> dict of extra metrics
        self.metrics_csv = metrics_csv  # optional path to append per-iter metrics
        self.history = deque(maxlen=config.replay_window)
        self.metrics = []

    def learn(self):
        prev_net = NeuralNet(self.game, self.config)
        for it in range(1, self.config.num_iters + 1):
            print(f"\n{'='*20} Iteration {it}/{self.config.num_iters} {'='*20}")
            # 1) self-play (parallel workers when configured)
            workers = getattr(self.config, "selfplay_workers", 1)
            if workers > 1:
                from src.core.parallel_selfplay import parallel_selfplay
                print(f"  spawning {workers} self-play workers "
                      f"on {getattr(self.config, 'selfplay_worker_device', 'cpu')} "
                      f"for {self.config.selfplay_games} games...", flush=True)
                iter_examples = parallel_selfplay(
                    self.game, self.net, self.config,
                    self.config.selfplay_games, workers,
                    worker_device=getattr(self.config, "selfplay_worker_device", "cpu"),
                )
                print(f"  self-play done: {len(iter_examples):,} examples", flush=True)
            else:
                iter_examples = []
                for _ in tqdm(range(self.config.selfplay_games), desc="Self-Play", unit="game"):
                    iter_examples += execute_episode(self.game, self.net, self.config)
            self.history.append(iter_examples)
            train_examples = [e for batch in self.history for e in batch]

            # 2) snapshot the current net as "previous", then train the candidate
            self.net.save_checkpoint("temp.pt")
            prev_net.load_checkpoint("temp.pt")
            pi_loss, v_loss = self.net.train(train_examples)

            # 3) gate: warmup iterations skip the arena and always accept the
            #    trained candidate (so the net can leave initialization before
            #    a noisy gate starts filtering); after that, standard head-to-head.
            warmup = it <= self.config.warmup_iters
            if warmup:
                nwins = owins = draws = 0
                accepted = True
                self.net.save_checkpoint("best.pt")
            else:
                new_p = greedy_mcts_player(self.game, self.net, self.config)
                old_p = greedy_mcts_player(self.game, prev_net, self.config)
                nwins, owins, draws = play_match(
                    self.game, new_p, old_p, self.config.arena_games,
                    max_moves=self.config.max_game_len,
                )
                decided = nwins + owins
                accepted = (decided > 0
                            and nwins / decided >= self.config.update_threshold)
                if accepted:
                    self.net.save_checkpoint("best.pt")
                else:
                    self.net.load_checkpoint("temp.pt")  # revert to previous weights
            # always save the latest weights so a long run is resumable / crash-safe
            self.net.save_checkpoint("latest.pt")

            row = {
                "iter": it,
                "examples": len(train_examples),
                "pi_loss": pi_loss,
                "v_loss": v_loss,
                "arena": (nwins, owins, draws),
                "accepted": accepted,
                "warmup": warmup,
            }
            if self.eval_hook is not None:
                row.update(self.eval_hook(it, self.net))
            self.metrics.append(row)
            self._log(row)
            self._append_csv(row)
        return self.metrics

    def _append_csv(self, row):
        if not self.metrics_csv:
            return
        os.makedirs(os.path.dirname(self.metrics_csv) or ".", exist_ok=True)
        nwins, owins, draws = row["arena"]
        skip = {"iter", "pi_loss", "v_loss", "arena", "accepted", "examples", "warmup"}
        flat = {
            "iter": row["iter"],
            "examples": row["examples"],
            "pi_loss": round(row["pi_loss"], 5),
            "v_loss": round(row["v_loss"], 5),
            "arena_new": nwins,
            "arena_old": owins,
            "arena_draw": draws,
            "accepted": int(row["accepted"]),
            "warmup": int(row.get("warmup", False)),
            "extra": json.dumps({k: v for k, v in row.items() if k not in skip}),
        }
        write_header = not os.path.exists(self.metrics_csv)
        with open(self.metrics_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(flat))
            if write_header:
                writer.writeheader()
            writer.writerow(flat)

    @staticmethod
    def _log(row):
        base = (
            f"[iter {row['iter']:>2}] "
            f"pi_loss={row['pi_loss']:.3f} v_loss={row['v_loss']:.3f} "
            f"arena(new/old/draw)={row['arena']} "
            f"{('ACCEPT(warmup)' if row.get('warmup') else ('ACCEPT' if row['accepted'] else 'reject'))}"
        )
        skip = {"iter", "pi_loss", "v_loss", "arena", "accepted", "examples", "warmup"}
        extra = "  ".join(f"{k}={v}" for k, v in row.items() if k not in skip)
        print(base + ("  " + extra if extra else ""), flush=True)
