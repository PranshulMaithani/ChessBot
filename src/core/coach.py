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

from collections import deque

from src.core.arena import greedy_mcts_player, play_match
from src.core.selfplay import execute_episode
from src.nn.net import NeuralNet


class Coach:
    def __init__(self, game, net, config, eval_hook=None):
        self.game = game
        self.net = net
        self.config = config
        self.eval_hook = eval_hook  # optional: (iter, net) -> dict of extra metrics
        self.history = deque(maxlen=config.replay_window)
        self.metrics = []

    def learn(self):
        prev_net = NeuralNet(self.game, self.config)
        for it in range(1, self.config.num_iters + 1):
            # 1) self-play
            iter_examples = []
            for _ in range(self.config.selfplay_games):
                iter_examples += execute_episode(self.game, self.net, self.config)
            self.history.append(iter_examples)
            train_examples = [e for batch in self.history for e in batch]

            # 2) snapshot the current net as "previous", then train the candidate
            self.net.save_checkpoint("temp.pt")
            prev_net.load_checkpoint("temp.pt")
            pi_loss, v_loss = self.net.train(train_examples)

            # 3) gate: candidate vs previous
            new_p = greedy_mcts_player(self.game, self.net, self.config)
            old_p = greedy_mcts_player(self.game, prev_net, self.config)
            nwins, owins, draws = play_match(self.game, new_p, old_p, self.config.arena_games)
            decided = nwins + owins
            frac = nwins / decided if decided else 0.0
            accepted = frac >= self.config.update_threshold

            if accepted:
                self.net.save_checkpoint("best.pt")
            else:
                self.net.load_checkpoint("temp.pt")  # revert to previous weights

            row = {
                "iter": it,
                "examples": len(train_examples),
                "pi_loss": pi_loss,
                "v_loss": v_loss,
                "arena": (nwins, owins, draws),
                "accepted": accepted,
            }
            if self.eval_hook is not None:
                row.update(self.eval_hook(it, self.net))
            self.metrics.append(row)
            self._log(row)
        return self.metrics

    @staticmethod
    def _log(row):
        base = (
            f"[iter {row['iter']:>2}] "
            f"pi_loss={row['pi_loss']:.3f} v_loss={row['v_loss']:.3f} "
            f"arena(new/old/draw)={row['arena']} "
            f"{'ACCEPT' if row['accepted'] else 'reject'}"
        )
        skip = {"iter", "pi_loss", "v_loss", "arena", "accepted", "examples"}
        extra = "  ".join(f"{k}={v}" for k, v in row.items() if k not in skip)
        print(base + ("  " + extra if extra else ""), flush=True)
