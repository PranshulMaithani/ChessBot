"""A persistent pool of self-play worker processes.

The existing `parallel_selfplay` respawns N workers every iteration: each one
imports torch (~3-5 s on Windows), allocates a fresh CUDA context (~500 MB-1 GB),
torch.loads the latest checkpoint, and pays cuDNN's first-forward autotune cost.
On a multi-hour training run this adds up to many minutes of pure overhead, all
of it wall-clock you never see translated into games.

This module keeps workers alive for the lifetime of the training run. Each
iteration the main process broadcasts the current weights and a job count;
workers reload (~100 ms), generate games, and block on the next job. The
respawn cost goes away, CUDA contexts and cuDNN benchmarks are amortized
across all iterations.

Tradeoff: worker CUDA contexts stay resident even during the training and
arena phases (when only the main process needs the GPU). On 8 GB with a 96ch
x 8 net this is fine — 4 worker contexts use about 2 GB and the trainer
backward pass fits comfortably in the remaining 4-5 GB.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import sys
import tempfile
from dataclasses import asdict
from typing import List, Tuple

import torch

# Workers spawned via 'spawn' re-import this module fresh; project root on
# sys.path so `from src.* import ...` resolves the same as in the main process.
_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_SHUTDOWN = "__shutdown__"
_PLAY = "play"


def _persistent_worker(worker_id, game, config_dict, in_queue, out_queue):
    """Worker entry point. Loops on `in_queue` for jobs until told to shut down.

    Each message is a tuple ``(cmd, n_games, weights_path)``:
        ("play", n, path)  -> reload weights from `path`, play `n` games,
                              put (worker_id, examples) on out_queue.
        ("__shutdown__",): exit cleanly.

    The net + CUDA context live for the lifetime of the worker, so the first
    job pays the cold-start cost (~3-5s + cuDNN warmup) and subsequent jobs
    are bottlenecked only by actual self-play.
    """
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)
    from src.config import Config
    from src.core.selfplay import execute_episode
    from src.nn.net import NeuralNet

    config = Config(**config_dict)
    net = NeuralNet(game, config)

    # Same Ampere niceties as the main process; cheap and cumulative.
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    while True:
        msg = in_queue.get()
        cmd = msg[0]
        if cmd == _SHUTDOWN:
            return
        if cmd != _PLAY:
            continue  # unknown command; ignore to stay robust

        _, n_games, weights_path = msg
        if weights_path and os.path.exists(weights_path):
            # Always reload — main process writes fresh weights every iter.
            ckpt = torch.load(weights_path, map_location=net.device)
            net.model.load_state_dict(ckpt["state_dict"])

        examples = []
        for _ in range(n_games):
            examples += execute_episode(game, net, config)
        out_queue.put((worker_id, examples))


class PersistentSelfplayPool:
    """Worker pool whose lifetime spans a whole training run.

    Use as a context manager so workers exit cleanly even on Ctrl-C::

        with PersistentSelfplayPool(game, cfg, n_workers, "cuda") as pool:
            for it in range(iters):
                examples = pool.run(net, cfg.selfplay_games)
                ...
    """

    def __init__(self, game, config, num_workers: int, worker_device: str = "cpu"):
        self.game = game
        self.config = config
        self.num_workers = num_workers
        self.worker_device = worker_device
        self.in_queues: list = []
        self.out_queue = None
        self.procs: list = []
        # Stable temp file location for weight handoff. Re-used across iters;
        # workers always reload, so reusing the path is correct (and tidier).
        self._weights_path = os.path.join(
            tempfile.gettempdir(), f"chesseng_pool_{os.getpid()}.pt"
        )

    # --- lifecycle ---------------------------------------------------------

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.shutdown()

    def start(self):
        if self.procs:  # idempotent
            return
        ctx = mp.get_context("spawn")
        worker_config_dict = asdict(self.config)
        worker_config_dict["device"] = self.worker_device
        self.out_queue = ctx.Queue()
        for wid in range(self.num_workers):
            inq = ctx.Queue()
            p = ctx.Process(
                target=_persistent_worker,
                args=(wid, self.game, worker_config_dict, inq, self.out_queue),
                daemon=True,
            )
            p.start()
            self.in_queues.append(inq)
            self.procs.append(p)

    def shutdown(self):
        for q in self.in_queues:
            try:
                q.put((_SHUTDOWN,))
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
        self.in_queues.clear()
        self.procs.clear()
        try:
            os.remove(self._weights_path)
        except OSError:
            pass

    # --- one iteration's worth of work -------------------------------------

    def run(self, net, num_games: int) -> List[Tuple]:
        """Broadcast `net`'s current weights, play `num_games` across the pool,
        return the aggregated example list (same format as execute_episode)."""
        if not self.procs:
            raise RuntimeError("pool not started; call start() or use as a context manager")

        # Stash current weights for workers to torch.load.
        torch.save({"state_dict": net.model.state_dict()}, self._weights_path)

        # Split games as evenly as possible.
        per = [num_games // self.num_workers] * self.num_workers
        for i in range(num_games % self.num_workers):
            per[i] += 1

        dispatched = 0
        for wid, n in enumerate(per):
            if n > 0:
                self.in_queues[wid].put((_PLAY, n, self._weights_path))
                dispatched += 1

        all_examples: list = []
        for _ in range(dispatched):
            _, examples = self.out_queue.get()
            all_examples.extend(examples)
        return all_examples
