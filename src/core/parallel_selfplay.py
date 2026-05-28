"""Parallel self-play across multiple processes.

Self-play in a single Python process is dominated by MCTS bookkeeping (dict
ops, board copies) — even with BatchedMCTS keeping the GPU busy, the GIL keeps
one core working at a time and 7 cores idle. Spawning N independent worker
processes side-steps the GIL: each generates games in its own interpreter
from a shared checkpoint, then the main loop aggregates the examples and
trains as before.

Tradeoffs
---------
worker_device="cpu" (default)
    No GPU contention; you can run many workers (>= core count). Each game is
    slower individually (CPU inference is ~3-5x slower than GPU for our nets),
    but total throughput scales near-linearly with cores. Best choice on a
    laptop where the GPU is the scarce resource.

worker_device="cuda"
    Each worker spawns its own CUDA context (~500 MB-1 GB overhead) and
    serialises through the GPU at the kernel level. 2-3 workers fits in 8 GB
    VRAM with a 96ch x 8 net; more risks OOM. Faster per game.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import sys
import tempfile
from dataclasses import asdict
from typing import List, Tuple

import torch

# Workers spawned with the 'spawn' start method re-import this file fresh; make
# sure the project root is on sys.path so `from src.* import ...` works.
_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _worker(
    worker_id: int,
    game,
    config_dict: dict,
    num_games: int,
    ckpt_path: str,
    return_queue,
):
    """Worker entry point. Reconstructs config + net, generates `num_games`
    self-play episodes, returns them through the queue."""
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)
    from src.config import Config
    from src.core.selfplay import execute_episode
    from src.nn.net import NeuralNet

    config = Config(**config_dict)
    net = NeuralNet(game, config)
    if ckpt_path and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=net.device)
        net.model.load_state_dict(ckpt["state_dict"])

    examples = []
    for _ in range(num_games):
        examples += execute_episode(game, net, config)
    return_queue.put((worker_id, examples))


def parallel_selfplay(
    game,
    net,
    config,
    num_games: int,
    num_workers: int,
    worker_device: str = "cpu",
) -> List[Tuple]:
    """Run `num_games` self-play games across `num_workers` processes.
    Returns the aggregated example list (same format as execute_episode)."""
    if num_workers <= 1 or num_games <= 1:
        from src.core.selfplay import execute_episode
        examples = []
        for _ in range(num_games):
            examples += execute_episode(game, net, config)
        return examples

    # Stash current weights so workers can load them.
    tmp_ckpt = os.path.join(
        tempfile.gettempdir(), f"chesseng_worker_init_{os.getpid()}.pt"
    )
    torch.save({"state_dict": net.model.state_dict()}, tmp_ckpt)

    # Workers may run on a different device than the main process.
    worker_config_dict = asdict(config)
    worker_config_dict["device"] = worker_device

    # Split games evenly; remainder goes to the first few workers.
    games_per_worker = [num_games // num_workers] * num_workers
    for i in range(num_games % num_workers):
        games_per_worker[i] += 1
    games_per_worker = [g for g in games_per_worker if g > 0]

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = []
    for wid, g in enumerate(games_per_worker):
        p = ctx.Process(
            target=_worker,
            args=(wid, game, worker_config_dict, g, tmp_ckpt, q),
        )
        p.start()
        procs.append(p)

    all_examples: List[Tuple] = []
    for _ in procs:
        _, examples = q.get()
        all_examples.extend(examples)
    for p in procs:
        p.join()

    try:
        os.remove(tmp_ckpt)
    except OSError:
        pass

    return all_examples
