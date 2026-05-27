"""Hyperparameters for the AlphaZero training loop.

Defaults are tuned for fast iteration on small games (Tic-Tac-Toe / Connect 4).
They will be overridden per-stage; chess will want a bigger net and more sims.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Config:
    # --- neural net ---
    num_channels: int = 64       # conv width
    num_res_blocks: int = 4      # residual tower depth

    # --- MCTS ---
    num_sims: int = 50           # tree-search simulations per move
    c_puct: float = 1.5          # exploration constant in PUCT
    dirichlet_alpha: float = 0.5 # root noise concentration (smaller = peakier)
    dirichlet_eps: float = 0.25  # how much root noise to mix in during self-play

    # --- self-play ---
    num_iters: int = 15          # outer loop: (self-play -> train -> arena)
    selfplay_games: int = 40     # games generated per iteration
    temp_threshold: int = 6      # after this many plies, play greedily (temp->0)

    # --- training ---
    epochs: int = 10
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    replay_window: int = 20      # keep examples from the last N iterations

    # --- arena (gatekeeping) ---
    arena_games: int = 24
    update_threshold: float = 0.55  # new net must win this share of decided games

    # --- io / misc ---
    device: str = "cuda"         # falls back to cpu automatically if unavailable
    checkpoint_dir: str = "models"
    seed: int = 0
