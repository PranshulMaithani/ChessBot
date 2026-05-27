"""Thin wrapper around :class:`AlphaZeroNet`: device handling, prediction,
training step, and checkpointing. Game-agnostic — it reads the board shape /
channels / action size from the game's encoder.
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from src.nn.model import AlphaZeroNet


class NeuralNet:
    def __init__(self, game, config):
        self.game = game
        self.config = config

        sample = game.encode(game.get_canonical_form(game.get_init_state(), 1))
        in_ch = sample.shape[0]
        h, w = game.get_board_shape()
        self.action_size = game.get_action_size()

        use_cuda = config.device == "cuda" and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.model = AlphaZeroNet(
            in_ch, h, w, self.action_size, config.num_channels, config.num_res_blocks
        ).to(self.device)

    def predict(self, canonical_board):
        """Single board -> (policy probabilities, value scalar)."""
        x = torch.from_numpy(self.game.encode(canonical_board)).unsqueeze(0).float().to(self.device)
        self.model.eval()
        with torch.no_grad():
            logits, v = self.model(x)
            p = torch.softmax(logits, dim=1)[0]
        return p.cpu().numpy(), float(v.item())

    def train(self, examples):
        """examples: list of (canonical_board, pi, z). Returns (avg_pi_loss, avg_v_loss)."""
        self.model.train()
        optimizer = optim.Adam(
            self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay
        )
        bs = self.config.batch_size
        n = len(examples)
        tot_pi = tot_v = 0.0
        n_batches = 0
        for _ in range(self.config.epochs):
            np.random.shuffle(examples)
            for i in range(0, n, bs):
                batch = examples[i : i + bs]
                boards = torch.from_numpy(
                    np.stack([self.game.encode(b) for b, _, _ in batch])
                ).float().to(self.device)
                target_pi = torch.from_numpy(
                    np.stack([p for _, p, _ in batch]).astype(np.float32)
                ).to(self.device)
                target_v = torch.tensor(
                    [z for _, _, z in batch], dtype=torch.float32, device=self.device
                )

                logits, v = self.model(boards)
                l_pi = -(target_pi * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
                l_v = F.mse_loss(v, target_v)
                loss = l_pi + l_v

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                tot_pi += l_pi.item()
                tot_v += l_v.item()
                n_batches += 1
        denom = max(n_batches, 1)
        return tot_pi / denom, tot_v / denom

    def save_checkpoint(self, name):
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        torch.save({"state_dict": self.model.state_dict()},
                   os.path.join(self.config.checkpoint_dir, name))

    def load_checkpoint(self, name):
        path = os.path.join(self.config.checkpoint_dir, name)
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["state_dict"])
