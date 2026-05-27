"""The policy/value network — a small AlphaZero-style residual CNN.

Input : (C, H, W) encoded canonical board.
Output: policy logits over all actions, and a scalar value in [-1, 1]
        (expected game result from the side-to-move's perspective).

The same architecture serves every game; only the input channels, board size,
and action count change. We scale ``num_channels`` / ``num_res_blocks`` up for
chess later.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return F.relu(x + y)  # residual skip connection


class AlphaZeroNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        board_h: int,
        board_w: int,
        action_size: int,
        num_channels: int = 64,
        num_res_blocks: int = 4,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, num_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=True),
        )
        self.tower = nn.Sequential(*[ResBlock(num_channels) for _ in range(num_res_blocks)])

        # policy head: 1x1 conv -> flatten -> linear to action logits
        self.p_conv = nn.Sequential(
            nn.Conv2d(num_channels, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.p_fc = nn.Linear(32 * board_h * board_w, action_size)

        # value head: 1x1 conv -> flatten -> linear -> scalar in [-1, 1]
        self.v_conv = nn.Sequential(
            nn.Conv2d(num_channels, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.v_fc1 = nn.Linear(32 * board_h * board_w, 128)
        self.v_fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.stem(x)
        x = self.tower(x)

        p = self.p_conv(x).flatten(1)
        p = self.p_fc(p)  # logits (softmax applied by the caller)

        v = self.v_conv(x).flatten(1)
        v = F.relu(self.v_fc1(v))
        v = torch.tanh(self.v_fc2(v)).squeeze(-1)
        return p, v
