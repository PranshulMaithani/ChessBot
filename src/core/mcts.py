"""
Monte Carlo Tree Search guided by the policy/value net (PUCT).

All search happens on canonical boards (side-to-move is always +1), so every
node is viewed from the perspective of the player about to move.

Key fixes:
- Repetition detection inside a single simulation path
- Hard maximum search depth
- Safe recursion (no infinite descent through cycles)

State is stored in plain dicts keyed by game.string_key(board):

    Qsa  (s,a) -> mean action value
    Nsa  (s,a) -> visit count
    Ns   s     -> visits to the node
    Ps   s     -> net policy prior
    Es   s     -> cached terminal value
    Vs   s     -> legal-move mask
"""

from __future__ import annotations

import math
import numpy as np

EPS = 1e-8
MAX_SEARCH_DEPTH = 512


class MCTS:
    def __init__(self, game, net, config):
        self.game = game
        self.net = net

        self.c_puct = config.c_puct
        self.num_sims = config.num_sims
        self.dirichlet_alpha = config.dirichlet_alpha
        self.dirichlet_eps = config.dirichlet_eps

        self.Qsa = {}
        self.Nsa = {}
        self.Ns = {}

        self.Ps = {}
        self.Es = {}
        self.Vs = {}

    def get_action_prob(self, board, temp=1.0, add_root_noise=False):
        """
        Run simulations from board and return a policy over actions
        proportional to visit counts^(1/temp).

        temp=0 => greedy.
        """

        self.search(board, depth=0, path=set())

        s = self.game.string_key(board)

        if add_root_noise and s in self.Ps:
            self._apply_dirichlet(s)

        for _ in range(max(self.num_sims - 1, 0)):
            self.search(board, depth=0, path=set())

        counts = np.array(
            [self.Nsa.get((s, a), 0) for a in range(self.game.get_action_size())],
            dtype=np.float64,
        )

        if temp == 0:
            best = np.flatnonzero(counts == counts.max())
            probs = np.zeros_like(counts)
            probs[np.random.choice(best)] = 1.0
            return probs

        counts = counts ** (1.0 / temp)

        total = counts.sum()

        if total <= 0:
            valid = self.Vs[s].astype(np.float64)
            return valid / valid.sum()

        return counts / total

    def _apply_dirichlet(self, s):
        """
        Mix Dirichlet noise into the root prior for self-play exploration.
        """

        idx = np.flatnonzero(self.Vs[s])

        if idx.size == 0:
            return

        noise = np.random.dirichlet(
            [self.dirichlet_alpha] * idx.size
        )

        p = self.Ps[s].copy()

        p[idx] = (
            (1 - self.dirichlet_eps) * p[idx]
            + self.dirichlet_eps * noise
        )

        self.Ps[s] = p

    def search(self, board, depth=0, path=None):
        """
        One MCTS simulation.

        Returns:
            value from the CURRENT player's perspective.
        """

        if path is None:
            path = set()

        s = self.game.string_key(board)

        # ------------------------------------------------------------
        # Repetition detection
        # ------------------------------------------------------------

        if s in path:
            return 0.0

        # ------------------------------------------------------------
        # Emergency recursion guard
        # ------------------------------------------------------------

        if depth >= MAX_SEARCH_DEPTH:
            return 0.0

        # ------------------------------------------------------------
        # Terminal check
        # ------------------------------------------------------------

        if s not in self.Es:
            self.Es[s] = self.game.get_game_ended(board, 1)

        if self.Es[s] != 0:
            return -self.Es[s]

        # ------------------------------------------------------------
        # Leaf expansion
        # ------------------------------------------------------------

        if s not in self.Ps:
            p, v = self.net.predict(board)

            valid = self.game.get_valid_moves(board, 1)

            p = p * valid

            if p.sum() > 0:
                p = p / p.sum()
            else:
                p = valid / valid.sum()

            self.Ps[s] = p
            self.Vs[s] = valid
            self.Ns[s] = 0

            return -v

        # ------------------------------------------------------------
        # Select action via PUCT
        # ------------------------------------------------------------

        valid = self.Vs[s]

        sqrt_ns = math.sqrt(self.Ns[s] + EPS)

        best_u = -float("inf")
        best_a = -1

        for a in np.flatnonzero(valid):
            a = int(a)

            if (s, a) in self.Qsa:
                u = (
                    self.Qsa[(s, a)]
                    + self.c_puct
                    * self.Ps[s][a]
                    * sqrt_ns
                    / (1 + self.Nsa[(s, a)])
                )
            else:
                u = (
                    self.c_puct
                    * self.Ps[s][a]
                    * sqrt_ns
                )

            if u > best_u:
                best_u = u
                best_a = a

        a = best_a

        # ------------------------------------------------------------
        # Recurse
        # ------------------------------------------------------------

        next_board, next_player = self.game.get_next_state(
            board,
            1,
            a,
        )

        next_canon = self.game.get_canonical_form(
            next_board,
            next_player,
        )

        path.add(s)

        v = self.search(
            next_canon,
            depth=depth + 1,
            path=path,
        )

        path.remove(s)

        # ------------------------------------------------------------
        # Backprop
        # ------------------------------------------------------------

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (
                self.Nsa[(s, a)] * self.Qsa[(s, a)] + v
            ) / (self.Nsa[(s, a)] + 1)

            self.Nsa[(s, a)] += 1

        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1

        return -v