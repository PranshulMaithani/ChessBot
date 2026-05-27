"""Monte Carlo Tree Search guided by the policy/value net (PUCT).

All search happens on *canonical* boards (side-to-move is always +1), so every
node is viewed "from the perspective of the player about to move". Values are
negated as we recurse, which is how the single +1 viewpoint stays consistent
for both players.

State is stored in plain dicts keyed by ``game.string_key(board)``:
    Qsa  (s,a) -> mean action value
    Nsa  (s,a) -> visit count
    Ns   s     -> visits to the node
    Ps   s     -> net policy prior (masked to legal moves)
    Es   s     -> cached terminal value (0 if non-terminal)
    Vs   s     -> legal-move mask
"""
from __future__ import annotations

import math

import numpy as np

EPS = 1e-8


class MCTS:
    def __init__(self, game, net, config):
        self.game = game
        self.net = net
        self.c_puct = config.c_puct
        self.num_sims = config.num_sims
        self.dirichlet_alpha = config.dirichlet_alpha
        self.dirichlet_eps = config.dirichlet_eps

        self.Qsa, self.Nsa, self.Ns = {}, {}, {}
        self.Ps, self.Es, self.Vs = {}, {}, {}

    def get_action_prob(self, board, temp=1.0, add_root_noise=False):
        """Run simulations from ``board`` and return a policy over actions
        proportional to (visit count) ** (1/temp). ``temp=0`` => greedy."""
        self.search(board)  # expand the root first
        s = self.game.string_key(board)
        if add_root_noise and s in self.Ps:
            self._apply_dirichlet(s)
        for _ in range(max(self.num_sims - 1, 0)):
            self.search(board)

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
        if total <= 0:  # net gave zero mass to all legal moves; fall back to uniform
            valid = self.Vs[s].astype(np.float64)
            return valid / valid.sum()
        return counts / total

    def _apply_dirichlet(self, s):
        """Mix Dirichlet noise into the root prior for self-play exploration."""
        idx = np.flatnonzero(self.Vs[s])
        if idx.size == 0:
            return
        noise = np.random.dirichlet([self.dirichlet_alpha] * idx.size)
        p = self.Ps[s].copy()
        p[idx] = (1 - self.dirichlet_eps) * p[idx] + self.dirichlet_eps * noise
        self.Ps[s] = p

    def search(self, board):
        """One simulation. Returns the value from the *current* player's view."""
        s = self.game.string_key(board)

        if s not in self.Es:
            self.Es[s] = self.game.get_game_ended(board, 1)
        if self.Es[s] != 0:  # terminal node
            return -self.Es[s]

        if s not in self.Ps:  # leaf: expand with the net and stop
            p, v = self.net.predict(board)
            valid = self.game.get_valid_moves(board, 1)
            p = p * valid
            p = p / p.sum() if p.sum() > 0 else valid / valid.sum()
            self.Ps[s], self.Vs[s], self.Ns[s] = p, valid, 0
            return -v

        # internal node: pick the action maximizing the PUCT score
        valid = self.Vs[s]
        sqrt_ns = math.sqrt(self.Ns[s] + EPS)
        best_u, best_a = -float("inf"), -1
        for a in np.flatnonzero(valid):
            a = int(a)
            if (s, a) in self.Qsa:
                u = self.Qsa[(s, a)] + self.c_puct * self.Ps[s][a] * sqrt_ns / (1 + self.Nsa[(s, a)])
            else:
                u = self.c_puct * self.Ps[s][a] * sqrt_ns  # Q assumed 0 for unvisited
            if u > best_u:
                best_u, best_a = u, a
        a = best_a

        next_board, next_player = self.game.get_next_state(board, 1, a)
        next_canon = self.game.get_canonical_form(next_board, next_player)
        v = self.search(next_canon)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1
        else:
            self.Qsa[(s, a)], self.Nsa[(s, a)] = v, 1
        self.Ns[s] += 1
        return -v
