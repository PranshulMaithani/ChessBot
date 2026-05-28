"""Batched MCTS with virtual loss.

The dominant cost of plain MCTS on a GPU is single-board net.predict calls —
one forward pass per leaf expansion. Running K simulations in parallel and
collecting their leaves into one batched forward pass cuts that cost by ~K
(modulo bookkeeping overhead). Virtual loss steers the parallel descents away
from the same path so they actually explore different branches.

Drop-in API-compatible with :class:`src.core.mcts.MCTS`. With
``mcts_batch_size=1`` it behaves like a single-threaded iterative MCTS.

Like the simple MCTS it tracks the state keys visited along *this* descent
(``seen`` set) and treats re-entry as a draw — the same fix that closed the
overnight crash on chess.
"""
from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

EPS = 1e-8
MAX_SEARCH_DEPTH = 512


class BatchedMCTS:
    def __init__(self, game, net, config):
        self.game = game
        self.net = net
        self.c_puct = config.c_puct
        self.num_sims = config.num_sims
        self.batch_size = max(1, getattr(config, "mcts_batch_size", 1))
        self.virtual_loss = getattr(config, "virtual_loss", 1.0)
        self.dirichlet_alpha = config.dirichlet_alpha
        self.dirichlet_eps = config.dirichlet_eps

        self.Qsa, self.Nsa, self.Ns = {}, {}, {}
        self.Ps, self.Es, self.Vs = {}, {}, {}
        self.virtual_n = {}  # (s, a) -> in-flight virtual visits
        # MCTS revisits the same (state, action) pair dozens of times via
        # transpositions; computing the child canonical board each time means
        # repeated chess.Board.copy + push + mirror. Caching the result per
        # (s, a) eats this cost once. Lives for the duration of the episode
        # (the MCTS is re-instantiated each episode, so memory is bounded).
        self._next_cache: dict = {}

    def get_action_prob(self, board, temp=1.0, add_root_noise=False):
        # Expand the root first (one sim) so we have a Ps to perturb.
        self._run_single_sim(board)
        s = self.game.string_key(board)
        if add_root_noise and s in self.Ps:
            self._apply_dirichlet(s)

        remaining = max(self.num_sims - 1, 0)
        while remaining > 0:
            k = min(self.batch_size, remaining)
            self._run_batch(board, k)
            remaining -= k

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
        idx = np.flatnonzero(self.Vs[s])
        if idx.size == 0:
            return
        noise = np.random.dirichlet([self.dirichlet_alpha] * idx.size)
        p = self.Ps[s].copy()
        p[idx] = (1 - self.dirichlet_eps) * p[idx] + self.dirichlet_eps * noise
        self.Ps[s] = p

    # --- simulation core ---------------------------------------------------

    def _run_single_sim(self, root):
        path, leaf, term_value = self._descend(root)
        if term_value is not None:
            self._backup(path, term_value)
        else:
            p, v = self.net.predict(leaf)
            self._expand(leaf, p)
            self._backup(path, v)

    def _run_batch(self, root, k: int):
        """Descend k simulations in parallel (virtual loss); evaluate all leaves
        in one batched net call; back up all paths."""
        pending: List[Tuple[List[Tuple[str, int]], object]] = []
        for _ in range(k):
            path, leaf, term_value = self._descend(root)
            if term_value is not None:
                self._backup(path, term_value)
            else:
                self._apply_virtual_loss(path)
                pending.append((path, leaf))
        if not pending:
            return
        policies, values = self.net.predict_batch([leaf for _, leaf in pending])
        for (path, leaf), p, v in zip(pending, policies, values):
            self._remove_virtual_loss(path)
            self._expand(leaf, p)
            self._backup(path, float(v))

    def _descend(self, board):
        """From `board`, walk down to (a) a leaf to be expanded, (b) a terminal,
        (c) a repeated state in this descent, or (d) the depth cap. Exactly one
        of leaf_board / terminal_value is non-None on return."""
        path: List[Tuple[str, int]] = []
        seen = set()
        cur = board
        # Prefer the cheap no-claim game-end check inside the tree if the game
        # offers one (chess does — saves ~20% of self-play time by skipping
        # the threefold-repetition scan on hypothetical states). The `seen`
        # set above already handles repetition correctness for this descent.
        game_ended = getattr(self.game, "get_game_ended_no_claim",
                             self.game.get_game_ended)
        for _ in range(MAX_SEARCH_DEPTH):
            s = self.game.string_key(cur)
            if s in seen:
                return path, None, 0.0           # repetition -> draw
            if s not in self.Es:
                self.Es[s] = game_ended(cur, 1)
            if self.Es[s] != 0:
                return path, None, self.Es[s]    # terminal
            if s not in self.Ps:
                return path, cur, None           # leaf to be expanded
            seen.add(s)
            a = self._select(s)
            path.append((s, a))
            # Cached child canonical board for (s, a) — chess.Board ops are C
            # but still ~130 us each via Python, and transpositions revisit
            # the same (s, a) dozens of times per game.
            key = (s, a)
            child = self._next_cache.get(key)
            if child is None:
                nb, npl = self.game.get_next_state(cur, 1, a)
                child = self.game.get_canonical_form(nb, npl)
                self._next_cache[key] = child
            cur = child
        return path, None, 0.0                   # depth cap -> draw

    def _expand(self, board, policy):
        s = self.game.string_key(board)
        if s in self.Ps:
            # another sim in the same batch already expanded this leaf — no-op
            return
        valid = self.game.get_valid_moves(board, 1)
        p = policy * valid
        p = p / p.sum() if p.sum() > 0 else valid / valid.sum()
        self.Ps[s] = p
        self.Vs[s] = valid
        self.Ns[s] = 0

    def _select(self, s):
        """PUCT selection that's aware of virtual visits in flight."""
        valid = self.Vs[s]
        # include virtual visits in the parent count for the sqrt(N_total) term
        virt_total = sum(self.virtual_n.get((s, int(a)), 0) for a in np.flatnonzero(valid))
        sqrt_ns = math.sqrt(self.Ns[s] + virt_total + EPS)
        best_u, best_a = -float("inf"), -1
        for a in np.flatnonzero(valid):
            a = int(a)
            real_n = self.Nsa.get((s, a), 0)
            real_q = self.Qsa.get((s, a), 0.0)
            virt = self.virtual_n.get((s, a), 0)
            total = real_n + virt
            if total > 0:
                # Treat each in-flight sim as if it returned -virtual_loss (the
                # worst possible outcome for the side to move) — this is the
                # virtual-loss trick that spreads parallel descents.
                q_eff = (real_n * real_q + virt * (-self.virtual_loss)) / total
            else:
                q_eff = 0.0
            u = q_eff + self.c_puct * self.Ps[s][a] * sqrt_ns / (1 + total)
            if u > best_u:
                best_u, best_a = u, a
        return best_a

    def _apply_virtual_loss(self, path):
        for s, a in path:
            self.virtual_n[(s, a)] = self.virtual_n.get((s, a), 0) + 1

    def _remove_virtual_loss(self, path):
        for s, a in path:
            self.virtual_n[(s, a)] -= 1
            if self.virtual_n[(s, a)] == 0:
                del self.virtual_n[(s, a)]

    def _backup(self, path, leaf_value):
        v = leaf_value
        for s, a in reversed(path):
            v = -v  # flip to the parent's perspective
            if (s, a) in self.Qsa:
                self.Qsa[(s, a)] = (
                    self.Nsa[(s, a)] * self.Qsa[(s, a)] + v
                ) / (self.Nsa[(s, a)] + 1)
                self.Nsa[(s, a)] += 1
            else:
                self.Qsa[(s, a)], self.Nsa[(s, a)] = v, 1
            self.Ns[s] += 1
