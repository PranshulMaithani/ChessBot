"""The Game interface for AlphaZero-style self-play.

Every game (Tic-Tac-Toe, Connect 4, Chess) implements this interface, so the
AlphaZero core — MCTS, self-play, training — is written **once** and reused
across all of them.

Conventions
-----------
players
    Two players, ``+1`` and ``-1``. ``+1`` always moves first from the initial
    state.

state / board
    A NumPy array describing the position, stored in *absolute* terms (it does
    not change with whose turn it is). Whose turn it is, is tracked separately
    by passing ``player`` around.

canonical form
    The board as seen by the *player to move*, rearranged so the side to move
    is always ``+1``. The neural net and MCTS only ever see canonical boards,
    so they never need to know whose turn it is. For symmetric games the
    canonical form is simply ``board * player``; for chess it means mirroring
    the board so the side to move is "white at the bottom".

actions
    Encoded as integers in ``[0, action_size)``. :meth:`get_valid_moves`
    returns a binary mask of legal actions of length ``action_size``.

outcomes
    :meth:`get_game_ended` returns ``0`` while the game is ongoing, ``+1`` if
    the queried player has won, ``-1`` if they have lost, and a tiny non-zero
    value (``1e-4``) for a draw (so "draw" is distinguishable from "ongoing").
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Game(ABC):
    # --- static description of the game -------------------------------------
    @abstractmethod
    def get_init_state(self) -> np.ndarray:
        """Return the starting board (absolute form)."""

    @abstractmethod
    def get_board_shape(self) -> tuple[int, int]:
        """Return ``(rows, cols)`` of the board."""

    @abstractmethod
    def get_action_size(self) -> int:
        """Total number of distinct actions (size of the policy head)."""

    # --- dynamics -----------------------------------------------------------
    @abstractmethod
    def get_next_state(
        self, board: np.ndarray, player: int, action: int
    ) -> tuple[np.ndarray, int]:
        """Apply ``action`` for ``player``; return ``(next_board, next_player)``.

        Must not mutate the input ``board``.
        """

    @abstractmethod
    def get_valid_moves(self, board: np.ndarray, player: int) -> np.ndarray:
        """Binary vector of length ``action_size``; ``1`` where the move is legal."""

    @abstractmethod
    def get_game_ended(self, board: np.ndarray, player: int) -> float:
        """``0`` if ongoing; ``+1`` if ``player`` won, ``-1`` if lost, ``1e-4`` draw."""

    # --- perspective / encoding --------------------------------------------
    @abstractmethod
    def get_canonical_form(self, board: np.ndarray, player: int) -> np.ndarray:
        """Return the board from ``player``'s perspective, side-to-move as ``+1``."""

    @abstractmethod
    def encode(self, canonical_board: np.ndarray) -> np.ndarray:
        """Turn a canonical board into a ``(C, H, W)`` float32 tensor for the net."""

    @abstractmethod
    def string_key(self, canonical_board: np.ndarray) -> str:
        """A hashable key for a canonical board (used to cache MCTS nodes)."""

    # --- optional: data augmentation ---------------------------------------
    def get_symmetries(
        self, canonical_board: np.ndarray, pi: np.ndarray
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Return ``(board, pi)`` pairs equivalent under the game's symmetries.

        Used to multiply training data for free. The default returns just the
        identity; games with rotational/reflective symmetry should override it.
        ``pi`` is a policy vector of length ``action_size``.
        """
        return [(canonical_board, pi)]
