# Build Journal

A running log of decisions, experiments, and results. Newest entries on top.

---

## 2026-05-28 — Day 0: foundations

**Goal:** stand up the project and the core abstraction.

**Decisions**
- **Method:** AlphaZero-style self-play RL (MCTS + a policy/value neural net).
  Chosen for the learning journey; we know it won't reach engine-grade strength
  on one laptop GPU, and we'll optimize within that budget.
- **Staged build:** validate the entire loop on Tic-Tac-Toe → Connect 4 →
  full chess. The engine is written once against a `Game` interface, so each
  stage is a plug-in.
- **Stack:** Python 3.10, PyTorch 2.11 + CUDA 12.8 (RTX 3070 Laptop, 8 GB),
  `python-chess` for move generation (no point reinventing legal-move logic).
- **Board legality for chess:** delegate to `python-chess`. Our `Game` layer
  only adapts it to the AlphaZero interface (canonical form, action encoding).

**Done today**
- Repo scaffold, `.gitignore`, requirements.
- `Game` interface (`src/games/base.py`).
- Tic-Tac-Toe implementation + random-game sanity tests.

**Result — Stage 0 gate: PASSED**
- Built the full loop: ResNet policy/value net, PUCT MCTS, self-play, trainer,
  arena gate (`src/nn`, `src/core`).
- Validation run (12 iters, 25 games/iter, 30 sims): the agent reached
  **perfect play** — vs a minimax solver it went 0W/0L/40D (optimal TTT is a
  forced draw), and 37/0/3 vs a random mover. Zero losses by iter 5.
- `pi_loss` 1.57 -> 1.09, `v_loss` 0.45 -> 0.22.
- Note: once the game is solved, the arena correctly rejects all further
  candidates (a new net can't beat an already-perfect one). Expected, not a bug.
- Repro: `python scripts/train_tictactoe.py --iters 12 --games 25 --sims 30`

**Next**
- Stage 0.5: Connect 4 (real MCTS+NN stress test at small scale).
- Persist per-iteration metrics + plot the learning curve (for the journal).
- Then Stage 1: the chess adapter.

**Open questions**
- Net size vs. MCTS sims trade-off on 8 GB VRAM (revisit at Stage 1).
