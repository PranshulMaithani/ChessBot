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

**Next**
- Neural net (small ResNet, policy + value heads).
- MCTS.
- Self-play + training loop; confirm Tic-Tac-Toe converges to perfect play.

**Open questions**
- Net size vs. MCTS sims trade-off on 8 GB VRAM (revisit at Stage 1).
