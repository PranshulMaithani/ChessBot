# Build Journal

A running log of decisions, experiments, and results. Newest entries on top.

---

## 2026-05-28 — Stage 1: chess pipeline live

**Goal:** plug chess into the validated core and start a self-play run.

**Built**
- `src/games/chess_game.py`: chess over python-chess with
  - canonical form = always white-to-move (mirror when black to move),
  - AlphaZero 8x8x73 = 4672 action encoding (queen / knight / underpromotion),
  - 18-plane board encoding (12 pieces + 4 castling + ep + halfmove clock).
- 8 correctness tests (`tests/test_chess.py`): encoding bijectivity, canonical
  mirroring, terminal values, random-game consistency. All pass.
- Generic upgrades: ply cap (adjudicate draw), per-iteration `latest.pt`
  checkpoint, CSV metrics, `scripts/plot_metrics.py`.
- `scripts/train_chess.py`: resumable overnight runner.

**Sanity check**
- Smoke run completed on GPU; untrained `pi_loss = 8.46 ~= ln(4672)`, i.e. the
  policy starts exactly uniform over the legal action space (as it should).

**Starting config (laptop overnight):** 64ch x6 ResNet, 80 MCTS sims,
16 games/iter, replay window 8, 200-ply cap. Progress probe = raw-policy
(no search) win rate vs a random mover.

**Known limits / Stage 2 levers**
- 80 sims is low (AlphaZero used 800) — sims is the main strength/speed knob.
- MCTS is single-board + recursive + `board.copy()`; batched inference and
  make/undo are the big speedups.
- No board-history planes yet.

**Next**
- Train overnight; morning: plot the curve and play a game against it.

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
