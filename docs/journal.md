# Build Journal

A running log of decisions, experiments, and results. Newest entries on top.

---

## 2026-05-28 — Day 2 (cont.): supervised pretraining pipeline

**The pragmatic answer to chess cold-start: AlphaGo-style hybrid.**
Pure AlphaZero takes days on this hardware to break out of "shuffle to draw".
Pretrain the same net on human master games first; self-play continues from
there. AlphaGo (the predecessor) did this and was world-champion strong.

**Shipped**
- `src/data/pgn.py`: PGN → `(planes, action_idx, z)` extraction. Reuses
  `ChessGame.encode` and `move_to_index` so the net architecture / checkpoints
  are identical between supervised pretraining and self-play.
- `scripts/pretrain_chess.py`: supervised training loop (policy CE on the
  played move, value MSE on the game outcome). Same losses as self-play, just
  with human-game targets instead of MCTS targets. Saves
  `models/chess/pretrained.pt`.
- `scripts/train_chess.py --init-from <path>`: warm-start the self-play
  trainer from a checkpoint (e.g. the pretrained one).
- `docs/DATA.md`: download instructions, Lichess Elite default.

**Verified**
- 6 new PGN tests pass: mirror-move round-trip, white-perspective value
  targets in a white-won game (and inverse for black-won), Elo filter,
  skip-book advance, max-positions cap. All other suites green (31 tests
  total).

**Why this is expected to work where pure self-play stalled**
- Value head sees real ±1 outcomes from move 1 instead of mostly-zero draws.
- Policy head learns chess move shapes (opening principles, tactics) before
  MCTS ever runs.
- Resignation (Stage 2 fix) actually fires correctly once the value head is
  calibrated.
- BatchedMCTS (Stage 2 fix) makes the self-play continuation 5-10x faster.
- Combined: the chess self-play loop becomes productive on a laptop GPU
  instead of slogging at the plateau.

---

## 2026-05-28 — Day 2: Connect 4 success + Stage 2 engineering

**Connect 4: trained from zero in ~3 hours.**
- All 60 iterations completed.
- pi_loss 1.67 -> 0.42, v_loss 0.31 -> 0.17.
- vs random: 13/7/0 at iter 1 -> **20/0/0** by iter 26 and consistently after.
  The bot reliably wins Connect 4 against a random mover. First real "trained
  from zero by self-play" deliverable.
- Plot: `docs/plots/connect4_progress.png`. Play it: `scripts/play_connect4.py`.

**Chess: still plateauing on the laptop (as predicted).**
- Warmup gate worked (iters 1-5 each ACCEPT). pi_loss collapsed 5.83 -> 1.08.
- Iters 6-20 all-draws arena, all rejected -- the chess cold-start ceiling
  (80 sims is just below the threshold for finding decisive lines from a fresh
  net on bare boards). Iter 21 *did* get a marginal promotion (1/0/11).

**Stage 2 engineering shipped (toward breaking the chess plateau)**
- `src/core/batched_mcts.py`: BatchedMCTS with virtual loss. Runs `K`
  simulations in parallel and evaluates their leaves in one GPU forward pass
  -- the canonical AlphaZero speedup. Expected ~5-10x throughput on chess.
- `src/nn/net.py::predict_batch`: GPU-batched policy/value inference.
- `src/core/selfplay.py`: **resignation** -- if the net thinks the side to
  move is losing badly for N plies, end the game with a real -1/+1 label.
  Turns the value-uninformative "shuffle to the cap" into honest training
  signal (the actual fix for the chess draw plateau).
- `src/core/mcts_factory.py`: switches between simple/Batched MCTS by config.
- New chess config: `mcts_batch_size=16`, `resign_threshold=-0.85`.

**Cloud-ready**
- `docs/CLOUD.md`: Kaggle (30h/week T4) and Colab (~12h, Drive-persisted)
  setup with copy-pasteable cells and recommended config (`--mcts-batch 32
  --channels 96 --res-blocks 8 --resign-threshold -0.85`).

**Play the bots**
- `scripts/play_connect4.py` (interactive, ASCII board).
- `scripts/play.py` (chess; already existed).

**Verified**
- All test suites pass: TTT 4/4, Connect 4 8/8, chess 9/9, BatchedMCTS 4/4
  (including the King-vs-King repetition regression at 200 sims with
  `mcts_batch_size=16`).

**Next**
- Restart chess locally with `--resume`: should run faster and break the
  draw plateau.
- Push repo to GitHub, then start a Kaggle session for serious chess training.

---

## 2026-05-28 — Day 1: triage + Connect 4 detour

**Two things broke on the chess overnight run.**

1. **Silent crash on iter 1.** Native stack overflow from MCTS recursing
   infinitely through cycles of fully-expanded positions (a weak net wanders
   into King-vs-King-style shuffles, those positions all expand, and a later
   simulation descends through the cycle forever — no Python traceback because
   the C stack dies first). Patched in-session: MCTS now tracks states visited
   in the current descent and treats repetition as a draw, with a
   `MAX_SEARCH_DEPTH` backstop. Locked in by a new regression test
   (`test_mcts_terminates_on_repetition_cycles` — King-vs-King at 200 sims).

2. **14 iterations ran but the net never left initialization.** Arena rejected
   nearly every candidate (iter 1: 0-11-1, then mostly 1-8-3 sort of patterns),
   so coach reloaded `temp.pt` each time. Cold-start failure: 80 sims on an
   untrained net is essentially a random walk, 200-ply self-play games are
   adjudicated as draws (value ~ 0), so training perturbs the net into
   worse-than-uniform territory and the gate (correctly) rejects it. Forever.
   Also: `play_game` had no length cap, so arena ate ~60% of each iteration.

**Fixes shipped**
- `play_game` / `play_match` accept `max_moves`; chess passes
  `max_game_len=200` to the arena too.
- coach: optional warmup phase skips the arena and unconditionally accepts
  the trained candidate for the first N iters; chess uses `warmup_iters=5`.
- chess `update_threshold` 0.55 -> 0.52 (12 arena games is too noisy for 0.55).
- per-game checkpoint dirs: `models/chess/`, `models/connect4/`,
  `models/tictactoe/` (so runs of different games don't overwrite each other).

**Stage 0.5: Connect 4 (built today)**
- `src/games/connect4.py` + 8 passing tests + `scripts/train_connect4.py`.
- Smoke run: a 16-channel untrained net + 10 MCTS sims already won 17 / 20
  vs a random opponent. Connect 4 produces decisive games immediately, so
  the value head gets honest signal from iteration 1 — exactly what chess
  cold-start lacks on this hardware.

**Plan**
- Tonight: `train_connect4.py` (a few hours, will show a real learning curve).
- Then revisit chess with the fixes; arena gate now actually lets the net
  evolve.

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
