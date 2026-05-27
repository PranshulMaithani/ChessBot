# Chesseng — a chess engine that learns from zero by self-play

An AlphaZero-style chess engine trained **locally** through self-play
reinforcement learning. No human games, no handcrafted evaluation — the bot
starts knowing only the rules and gets better by playing itself.

> Hardware target: a single laptop GPU (RTX 3070, 8 GB). We will not reach
> Stockfish strength, but we *will* watch an engine climb from random moves to
> real chess, and document every step.

## How it works (the loop)

```
        ┌─────────────┐   games    ┌──────────┐   better net   ┌────────┐
        │  Self-play  │──────────► │  Train   │──────────────► │  Arena │
        │ (MCTS + NN) │            │  (NN)    │                │ eval   │
        └─────▲───────┘            └──────────┘                └───┬────┘
              │                                                    │
              └──────────────  keep net if it wins  ◄──────────────┘
```

- **Neural net** maps a board to a *policy* (which moves look good) and a
  *value* (who is winning).
- **MCTS** uses the net to search a few hundred positions deep and pick a move.
- **Self-play** generates training games using that search.
- **Train** updates the net to predict the search's choices and the game's
  outcome.
- **Arena** pits the new net against the old one; we only promote it if it
  actually plays better.

## Staged plan

| Stage | Game | Goal |
|-------|------|------|
| 0  | Tic-Tac-Toe | Validate the full pipeline learns (minutes) |
| 0.5| Connect 4   | Test MCTS+NN at small scale (hours) |
| 1  | Chess       | The real thing (days of self-play) |
| 2  | Chess       | Strength tuning + Elo tracking |

The AlphaZero core (`src/core`, `src/nn`) is written **once** against the
`Game` interface in `src/games/base.py`. Each stage just plugs in a new game.

## Setup

```bash
pip install -r requirements.txt
# PyTorch with CUDA — see requirements.txt (already installed on the dev machine)
```

## Run

```bash
python tests/test_games.py            # sanity-check Tic-Tac-Toe
python tests/test_chess.py            # sanity-check the chess adapter

python scripts/train_tictactoe.py     # Stage 0: converges to perfect play

python scripts/train_chess.py         # Stage 1: self-play chess (runs until stopped)
python scripts/train_chess.py --resume   # continue from models/latest.pt
python scripts/plot_metrics.py        # draw the learning curve -> docs/plots/
```

Training saves `models/latest.pt` every iteration and `models/best.pt` on each
arena promotion, so it is safe to Ctrl-C and `--resume`.

## Journal

Progress, decisions, and results are logged in [`docs/journal.md`](docs/journal.md).
