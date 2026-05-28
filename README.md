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
python tests/test_games.py             # sanity-check Tic-Tac-Toe
python tests/test_connect4.py          # sanity-check Connect 4
python tests/test_chess.py             # sanity-check the chess adapter (incl. MCTS regression)

python scripts/train_tictactoe.py      # Stage 0   : converges to perfect play (minutes)
python scripts/train_connect4.py       # Stage 0.5 : decisive learning signal (hours)
python scripts/train_chess.py          # Stage 1   : self-play chess (runs until stopped)

# All trainers support:
#   --resume         continue from models/<game>/latest.pt
#   --smoke          tiny end-to-end sanity run
python scripts/plot_metrics.py --csv runs/connect4_metrics.csv \
                               --out docs/plots/connect4_progress.png
python scripts/play.py                 # play a game vs the trained chess bot
python scripts/play_connect4.py        # play a game vs the trained Connect 4 bot
```

## Cloud training (Kaggle / Colab)

See [`docs/CLOUD.md`](docs/CLOUD.md) for Kaggle (30h/week T4) and Colab
(~12h/session, Drive-persisted) setup. Recommended cloud config bumps
`--sims 200 --mcts-batch 32 --channels 96 --res-blocks 8` for ~5x faster
iterations on the better GPUs.

Each trainer saves `models/<game>/latest.pt` every iteration and
`models/<game>/best.pt` on each arena promotion, so runs are safe to Ctrl-C
and `--resume`. Per-iteration metrics land in `runs/<game>_metrics.csv`.

## Journal

Progress, decisions, and results are logged in [`docs/journal.md`](docs/journal.md).
