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
python tests/test_games.py      # sanity-check the game implementations
```
(more entry points added as stages land)

## Journal

Progress, decisions, and results are logged in [`docs/journal.md`](docs/journal.md).
