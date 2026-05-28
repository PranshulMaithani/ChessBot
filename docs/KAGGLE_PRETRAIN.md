# Pretrain Chess on Kaggle — copy-paste cells (128ch × 10 blocks)

Walks through pretraining the chess net on a Lichess Elite PGN via a free
Kaggle GPU session. Run this once; download the resulting `pretrained.pt`;
then continue with self-play locally via `--init-from`.

> **Architecture: 128 channels × 10 residual blocks.** This is the
> "use my 8 GB VRAM" preset — substantially stronger ceiling than the older
> 96×8 default. If you want the smaller 96×8 net (faster iters, lower
> ceiling), swap `--channels 96 --res-blocks 8` in Cells 3.5 and 4 and use
> `--batch-size 512` instead of `384`.

## Before you start

1. **Push your latest commits from the laptop** so Kaggle clones up-to-date code:
   ```powershell
   git push
   ```
2. New Kaggle notebook → https://kaggle.com/code → **+ New Notebook**.
3. Right panel:
   - **Accelerator** → GPU T4 x2 (or any GPU).
   - **Internet** → ON  (required for `git clone` and PGN download).

Paste each block below into its own notebook cell, in order. Run them top-to-bottom.

---

## Cell 1 — verify GPU is attached  *(Code)*

```python
!nvidia-smi
```

You should see a `Tesla T4` or `P100`. If you see nothing, the accelerator setting didn't take — flip it and reload.

---

## Cell 2 — clone repo + install deps  *(Code)*

```python
%cd /kaggle/working
!git clone https://github.com/PranshulMaithani/ChessBot.git
%cd ChessBot
!pip install -q -r requirements.txt
```

---

## Cell 3 — download Lichess Elite database  *(Code, ~30–90 s)*

```python
!mkdir -p data
!curl -fL -o data/elite.zip "https://database.nikonoel.fr/lichess_elite_2024-10.zip"
!cd data && unzip -o elite.zip && rm elite.zip
!ls -lh data/
```

If the download 404s, that month isn't hosted anymore — open
https://database.nikonoel.fr/ in your browser, pick any month listed, and
swap the year-month in the URL above. Then adjust `--pgn` in Cells 3.5 and 4
to match the extracted `.pgn` filename.

---

## (Optional) Cell 3.5 — quick smoke before the long run  *(Code, ~3 min)*

```python
!python scripts/pretrain_chess.py \
    --pgn data/lichess_elite_2024-10.pgn \
    --max-positions 30000 --epochs 1 \
    --channels 128 --res-blocks 10 \
    --batch-size 256 \
    --out models/chess/_smoke.pt
```

Use this to confirm the pipeline runs end-to-end before committing 90+ min
to the real run. Delete the smoke checkpoint afterwards:
`!rm models/chess/_smoke.pt`.

If you see OOM here, drop `--batch-size` to `192`.

---

## Cell 4 — pretrain  *(Code, ~60–120 min on T4)*

```python
!python scripts/pretrain_chess.py \
    --pgn data/lichess_elite_2024-10.pgn \
    --max-positions 800000 --epochs 4 \
    --channels 128 --res-blocks 10 \
    --batch-size 384
```

`--batch-size 384` is the safe-margin value for 128×10 on T4 (15 GB VRAM).
If you get OOM, drop to 256. If VRAM utilisation stays well under 10 GB on
the first few batches, you can bump to 512 for ~25% faster epochs.

You'll see per-epoch lines like:

```
epoch 1/4  train pi=4.21 v=0.61 top1=0.18  |  val pi=4.05 v=0.55 top1=0.21  (480.5s)
```

What to watch:
- **`top1`** = "how often net's argmax matches the actually-played master move."
  Starts ~0.07 (random over 14 legal moves), should climb past **0.42** by the
  last epoch with the bigger 128×10 net (vs ~0.40 for 96×8).
- **`pi`** = policy cross-entropy — should drop ~4.5 → ~2.4.
- **`v`** = value MSE — should drop ~0.6 → ~0.32.
- **`val` columns** track held-out positions — `val` close to `train` means
  no overfitting, `val` >> `train` means too many epochs.
- Each epoch on T4 takes ~15-25 min at this size. If you see >35 min/epoch,
  your batch is starving the GPU — bump `--batch-size` if VRAM allows.

---

## Cell 5 — verify the checkpoint  *(Code)*

```python
!ls -lh models/chess/
```

Expect `pretrained.pt` around **~45-50 MB** at the 128×10 config (state-dict
only — no optimizer state saved).

---

## (Optional) Cell 6 — sanity-check: predict on the start position  *(Code)*

```python
import sys, torch
sys.path.insert(0, '/kaggle/working/ChessBot')
from src.config import Config
from src.games import ChessGame
from src.nn.net import NeuralNet

cfg = Config(num_channels=128, num_res_blocks=10, device='cuda',
             checkpoint_dir='models/chess')
game = ChessGame()
net = NeuralNet(game, cfg)
net.load_checkpoint('pretrained.pt')

p, v = net.predict(game.get_init_state())
top5 = sorted(range(len(p)), key=lambda i: -p[i])[:5]
print("value of start position:", round(v, 3))
print("top-5 actions (index, prob):", [(i, round(float(p[i]), 3)) for i in top5])
```

A well-pretrained net gives **`value ≈ 0`** (start is balanced) and puts most
mass on a handful of common opening moves (the indices for `1.e4`, `1.d4`,
`1.Nf3`, `1.c4`).

---

## When training finishes — download the checkpoint

1. Click **Save Version → Save & Run All** at the top right of Kaggle (this
   finalises outputs).
2. After the version saves, the right-side panel → **Output** lists files.
   Find `models/chess/pretrained.pt` and download.
3. On the laptop, place it at `models/chess/pretrained.pt` (overwriting any
   older 96×8 file).
4. Delete or rename the old self-play checkpoints — they're 96×8 and
   incompatible with the new 128×10 net:
   ```powershell
   cd D:\GoodProjects\Chesseng\models\chess
   Rename-Item best.pt best_96x8.pt -ErrorAction SilentlyContinue
   Rename-Item latest.pt latest_96x8.pt -ErrorAction SilentlyContinue
   Remove-Item temp.pt -ErrorAction SilentlyContinue
   ```
5. Continue with self-play, matching the net architecture you pretrained:
   ```powershell
   python scripts\train_chess.py `
       --init-from models\chess\pretrained.pt `
       --channels 128 --res-blocks 10 `
       --sims 120 --mcts-batch 32 `
       --workers 4 --worker-device cuda `
       --games 16
   ```
   The `--channels` / `--res-blocks` flags **must match** the pretrain
   config or `load_state_dict` will refuse with a shape mismatch. No
   `--resume` on the first run — that flag looks for `latest.pt`, which
   you just renamed.

---

## Tips for multi-session pretraining

Kaggle sessions auto-kill at 9 h. If you want longer/more epochs:

- Re-run Cell 4 with a different `--max-positions` and a different `--out`
  path in subsequent sessions (e.g. `--out models/chess/pretrained_v2.pt`).
- Or: bump `--epochs` in a single session — most gain comes in the first 3-4
  epochs anyway. Diminishing returns past that.

The 30 h weekly Kaggle quota easily covers a 128×10 pretrain (~2 h) plus
several follow-up sessions. Use the rest of the quota for **self-play
continuation** if you don't want to run it locally — see
`docs/KAGGLE_TRAIN.md` if/when that's written.
