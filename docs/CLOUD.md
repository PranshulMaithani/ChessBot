# Cloud training guide — Kaggle & Colab

Train on Kaggle (30h GPU / week, ~9h per session) and Colab (~12h per session)
while the laptop does play / iteration. Same code, just a different launcher.

## One-time setup: push to GitHub

The cloud notebooks `git clone` the repo. Push yours first — private is fine:

```powershell
gh repo create chesseng --private --source . --push
# or, manually:
#   git remote add origin https://github.com/<you>/chesseng.git
#   git push -u origin master
```

Replace `<you>/chesseng` below with your actual repo path.

---

## Kaggle (T4 / P100, 30h/week, ~9h per session)

1. **New Notebook** → right panel:
   - **Accelerator**: GPU T4 x2 (or any GPU available).
   - **Internet**: **ON** (needed for `git clone` and `pip install`).
2. Paste these cells:

   **Setup (cell 1):**
   ```bash
   !nvidia-smi
   !git clone https://github.com/<you>/chesseng.git
   %cd chesseng
   !pip install -q chess tqdm matplotlib
   ```

   **Train (cell 2):**
   ```bash
   !python scripts/train_chess.py \
       --sims 200 --mcts-batch 32 \
       --channels 96 --res-blocks 8 \
       --games 32 --arena 16 \
       --resign-threshold -0.85
   ```

3. When the session ends:
   - Files in `/kaggle/working/chesseng/models/chess/` and `runs/` are
     downloadable from the right-side **Output** panel.
   - For the next session, upload them back to the same paths and add
     `--resume`. Or push them to a Kaggle Dataset and attach it.

**Quick wallclock estimate**: T4 + the config above ≈ **5–8 min/iter**, so
~70-100 iters per 9h session — vs ~25 iters/9h on the laptop.

---

## Colab (T4 free, ~12h/session)

Colab has a major advantage: **mount Google Drive** so checkpoints survive
across sessions automatically — no upload/download dance.

1. New Notebook → **Runtime → Change runtime type → GPU**.
2. Cells:

   **Mount Drive (once per session):**
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   !mkdir -p /content/drive/MyDrive/Chesseng/models/chess
   !mkdir -p /content/drive/MyDrive/Chesseng/runs
   ```

   **Setup + symlink so saves land on Drive:**
   ```bash
   %cd /content
   !rm -rf chesseng
   !git clone https://github.com/<you>/chesseng.git
   %cd chesseng
   !pip install -q chess tqdm matplotlib
   !rm -rf models runs
   !ln -sfn /content/drive/MyDrive/Chesseng/models models
   !ln -sfn /content/drive/MyDrive/Chesseng/runs   runs
   ```

   **Train (resume from Drive if anything's there):**
   ```bash
   !python scripts/train_chess.py \
       --sims 200 --mcts-batch 32 \
       --channels 96 --res-blocks 8 \
       --games 32 --arena 16 \
       --resign-threshold -0.85 \
       --resume
   ```

When Colab disconnects, just open the notebook again and rerun the cells —
weights are on Drive, training picks up where it left off.

---

## Sync back to the laptop

After cloud training, pull the checkpoints down:

- **Kaggle**: download `models/chess/best.pt` + `runs/chess_metrics.csv`
  from the Output panel.
- **Colab**: they're already on your Drive; just copy locally or use Drive
  for Desktop sync.

Drop them into `models/chess/` and `runs/` on the laptop, then:

```powershell
python scripts/play.py                       # play the cloud-trained bot
python scripts/train_chess.py --resume       # keep refining locally
python scripts/plot_metrics.py --csv runs/chess_metrics.csv
```

---

## Recommended cloud config for chess

```
--sims 200          # depth: stronger play, decisive games
--mcts-batch 32     # parallel sims per GPU forward — the big speedup
--channels 96       # net width (was 64 on laptop)
--res-blocks 8      # net depth (was 6)
--games 32          # self-play games per iter
--arena 16
--resign-threshold -0.85   # kills the shuffle-to-draw plateau
```

If you have an A100 (Colab paid), bump `--channels 128 --res-blocks 12
--mcts-batch 64` for a bigger model that still fits in VRAM.

---

## Strategy across the three environments

| Where | What it's best at |
|---|---|
| **Laptop (RTX 3070, 8GB)** | playing the bot, iterating on code, small smokes |
| **Colab (T4, Drive)** | uninterrupted long runs — Drive auto-syncs |
| **Kaggle (T4/P100, 30h/wk)** | bursty heavy compute when you've got the time budget |

Recommendation: keep one canonical `models/chess/best.pt` you trust, drive
improvements from whichever environment is most convenient at the time, and
sync back through your repo / Drive.
