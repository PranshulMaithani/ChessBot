# Datasets for supervised pretraining

`scripts/pretrain_chess.py` reads standard PGN. Any source works; the
recommended one is the **Lichess Elite Database** — curated Lichess games
where both players are rated 2400+.

## Lichess Elite Database (recommended)

https://database.nikonoel.fr/

Monthly dumps, ~1–3 GB compressed each (~1–3M games per month).
You do **not** need millions of positions for pretraining to help massively;
`--max-positions 500000` is plenty (≈1–3 % of one month).

PowerShell:
```powershell
mkdir -p data
cd data
# pick any recent month; example: October 2024
Invoke-WebRequest -Uri "https://database.nikonoel.fr/lichess_elite_2024-10.zip" `
                  -OutFile "lichess_elite_2024-10.zip"
Expand-Archive lichess_elite_2024-10.zip -DestinationPath .
cd ..
```

Bash/git-bash:
```bash
mkdir -p data && cd data
curl -L -o lichess_elite_2024-10.zip \
    https://database.nikonoel.fr/lichess_elite_2024-10.zip
unzip lichess_elite_2024-10.zip
cd ..
```

That gives you `data/lichess_elite_2024-10.pgn` (or similar).

## Alternative: full Lichess database

For the largest possible pool: https://database.lichess.org/.
Files are huge (50+ GB). Apply `--min-elo 2200` aggressively.

## Alternative: CCRL engine-vs-engine

http://ccrl.chessdom.com/ccrl/ — engine matches at deep search.
Smaller, very high quality, mostly decisive.

## Tip — pretrain on Kaggle

Pretraining is data-heavy and benefits from a faster GPU. The optimal use
of your weekly Kaggle quota is:

1. Upload the PGN as a Kaggle dataset (or `wget` it inside the notebook).
2. Run `pretrain_chess.py` there (Kaggle's T4 → ~30–60 min for 500k
   positions × 3 epochs).
3. Download the resulting `models/chess/pretrained.pt`.
4. Continue with `train_chess.py --init-from models/chess/pretrained.pt`
   wherever (laptop, Colab, another Kaggle session).
