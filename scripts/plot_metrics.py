"""Plot training metrics from a coach CSV into a PNG for the journal.

    python scripts/plot_metrics.py                       # chess defaults
    python scripts/plot_metrics.py --csv runs/foo.csv --out docs/plots/foo.png
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402


def load(csv_path):
    rows = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            extra = json.loads(row.get("extra") or "{}")
            rows.append((row, extra))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="runs/chess_metrics.csv")
    ap.add_argument("--out", default="docs/plots/chess_progress.png")
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        raise SystemExit(f"no metrics yet at {args.csv} — run training first")
    rows = load(args.csv)
    iters = [int(r["iter"]) for r, _ in rows]
    pi = [float(r["pi_loss"]) for r, _ in rows]
    v = [float(r["v_loss"]) for r, _ in rows]

    # pull "vs_random[W/L/D]" win-rate from the extra column if present
    win_rate = []
    for _, extra in rows:
        wld = extra.get("vs_random[W/L/D]")
        if wld and sum(wld) > 0:
            win_rate.append(wld[0] / sum(wld))
        else:
            win_rate.append(None)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(iters, pi, label="policy loss")
    ax1.plot(iters, v, label="value loss")
    ax1.set(xlabel="iteration", ylabel="loss", title="Training loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    if any(w is not None for w in win_rate):
        xs = [i for i, w in zip(iters, win_rate) if w is not None]
        ys = [w for w in win_rate if w is not None]
        ax2.plot(xs, ys, marker="o", color="green")
        ax2.set(xlabel="iteration", ylabel="win rate", ylim=(0, 1),
                title="Win rate vs random")
        ax2.grid(alpha=0.3)
    else:
        ax2.set_axis_off()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=110)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
