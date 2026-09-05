#!/usr/bin/env python3
"""Plot training curves from trainer_state.json.

    python code/plot_training.py --state private/adapter/checkpoint-N/trainer_state.json \\
        --out results/training_curves.png

The input holds metrics only (loss, accuracy, learning rate) — no training
data — so the resulting figure is safe to commit.
"""
import argparse, json, pathlib, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "common"))
from plotstyle import apply, NV

import pathlib as _pl, sys as _sys
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[3] / "common"))
from cli import step, ok, info, warn, die, kv, done, Timer  # noqa: E402



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="Day 00 · LoRA r=16 all-linear, Qwen3.5-9B, 169 samples")
    a = ap.parse_args()

    st = json.load(open(a.state))
    rows = [r for r in st["log_history"] if "loss" in r and "epoch" in r]
    ep = [r["epoch"] for r in rows]
    loss = [r["loss"] for r in rows]
    acc = [r.get("mean_token_accuracy") for r in rows]
    lr = [r.get("learning_rate") for r in rows]

    apply()
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))

    ax = axes[0]
    ax.plot(ep, loss, marker="o", color=NV["green"])
    ax.set_xlabel("epoch"); ax.set_ylabel("train loss"); ax.set_title("Loss")
    ax.annotate(f"{loss[0]:.2f}", (ep[0], loss[0]), textcoords="offset points", xytext=(6, 6), fontsize=9, color=NV["gray"])
    ax.annotate(f"{min(loss):.2f}", (ep[loss.index(min(loss))], min(loss)), textcoords="offset points", xytext=(6, -12), fontsize=9, color=NV["gray"])

    ax = axes[1]
    if all(v is not None for v in acc):
        ax.plot(ep, acc, marker="o", color=NV["dark"])
    ax.set_xlabel("epoch"); ax.set_ylabel("mean token accuracy"); ax.set_title("Token accuracy")
    ax.set_ylim(0, 1)

    ax = axes[2]
    if all(v is not None for v in lr):
        ax.plot(ep, lr, marker="o", color=NV["blue"])
    ax.set_xlabel("epoch"); ax.set_ylabel("learning rate"); ax.set_title("LR schedule (3-step warmup, cosine)")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    fig.suptitle(a.title, x=0.01, ha="left", fontsize=12, fontweight="bold", color=NV["dark"])
    fig.tight_layout()
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out)
    step("Training curves plotted")
    kv("logged points", len(rows))
    kv("train loss", f"{loss[0]:.3f} → {loss[-1]:.3f}", f"(min {min(loss):.3f})")
    info(f"written to {a.out}")


if __name__ == "__main__":
    main()
