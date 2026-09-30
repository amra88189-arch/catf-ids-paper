#!/usr/bin/env python3
"""Fig. 5: the two thresholds over alternating attack-heavy / normal-heavy blocks,
adaptive vs frozen, from the threshold stress test of configuration D.

    python make_fig_threshold_trace.py
    python make_fig_threshold_trace.py --events threshold_stress_D/stress_events.csv.gz
                                       --schedule waves_medium --n 600 --path stream

Reads the per-event file threshold_stress.py wrote (nothing is re-run) and writes
CATF-IDS_Fig5_threshold_trace.{pdf,png,svg} next to this script.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
NAVY, RED, INK, SHADE = "#1F4D78", "#C0504D", "#222222", "#DCE6F1"
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "legend.fontsize": 8, "axes.edgecolor": INK, "axes.linewidth": 0.8, "savefig.dpi": 300,
})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default=os.path.join(HERE, "threshold_stress_D", "stress_events.csv.gz"))
    ap.add_argument("--schedule", default="waves_medium")
    ap.add_argument("--path", default="stream", choices=["stream", "cv"])
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--floor", type=float, default=0.35)
    a = ap.parse_args()
    if not os.path.exists(a.events):
        raise SystemExit(f"{a.events} not found: run threshold_stress.py first")
    ev = pd.read_csv(a.events)
    g = ev[(ev["path"] == a.path) & (ev["schedule"] == a.schedule)]
    if a.path == "cv":
        g = g[g["fold"] == 1]
    if not len(g):
        raise SystemExit(f"no events for path={a.path} schedule={a.schedule}")

    fig, ax = plt.subplots(figsize=(6.3, 2.7))
    ad = g[g["arm"] == "adaptive"].sort_values("pos").head(a.n)
    fr = g[g["arm"] == "frozen"].sort_values("pos").head(a.n)
    x = ad["pos"].to_numpy()
    ax.fill_between(x, 0, ad["share"].to_numpy(), step="post", color=SHADE, linewidth=0,
                    label="attack share (target)")
    ax.axhline(a.floor, color="#7F7F7F", linewidth=0.8, linestyle=":")
    ax.text(x[-1], a.floor - 0.018, f"floor {a.floor:.2f}", ha="right", va="top",
            fontsize=7.5, color="#555555")
    ax.plot(ad["pos"], ad["tau_h"], color=NAVY, linewidth=1.3, label=r"$\tau_{high}$, adaptive")
    ax.plot(fr["pos"], fr["tau_h"], color=NAVY, linewidth=1.1, linestyle="--",
            label=r"$\tau_{high}$, frozen")
    ax.plot(ad["pos"], ad["gate"], color=RED, linewidth=1.1,
            label=r"$\tau_{low}$ at the escalation gate, adaptive")
    ax.plot(fr["pos"], fr["gate"], color=RED, linewidth=1.0, linestyle="--",
            label=r"$\tau_{low}$ at the escalation gate, frozen")
    ax.set_xlim(x[0], x[-1]); ax.set_ylim(0, 1)
    ax.set_xlabel("event position in the reordered stream")
    ax.set_ylabel("threshold / attack share")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, frameon=False,
              handlelength=2.4, columnspacing=1.6)
    fig.tight_layout()
    for ext in ("pdf", "png", "svg"):
        fig.savefig(os.path.join(HERE, f"CATF-IDS_Fig5_threshold_trace.{ext}"), bbox_inches="tight")
    print("wrote CATF-IDS_Fig5_threshold_trace.pdf / .png / .svg")
    n_forced = int(ad["tau_h"].isna().sum())
    if n_forced:
        print(f"  ({n_forced} forced alerts in the window leave gaps: they never reach the thresholds)")


if __name__ == "__main__":
    main()
