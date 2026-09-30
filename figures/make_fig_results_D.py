#!/usr/bin/env python3
"""Draw the result figures of Section 6 from the definitive run (configuration D).

    python make_fig_results_D.py     # needs matplotlib

  Fig. 4  CATF-IDS_Fig4_confusion.{pdf,png,svg}   confusion matrices, CV pooled and stream
  Fig. 6  CATF-IDS_Fig6_injection.{pdf,png,svg}   injected jumps caught, by size

Every number is copied from CATF-IDS_Definitive_Run_D.md (final_run.py, catf_ids.py
and policy_study.py outputs, all checks passed). Change a number here and re-run.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
NAVY, BLUE, RED, GREY, INK = "#1F4D78", "#4F81BD", "#C0504D", "#7F7F7F", "#222222"
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
    "axes.edgecolor": INK, "axes.linewidth": 0.8, "savefig.dpi": 300,
})
CM = LinearSegmentedColormap.from_list("navy", ["#FFFFFF", "#DCE6F1", "#4F81BD", NAVY])


def save(fig, stem):
    for ext in ("pdf", "png", "svg"):
        fig.savefig(os.path.join(HERE, f"{stem}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {stem}.pdf / .png / .svg")


# ── Fig. 4: confusion matrices ────────────────────────────────────────
# rows: true normal, true attack; columns: predicted normal, predicted attack
PANELS = [("(a) Cross-validation, five folds pooled\n13,256 records (balanced)",
           [[5495, 1133], [213, 6415]]),
          ("(b) Full stream\n24,934 records (73% attack)",
           [[5456, 1172], [657, 17649]])]


def fig4():
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.9))
    for ax, (title, m) in zip(axes, PANELS):
        share = [[v / sum(row) for v in row] for row in m]
        ax.imshow(share, cmap=CM, vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                dark = share[i][j] > 0.55
                col = "white" if dark else INK
                ax.text(j, i - 0.10, f"{m[i][j]:,}", ha="center", va="center",
                        fontsize=11, fontweight="bold", color=col)
                ax.text(j, i + 0.20, f"{100 * share[i][j]:.1f}% of row", ha="center",
                        va="center", fontsize=8, color=col)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["normal", "attack"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["normal", "attack"])
        ax.set_xlabel("predicted"); ax.set_ylabel("fused label")
        ax.set_title(title, pad=6)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xticks([0.5], minor=True); ax.set_yticks([0.5], minor=True)
        ax.grid(which="minor", color="white", linewidth=2)
        ax.tick_params(which="both", length=0)
    fig.tight_layout(w_pad=2.5)
    save(fig, "CATF-IDS_Fig4_confusion")


# ── Fig. 6: injected device-reading jumps caught that nothing else flags ─
SIZES = [0.5, 1, 2, 5]
ARMS = [("calibrated limits (the evaluated system)", [65.4, 82.1, 82.1, 82.1], NAVY, "o", "-"),
        ("original fixed conditions", [42.3, 28.2, 19.2, 62.8], RED, "s", "--"),
        ("learned path alone (no forced alerts)", [34.6, 24.4, 19.2, 19.2], GREY, "^", ":")]


def fig6():
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for lab, ys, col, mk, ls in ARMS:
        ax.plot(SIZES, ys, ls, color=col, marker=mk, markersize=5, linewidth=1.6, label=lab)
        below = col == GREY                      # the learned path's labels go underneath
        for x, y in zip(SIZES, ys):
            ax.annotate(f"{y:.0f}", (x, y), textcoords="offset points",
                        xytext=(0, -12 if below else 6), ha="center", fontsize=7.5, color=col)
    ax.set_xscale("log")
    ax.set_xticks(SIZES); ax.set_xticklabels([f"{s:g}×" for s in SIZES])
    ax.minorticks_off()
    ax.set_xlim(0.4, 6.3); ax.set_ylim(0, 100)
    ax.set_xlabel("size of the jump (× the device's normal range)")
    ax.set_ylabel("injected records caught (%)")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#E3E3E3", linewidth=0.6)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), frameon=False, ncol=1,
              handlelength=2.6)
    fig.tight_layout()
    save(fig, "CATF-IDS_Fig6_injection")


if __name__ == "__main__":
    fig4()
    fig6()
