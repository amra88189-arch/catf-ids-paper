#!/usr/bin/env python3
"""
make_fig_kappa.py — the κ figure (§5.7), from the verified sweep only.

Every number below is copied from CATF-IDS_Kappa_Sweep_fsm3.md, which records
the diag_align_fsm3.py run on the frozen loaders at 2,000 rows per file. No
value is computed from anything else. Intervals and κ_max are derived from the
printed agreement and marginal rates by the formulas in that file.

Writes CATF-IDS_Fig_kappa_sweep.pdf (vector, for the manuscript) and .png.
"""
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# tol, rows, net%, dev%, host%, agree n=d, agree n=h, agree d=h, κ n/h, κ n/d, κ d/h
SWEEP = [
    (1,     58,    .776, .655, .948, .638, .793, .603, .181,  .126, -.099),
    (5,     710,   .690, .720, .970, .582, .694, .704, .048, -.004, -.009),
    (15,    4053,  .613, .718, .949, .555, .626, .721, .061,  .012,  .082),
    (30,    8906,  .593, .680, .935, .528, .610, .685, .070, -.012,  .083),
    (60,    15233, .566, .621, .902, .498, .590, .647, .082, -.037,  .123),
    (120,   20702, .554, .542, .843, .480, .579, .608, .090, -.049,  .169),
    (300,   28581, .533, .432, .779, .481, .579, .559, .126, -.028,  .180),
    (900,   35483, .512, .374, .721, .491, .573, .555, .137, -.011,  .199),
    (3600,  38244, .510, .348, .687, .492, .578, .568, .150, -.009,  .225),
]
NET_ROWS = 45500


def chance(a, b):
    return a * b + (1 - a) * (1 - b)


def ci95(p_o, p_e, n):
    return 1.96 * math.sqrt(p_o * (1 - p_o) / (n * (1 - p_e) ** 2))


def kmax(a, b):
    p_e = chance(a, b)
    return ((1 - abs(a - b)) - p_e) / (1 - p_e)


tol = [r[0] for r in SWEEP]
k_nh = [r[8] for r in SWEEP]
k_nd = [r[9] for r in SWEEP]
k_dh = [r[10] for r in SWEEP]
e_nh = [ci95(r[6], chance(r[2], r[4]), r[1]) for r in SWEEP]
e_nd = [ci95(r[5], chance(r[2], r[3]), r[1]) for r in SWEEP]
kmax_nh = [kmax(r[2], r[4]) for r in SWEEP]
kept = [100 * r[1] / NET_ROWS for r in SWEEP]

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8,
    "axes.titlesize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "axes.linewidth": 0.6, "lines.linewidth": 1.0,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.0, 2.6),
                             gridspec_kw={"width_ratios": [2.1, 1]})

# ── (a) κ against tolerance ────────────────────────────────────────────
ax.axhspan(-0.2, 0.20, color="0.93", zorder=0)
ax.text(4000, -0.185, "shaded: slight agreement or less (κ < 0.20)", fontsize=6.5, color="0.45", ha="right", va="bottom")
ax.axhline(0, color="0.55", lw=0.5, zorder=1)
ax.axvline(300, color="0.55", lw=0.5, ls=":", zorder=1)
ax.text(300 * 0.92, -0.115, "pipeline tolerance", fontsize=6.5, color="0.35", ha="right", va="bottom")

ax.plot(tol, kmax_nh, color="0.55", ls="--", lw=0.8, label=r"$\kappa_{\max}$, network/host")
ax.errorbar(tol, k_nh, yerr=e_nh, color="black", marker="o", ms=3.5, capsize=2,
            lw=1.2, elinewidth=0.7, label="network / host (primary)", zorder=4)
ax.errorbar(tol, k_nd, yerr=e_nd, color="0.35", marker="s", ms=3, capsize=1.5,
            lw=0.8, elinewidth=0.5, ls="-.", label="network / device", zorder=3)
ax.plot(tol, k_dh, color="0.35", marker="^", ms=3.2, lw=0.8, ls=":",
        label="device / host", zorder=3)
ax.text(1.12, 0.47, "n = 58", fontsize=6.5, color="0.3")

ax.set_xscale("log")
ax.set_xticks(tol)
ax.set_xticklabels(["1", "5", "15", "30", "60", "120", "300", "900", "3600"])
ax.set_xlim(0.8, 4500)
ax.set_ylim(-0.2, 0.8)
ax.set_xlabel("merge tolerance (s)")
ax.set_ylabel(r"Cohen's $\kappa$")
ax.set_title("(a) label agreement after the join, corrected for chance", loc="left")
ax.legend(loc="upper left", bbox_to_anchor=(0.13, 1.0), frameon=False, ncol=2,
          columnspacing=1.2, handlelength=2.2)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

# ── (b) what each join keeps ──────────────────────────────────────────
bx.plot(tol, kept, color="black", marker="o", ms=3)
bx.set_xscale("log")
bx.set_yscale("log")
bx.set_xticks([1, 10, 100, 1000])
bx.set_xticklabels(["1", "10", "100", "1000"])
bx.set_xlim(0.8, 4500)
bx.set_ylim(0.05, 150)
bx.set_yticks([0.1, 1, 10, 100])
bx.set_yticklabels(["0.1", "1", "10", "100"])
bx.axvline(300, color="0.55", lw=0.5, ls=":")
bx.set_xlabel("merge tolerance (s)")
bx.set_ylabel("network records retained (%)")
bx.set_title("(b) rows surviving the join", loc="left")
for s in ("top", "right"):
    bx.spines[s].set_visible(False)

fig.tight_layout(w_pad=2.0)
fig.savefig("CATF-IDS_Fig_kappa_sweep.pdf", bbox_inches="tight")
fig.savefig("CATF-IDS_Fig_kappa_sweep.png", dpi=300, bbox_inches="tight")
print("wrote CATF-IDS_Fig_kappa_sweep.pdf / .png")
for t, a, b in zip(tol, k_nh, kmax_nh):
    print(f"  {t:>5} s  κ n/h {a:+.3f}  κmax {b:.3f}")
