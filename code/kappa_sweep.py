"""
kappa_sweep.py -- the cross-modal alignment test of the paper (Fig. 4).

Joins the network, device and Linux labels on the nearest timestamp (network as
base, merge_asof direction='nearest', the mechanics fuse_datasets uses) at nine
tolerances from 1 s to 1 h, and reports match rates, raw agreement and Cohen's
kappa for each pair. It uses catf_ids.py's own loaders and preprocessing and the
pipeline's sampling rule (2,000 records per network file at --n 46000).
It trains nothing; expect a few minutes.

    python kappa_sweep.py --base "<folder with the Processed_*_dataset folders>"
"""
import argparse
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=46000,
                help="same meaning as the pipeline's --n (default 46000)")
ap.add_argument("--target", default="catf_ids.py",
                help="pipeline module to borrow loaders from")
ap.add_argument("--base", default=os.environ.get("CATF_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "TON_IoT", "Processed_datasets")),
                help="folder holding the Processed_*_dataset folders of TON_IoT")
args = ap.parse_args()

HERE   = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.join(HERE, args.target)
if not os.path.exists(TARGET):
    sys.exit(f"FAIL: {TARGET} not found. Put this script beside the frozen pipeline.")

# Import the pipeline as a module without running its __main__ block.
_saved_argv = sys.argv
sys.argv = [TARGET]                      # the pipeline reads sys.argv at import
spec = importlib.util.spec_from_file_location("ids_pipeline", TARGET)
ids  = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ids)
sys.argv = _saved_argv

for fn in ("load_multi", "preprocess_network", "preprocess_iot_multi",
           "preprocess_linux_multi"):
    if not hasattr(ids, fn):
        sys.exit(f"FAIL: {os.path.basename(TARGET)} has no {fn}() — wrong file?")

BASE = args.base
NET  = [os.path.join(BASE, "Processed_Network_dataset", f"Network_dataset_{i}.csv") for i in range(1, 24)]
IOT  = [os.path.join(BASE, "Processed_IoT_dataset", f"IoT_{n}.csv") for n in
        ("Fridge", "Garage_Door", "GPS_Tracker", "Modbus", "Motion_Light",
         "Thermostat", "Weather")]
LOG  = [os.path.join(BASE, "Processed_Linux_dataset", f"{n}.csv") for n in
        ("linux_disk_1", "linux_disk_2", "linux_memory1", "linux_memory2",
         "Linux_process_1", "Linux_process_2")]

# The pipeline's own sizing rule (run_pipeline_multi).
N_PER_FILE = max(500, args.n // max(len(NET), 1))

print("=" * 86)
print(f"  loaders from : {os.path.basename(TARGET)}")
print(f"  rows per file: {N_PER_FILE}   (pipeline rule at --n {args.n})")
print("=" * 86)

raw_net = ids.load_multi(NET, "Network",   n_per_file=N_PER_FILE)
raw_iot = ids.load_multi(IOT, "IoT",       n_per_file=N_PER_FILE)
raw_log = ids.load_multi(LOG, "Linux Log", n_per_file=N_PER_FILE)

net = ids.preprocess_network(raw_net)[["ts", "label"]].rename(columns={"label": "net_label"})
iot = ids.preprocess_iot_multi(raw_iot)[["ts", "label"]].rename(columns={"label": "iot_label"})
log = ids.preprocess_linux_multi(raw_log)[["ts", "label"]].rename(columns={"label": "log_label"})

for d in (net, iot, log):
    d.sort_values("ts", inplace=True)
    d.reset_index(drop=True, inplace=True)


def kappa(a, b):
    """Cohen's κ for two binary columns, plus the chance agreement it corrects for."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    p_o = float((a == b).mean())
    pa, pb = a.mean(), b.mean()
    p_e = pa * pb + (1 - pa) * (1 - pb)
    k = (p_o - p_e) / (1 - p_e) if (1 - p_e) > 1e-9 else float("nan")
    return k, p_o, p_e


print("\n" + "=" * 86)
print("  TIME-ALIGNMENT SWEEP — frozen-lineage loaders")
print("=" * 86)
print(f"  records after preprocessing — net {len(net):,}  iot {len(iot):,}  log {len(log):,}")
print(f"  source attack rates         — net {net['net_label'].mean():.1%}  "
      f"iot {iot['iot_label'].mean():.1%}  log {log['log_label'].mean():.1%}")
print(f"  ts ranges — net [{net['ts'].min():.0f}, {net['ts'].max():.0f}]  "
      f"iot [{iot['ts'].min():.0f}, {iot['ts'].max():.0f}]  "
      f"log [{log['ts'].min():.0f}, {log['ts'].max():.0f}]")

hdr = (f"\n  {'tol(s)':>7} {'rows':>7} {'kept':>6} | {'net%':>6} {'iot%':>6} {'log%':>6} |"
       f" {'n=i':>6} {'n=l':>6} {'i=l':>6} | {'all3':>6} |"
       f" {'κ n/l':>7} {'chance':>7} | {'κ n/i':>7} {'κ i/l':>7}")
print(hdr)
print("  " + "-" * 108)

rows = []
for tol in (1, 5, 15, 30, 60, 120, 300, 900, 3600):
    m = pd.merge_asof(net, iot, on="ts", direction="nearest", tolerance=tol)
    m = pd.merge_asof(m.sort_values("ts"), log, on="ts", direction="nearest", tolerance=tol)
    m = m.dropna(subset=["net_label", "iot_label", "log_label"])
    if len(m) < 50:
        print(f"  {tol:>7} {len(m):>7}   — too few rows to measure")
        rows.append((tol, len(m), None))
        continue
    k_nl, a_nl, e_nl = kappa(m["net_label"], m["log_label"])
    k_ni, a_ni, _    = kappa(m["net_label"], m["iot_label"])
    k_il, a_il, _    = kappa(m["iot_label"], m["log_label"])
    all3 = float(((m["net_label"] == m["iot_label"]) &
                  (m["net_label"] == m["log_label"])).mean())
    mark = "  <- pipeline tolerance" if tol == 300 else ""
    print(f"  {tol:>7} {len(m):>7} {len(m)/len(net):>5.1%} |"
          f" {m['net_label'].mean():>5.1%} {m['iot_label'].mean():>5.1%} {m['log_label'].mean():>5.1%} |"
          f" {a_ni:>5.1%} {a_nl:>5.1%} {a_il:>5.1%} | {all3:>5.1%} |"
          f" {k_nl:>7.3f} {e_nl:>6.1%} | {k_ni:>7.3f} {k_il:>7.3f}{mark}")
    rows.append((tol, len(m), (k_nl, k_ni, k_il, a_nl, e_nl)))

# ── the one-screen answer the paper needs ─────────────────────────────
valid = [(t, n, r) for t, n, r in rows if r is not None]
print("\n" + "=" * 86)
print("  PASTE THIS BLOCK BACK")
print("=" * 86)
if not valid:
    print("  no tolerance produced enough rows — the claim cannot be checked this way")
else:
    knl = [r[0] for _, _, r in valid]
    tmax = max(valid, key=lambda x: x[2][0])
    print(f"  tolerances measured          : {len(valid)} of 9")
    print(f"  κ net/host, min .. max        : {min(knl):.3f} .. {max(knl):.3f}"
          f"   (max at {tmax[0]} s)")
    tight = [r[0] for t, _, r in valid if t <= 30]
    loose = [r[0] for t, _, r in valid if t >= 300]
    if tight and loose:
        print(f"  κ net/host, mean ≤30 s vs ≥300 s: {np.mean(tight):.3f} vs {np.mean(loose):.3f}")
    r300 = next((r for t, _, r in valid if t == 300), None)
    if r300:
        print(f"  at 300 s (pipeline)           : κ n/l {r300[0]:.3f}, raw agreement "
              f"{r300[3]:.1%} against chance {r300[4]:.1%}")
    print("\n  How §5.7 reads against this:")
    if max(knl) < 0.20 and (not tight or not loose or np.mean(tight) - np.mean(loose) < 0.10):
        print("   'agreement does not recover at any tolerance' — SUPPORTED: κ stays low")
        print("   everywhere and tightening the join does not raise it materially.")
    else:
        print("   'agreement does not recover at any tolerance' — NOT CLEARLY SUPPORTED.")
        print("   κ rises somewhere or tightening helps. §1, §5.7 and §9 need rewording")
        print("   before §8 is written. Send me the full table.")

# ── the consensus rule, as the pipeline actually applies it ───────────
print("\n" + "=" * 86)
print("  CONSENSUS RULE — enumerated (pipeline weights, threshold 0.45)")
print("=" * 86)
for name, W in (("3 sources (no Windows)", {"net": .45, "iot": .20, "log": .20}),
                ("4 sources (as the pipeline votes)", {"net": .45, "iot": .20, "log": .20, "win": .15})):
    keys = list(W)
    same = flips = 0
    for bits in range(1 << len(keys)):
        v = {k: (bits >> i) & 1 for i, k in enumerate(keys)}
        final = int(sum(W[k] * v[k] for k in keys) >= 0.45)
        same += int(final == v["net"])
        flips += int(v["net"] == 0 and final == 1)
    print(f"  {name:<36} final == net label in {same}/{1 << len(keys)} combinations;"
          f" net-normal records turned attack in {flips}")
print("  With four sources, a net-normal record becomes attack only when device, Linux")
print("  and Windows all say attack — the mechanism behind §4.4's 4,504 relabellings.")
