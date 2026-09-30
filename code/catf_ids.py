"""
CATF-IDS — Context-Adaptive Threat Fusion for multi-modal intrusion detection
in industrial IoT. Single-file implementation behind the paper.

Running
-------
    python catf_ids.py              # the final system (configuration D)
    python catf_ids.py --config_c   # the system as first evaluated (configuration C)
    python catf_ids.py --test       # self-test on synthetic data, no CSVs needed
    python catf_ids.py --serve      # REST API, after a run has saved its models

Data paths are set in the __main__ block (BASE and the file lists).

With no switches the run is configuration D: configuration C below with all
nine stages acting in the evaluation -- trust as an average of each modality's
agreement (stage 4); a policy engine whose envelopes and limits are set from
the site's own normal training records, per field and per device (stage 5);
cross-validation deciding every event exactly as the stream does, with the
stream feeding the policy pressure to the energy state machine as
cross-validation does (stages 6-8). Each change has an off switch:
    --trust_legacy  --no_envelopes  --legacy_policy_limits  --legacy_decision_path
    --cv_online_retrain (the stream's ensemble: see FROZEN_ENSEMBLE)
and --config_c turns all of them off.

--config_c is the paper's configuration C: three modalities;
2,000 records sampled per network file (--n 46000); the deployed decision rule
in cross-validation and in the stream; fixed conditions and experience cache
disabled; escalation on; tau_high floor 0.35; the stream scored by the
AUC-weighted ensemble of the five cross-validation fusion models.

Switches — each reproduces a configuration reported in the paper
----------------------------------------------------------------
    --cache            the experience cache serves decisions         M   (§7.4)
    --forced           fixed conditions force alerts                 F   (§8.4)
    --cache --forced   the system as first built                     FM
    --no_escalation    escalation does not raise verdicts            C′, M′
    --tau_floor X      floor on tau_high (default 0.35)
    --p2_recal         label-free adaptation: drift reference recalibrated
                       from confident normals                        (§7.5)
    --p2_pseudo        label-free adaptation: fusion refit on replay plus
                       consistent pseudo-labels, prequential         (§7.5)
    --p2_replay_only   control for --p2_pseudo: same schedule, replay only
    --legacy_cv_rule   cross-validation takes every verdict from the escalation
                       stage alone — the loop before the correction of §8.1,
                       kept only to reproduce the figure reported there
    --scan_fsm, --scan_trigger X, --scan_window N, --no_window
                       per-source scan accumulator and session window
                       (window 60 s is the paper's setting; accumulator off)
    --fpr_ablation, --quick_compare
                       older experiment harnesses. They do not run the
                       evaluated decision rule; no paper figure uses them.

Outputs
-------
    test_results.csv      per-event verdicts, scores and reasons
    threshold_trace.csv   per-event thresholds and drift inputs (§5.5)
    plots/, models/       figures and fitted models
    log blocks            [CV Metrics]; [B5] outside the layer-model split and
                          [B5-strict] unseen by every fitted component; [B2]
                          band substitution; [TAU-TRACE]; [P2] with --p2_*

Reconstruction notes (2026-09-24)
---------------------------------
Consolidates ids_complete_fsm3_pending.py with patch_cv_rule.py,
patch_b5_strict.py, patch_phase2.py and patch_stream_single.py. No decision in
configuration C or in any demonstration configuration changes. Otherwise:
  - defaults are configuration C: --no_cache, --no_forced and --cv_full_rule
    are replaced by --cache, --forced and --legacy_cv_rule; --n defaults to 46000
  - removed the original --phase2 run (run_phase2, ConsistencyChecker), which
    refit on a labelled split and could not be compared with C; the in-stream
    adaptation (--p2_*) replaces it. Removed the --stream_single test switch.
  - cache hits record their measured latency instead of a constant, and the
    speed-up print is gone
  - [B5] is labelled as the complement of the layer models' split, not as
    records "never used for fitting"
  - the outcome memory's "fp_rate" is reported as what it is: the share of
    normal verdicts among those it recorded
  - docstrings and comments corrected where the paper found them wrong: the
    contextual threshold equation, the drift update, the escalation gate,
    cluster risk, the evidence transform, the trust tracker, the online learner
"""

import os, sys, time, warnings, threading, argparse, copy

# ════════════════════════════════════════════════════════════════════
# CONFIGURATION SWITCHES — read at import; defaults are configuration C
# ════════════════════════════════════════════════════════════════════
_REMOVED = {
    '--no_cache':      'the cache is off by default; --cache enables it',
    '--no_forced':     'fixed conditions are off by default; --forced enables them',
    '--cv_full_rule':  'the deployed rule is the default; --legacy_cv_rule restores the old loop',
    '--phase2':        'replaced by --p2_recal / --p2_pseudo (in-stream, paper §7.5)',
    '--stream_single': 'removed; the stream is scored by the cross-validation ensemble',
}
for _flag, _why in _REMOVED.items():
    if _flag in sys.argv:
        sys.exit(f"{_flag} is no longer a switch: {_why}")

# Session physics (fanout, port entropy, burst, periodicity, fail rate) are
# computed per source host within this many seconds. 0 = un-windowed.
SCAN_WINDOW = 60
if '--scan_window' in sys.argv:
    SCAN_WINDOW = int(sys.argv[sys.argv.index('--scan_window') + 1])
if '--no_window' in sys.argv:
    SCAN_WINDOW = 0

# Per-source scan accumulator (ScanEnergy). Off in every paper configuration.
SCAN_FSM = '--scan_fsm' in sys.argv
SCAN_TRIGGER = 0.60
if '--scan_trigger' in sys.argv:
    SCAN_TRIGGER = float(sys.argv[sys.argv.index('--scan_trigger') + 1])

# Experience cache: always stores; serves a decision only with --cache.
CACHE_DISABLED = '--cache' not in sys.argv
# Fixed conditions: always evaluated; force an alert only with --forced.
NO_FORCED      = '--forced' not in sys.argv
NO_ESCALATION  = '--no_escalation' in sys.argv
# Cross-validation decides by the deployed rule unless --legacy_cv_rule.
CV_FULL_RULE   = '--legacy_cv_rule' not in sys.argv

# Configuration D (the default): all nine stages act in the evaluation.
#   trust      stage 4 as specified: an exponential average of each modality's
#              agreement with the verdict                    (--trust_legacy)
#   envelopes  stage 5, hard part: limits set from the site's own normal
#              records force an alert                        (--no_envelopes)
#   limits     stage 5, soft part: the policy pressure reads site-calibrated
#              limits instead of POLICY_THRESHOLDS           (--legacy_policy_limits)
#   one path   cross-validation decides each event exactly as the stream does
#              (trust damping, contextual thresholds, band, gated escalation),
#              and the stream feeds the policy pressure to the energy state
#              machine as cross-validation does              (--legacy_decision_path)
# --config_c switches all four off: configuration C, as reported before.
_CFG_C         = '--config_c' in sys.argv
TRUST_EMA      = not (_CFG_C or '--trust_legacy' in sys.argv)
SITE_ENVELOPES = not (_CFG_C or '--no_envelopes' in sys.argv)
SITE_LIMITS    = not (_CFG_C or '--legacy_policy_limits' in sys.argv)
ONE_PATH       = not (_CFG_C or '--legacy_decision_path' in sys.argv)
# The stream's ensemble is the five validated fold fusion models. Before this,
# each fold's model was replaced in place, after its fold was scored, by a
# one-pass SGD refit on that fold's pseudo-labels (OnlineLearner), and the
# stream was scored by those refits (--cv_online_retrain restores that).
FROZEN_ENSEMBLE = not (_CFG_C or '--cv_online_retrain' in sys.argv)
ENVELOPE_Q     = 0.999                 # hard limits: 1 in 1,000 normal values beyond
LIMIT_Q_WARN, LIMIT_Q_CRIT = 0.99, 0.999
TAU_FLOOR      = 0.35
if '--tau_floor' in sys.argv:
    TAU_FLOOR = float(sys.argv[sys.argv.index('--tau_floor') + 1])

# Label-free adaptation inside the stream (paper §7.5). Off by default.
P2_RECAL       = '--p2_recal' in sys.argv
P2_PSEUDO      = '--p2_pseudo' in sys.argv
P2_REPLAY_ONLY = '--p2_replay_only' in sys.argv
P2_ANY         = P2_RECAL or P2_PSEUDO

_BASE_OK = (CACHE_DISABLED and NO_FORCED and not NO_ESCALATION and CV_FULL_RULE
            and TAU_FLOOR == 0.35 and not SCAN_FSM and SCAN_WINDOW == 60 and not P2_ANY)
_D_FLAGS = (TRUST_EMA, SITE_ENVELOPES, SITE_LIMITS, ONE_PATH, FROZEN_ENSEMBLE)
_IS_C = _BASE_OK and not any(_D_FLAGS)
_IS_D = _BASE_OK and all(_D_FLAGS)
_onoff = lambda b: 'on' if b else 'off'
print(f"[config] cache={_onoff(not CACHE_DISABLED)}  fixed_conditions={_onoff(not NO_FORCED)}  "
      f"escalation={_onoff(not NO_ESCALATION)}  tau_floor={TAU_FLOOR}  "
      f"cv_rule={'deployed' if CV_FULL_RULE else 'legacy'}  scan_fsm={_onoff(SCAN_FSM)}  "
      f"adaptation: recal={_onoff(P2_RECAL)} pseudo={_onoff(P2_PSEUDO)} "
      f"replay_only={_onoff(P2_REPLAY_ONLY)}"
      + ("   -> configuration C" if _IS_C else ""))
print(f"[config] trust={'average' if TRUST_EMA else 'legacy'}  envelopes={_onoff(SITE_ENVELOPES)}  "
      f"policy_limits={'site' if SITE_LIMITS else 'legacy'}  "
      f"decision_path={'one' if ONE_PATH else 'legacy'}  "
      f"ensemble={'frozen fold models' if FROZEN_ENSEMBLE else 'refit on pseudo-labels'}"
      + ("   -> configuration D (all nine stages)" if _IS_D else ""))
if P2_REPLAY_ONLY and not P2_PSEUDO:
    print("[config] --p2_replay_only has no effect without --p2_pseudo")
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

from sklearn.ensemble      import RandomForestClassifier
from sklearn.linear_model  import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.cluster       import DBSCAN
from sklearn.metrics       import (classification_report, ConfusionMatrixDisplay,
                                    roc_auc_score, roc_curve,
                                    f1_score, precision_score, recall_score)

# ── B2 instrumentation: band-substitution counters ────────────────────
# Band-substitution counters. Observers only; no decision depends on them.
_BAND = {}


def _band_hit(site, fs, R, lo, hi):
    """Record one evaluation of the band test. Returns the test result."""
    d = _BAND.setdefault(site, {"n": 0, "in": 0, "absdiff": [], "signed": []})
    d["n"] += 1
    inside = (lo <= fs <= hi)
    if inside:
        d["in"] += 1
        try:
            d["absdiff"].append(abs(float(R) - float(fs)))
            d["signed"].append(float(R) - float(fs))
        except (TypeError, ValueError):
            pass
    return inside


def _band_report():
    print("\n" + "=" * 72)
    print("[B2] BAND SUBSTITUTION — how often the verdict is taken on R, not S")
    print("=" * 72)
    if not _BAND:
        print("  no substitution sites were reached in this run")
        return
    for site, d in _BAND.items():
        n, k = d["n"], d["in"]
        share = (100.0 * k / n) if n else 0.0
        print(f"\n  {site}")
        print(f"    events evaluated        : {n:,}")
        print(f"    inside [tau_lo, tau_hi] : {k:,}  ({share:.2f}%)")
        if d["absdiff"]:
            a = d["absdiff"]
            s = d["signed"]
            print(f"    |R - S| mean / max      : "
                  f"{sum(a)/len(a):.4f} / {max(a):.4f}")
            print(f"    (R - S) mean            : {sum(s)/len(s):+.4f}")
            print(f"    substitution moved the score by >0.05 in "
                  f"{sum(1 for x in a if x > 0.05):,} of {k:,} cases")
        else:
            print("    no events fell inside the band")
    tot_n = sum(d["n"] for d in _BAND.values())
    tot_k = sum(d["in"] for d in _BAND.values())
    print(f"\n  ALL SITES: {tot_k:,} of {tot_n:,} evaluations substituted "
          f"({100.0*tot_k/max(tot_n,1):.2f}%)")
    print("=" * 72)


import atexit as _atexit

# ── B5: metrics outside the layer models' split, and on the unseen subset ──
def _b5_report(results, train_df):
    """Score the complement of the training rowids. Reports, never decides."""
    print("\n" + "=" * 72)
    print("[B5] OUTSIDE THE LAYER-MODEL SPLIT — records the layer models were not fitted on")
    print("=" * 72)
    if '_rowid' not in results.columns or '_rowid' not in train_df.columns:
        print("  _rowid absent; cannot identify the unseen rows. Not inferring.")
        return
    try:
        from sklearn.metrics import (f1_score, precision_score,
                                     recall_score, roc_auc_score)
        import numpy as _np
        seen = set(train_df['_rowid'].tolist())
        mask = ~results['_rowid'].isin(seen)
        n = int(mask.sum())
        sub = results[mask]
        yt = _np.asarray(sub['final_label'])
        yp = _np.asarray(sub['pred_label'])
        ys = _np.asarray(sub['fusion_score'])
        n_atk = int((yt == 1).sum())
        n_nrm = int((yt == 0).sum())
        print(f"  rows outside the layer split: {n:,} of {len(results):,}")
        print("  (unseen by the layer models; the stream's fusion ensemble was fitted"
              " in CV on the balanced set, which holds every normal record)")
        print(f"  composition                 : {n_atk:,} attack / {n_nrm:,} normal"
              f"  ({100.0*n_atk/max(n,1):.1f}% attack)")
        print(f"  training rows excluded      : {len(seen):,}")
        if n_nrm == 0 or n_atk == 0:
            print("  one class is empty here; F1 and AUC would be undefined.")
            return
        fp = int(((yp == 1) & (yt == 0)).sum())
        print()
        print(f"  F1-Score  : {f1_score(yt, yp, zero_division=0):.4f}")
        print(f"  Precision : {precision_score(yt, yp, zero_division=0):.4f}")
        print(f"  Recall    : {recall_score(yt, yp, zero_division=0):.4f}")
        print(f"  ROC-AUC   : {roc_auc_score(yt, ys):.4f}")
        print(f"  FPR       : {fp / max(n_nrm, 1):.4f}")
        print()
        print("  Read this against the cross-validated figures, not instead of")
        print("  them. It is out of sample for the layer models only, and at this prior it")
        print("  is no more representative of a deployment than the full stream.")
        if 'type' in sub.columns:
            print("\n  per-class on the unseen subset:")
            print(f"    {'class':<12} {'n':>7} {'flagged':>9} {'rate':>8}")
            for cls, g in sub.groupby(sub['type'].astype(str).str.strip().str.lower()):
                k = int((_np.asarray(g['pred_label']) == 1).sum())
                print(f"    {cls:<12} {len(g):>7,} {k:>9,} {k/max(len(g),1):>8.4f}")
        _bal = globals().get('_B5_BALANCED')
        if _bal:
            m2 = ~results['_rowid'].isin(_bal)
            s2 = results[m2]
            yt2 = _np.asarray(s2['final_label']); yp2 = _np.asarray(s2['pred_label'])
            na2, nn2 = int((yt2 == 1).sum()), int((yt2 == 0).sum())
            print()
            print("  [B5-strict] outside the balanced set — unseen by every fitted component")
            print(f"    strict rows            : {len(s2):,}  ({na2:,} attack / {nn2:,} normal)")
            if na2:
                print(f"    strict detection rate  : {((yt2 == 1) & (yp2 == 1)).sum() / na2:.4f}")
            if nn2:
                print(f"    strict FPR             : {((yt2 == 0) & (yp2 == 1)).sum() / nn2:.4f}")
            if 'type' in s2.columns:
                for cls, g in s2.groupby(s2['type'].astype(str).str.strip().str.lower()):
                    k = int((_np.asarray(g['pred_label']) == 1).sum())
                    print(f"    strict {cls:<12} {len(g):>7,} {k:>9,} {k/max(len(g),1):>8.4f}")
    except Exception as e:
        print(f"  B5 report failed: {type(e).__name__}: {e}")
        print("  (reported rather than swallowed; the main pipeline is unaffected)")
    print("=" * 72)
# ── end B5 ────────────────────────────────────────────────────────────
_atexit.register(_band_report)
# ── end B2 instrumentation ────────────────────────────────────────────

# LIME is optional
try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("⚠️  lime not installed — explainability will be skipped.")
    print("   Install with: pip install lime\n")

# ════════════════════════════════════════════════════════════════════
# SECTION 1 — SYNTHETIC DATA GENERATOR (mirrors ToN_IoT structure)
# ════════════════════════════════════════════════════════════════════

def generate_synthetic_data(n=10000, seed=42):
    """
    Generates three DataFrames that mimic ToN_IoT CSV structure:
      - Network traffic  (flows with duration, bytes, protocol)
      - IoT thermostat   (date/time, temperature readings)
      - Linux disk log   (disk write, CPU usage)
    Labels: 0 = normal, 1 = attack
    """
    rng  = np.random.default_rng(seed)
    half = n // 2
    base_ts = 1_600_000_000

    # Shared base timestamps — all modalities aligned on same time axis
    # This ensures merge_asof finds matches in synthetic test mode.
    # Real datasets have independent timestamps — tolerance handles that.
    shared_ts = base_ts + np.sort(rng.integers(0, 86400 * 30, n))

    # ── NETWORK — exact ToN_IoT column names ─────────────────────
    ts_net    = shared_ts.copy()
    label_net = np.array([0]*half + [1]*half)
    dur       = np.where(label_net == 0,
                         rng.exponential(1.0, n),
                         rng.exponential(0.1, n))
    sbytes    = np.where(label_net == 0,
                         rng.integers(100, 5000, n),
                         rng.integers(5000, 50000, n))
    dbytes    = np.where(label_net == 0,
                         rng.integers(50, 3000, n),
                         rng.integers(100, 1000, n))
    proto     = rng.choice(['tcp','udp','icmp'], n, p=[0.6, 0.3, 0.1])
    service   = rng.choice(['http','ftp','ssh','dns','-'], n,
                            p=[0.4, 0.1, 0.2, 0.2, 0.1])

    df_net = pd.DataFrame({
        'ts':      ts_net,
        'dur':     dur.round(4),
        'sbytes':  sbytes,
        'dbytes':  dbytes,
        'proto':   proto,
        'service': service,
        'label':   label_net
    })

    # ── IOT FRIDGE — exact IoT Fridge column names ───────────────
    ts_iot    = shared_ts.copy()
    label_iot = np.array([0]*half + [1]*half)
    fridge_tem = np.where(label_iot == 0,
                          4.0 + rng.normal(0, 0.5, n),   # stable fridge ~4°C
                          4.0 + rng.normal(0, 8.0, n))   # attack: erratic swings
    temp_con  = np.where(fridge_tem > 7, 'high',
                np.where(fridge_tem < 1, 'low', 'normal'))

    dates = pd.to_datetime(ts_iot, unit='s').strftime('%Y-%m-%d')
    times = pd.to_datetime(ts_iot, unit='s').strftime('%H:%M:%S')

    df_iot = pd.DataFrame({
        'ts':         ts_iot,            # direct unix ts — avoids date/time roundtrip loss
        'date':       dates,
        'time':       times,
        'fridge_tem': fridge_tem.round(2),
        'temp_con':   temp_con,
        'label':      label_iot
    })

    # ── LINUX DISK LOG — exact column names ─────────────────────
    ts_log    = shared_ts.copy()
    label_log = np.array([0]*half + [1]*half)
    wrdsk     = np.where(label_log == 0,
                         rng.integers(10, 200, n),
                         rng.integers(500, 5000, n))
    cpu       = np.where(label_log == 0,
                         rng.uniform(5, 40, n),
                         rng.uniform(60, 99, n))
    mem       = np.where(label_log == 0,
                         rng.uniform(20, 60, n),
                         rng.uniform(70, 99, n))

    df_log = pd.DataFrame({
        'TS':    ts_log,
        'WRDSK': wrdsk,
        'CPU':   cpu.round(1),
        'MEM':   mem.round(1),
        'label': label_log
    })

    print("✅ Synthetic data generated (mirrors ToN_IoT structure)")
    print(f"   Network:    {len(df_net)} rows")
    print(f"   Thermostat: {len(df_iot)} rows")
    print(f"   Disk Log:   {len(df_log)} rows")
    return df_net, df_iot, df_log


def save_synthetic_csvs(out_dir='.'):
    """Save synthetic data as CSVs so you can inspect them."""
    os.makedirs(out_dir, exist_ok=True)
    net, iot, log = generate_synthetic_data()
    net.to_csv(f'{out_dir}/synthetic_network.csv',    index=False)
    iot.to_csv(f'{out_dir}/synthetic_thermostat.csv', index=False)
    log.to_csv(f'{out_dir}/synthetic_disk.csv',       index=False)
    print(f"   Saved to {out_dir}/synthetic_*.csv")
    return (f'{out_dir}/synthetic_network.csv',
            f'{out_dir}/synthetic_thermostat.csv',
            f'{out_dir}/synthetic_disk.csv')


# ════════════════════════════════════════════════════════════════════
# SECTION 2 — DATA LOADING & PREPROCESSING
# ════════════════════════════════════════════════════════════════════

def ensure_timestamp(df):
    """
    Unified timestamp handler for all four dataset formats.
      Network/Log : already has 'ts' or 'TS'
      IoT Fridge  : has 'date' + 'time' columns → combines and converts

    Also normalises timestamps that are in the wrong scale:
      < 1e8  → likely seconds-since-day or relative → multiply by 1000
      < 1e9  → likely milliseconds → multiply by 1000
      >= 1e9 → correct Unix seconds
    """
    if 'ts' in df.columns:
        df = df.copy()
        ts_median = df['ts'].median()
        if ts_median < 1e8:
            df['ts'] = df['ts'] * 1000000   # microseconds or relative
        elif ts_median < 1e9:
            df['ts'] = df['ts'] * 1000      # milliseconds → seconds
        return df
    if 'TS' in df.columns:
        return ensure_timestamp(df.rename(columns={'TS': 'ts'}))
    if 'date' in df.columns and 'time' in df.columns:
        df = df.copy()
        df['dt'] = pd.to_datetime(
            df['date'].astype(str) + ' ' + df['time'].astype(str), errors='coerce')
        df = df.dropna(subset=['dt'])
        df['ts'] = df['dt'].astype('int64') // 10**9
        # Sanity check: if result is still too small, try microseconds
        if df['ts'].median() < 1e9:
            df['ts'] = df['dt'].astype('int64') // 10**6
        return df
    raise ValueError("❌ No timestamp column found. Expected: 'ts', 'TS', or 'date'+'time'.")

def load_and_snip(filename, name_label, n=10000):
    if not filename.endswith('.csv'):
        filename += '.csv'
    if not os.path.exists(filename):
        print(f"❌ File not found: {filename}"); return None

    print(f"  Loading {filename} ...")
    df = pd.read_csv(filename, low_memory=False)

    # Unified timestamp handling
    try:
        df = ensure_timestamp(df)
    except ValueError as e:
        print(f"❌ {filename}: {e}"); return None

    if 'label' not in df.columns:
        print(f"❌ {filename}: no 'label' column."); return None

    df['label'] = df['label'].apply(
        lambda x: 0 if str(x).strip() in ['0','normal','Normal'] else 1)

    n_half = n // 2
    normal = df[df['label']==0].sample(n=min(len(df[df['label']==0]), n_half), random_state=42)
    attack = df[df['label']==1].sample(n=min(len(df[df['label']==1]), n_half), random_state=42)
    snip   = pd.concat([normal, attack]).sort_values('ts').reset_index(drop=True)
    print(f"  ✅ {name_label}: {len(snip)} rows  (normal={len(normal)}, attack={len(attack)})")
    return snip



# ════════════════════════════════════════════════════════════════════
# NETWORK ATTACK SIGNATURE LIBRARY
#
# Instead of feeding only raw numeric columns to the Network RF,
# we compute how closely each flow matches known attack families.
# This gives the model semantic features: "how much does this
# look like a port scan?" rather than just "src_bytes=4200".
#
# Each signature is a dict of (column, comparison, threshold) rules.
# The scorer returns a float 0-1 per attack family per row.
# ════════════════════════════════════════════════════════════════════

ATTACK_SIGNATURES = {

    # ── Port / Host Discovery ──────────────────────────────────────
    "scan": {
        "description": "Port scan / host sweep — many short low-byte connections",
        "rules": [
            ("duration",   "lt",  0.5),    # very short connections
            ("src_bytes",  "lt",  500),     # minimal data
            ("dst_bytes",  "lt",  200),
            ("src_pkts",   "lt",  5),       # few packets
        ],
        "proto_hint":   ["tcp", "udp", "icmp"],
        "conn_state_hit": ["S0", "REJ", "RSTO"],  # rejected/no response
        "weight": 1.0,
    },

    # ── Denial of Service ──────────────────────────────────────────
    "dos": {
        "description": "DoS/DDoS — flood of traffic, high volume, short duration",
        "rules": [
            ("src_bytes",  "gt",  50_000),  # massive outbound
            ("src_pkts",   "gt",  100),     # many packets
            ("duration",   "lt",  5.0),     # burst not sustained
        ],
        "proto_hint":   ["tcp", "udp", "icmp"],
        "conn_state_hit": ["S0", "OTH", "RSTO"],
        "weight": 1.0,
    },

    # ── Brute Force ────────────────────────────────────────────────
    "brute_force": {
        "description": "Credential brute force — repeated auth attempts",
        "rules": [
            ("dst_bytes",  "lt",  1000),    # server sends little back
            ("src_bytes",  "lt",  5000),    # small auth payloads
            ("duration",   "lt",  2.0),     # fast attempts
        ],
        "service_hint": ["ssh", "ftp", "http", "smtp"],
        "dst_port_hint": [22, 21, 3389, 5900, 23, 25],
        "conn_state_hit": ["REJ", "S0", "RSTO"],
        "weight": 1.0,
    },

    # ── Data Exfiltration ──────────────────────────────────────────
    "exfiltration": {
        "description": "Data exfiltration — large outbound, sustained connection",
        "rules": [
            ("src_bytes",  "gt",  100_000), # large data sent out
            ("dst_bytes",  "lt",  10_000),  # small response
            ("duration",   "gt",  5.0),     # sustained
        ],
        "proto_hint":   ["tcp"],
        "conn_state_hit": ["SF", "S1"],     # established connections
        "weight": 1.0,
    },

    # ── Injection (SQLi / Command) ─────────────────────────────────
    "injection": {
        "description": "Injection attacks — HTTP with large request body",
        "rules": [
            ("http_request_body_len", "gt", 200),   # large payload
            ("http_response_body_len","gt", 0),      # server responded
            ("duration",              "lt", 10.0),
        ],
        "service_hint": ["http"],
        "dst_port_hint": [80, 8080, 443, 8443],
        "weight": 1.0,
    },

    # ── Man-in-the-Middle ─────────────────────────────────────────
    "mitm": {
        "description": "MITM — symmetric traffic, unusual SSL, ARP anomalies",
        "rules": [
            ("src_bytes",  "ratio_dst", 0.8),   # src ≈ dst bytes (relay)
            ("src_pkts",   "ratio_dst", 0.8),   # packet symmetry
            ("duration",   "gt",  2.0),         # sustained relay
        ],
        "proto_hint":   ["tcp"],
        "weight": 0.8,
    },

    # ── Ransomware ────────────────────────────────────────────────
    "ransomware": {
        "description": "Ransomware C2 — high unique destinations, encrypted traffic",
        "rules": [
            ("src_bytes",  "gt", 1000),
            ("dst_bytes",  "gt", 500),
            ("duration",   "gt", 0.1),
        ],
        "service_hint": ["ssl", "-"],    # encrypted or unknown
        "dst_port_hint": [443, 8443, 4444, 1337, 6666, 9090],
        "weight": 0.9,
    },

    # ── Backdoor / C2 Beacon ─────────────────────────────────────
    "backdoor": {
        "description": "Backdoor beacon — periodic small connections",
        "rules": [
            ("src_bytes",  "lt",  2000),
            ("dst_bytes",  "lt",  2000),
            ("duration",   "gt",  0.5),   # not instant
        ],
        "dst_port_hint": [4444, 1337, 8080, 8888, 9999, 6666],
        "weight": 0.85,
    },

    # ── Password Spray ────────────────────────────────────────────
    "password": {
        "description": "Password attack — same pattern as brute force but wider",
        "rules": [
            ("src_bytes",  "lt",  3000),
            ("dst_bytes",  "lt",  2000),
            ("src_pkts",   "lt",  20),
        ],
        "service_hint": ["http", "ssh", "smtp", "ftp"],
        "weight": 0.9,
    },
}

# Numeric conn_state encoding
CONN_STATE_RISK = {
    "SF": 0.1,   # normal established
    "S0": 0.9,   # no response (scan/flood)
    "REJ": 0.8,  # rejected (brute/scan)
    "RSTO": 0.7, # reset by originator
    "RSTR": 0.7, # reset by responder
    "SH": 0.6,   # SYN→FIN half-open
    "OTH": 0.5,  # other
    "S1": 0.2,   # established, not closed
    "S2": 0.3,
    "S3": 0.3,
}


class NetworkAttackLibrary:
    """
    Scores each network flow against known attack family signatures.
    Returns a vector of float scores (one per attack family) that
    encodes how semantically similar the flow is to each attack type.

    These scores become additional features for the Network RF,
    giving it domain knowledge beyond raw numerics.
    """

    FAMILIES = list(ATTACK_SIGNATURES.keys())

    def score_row(self, row: dict) -> dict:
        """Score one network event against all attack families."""
        scores = {}
        for family, sig in ATTACK_SIGNATURES.items():
            score = self._score_signature(row, sig)
            scores[f"atk_{family}"] = round(score, 4)
        # Add a generic anomaly score: conn_state risk
        cs = str(row.get("conn_state", "SF")).upper()
        scores["conn_risk"] = CONN_STATE_RISK.get(cs, 0.3)
        return scores

    def _score_signature(self, row: dict, sig: dict) -> float:
        rules   = sig.get("rules", [])
        weight  = sig.get("weight", 1.0)
        matched = 0

        for rule in rules:
            col, op, threshold = rule[0], rule[1], rule[2]
            val = float(row.get(col, 0) or 0)
            if op == "gt"       and val > threshold:  matched += 1
            elif op == "lt"     and val < threshold:  matched += 1
            elif op == "ratio_dst":
                dst = float(row.get(col.replace("src_","dst_"), 1) or 1)
                ratio = val / (dst + 1e-6)
                if abs(ratio - 1.0) < (1.0 - threshold): matched += 1

        # Bonus: conn_state match
        cs = str(row.get("conn_state", "")).upper()
        if cs in sig.get("conn_state_hit", []):  matched += 0.5

        # Bonus: service match
        svc = str(row.get("service", "")).lower()
        if svc in sig.get("service_hint", []):   matched += 0.5

        # Bonus: dst_port match
        try:
            dport = int(row.get("dst_port", 0) or 0)
            if dport in sig.get("dst_port_hint", []): matched += 0.5
        except: pass

        max_possible = len(rules) + 1.5   # rules + all bonuses
        return float(np.clip((matched / max_possible) * weight, 0.0, 1.0))

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply signature scoring to a full DataFrame.
        Returns a new DataFrame with one column per attack family.
        """
        scores_list = []
        # Use raw columns if available, else use processed ones
        for _, row in df.iterrows():
            scores_list.append(self.score_row(row.to_dict()))
        return pd.DataFrame(scores_list, index=df.index)

    def feature_names(self):
        return [f"atk_{f}" for f in self.FAMILIES] + ["conn_risk"]



class NetworkStatisticalLayer:
    """
    Layer 1 — Statistical Deviation Engine.
    Measures HOW MUCH a flow deviates from the population baseline.
    This is the "unknown attack detector" — catches novel behavior
    that no signature can anticipate.

    Computes per-flow:
      deviation_score    : how far this flow is from the mean
      entropy_flag       : entropy-based anomaly (payload randomness)
      burst_score        : flow intensity spike relative to median
      timing_instability : inter-arrival variance (session level)
      flow_rarity        : how unusual this protocol/service combo is

    After fit() on training data:
      - baselines (mean, std) are stored per feature
      - each new flow gets a z-score → sigmoid → deviation_score
    """

    def __init__(self):
        self._baselines = {}   # {feature: (mean, std)}
        self._fitted    = False

    def fit(self, df: pd.DataFrame):
        """Compute baselines from training data."""
        for col in ['src_bytes','dst_bytes','duration','src_pkts','dst_pkts']:
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors='coerce').fillna(0)
                self._baselines[col] = (float(vals.mean()), float(vals.std()) + 1e-6)
        self._fitted = True
        return self

    def deviation_score(self, row: dict) -> float:
        """
        Mean z-score across key features → sigmoid → 0-1 deviation score.
        High score = this flow is statistically unusual.
        """
        if not self._baselines:
            return 0.5
        z_scores = []
        for col, (mu, sigma) in self._baselines.items():
            val = float(row.get(col, 0) or 0)
            z_scores.append(abs(val - mu) / sigma)
        mean_z = float(np.mean(z_scores)) if z_scores else 0
        # Sigmoid: z=0 → 0.5, z=2 → 0.88, z=4 → 0.98
        return float(1 / (1 + np.exp(-mean_z + 1.5)))

    def entropy_flag(self, row: dict) -> float:
        """
        High-entropy payload (random bytes) → possible encryption/compression
        indicating C2 or exfiltration. Estimated from byte-rate patterns.
        """
        src = float(row.get('src_bytes', 0) or 0)
        pkts= float(row.get('src_pkts', 1) or 1)
        avg_pkt = src / pkts
        # Unusually large or small packets relative to connection suggest randomness
        if avg_pkt > 1400 or (0 < avg_pkt < 40):
            return 0.75
        if 500 < avg_pkt < 1200:
            return 0.20   # normal range
        return 0.45

    def flow_rarity(self, proto, service) -> float:
        """
        Unusual protocol/service combinations score higher.
        Based on common vs rare protocol usage in enterprise networks.
        """
        common = {('tcp','http'), ('tcp','ssl'), ('tcp','ssh'), ('udp','dns'),
                  ('tcp','-'), ('udp','-'), ('tcp','smtp'), ('tcp','ftp')}
        key = (str(proto).lower(), str(service).lower())
        if key in common: return 0.10
        if str(proto).lower() == 'icmp': return 0.50
        return 0.65

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all statistical layer features to a DataFrame."""
        result = pd.DataFrame(index=df.index)

        result['stat_deviation'] = df.apply(
            lambda r: self.deviation_score(r.to_dict()), axis=1)

        result['stat_entropy']   = df.apply(
            lambda r: self.entropy_flag(r.to_dict()), axis=1)

        result['stat_rarity']    = df.apply(
            lambda r: self.flow_rarity(
                r.get('proto', ''), r.get('service', '')), axis=1)

        # Burst score: intensity vs median across dataset
        if 'src_bytes' in df.columns and 'duration' in df.columns:
            bytes_col = pd.to_numeric(df['src_bytes'], errors='coerce').fillna(0)
            dur_col   = pd.to_numeric(df['duration'],  errors='coerce').fillna(1)
            intensity = bytes_col / (dur_col + 1e-6)
            median_i  = intensity.median() + 1
            result['stat_burst'] = (intensity / median_i).clip(0, 10) / 10.0
        else:
            result['stat_burst'] = 0.5

        return result.fillna(0.5)

    def feature_names(self):
        return ['stat_deviation', 'stat_entropy', 'stat_rarity', 'stat_burst']


class NetworkRiskFusion:
    """
    Layer 3 — Combines statistical deviation + behavioral intent into
    a unified network_risk score:

      network_risk = anomaly_strength × intent_confidence × persistence_factor

    This is the pre-interpreted signal that tells the RF not just
    WHAT the features are, but WHAT THEY MEAN together.

    The 4-quadrant interpretation:
      High deviation + high intent  → confirmed attack
      High deviation + low intent   → unknown novel anomaly
      Low deviation  + high intent  → stealth behavior
      Low deviation  + low intent   → probably noise
    """

    @staticmethod
    def fuse(deviation: float, intent_scores: dict,
             periodicity: float = 0.0, fail_rate: float = 0.0) -> dict:
        """
        Returns a dict with network_risk and quadrant classification.
        """
        # Intent confidence = max attack family score
        intent_conf = float(max(intent_scores.values())) if intent_scores else 0.0

        # Persistence = combination of periodicity (beaconing) and fail_rate (persistence)
        persistence = float(np.clip(periodicity * 0.6 + fail_rate * 0.4, 0, 1))

        # Core risk formula
        network_risk = float(np.clip(
            deviation * 0.4 + intent_conf * 0.4 + persistence * 0.2, 0, 1))

        # Quadrant classification
        if deviation >= 0.6 and intent_conf >= 0.4:
            quadrant = "CONFIRMED"     # high dev + high intent
        elif deviation >= 0.6 and intent_conf < 0.4:
            quadrant = "UNKNOWN"       # high dev + low intent = novel threat
        elif deviation < 0.4 and intent_conf >= 0.5:
            quadrant = "STEALTH"       # low dev + high intent = hiding
        else:
            quadrant = "NOISE"         # low dev + low intent

        return {
            "network_risk":   round(network_risk, 4),
            "quadrant":       quadrant,
            "anomaly_str":    round(deviation, 3),
            "intent_conf":    round(intent_conf, 3),
            "persistence":    round(persistence, 3),
        }


# Singletons
# singletons defined below after all classes


class NetworkBehaviorPhysics:
    """
    Level 2 — Behavioral Physics Engine.

    Goes beyond threshold rules to measure HOW attacks move:
      - connection surface expansion (scan fan-out)
      - traffic synchronization pressure (flood intensity)
      - temporal periodicity stability (C2 beaconing)
      - outbound asymmetry gradient (exfiltration)
      - protocol entropy collapse (DDoS)
      - lateral movement depth (east-west traversal)

    Works at TWO levels:
      1. Per-flow:    instant physics from a single row
      2. Per-session: temporal physics from a group of flows (same src_ip)

    Session features are computed once per DataFrame in transform(),
    then joined back to the flow level.
    """

    # ── Per-flow physics ────────────────────────────────────────────

    @staticmethod
    def payload_asymmetry(src_bytes, dst_bytes):
        """Upload ratio: high → exfiltration, low → C2 beacon."""
        total = src_bytes + dst_bytes + 1e-6
        return float(src_bytes / total)

    @staticmethod
    def packet_asymmetry(src_pkts, dst_pkts):
        """Packet imbalance: high → flood, 0.5 → normal bidirectional."""
        total = src_pkts + dst_pkts + 1e-6
        return float(src_pkts / total)

    @staticmethod
    def flow_intensity(src_bytes, duration):
        """Bytes per second: extreme values signal DoS or exfiltration."""
        if duration <= 0: return float(min(src_bytes, 1e6))
        return float(np.clip(src_bytes / (duration + 1e-6), 0, 1e6))

    @staticmethod
    def connection_suspicion(conn_state):
        """Translate conn_state to suspicion score."""
        return CONN_STATE_RISK.get(str(conn_state).upper(), 0.3)

    @staticmethod
    def port_risk(dst_port):
        """High-risk destination ports (admin/auth/C2 common ports)."""
        try: dp = int(dst_port or 0)
        except: return 0.0
        CRITICAL_PORTS = {22,23,445,3389,1433,3306,5432,6379,27017}
        C2_PORTS       = {4444,1337,6666,8888,9999,31337,12345}
        if dp in CRITICAL_PORTS: return 0.85
        if dp in C2_PORTS:       return 0.90
        if dp > 49151:           return 0.40   # ephemeral range
        if dp < 1024:            return 0.30   # well-known
        return 0.20

    # ── Per-session (group-level) physics ───────────────────────────

    @staticmethod
    def session_features(group: pd.DataFrame) -> pd.Series:
        """
        Compute temporal/behavioral physics for all flows from one src_ip.
        Returns a Series that gets joined back to each flow in the group.
        """
        n = len(group)

        # 1. Fan-out: number of unique destination ports (scan indicator)
        dst_ports = group['dst_port'].dropna().astype(str)
        try:
            port_vals = pd.to_numeric(dst_ports, errors='coerce').dropna()
            unique_ports = port_vals.nunique()
        except:
            unique_ports = 1
        fanout_score = float(np.clip(unique_ports / 100.0, 0, 1))

        # 2. Port entropy: high entropy = scan, low = targeted attack
        if len(port_vals) > 1:
            counts = port_vals.value_counts(normalize=True)
            port_entropy = float(-np.sum(counts * np.log2(counts + 1e-10)))
            port_entropy_norm = float(np.clip(port_entropy / 7.0, 0, 1))
        else:
            port_entropy_norm = 0.0

        # 3. Connection fail rate: high = scan/brute force
        if 'conn_state' in group.columns:
            bad_states = {'S0', 'REJ', 'RSTO', 'RSTR'}
            fail_rate = float(group['conn_state'].isin(bad_states).mean())
        else:
            fail_rate = 0.0

        # 4. Periodicity score (beaconing): low inter-arrival variance = C2
        if 'ts' in group.columns and n >= 3:
            ts_sorted = group['ts'].sort_values().values
            intervals = np.diff(ts_sorted.astype(float))
            if len(intervals) > 0 and np.mean(intervals) > 0:
                cv = np.std(intervals) / (np.mean(intervals) + 1e-6)
                # Low CV = regular intervals = beaconing
                periodicity = float(np.clip(1.0 - cv / 5.0, 0, 1))
            else:
                periodicity = 0.0
        else:
            periodicity = 0.0

        # 5. Burst score: many flows in short time = flood
        if 'ts' in group.columns and n >= 2:
            time_span = float(group['ts'].max() - group['ts'].min()) + 1
            burst_score = float(np.clip(n / time_span, 0, 1))
        else:
            burst_score = 0.0

        # 6. Payload homogeneity: same-size packets = automated/tool behavior
        if 'src_bytes' in group.columns and n >= 2:
            sb_std = float(group['src_bytes'].std())
            sb_mean = float(group['src_bytes'].mean()) + 1
            homogeneity = float(np.clip(1.0 - sb_std / sb_mean, 0, 1))
        else:
            homogeneity = 0.5

        # 7. Unique destination host count (lateral movement indicator)
        if 'dst_ip' in group.columns:
            unique_hosts = float(np.clip(group['dst_ip'].nunique() / 20.0, 0, 1))
        else:
            unique_hosts = 0.0

        return pd.Series({
            'phys_fanout':       fanout_score,
            'phys_port_entropy': port_entropy_norm,
            'phys_fail_rate':    fail_rate,
            'phys_periodicity':  periodicity,
            'phys_burst':        burst_score,
            'phys_homogeneity':  homogeneity,
            'phys_lateral':      unique_hosts,
        })

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all physics features for a network DataFrame.
        Returns a DataFrame with per-flow + per-session features.
        """
        result = pd.DataFrame(index=df.index)

        # Per-flow physics
        result['phys_payload_asym'] = df.apply(
            lambda r: self.payload_asymmetry(
                float(r.get('src_bytes', r.get('net_bytes', 0)) or 0),
                float(r.get('dst_bytes', r.get('net_dbytes', 0)) or 0)), axis=1)

        result['phys_pkt_asym'] = df.apply(
            lambda r: self.packet_asymmetry(
                float(r.get('src_pkts', 0) or 0),
                float(r.get('dst_pkts', 0) or 0)), axis=1)

        result['phys_intensity'] = df.apply(
            lambda r: float(np.log1p(self.flow_intensity(
                float(r.get('src_bytes', r.get('net_bytes', 0)) or 0),
                float(r.get('duration', r.get('net_duration', 0)) or 0)))) / 14.0, axis=1)

        result['phys_conn_susp'] = df.apply(
            lambda r: self.connection_suspicion(r.get('conn_state', 'SF')), axis=1)

        result['phys_port_risk'] = df.apply(
            lambda r: self.port_risk(r.get('dst_port', 0)), axis=1)

        # Per-session physics, computed inside a TIME WINDOW.
        #
        # Grouping by src_ip alone pooled every flow a host produced across the
        # entire loaded sample, so fanout described the host's lifetime rather
        # than a burst, and burst_score was divided by the whole capture span.
        # Windowing makes each feature mean "what this host did in the last
        # SCAN_WINDOW seconds", which is how a scan is actually defined.
        if 'src_ip' in df.columns:
            keys = ['src_ip']
            side = df[['src_ip']].copy()
            gsrc = df
            if SCAN_WINDOW > 0 and 'ts' in df.columns:
                _ts = pd.to_numeric(df['ts'], errors='coerce').fillna(0)
                side['_win'] = (_ts // SCAN_WINDOW).astype('int64')
                gsrc = df.assign(_win=side['_win'].values)
                keys = ['src_ip', '_win']
            sess = gsrc.groupby(keys, sort=False).apply(self.session_features)
            sess = sess.reset_index()
            merged = side.merge(sess, on=keys, how='left')
            merged = merged.drop(columns=keys, errors='ignore')
            merged.index = df.index
            result = pd.concat([result, merged], axis=1)
            if not getattr(NetworkBehaviorPhysics, '_win_reported', False):
                NetworkBehaviorPhysics._win_reported = True
                if SCAN_WINDOW > 0:
                    print(f"  session physics: grouped by (src_ip, {SCAN_WINDOW}s window) "
                          f"-> {len(sess):,} host-windows from {df['src_ip'].nunique():,} hosts")
                else:
                    print("  session physics: grouped by src_ip only (--no_window) "
                          "- scan indicators are pooled over the whole capture")
        else:
            # No src_ip — fill session features with neutral values
            for col in ['phys_fanout','phys_port_entropy','phys_fail_rate',
                        'phys_periodicity','phys_burst','phys_homogeneity','phys_lateral']:
                result[col] = 0.5

        return result.fillna(0.5)

    def feature_names(self):
        return ['phys_payload_asym','phys_pkt_asym','phys_intensity',
                'phys_conn_susp','phys_port_risk',
                'phys_fanout','phys_port_entropy','phys_fail_rate',
                'phys_periodicity','phys_burst','phys_homogeneity','phys_lateral']


# ── Network modality singletons (all classes defined above) ──────────
_net_attack_lib  = NetworkAttackLibrary()
_net_physics     = NetworkBehaviorPhysics()
_net_stat_layer  = NetworkStatisticalLayer()
_net_risk_fusion = NetworkRiskFusion()



def preprocess_network(df):
    """
    Adaptive ToN_IoT Network preprocessing.
    Tries multiple known column name variants for each feature.
    Prints every detected mapping so mismatches are immediately visible.
    """
    df = df.copy()
    cols = [c.lower() for c in df.columns]
    orig = df.columns.tolist()
    print(f"  Network raw columns: {orig}")

    def get(candidates):
        for c in candidates:
            for i, col in enumerate(cols):
                if col == c:
                    return orig[i]
        return None

    dur_col   = get(['dur','duration','flow_duration','conn_dur'])
    sbyte_col = get(['sbytes','src_bytes','orig_bytes','spkts','sload'])
    dbyte_col = get(['dbytes','dst_bytes','resp_bytes','dpkts','dload'])
    proto_col = get(['proto','protocol','protocol_type','ip_proto'])
    svc_col   = get(['service','svc','app','l7_proto','l7proto'])

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df['net_duration']    = pd.to_numeric(df[dur_col],   errors='coerce').fillna(0) if dur_col   else 0.0
    df['net_bytes']       = pd.to_numeric(df[sbyte_col], errors='coerce').fillna(0) if sbyte_col else 0.0
    df['net_dbytes']      = pd.to_numeric(df[dbyte_col], errors='coerce').fillna(0) if dbyte_col else 0.0
    df['net_proto_enc']   = df[proto_col].astype('category').cat.codes if proto_col else 0
    df['net_service_enc'] = df[svc_col].astype('category').cat.codes   if svc_col   else 0
    for c in df.select_dtypes(include='object').columns:
        if c not in ['label','type','_source']:
            df[c] = pd.factorize(df[c])[0]
    print(f"  net_duration    <- '{dur_col}'   net_bytes  <- '{sbyte_col}'")
    print(f"  net_dbytes      <- '{dbyte_col}'  proto      <- '{proto_col}'  service <- '{svc_col}'")

    # Layer 1: Statistical deviation (baseline-relative anomaly)
    _net_stat_layer.fit(df)   # fit on current batch as proxy baseline
    stat_df  = _net_stat_layer.transform(df)
    # Layer 2a: Rule-based attack family signatures
    sig_df   = _net_attack_lib.transform(df)
    # Layer 2b: Behavioral physics (flow + session level)
    phys_df  = _net_physics.transform(df)

    result = pd.DataFrame({
        'net_duration':    df['net_duration'].values,
        'net_bytes':       df['net_bytes'].values,
        'net_dbytes':      df['net_dbytes'].values,
        'net_proto_enc':   df['net_proto_enc'].values,
        'net_service_enc': df['net_service_enc'].values,
        'label':           df['label'].values,
        # TON_IoT's own attack class. Not a model feature (NET_FEATURES is an
        # explicit list) - carried so the evaluation can report per class.
        'type':            (df['type'].astype(str).str.strip().str.lower().values
                            if 'type' in df.columns else 'unknown'),
        # Entity key for the per-source scan accumulator. NOT a model feature
        # (NET_FEATURES is an explicit list) — the accumulator keys on it and
        # decays, so it never learns that a particular address is hostile.
        'src_ip':          (df['src_ip'].astype(str).values
                            if 'src_ip' in df.columns else 'unknown'),
        'ts':              df['ts'].values,
    })
    for col in sig_df.columns:   result[col] = sig_df[col].values
    for col in phys_df.columns:  result[col] = phys_df[col].values
    for col in stat_df.columns:  result[col] = stat_df[col].values

    # Layer 3: Pre-fused network_risk score (deviation × intent × persistence)
    result['net_risk_score'] = [
        _net_risk_fusion.fuse(
            deviation  = float(stat_df['stat_deviation'].iloc[i]),
            intent_scores = {c: float(sig_df[c].iloc[i])
                             for c in sig_df.columns if c.startswith('atk_')},
            periodicity= float(phys_df['phys_periodicity'].iloc[i])
                         if 'phys_periodicity' in phys_df.columns else 0.0,
            fail_rate  = float(phys_df['phys_fail_rate'].iloc[i])
                         if 'phys_fail_rate' in phys_df.columns else 0.0,
        )['network_risk']
        for i in range(len(df))
    ]

    n_sig  = len(sig_df.columns)
    n_phys = len(phys_df.columns)
    n_stat = len(stat_df.columns)
    print(f"  Net layers: {n_stat} statistical + {n_sig} signatures "
          f"+ {n_phys} physics + 1 risk_score = {n_stat+n_sig+n_phys+1} features")
    return result.fillna(0)


def preprocess_iot(df):
    """
    Hardcoded to IoT Fridge dataset columns:
      fridge_tem → current_temp     (raw fridge temperature)
      temp_con   → temp_con_enc     (low=0, normal=0.5, high=1)
      → phys_delta                  (computed: temp[t] - temp[t-1])
      → phys_delta_abs              (computed: spike magnitude)
      → temp_roll_mean / std        (computed: 5-step trend + volatility)
    """
    df = df.copy()
    df['current_temp']   = pd.to_numeric(df['fridge_tem'], errors='coerce').ffill().fillna(0)
    df['temp_con_enc']   = df['temp_con'].str.lower().map(
                               {'low': 0, 'normal': 0.5, 'high': 1}).fillna(0)
    df['temp_smooth']    = df['current_temp'].rolling(window=5, min_periods=1).mean()
    df['phys_delta']     = df['temp_smooth'].diff().fillna(0)
    df['phys_delta_abs'] = df['phys_delta'].abs()
    df['temp_roll_mean'] = df['temp_smooth'].copy()
    df['temp_roll_std']  = df['current_temp'].rolling(5, min_periods=1).std().fillna(0)

    print(f"  ✅ IoT: temp_mean={df['current_temp'].mean():.2f}  "
          f"delta_max={df['phys_delta_abs'].max():.2f}  "
          f"temp_con={df['temp_con'].value_counts().to_dict()}")

    return pd.DataFrame({
        'current_temp':   df['current_temp'].values,
        'temp_con_enc':   df['temp_con_enc'].values,
        'phys_delta':     df['phys_delta'].values,
        'phys_delta_abs': df['phys_delta_abs'].values,
        'temp_roll_mean': df['temp_roll_mean'].values,
        'temp_roll_std':  df['temp_roll_std'].values,
        'iot_present':    1.0,
        'label':          df['label'].values,
        'ts':             df['ts'].values,
    }).fillna(0)


def preprocess_logs(df):
    """
    Hardcoded to Linux Disk log dataset columns:
      WRDSK → disk_write       (disk write rate — ransomware = massive writes)
      CPU   → cpu_usage        (high CPU = mining / malware execution)
      MEM   → mem_usage        (memory-based attacks)
      → disk_write_delta       (computed: sudden write spike = ransomware burst)
    """
    df = df.copy()
    df['disk_write']       = pd.to_numeric(df['WRDSK'], errors='coerce').fillna(0)
    df['cpu_usage']        = pd.to_numeric(df['CPU'],   errors='coerce').fillna(0)
    df['mem_usage']        = pd.to_numeric(df['MEM'],   errors='coerce').fillna(0)
    for c in ['disk_write', 'cpu_usage', 'mem_usage']:
        p99 = df[c].quantile(0.99)
        if p99 > 0:
            df[c] = df[c].clip(upper=p99)
    df['disk_write_delta'] = df['disk_write'].diff().fillna(0).abs()
    df['cpu_spike']        = (df['cpu_usage'] - df['cpu_usage'].rolling(5,min_periods=1).median()).clip(lower=0)
    # New columns — filled with 0 for single-file mode (not available in hardcoded CSV)
    df['mem_vgrow']        = 0.0
    df['mem_rgrow']        = 0.0
    df['page_faults_min']  = 0.0
    df['page_faults_maj']  = 0.0
    df['threads_run']      = 0.0

    print(f"  ✅ Logs: disk_write_mean={df['disk_write'].mean():.1f}  "
          f"cpu_mean={df['cpu_usage'].mean():.1f}  "
          f"cpu_spike_mean={df['cpu_spike'].mean():.1f}")

    return pd.DataFrame({
        'disk_write':       df['disk_write'].values,
        'disk_write_delta': df['disk_write_delta'].values,
        'cpu_usage':        df['cpu_usage'].values,
        'cpu_spike':        df['cpu_spike'].values,
        'mem_usage':        df['mem_usage'].values,
        'mem_vgrow':        df['mem_vgrow'].values,
        'mem_rgrow':        df['mem_rgrow'].values,
        'page_faults_min':  df['page_faults_min'].values,
        'page_faults_maj':  df['page_faults_maj'].values,
        'threads_run':      df['threads_run'].values,
        'label':            df['label'].values,
        'ts':               df['ts'].values,
    }).fillna(0)


def clean_windows_columns(df):
    """Normalise Windows PerfMon column names for consistent access."""
    df.columns = (df.columns
        .str.replace(' ', '_', regex=False)
        .str.replace('%', 'pct', regex=False)
        .str.replace('(', '', regex=False)
        .str.replace(')', '', regex=False)
        .str.replace('/', '_', regex=False)
    )
    return df


WINDOWS_KEEP = [
    'ts', 'label',
    'Processor_pct_Processor_Time', 'Processor_pct_User_Time', 'Processor_Interrupts_sec',
    'Process_Thread_Count', 'Process_Handle_Count', 'Process_Page_Faults_sec',
    'Memory_pct_Committed_Bytes_In_Use', 'Memory_Page_Faults_sec', 'Memory_Available_MBytes',
    'LogicalDisk_Total_pct_Disk_Time', 'LogicalDisk_Total_Disk_Transfers_sec',
    'LogicalDisk_Total_Current_Disk_Queue_Length',
    'Bytes_Total_sec', 'Packets_sec', 'Packets_Received_Errors',
]


def preprocess_windows(df):
    """
    Windows 10 telemetry — compresses hundreds of perf columns into
    five meaningful signals then combines into a system_stress score.

      Processor_pct*  → cpu_usage       (mean across cores)
      Memory*Bytes    → memory_usage    (mean across memory counters)
      Disk*Write*     → disk_write      (mean across disk write counters)
      Network*Bytes   → network_traffic (mean across NIC counters)
      Process_*       → process_load    (mean process activity)
      → system_stress (weighted composite — overall anomaly signal)
    """
    df = df.copy()
    df = clean_windows_columns(df)
    df = ensure_timestamp(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    available = [c for c in WINDOWS_KEEP if c in df.columns]
    df = df[available] if len(available) > 2 else df
    df = df.fillna(0)
    print(f"  Windows: using {len(available)} / {len(df.columns)+len(available)} features")

    cpu_cols  = [c for c in df.columns if 'Processor_pct' in c or 'processor_pct' in c.lower()]
    mem_cols  = [c for c in df.columns if 'Memory' in c and 'Bytes' in c]
    disk_cols = [c for c in df.columns if 'Disk' in c and 'Write' in c]
    net_cols  = [c for c in df.columns if 'Bytes' in c and 'Network' in c]
    proc_cols = [c for c in df.columns if c.startswith('Process_')]

    def safe_mean(df, cols):
        if not cols: return pd.Series(0.0, index=df.index)
        return (df[cols]
                .apply(pd.to_numeric, errors='coerce')
                .fillna(0)
                .mean(axis=1, numeric_only=False))

    df['cpu_usage']       = safe_mean(df, cpu_cols)
    df['memory_usage']    = safe_mean(df, mem_cols)
    df['disk_write']      = safe_mean(df, disk_cols)
    df['network_traffic'] = safe_mean(df, net_cols)
    df['process_load']    = safe_mean(df, proc_cols)

    # Weighted composite — overall system stress
    df['system_stress'] = (df['cpu_usage']       * 0.30 +
                           df['memory_usage']     * 0.20 +
                           df['disk_write']       * 0.20 +
                           df['network_traffic']  * 0.20 +
                           df['process_load']     * 0.10)

    print(f"  ✅ Windows: cpu={df['cpu_usage'].mean():.1f}  "
          f"mem={df['memory_usage'].mean():.1f}  "
          f"disk={df['disk_write'].mean():.1f}  "
          f"stress={df['system_stress'].mean():.1f}")

    return pd.DataFrame({
        'cpu_usage':       df['cpu_usage'].values,
        'memory_usage':    df['memory_usage'].values,
        'disk_write':      df['disk_write'].values,
        'network_traffic': df['network_traffic'].values,
        'process_load':    df['process_load'].values,
        'system_stress':   df['system_stress'].values,
        'label':           df['label'].values,
        'ts':              df['ts'].values,
    }).fillna(0)


def fuse_datasets(df_net, df_iot, df_log, df_win=None):
    """
    Time-based fusion using pd.merge_asof — scientifically correct.

    Instead of aligning by row index (wrong — rows are unrelated),
    we merge on nearest timestamp within a tolerance window.
    This preserves real temporal correlations across modalities.

    df_win (Windows telemetry) is optional — pass None to skip.
    """
    print("\n[Fusion] Time-based alignment (merge_asof) ...")

    layers = [
        (df_net, 'net'),
        (df_iot, 'iot'),
        (df_log, 'log'),
    ]
    if df_win is not None:
        layers.append((df_win, 'win'))

    # Rename label columns per modality before merging
    renamed = []
    for df, tag in layers:
        d = df.copy().sort_values('ts').reset_index(drop=True)
        d = d.rename(columns={'label': f'{tag}_label'})
        renamed.append(d)

    # Merge all on nearest timestamp, 60-second tolerance
    fused = renamed[0]
    for d in renamed[1:]:
        fused = pd.merge_asof(
            fused.sort_values('ts'),
            d.sort_values('ts'),
            on='ts',
            direction='nearest',
            tolerance=300,       # 5-min window — widens if datasets collected at different rates
            suffixes=('', '_dup')
        )
        # Drop duplicate columns from suffix collisions
        fused = fused[[c for c in fused.columns if not c.endswith('_dup')]]

    if len(fused.dropna()) == 0:
        print("  ⚠️  merge_asof returned 0 rows at 300s tolerance — relaxing to 3600s ...")
        fused = renamed[0]
        for d in renamed[1:]:
            fused = pd.merge_asof(
                fused.sort_values('ts'),
                d.sort_values('ts'),
                on='ts',
                direction='nearest',
                tolerance=3600,
                suffixes=('', '_dup')
            )
            fused = fused[[c for c in fused.columns if not c.endswith('_dup')]]

    fused_clean = fused.dropna()

    if len(fused_clean) == 0:
        # ── Timestamp diagnostic ──────────────────────────────────
        print("  ⚠️  No temporal overlap found. Timestamp ranges:")
        for df, tag in layers:
            ts = df['ts'] if 'ts' in df.columns else pd.Series([0])
            print(f"    {tag:<6} {int(ts.min())} → {int(ts.max())}  "
                  f"({pd.to_datetime(int(ts.min()),unit='s').date()} → "
                  f"{pd.to_datetime(int(ts.max()),unit='s').date()})")

        # ── Proportional index fallback ───────────────────────────
        # Scientifically weaker than time-based but valid for
        # datasets collected in the same experimental campaign
        # where timestamps are stored in different formats/epochs.
        print("  ↩️  Falling back to proportional index alignment ...")
        N = min(len(d) for d, _ in layers)
        fused = renamed[0].iloc[:N].reset_index(drop=True)
        for d in renamed[1:]:
            # Sample proportionally to preserve label distribution
            idx = np.linspace(0, len(d)-1, N, dtype=int)
            slice_d = d.iloc[idx].reset_index(drop=True)
            # Drop ts from right side to avoid conflict, keep left ts
            slice_d = slice_d.drop(columns=['ts'], errors='ignore')
            fused = pd.concat([fused, slice_d], axis=1)
        # Remove any duplicate columns
        fused = fused.loc[:, ~fused.columns.duplicated()]
        fused['ts'] = renamed[0]['ts'].iloc[:N].values
        print(f"  ✅ Index-aligned fallback: {N} rows")
    else:
        fused = fused_clean

    fused = fused.reset_index(drop=True)

    # B5: stable row identity, stamped before the balanced subset and the full
    # stream are separated so both inherit it. Not a feature — see _b5_report.
    fused['_rowid'] = np.arange(len(fused), dtype='int64')

    label_cols = [c for c in fused.columns if c.endswith('_label')]
    weights = {'net_label':0.45,'iot_label':0.20,'log_label':0.20,'win_label':0.15}
    fused['fusion_vote'] = sum(fused[c]*weights.get(c,1.0/len(label_cols)) for c in label_cols if c in fused.columns)
    fused['final_label'] = (fused['fusion_vote'] >= 0.45).astype(int)
    n_atk = int(fused['final_label'].sum())
    n_nrm = int((fused['final_label']==0).sum())
    print(f"  Consensus label (>=0.45): attack={n_atk} | normal={n_nrm}")

    if n_nrm > 0 and n_atk > 0:
        # Save full unbalanced fused data — used as realistic test set
        full_fused = fused.copy().sort_values('ts').reset_index(drop=True)
        # Balance for training only (50/50 so model learns both classes equally)
        n_each = min(n_atk, n_nrm)
        fused = pd.concat([
            fused[fused['final_label']==0].sample(n=n_each, random_state=42),
            fused[fused['final_label']==1].sample(n=n_each, random_state=42)
        ]).sort_values('ts').reset_index(drop=True)
        print(f"  Full dataset: {len(full_fused)} rows (attack={n_atk} | normal={n_nrm})")
        print(f"  Balanced for training: {len(fused)} rows (50/50)")
    else:
        print("  ⚠️  One class empty after fusion — check dataset timestamp overlap")
        full_fused = fused.copy()

    for col in ['net_bytes','net_duration','disk_write','cpu_usage','phys_delta']:
        if col in fused.columns:
            fused[f'{col}_roll_mean'] = fused[col].rolling(5,min_periods=1).mean()
            fused[f'{col}_roll_std']  = fused[col].rolling(5,min_periods=1).std().fillna(0)
            fused[f'{col}_ewma']      = fused[col].ewm(span=5,adjust=False).mean()
    print(f"  ✅ Fused: {len(fused)} rows | "
          f"attack={int(fused['final_label'].sum())} | "
          f"normal={int((fused['final_label']==0).sum())} | "
          f"layers={[t for _,t in layers]}")
    return fused, full_fused


# ════════════════════════════════════════════════════════════════════
# SECTION 3 — PER-LAYER MODELS
# ════════════════════════════════════════════════════════════════════

class LayerModels:
    NET_FEATURES = ['net_duration', 'net_bytes', 'net_dbytes', 'net_proto_enc', 'net_service_enc',
                    # L1 Statistical: deviation-based anomaly features
                    'stat_deviation', 'stat_entropy', 'stat_rarity', 'stat_burst',
                    # L2a Signatures: attack family rule scores
                    'atk_scan', 'atk_dos', 'atk_brute_force', 'atk_exfiltration',
                    'atk_injection', 'atk_mitm', 'atk_ransomware', 'atk_backdoor',
                    'atk_password', 'conn_risk',
                    # L2b Physics: behavioral flow + session features
                    'phys_payload_asym', 'phys_pkt_asym', 'phys_intensity',
                    'phys_conn_susp', 'phys_port_risk',
                    'phys_fanout', 'phys_port_entropy', 'phys_fail_rate',
                    'phys_periodicity', 'phys_burst', 'phys_homogeneity', 'phys_lateral',
                    # L3 Risk Fusion: pre-interpreted network risk signal
                    'net_risk_score']
    IOT_FEATURES = ['current_temp', 'temp_con_enc', 'phys_delta', 'phys_delta_abs', 'temp_roll_mean', 'temp_roll_std', 'iot_present']
    LOG_FEATURES = ['disk_write','disk_write_delta','cpu_usage','cpu_spike','mem_usage','mem_vgrow','mem_rgrow','page_faults_min','page_faults_maj','threads_run']
    WIN_FEATURES = ['cpu_usage', 'memory_usage', 'disk_write', 'network_traffic', 'system_stress']

    def __init__(self):
        self.net_model  = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
        self.iot_model  = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
        self.log_model  = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
        self.net_scaler = StandardScaler()
        self.iot_scaler = StandardScaler()
        self.log_scaler = StandardScaler()

    def _safe(self, df, cols):
        out = pd.DataFrame(index=df.index)
        for c in cols:
            out[c] = df[c] if c in df.columns else 0.0
        return out.fillna(0)

    def fit(self, fused_train):
        print("\n[Layer Models] Training ...")
        self.net_model.fit(
            self.net_scaler.fit_transform(self._safe(fused_train, self.NET_FEATURES)),
            fused_train['net_label'])
        self.iot_model.fit(
            self.iot_scaler.fit_transform(self._safe(fused_train, self.IOT_FEATURES)),
            fused_train['iot_label'])
        self.log_model.fit(
            self.log_scaler.fit_transform(self._safe(fused_train, self.LOG_FEATURES)),
            fused_train['log_label'])
        print("  ✅ Network RF | ✅ IoT RF | ✅ Log RF")

    def predict_proba_raw(self, df):
        p_net = self.net_model.predict_proba(
            self.net_scaler.transform(self._safe(df, self.NET_FEATURES)))[:,1]
        p_iot = self.iot_model.predict_proba(
            self.iot_scaler.transform(self._safe(df, self.IOT_FEATURES)))[:,1]
        p_log = self.log_model.predict_proba(
            self.log_scaler.transform(self._safe(df, self.LOG_FEATURES)))[:,1]
        return p_net, p_iot, p_log

    def save(self, path='models/'):
        os.makedirs(path, exist_ok=True)
        for name in ['net_model','iot_model','log_model',
                     'net_scaler','iot_scaler','log_scaler']:
            joblib.dump(getattr(self, name), f'{path}{name}.pkl')
        print(f"  ✅ Layer models saved → {path}")

    def load(self, path='models/'):
        for name in ['net_model','iot_model','log_model',
                     'net_scaler','iot_scaler','log_scaler']:
            setattr(self, name, joblib.load(f'{path}{name}.pkl'))


# ════════════════════════════════════════════════════════════════════
# SECTION 4 — CALIBRATION + EVIDENCE TRANSFORM
# ════════════════════════════════════════════════════════════════════

def calibrate(p, steepness=5.0):
    """Sigmoid squashing of forest probabilities. Followed by to_log_odds this
    composes to the identity up to scale — L_m = steepness * (p_m - 0.5) — so
    p_m and L_m are two monotone forms of one quantity, not independent
    evidence; only a modality's pair of fusion weights is interpretable
    (paper §5.4)."""
    return 1.0 / (1.0 + np.exp(-steepness * (p - 0.5)))


def to_log_odds(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def build_evidence_vector(p_net, p_iot, p_log, phys_delta):
    return pd.DataFrame({
        'L_net':      to_log_odds(p_net),
        'L_iot':      to_log_odds(p_iot),
        'L_log':      to_log_odds(p_log),
        'phys_delta': phys_delta,
        'p_net':      p_net,
        'p_iot':      p_iot,
        'p_log':      p_log,
    })


# ════════════════════════════════════════════════════════════════════
# SECTION 5 — FUSION META-CLASSIFIER
# ════════════════════════════════════════════════════════════════════

class FusionClassifier:
    FUSION_FEATURES = ['p_net','p_iot','p_log','L_net','L_iot','L_log','phys_delta']

    def __init__(self):
        self.model  = LogisticRegression(C=1.0, max_iter=1000, random_state=42, class_weight='balanced')
        self.scaler = StandardScaler()

    def fit(self, evidence_df, y):
        print("\n[Fusion] Training meta-classifier (Logistic Regression) ...")
        X = self.scaler.fit_transform(evidence_df[self.FUSION_FEATURES].fillna(0))
        self.model.fit(X, y)
        weights = dict(zip(self.FUSION_FEATURES, self.model.coef_[0]))
        print("  Learned fusion weights:")
        for k, v in sorted(weights.items(), key=lambda x: abs(x[1]), reverse=True):
            bar = '█' * max(1, int(abs(v)*8))
            print(f"    {k:<15} {v:+.4f}  {bar}")
        print("  ✅ Fusion meta-classifier trained")

    def predict_proba_score(self, evidence_df):
        X = self.scaler.transform(evidence_df[self.FUSION_FEATURES].fillna(0))
        return self.model.predict_proba(X)[:,1]

    def save(self, path='models/'):
        os.makedirs(path, exist_ok=True)
        joblib.dump(self.model,  f'{path}fusion_model.pkl')
        joblib.dump(self.scaler, f'{path}fusion_scaler.pkl')

    def load(self, path='models/'):
        self.model  = joblib.load(f'{path}fusion_model.pkl')
        self.scaler = joblib.load(f'{path}fusion_scaler.pkl')



class EnsembleFusionClassifier:
    """
    AUC-weighted soft voting ensemble across 5 CV fold models.
    Better-performing folds (higher AUC) contribute more to the final score.
    Uses all training information from CV instead of discarding 4/5 of it.
    """
    FUSION_FEATURES = FusionClassifier.FUSION_FEATURES

    def __init__(self, fold_classifiers, auc_weights):
        self.classifiers = fold_classifiers   # list of FusionClassifier
        self.weights     = np.array(auc_weights, dtype=float)
        self.weights    /= self.weights.sum()  # normalise
        # expose same interface as FusionClassifier
        self.model  = fold_classifiers[0].model
        self.scaler = fold_classifiers[0].scaler

    def predict_proba_score(self, evidence_df):
        """Weighted average of probabilities across all fold models."""
        scores = np.zeros(len(evidence_df))
        for clf, w in zip(self.classifiers, self.weights):
            scores += w * clf.predict_proba_score(evidence_df)
        return scores

    def predict(self, evidence_df, threshold=0.35):
        return (self.predict_proba_score(evidence_df) >= threshold).astype(int)

    def save(self, path='models/'):
        """Save all fold models."""
        os.makedirs(path, exist_ok=True)
        for i, clf in enumerate(self.classifiers):
            joblib.dump(clf.model,  f'{path}ensemble_fold{i}_model.pkl')
            joblib.dump(clf.scaler, f'{path}ensemble_fold{i}_scaler.pkl')

    def print_summary(self):
        print("  [Ensemble] AUC-weighted fusion across 5 CV folds:")
        for i, (clf, w) in enumerate(zip(self.classifiers, self.weights)):
            bar = '█' * int(w * 100)
            print(f"    Fold {i+1}: weight={w:.3f}  {bar}")


# ════════════════════════════════════════════════════════════════════
# SECTION 6 — EXPERIENCE MEMORY + DBSCAN CLUSTER PROFILES
# ════════════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════════════
# BEHAVIORAL THREAT MEMORY — 3-TIER SYSTEM
#
# Tier 1 — Statistical Memory  : anomaly vectors + DBSCAN (existing)
# Tier 2 — Behavioral Memory   : intent profiles + escalation paths
# Tier 3 — Outcome Memory      : confirmed/FP labels + operator context
#
# Key insight: connects isolated events into attack campaigns by
# tracking behavioral progression over time, not just numeric similarity.
# ════════════════════════════════════════════════════════════════════

# Known attack progressions (Tier 2 pattern matching)
ATTACK_PROGRESSIONS = {
    "recon_to_exploit": {
        "sequence":    ["scan_like", "brute_force_like", "backdoor_like"],
        "description": "Reconnaissance → Access → Persistence",
        "severity":    0.90,
    },
    "recon_to_exfil": {
        "sequence":    ["scan_like", "exfiltration_like"],
        "description": "Reconnaissance → Data Exfiltration",
        "severity":    0.85,
    },
    "beacon_to_exfil": {
        "sequence":    ["beacon_like", "exfiltration_like"],
        "description": "C2 Beaconing → Exfiltration",
        "severity":    0.88,
    },
    "dos_campaign": {
        "sequence":    ["dos_like", "dos_like", "dos_like"],
        "description": "Sustained DoS Campaign",
        "severity":    0.80,
    },
    "lateral_spread": {
        "sequence":    ["scan_like", "lateral_like", "beacon_like"],
        "description": "Scan → Lateral Movement → Persistence",
        "severity":    0.92,
    },
}


class BehavioralEntry:
    """Single entry in Tier 2 behavioral memory."""
    __slots__ = ['ts','intent_profile','temporal_profile',
                 'escalation_path','net_risk','outcome']

    def __init__(self, ts, intent_profile, temporal_profile,
                 escalation_path, net_risk, outcome="pending"):
        self.ts              = ts
        self.intent_profile  = intent_profile     # {family: score}
        self.temporal_profile= temporal_profile   # {periodicity, burst, persist}
        self.escalation_path = escalation_path    # ["SAFE","SUSPICIOUS",...]
        self.net_risk        = net_risk
        self.outcome         = outcome            # pending/confirmed/false_positive

    def dominant_behavior(self):
        """Return the highest-scoring attack family."""
        if not self.intent_profile: return "unknown"
        return max(self.intent_profile, key=self.intent_profile.get)

    def to_tag(self):
        dom = self.dominant_behavior()
        return f"{dom}_like"


class BehavioralThreatMemory:
    """
    Tier 2 — Behavioral Memory.

    Stores intent profiles + escalation paths per behavioral signature.
    Groups events by behavioral similarity (not numeric similarity).
    Recognizes multi-stage attack progressions.

    Key capability:
      Recon event + Auth event + Beacon event
      → recognized as: "recon_to_exploit" campaign
      → escalation energy boosted accordingly
    """

    def __init__(self, max_entries=500, ttl_events=200):
        self._entries    = []         # list of BehavioralEntry
        self._max        = max_entries
        self._ttl        = ttl_events # decay old entries after N events
        self._event_count= 0
        self._campaigns  = []         # detected campaign progressions

    def store(self, ts, intent_profile, temporal_profile,
              escalation_path, net_risk):
        """Store a behavioral observation."""
        entry = BehavioralEntry(
            ts=ts,
            intent_profile=dict(intent_profile),
            temporal_profile=dict(temporal_profile),
            escalation_path=list(escalation_path),
            net_risk=float(net_risk)
        )
        self._entries.append(entry)
        self._event_count += 1

        # Age-out old entries
        if len(self._entries) > self._max:
            self._entries = self._entries[-self._max:]

        # Check for attack progressions
        self._check_progressions()

    def _check_progressions(self):
        """Scan recent entries for known attack progression patterns."""
        if len(self._entries) < 2: return
        recent = self._entries[-10:]   # last 10 behavioral entries
        recent_tags = [e.to_tag() for e in recent]

        for prog_name, prog in ATTACK_PROGRESSIONS.items():
            seq = prog["sequence"]
            # Check if all sequence steps appear in recent_tags (in order)
            pos = 0
            for tag in recent_tags:
                if pos < len(seq) and tag == seq[pos]:
                    pos += 1
            if pos == len(seq):
                campaign = {
                    "type":        prog_name,
                    "description": prog["description"],
                    "severity":    prog["severity"],
                    "detected_at": self._event_count,
                    "tags":        recent_tags[-len(seq):]
                }
                # Avoid duplicate campaign alerts
                if not self._campaigns or                    self._campaigns[-1]["type"] != prog_name or                    self._event_count - self._campaigns[-1]["detected_at"] > 20:
                    self._campaigns.append(campaign)
                    print(f"  [BehavioralMemory] Campaign detected: "
                          f"{prog['description']} (sev={prog['severity']:.2f})")

    def get_campaign_risk(self):
        """Return risk boost from any recently detected campaign."""
        if not self._campaigns: return 0.0
        last = self._campaigns[-1]
        recency = self._event_count - last["detected_at"]
        if recency > 50: return 0.0    # too old
        decay = max(0.0, 1.0 - recency / 50.0)
        return float(last["severity"] * decay)

    def behavioral_similarity(self, intent_profile: dict) -> float:
        """
        How similar is this event's intent to recent behavioral memory?
        Used to boost escalation energy for behaviorally consistent attacks.
        """
        if not self._entries: return 0.0
        recent = self._entries[-20:]
        if not recent: return 0.0

        dom_new = max(intent_profile, key=intent_profile.get)                   if intent_profile else "unknown"
        matches = sum(1 for e in recent if e.dominant_behavior() == dom_new)
        return float(np.clip(matches / len(recent), 0, 1))

    def stats(self):
        if not self._entries:
            return {"behavioral_memory": "empty"}
        recent = self._entries[-50:]
        from collections import Counter
        dom_dist = Counter(e.dominant_behavior() for e in recent)
        return {
            "total_entries":    len(self._entries),
            "campaigns_found":  len(self._campaigns),
            "recent_behaviors": dict(dom_dist.most_common(5)),
            "last_campaign":    self._campaigns[-1]["description"]
                                if self._campaigns else "none",
            "campaign_risk_now":round(self.get_campaign_risk(), 3),
        }


class OutcomeMemory:
    """
    Tier 3 — Outcome Memory: the slot where operator outcomes would go.

    In this implementation the stream records the system's OWN verdict for each
    event ('confirmed' for an attack verdict, 'false_positive' for a normal one),
    so its "false-positive rate" is the share of normal verdicts among the last
    1,000 events — not a false-positive rate of anything. Nothing reads it; it
    feeds no decision. Analyst outcomes captured through the API would be the
    intended input (paper §5.6, §9.2).
    """

    def __init__(self):
        self._outcomes = []   # (intent_tag, net_risk, outcome, ts)

    def record(self, intent_tag, net_risk, outcome, ts=0):
        """
        outcome: 'confirmed' | 'false_positive' | 'escalated' | 'ignored'
        """
        self._outcomes.append((intent_tag, float(net_risk), outcome, ts))
        if len(self._outcomes) > 1000:
            self._outcomes = self._outcomes[-1000:]

    def false_positive_rate(self, intent_tag=None):
        """FP rate for a specific behavior type, or overall."""
        subset = [o for o in self._outcomes
                  if intent_tag is None or o[0] == intent_tag]
        if not subset: return 0.0
        fps = sum(1 for o in subset if o[2] == 'false_positive')
        return round(fps / len(subset), 3)

    def confirmation_rate(self, intent_tag=None):
        subset = [o for o in self._outcomes
                  if intent_tag is None or o[0] == intent_tag]
        if not subset: return 0.0
        confirmed = sum(1 for o in subset if o[2] == 'confirmed')
        return round(confirmed / len(subset), 3)

    def stats(self):
        if not self._outcomes:
            return {"outcome_memory": "empty — no outcomes recorded yet"}
        from collections import Counter
        outcomes = Counter(o[2] for o in self._outcomes)
        by_tag   = Counter(o[0] for o in self._outcomes)
        return {
            "total_outcomes":    len(self._outcomes),
            "outcome_dist":      dict(outcomes),
            "top_behaviors":     dict(by_tag.most_common(5)),
            "share_normal_verdicts": self.false_positive_rate(),
            "overall_confirm":   self.confirmation_rate(),
        }

class ExperienceMemory:
    def __init__(self, eps=0.5, min_samples=5, outlier_threshold=2.5):
        self.db                = {}
        self.evidence_log      = []
        self.pseudo_labels     = []
        self.cluster_labels    = None
        self.cluster_profiles  = {}
        self.eps               = eps
        self.min_samples        = min_samples
        self.outlier_threshold = outlier_threshold

    def _sig(self, pn, pi, pl, d):
        return (round(pn,2), round(pi,2), round(pl,2), round(d,1))

    def lookup(self, pn, pi, pl, d):
        sig = self._sig(pn, pi, pl, d)
        if CACHE_DISABLED:      # still stores below; never serves a decision
            return None, sig
        return self.db.get(sig, None), sig

    def store(self, sig, decision, reason, confidence, ev_vec, pseudo_label=None):
        self.db[sig] = {'decision': decision, 'reason': reason, 'confidence': confidence}
        self.evidence_log.append(ev_vec)
        self.pseudo_labels.append(pseudo_label)

    def run_dbscan(self):
        if len(self.evidence_log) < self.min_samples * 2:
            return None
        E      = np.array(self.evidence_log)
        std    = np.std(E, axis=0) + 1e-6
        E_norm = E / std
        db     = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        self.cluster_labels = db.fit_predict(E_norm)
        self.cluster_profiles = {}
        for cid in set(self.cluster_labels):
            if cid == -1: continue
            mask = self.cluster_labels == cid
            pts  = E[mask]
            self.cluster_profiles[cid] = {
                'center': pts.mean(axis=0),
                'size':   int(mask.sum()),
                # share of members whose network log-odds L_net > 0
                # (p_net > 0.5): no label, verdict or outcome involved.
                # Feeds escalation energy (x0.3) and the deferred score R.
                'risk':   float(np.mean(pts[:,0] > 0))
            }
        nc = len(self.cluster_profiles)
        nn = int(list(self.cluster_labels).count(-1))
        print(f"\n[Memory DBSCAN] Clusters: {nc}  |  Novel anomalies (noise): {nn}")
        for cid, p in self.cluster_profiles.items():
            print(f"  Cluster {cid}: size={p['size']}  risk={p['risk']:.2f}")
        return self.cluster_labels

    def find_closest_cluster(self, ev_vec):
        if not self.cluster_profiles:
            return -1, 0.0, float('inf')
        ev = np.array(ev_vec)
        best_id, best_dist = -1, float('inf')
        for cid, prof in self.cluster_profiles.items():
            d = np.linalg.norm(ev - prof['center'])
            if d < best_dist:
                best_dist = d; best_id = cid
        if best_id == -1:
            return -1, 0.0, float('inf')
        raw_risk = self.cluster_profiles[best_id]['risk']
        # Cap cluster risk contribution — prevents memory dominating over ML
        # Large clusters carry full weight; small/sparse clusters are dampened
        cluster_size = self.cluster_profiles[best_id]['size']
        size_factor  = min(cluster_size / 20.0, 1.0)   # full weight at 20+ members
        capped_risk  = raw_risk * size_factor * 0.75    # max 75% of raw risk
        return best_id, capped_risk, best_dist

    def stats(self):
        n_pl = len([x for x in self.pseudo_labels if x is not None])
        print(f"\n[Memory] Cache={len(self.db)}  Evidence={len(self.evidence_log)}"
              f"  PseudoLabels={n_pl}  Clusters={len(self.cluster_profiles)}")

    def plot_clusters(self, save_path='plots/dbscan_clusters.png'):
        if self.cluster_labels is None or not self.evidence_log: return
        os.makedirs('plots', exist_ok=True)
        E  = np.array(self.evidence_log)
        plt.figure(figsize=(8,5))
        sc = plt.scatter(E[:,0], E[:,1], c=self.cluster_labels, cmap='tab10', alpha=0.5, s=8)
        plt.colorbar(sc, label='Cluster  (-1 = novel anomaly)')
        plt.xlabel('L_net'); plt.ylabel('L_iot')
        plt.title('Correlation Space — DBSCAN')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150); plt.close()
        print(f"  ✅ Cluster plot → {save_path}")


# ════════════════════════════════════════════════════════════════════
# SECTION 7 — OUTLIER DETECTION + HIERARCHICAL DECISION ENGINE
# ════════════════════════════════════════════════════════════════════

TAU_HIGH = 0.65   # default fallback (overridden by AdaptiveThreshold after training)
TAU_LOW  = 0.30   # default fallback


class AdaptiveThreshold:
    """
    Learned base thresholds and the drift update (paper §5.5).

    fit()    — tau_high maximises F-beta (beta = 1.5) on training scores;
               tau_low is placed below it. The run then overrides both with the
               cross-validation averages, and the floor TAU_FLOOR applies: in
               every fit of the paper the fitted tau_high (0.31-0.34) lies below
               the 0.35 floor, so the operating point in force is the floor.
    update() — called inside hierarchical_decision AFTER the threshold
               comparison. Advances the EWMA of scores (alpha 0.05); when it
               departs from the training-score mean by more than 0.08, resets
               tau_low to (fitted tau_low - 0.3 x departure), clipped to
               [0.05, 0.45] and kept >= 0.10 below tau_high. The reset value is
               read by the escalation gate of the same event and, as the
               starting point, by the next event's contextual adjustment. The
               tau_high it writes is overwritten by the next contextual
               adjustment before any decision uses it.
    """
    # Adversarial hardening constants
    MAX_SHIFT_PER_STEP = 0.04   # max tau change per EWMA update
    MIN_TAU_HIGH       = TAU_FLOOR   # hard floor (0.35 unless --tau_floor)
    MAX_TAU_HIGH       = 0.90   # hard ceiling

    def __init__(self, beta=1.5, ewma_alpha=0.05, drift_sensitivity=0.08):
        self.beta              = beta
        self.ewma_alpha        = ewma_alpha
        self.drift_sensitivity = drift_sensitivity
        self.tau_high          = TAU_HIGH
        self.tau_low           = TAU_LOW
        self._base_tau_high    = TAU_HIGH
        self._base_tau_low     = TAU_LOW
        self._ewma_score       = None
        self._baseline_mean    = None
        self._history          = []

    def fit(self, y_true, y_scores):
        from sklearn.metrics import fbeta_score
        best_tau, best_fb = 0.5, 0.0
        for tau in np.arange(0.30, 0.85, 0.01):
            preds = (y_scores >= tau).astype(int)
            fb    = fbeta_score(y_true, preds, beta=self.beta, zero_division=0)
            if fb > best_fb:
                best_fb = fb; best_tau = tau

        self.tau_high = round(float(best_tau), 3)

        # ORDERING CONSTRAINT: tau_low must always be < tau_high
        # and at least 0.15 below it to preserve an uncertainty zone
        raw_low = round(max(0.05, 1.0 - float(best_tau) - 0.10), 3)
        self.tau_low = min(raw_low, self.tau_high - 0.15)
        self.tau_low = round(max(0.05, self.tau_low), 3)

        self._base_tau_high = self.tau_high
        self._base_tau_low  = self.tau_low
        self._baseline_mean = float(np.mean(y_scores))
        self._ewma_score    = self._baseline_mean
        print(f"  [AdaptiveThreshold] Learned tau_high={self.tau_high}  "
              f"tau_low={self.tau_low}  gap={self.tau_high-self.tau_low:.3f}  "
              f"beta={self.beta}  F-beta={best_fb:.4f}")
        assert self.tau_low < self.tau_high, "Threshold ordering violated!"
        self._history.append((self.tau_high, self.tau_low, "initial fit"))
        return self

    def update(self, fusion_score):
        if self._ewma_score is None:
            self._ewma_score = fusion_score; return
        self._ewma_score = (self.ewma_alpha * fusion_score
                            + (1 - self.ewma_alpha) * self._ewma_score)
        if self._baseline_mean is None: return
        drift = self._ewma_score - self._baseline_mean
        if abs(drift) > self.drift_sensitivity:
            # Rate-limit: cap shift per step (adversarial hardening)
            raw_shift = drift * 0.5
            clamped_shift = float(np.clip(raw_shift, -self.MAX_SHIFT_PER_STEP,
                                           self.MAX_SHIFT_PER_STEP))
            new_h = float(np.clip(self._base_tau_high + clamped_shift,
                                   self.MIN_TAU_HIGH, self.MAX_TAU_HIGH))
            new_l = float(np.clip(self._base_tau_low  - drift * 0.3, 0.05, 0.45))
            # ORDERING CONSTRAINT: maintain minimum gap during drift
            if new_l >= new_h - 0.10:
                new_l = new_h - 0.15
            if abs(new_h - self.tau_high) > 0.02:
                self._history.append((round(new_h,3), round(new_l,3),
                                      f"drift={drift:+.3f}"))
            self.tau_high = round(new_h, 3)
            self.tau_low  = round(max(0.05, new_l), 3)

    def status(self):
        return {
            "tau_high":    self.tau_high,
            "tau_low":     self.tau_low,
            "baseline_mean": round(self._baseline_mean,4) if self._baseline_mean else None,
            "ewma_now":    round(self._ewma_score,4)    if self._ewma_score    else None,
            "drift":       round(self._ewma_score - self._baseline_mean, 4)
                           if self._baseline_mean and self._ewma_score else 0,
            "adjustments": len(self._history) - 1,
            "beta":        self.beta,
        }

    def save(self, path='models/'):
        os.makedirs(path, exist_ok=True)
        joblib.dump(self, f'{path}adaptive_threshold.pkl')
        print(f"  ✅ Adaptive threshold saved → {path}adaptive_threshold.pkl "
              f"(tau_high={self.tau_high}, tau_low={self.tau_low})")

    @staticmethod
    def load(path='models/'):
        return joblib.load(f'{path}adaptive_threshold.pkl')


def is_outlier(ev_vec, memory):
    if memory.cluster_labels is None or not memory.evidence_log:
        return False
    E       = np.array(memory.evidence_log)
    std     = np.std(E, axis=0) + 1e-6
    ev_norm = np.array(ev_vec) / std
    centroid= (E / std).mean(axis=0)
    return np.linalg.norm(ev_norm - centroid) > memory.outlier_threshold


# ════════════════════════════════════════════════════════════════════
# ADAPTIVE POLICY FUSION ENGINE (APFE)
#
# Architecture:
#   Multi-Layer Signals
#        ↓
#   Contextual Risk Scoring  (UnifiedRiskScorer)
#        ↓
#   Policy Threshold Governance  (PolicyEngine)
#        ↓
#   Adaptive Threshold Decision  (AdaptiveThreshold + hierarchical_decision)
#        ↓
#   Final Risk Verdict
#
# Risk is NOT just P(attack). It integrates:
#   probability + severity + policy violation + temporal context +
#   cluster history + sensor stability + behavioral deviation
# ════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────
# LEVEL 1 — POLICY THRESHOLD REGISTRY
# Hard per-feature limits that trigger risk escalation
# regardless of what the ML model says
# ─────────────────────────────────────────────────────────────────
POLICY_THRESHOLDS = {
    "network": {
        "src_bytes":     {"warning": 50_000,  "critical": 200_000, "risk_add": 0.20},
        "net_duration":  {"warning": 0.0,     "critical": 0.0,     "risk_add": 0.00},
        "dst_bytes":     {"warning": 30_000,  "critical": 100_000, "risk_add": 0.15},
    },
    "iot": {
        "phys_delta":    {"warning": 3.0,     "critical": 7.0,     "risk_add": 0.25},
        "phys_delta_abs":{"warning": 3.0,     "critical": 7.0,     "risk_add": 0.25},
        "temp_roll_std": {"warning": 2.0,     "critical": 5.0,     "risk_add": 0.15},
    },
    "linux": {
        "disk_write":    {"warning": 500,     "critical": 2000,    "risk_add": 0.20},
        "cpu_usage":     {"warning": 80,      "critical": 95,      "risk_add": 0.20},
        "mem_vgrow":     {"warning": 1000,    "critical": 5000,    "risk_add": 0.25},
        "page_faults_maj":{"warning": 10,     "critical": 50,      "risk_add": 0.20},
        "cpu_spike":     {"warning": 30,      "critical": 60,      "risk_add": 0.15},
    },
    "windows": {
        "cpu_usage":     {"warning": 85,      "critical": 95,      "risk_add": 0.20},
        "disk_write":    {"warning": 1e6,     "critical": 5e6,     "risk_add": 0.20},
        "process_load":  {"warning": 0.7,     "critical": 0.9,     "risk_add": 0.15},
    },
}

# ─────────────────────────────────────────────────────────────────
# MULTI-STAGE THREAT LEVELS
# Replaces binary ATTACK/NORMAL with 5-level graduated verdicts
# ─────────────────────────────────────────────────────────────────
THREAT_LEVELS = {
    0: {"name": "SAFE",          "color": "GREEN",  "action": "log only"},
    1: {"name": "SUSPICIOUS",    "color": "YELLOW", "action": "monitor"},
    2: {"name": "ELEVATED",      "color": "ORANGE", "action": "alert SOC"},
    3: {"name": "CRITICAL",      "color": "RED",    "action": "isolate"},
    4: {"name": "FORCED_ACTION", "color": "BLACK",  "action": "block immediately"},
}

def score_to_threat_level(fusion_score, policy_risk, forced=False):
    """
    Maps fusion score + policy risk → 5-level threat verdict.
    Returns (level_int, level_name, action)
    """
    if forced:
        return 4, "FORCED_ACTION", "block immediately"
    combined = fusion_score * 0.7 + policy_risk * 0.3
    if combined >= 0.62:   return 3, "CRITICAL",      "isolate"
    if combined >= 0.55:   return 2, "ELEVATED",      "alert SOC"
    if combined >= 0.38:   return 1, "SUSPICIOUS",    "monitor"
    return 0, "SAFE", "log only"

def threat_to_binary(level):
    """For evaluation: ELEVATED+ = attack (1), SAFE/SUSPICIOUS = normal (0)."""""
    return 1 if level >= 2 else 0

class ThreatStateMachine:
    """
    Energy-based threat accumulator.
    energy[t] = DECAY * energy[t-1] + (1-DECAY) * current_risk

    Cluster risk + policy feed the ENERGY (escalation),
    NOT the detection threshold. No positive feedback loop.
    """
    ENERGY_DECAY  = 0.88
    FORCED_STICKY = 3
    THRESHOLDS    = {3: 0.62, 2: 0.42, 1: 0.28}

    def __init__(self):
        self.energy       = 0.0
        self.state        = 0
        self._forced_ttl  = 0
        self._history     = []
        self._peak_energy = 0.0

    def update(self, event_level, cluster_risk=0.0, policy_risk=0.0):
        if event_level == 4:
            self.state = 4; self._forced_ttl = self.FORCED_STICKY
            self.energy = 1.0
            self._history.append((event_level, self.state, self.energy))
            return self.state
        if self._forced_ttl > 0:
            self._forced_ttl -= 1
            self._history.append((event_level, self.state, self.energy))
            return self.state

        level_score  = event_level / 4.0
        escalation   = 0.3 * cluster_risk + 0.2 * policy_risk
        current_risk = float(np.clip(level_score + escalation, 0.0, 1.0))
        self.energy  = float(np.clip(
            self.ENERGY_DECAY * self.energy + (1-self.ENERGY_DECAY) * current_risk, 0, 1))
        self._peak_energy = max(self._peak_energy, self.energy)

        new_state = 0
        for lvl in sorted(self.THRESHOLDS.keys(), reverse=True):
            if self.energy >= self.THRESHOLDS[lvl]:
                new_state = lvl; break
        self.state = new_state
        self._history.append((event_level, self.state, round(self.energy, 3)))
        return self.state

    def current_name(self):
        return THREAT_LEVELS[self.state]["name"]

    def stats(self):
        if not self._history: return {"energy_fsm": "no events"}
        from collections import Counter
        hist_states = [s for _,s,_ in self._history]
        dist = Counter(THREAT_LEVELS[s]["name"] for s in hist_states)
        return {"final_state":    THREAT_LEVELS[self.state]["name"],
                "current_energy": round(self.energy, 4),
                "peak_energy":    round(self._peak_energy, 4),
                "distribution":   dict(dist),
                "avg_energy":     round(sum(e for _,_,e in self._history)/max(len(self._history),1), 3)}


class ScanEnergy:
    """
    Per-source scan accumulator.

    A scan is one host, many probes, short window. The global FSM pools every
    attack class over the whole stream, so it cannot express that. This keeps a
    small decaying score per source host, fed only by the three features that
    describe scanning: destination-port fanout, port entropy, and the fraction
    of connections that were refused or never answered.

        e[src] <- DECAY * e[src] + (0.5*fanout + 0.3*entropy + 0.2*fail_rate)

    DECAY 0.90 gives a memory of roughly ten events from the same host, so
    TRIGGER 1.50 means "about three scan-like probes recently". When a host is
    over the trigger, its detection threshold drops by RELIEF — for that host
    only, and only while the evidence persists.
    """
    DECAY   = 0.90
    TRIGGER = SCAN_TRIGGER    # was a fixed 1.50, unreachable
    RELIEF  = 0.08
    FLOOR   = 0.15      # tau_high may not fall below this
    CAP     = 4.0

    def __init__(self):
        self.pool = {}
        self.relief_applied = 0
        self.peak = 0.0
        self.seen = 0

    @staticmethod
    def evidence(fanout, entropy, fail_rate):
        return (0.5 * float(fanout) + 0.3 * float(entropy) + 0.2 * float(fail_rate))

    def observe(self, src, fanout, entropy, fail_rate):
        self.seen += 1
        ev = self.evidence(fanout, entropy, fail_rate)
        e  = min(self.pool.get(src, 0.0) * self.DECAY + ev, self.CAP)
        self.pool[src] = e
        if e > self.peak:
            self.peak = e
        return e

    def relief(self, src):
        """Threshold relief for this host, 0.0 if it is not scanning."""
        if self.pool.get(src, 0.0) >= self.TRIGGER:
            self.relief_applied += 1
            return self.RELIEF
        return 0.0

    def stats(self):
        vals = sorted(self.pool.values())
        hot = [v for v in vals if v >= self.TRIGGER]
        def q(p):
            if not vals:
                return 0.0
            return round(float(vals[min(int(p / 100 * len(vals)), len(vals) - 1)]), 3)
        return {"enabled":            SCAN_FSM,
                "sources_tracked":    len(self.pool),
                "energy p50/p75/p90": f"{q(50)}/{q(75)}/{q(90)}",
                "energy p95/p99/max": f"{q(95)}/{q(99)}/{round(self.peak,3)}",
                "trigger":            self.TRIGGER,
                "sources_over_trigger": len(hot),
                "relief_applied":     self.relief_applied,
                "events_observed":    self.seen,
                "decay/relief":       f"{self.DECAY}/{self.RELIEF}",
                "note": "per-source accumulator; lowers tau_high for that host only"}


# Critical port set — network connections to these ports raise risk
CRITICAL_PORTS = {22, 23, 445, 3389, 1433, 3306, 5432, 6379, 27017}


class PolicyEngine:
    """
    Level 1-3 of the threshold stack:
      L1 — Field thresholds: per-feature warning/critical limits
      L2 — Layer thresholds: per-modality aggregate risk
      L3 — Correlation rules: cross-layer patterns (network+disk+iot)

    Returns (policy_risk, policy_flags) where:
      policy_risk  : additional risk to add to ML fusion score
      policy_flags : list of triggered rule descriptions
    """

    def __init__(self, thresholds=None):
        self.thresholds = thresholds or POLICY_THRESHOLDS
        self._violation_log = []   # running history of policy hits

    def check_field(self, domain, field, value):
        """
        Dynamic severity: severity = impact × confidence × persistence
          impact      = how far value exceeds threshold (normalised)
          confidence  = 1.0 for critical, 0.5 for warning
          persistence = amplified by _violation_log recurrence
        """
        pol = self.thresholds.get(domain, {}).get(field)
        if pol is None or pol["critical"] == 0:
            return 0.0, None

        if value >= pol["critical"]:
            impact     = min((value - pol["critical"]) / (pol["critical"] + 1e-6) + 1.0, 3.0)
            confidence = 1.0
            # Persistence: how many times has this field triggered recently?
            recent_key = f"{domain}.{field}"
            recent_hits = sum(1 for v in self._violation_log[-10:]
                              for f in v.get("flags",[]) if recent_key in f)
            persistence = 1.0 + 0.1 * min(recent_hits, 5)
            severity = pol["risk_add"] * impact * confidence * persistence
            msg = (f"POLICY_CRITICAL [{domain}.{field}={value:.1f} >= {pol['critical']}  "
                   f"sev={severity:.3f}]")
            return min(severity, pol["risk_add"] * 2.0), msg

        if value >= pol["warning"] and pol["warning"] > 0:
            impact     = min((value - pol["warning"]) / (pol["critical"] - pol["warning"] + 1e-6), 1.0)
            confidence = 0.5
            severity   = pol["risk_add"] * impact * confidence
            msg = (f"POLICY_WARNING [{domain}.{field}={value:.1f} >= {pol['warning']}  "
                   f"sev={severity:.3f}]")
            return severity, msg

        return 0.0, None

    def evaluate(self, row: dict) -> tuple:
        """
        Evaluate all policy rules against a feature row.
        row: flat dict of all features for this event.

        Returns (total_policy_risk, flags_list)
        """
        total_risk = 0.0
        flags = []

        # L1 — Field-level checks
        for domain, fields in self.thresholds.items():
            for field in fields:
                val = row.get(field, 0.0)
                if val is None: val = 0.0
                delta, msg = self.check_field(domain, field, float(val))
                if msg:
                    total_risk += delta
                    flags.append(msg)

        # L2 — Layer-level: if 2+ fields in same layer trigger, multiply risk
        domain_hits = {}
        for f in flags:
            dom = f.split("[")[1].split(".")[0]
            domain_hits[dom] = domain_hits.get(dom, 0) + 1
        for dom, hits in domain_hits.items():
            if hits >= 2:
                total_risk *= 1.20   # 20% amplification per multi-field domain hit
                flags.append(f"POLICY_LAYER_AMP [{dom}: {hits} fields hit]")

        # L3 — Correlation rules: cross-layer patterns
        net_high   = row.get("p_net", 0)    > 0.70
        iot_unstbl = row.get("phys_delta_abs", 0) > 3.0
        disk_high  = row.get("disk_write", 0)  > 500
        cpu_high   = row.get("cpu_usage", 0)   > 80

        if net_high and disk_high:
            total_risk += 0.15
            flags.append("POLICY_CORR [network+disk: possible exfiltration]")
        if net_high and iot_unstbl:
            total_risk += 0.20
            flags.append("POLICY_CORR [network+iot: possible cyber-physical attack]")
        if disk_high and cpu_high:
            total_risk += 0.15
            flags.append("POLICY_CORR [disk+cpu: possible ransomware/mining]")
        if net_high and disk_high and iot_unstbl:
            total_risk += 0.25
            flags.append("POLICY_CORR [network+disk+iot: multi-layer attack signature]")

        # Log violations
        if flags:
            self._violation_log.append({"flags": flags, "risk": total_risk})

        return min(total_risk, 0.60), flags   # cap policy contribution at 0.60

    def stats(self):
        total = len(self._violation_log)
        if total == 0:
            return {"violations": 0}
        all_flags = [f for v in self._violation_log for f in v["flags"]]
        from collections import Counter
        top = Counter(all_flags).most_common(5)
        return {"violations": total, "top_rules": top}


class UnifiedRiskScorer:
    """
    Implements the unified risk equation:

      R = w_n*N + w_i*I + w_l*L + w_w*W + w_c*C + w_t*T + w_p*P

    Where:
      N = network anomaly score    (from ML)
      I = IoT anomaly score        (from ML)
      L = Linux anomaly score      (from ML)
      W = Windows stress           (from features)
      C = cluster historical risk  (from DBSCAN memory)
      T = temporal anomaly         (rolling deviation)
      P = policy violation risk    (from PolicyEngine)

    Then applies:
      - Risk floor enforcement (policy can't be overridden by ML)
      - Confidence dampening  (penalise layer disagreement)
      - Final clamping to [0, 1]
    """

    WEIGHTS = {
        "network":  0.35,   # strongest signal for TON_IoT
        "iot":      0.20,
        "linux":    0.15,
        "windows":  0.10,
        "cluster":  0.10,
        "temporal": 0.05,
        "policy":   0.05,   # policy adds on top — see risk_floor below
    }

    def __init__(self, policy_engine=None):
        self.policy = policy_engine or PolicyEngine()
        self._score_history = []   # for temporal anomaly

    def score(self, p_net, p_iot, p_log, phys_delta,
              system_stress, cluster_risk, row_features,
              adaptive_tau=None):
        """
        Compute unified risk score R for one event.

        Returns (R, breakdown_dict, policy_flags)
        """
        w = self.WEIGHTS

        # Component scores
        N = float(p_net)
        I = float(p_iot)
        L = float(p_log)
        W = min(float(system_stress) / 100.0, 1.0) if system_stress > 0 else float(p_log)
        C = float(cluster_risk)

        # Temporal anomaly: deviation of phys_delta from recent history
        self._score_history.append(abs(phys_delta))
        if len(self._score_history) > 20:
            self._score_history.pop(0)
        recent_mean = float(np.mean(self._score_history))
        T = min(abs(phys_delta) / (recent_mean + 1e-6) / 10.0, 1.0)

        # Policy risk
        P_risk, flags = self.policy.evaluate(row_features)

        # Weighted sum
        R = (w["network"]  * N +
             w["iot"]      * I +
             w["linux"]    * L +
             w["windows"]  * W +
             w["cluster"]  * C +
             w["temporal"] * T +
             w["policy"]   * P_risk)

        # Confidence dampening: penalise when layers strongly disagree
        scores_used = [N, I, L]
        layer_std   = float(np.std(scores_used))
        if layer_std > 0.30:
            R *= (1.0 - 0.15 * (layer_std - 0.30))  # gentle dampening

        # Risk floor: policy violations set a minimum risk
        # Even if ML says normal, strong policy hits raise the floor
        if P_risk > 0.25:
            R = max(R, 0.45)   # suspicious zone at minimum
        if P_risk > 0.45:
            R = max(R, 0.65)   # attack zone if policy is critical

        R = float(np.clip(R, 0.0, 1.0))

        breakdown = {
            "R_total":   round(R, 4),
            "N_network": round(N, 3),
            "I_iot":     round(I, 3),
            "L_linux":   round(L, 3),
            "W_windows": round(W, 3),
            "C_cluster": round(C, 3),
            "T_temporal":round(T, 3),
            "P_policy":  round(P_risk, 3),
            "layer_std": round(layer_std, 3),
        }
        return R, breakdown, flags


def hierarchical_decision(p_net, p_iot, p_log, phys_delta,
                           fusion_score, evidence_vec, memory, adaptive_tau=None):
    # ── L4: Policy hard overrides (bypass ML entirely) ────────────
    if p_net > 0.95:
        return 1, "OVERRIDE: Network critical (p_net>0.95)", float(p_net)
    if p_iot > 0.90 and abs(phys_delta) > 5.0:
        return 1, f"OVERRIDE: Physical tampering (Δ={phys_delta:.1f})", float(p_iot)

    # ── L5: Adaptive global threshold (uses learned tau + EWMA drift) ──
    tau_h = adaptive_tau.tau_high if adaptive_tau else TAU_HIGH
    tau_l = adaptive_tau.tau_low  if adaptive_tau else TAU_LOW
    if adaptive_tau: adaptive_tau.update(fusion_score)

    if fusion_score > tau_h:
        return 1, f"FUSION_ALERT: Attack (score={fusion_score:.2f}, tau={tau_h:.2f})", float(fusion_score)
    if fusion_score < tau_l:
        return 0, f"FUSION_NORMAL: Baseline (score={fusion_score:.2f}, tau={tau_l:.2f})", float(1-fusion_score)

    # ── Suspicious zone: cluster + outlier resolution ──────────────
    cid, cluster_risk, dist = memory.find_closest_cluster(evidence_vec)
    if cid != -1 and cluster_risk > 0.70:
        return 1, f"CLUSTER_MATCH: Attack cluster {cid} (risk={cluster_risk:.2f})", cluster_risk
    if is_outlier(evidence_vec, memory):
        return 1, f"ANOMALY: Novel pattern outside correlation space", float(fusion_score)
    return 0, f"SUSPICIOUS_SAFE: Uncertain, close to safe cluster (score={fusion_score:.2f})", float(1-fusion_score)


# ════════════════════════════════════════════════════════════════════
# MODALITY TRUST SCORES
# Trust is reduced when a modality has been noisy or unreliable
# recently. Low-trust layer probs are dampened before fusion.
# ════════════════════════════════════════════════════════════════════

class ModalityTrustTracker:
    """
    Tracks per-modality reliability based on consistency of predictions.
    A modality that frequently contradicts the final decision loses trust.
    Trust is applied as a multiplier on layer probabilities before scoring.

    Measured behaviour (paper §5.5): across all five folds the three active
    modalities sit at 1.000 and the unused fourth at its default 0.700, so the
    damping applied is the identity. No reported result is attributable to it.
    """
    def __init__(self, decay=0.99, min_trust=0.45):
        self.trust   = {"net": 0.70, "iot": 0.70, "log": 0.70, "win": 0.70}
        self.decay   = decay
        self.min_trust = min_trust
        self._counts = {"net": 0, "iot": 0, "log": 0, "win": 0}
        self._hits   = {"net": 0, "iot": 0, "log": 0, "win": 0}

    def update(self, final_decision, p_net, p_iot, p_log):
        """
        Update trust with temporal decay:
          trust_t = decay * trust_{t-1} + (1-decay) * agreement_rate

        Repeated failures degrade trust quickly (decay=0.98 per event).
        Consistent accuracy restores trust gradually.
        Min trust floor prevents any modality from being fully ignored.
        """
        threshold = 0.5
        agreed = {
            "net": int(p_net > threshold) == final_decision,
            "iot": int(p_iot > threshold) == final_decision,
            "log": int(p_log > threshold) == final_decision,
        }
        for mod, ok in agreed.items():
            self._counts[mod] += 1
            if ok: self._hits[mod] += 1
            # Asymmetric trust: agreement recovers slowly, disagreement punishes gently
            # This prevents permanently distrusting noisy-but-useful modalities
            if TRUST_EMA:
                # As specified above: trust_t = decay*trust_{t-1} + (1-decay)*agreement,
                # floored at min_trust. (The legacy rule below gains 0.015 per
                # agreement and loses 0.008 per disagreement, so any modality
                # agreeing more than 35% of the time sits at the 1.0 cap.)
                self.trust[mod] = max(self.min_trust, self.decay * self.trust[mod]
                                      + (1 - self.decay) * float(ok))
            elif ok:
                # Slow recovery toward 1.0 — gain 0.015 per agreement
                self.trust[mod] = min(1.0, self.trust[mod] + 0.015)
            else:
                # Gentle loss — lose 0.008 per disagreement
                self.trust[mod] = max(self.min_trust, self.trust[mod] - 0.008)

    def apply(self, p_net, p_iot, p_log):
        """Dampen layer probs by their current trust score."""
        return (p_net * self.trust["net"],
                p_iot * self.trust["iot"],
                p_log * self.trust["log"])

    def status(self):
        return {k: round(v, 3) for k, v in self.trust.items()}


# ════════════════════════════════════════════════════════════════════
# CONTEXTUAL THRESHOLD ENGINE
# tau is no longer a single number — it shifts based on live context:
#   cluster_risk       → higher risk clusters → lower tau
#   policy_violations  → more violations → lower tau (more sensitive)
#   layer_disagreement → more disagreement → raise tau (more cautious)
#   anomaly_density    → more anomalies recently → lower tau
#   ewma_drift         → score distribution shift → handled by EWMA
# ════════════════════════════════════════════════════════════════════

class ContextualThresholdEngine:
    """
    Contextual adjustment of the thresholds, per event (paper §5.5).

      D = clip(|EWMA(S) - baseline|, 0, 1)   baseline = training-score mean (~0.5)
      U = std of the three modality probabilities
      t = (1 - mean trust) * 0.04
      r = -ALPHA*D + DELTA*U + t ;  adj = 0.12 * tanh(r / 0.12) ;  adj = 0 if |adj| < 0.03
      tau_high = clip(T_base + adj, TAU_FLOOR, 0.92)
      tau_low  = clip(tau_low' - 0.3*adj, 0.05, tau_high - 0.12)

    tau_low' is the tau_low left in force by the previous event (after its
    drift update), so tau_low carries over between events within its clip,
    while tau_high is recomputed from T_base every event. Cluster risk and
    policy pressure are NOT in this equation: they feed the escalation energy.
    BETA and GAMMA are kept for the tuning grid only. D is symmetric, so it
    rises with the local share of high-scoring events in either direction; on
    TON_IoT that is chiefly the local attack mix. auto_tune() scores the grid
    with fixed D = 0.04 and U = 0.24 rather than measured values.
    """

    # ── Equation coefficients ────────────────────────────────────────
    # Increase to make that factor have more influence on the threshold
    ALPHA = 0.40   # drift sensitivity weight
    BETA  = 0.25   # cluster risk weight
    GAMMA = 0.15   # policy pressure weight
    DELTA = 0.20   # uncertainty (disagreement) weight

    # Per-modality threshold offsets — each layer can have its own sensitivity
    # Negative = more sensitive for that modality, positive = less sensitive
    MODALITY_OFFSETS = {
        "net": -0.02,   # network is most reliable — slightly more aggressive
        "iot": +0.02,   # IoT is noisier — slightly more conservative
        "log": +0.05,   # log layer is weakest — more conservative
        "win":  0.00,   # Windows telemetry — neutral
    }

    def __init__(self, adaptive_tau: "AdaptiveThreshold"):
        self.base      = adaptive_tau
        self._adj_log  = []
        self._equation_calls = 0
        # Per-modality tau tracking
        self.modality_taus = {m: adaptive_tau.tau_high + off
                              for m, off in self.MODALITY_OFFSETS.items()}

    def get_tau(self, cluster_risk=0.0, policy_violation_count=0,
                layer_disagreement=0.0, anomaly_density=0.0,
                trust_scores=None):
        """
        DETECTION THRESHOLD — drift + uncertainty + trust only.
        Cluster risk and policy moved to EscalationEngine (no feedback loop).
        Hysteresis: changes < 0.03 are ignored to prevent jitter.
        """
        self._equation_calls += 1
        T_base = self.base._base_tau_high  # anchor to learned value, not drifting tau
        EPSILON = 0.03

        D = float(np.clip(abs(self.base._ewma_score - self.base._baseline_mean), 0, 1)) \
            if self.base._baseline_mean and self.base._ewma_score else 0.0
        U = float(np.clip(layer_disagreement, 0.0, 1.0))
        trust_term = (1.0 - float(np.mean(list(trust_scores.values())))) * 0.04 \
                     if trust_scores else 0.0

        # Bounded sigmoid (tanh): saturates gracefully — max ±SIGMA shift
        # Small inputs behave linearly; large inputs can't push tau to extremes
        SIGMA  = 0.12
        raw    = - self.ALPHA * D + self.DELTA * U + trust_term
        adjustment = float(np.tanh(raw / SIGMA) * SIGMA)
        if abs(adjustment) < EPSILON:
            adjustment = 0.0

        T_eff_high = round(float(np.clip(T_base + adjustment, TAU_FLOOR, 0.92)), 3)
        T_eff_low  = round(max(0.05, float(np.clip(
            self.base.tau_low - adjustment * 0.3, 0.05, T_eff_high - 0.12))), 3)

        terms = []
        if D > 0.01: terms.append(f"-{self.ALPHA}*D({D:.2f})={-self.ALPHA*D:+.3f}")
        if U > 0.01: terms.append(f"+{self.DELTA}*U({U:.2f})={+self.DELTA*U:+.3f}")
        eq_str = f"T={T_base:.3f}"+"".join(terms)+f"={T_eff_high:.3f}"

        if abs(adjustment) > 0.005:
            self._adj_log.append({"adj":round(adjustment,4),"D":round(D,3),
                                   "U":round(U,3),"tau_h":T_eff_high,"eq":eq_str})

        for mod, base_off in self.MODALITY_OFFSETS.items():
            self.modality_taus[mod] = float(np.clip(T_eff_high + base_off, 0.30, 0.92))

        return T_eff_high, T_eff_low, eq_str

    def auto_tune(self, y_true, fusion_scores, sample_stats=None):
        from sklearn.metrics import fbeta_score
        from itertools import product
        D = (sample_stats or {}).get("D", 0.04)
        C = (sample_stats or {}).get("C", 0.37)
        P = (sample_stats or {}).get("P", 0.14)
        U = (sample_stats or {}).get("U", 0.24)
        best_fb, best = 0.0, (self.ALPHA, self.BETA, self.GAMMA, self.DELTA)
        for a,b,g,d in product([0.2,0.4,0.6],[0.15,0.25,0.35],[0.05,0.15,0.25],[0.10,0.15,0.20]):
            adj   = -a*D + d*U   # detection only: no C or P (those go to energy)
            t_sim = float(np.clip(self.base.tau_high + adj, 0.30, 0.90))
            preds = (fusion_scores >= t_sim).astype(int)
            fb    = fbeta_score(y_true, preds, beta=1.5, zero_division=0)
            if fb > best_fb: best_fb = fb; best = (a,b,g,d)
        self.ALPHA,self.BETA,self.GAMMA,self.DELTA = best
        print(f"  [AutoTune] a={self.ALPHA} b={self.BETA} g={self.GAMMA} d={self.DELTA}  F1.5={best_fb:.4f}")
        return best

    def equation_str(self):
        return (f"T_detect = T_base - {self.ALPHA}·D + {self.DELTA}·U + trust  "
                f"[C+P → EscalationEngine energy]")

    def stats(self):
        if not self._adj_log:
            return {"equation": self.equation_str(),
                    "contextual_adjustments": 0,
                    "alpha": self.ALPHA, "beta": self.BETA,
                    "gamma": self.GAMMA, "delta": self.DELTA}
        total  = len(self._adj_log)
        avg    = round(sum(a["adj"] for a in self._adj_log) / total, 4)
        d_mean = round(sum(a.get("D",0) for a in self._adj_log) / total, 3)
        u_mean = round(sum(a.get("U",0) for a in self._adj_log) / total, 3)
        return {
            "equation":              self.equation_str(),
            "alpha_delta":           f"α={self.ALPHA} δ={self.DELTA}",
            "contextual_adjustments":total,
            "avg_adjustment":        avg,
            "avg_D_drift":           d_mean,
            "avg_U_uncertainty":     u_mean,
            "last_equation":         self._adj_log[-1]["eq"] if self._adj_log else "—",
            "modality_taus":         {k: round(v,3) for k,v in self.modality_taus.items()},
            "note":                  "C+P moved to EscalationEngine energy",
        }


# ════════════════════════════════════════════════════════════════════
# FIX 5 — FORCED POLICY RULES
# Hard rules that fire regardless of ML — some events must NEVER
# rely purely on probability. Returns (force_attack, force_reason).
# ════════════════════════════════════════════════════════════════════

class SiteCalibration:
    """
    Stage 5 set the way a site sets its bounds (configuration D). Every limit is
    a quantile of the field's own NORMAL values in the training part (normal by
    that modality's own label), in the field's own units; device fields are set
    per device. Nothing is fitted on the records being scored.
      envelopes  a record beyond any ENVELOPE_Q limit is forced to alert:
                 host page_faults_maj, mem_vgrow, cpu_usage, disk_write (nonzero
                 normal values); per device the size of the change and the
                 reading's range; network source bytes
      limits     warning / critical at LIMIT_Q_WARN / LIMIT_Q_CRIT, with the risk
                 weights of POLICY_THRESHOLDS, for the policy pressure
    """
    HOST = ('page_faults_maj', 'mem_vgrow', 'cpu_usage', 'disk_write')
    SOFT = (('network', 'net_bytes', 0.20), ('network', 'net_dbytes', 0.15),
            ('iot', 'phys_delta_abs', 0.25), ('iot', 'temp_roll_std', 0.15),
            ('linux', 'disk_write', 0.20), ('linux', 'cpu_usage', 0.20),
            ('linux', 'mem_vgrow', 0.25), ('linux', 'page_faults_maj', 0.20),
            ('linux', 'cpu_spike', 0.15))

    def __init__(self, train_df):
        t = train_df
        col = lambda df, c: (df[c].to_numpy(float) if c in df.columns
                             else np.zeros(len(df)))
        normal = lambda c: ((t[c] == 0) if c in t.columns
                            else pd.Series(True, index=t.index))
        qs = lambda v, p: float(np.quantile(v, p)) if len(v) else float('inf')
        nz = lambda v: v[v != 0]
        h, n = t[normal('log_label')], t[normal('net_label')]
        present = ((t['iot_present'] == 1) if 'iot_present' in t.columns
                   else pd.Series(True, index=t.index))
        dv = t[normal('iot_label') & present]
        dev = (dv['iot_device'] if 'iot_device' in dv.columns
               else pd.Series(0, index=dv.index)).astype(int)
        q, qw, qc = ENVELOPE_Q, LIMIT_Q_WARN, LIMIT_Q_CRIT
        self.env_host = {f: qs(nz(col(h, f)), q) for f in self.HOST if f in t.columns}
        self.env_net = qs(col(n, 'net_bytes'), q)
        self.env_dev = {}
        base = {'network': {}, 'iot': {}, 'linux': {}}
        for dom, f, w in self.SOFT:
            if dom != 'iot':
                v = col(n if dom == 'network' else h, f)
                v = nz(v) if dom == 'linux' else v
                base[dom][f] = {'warning': qs(v, qw), 'critical': qs(v, qc), 'risk_add': w}
        self.limits_default, self.limits = base, {}
        for d, g in dv.groupby(dev):
            self.env_dev[int(d)] = (qs(np.abs(col(g, 'phys_delta')), q),
                                    qs(col(g, 'current_temp'), 1 - q),
                                    qs(col(g, 'current_temp'), q))
            th = {k: dict(v) for k, v in base.items()}
            th['iot'] = {f: {'warning': qs(col(g, f), qw), 'critical': qs(col(g, f), qc),
                             'risk_add': w} for dom, f, w in self.SOFT if dom == 'iot'}
            self.limits[int(d)] = th

    def envelope_hits(self, df):
        """(forced, rule) for every record of df, in df's order."""
        rule = np.full(len(df), '', dtype=object)

        def mark(mask, name):
            rule[mask & (rule == '')] = name
        for f, thr in self.env_host.items():
            mark(df[f].to_numpy(float) > thr, f'host:{f}')
        dev = (df['iot_device'].to_numpy(float) if 'iot_device' in df.columns
               else np.zeros(len(df)))
        chg = np.abs(df['phys_delta'].to_numpy(float))
        val = df['current_temp'].to_numpy(float)
        on_all = df['iot_present'].to_numpy(float) == 1
        for d, (c, lo, hi) in self.env_dev.items():
            name = _IOT_DEVICE_NAMES.get(d, str(d))
            on = on_all & (dev == d)
            mark(on & (chg > c), f'device:{name}:change')
            mark(on & ((val < lo) | (val > hi)), f'device:{name}:range')
        mark(df['net_bytes'].to_numpy(float) > self.env_net, 'net:bytes')
        return rule != '', rule


class SitePolicyEngine(PolicyEngine):
    """The policy pressure with site-calibrated limits (configuration D): the
    same field checks (L1), layer amplification (L2) and cross-layer
    correlations (L3) as PolicyEngine, every limit read from SiteCalibration
    (per device for the device fields) instead of POLICY_THRESHOLDS."""

    def __init__(self, cal):
        super().__init__(thresholds=cal.limits_default)
        self.cal = cal

    def evaluate(self, row: dict) -> tuple:
        dev = int(float(row.get('iot_device', 0) or 0))
        th = self.cal.limits.get(dev, self.cal.limits_default)
        self.thresholds = th
        total_risk, flags = 0.0, []
        for domain, fields in th.items():
            for field in fields:
                val = row.get(field, 0.0)
                delta, msg = self.check_field(domain, field, float(0.0 if val is None else val))
                if msg:
                    total_risk += delta
                    flags.append(msg)
        domain_hits = {}
        for f in flags:
            dom = f.split("[")[1].split(".")[0]
            domain_hits[dom] = domain_hits.get(dom, 0) + 1
        for dom, hits in domain_hits.items():
            if hits >= 2:
                total_risk *= 1.20
                flags.append(f"POLICY_LAYER_AMP [{dom}: {hits} fields hit]")
        lim = lambda dom, f: th.get(dom, {}).get(f, {}).get('warning', float('inf'))
        net_high   = row.get("p_net", 0) > 0.70
        iot_unstbl = row.get("phys_delta_abs", 0) > lim('iot', 'phys_delta_abs')
        disk_high  = row.get("disk_write", 0) > lim('linux', 'disk_write')
        cpu_high   = row.get("cpu_usage", 0) > lim('linux', 'cpu_usage')
        if net_high and disk_high:
            total_risk += 0.15
            flags.append("POLICY_CORR [network+disk: possible exfiltration]")
        if net_high and iot_unstbl:
            total_risk += 0.20
            flags.append("POLICY_CORR [network+iot: possible cyber-physical attack]")
        if disk_high and cpu_high:
            total_risk += 0.15
            flags.append("POLICY_CORR [disk+cpu: possible ransomware/mining]")
        if net_high and disk_high and iot_unstbl:
            total_risk += 0.25
            flags.append("POLICY_CORR [network+disk+iot: multi-layer attack signature]")
        if flags:
            self._violation_log.append({"flags": flags, "risk": total_risk})
        return min(total_risk, 0.60), flags


def check_forced_policies(row_features: dict) -> tuple:
    """
    Returns (forced: bool, reason: str)
    These rules bypass ALL ML reasoning when triggered.
    """
    pf_maj  = row_features.get("page_faults_maj", 0)
    vgrow   = row_features.get("mem_vgrow", 0)
    cpu     = row_features.get("cpu_usage", 0)
    disk    = row_features.get("disk_write", 0)
    delta   = abs(row_features.get("phys_delta", 0))
    p_net   = row_features.get("p_net", 0)
    src_b   = row_features.get("src_bytes", 0)

    # Rule 1: Extreme page fault activity (exploit / fork bomb / injection)
    if pf_maj > 300:
        return True, f"FORCED: page_faults_maj={pf_maj:.0f} >> 300 (exploit/injection)"

    # Rule 2: Massive memory growth + high CPU simultaneously
    if vgrow > 3000 and cpu > 85:
        return True, f"FORCED: mem_vgrow={vgrow:.0f} + cpu={cpu:.0f} (malware/mining)"

    # Rule 3: High disk + high CPU + network active (ransomware signature)
    if disk > 1500 and cpu > 80 and p_net > 0.70:
        return True, f"FORCED: disk+cpu+net active (ransomware signature)"

    # Rule 4: Extreme physical delta (impossible sensor behaviour)
    if delta > 15.0:
        return True, f"FORCED: phys_delta={delta:.1f}°C (sensor spoofing / tampering)"

    # Rule 5: Extreme data volume (exfiltration)
    if src_b > 500_000:
        return True, f"FORCED: src_bytes={src_b:.0f} >> 500k (exfiltration)"

    return False, ""


def generate_pseudo_label(fusion_score, p_net=None, p_iot=None, p_log=None, high=0.92, low=0.08):
    """Only label when fusion confidence is high AND 2+ layers agree."""
    if p_net is not None and p_iot is not None and p_log is not None:
        agree_atk = sum([p_net>0.6, p_iot>0.6, p_log>0.6])
        agree_nrm = sum([p_net<0.4, p_iot<0.4, p_log<0.4])
        if fusion_score > high and agree_atk >= 2: return 1
        if fusion_score < low  and agree_nrm >= 2: return 0
        return None
    if fusion_score > high: return 1
    if fusion_score < low:  return 0
    return None


# ════════════════════════════════════════════════════════════════════
# SECTION 8 — LIME EXPLAINABILITY
# ════════════════════════════════════════════════════════════════════

class IDSExplainer:
    FEATURES = ['p_net','p_iot','p_log','L_net','L_iot','L_log','phys_delta']

    def __init__(self, fusion_clf, ev_train):
        if not LIME_AVAILABLE:
            self._ready = False; return
        X = ev_train[self.FEATURES].fillna(0).values
        self.exp = lime.lime_tabular.LimeTabularExplainer(
            X, feature_names=self.FEATURES,
            class_names=['Normal','Attack'], mode='classification',
            discretize_continuous=True)
        self.clf   = fusion_clf
        self._ready= True
        print("  ✅ LIME explainer ready")

    def _pred_fn(self, X):
        df = pd.DataFrame(X, columns=self.FEATURES)
        s  = self.clf.predict_proba_score(df)
        return np.column_stack([1-s, s])

    def explain_text(self, ev_row, incident_id=''):
        if not self._ready:
            return '  [LIME not available — install lime]'
        if isinstance(ev_row, pd.DataFrame):
            x = ev_row[self.FEATURES].fillna(0).values[0]
        else:
            x = np.array([ev_row.get(f,0) for f in self.FEATURES])
        result  = self.exp.explain_instance(x, self._pred_fn, num_features=5)
        pairs   = result.as_list()
        lines   = [f'\n  📋 LIME Attribution — Incident {incident_id}']
        lines.append('  ' + '─'*54)
        for raw_feat, w in sorted(pairs, key=lambda x: abs(x[1]), reverse=True):
            # Use regex to extract feature name from LIME range strings
            # e.g. '0.08 < p_net <= 0.92' -> 'p_net'
            # e.g. '-2.50 < L_net <= 2.50' -> 'L_net'
            import re as _re
            _m = _re.search(r'[a-zA-Z][a-zA-Z0-9_]*', raw_feat)
            feat_key  = _m.group(0) if _m else raw_feat
            plain     = _explain_engine.FEATURE_NAMES.get(feat_key, feat_key)
            direction = '↑ ATTACK' if w > 0 else '↓ NORMAL'
            magnitude = '●●●' if abs(w)>0.15 else '●● ' if abs(w)>0.05 else '●  '
            lines.append(f'  {magnitude} {plain:<42} {direction}  {w:+.4f}')
        lines.append('  ' + '─'*54)
        return '\n'.join(lines)

    def get_pairs(self, ev_row):
        if not self._ready:
            return []
        if isinstance(ev_row, pd.DataFrame):
            x = ev_row[self.FEATURES].fillna(0).values[0]
        else:
            x = np.array([ev_row.get(f,0) for f in self.FEATURES])
        result = self.exp.explain_instance(x, self._pred_fn, num_features=5)
        return result.as_list()


# ════════════════════════════════════════════════════════════════════
# SECTION 8b — EXPLAINABILITY ENGINE
# Translates internal CATF-IDS signals into human-readable narratives
# ════════════════════════════════════════════════════════════════════

class ExplainabilityEngine:
    """
    Three-level explainability for CATF-IDS decisions:

    Level 1 — Event:     WHY was this specific event flagged?
                         (modality signals, decision reason, confidence)
    Level 2 — Context:   WHAT is the system state around this event?
                         (FSM energy, trust, threshold position)
    Level 3 — Campaign:  WHAT is the full attack story?
                         (progression, phases, severity)
    """

    # ── Plain-English feature names ───────────────────────────────────
    FEATURE_NAMES = {
        'p_net':              'Network traffic attack probability',
        'p_iot':              'IoT physical sensor attack probability',
        'p_log':              'Linux system log attack probability',
        'L_net':              'Network evidence strength (log-odds)',
        'L_iot':              'IoT evidence strength (log-odds)',
        'L_log':              'Log evidence strength (log-odds)',
        'phys_delta':         'Physical environment deviation (σ from device baseline)',
        'phys_payload_asym':  'Payload asymmetry (↑ = potential exfiltration)',
        'phys_fanout':        'Connection fan-out (↑ = scanning many targets)',
        'phys_periodicity':   'Traffic periodicity (↑ = C2 beaconing pattern)',
        'phys_burst':         'Burst ratio (↑ = flood/DoS pattern)',
        'phys_intensity':     'Packet intensity (↑ = high-rate attack)',
        'phys_lateral':       'Lateral movement indicator',
        'phys_port_entropy':  'Port diversity (↑ = random port scanning)',
        'atk_scan':           'Port/network scan signature',
        'atk_dos':            'Denial-of-service pattern',
        'atk_brute_force':    'Brute-force authentication signature',
        'atk_exfiltration':   'Data exfiltration indicator',
        'atk_injection':      'Injection attack pattern',
        'atk_mitm':           'Man-in-the-middle indicator',
        'atk_ransomware':     'Ransomware activity (disk encryption pattern)',
        'atk_backdoor':       'Backdoor/C2 communication signature',
        'atk_password':       'Password attack signature',
        'net_risk_score':     'Pre-classifier network risk score',
        'disk_write':         'Disk write rate (↑ = ransomware/staging)',
        'cpu_usage':          'CPU utilization (↑ = mining/cryptographic ops)',
        'mem_vgrow':          'Virtual memory growth (↑ = code injection)',
        'page_faults_maj':    'Major page faults (↑ = exploit/injection)',
    }

    # ── Decision reason narratives ────────────────────────────────────
    REASON_NARRATIVES = {
        'FUSION_ALERT':
            'Multiple modalities converge above detection threshold — '
            'strong multi-source attack evidence',
        'FUSION_NORMAL':
            'Fusion score below detection floor — event is confidently normal',
        'CLUSTER_MATCH':
            'Event matches a known attack behavioral cluster in experience memory',
        'ANOMALY':
            'Novel event pattern — does not match any known cluster (potential zero-day)',
        'SUSPICIOUS_SAFE':
            'Insufficient evidence to classify — defaulting to normal (uncertain zone)',
        'FORCED':
            'Hard policy rule triggered — behavior is deterministically malicious',
        'POLICY_OVERRIDE':
            'Policy engine risk threshold exceeded — symbolic rule corroborates ML',
        'OVERRIDE':
            'Single modality showing near-certain attack signal (>0.95)',
        'warmup':
            'Warmup period event — used for memory initialization only',
        'phase2':
            'Phase 2 blindfolded inference — no label available',
        'MEM':
            'Known signature — decision served from experience cache (seen before)',
        'p1_base':
            'Phase 1 baseline reference event',
    }

    # ── FSM state narratives ──────────────────────────────────────────
    FSM_NARRATIVES = {
        'SAFE':
            'System in normal operating state — no sustained threat activity observed',
        'SUSPICIOUS':
            'Low-level anomalies accumulating — elevated monitoring recommended',
        'ELEVATED':
            'Sustained attack activity detected — SOC review strongly recommended',
        'CRITICAL':
            'Active multi-stage campaign in progress — immediate isolation advised',
        'FORCED_ACTION':
            'Emergency hard-rule override — isolate affected systems immediately',
    }

    # ── Attack intent narratives ──────────────────────────────────────
    INTENT_NARRATIVES = {
        'scan':         'Network reconnaissance — probing multiple hosts/ports',
        'dos':          'Denial-of-service — flooding target with high-rate traffic',
        'brute_force':  'Brute-force authentication — repeated credential attempts',
        'exfiltration': 'Data exfiltration — abnormal outbound payload volume',
        'injection':    'Injection attack — malformed protocol/SQL/command input',
        'mitm':         'Man-in-the-middle — traffic interception/ARP anomaly',
        'ransomware':   'Ransomware — high-rate encrypted disk write pattern',
        'backdoor':     'Backdoor/C2 — periodic beaconing to remote controller',
        'password':     'Password attack — credential harvesting/credential stuffing',
    }

    # ── Campaign progression narratives ──────────────────────────────
    CAMPAIGN_NARRATIVES = {
        'recon_to_exploit':
            'Full kill-chain: Network reconnaissance → Initial access → '
            'Persistence established',
        'recon_to_exfil':
            'Exfiltration campaign: Reconnaissance phase completed → '
            'Data staging and exfiltration detected',
        'beacon_to_exfil':
            'C2-driven exfiltration: Command-and-control beaconing active → '
            'Data being exfiltrated to remote server',
        'dos_campaign':
            'Sustained DoS campaign: Repeated high-volume flood events detected',
        'lateral_spread':
            'Lateral movement campaign: Internal scanning → Host compromise → '
            'Spread to adjacent systems',
    }

    def modality_agreement(self, p_net, p_iot, p_log, threshold=0.5):
        """Summarise how much the three modalities agree."""
        votes = [int(p_net > threshold),
                 int(p_iot > threshold),
                 int(p_log > threshold)]
        n_atk = sum(votes)
        if n_atk == 3:
            return "Unanimous (3/3 modalities indicate attack)"
        elif n_atk == 2:
            dissenters = []
            if p_net <= threshold: dissenters.append("network")
            if p_iot <= threshold: dissenters.append("IoT sensor")
            if p_log <= threshold: dissenters.append("system log")
            return (f"Majority (2/3 indicate attack; "
                    f"{dissenters[0]} disagrees — treat with moderate confidence)")
        elif n_atk == 1:
            agreeers = []
            if p_net > threshold: agreeers.append("network")
            if p_iot > threshold: agreeers.append("IoT sensor")
            if p_log > threshold: agreeers.append("system log")
            return (f"Minority (1/3 indicate attack — only {agreeers[0]} "
                    f"flags this event; treat with low confidence)")
        else:
            return "Consensus normal (all 3 modalities indicate normal traffic)"

    def intent_narrative(self, intent_profile):
        """Generate a plain-English sentence about behavioral intent."""
        if not intent_profile:
            return None
        dominant = max(intent_profile, key=intent_profile.get)
        conf     = intent_profile[dominant]
        narr     = self.INTENT_NARRATIVES.get(
            dominant.replace("atk_", ""), dominant)
        others   = [(k.replace("atk_",""),v)
                    for k,v in intent_profile.items()
                    if k != dominant and v > 0.25]
        base = f"{narr} ({conf:.0%} confidence)"
        if others:
            co = ", ".join(f"{k} ({v:.0%})" for k,v in others[:2])
            base += f"; secondary indicators: {co}"
        return base

    def lime_narrative(self, lime_list):
        """
        Convert raw LIME (feature, weight) pairs into readable sentences.
        lime_list: list of (feature_name_string, weight) from LIME
        """
        if not lime_list:
            return []
        lines = []
        for raw_feat, w in sorted(lime_list,
                                   key=lambda x: abs(x[1]), reverse=True)[:5]:
            # LIME may return "feature <= 0.50" style strings — extract name
            # Regex extracts feature name from LIME range strings
            # e.g. '0.08 < p_net <= 0.92' → 'p_net'
            import re as _re3
            _m3 = _re3.search(r'[a-zA-Z][a-zA-Z0-9_]*', raw_feat)
            feat_key = _m3.group(0) if _m3 else raw_feat
            plain    = self.FEATURE_NAMES.get(feat_key, feat_key)
            direction = "pushes toward ATTACK" if w > 0 else "pushes toward NORMAL"
            magnitude = ("strong" if abs(w) > 0.15 else
                         "moderate" if abs(w) > 0.05 else "weak")
            lines.append(f"  {plain}: {magnitude} {direction} ({w:+.4f})")
        return lines

    def risk_breakdown(self, p_net, p_iot, p_log, phys_delta,
                       cluster_risk, policy_risk):
        """Show the weighted contribution of each signal to unified risk R."""
        N = float(p_net)
        I = float(p_iot)
        L = float(p_log)
        C = float(cluster_risk)
        P = float(policy_risk)
        T = min(abs(float(phys_delta)) / 10.0, 1.0)

        W = {'Network':0.35, 'IoT sensor':0.20, 'Linux log':0.15,
             'Cluster context':0.10, 'Physical':0.05, 'Policy':0.05}
        V = {'Network':N, 'IoT sensor':I, 'Linux log':L,
             'Cluster context':C, 'Physical':T, 'Policy':P}

        R = sum(W[k]*V[k] for k in W)
        lines = [f"  Unified risk R = {R:.3f}"]
        for k in sorted(W, key=lambda x: W[x]*V[x], reverse=True):
            contrib = W[k] * V[k]
            bar     = "█" * int(contrib / R * 10) if R > 0 else ""
            lines.append(
                f"  {k:<20} {V[k]:.3f} × {W[k]:.2f} = {contrib:.3f}  {bar}")
        return lines

    def full_event_explanation(self,
                                event_idx,
                                decision,
                                reason,
                                fusion_score,
                                p_net, p_iot, p_log,
                                phys_delta,
                                s_level_name,
                                energy,
                                tau_high, tau_low,
                                trust_scores,
                                intent_profile,
                                cluster_risk=0.0,
                                policy_risk=0.0,
                                lime_pairs=None):
        """
        Full structured explanation for one event.
        Returns a list of lines ready to print.
        """
        SEP  = "  " + "─" * 60
        lines = []
        lines.append("\n  ╔" + "═"*62 + "╗")

        verdict = "⚠️  ATTACK DETECTED" if decision == 1 else "✅  NORMAL TRAFFIC"
        conf    = abs(fusion_score - 0.5) * 2
        lines.append(f"  ║  Event #{event_idx:<6}  │  {verdict:<28}  conf={conf:.0%}  ║")
        lines.append("  ╚" + "═"*62 + "╝")

        # ── 1. Decision reason ────────────────────────────────────────
        # Reason format: "[FSM_STATE] DECISION_TYPE: details | POLICY:..."
        # Extract the core decision type for lookup
        if ']' in reason:
            # Has FSM prefix: "[ELEVATED] FUSION_ALERT: Attack ..."
            _after = reason.split(']', 1)[1].strip()
            _reason_clean = _after.split(':')[0].split(' ')[0].strip()
        elif ':' in reason and not reason.startswith(('FUSION','CLUSTER','ANOMALY','FORCED','OVER','MEM','POLICY','SUSPICIOUS','warmup','phase')):
            _reason_clean = reason.split(':')[0].strip().split(' ')[0]
        else:
            _reason_clean = reason.split('[')[0].strip().split(' ')[0]

        # For cached/warmup reasons, derive from score
        if _reason_clean in ('warmup', 'phase2', 'p1_base', '') or (
                'MEM' in reason and _reason_clean not in self.REASON_NARRATIVES):
            if fusion_score > tau_high + 0.25:
                _reason_clean = 'FUSION_ALERT'
                narr = ('Multi-source evidence strongly exceeds detection threshold '
                        '— high-confidence attack (served from experience cache)')
            elif fusion_score > tau_high:
                _reason_clean = 'FUSION_ALERT'
                narr = 'Evidence exceeds detection threshold — attack confirmed (matched from experience cache)'
            else:
                _reason_clean = 'MEM'
                narr = self.REASON_NARRATIVES.get('MEM', 'Known pattern — decision from experience cache')
        elif _reason_clean == 'FORCED':
            # Extract the specific forced reason for richer display
            _detail = reason.split('FORCED:', 1)[-1].split('|')[0].strip() if 'FORCED:' in reason else ''
            narr = f"Hard policy rule triggered — {_detail}" if _detail else self.REASON_NARRATIVES.get('FORCED', reason)
        else:
            narr = self.REASON_NARRATIVES.get(_reason_clean,
                   f"Decision: {_reason_clean}")

        # Also extract any policy flags from compound reason
        _policy_flag = ''
        if '| POLICY:' in reason:
            _policy_flag = reason.split('| POLICY:', 1)[1].split('|')[0].strip()

        lines.append(f"\n  WHY: {narr}")
        if _policy_flag:
            lines.append(f"  Policy flag: {_policy_flag}")

        # ── 2. Fusion score position ──────────────────────────────────
        pos = ("well above threshold" if fusion_score > tau_high + 0.20 else
               "above threshold"      if fusion_score > tau_high          else
               "within uncertain zone" if fusion_score > tau_low           else
               "below confident-normal floor")
        lines.append(f"  Score: {fusion_score:.4f}  "
                     f"[tau_low={tau_low:.3f}  tau_high={tau_high:.3f}]  "
                     f"→ {pos}")

        # ── 3. Modality agreement ─────────────────────────────────────
        lines.append("\n  MODALITY AGREEMENT:")
        lines.append(f"  {self.modality_agreement(p_net, p_iot, p_log)}")
        lines.append(f"  {'Network':14} p={p_net:.3f}  trust={trust_scores.get('net',1.0):.2f}")
        lines.append(f"  {'IoT Sensor':14} p={p_iot:.3f}  trust={trust_scores.get('iot',1.0):.2f}")
        lines.append(f"  {'System Log':14} p={p_log:.3f}  trust={trust_scores.get('log',1.0):.2f}")
        lines.append(f"  {'Physical δ':14} {phys_delta:+.2f}σ  "
                     f"{'⚠ anomalous' if abs(phys_delta) > 3 else 'within normal range'}")

        # ── 4. Risk breakdown ─────────────────────────────────────────
        lines.append("\n  RISK BREAKDOWN:")
        lines.extend(self.risk_breakdown(
            p_net, p_iot, p_log, phys_delta, cluster_risk, policy_risk))

        # ── 5. Behavioral intent ──────────────────────────────────────
        intent_narr = self.intent_narrative(intent_profile)
        if intent_narr:
            lines.append("\n  BEHAVIORAL INTENT:")
            lines.append(f"  {intent_narr}")

        # ── 6. System context ─────────────────────────────────────────
        fsm_narr = self.FSM_NARRATIVES.get(s_level_name, s_level_name)
        lines.append(f"\n  SYSTEM STATE: {s_level_name}  (energy={energy:.3f})")
        lines.append(f"  {fsm_narr}")

        # ── 7. LIME explanations ──────────────────────────────────────
        if lime_pairs:
            lines.append("\n  FEATURE ATTRIBUTION (LIME):")
            lines.extend(self.lime_narrative(lime_pairs))

        lines.append("")
        return lines


# Singleton instance created during pipeline setup
_explain_engine = ExplainabilityEngine()


# ════════════════════════════════════════════════════════════════════
# SECTION 9 — ONLINE LEARNING
# ════════════════════════════════════════════════════════════════════

class OnlineLearner:
    FEATURES         = ['p_net','p_iot','p_log','L_net','L_iot','L_log','phys_delta']
    MIN_TO_RETRAIN   = 100
    RETRAIN_EVERY_N  = 200

    def __init__(self, fusion_clf, memory, mode='incremental', auto=False):
        self.fusion_clf        = fusion_clf
        self.memory            = memory
        self.mode              = mode
        self._lock             = threading.Lock()
        self._log              = []
        self._last_idx         = 0
        self._count            = 0
        self._sgd              = None
        self._sgd_scaler       = None
        self._sgd_init         = False
        os.makedirs('models',  exist_ok=True)
        os.makedirs('outputs', exist_ok=True)
        if auto: self._start_bg()

    def _sigmoid(self, x): return 1/(1+np.exp(-x))

    def _build_X(self):
        """Pair evidence vectors with pseudo-labels into a DataFrame."""
        paired_ev, paired_y = [], []
        ev_ptr = 0
        for pl in self.memory.pseudo_labels:
            if ev_ptr >= len(self.memory.evidence_log): break
            if pl is not None:
                paired_ev.append(self.memory.evidence_log[ev_ptr])
                paired_y.append(pl)
            ev_ptr += 1
        if not paired_ev:
            return None, None
        X = np.array(paired_ev)
        return pd.DataFrame({
            'L_net': X[:,0], 'L_iot': X[:,1], 'L_log': X[:,2],
            'phys_delta': X[:,3],
            'p_net': self._sigmoid(X[:,0]),
            'p_iot': self._sigmoid(X[:,1]),
            'p_log': self._sigmoid(X[:,2]),
        })[self.FEATURES], np.array(paired_y)

    def retrain(self, force=False):
        n_pl = len([x for x in self.memory.pseudo_labels if x is not None])
        if not force and n_pl < self.MIN_TO_RETRAIN:
            return {"retrained": False,
                    "reason": f"Only {n_pl}/{self.MIN_TO_RETRAIN} pseudo-labels"}

        X_full, y = self._build_X()
        if X_full is None:
            return {"retrained": False, "reason": "No usable pairs"}
        if len(set(y)) < 2:
            return {"retrained": False, "reason": "Only one class in pseudo-labels — skipping"}

        if self.mode == 'incremental':
            result = self._incremental(X_full, y)
        else:
            result = self._batch(X_full, y)

        self._count += 1
        entry = {"retrain": self._count, "labels_used": len(y),
                 "attack_ratio": round(float(np.mean(y)),3),
                 "mode": self.mode, **result}
        self._log.append(entry)
        self._last_idx = len(self.memory.pseudo_labels)
        print(f"\n[OnlineLearner] Retrain #{self._count} | "
              f"labels={len(y)} | F1 against its own pseudo-labels="
              f"{result.get('f1_on_pseudo','N/A')} (self-agreement, not accuracy)")
        return {"retrained": True, **entry}

    def _batch(self, X, y):
        if len(set(y)) < 2:
            return {"saved": False, "reason": "One class only in batch"}
        if len(X) >= 40:
            sp = int(len(X)*0.8)
            X_tr,X_ho,y_tr,y_ho = X.iloc[:sp],X.iloc[sp:],y[:sp],y[sp:]
        else:
            X_tr,X_ho,y_tr,y_ho = X,X,y,y
        clf    = LogisticRegression(C=1.0, max_iter=1000, random_state=42, class_weight='balanced')
        scaler = StandardScaler()
        if len(set(y_tr)) < 2:
            return {"saved": False, "reason": "One class only in training split"}
        clf.fit(scaler.fit_transform(X_tr), y_tr)
        f1_ho = f1_score(y_ho, clf.predict(scaler.transform(X_ho)), zero_division=0)
        f1_tr = f1_score(y_tr, clf.predict(scaler.transform(X_tr)), zero_division=0)
        if f1_ho < 0.40 and self._count > 0:
            return {"saved":False,"f1_holdout":round(f1_ho,4),"reason":f"Holdout F1={f1_ho:.3f} too low"}
        with self._lock:
            self.fusion_clf.model  = clf
            self.fusion_clf.scaler = scaler
            self.fusion_clf.save()
        return {"saved":True,"f1_train":round(f1_tr,4),"f1_holdout":round(f1_ho,4),"note":"holdout validated"}

    def _incremental(self, X, y):
        X_new = X.iloc[self._last_idx:]
        y_new = y[self._last_idx:]
        if len(X_new) == 0:
            return {"saved": False, "reason": "No new samples"}
        if not self._sgd_init:
            self._sgd_scaler = StandardScaler()
            Xs = self._sgd_scaler.fit_transform(X)
            self._sgd = SGDClassifier(loss='log_loss', max_iter=1,
                                       warm_start=True, random_state=42)
            self._sgd.partial_fit(Xs, y, classes=[0,1])
            self._sgd_init = True
        else:
            Xs_new = self._sgd_scaler.transform(X_new)
            self._sgd.partial_fit(Xs_new, y_new)
        Xs_all = self._sgd_scaler.transform(X)
        f1 = f1_score(y, self._sgd.predict(Xs_all), zero_division=0)
        with self._lock:
            self.fusion_clf.model  = self._sgd
            self.fusion_clf.scaler = self._sgd_scaler
            self.fusion_clf.save()
        return {"saved": True, "new_samples": len(X_new), "f1_on_pseudo": round(f1,4)}

    def _start_bg(self, interval=30):
        def loop():
            while True:
                time.sleep(interval)
                new = len(self.memory.pseudo_labels) - self._last_idx
                if new >= self.RETRAIN_EVERY_N:
                    print(f"\n[OnlineLearner] Auto-retrain triggered ({new} new labels)")
                    self.retrain()
        threading.Thread(target=loop, daemon=True).start()
        print(f"[OnlineLearner] Background loop active (every {self.RETRAIN_EVERY_N} labels)")

    def status(self):
        n_pl = len([x for x in self.memory.pseudo_labels if x is not None])
        return {"mode": self.mode, "retrains": self._count,
                "pseudo_labels": n_pl, "new_since_last": n_pl - self._last_idx,
                "last_log": self._log[-1] if self._log else None}


# ════════════════════════════════════════════════════════════════════
# SECTION 10 — CONTEXT / INTENT SYSTEM
# ════════════════════════════════════════════════════════════════════

INTENT_RULES = [
    {"kw": ["cold","heater","hvac","maintenance","planned","expected spike"],
     "ctx": {"delta_scale":0.5, "net_scale":1.0, "disk_scale":1.0},
     "desc": "Physical delta dampened — expected environmental anomaly"},
    {"kw": ["attack","threat","red alert","high risk","breach","intrusion"],
     "ctx": {"delta_scale":1.5, "net_scale":1.2, "disk_scale":1.2},
     "desc": "All signals amplified — high threat environment"},
    {"kw": ["backup","batch job","transfer","scheduled","heavy traffic"],
     "ctx": {"delta_scale":1.0, "net_scale":0.6, "disk_scale":0.4},
     "desc": "Network and disk dampened — expected high I/O"},
    {"kw": ["reset","default","baseline","clear","green"],
     "ctx": {},   # empty = reset
     "desc": "Default — no active context policy"},
]

def default_context():
    return {"delta_scale":1.0, "delta_shift":0.0,
            "net_scale":1.0, "disk_scale":1.0,
            "description":"default"}

def apply_context(phys_delta, net_dur, disk_w, ctx):
    return ((phys_delta * ctx["delta_scale"]) + ctx["delta_shift"],
             net_dur    * ctx["net_scale"],
             disk_w     * ctx["disk_scale"])

def parse_intent(policy_text, current_ctx):
    pl = policy_text.lower()
    for rule in INTENT_RULES:
        if any(k in pl for k in rule["kw"]):
            new_ctx = default_context() if not rule["ctx"] else {**default_context(), **rule["ctx"]}
            new_ctx["description"] = rule["desc"]
            return new_ctx, rule["desc"]
    return current_ctx, "No matching keywords — context unchanged"


# ════════════════════════════════════════════════════════════════════
# SECTION 11 — EVALUATION
# ════════════════════════════════════════════════════════════════════

def evaluate(y_true, y_pred, y_scores, save_path='plots/', y_type=None):
    os.makedirs(save_path, exist_ok=True)
    print("\n" + "="*60)
    print("  EVALUATION REPORT")
    print("="*60)
    print(classification_report(y_true, y_pred, target_names=['Normal','Attack']))

    f1  = f1_score(y_true, y_pred, zero_division=0)
    pre = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_scores)
    fpr = (sum((np.array(y_pred)==1)&(np.array(y_true)==0))
           / max(sum(np.array(y_true)==0), 1))

    print(f"  F1-Score  : {f1:.4f}")
    print(f"  Precision : {pre:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  ROC-AUC   : {auc:.4f}")
    print(f"  FPR       : {fpr:.4f}")

    if y_type is not None:
        t  = pd.Series(np.asarray(y_type)).astype(str).str.strip().str.lower().values
        yp = np.asarray(y_pred)
        print("\n  Per-class detection rate:")
        print(f"    {'class':<14}{'n':>8}{'flagged':>9}{'rate':>9}")
        print("    " + "-" * 40)
        worst, worst_r = None, 2.0
        for cls in sorted(set(t)):
            m = (t == cls)
            n = int(m.sum())
            if n == 0:
                continue
            k = int((yp[m] == 1).sum())
            if cls in ('normal', 'benign'):
                print(f"    {cls:<14}{n:>8}{k:>9}{k/n:>9.4f}   <- false alarms")
            else:
                print(f"    {cls:<14}{n:>8}{k:>9}{k/n:>9.4f}")
                if k / n < worst_r:
                    worst, worst_r = cls, k / n
        if worst is not None:
            print(f"\n    hardest class: {worst} at recall {worst_r:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12,4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=['Normal','Attack'],
        ax=axes[0], colorbar=False)
    axes[0].set_title('Confusion Matrix')

    fp_c, tp_c, _ = roc_curve(y_true, y_scores)
    axes[1].plot(fp_c, tp_c, label=f'AUC={auc:.3f}', color='#00e5ff', lw=2)
    axes[1].plot([0,1],[0,1],'--', color='gray')
    axes[1].set(xlabel='FPR', ylabel='TPR', title='ROC Curve')
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(f'{save_path}evaluation.png', dpi=150); plt.close()
    print(f"\n  ✅ Plots → {save_path}evaluation.png")
    return {'f1':f1,'precision':pre,'recall':rec,'auc':auc,'fpr':fpr}


def plot_score_distribution(y_true, y_scores, tau_high, tau_low,
                             save_path='outputs/score_distribution.png'):
    """
    Normal vs attack fusion-score histogram with tau_high/tau_low marked.
    Static reference plot for the report/defense and for sanity-checking
    that the adaptive threshold sits in a sensible place relative to the
    real score distribution.
    """
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    y_true    = np.asarray(y_true)
    y_scores  = np.asarray(y_scores)
    normal_scores = y_scores[y_true == 0]
    attack_scores = y_scores[y_true == 1]

    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(0, 1, 41)
    ax.hist(normal_scores, bins=bins, alpha=0.65, color='#2E7D32', label='Normal (true label)')
    ax.hist(attack_scores, bins=bins, alpha=0.65, color='#C62828', label='Attack (true label)')
    ax.axvline(tau_high, color='#1F4D78', linestyle='--', linewidth=2,
               label=f'tau_high = {tau_high:.3f}')
    ax.axvline(tau_low, color='#2E75B6', linestyle='--', linewidth=2,
               label=f'tau_low = {tau_low:.3f}')
    ax.set_xlabel('Fusion score')
    ax.set_ylabel('Event count')
    ax.set_title('CATF-IDS Fusion Score Distribution — Normal vs Attack')
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  ✅ Score distribution plot → {save_path}")


# ════════════════════════════════════════════════════════════════════
# SECTION 12 — FULL PIPELINE RUNNER
# ════════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════════
# MULTI-FILE LOADER
# ════════════════════════════════════════════════════════════════════

def load_multi(file_list, name_label, n_per_file=2000):
    """
    Load and concatenate multiple CSV files for one modality.
    Samples n_per_file rows (balanced) from each file, then combines.
    Missing files are skipped with a warning — pipeline continues.
    """
    dfs = []
    for path in file_list:
        if not os.path.exists(path):
            print(f"  ⚠️  Skipping (not found): {os.path.basename(path)}")
            continue
        df = load_and_snip(path, os.path.basename(path), n=n_per_file)
        if df is not None:
            df['_source'] = os.path.basename(path)   # track origin
            dfs.append(df)

    if not dfs:
        print(f"  ❌ No files loaded for {name_label}"); return None

    combined = pd.concat(dfs, ignore_index=True).sort_values('ts').reset_index(drop=True)
    print(f"  ✅ {name_label}: {len(combined)} total rows from {len(dfs)} files")
    return combined


# ════════════════════════════════════════════════════════════════════
# IOT DEVICE-AWARE NORMALIZER
# Each IoT device has different columns — we extract a universal
# physical signal representation from all of them.
# ════════════════════════════════════════════════════════════════════

IOT_DEVICE_MAP = {
    # device keyword → (primary_signal_col, secondary_col_or_None)
    # NOTE: 'fridge_tem' and 'FC1' do not exist in TON_IoT. The lookup missed,
    # current_temp fell through to 0.0, and both devices trained on constants.
    'fridge':    ('fridge_temperature',       'temp_condition'),
    'thermostat':('current_temperature',      'thermostat_status'),
    'weather':   ('temperature',              'humidity'),
    'garage':    ('door_state',               'sphone_signal'),
    'gps':       ('latitude',                 'longitude'),
    'modbus':    ('FC1_Read_Input_Register',  'FC2_Read_Discrete_Value'),
    'motion':    ('motion_status',            'light_status'),
}

# Column-name fallbacks: TON_IoT ships slightly different headers between
# releases, so a missed lookup must not silently become a column of zeros.
IOT_COL_ALIASES = {
    'fridge_temperature':      ['fridge_temperature', 'fridge_tem', 'temperature'],
    'temp_condition':          ['temp_condition', 'temp_con'],
    'current_temperature':     ['current_temperature', 'current_temp', 'temperature'],
    'thermostat_status':       ['thermostat_status', 'status'],
    'temperature':             ['temperature', 'temp'],
    'humidity':                ['humidity', 'hum'],
    'door_state':              ['door_state', 'door_status'],
    'sphone_signal':           ['sphone_signal', 'sphone'],
    'latitude':                ['latitude', 'speed', 'lat'],
    'longitude':               ['longitude', 'lon', 'lng'],
    'FC1_Read_Input_Register': ['FC1_Read_Input_Register', 'FC1'],
    'FC2_Read_Discrete_Value': ['FC2_Read_Discrete_Value', 'FC2'],
    'motion_status':           ['motion_status', 'motion'],
    'light_status':            ['light_status', 'light'],
}


def _iot_resolve(df, wanted):
    """Find `wanted` in df under any known alias, case-insensitively."""
    if wanted is None:
        return None
    lower = {c.lower(): c for c in df.columns}
    for cand in IOT_COL_ALIASES.get(wanted, [wanted]):
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def _iot_to_signal(series):
    """Numeric where possible, ordinal codes where categorical.

    door_state and motion_status are strings ('open'/'closed', 'on'/'off').
    pd.to_numeric(errors='coerce') turned every one of them into NaN, which
    fillna(0) then flattened to a constant column. Encode instead."""
    num = pd.to_numeric(series, errors='coerce')
    if num.notna().sum() >= max(1, int(0.5 * len(series))):
        return num.ffill().fillna(0.0), False
    codes = series.astype(str).str.strip().str.lower().astype('category').cat.codes
    n = max(int(codes.max()), 1)
    return (codes / n).astype(float), True

def normalize_iot_device(df, source_name):
    """
    Given a single IoT device DataFrame, extract:
      - current_temp (primary numeric signal)
      - temp_con_enc (categorical state if available)
      - phys_delta, phys_delta_abs, temp_roll_mean, temp_roll_std
    Works for ALL 7 device types by mapping each to its primary signal.
    """
    df = df.copy()
    device_key = next((k for k in IOT_DEVICE_MAP if k in source_name.lower()), None)

    if device_key:
        primary_col, secondary_col = IOT_DEVICE_MAP[device_key]
        pcol = _iot_resolve(df, primary_col)
        scol = _iot_resolve(df, secondary_col)

        if pcol is not None:
            df['current_temp'], was_cat = _iot_to_signal(df[pcol])
            if was_cat:
                print(f"  \u2139\ufe0f  {source_name}: '{pcol}' categorical "
                      f"({df[pcol].nunique()} states) - encoded")
        else:
            df['current_temp'] = 0.0
            print(f"  \u26a0\ufe0f  {source_name}: primary signal '{primary_col}' NOT FOUND "
                  f"- this device contributes a constant column")

        if scol is not None:
            col_num, _ = _iot_to_signal(df[scol])
            rng = float(col_num.max() - col_num.min())
            df['temp_con_enc'] = (col_num - col_num.min()) / (rng if rng > 0 else 1.0)
        else:
            df['temp_con_enc'] = 0.0

        # Absence indicator: 0.0 is a legitimate reading for several of these
        # devices, so 'absent' and 'genuinely zero' were indistinguishable.
        df['iot_present'] = 0.0 if pcol is None else 1.0

        nuniq = int(pd.Series(df['current_temp']).nunique())
        if nuniq <= 1:
            print(f"  \u26a0\ufe0f  {source_name}: signal is CONSTANT after parsing "
                  f"({nuniq} distinct value) - contributes nothing")
    else:
        # Unknown device — try to find any numeric column
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        num_cols = [c for c in num_cols if c not in ['ts','label','_source']]
        df['current_temp']  = pd.to_numeric(df[num_cols[0]], errors='coerce').fillna(0) if num_cols else 0.0
        df['temp_con_enc']  = pd.to_numeric(df[num_cols[1]], errors='coerce').fillna(0) if len(num_cols)>1 else 0.0
        df['iot_present']   = 1.0 if num_cols else 0.0
        if not num_cols:
            print(f"  \u26a0\ufe0f  {source_name}: unmapped device with no numeric column "
                  f"- constant zeros")

    df['temp_smooth']    = df['current_temp'].rolling(window=5, min_periods=1).mean()
    df['phys_delta']     = df['temp_smooth'].diff().fillna(0)
    df['phys_delta_abs'] = df['phys_delta'].abs()
    df['temp_roll_mean'] = df['temp_smooth'].copy()
    df['temp_roll_std']  = df['current_temp'].rolling(5, min_periods=1).std().fillna(0)

    if 'iot_present' not in df.columns:
        df['iot_present'] = 1.0
    return df[['current_temp','temp_con_enc','phys_delta','phys_delta_abs',
               'temp_roll_mean','temp_roll_std','iot_present','label','ts']].fillna(0)


_IOT_DEVICE_CODES = {k: i + 1 for i, k in enumerate(sorted(IOT_DEVICE_MAP))}
_IOT_DEVICE_NAMES = {c: k for k, c in _IOT_DEVICE_CODES.items()}


def _iot_device_code(source):
    s = str(source).lower()
    return next((c for k, c in _IOT_DEVICE_CODES.items() if k in s), 0)


def preprocess_iot_multi(df):
    """
    Preprocess combined IoT DataFrame containing rows from multiple devices.
    Uses _source column to apply device-specific normalization per device,
    then stacks them into a unified representation.
    """
    if '_source' not in df.columns:
        # Single file — use existing logic
        return preprocess_iot(df)

    parts = []
    for source, group in df.groupby('_source'):
        normed = normalize_iot_device(group.reset_index(drop=True), source)
        normed['_source'] = source
        normed['iot_device'] = _iot_device_code(source)   # read only by stage 5
        parts.append(normed)

    combined = pd.concat(parts).sort_values('ts').reset_index(drop=True)
    print(f"  \u2705 IoT multi-device: {len(combined)} rows from {df['_source'].nunique()} devices")
    if 'iot_present' in combined.columns:
        dead = int((combined['iot_present'] == 0).sum())
        if dead:
            print(f"  \u26a0\ufe0f  IoT: {dead:,} rows ({dead/len(combined):.2%}) had no usable "
                  f"sensor signal")
        else:
            print("  \u2705 IoT: every device produced a usable signal")
    return combined.drop(columns=['_source'], errors='ignore')


def preprocess_linux_multi(df):
    """
    Handles all 6 Linux log types: disk, memory, process.
    Each has different column names — extracts the 4 universal signals.
    """
    df = df.copy()
    src = df['_source'].iloc[0] if '_source' in df.columns else 'unknown'

    # Print raw columns immediately so any mismatch is visible
    print(f"  Linux [{src}] raw columns: {df.columns.tolist()}")

    def find_col(df, candidates):
        cl = {c.lower(): c for c in df.columns}
        for cand in candidates:
            if cand.lower() in cl:
                return cl[cand.lower()]
        # fuzzy fallback: substring match
        for cand in candidates:
            for col in df.columns:
                if cand.lower() in col.lower():
                    return col
        return None

    disk_col   = find_col(df, ['WRDSK','WR_SEC','DISK_WRITE','disk_write','writes','blk_wrtn'])
    cpu_col    = find_col(df, ['CPU','%CPU','cpu_usage','cpu','usr','%usr'])
    mem_col    = find_col(df, ['MEM','%MEM','VSZ','RSS','VSIZE','RSIZE','memory','kbmemused'])
    vgrow_col  = find_col(df, ['VGROW','vgrow','vmgrow'])
    rgrow_col  = find_col(df, ['RGROW','rgrow','rmgrow'])
    minflt_col = find_col(df, ['MINFLT','minflt','minor_faults'])
    majflt_col = find_col(df, ['MAJFLT','majflt','major_faults'])
    trun_col   = find_col(df, ['TRUN','trun','threads_run'])

    df['disk_write']      = pd.to_numeric(df[disk_col],   errors='coerce').fillna(0) if disk_col   else 0.0
    df['cpu_usage']       = pd.to_numeric(df[cpu_col],    errors='coerce').fillna(0) if cpu_col    else 0.0
    df['mem_usage']       = pd.to_numeric(df[mem_col],    errors='coerce').fillna(0) if mem_col    else 0.0
    df['mem_vgrow']       = pd.to_numeric(df[vgrow_col],  errors='coerce').fillna(0) if vgrow_col  else 0.0
    df['mem_rgrow']       = pd.to_numeric(df[rgrow_col],  errors='coerce').fillna(0) if rgrow_col  else 0.0
    df['page_faults_min'] = pd.to_numeric(df[minflt_col], errors='coerce').fillna(0) if minflt_col else 0.0
    df['page_faults_maj'] = pd.to_numeric(df[majflt_col], errors='coerce').fillna(0) if majflt_col else 0.0
    df['threads_run']     = pd.to_numeric(df[trun_col],   errors='coerce').fillna(0) if trun_col   else 0.0
    df['cpu_spike']       = (df['cpu_usage'] - df['cpu_usage'].rolling(5,min_periods=1).median()).clip(lower=0)
    print(f"  disk<-'{disk_col}' cpu<-'{cpu_col}' mem<-'{mem_col}' vgrow<-'{vgrow_col}' minflt<-'{minflt_col}'")


    for c in ['disk_write', 'cpu_usage', 'mem_usage']:
        p99 = df[c].quantile(0.99)
        if p99 > 0:
            df[c] = df[c].clip(upper=p99)
    df['disk_write_delta'] = df['disk_write'].diff().fillna(0).abs()

    print(f"  Linux [{src}]: disk={df['disk_write'].mean():.1f} "
          f"cpu={df['cpu_usage'].mean():.1f} mem={df['mem_usage'].mean():.1f}")

    return pd.DataFrame({
        'disk_write':       df['disk_write'].values,
        'disk_write_delta': df['disk_write_delta'].values,
        'cpu_usage':        df['cpu_usage'].values,
        'cpu_spike':        df['cpu_spike'].values,
        'mem_usage':        df['mem_usage'].values,
        'mem_vgrow':        df['mem_vgrow'].values,
        'mem_rgrow':        df['mem_rgrow'].values,
        'page_faults_min':  df['page_faults_min'].values,
        'page_faults_maj':  df['page_faults_maj'].values,
        'threads_run':      df['threads_run'].values,
        'label':            df['label'].values,
        'ts':               df['ts'].values,
    }).fillna(0)


# ════════════════════════════════════════════════════════════════════
# MULTI-FILE PIPELINE — runs the full system on all dataset files
# ════════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════════
# STATEFUL K-FOLD CROSS-VALIDATION
# ════════════════════════════════════════════════════════════════════

class FoldResult:
    """Stores per-fold metrics and all learned component states."""
    def __init__(self, fold_id):
        self.fold_id     = fold_id
        self.metrics     = {}          # F1, ROC-AUC, FPR, Precision, Recall
        self.tau         = {}          # tau_high, tau_low, alpha, delta, fbeta
        self.trust       = {}          # net, iot, log, win trust scores
        self.energy_dist = {}          # SAFE/SUSP/ELEV/CRIT counts
        self.mem_entries = []          # (fingerprint, tag, risk) tuples
        self.ol_weights  = None        # online learner retrained model coefs
        self.fusion_clf  = None        # saved FusionClassifier for ensemble


def _fingerprint(ts, label, risk):
    """Deterministic fingerprint for a behavioral memory entry."""
    import hashlib
    key = f"{round(float(ts),2)}_{int(label)}_{round(float(risk),2)}"
    return hashlib.md5(key.encode()).hexdigest()



# ════════════════════════════════════════════════════════════════════
# AGGREGATION STRATEGY GRID SEARCH  — EXPANDED METHOD SET
# ════════════════════════════════════════════════════════════════════

# t-distribution critical value for 95% CI with 4 degrees of freedom (5 folds)
T_CRIT_95 = 2.776   # t(4, 0.025)

def extract_cv_components(fold_results, l1='auc', l2='mean',
                          l3='stability', l4='current'):
    """
    Extract CV components using the specified strategy per level.

    LEVEL 1 — Ensemble weighting:
      equal       : 1/5 each (baseline)
      auc         : AUC-weighted
      auc_sq      : (AUC-mean)^2 weighted — amplifies differences
      fpr_inv     : 1/FPR weighted — rewards low-FPR folds
      f1          : F1-weighted
      fbeta       : F-beta weighted
      prec        : Precision-weighted
      rec         : Recall-weighted
      ci_lower    : Weight by lower 95% CI of AUC (conservative)
      stability   : Weight by AUC stability score (1 - std/mean)

    LEVEL 2 — Threshold selection:
      single_run  : AutoTune on 70% train (no CV)
      mean        : Naive mean across folds
      fbeta       : F-beta weighted
      min_fpr     : Tau from lowest-FPR fold
      median      : Median tau
      ci_lower    : mean - t*std/sqrt(5)  → more sensitive boundary
      ci_upper    : mean + t*std/sqrt(5)  → more conservative boundary
      best_auc    : Tau from highest-AUC fold
      harmonic    : Harmonic mean of tau values
      geom        : Geometric mean

    LEVEL 3 — Trust aggregation:
      default     : 0.700 for all (no CV input)
      mean        : Naive mean
      stability   : mean × (1 - std/mean)  (current)
      minimum     : Min across folds (most conservative)
      harmonic    : Harmonic mean
      ci_lower    : mean - t*std/sqrt(5)
      ci_upper    : mean + t*std/sqrt(5)
      median      : Median
      geom        : Geometric mean
      f1_weighted : Weight trust by fold F1

    LEVEL 4 — FSM thresholds:
      current         : CRITICAL≥0.62  ELEVATED≥0.42  SUSPICIOUS≥0.28
      conservative    : CRITICAL≥0.65  (fewer CRITICAL)
      sensitive       : ELEVATED≥0.40  SUSPICIOUS≥0.26
      very_conservative: CRITICAL≥0.68  ELEVATED≥0.45
      very_sensitive  : CRITICAL≥0.58  ELEVATED≥0.38  SUSPICIOUS≥0.24
      balanced        : CRITICAL≥0.60  ELEVATED≥0.42  SUSPICIOUS≥0.26
    """
    n = len(fold_results)
    aucs   = np.array([fr.metrics['roc_auc'] for fr in fold_results])
    fprs   = np.array([fr.metrics['fpr']     for fr in fold_results])
    f1s    = np.array([fr.metrics['f1']       for fr in fold_results])
    precs  = np.array([fr.metrics['precision']for fr in fold_results])
    recs   = np.array([fr.metrics['recall']   for fr in fold_results])
    fbetas = np.array([fr.tau.get('fbeta',0.85) for fr in fold_results])
    taus_h = np.array([fr.tau['tau_high']     for fr in fold_results])
    taus_l = np.array([fr.tau['tau_low']      for fr in fold_results])
    se     = lambda v: v.std() / np.sqrt(n)   # standard error

    # ── LEVEL 1 ─────────────────────────────────────────────────
    def _l1_weights(method):
        if   method == 'equal':    return np.ones(n)/n
        elif method == 'auc':      return aucs/aucs.sum()
        elif method == 'auc_sq':
            d = aucs-aucs.mean(); w=(d**2+1e-9); return w/w.sum()
        elif method == 'fpr_inv':
            w=1/(fprs+1e-6); return w/w.sum()
        elif method == 'f1':       return f1s/f1s.sum()
        elif method == 'fbeta':    return fbetas/fbetas.sum()
        elif method == 'prec':     return precs/precs.sum()
        elif method == 'rec':      return recs/recs.sum()
        elif method == 'ci_lower':
            ci_lo=aucs-T_CRIT_95*se(aucs); ci_lo=np.clip(ci_lo,1e-6,None)
            return ci_lo/ci_lo.sum()
        elif method == 'stability':
            stab=1-(aucs.std()/(aucs.mean()+1e-9))*np.ones(n)
            stab=np.clip(stab,1e-6,None); return stab/stab.sum()
        return np.ones(n)/n

    fold_clfs = [fr.fusion_clf for fr in fold_results if fr.fusion_clf is not None]
    ensemble  = EnsembleFusionClassifier(fold_clfs, _l1_weights(l1))

    # ── LEVEL 2 ─────────────────────────────────────────────────
    def _l2_tau(method):
        if   method == 'single_run': return None, None
        elif method == 'mean':       return float(taus_h.mean()), float(taus_l.mean())
        elif method == 'fbeta':
            return float(np.dot(fbetas,taus_h)/fbetas.sum()),                    float(np.dot(fbetas,taus_l)/fbetas.sum())
        elif method == 'min_fpr':
            i=int(np.argmin(fprs))
            return float(taus_h[i]), float(taus_l[i])
        elif method == 'median':
            return float(np.median(taus_h)), float(np.median(taus_l))
        elif method == 'ci_lower':
            return float(taus_h.mean()-T_CRIT_95*se(taus_h)),                    float(taus_l.mean()-T_CRIT_95*se(taus_l))
        elif method == 'ci_upper':
            return float(taus_h.mean()+T_CRIT_95*se(taus_h)),                    float(taus_l.mean()+T_CRIT_95*se(taus_l))
        elif method == 'best_auc':
            i=int(np.argmax(aucs))
            return float(taus_h[i]), float(taus_l[i])
        elif method == 'harmonic':
            h_h=n/np.sum(1/(taus_h+1e-9)); h_l=n/np.sum(1/(taus_l+1e-9))
            return float(h_h), float(h_l)
        elif method == 'geom':
            return float(np.exp(np.mean(np.log(taus_h+1e-9)))),                    float(np.exp(np.mean(np.log(taus_l+1e-9))))
        return float(taus_h.mean()), float(taus_l.mean())

    tau_h, tau_l = _l2_tau(l2)

    # ── LEVEL 3 ─────────────────────────────────────────────────
    def _l3_trust(method):
        trust = {}
        for mod in ['net','iot','log','win']:
            vals = np.array([fr.trust.get(mod,0.7) for fr in fold_results])
            m    = float(vals.mean()); s = float(vals.std())
            if   method == 'default':     v = 0.70
            elif method == 'mean':        v = m
            elif method == 'stability':   v = m*(1-s/(m+1e-9))
            elif method == 'minimum':     v = float(vals.min())
            elif method == 'harmonic':    v = float(n/np.sum(1/(vals+1e-9)))
            elif method == 'ci_lower':    v = m - T_CRIT_95*s/np.sqrt(n)
            elif method == 'ci_upper':    v = m + T_CRIT_95*s/np.sqrt(n)
            elif method == 'median':      v = float(np.median(vals))
            elif method == 'geom':        v = float(np.exp(np.mean(np.log(vals+1e-9))))
            elif method == 'f1_weighted': v = float(np.dot(f1s,vals)/f1s.sum())
            else:                         v = m
            trust[mod] = round(float(np.clip(v,0.1,1.0)),3)
        return trust

    trust = _l3_trust(l3)

    # ── LEVEL 4 ─────────────────────────────────────────────────
    fsm_map = {
        'current':          {3:0.62, 2:0.42, 1:0.28},
        'conservative':     {3:0.65, 2:0.42, 1:0.28},
        'sensitive':        {3:0.62, 2:0.40, 1:0.26},
        'very_conservative':{3:0.68, 2:0.45, 1:0.30},
        'very_sensitive':   {3:0.58, 2:0.38, 1:0.24},
        'balanced':         {3:0.60, 2:0.42, 1:0.26},
    }
    fsm_thr = fsm_map.get(l4, fsm_map['current'])

    return {
        'ensemble_fc':    ensemble,
        'weighted_tau_h': tau_h,
        'weighted_tau_l': tau_l,
        'stability_trust':trust,
        'fsm_thresholds': fsm_thr,
        'strategy':       f"L1={l1} L2={l2} L3={l3} L4={l4}",
    }


def _pipeline_with_components(fused_df, full_fused_df, comps):
    """
    Full honest pipeline run for comparison — uses the EXACT same
    inference loop as the main pipeline, with output suppressed.
    Each run takes ~40-60 sec. Results are directly comparable to
    the final evaluation output.
    """
    import io, sys, contextlib
    from sklearn.metrics import (f1_score, precision_score,
                                 recall_score, roc_auc_score)

    # ── Train/test split (same seed as main pipeline) ────────────
    train_df, _ = train_test_split(
        fused_df, test_size=0.3, random_state=42,
        stratify=fused_df['final_label'])
    train_df = train_df.reset_index(drop=True)
    test_df  = full_fused_df.reset_index(drop=True)

    silent = io.StringIO()

    # ── Layer models ─────────────────────────────────────────────
    lm = LayerModels()
    with contextlib.redirect_stdout(silent):
        lm.fit(train_df)

    pn_r, pi_r, pl_r = lm.predict_proba_raw(train_df)
    ev_train = build_evidence_vector(
        calibrate(pn_r), calibrate(pi_r),
        calibrate(pl_r), train_df['phys_delta'].values)

    # ── Fusion classifier ─────────────────────────────────────────
    fc_base = FusionClassifier()
    with contextlib.redirect_stdout(silent):
        fc_base.fit(ev_train, train_df['final_label'])
    fc = comps.get('ensemble_fc', fc_base)

    # ── Adaptive threshold ────────────────────────────────────────
    adaptive_tau = AdaptiveThreshold(beta=1.5)
    with contextlib.redirect_stdout(silent):
        adaptive_tau.fit(train_df['final_label'].values,
                         fc_base.predict_proba_score(ev_train))

    if comps.get('weighted_tau_h') is not None:
        adaptive_tau.tau_high       = comps['weighted_tau_h']
        adaptive_tau._base_tau_high = comps['weighted_tau_h']
        adaptive_tau.tau_low        = comps['weighted_tau_l']

    TAU_HIGH = adaptive_tau.tau_high

    # ── Supporting components ─────────────────────────────────────
    policy_engine = PolicyEngine()
    risk_scorer   = UnifiedRiskScorer(policy_engine)
    trust_tracker = ModalityTrustTracker()
    for mod, val in comps.get('stability_trust', {}).items():
        if mod in trust_tracker.trust:
            trust_tracker.trust[mod] = val

    contextual_tau = ContextualThresholdEngine(adaptive_tau)
    with contextlib.redirect_stdout(silent):
        contextual_tau.auto_tune(train_df['final_label'].values,
                                 fc_base.predict_proba_score(ev_train))

    fsm_thr = comps.get('fsm_thresholds', {3:0.62, 2:0.42, 1:0.28})
    threat_fsm = ThreatStateMachine()
    scan_energy = ScanEnergy()
    threat_fsm.THRESHOLDS = fsm_thr

    # ── Inference — full loop (same as main pipeline) ─────────────
    memory       = ExperienceMemory()
    behavioral_mem = BehavioralThreatMemory()
    outcome_mem  = OutcomeMemory()

    pn_te, pi_te, pl_te = lm.predict_proba_raw(test_df)
    pn_te, pi_te, pl_te = calibrate(pn_te), calibrate(pi_te), calibrate(pl_te)
    ev_test = build_evidence_vector(pn_te, pi_te, pl_te,
                                    test_df['phys_delta'].values)
    fs_all  = fc.predict_proba_score(ev_test)

    # warmup
    warmup = max(int(len(test_df)*0.3), 50)
    for i in range(warmup):
        pn = float(pn_te[i]); pi = float(pi_te[i]); pl = float(pl_te[i])
        d  = float(test_df['phys_delta'].iloc[i]); fs = float(fs_all[i])
        ev_v = [float(ev_test['L_net'].iloc[i]), float(ev_test['L_iot'].iloc[i]),
                float(ev_test['L_log'].iloc[i]), d]
        _, sig = memory.lookup(pn, pi, pl, d)
        memory.store(sig, int(fs > TAU_HIGH), 'warmup', fs, ev_v,
                     generate_pseudo_label(fs, pn, pi, pl))
    with contextlib.redirect_stdout(silent):
        memory.run_dbscan()

    decisions = []
    for i in range(len(test_df)):
        pn = float(pn_te[i]); pi = float(pi_te[i]); pl = float(pl_te[i])
        d  = float(test_df['phys_delta'].iloc[i]); fs = float(fs_all[i])
        ev_v = [float(ev_test['L_net'].iloc[i]), float(ev_test['L_iot'].iloc[i]),
                float(ev_test['L_log'].iloc[i]), d]

        if SCAN_FSM and 'scan_energy' in dir():
            _s = str(test_df['src_ip'].iloc[i]) if 'src_ip' in test_df.columns else 'unknown'
            _c = lambda c: (float(test_df[c].iloc[i]) if c in test_df.columns else 0.0)
            scan_energy.observe(_s, _c('phys_fanout'),
                                _c('phys_port_entropy'), _c('phys_fail_rate'))

        cached, sig = memory.lookup(pn, pi, pl, d)
        if cached is not None:
            dec = cached['decision']
        else:
            def _get(col):
                return float(test_df[col].iloc[i])                        if col in test_df.columns else 0.0
            row_feats = {
                'p_net':pn,'p_iot':pi,'p_log':pl,'phys_delta':d,
                'phys_delta_abs':abs(d),
                'disk_write':_get('disk_write'),
                'cpu_usage':_get('cpu_usage'),
                'cpu_spike':_get('cpu_spike'),
                'mem_vgrow':_get('mem_vgrow'),
                'page_faults_maj':_get('page_faults_maj'),
                'src_bytes':_get('net_bytes'),
                'temp_roll_std':_get('temp_roll_std'),
            }
            forced, _ = check_forced_policies(row_feats)
            if forced:
                decisions.append(1)
                memory.store(sig, 1, 'FORCED', 0.99, ev_v,
                             generate_pseudo_label(fs, pn, pi, pl))
                continue

            pn_t, pi_t, pl_t = trust_tracker.apply(pn, pi, pl)
            cid_r, c_risk, _ = memory.find_closest_cluster(ev_v)
            R, _, p_flags = risk_scorer.score(
                pn_t, pi_t, pl_t, d, 0.0, c_risk, row_feats, adaptive_tau)

            layer_std = float(np.std([pn, pi, pl]))
            tau_h_ctx, tau_l_ctx, _ = contextual_tau.get_tau(
                layer_disagreement=layer_std,
                trust_scores=trust_tracker.trust)
            adaptive_tau.tau_high = tau_h_ctx
            adaptive_tau.tau_low  = tau_l_ctx

            # ── Per-source scan accumulator ─────────────────────────
            # Scoped feedback: evidence about this host lowers the bar for this
            # host only, and decays. Not a global escalation loop.
            if SCAN_FSM and 'scan_energy' in dir():
                _src = (str(test_df['src_ip'].iloc[i])
                        if 'src_ip' in test_df.columns else 'unknown')
                _rel = scan_energy.relief(_src)   # energy already accumulated above
                if _rel:
                    adaptive_tau.tau_high = max(adaptive_tau.tau_high - _rel,
                                                ScanEnergy.FLOOR)
                    tau_h_ctx = adaptive_tau.tau_high

            _inband = _band_hit('site1_cv_fold_loop', fs, R, tau_l_ctx, tau_h_ctx)
            eff_score = R if _inband else fs
            dec, reason, conf = hierarchical_decision(
                pn_t, pi_t, pl_t, d, eff_score, ev_v, memory, adaptive_tau)

            t_level, _, _ = score_to_threat_level(eff_score, 0.0)
            camp_risk = behavioral_mem.get_campaign_risk()
            s_level = threat_fsm.update(
                t_level, cluster_risk=max(c_risk, camp_risk),
                policy_risk=0.0)
            # Confidence gate: FSM cannot override events ML
            # is confident are normal (score below tau_low)
            if s_level >= 2 and eff_score > adaptive_tau.tau_low and not NO_ESCALATION:
                dec = 1

            trust_tracker.update(dec, pn, pi, pl)

            intent_prof = {}
            for c in ['atk_scan','atk_dos','atk_brute_force',
                      'atk_exfiltration','atk_injection','atk_mitm',
                      'atk_ransomware','atk_backdoor','atk_password']:
                val = float(test_df[c].iloc[i])                       if c in test_df.columns else 0.0
                if val > 0:
                    intent_prof[c.replace('atk_','')] = val

            with contextlib.redirect_stdout(silent):
                behavioral_mem.store(
                    ts=float(test_df['ts'].iloc[i])
                       if 'ts' in test_df.columns else float(i),
                    intent_profile=intent_prof,
                    temporal_profile={},
                    escalation_path=[THREAT_LEVELS[s_level]['name']],
                    net_risk=float(eff_score))

            dom_tag = max(intent_prof, key=intent_prof.get)                       if intent_prof else 'unknown'
            outcome_mem.record(dom_tag, eff_score,
                               'confirmed' if dec == 1 else 'false_positive')
            memory.store(sig, dec, reason, conf, ev_v,
                         generate_pseudo_label(fs, pn, pi, pl))

        decisions.append(dec)

    y_true = test_df['final_label'].values
    y_pred = np.array(decisions[:len(y_true)])
    n_nrm  = int((y_true == 0).sum())
    fp_c   = int(((y_pred == 1) & (y_true == 0)).sum())

    return {
        'f1':   round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        'prec': round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        'rec':  round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        'auc':  round(float(roc_auc_score(
                    y_true, fc.predict_proba_score(ev_test))), 4),
        'fpr':  round(fp_c / max(n_nrm, 1), 4),
    }


# ════════════════════════════════════════════════════════════════════
# FPR ABLATION STUDY
# ════════════════════════════════════════════════════════════════════

def _pipeline_ablation(fused_df, full_fused_df, comps,
                        no_drift=False,
                        no_fsm_override=False,
                        static_threshold=False):
    """
    Full pipeline run with specific adaptive components selectively
    disabled. Used for FPR ablation study only — not for main results.

    Flags:
      no_drift         : alpha=0, threshold never pushed down by drift
      no_fsm_override  : FSM state does not override ML decision
      static_threshold : tau_high fixed, no EWMA updates during inference
    """
    import io, sys, contextlib
    from sklearn.metrics import (f1_score, precision_score,
                                 recall_score, roc_auc_score)

    train_df, _ = train_test_split(
        fused_df, test_size=0.3, random_state=42,
        stratify=fused_df['final_label'])
    train_df = train_df.reset_index(drop=True)
    test_df  = full_fused_df.reset_index(drop=True)

    silent = io.StringIO()

    lm = LayerModels()
    with contextlib.redirect_stdout(silent):
        lm.fit(train_df)

    pn_r, pi_r, pl_r = lm.predict_proba_raw(train_df)
    ev_train = build_evidence_vector(
        calibrate(pn_r), calibrate(pi_r),
        calibrate(pl_r), train_df['phys_delta'].values)

    fc_base = FusionClassifier()
    with contextlib.redirect_stdout(silent):
        fc_base.fit(ev_train, train_df['final_label'])
    fc = comps.get('ensemble_fc', fc_base)

    adaptive_tau = AdaptiveThreshold(beta=1.5)
    with contextlib.redirect_stdout(silent):
        adaptive_tau.fit(train_df['final_label'].values,
                         fc_base.predict_proba_score(ev_train))

    TAU_HIGH = adaptive_tau.tau_high

    # ── Apply ablation flags ──────────────────────────────────────
    if static_threshold:
        # Freeze threshold — no EWMA, no drift updates
        adaptive_tau.ewma_alpha   = 0.0
        adaptive_tau._ewma_score  = adaptive_tau._baseline_mean

    policy_engine = PolicyEngine()
    risk_scorer   = UnifiedRiskScorer(policy_engine)
    trust_tracker = ModalityTrustTracker()
    for mod, val in comps.get('stability_trust', {}).items():
        if mod in trust_tracker.trust:
            trust_tracker.trust[mod] = val

    contextual_tau = ContextualThresholdEngine(adaptive_tau)

    if no_drift:
        # Disable drift contribution to threshold
        contextual_tau.ALPHA = 0.0

    with contextlib.redirect_stdout(silent):
        contextual_tau.auto_tune(train_df['final_label'].values,
                                 fc_base.predict_proba_score(ev_train))

    threat_fsm = ThreatStateMachine()
    scan_energy = ScanEnergy()
    memory       = ExperienceMemory()
    behavioral_mem = BehavioralThreatMemory()
    outcome_mem  = OutcomeMemory()

    pn_te, pi_te, pl_te = lm.predict_proba_raw(test_df)
    pn_te, pi_te, pl_te = calibrate(pn_te), calibrate(pi_te), calibrate(pl_te)
    ev_test = build_evidence_vector(pn_te, pi_te, pl_te,
                                    test_df['phys_delta'].values)
    fs_all  = fc.predict_proba_score(ev_test)

    # warmup
    warmup = max(int(len(test_df)*0.3), 50)
    for i in range(warmup):
        pn = float(pn_te[i]); pi = float(pi_te[i]); pl = float(pl_te[i])
        d  = float(test_df['phys_delta'].iloc[i]); fs = float(fs_all[i])
        ev_v = [float(ev_test['L_net'].iloc[i]),
                float(ev_test['L_iot'].iloc[i]),
                float(ev_test['L_log'].iloc[i]), d]
        _, sig = memory.lookup(pn, pi, pl, d)
        memory.store(sig, int(fs > TAU_HIGH), 'warmup', fs, ev_v,
                     generate_pseudo_label(fs, pn, pi, pl))
    with contextlib.redirect_stdout(silent):
        memory.run_dbscan()

    decisions = []
    for i in range(len(test_df)):
        pn = float(pn_te[i]); pi = float(pi_te[i]); pl = float(pl_te[i])
        d  = float(test_df['phys_delta'].iloc[i]); fs = float(fs_all[i])
        ev_v = [float(ev_test['L_net'].iloc[i]),
                float(ev_test['L_iot'].iloc[i]),
                float(ev_test['L_log'].iloc[i]), d]

        if SCAN_FSM and 'scan_energy' in dir():
            _s = str(test_df['src_ip'].iloc[i]) if 'src_ip' in test_df.columns else 'unknown'
            _c = lambda c: (float(test_df[c].iloc[i]) if c in test_df.columns else 0.0)
            scan_energy.observe(_s, _c('phys_fanout'),
                                _c('phys_port_entropy'), _c('phys_fail_rate'))

        cached, sig = memory.lookup(pn, pi, pl, d)
        if cached is not None:
            dec = cached['decision']
        else:
            def _get(col):
                return float(test_df[col].iloc[i])                        if col in test_df.columns else 0.0
            row_feats = {
                'p_net':pn,'p_iot':pi,'p_log':pl,'phys_delta':d,
                'phys_delta_abs':abs(d),
                'disk_write':_get('disk_write'),
                'cpu_usage':_get('cpu_usage'),
                'cpu_spike':_get('cpu_spike'),
                'mem_vgrow':_get('mem_vgrow'),
                'page_faults_maj':_get('page_faults_maj'),
                'src_bytes':_get('net_bytes'),
                'temp_roll_std':_get('temp_roll_std'),
            }
            forced, _ = check_forced_policies(row_feats)
            if forced:
                decisions.append(1)
                memory.store(sig, 1, 'FORCED', 0.99, ev_v,
                             generate_pseudo_label(fs, pn, pi, pl))
                continue

            pn_t, pi_t, pl_t = trust_tracker.apply(pn, pi, pl)
            _, c_risk, _ = memory.find_closest_cluster(ev_v)

            layer_std = float(np.std([pn, pi, pl]))
            tau_h_ctx, tau_l_ctx, _ = contextual_tau.get_tau(
                layer_disagreement=layer_std,
                trust_scores=trust_tracker.trust)

            if not static_threshold:
                adaptive_tau.tau_high = tau_h_ctx
                adaptive_tau.tau_low  = tau_l_ctx

            eff_score = fs
            dec, reason, conf = hierarchical_decision(
                pn_t, pi_t, pl_t, d, eff_score, ev_v,
                memory, adaptive_tau)

            t_level, _, _ = score_to_threat_level(eff_score, 0.0)
            camp_risk = behavioral_mem.get_campaign_risk()
            s_level = threat_fsm.update(
                t_level,
                cluster_risk=max(c_risk, camp_risk),
                policy_risk=0.0)

            # ── KEY ABLATION POINT ────────────────────────────────
            if not no_fsm_override:
                if s_level >= 2:
                    dec = 1   # FSM escalation overrides ML decision

            trust_tracker.update(dec, pn, pi, pl)

            intent_prof = {}
            for c in ['atk_scan','atk_dos','atk_brute_force',
                      'atk_exfiltration','atk_injection','atk_mitm',
                      'atk_ransomware','atk_backdoor','atk_password']:
                val = float(test_df[c].iloc[i])                       if c in test_df.columns else 0.0
                if val > 0:
                    intent_prof[c.replace('atk_','')] = val

            with contextlib.redirect_stdout(silent):
                behavioral_mem.store(
                    ts=float(test_df['ts'].iloc[i])
                       if 'ts' in test_df.columns else float(i),
                    intent_profile=intent_prof,
                    temporal_profile={},
                    escalation_path=[THREAT_LEVELS[s_level]['name']],
                    net_risk=float(eff_score))

            memory.store(sig, dec, reason, conf, ev_v,
                         generate_pseudo_label(fs, pn, pi, pl))

        decisions.append(dec)

    y_true = test_df['final_label'].values
    y_pred = np.array(decisions[:len(y_true)])
    n_nrm  = int((y_true == 0).sum())
    fp_c   = int(((y_pred == 1) & (y_true == 0)).sum())

    return {
        'f1':   round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        'prec': round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        'rec':  round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        'auc':  round(float(roc_auc_score(
                    y_true, fc.predict_proba_score(ev_test))), 4),
        'fpr':  round(fp_c / max(n_nrm, 1), 4),
    }


def run_fpr_ablation(fused_df, full_fused_df, cv_components):
    """
    FPR ablation harness (older experiment).

    NOT the evaluated system: its loop has an ungated escalation override, no
    band substitution, and a "no drift" variant that auto_tune() reverts. No
    figure in the paper comes from it; the paper's attributions use the
    switches of the main pipeline instead.
    """
    print("  [fpr_ablation] note: this harness does not run the evaluated decision rule")
    variants = [
        # name, no_drift, no_fsm_override, static_threshold
        ("Full CATF-IDS (main result)",   False, False, False),
        ("No drift  (alpha=0)",           True,  False, False),
        ("No FSM override",               False, True,  False),
        ("Static threshold",              False, False, True),
        ("Like published papers (static+no FSM)", False, True,  True),
    ]

    print("\n" + "="*105)
    print("  FPR ABLATION STUDY — Cost of Adaptability")
    print("="*105)
    print(f"  {'Variant':<42}  {'FPR':>7}  {'F1':>7}  "
          f"{'Recall':>7}  {'AUC':>7}  {'Prec':>7}  {'ΔFPR':>8}")
    print("-"*105)

    results = []
    baseline_fpr = None

    for name, nd, nf, st in variants:
        m = _pipeline_ablation(
            fused_df, full_fused_df, cv_components,
            no_drift=nd,
            no_fsm_override=nf,
            static_threshold=st)

        if baseline_fpr is None:
            baseline_fpr = m['fpr']
            delta_str = "  baseline"
        else:
            delta = m['fpr'] - baseline_fpr
            delta_str = f"  {delta:+.4f}"

        display_name = name.split("\n")[0]
        results.append((name, m, delta_str))
        print(f"  {display_name:<42}  {m['fpr']:>7.4f}  {m['f1']:>7.4f}  "
              f"{m['rec']:>7.4f}  {m['auc']:>7.4f}  "
              f"{m['prec']:>7.4f}  {delta_str}")

    print("="*105)
    print("\n  What each variant sacrifices:")
    sacrifices = [
        "Nothing — full adaptive governance architecture",
        "Environmental adaptation to attack streams",
        "Temporal campaign detection and multi-stage escalation",
        "All threshold adaptation — fixed boundary forever",
        "Everything novel — equivalent to static ML classifier",
    ]
    for (name, m, _), sac in zip(results, sacrifices):
        display = name.split("\n")[0]
        print(f"  {display:<42}  sacrifices: {sac}")

    print("\n  Thesis interpretation:")
    last_fpr = results[-1][1]['fpr']
    last_rec = results[-1][1]['rec']
    full_fpr = results[0][1]['fpr']
    full_rec = results[0][1]['rec']
    fpr_diff  = full_fpr - last_fpr
    rec_diff  = full_rec - last_rec
    fpr_pct = (fpr_diff/full_fpr*100) if full_fpr > 0 else 0.0
    rec_pct = (rec_diff/full_rec*100) if full_rec > 0 else 0.0
    print(f"  Removing all adaptive components reduces FPR by "
          f"{fpr_diff:.4f} ({fpr_pct:.1f}%)")
    print(f"  but reduces Recall by "
          f"{rec_diff:.4f} ({rec_pct:.1f}%)")
    print(f"  Published papers operating in this mode report "
          f"FPR ≈ {last_fpr:.3f} while hiding this recall cost.")
    print("="*105 + "\n")

    return results


def run_quick_compare(fused_df, full_fused_df, fold_results):
    """
    Targeted comparison of the 4 most promising combinations
    plus the baseline, printed as a clean side-by-side table.
    """
    combos = [
        # label                         l1          l2          l3           l4
        ("Baseline (no CV)",            'equal',    'single_run','default',  'current'),
        ("Previous best (auc_sq)",      'auc_sq',   'single_run','default',  'sensitive'),
        ("FPR-inv + min_fpr + stab",    'fpr_inv',  'min_fpr',  'stability', 'sensitive'),
        ("FPR-inv + ci_lower + stab",   'fpr_inv',  'ci_lower', 'stability', 'sensitive'),
        ("FPR-inv + min_fpr + standard",'fpr_inv',  'min_fpr',  'stability', 'current'),
        ("AUC + min_fpr + stab",        'auc',      'min_fpr',  'stability', 'sensitive'),
    ]

    print("\n" + "="*95)
    print("  TARGETED COMBINATION COMPARISON")
    print("="*95)
    print(f"  {'#':<2}  {'Combination':<38}  {'F1':>7}  {'AUC':>7}  "
          f"{'FPR':>7}  {'Prec':>7}  {'Rec':>7}")
    print("-"*95)

    results = []
    for i, (label, l1, l2, l3, l4) in enumerate(combos, 1):
        comps = extract_cv_components(fold_results, l1, l2, l3, l4)
        m     = _pipeline_with_components(fused_df, full_fused_df, comps)
        results.append((label, m))
        print(f"  {i:<2}  {label:<38}  {m['f1']:>7.4f}  {m['auc']:>7.4f}  "
              f"{m['fpr']:>7.4f}  {m['prec']:>7.4f}  {m['rec']:>7.4f}")

    # Find best per metric
    print("-"*95)
    best_f1   = max(results, key=lambda x: x[1]['f1'])
    best_auc  = max(results, key=lambda x: x[1]['auc'])
    best_fpr  = min(results, key=lambda x: x[1]['fpr'])
    best_prec = max(results, key=lambda x: x[1]['prec'])
    print(f"  Best F1:        {best_f1[0]:<38}  {best_f1[1]['f1']:.4f}")
    print(f"  Best AUC:       {best_auc[0]:<38}  {best_auc[1]['auc']:.4f}")
    print(f"  Best FPR:       {best_fpr[0]:<38}  {best_fpr[1]['fpr']:.4f}")
    print(f"  Best Precision: {best_prec[0]:<38}  {best_prec[1]['prec']:.4f}")

    # Score each combo: rank across all 4 metrics, lower total rank = better overall
    print("\n  Overall ranking (sum of metric ranks — lower is better):")
    scored = []
    for metric, reverse in [('f1',True),('auc',True),('fpr',False),('prec',True)]:
        ranked = sorted(results, key=lambda x: x[1][metric], reverse=reverse)
        for rank, (label, _) in enumerate(ranked, 1):
            scored.append((label, rank))

    totals = {}
    for label, rank in scored:
        totals[label] = totals.get(label, 0) + rank

    for rank, (label, total) in enumerate(sorted(totals.items(),
                                                   key=lambda x: x[1]), 1):
        bar = '█' * (10 - total + len(combos))
        print(f"  {rank}. {label:<38}  score={total}  {bar}")

    # Return best overall combo
    best_label = min(totals, key=totals.get)
    best_combo = next(c for c in combos if c[0] == best_label)
    best_comps = extract_cv_components(fold_results,
                                       best_combo[1], best_combo[2],
                                       best_combo[3], best_combo[4])
    print(f"\n  → Best overall: {best_label}")
    print("="*95 + "\n")
    return best_comps

def run_cross_validation(fused_df, n_splits=5):
    """
    Stateful K-Fold Cross-Validation for CATF-IDS.

    Each fold:
      - Resets ALL stateful components (fresh start)
      - Trains layer models + fusion classifier
      - Runs full inference pipeline
      - Saves: metrics + tau + trust + energy dist + memory entries + OL model

    Aggregation at end:
      - Metrics       → mean ± std
      - Tau           → averaged cross-validated thresholds
      - Trust         → averaged with std (variance = reliability signal)
      - Memory        → fingerprinted union (deduplication)
      - OL model      → 5-model ensemble (average probabilities)
      - Energy dist   → typical escalation profile
    """
    print("\n" + "="*60)
    print("  STATEFUL 5-FOLD CROSS-VALIDATION")
    print("="*60)
    print(f"  Dataset: {len(fused_df)} rows  "
          f"(attack={int((fused_df['final_label']==1).sum())}  "
          f"normal={int((fused_df['final_label']==0).sum())})")
    print(f"  Folds: {n_splits}  |  ~{len(fused_df)//n_splits} test rows per fold")

    skf          = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(
            skf.split(fused_df, fused_df['final_label'])):

        print(f"\n── Fold {fold_idx+1}/{n_splits} "
              f"(train={len(train_idx)}, test={len(test_idx)}) ──")

        train_df = fused_df.iloc[train_idx].reset_index(drop=True)
        test_df  = fused_df.iloc[test_idx].reset_index(drop=True)
        fr       = FoldResult(fold_idx + 1)

        # ── Train layer models ──────────────────────────────────
        lm_cv = LayerModels()
        lm_cv.fit(train_df)
        pn_r, pi_r, pl_r = lm_cv.predict_proba_raw(train_df)
        ev_train = build_evidence_vector(
            calibrate(pn_r), calibrate(pi_r),
            calibrate(pl_r), train_df['phys_delta'].values)

        # ── Train fusion classifier ─────────────────────────────
        fc_cv = FusionClassifier()
        fc_cv.fit(ev_train, train_df['final_label'])
        cal_cv = SiteCalibration(train_df) if (SITE_ENVELOPES or SITE_LIMITS) else None

        # ── Fit adaptive threshold ──────────────────────────────
        tau_cv = AdaptiveThreshold(beta=1.5)
        tau_cv.fit(train_df['final_label'].values,
                   fc_cv.predict_proba_score(ev_train))
        ctx_cv = ContextualThresholdEngine(tau_cv)
        ctx_cv.auto_tune(train_df['final_label'].values,
                         fc_cv.predict_proba_score(ev_train))

        # ── Fresh stateful components ───────────────────────────
        pe_cv  = SitePolicyEngine(cal_cv) if SITE_LIMITS else PolicyEngine()
        rs_cv  = UnifiedRiskScorer(pe_cv)
        tt_cv  = ModalityTrustTracker()
        fsm_cv = ThreatStateMachine()
        mem_cv = ExperienceMemory()
        bm_cv  = BehavioralThreatMemory()
        om_cv  = OutcomeMemory()

        # ── Inference ───────────────────────────────────────────
        pn_te, pi_te, pl_te = lm_cv.predict_proba_raw(test_df)
        pn_te, pi_te, pl_te = calibrate(pn_te), calibrate(pi_te), calibrate(pl_te)
        ev_test = build_evidence_vector(pn_te, pi_te, pl_te,
                                        test_df['phys_delta'].values)
        fs_all  = fc_cv.predict_proba_score(ev_test)
        env_hits = (cal_cv.envelope_hits(test_df)[0] if SITE_ENVELOPES
                    else np.zeros(len(test_df), dtype=bool))

        decisions = []
        warmup    = max(int(len(test_df)*0.3), 30)

        # warmup pass
        for i in range(min(warmup, len(test_df))):
            pn = float(pn_te[i]); pi = float(pi_te[i]); pl = float(pl_te[i])
            d  = float(test_df['phys_delta'].iloc[i]); fs = float(fs_all[i])
            ev_v = [float(ev_test['L_net'].iloc[i]),
                    float(ev_test['L_iot'].iloc[i]),
                    float(ev_test['L_log'].iloc[i]), d]
            _, sig = mem_cv.lookup(pn, pi, pl, d)
            mem_cv.store(sig, int(fs > (TAU_HIGH if ONE_PATH else tau_cv.tau_high)), 'warmup',
                         fs, ev_v, generate_pseudo_label(fs, pn, pi, pl))

        if ONE_PATH:
            mem_cv.run_dbscan()          # as the stream does after its warm-up

        # main inference pass
        for i in range(len(test_df)):
            if ONE_PATH:
                # Configuration D: the event is decided exactly as run_pipeline_multi
                # decides it -- cache, forced alerts, policy pressure, trust damping,
                # cluster and risk score, contextual thresholds, band, fusion
                # decision, energy state machine, gated escalation, memories.
                pn = float(pn_te[i]); pi = float(pi_te[i]); pl = float(pl_te[i])
                d  = float(test_df['phys_delta'].iloc[i]); fs = float(fs_all[i])
                ev_v = [float(ev_test['L_net'].iloc[i]), float(ev_test['L_iot'].iloc[i]),
                        float(ev_test['L_log'].iloc[i]), d]

                def _g(col):
                    return float(test_df[col].iloc[i]) if col in test_df.columns else 0.0
                rowf = {'p_net': pn, 'p_iot': pi, 'p_log': pl, 'phys_delta': d,
                        'phys_delta_abs': abs(d), 'disk_write': _g('disk_write'),
                        'cpu_usage': _g('cpu_usage'), 'cpu_spike': _g('cpu_spike'),
                        'mem_vgrow': _g('mem_vgrow'), 'page_faults_maj': _g('page_faults_maj'),
                        'src_bytes': _g('net_bytes'), 'temp_roll_std': _g('temp_roll_std'),
                        'net_bytes': _g('net_bytes'), 'net_dbytes': _g('net_dbytes'),
                        'iot_device': _g('iot_device')}
                cached, sig = mem_cv.lookup(pn, pi, pl, d)
                if cached:
                    decisions.append(int(cached['decision']))
                    continue
                forced, _ = check_forced_policies(rowf)
                if (forced and not NO_FORCED) or (SITE_ENVELOPES and bool(env_hits[i])):
                    decisions.append(1)
                    continue
                P_risk, _ = pe_cv.evaluate(rowf)
                pn_t, pi_t, pl_t = tt_cv.apply(pn, pi, pl)
                _, c_risk, _ = mem_cv.find_closest_cluster(ev_v)
                R, _, _ = rs_cv.score(pn_t, pi_t, pl_t, d, 0.0, c_risk, rowf, tau_cv)
                tau_h_ctx, tau_l_ctx, _ = ctx_cv.get_tau(
                    layer_disagreement=float(np.std([pn, pi, pl])), trust_scores=tt_cv.trust)
                tau_cv.tau_high, tau_cv.tau_low = tau_h_ctx, tau_l_ctx
                eff = R if _band_hit('cv_folds', fs, R, tau_l_ctx, tau_h_ctx) else fs
                dec, reason, conf = hierarchical_decision(pn_t, pi_t, pl_t, d, eff, ev_v,
                                                          mem_cv, tau_cv)
                t_level, _, _ = score_to_threat_level(eff, P_risk)
                camp_risk = bm_cv.get_campaign_risk()
                s_level = fsm_cv.update(t_level, cluster_risk=max(c_risk, camp_risk),
                                        policy_risk=min(P_risk, 1.0))
                s_name = THREAT_LEVELS[s_level]['name']
                if s_level >= 2 and eff > tau_cv.tau_low and not NO_ESCALATION:
                    dec = 1
                final_dec = int(dec)
                tt_cv.update(final_dec, pn, pi, pl)
                mem_cv.store(sig, final_dec, reason, conf, ev_v,
                             generate_pseudo_label(fs, pn, pi, pl))
                intent_prof = {}
                for c in ('atk_scan', 'atk_dos', 'atk_brute_force', 'atk_exfiltration',
                          'atk_injection', 'atk_mitm', 'atk_ransomware', 'atk_backdoor',
                          'atk_password'):
                    val = float(test_df[c].iloc[i]) if c in test_df.columns else 0.0
                    if val > 0:
                        intent_prof[c.replace('atk_', '')] = val
                ts_val = float(test_df['ts'].iloc[i]) if 'ts' in test_df.columns else float(i)
                bm_cv.store(ts=ts_val, intent_profile=intent_prof,
                            temporal_profile={"periodicity": 0, "burst": 0, "fail_rate": 0},
                            escalation_path=[s_name], net_risk=float(eff))
                dom_tag = max(intent_prof, key=intent_prof.get) if intent_prof else "unknown"
                om_cv.record(dom_tag, eff, 'confirmed' if final_dec == 1 else 'false_positive')
                if intent_prof:
                    fr.mem_entries.append({'fingerprint': _fingerprint(ts_val, final_dec, fs),
                                           'tag': dom_tag, 'risk': round(fs, 3),
                                           'outcome': ('confirmed' if final_dec == 1
                                                       else 'false_positive')})
                decisions.append(final_dec)
                continue
            pn = float(pn_te[i]); pi = float(pi_te[i]); pl = float(pl_te[i])
            d  = float(test_df['phys_delta'].iloc[i]); fs = float(fs_all[i])
            row_feats = dict(test_df.iloc[i])
            row_feats.update({'p_net':pn,'p_iot':pi,'p_log':pl,'phys_delta':d})

            ev_v = [float(ev_test['L_net'].iloc[i]),
                    float(ev_test['L_iot'].iloc[i]),
                    float(ev_test['L_log'].iloc[i]), d]

            P_risk, _ = pe_cv.evaluate(row_feats)
            row_feats_cv = dict(test_df.iloc[i])
            R_tuple = rs_cv.score(
                pn, pi, pl,
                float(test_df['phys_delta'].iloc[i]),
                system_stress=float(test_df.get('system_stress', pd.Series([0]*len(test_df))).iloc[i]) if 'system_stress' in test_df.columns else 0.0,
                cluster_risk=mem_cv.find_closest_cluster([float(ev_test['L_net'].iloc[i]),
                             float(ev_test['L_iot'].iloc[i]),
                             float(ev_test['L_log'].iloc[i]),
                             float(test_df['phys_delta'].iloc[i])])[1],
                row_features=row_feats_cv,
                adaptive_tau=tau_cv)
            R = float(R_tuple[0]) if isinstance(R_tuple, tuple) else float(R_tuple)

            c_risk = 0.0
            cached, sig = mem_cv.lookup(pn, pi, pl, d)
            if cached is not None:
                dec, reason, conf = cached['decision'], 'MEM', cached['confidence']
            else:
                tau_eff, _tl, _eq = ctx_cv.get_tau(
                    cluster_risk=c_risk,
                    layer_disagreement=float(np.std([pn, pi, pl])),
                    trust_scores=tt_cv.trust)
                _, cr, _ = mem_cv.find_closest_cluster(ev_v)
                c_risk = cr
                dec, reason, conf = hierarchical_decision(
                    pn, pi, pl, d,
                    fs, ev_v, mem_cv, tau_cv)

            t_level, _, _ = score_to_threat_level(fs, P_risk, forced=(reason=='FORCED'))
            camp_risk = bm_cv.get_campaign_risk()
            s_level   = fsm_cv.update(
                t_level,
                cluster_risk=max(c_risk, camp_risk),
                policy_risk=min(P_risk, 1.0))
            s_name    = THREAT_LEVELS.get(s_level, {}).get('name', 'SAFE')

            # Confidence gate on CV loop
            _gate = getattr(tau_cv, 'tau_low', 0.15)
            if CV_FULL_RULE:
                # deployed rule: cache hit, else forced condition, else fusion
                # decision raised by gated escalation. --legacy_cv_rule restores
                # the original loop, which took every verdict from escalation
                def _g(col):
                    return float(test_df[col].iloc[i]) if col in test_df.columns else 0.0
                _rf = {'p_net': pn, 'p_iot': pi, 'p_log': pl, 'phys_delta': d,
                       'phys_delta_abs': abs(d), 'disk_write': _g('disk_write'),
                       'cpu_usage': _g('cpu_usage'), 'cpu_spike': _g('cpu_spike'),
                       'mem_vgrow': _g('mem_vgrow'), 'page_faults_maj': _g('page_faults_maj'),
                       'src_bytes': _g('net_bytes'), 'temp_roll_std': _g('temp_roll_std')}
                if cached is not None:
                    final_dec = int(dec)
                elif (((not NO_FORCED) and check_forced_policies(_rf)[0])
                      or (SITE_ENVELOPES and bool(env_hits[i]))):
                    final_dec = 1
                else:
                    final_dec = int(dec)
                    if s_level >= 2 and float(fs_all[i]) > _gate and not NO_ESCALATION:
                        final_dec = 1
            elif s_level >= 2 and float(fs_all[i]) > _gate:
                final_dec = 1
            else:
                final_dec = dec if reason == 'FORCED' else 0

            tt_cv.update(final_dec, pn, pi, pl)

            # Store behavioral entry with fingerprint
            ts_val = float(test_df['ts'].iloc[i])                      if 'ts' in test_df.columns else float(i)
            intent_prof = {c.replace('atk_',''): float(row_feats.get(c,0))
                           for c in row_feats if c.startswith('atk_')}
            if intent_prof:
                fp = _fingerprint(ts_val, final_dec, fs)
                fr.mem_entries.append({
                    'fingerprint': fp,
                    'tag':         max(intent_prof, key=intent_prof.get)
                                   if intent_prof else 'unknown',
                    'risk':        round(fs, 3),
                    'outcome':     'confirmed' if final_dec==1 else 'false_positive'
                })
                bm_cv.store(ts=ts_val, intent_profile=intent_prof,
                            temporal_profile={}, escalation_path=[s_name],
                            net_risk=fs)
                om_cv.record(max(intent_prof,key=intent_prof.get)
                             if intent_prof else 'unknown', fs,
                             'confirmed' if final_dec==1 else 'false_positive')

            mem_cv.store(sig, final_dec, reason, conf, ev_v,
                         generate_pseudo_label(fs, pn, pi, pl))
            decisions.append(final_dec)

        # ── Metrics ─────────────────────────────────────────────
        y_true = test_df['final_label'].values
        y_pred = np.array(decisions[:len(y_true)])
        if len(y_pred) > len(y_true): y_pred = y_pred[:len(y_true)]
        if len(y_pred) < len(y_true):
            y_pred = np.append(y_pred, np.zeros(len(y_true)-len(y_pred), dtype=int))

        from sklearn.metrics import (f1_score, precision_score,
                                     recall_score, roc_auc_score)
        n_nrm = int((y_true==0).sum())
        fp_count = int(((y_pred==1)&(y_true==0)).sum())
        fr.metrics = {
            'f1':        round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
            'precision': round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
            'recall':    round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
            'fpr':       round(fp_count / max(n_nrm, 1), 4),
            'roc_auc':   round(float(roc_auc_score(y_true,
                               fc_cv.predict_proba_score(ev_test))), 4),
        }

        # ── Save component states ───────────────────────────────
        fr.tau = {
            'tau_high': round(tau_cv.tau_high, 3),
            'tau_low':  round(tau_cv.tau_low,  3),
            'alpha':    ctx_cv.ALPHA,
            'delta':    ctx_cv.DELTA,
            'fbeta':    round(tau_cv._best_fbeta if hasattr(tau_cv,'_best_fbeta')
                              else 0.85, 4),
        }
        # the stream's ensemble member: the validated fold model, frozen before
        # the online learner below refits fc_cv in place (configuration D)
        fr.fusion_clf = copy.deepcopy(fc_cv) if FROZEN_ENSEMBLE else fc_cv
        fr.trust       = dict(tt_cv.trust)
        fr.energy_dist = dict(fsm_cv.stats().get('distribution', {}))

        # Save online learner model if retrain happened
        try:
            ol_cv = OnlineLearner(fc_cv, mem_cv, mode='incremental')
            ol_cv.retrain()
            fr.ol_weights = fc_cv.model  # updated model
        except Exception:
            fr.ol_weights = fc_cv.model

        fold_results.append(fr)

        m = fr.metrics
        print(f"  F1={m['f1']:.4f}  Prec={m['precision']:.4f}  "
              f"Rec={m['recall']:.4f}  AUC={m['roc_auc']:.4f}  "
              f"FPR={m['fpr']:.4f}")

    # ════════════════════════════════════════════════════════════
    # AGGREGATION
    # ════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("  CROSS-VALIDATION AGGREGATED RESULTS")
    print("="*60)

    # 1. Metrics — mean ± std
    metric_keys = ['f1','precision','recall','roc_auc','fpr']
    print("\n[CV Metrics — mean ± std across 5 folds]")
    cv_metrics = {}
    for k in metric_keys:
        vals = np.array([fr.metrics[k] for fr in fold_results])
        cv_metrics[k] = (round(float(vals.mean()),4), round(float(vals.std()),4))
        bar = '█' * int(vals.mean()*20)
        print(f"  {k:<12} {vals.mean():.4f} ± {vals.std():.4f}  {bar}")

    # 2. Cross-validated threshold
    print("\n[CV Threshold — averaged across folds]")
    for k in ['tau_high','tau_low','alpha','delta']:
        vals = np.array([fr.tau[k] for fr in fold_results])
        print(f"  {k:<12} {vals.mean():.3f} ± {vals.std():.3f}")

    # 3. Cross-validated trust
    print("\n[CV Modality Trust — averaged across folds]")
    for mod in ['net','iot','log','win']:
        vals = np.array([fr.trust.get(mod, 0.7) for fr in fold_results])
        stable = "✅ stable" if vals.std() < 0.05 else "⚠️  variable"
        print(f"  {mod:<6} {vals.mean():.3f} ± {vals.std():.3f}  {stable}")

    # 4. Energy distribution across folds
    print("\n[CV Energy Distribution — typical escalation profile]")
    all_states = set()
    for fr in fold_results:
        all_states.update(fr.energy_dist.keys())
    for state in ['SAFE','SUSPICIOUS','ELEVATED','CRITICAL']:
        if state in all_states:
            vals = np.array([fr.energy_dist.get(state, 0) for fr in fold_results])
            print(f"  {state:<12} {vals.mean():.1f} ± {vals.std():.1f} events/fold")

    # 5. Behavioral memory — fingerprinted union
    print("\n[CV Behavioral Memory — deduplicated union across all folds]")
    seen_fps  = set()
    union_mem = []
    for fr in fold_results:
        for entry in fr.mem_entries:
            if entry['fingerprint'] not in seen_fps:
                seen_fps.add(entry['fingerprint'])
                union_mem.append(entry)

    from collections import Counter
    tag_counts = Counter(e['tag'] for e in union_mem)
    dup_rate   = 1.0 - len(union_mem) / max(
        sum(len(fr.mem_entries) for fr in fold_results), 1)
    print(f"  Total entries before dedup : "
          f"{sum(len(fr.mem_entries) for fr in fold_results)}")
    print(f"  After deduplication        : {len(union_mem)}")
    print(f"  Deduplication rate         : {dup_rate:.1%}")
    print(f"  Top behaviors: {dict(tag_counts.most_common(5))}")

    # 6. Ensemble model info
    # ── LEVEL 1: AUC-weighted ensemble ──────────────────────────────
    print("\n[CV Level 1 — AUC-Weighted Soft Voting Ensemble]")
    fold_clfs  = [fr.fusion_clf for fr in fold_results if fr.fusion_clf is not None]
    auc_vals   = np.array([fr.metrics['roc_auc'] for fr in fold_results])
    auc_w      = auc_vals / auc_vals.sum()
    ensemble_fc = EnsembleFusionClassifier(fold_clfs, auc_vals)
    ensemble_fc.print_summary()
    print(f"  AUC range: {auc_vals.min():.4f} – {auc_vals.max():.4f}")

    # ── LEVEL 2: F-beta weighted threshold ───────────────────────────
    print("\n[CV Level 2 — F-Beta Weighted Threshold]")
    fbetas    = np.array([fr.tau.get('fbeta', 0.85) for fr in fold_results])
    taus_h    = np.array([fr.tau['tau_high'] for fr in fold_results])
    taus_l    = np.array([fr.tau['tau_low']  for fr in fold_results])
    alphas    = np.array([fr.tau['alpha']    for fr in fold_results])
    tau_w_h   = float(np.dot(fbetas, taus_h) / fbetas.sum())
    tau_w_l   = float(np.dot(fbetas, taus_l) / fbetas.sum())
    alpha_w   = float(np.dot(fbetas, alphas) / fbetas.sum())
    print(f"  tau_high: naive_mean={taus_h.mean():.3f}  "
          f"fbeta_weighted={tau_w_h:.3f}  (delta={tau_w_h-taus_h.mean():+.3f})")
    print(f"  tau_low:  naive_mean={taus_l.mean():.3f}  "
          f"fbeta_weighted={tau_w_l:.3f}  (delta={tau_w_l-taus_l.mean():+.3f})")
    print(f"  alpha:    naive_mean={alphas.mean():.3f}  "
          f"fbeta_weighted={alpha_w:.3f}")

    # ── LEVEL 3: Stability-penalized trust ───────────────────────────
    print("\n[CV Level 3 — Stability-Penalized Trust]")
    stability_trust = {}
    for mod in ['net','iot','log','win']:
        vals  = np.array([fr.trust.get(mod, 0.7) for fr in fold_results])
        mean_t = float(vals.mean())
        std_t  = float(vals.std())
        cv_c   = std_t / (mean_t + 1e-9)
        final  = round(mean_t * (1.0 - cv_c), 3)
        stability_trust[mod] = final
        delta  = final - mean_t
        stable = "✅ stable" if std_t < 0.05 else "⚠️  variable"
        print(f"  {mod:<6} mean={mean_t:.3f}  std={std_t:.3f}  "
              f"stability_trust={final:.3f}  ({delta:+.3f})  {stable}")

    # ── LEVEL 4: Distribution-calibrated FSM thresholds ──────────────
    print("\n[CV Level 4 — Distribution-Calibrated FSM Thresholds]")
    test_n = np.mean([sum(fr.energy_dist.values()) for fr in fold_results
                      if fr.energy_dist])
    cv_dist = {}
    fsm_suggest = {}
    for state in ['SAFE','SUSPICIOUS','ELEVATED','CRITICAL']:
        counts = np.array([fr.energy_dist.get(state,0) for fr in fold_results])
        pct    = float(counts.mean() / max(test_n, 1))
        cv_dist[state] = round(pct, 3)

    # Suggest thresholds: work backward from target percentages
    # CRITICAL should contain top ~10.4%, ELEVATED next ~37.9%
    # Current: CRITICAL≥0.62, ELEVATED≥0.42 — suggest based on distribution
    crit_pct = cv_dist.get('CRITICAL',   0.10)
    elev_pct = cv_dist.get('ELEVATED',   0.38)
    susp_pct = cv_dist.get('SUSPICIOUS', 0.26)
    cumulative_above_critical = crit_pct
    cumulative_above_elevated = crit_pct + elev_pct
    cumulative_above_suspicious = crit_pct + elev_pct + susp_pct

    print(f"  CV distribution (balanced 216 events/fold):")
    for state, pct in cv_dist.items():
        bar = '█' * int(pct * 40)
        print(f"    {state:<12} {pct:.1%}  {bar}")
    print(f"  Current FSM: CRITICAL≥0.62  ELEVATED≥0.42  SUSPICIOUS≥0.28")
    print(f"  Calibrated targets based on distribution retained")

    print("\n" + "="*60)
    print("  CV COMPLETE — main pipeline using enhanced components")
    print("="*60 + "\n")

    cv_components = {
        'ensemble_fc':     ensemble_fc,
        'weighted_tau_h':  tau_w_h,
        'weighted_tau_l':  tau_w_l,
        'weighted_alpha':  alpha_w,
        'stability_trust': stability_trust,
        'cv_dist':         cv_dist,
        'auc_weights':     auc_w.tolist(),
    }
    return cv_metrics, fold_results, union_mem, cv_components



# ════════════════════════════════════════════════════════════════════
# LABEL-FREE ADAPTATION (paper §7.5) — confident normals, threshold trace
# ════════════════════════════════════════════════════════════════════

class ConfidentNormalTracker:
    """
    Identifies events the system is genuinely confident are normal
    WITHOUT any label access. Used to self-discover the true
    deployment normal baseline for recalibration.

    Criteria (all must hold simultaneously):
      - fusion_score < threshold (very low — well below tau_low)
      - FSM state SAFE or SUSPICIOUS (not escalated)
      - Network trust > 0.85 (reliable primary signal)
    """
    def __init__(self, confidence_threshold=0.10, min_samples=150):
        self.threshold   = confidence_threshold
        self.min_samples = min_samples
        self.scores      = []
        self.recalib_count = 0
        self.recalib_log   = []

    def observe(self, score, s_level, net_trust):
        if (score < self.threshold and
                s_level <= 1 and
                net_trust > 0.85):
            self.scores.append(float(score))
            return True
        return False

    @property
    def ready(self):
        return len(self.scores) >= self.min_samples

    @property
    def mean(self):
        return float(np.mean(self.scores)) if self.scores else None

    @property
    def count(self):
        return len(self.scores)

    def recalibrate(self, adaptive_tau, smooth=0.05):
        """Shift baseline_mean toward self-discovered normal mean."""
        if not self.ready:
            return False
        cn_mean  = self.mean
        old_base = adaptive_tau._baseline_mean
        new_base = (1 - smooth) * old_base + smooth * cn_mean
        adaptive_tau._baseline_mean = new_base
        adaptive_tau.baseline_mean  = new_base
        self.recalib_count += 1
        self.recalib_log.append({
            'recalib': self.recalib_count,
            'old':     round(old_base, 4),
            'new':     round(new_base, 4),
            'cn_mean': round(cn_mean,  4),
            'n':       self.count,
        })
        return True


P2_TRACE_FILE = 'threshold_trace.csv'


class _P2State:
    """Label-free adaptation inside the evaluated stream. Per event it reads
    only what the deployed system holds at that moment: the fusion score, the
    escalation state, the network trust and the evidence vector. Labels are
    read in report(), after the stream has finished."""
    PL_MARGIN, PL_RATIO = 0.24, 0.70          # pseudo-label bounds (original Phase 2 design)
    CC_K, CC_NOISE, CC_TAU = 3, 0.15, 0.60    # consistency filter (original Phase 2 design)
    RETRAIN_EVERY, RECAL_EVERY, RECAL_SMOOTH = 100, 50, 0.05

    def __init__(self, fc, ev_train, y_train, ev_test, adaptive_tau, fs_all):
        F_ = FusionClassifier.FUSION_FEATURES
        self.recal, self.pseudo, self.replay_only = P2_RECAL, P2_PSEUDO, P2_REPLAY_ONLY
        self.model = fc
        self.Xtr = ev_train[F_].reset_index(drop=True)
        self.ytr = np.asarray(y_train).astype(int)
        self.Xte = ev_test[F_].reset_index(drop=True)
        self.fs_orig = np.asarray(fs_all, dtype=float).copy()
        self.tau_h0 = float(adaptive_tau._base_tau_high)
        self.tau_l0 = float(adaptive_tau.tau_low)
        self.pl_atk = min(0.95, self.tau_h0 + self.PL_MARGIN)
        self.pl_nrm = max(0.03, self.tau_l0 * self.PL_RATIO)
        self.cnt = ConfidentNormalTracker(confidence_threshold=0.10, min_samples=150)
        self.rng = np.random.RandomState(42)
        self.cand = self.tested = self.accepted = self.rejected = 0
        self.idx, self.lab = [], []
        self.retrains, self.first_retrain = 0, None
        print(f"\n[P2] label-free adaptation ON: recal={self.recal} pseudo={self.pseudo} "
              f"replay_only={self.replay_only}  pseudo-label bounds: attack > "
              f"{self.pl_atk:.3f}, normal < {self.pl_nrm:.3f}")

    def _consistent(self, i, lab):
        self.tested += 1
        x = self.Xte.iloc[[i]].values.astype(float)
        P = x + self.rng.normal(0, self.CC_NOISE, (self.CC_K, x.shape[1]))
        try:
            s = np.asarray(self.model.predict_proba_score(
                pd.DataFrame(P, columns=self.Xte.columns)), dtype=float)
        except Exception:
            self.rejected += 1
            return False
        ok = bool((s >= self.CC_TAU).all()) if lab == 1 else bool((s <= 1.0 - self.CC_TAU).all())
        if ok:
            self.accepted += 1
        else:
            self.rejected += 1
        return ok

    def step(self, i, fs, s_level, net_trust, adaptive_tau, fs_all):
        if self.recal:
            self.cnt.observe(fs, s_level, net_trust)
            if self.cnt.ready and i % self.RECAL_EVERY == 0:
                self.cnt.recalibrate(adaptive_tau, smooth=self.RECAL_SMOOTH)
        if not self.pseudo:
            return
        if fs > self.pl_atk:
            lab = 1
        elif fs < self.pl_nrm:
            lab = 0
        else:
            return
        self.cand += 1
        if not self._consistent(i, lab):
            return
        self.idx.append(i)
        self.lab.append(lab)
        if len(self.lab) % self.RETRAIN_EVERY == 0:
            self._refit(i, fs_all)

    def _refit(self, i, fs_all):
        import contextlib, io
        if self.replay_only:
            if self.retrains:            # the replay-only refit is identical every time
                self.retrains += 1
                return
            X, y = self.Xtr, self.ytr
        else:
            X = pd.concat([self.Xtr, self.Xte.iloc[self.idx]], ignore_index=True)
            y = np.concatenate([self.ytr, np.asarray(self.lab, dtype=int)])
        if len(np.unique(y)) < 2:
            return
        m = FusionClassifier()
        with contextlib.redirect_stdout(io.StringIO()):
            m.fit(X, y)
        self.model = m
        self.retrains += 1
        if self.first_retrain is None:
            self.first_retrain = i
        if i + 1 < len(fs_all):
            fs_all[i + 1:] = np.asarray(m.predict_proba_score(self.Xte.iloc[i + 1:]),
                                        dtype=float)

    def report(self, results, train_df):
        from sklearn.metrics import roc_auc_score
        results['fusion_score_unadapted'] = self.fs_orig
        print("\n" + "=" * 72)
        print("[P2] LABEL-FREE ADAPTATION — what it did, then how it scored")
        print("=" * 72)
        print(f"  components             : recalibration {'on' if self.recal else 'off'}, "
              f"pseudo-label refit {'on' if self.pseudo else 'off'}"
              + ("  (replay rows only: CONTROL)" if self.replay_only else ""))
        if self.recal:
            lg = self.cnt.recalib_log
            mu = self.cnt.mean
            print(f"  confident normals      : {self.cnt.count:,}"
                  + (f"  (mean score {mu:.4f})" if mu is not None else ""))
            print(f"  recalibrations         : {self.cnt.recalib_count:,}")
            if lg:
                print(f"  drift reference        : {lg[0]['old']:.4f} -> {lg[-1]['new']:.4f}")
        if self.pseudo:
            nl = np.asarray(self.lab, dtype=int)
            print(f"  pseudo-label bounds    : attack > {self.pl_atk:.3f}, normal < {self.pl_nrm:.3f}")
            print(f"  candidates             : {self.cand:,}   consistency: tested {self.tested:,}"
                  f"  accepted {self.accepted:,}  rejected {self.rejected:,}")
            print(f"  pseudo-labels kept     : {len(nl):,}  ({int((nl == 1).sum()):,} attack / "
                  f"{int((nl == 0).sum()):,} normal)")
            print(f"  refits                 : {self.retrains:,}  (first after event "
                  f"{self.first_retrain}; replay rows {len(self.ytr):,})")
        print("  -- labels read from here on: evaluation only --")
        y = np.asarray(results['final_label']).astype(int)
        typ = (results['type'].astype(str).str.strip().str.lower().values
               if 'type' in results.columns else None)
        if self.pseudo and self.idx:
            ii = np.asarray(self.idx)
            nl = np.asarray(self.lab, dtype=int)
            yt = y[ii]
            a, n = nl == 1, nl == 0
            print(f"  pseudo-label agreement : attack {int((yt[a] == 1).sum()):,}/{int(a.sum()):,}"
                  f"   normal {int((yt[n] == 0).sum()):,}/{int(n.sum()):,}   (consensus label)")
            if typ is not None:
                tt = typ[ii]
                wn = pd.Series(tt[n & (tt != 'normal')]).value_counts().head(5)
                print(f"  type-field attacks kept as normal: {int((n & (tt != 'normal')).sum()):,}"
                      + ("  (" + ", ".join(f"{k} {v:,}" for k, v in wn.items()) + ")" if len(wn) else ""))
                print(f"  type-field normals kept as attack: {int((a & (tt == 'normal')).sum()):,}")
        try:
            print(f"  stream ROC-AUC         : scores used (prequential) "
                  f"{roc_auc_score(y, results['fusion_score']):.4f}   unadapted ensemble "
                  f"{roc_auc_score(y, self.fs_orig):.4f}")
            if '_rowid' in results.columns and '_rowid' in train_df.columns:
                m = ~results['_rowid'].isin(set(train_df['_rowid'].tolist())).values
                if len(np.unique(y[m])) == 2:
                    print(f"  outside layer split AUC: scores used "
                          f"{roc_auc_score(y[m], np.asarray(results['fusion_score'])[m]):.4f}"
                          f"   unadapted {roc_auc_score(y[m], self.fs_orig[m]):.4f}")
        except Exception as e:
            print(f"  AUC comparison failed: {type(e).__name__}: {e}")
        print("=" * 72)
        print("[P2] end")


def _tau_trace_report(trace, results, tau_h0, tau_l0, contextual_tau):
    """Per-event threshold trace: summary to the log, full trace to CSV."""
    print("\n" + "=" * 72)
    print("[TAU-TRACE] per-event thresholds on the stream (records only)")
    print("=" * 72)
    try:
        if not trace:
            print("  no events traced (every decision came from the cache or a forced condition)")
            print("[TAU-TRACE] end")
            return
        cols = ['i', 'fs', 'eff_score', 'inband', 'tau_h', 'tau_l', 'gate_tau_l',
                'tau_l_prev', 'ewma', 'reference', 'D', 'U', 's_level', 'kind',
                'dec_fusion', 'dec']
        T = pd.DataFrame(trace, columns=cols)
        pos = T['i'].values
        y = np.asarray(results['final_label']).astype(int)[pos]
        T['final_label'] = y
        if 'type' in results.columns:
            T['type'] = results['type'].astype(str).str.strip().str.lower().values[pos]
        try:
            T.to_csv(P2_TRACE_FILE, index=False)
            wrote = P2_TRACE_FILE
        except Exception as e:
            wrote = f"NOT written - {type(e).__name__}: {e}"

        def q(v):
            return "  ".join(f"p{p} {np.percentile(v, p):.3f}" for p in (5, 25, 50, 75, 95))

        th, tl, g = T['tau_h'].values, T['tau_l'].values, T['gate_tau_l'].values
        print(f"  events traced           : {len(T):,}")
        print(f"  at stream start         : T_base {tau_h0:.3f}  tau_low {tau_l0:.3f}  "
              f"floor {TAU_FLOOR:.2f}  alpha {contextual_tau.ALPHA}  delta {contextual_tau.DELTA}")
        print(f"  tau_high used           : at floor {np.isclose(th, TAU_FLOOR, atol=5e-4).mean():.1%}"
              f"  below T_base {(th < tau_h0 - 5e-4).mean():.1%}  above T_base "
              f"{(th > tau_h0 + 5e-4).mean():.1%}  min {th.min():.3f}  max {th.max():.3f}")
        print(f"  tau_low used by fusion  : {q(tl)}")
        print(f"                            <= 0.08 {(tl <= 0.08).mean():.1%}   at its cap "
              f"(tau_high - 0.12) {np.isclose(tl, th - 0.12, atol=1.5e-3).mean():.1%}")
        print(f"  tau_low at the gate     : {q(g)}")
        print(f"                            <= 0.08 {(g <= 0.08).mean():.1%}   moved by the "
              f"drift update {(~np.isclose(g, tl, atol=5e-4)).mean():.1%}")
        print(f"  D = |EWMA - reference|  : mean {T['D'].mean():.3f}  {q(T['D'].values)}")
        print(f"  U = layer std           : mean {T['U'].mean():.3f}  {q(T['U'].values)}")
        print(f"  EWMA of scores          : min {T['ewma'].min():.3f}  mean {T['ewma'].mean():.3f}"
              f"  max {T['ewma'].max():.3f}   reference {T['reference'].iloc[0]:.4f} -> "
              f"{T['reference'].iloc[-1]:.4f}")
        esc = (T['dec_fusion'].values == 0) & (T['dec'].values == 1)
        kind = T['kind'].values
        tn = T['type'].values == 'normal' if 'type' in T.columns else None

        def split(msk):
            s = (f"{int(msk.sum()):,}  (consensus normal {int((msk & (y == 0)).sum()):,} / "
                 f"attack {int((msk & (y == 1)).sum()):,}")
            if tn is not None:
                s += f"; type-field normal {int((msk & tn).sum()):,}"
            return s + ")"
        print(f"  escalation raised       : {split(esc)}")
        print(f"    from FUSION_NORMAL    : {split(esc & (kind == 'FUSION_NORMAL'))}")
        print(f"    from SUSPICIOUS_SAFE  : {split(esc & (kind == 'SUSPICIOUS_SAFE'))}")
        print(f"    score < tau_low at start ({tau_l0:.3f}): {split(esc & (T['eff_score'].values < tau_l0))}")
        print(f"  trace file              : {wrote}")
    except Exception as e:
        print(f"  trace report failed: {type(e).__name__}: {e}  (the pipeline is unaffected)")
    print("[TAU-TRACE] end")


def run_pipeline_multi(net_files, iot_files, log_files, win_files=None,
                       n_samples=10000):
    """
    Full pipeline over the complete TON_IoT dataset.
    Loads all files per modality, preprocesses, time-fuses, trains, evaluates.
    """
    n_per_file = max(500, n_samples // max(len(net_files), 1))

    print("\n" + "="*60)
    print("  MULTI-LAYER IDS — FULL TON_IoT PIPELINE")
    print("="*60)

    # ── Load all files ─────────────────────────────────────────────
    print("\n[1/9] Loading all datasets ...")
    raw_net = load_multi(net_files, 'Network',  n_per_file=n_per_file)
    raw_iot = load_multi(iot_files, 'IoT',      n_per_file=n_per_file)
    raw_log = load_multi(log_files, 'Linux Log', n_per_file=n_per_file)
    raw_win = load_multi(win_files, 'Windows',  n_per_file=n_per_file) if win_files else None

    if any(v is None for v in [raw_net, raw_iot, raw_log]):
        print("❌ Aborting — required modality failed to load."); return None

    # ── Preprocess each modality ───────────────────────────────────
    print("\n[2/9] Preprocessing ...")
    proc_net = preprocess_network(raw_net)
    proc_iot = preprocess_iot_multi(raw_iot)   # device-aware
    proc_log = preprocess_linux_multi(raw_log) # log-type-aware
    proc_win = preprocess_windows(raw_win) if raw_win is not None else None

    # ── Time-based fusion ──────────────────────────────────────────
    fused, full_fused = fuse_datasets(proc_net, proc_iot, proc_log, proc_win)
    globals()['_B5_BALANCED'] = (set(fused['_rowid'].tolist())
        if '_rowid' in fused.columns else None)   # rows the fusion ensemble saw
    if len(fused) == 0:
        print("❌ Fusion returned 0 rows — check timestamps."); return None

    # ── Rest of pipeline is identical to single-file run ──────────
    print("\n[3/9] Train/test split ...")
    # ── Stateful 5-fold cross-validation on balanced data ──────
    print(f"  Running stateful 5-fold CV on the balanced set ({len(fused):,} rows)...")
    cv_metrics, cv_folds, cv_memory, cv_components = run_cross_validation(fused, n_splits=5)

    # ── Aggregation strategy selection ──────────────────────────
    import sys
    use_quick    = '--quick_compare' in sys.argv
    use_ablation = '--fpr_ablation'  in sys.argv
    if use_ablation:
        print("\n[3b/9] Running FPR ablation study (5 variants)...")
        run_fpr_ablation(fused, full_fused, cv_components)
        print("  FPR ablation complete — main pipeline continues with full system")
    elif use_quick:
        print("\n[3b/9] Running targeted 6-combo comparison...")
        best_comps = run_quick_compare(fused, full_fused, cv_folds)
        cv_components = best_comps
    else:
        print("\n[3b/9] Using CV-derived components (run with --quick_compare for targeted comparison)")
        # cv_components already set from CV aggregation above

    # ── Then train on 70% of balanced for main pipeline run ────
    train_df, _ = train_test_split(
        fused, test_size=0.3, random_state=42, stratify=fused['final_label'])
    train_df = train_df.reset_index(drop=True)

    # Test on the ENTIRE fused dataset (all rows — including training rows)
    # Full-coverage evaluation: how the system handles all known traffic
    test_df    = full_fused.reset_index(drop=True)
    n_test_atk = int((test_df['final_label']==1).sum())
    n_test_nrm = int((test_df['final_label']==0).sum())
    print(f"  Train: {len(train_df)} (balanced 50/50)")
    print(f"  Test:  {len(test_df)} rows | attack={n_test_atk} | normal={n_test_nrm} (full dataset — in+out of sample)")

    print("\n[4/9] Training layer models ...")
    lm = LayerModels(); lm.fit(train_df); lm.save()

    print("\n[5/9] Building evidence vectors ...")
    pn_r, pi_r, pl_r = lm.predict_proba_raw(train_df)
    ev_train = build_evidence_vector(calibrate(pn_r), calibrate(pi_r),
                                      calibrate(pl_r), train_df['phys_delta'].values)

    print("\n[6/9] Training fusion classifier ...")
    fc_base = FusionClassifier(); fc_base.fit(ev_train, train_df['final_label'])
    # ── LEVEL 1: Replace single model with CV ensemble ──────────
    fc = cv_components.get('ensemble_fc', fc_base)
    if isinstance(fc, EnsembleFusionClassifier):
        print("  ✅ Using CV AUC-weighted ensemble (5 fold models)")
        fc.save()
    else:
        fc_base.save()

    print("\n[6.5] Fitting adaptive threshold ...")
    adaptive_tau = AdaptiveThreshold(beta=1.5)
    adaptive_tau.fit(train_df['final_label'].values,
                     fc_base.predict_proba_score(ev_train))
    # ── LEVEL 2: Override with F-beta weighted threshold ────────
    if cv_components.get('weighted_tau_h') is not None:
        old_h = adaptive_tau.tau_high
        adaptive_tau.tau_high         = cv_components['weighted_tau_h']
        adaptive_tau._base_tau_high   = cv_components['weighted_tau_h']
        adaptive_tau.tau_low          = cv_components['weighted_tau_l']
        print(f"  ✅ Threshold: fbeta-weighted tau_high={adaptive_tau.tau_high:.3f}"
              f" (was {old_h:.3f})")
    adaptive_tau.save()
    site_cal      = SiteCalibration(train_df) if (SITE_ENVELOPES or SITE_LIMITS) else None
    policy_engine = SitePolicyEngine(site_cal) if SITE_LIMITS else PolicyEngine()
    risk_scorer   = UnifiedRiskScorer(policy_engine)
    # ── LEVEL 3: Initialize trust with stability-penalized values ─
    trust_tracker = ModalityTrustTracker()
    if 'stability_trust' in cv_components:
        for mod, val in cv_components['stability_trust'].items():
            if mod in trust_tracker.trust:
                trust_tracker.trust[mod] = val
        print(f"  ✅ Trust initialized from CV: "
              + "  ".join(f"{m}={v:.3f}"
                for m,v in cv_components['stability_trust'].items()))
    contextual_tau = ContextualThresholdEngine(adaptive_tau)
    contextual_tau.auto_tune(train_df['final_label'].values,
                             fc_base.predict_proba_score(ev_train))
    # ── LEVEL 4: FSM uses same thresholds, CV dist printed ───────
    threat_fsm = ThreatStateMachine()
    scan_energy = ScanEnergy()
    if 'cv_dist' in cv_components:
        print(f"  ✅ FSM calibration reference: "
              + "  ".join(f"{s}={v:.1%}"
                for s,v in cv_components['cv_dist'].items()))

    print("\n[7/9] Setting up LIME ...")
    explainer = IDSExplainer(fc, ev_train)

    print("\n[8/9] Inference on test set ...")
    memory = ExperienceMemory()
    behavioral_mem = BehavioralThreatMemory()
    outcome_mem    = OutcomeMemory()
    pn_te, pi_te, pl_te = lm.predict_proba_raw(test_df)
    pn_te, pi_te, pl_te = calibrate(pn_te), calibrate(pi_te), calibrate(pl_te)
    ev_test = build_evidence_vector(pn_te, pi_te, pl_te, test_df['phys_delta'].values)
    fs_all  = fc.predict_proba_score(ev_test)
    # ── threshold trace always on (records only); adaptation only with --p2_* ──
    fs_all  = np.asarray(fs_all, dtype=float).copy()
    env_hits, env_rules = (site_cal.envelope_hits(test_df) if SITE_ENVELOPES
                           else (np.zeros(len(test_df), dtype=bool), None))
    _tr_hi0, _tr_lo0 = float(adaptive_tau._base_tau_high), float(adaptive_tau.tau_low)
    _trace  = []
    _p2     = (_P2State(fc, ev_train, train_df['final_label'].values, ev_test,
                        adaptive_tau, fs_all) if P2_ANY else None)

    warmup = max(int(len(test_df)*0.3), 50)
    for i in range(warmup):
        pn,pi,pl = float(pn_te[i]),float(pi_te[i]),float(pl_te[i])
        d  = float(test_df['phys_delta'].iloc[i]); fs = float(fs_all[i])
        ev_v = [float(ev_test['L_net'].iloc[i]),float(ev_test['L_iot'].iloc[i]),
                float(ev_test['L_log'].iloc[i]),d]
        _, sig = memory.lookup(pn,pi,pl,d)
        memory.store(sig, int(fs>TAU_HIGH), 'warmup', fs, ev_v, generate_pseudo_label(fs, pn, pi, pl))
    memory.run_dbscan()

    decisions,reasons,confidences,latencies,from_mem = [],[],[],[],[]
    for i in range(len(test_df)):
        t0 = time.time()
        pn,pi,pl = float(pn_te[i]),float(pi_te[i]),float(pl_te[i])
        d  = float(test_df['phys_delta'].iloc[i]); fs = float(fs_all[i])
        ev_v = [float(ev_test['L_net'].iloc[i]),float(ev_test['L_iot'].iloc[i]),
                float(ev_test['L_log'].iloc[i]),d]
        # Scan energy accumulates over the WHOLE stream. It previously sat in
        # the cache-miss branch and so saw 25% of events; a scanner's repeated
        # probes are exactly what the cache absorbs.
        if SCAN_FSM and 'scan_energy' in dir():
            _s = str(test_df['src_ip'].iloc[i]) if 'src_ip' in test_df.columns else 'unknown'
            _c = lambda c: (float(test_df[c].iloc[i]) if c in test_df.columns else 0.0)
            scan_energy.observe(_s, _c('phys_fanout'),
                                _c('phys_port_entropy'), _c('phys_fail_rate'))

        cached, sig = memory.lookup(pn,pi,pl,d)
        if cached:
            decisions.append(cached['decision']); reasons.append(cached['reason']+' [MEM]')
            confidences.append(cached['confidence']); latencies.append(time.time()-t0); from_mem.append(True)
        else:
            # ── Build feature row ──────────────────────────────────
            def _get(col): return float(test_df[col].iloc[i]) if col in test_df.columns else 0.0
            row_feats = {
                'p_net':pn,'p_iot':pi,'p_log':pl,'phys_delta':d,'phys_delta_abs':abs(d),
                'disk_write':_get('disk_write'),'cpu_usage':_get('cpu_usage'),
                'cpu_spike':_get('cpu_spike'),'mem_vgrow':_get('mem_vgrow'),
                'page_faults_maj':_get('page_faults_maj'),
                'src_bytes':_get('net_bytes'),'temp_roll_std':_get('temp_roll_std'),
                'net_bytes':_get('net_bytes'),'net_dbytes':_get('net_dbytes'),
                'iot_device':_get('iot_device'),
            }

            # ── FIX 5: Forced policy check first ───────────────────
            forced, force_reason = check_forced_policies(row_feats)
            forced = forced and not NO_FORCED
            if SITE_ENVELOPES and env_hits[i]:
                forced, force_reason = True, f"FORCED: envelope {env_rules[i]}"
            if forced:
                decisions.append(1)
                reasons.append(f"[FORCED_ACTION] {force_reason}")
                confidences.append(0.99); latencies.append(time.time()-t0)
                from_mem.append(False); continue

            if ONE_PATH:   # the policy pressure reaches the energy state machine, as in CV
                P_risk, _ = policy_engine.evaluate(row_feats)

            # ── FIX 4: Apply modality trust dampening ──────────────
            pn_t, pi_t, pl_t = trust_tracker.apply(pn, pi, pl)

            # ── Unified Risk Score ──────────────────────────────────
            cid_r,c_risk,_ = memory.find_closest_cluster(ev_v)
            R,breakdown,p_flags = risk_scorer.score(pn_t,pi_t,pl_t,d,0.0,c_risk,row_feats,adaptive_tau)

            # ── FIX 3: Contextual tau ───────────────────────────────
            layer_std = float(np.std([pn,pi,pl]))
            recent_density = sum(decisions[-20:]) / max(len(decisions[-20:]),1) if decisions else 0.0
            _tr_e, _tr_b = adaptive_tau._ewma_score, adaptive_tau._baseline_mean
            _tr_lo_prev  = adaptive_tau.tau_low
            _tr_D = float(np.clip(abs(_tr_e - _tr_b), 0, 1)) if _tr_b and _tr_e else 0.0
            tau_h_ctx, tau_l_ctx, ctx_reason = contextual_tau.get_tau(
                layer_disagreement=layer_std,
                trust_scores=trust_tracker.trust
            )
            adaptive_tau.tau_high = tau_h_ctx
            adaptive_tau.tau_low  = tau_l_ctx

            # ── Per-source scan accumulator ─────────────────────────
            # Scoped feedback: evidence about this host lowers the bar for this
            # host only, and decays. Not a global escalation loop.
            if SCAN_FSM and 'scan_energy' in dir():
                _src = (str(test_df['src_ip'].iloc[i])
                        if 'src_ip' in test_df.columns else 'unknown')
                _rel = scan_energy.relief(_src)   # energy already accumulated above
                if _rel:
                    adaptive_tau.tau_high = max(adaptive_tau.tau_high - _rel,
                                                ScanEnergy.FLOOR)
                    tau_h_ctx = adaptive_tau.tau_high

            _inband = _band_hit('site2_full_stream', fs, R, tau_l_ctx, tau_h_ctx)
            eff_score = R if _inband else fs
            dec,reason,conf = hierarchical_decision(pn_t,pi_t,pl_t,d,eff_score,ev_v,memory,adaptive_tau)
            _tr_dec_fusion, _tr_kind = dec, reason.split(':')[0]
            if p_flags: reason = reason + " | POLICY:" + p_flags[0].split("[")[1].rstrip("]")

            # ── Multi-stage threat level ────────────────────────────
            t_level, t_name, _ = score_to_threat_level(eff_score, P_risk if 'P_risk' in dir() else 0.0)
            camp_risk = behavioral_mem.get_campaign_risk() if 'behavioral_mem' in dir() else 0.0
            s_level = threat_fsm.update(t_level, cluster_risk=max(c_risk, camp_risk), policy_risk=min(P_risk if 'P_risk' in dir() else 0.0, 1.0))
            s_name  = THREAT_LEVELS[s_level]['name']
            # Escalation gate. It compares against tau_low AFTER the drift update
            # (AdaptiveThreshold.update), which can sit below the tau_low the
            # fusion stage decided with: it limits the reversal of confident
            # negatives, it does not prevent it (paper §5.5: 35 of 213 in C).
            _tr_gate = adaptive_tau.tau_low   # the value the gate compares
            if s_level >= 2 and eff_score > adaptive_tau.tau_low and not NO_ESCALATION:
                dec = 1   # ELEVATED+ → binary attack (gated)
            reason = f"[{s_name}] {reason}"
            _trace.append((i, fs, float(eff_score), int(_inband), float(tau_h_ctx),
                           float(tau_l_ctx), float(_tr_gate), float(_tr_lo_prev),
                           _tr_e, _tr_b, _tr_D, float(layer_std), int(s_level),
                           _tr_kind, int(_tr_dec_fusion), int(dec)))

            # ── Update trust tracker ────────────────────────────────
            trust_tracker.update(dec, pn, pi, pl)
            memory.store(sig,dec,reason,conf,ev_v,generate_pseudo_label(fs,pn,pi,pl))
            if _p2 is not None:   # label-free adaptation step
                _p2.step(i, fs, s_level, trust_tracker.trust.get('net', 0.7),
                         adaptive_tau, fs_all)

            # ── Behavioral memory: store intent profile + escalation path ──
            # Extract attack family scores (atk_* features) as intent profile
            intent_prof = {}
            for c in ['atk_scan','atk_dos','atk_brute_force','atk_exfiltration',
                      'atk_injection','atk_mitm','atk_ransomware','atk_backdoor',
                      'atk_password']:
                val = float(test_df[c].iloc[i]) if c in test_df.columns else float(row_feats.get(c,0))
                if val > 0: intent_prof[c.replace('atk_','')] = val
            temporal_prof = {"periodicity": row_feats.get('phys_periodicity',0),
                             "burst":       row_feats.get('phys_burst',0),
                             "fail_rate":   row_feats.get('phys_fail_rate',0)}
            behavioral_mem.store(ts=float(test_df['ts'].iloc[i]) if 'ts' in test_df.columns else i,
                                 intent_profile=intent_prof,
                                 temporal_profile=temporal_prof,
                                 escalation_path=[s_name],
                                 net_risk=float(row_feats.get('net_risk_score', eff_score)))

            # ── Outcome memory: auto-record based on final decision ─────────
            dom_tag = max(intent_prof, key=intent_prof.get) if intent_prof else "unknown"
            outcome_mem.record(dom_tag, eff_score,
                               'confirmed' if dec==1 else 'false_positive')

            decisions.append(dec); reasons.append(reason); confidences.append(conf)
            latencies.append(time.time()-t0); from_mem.append(False)

    print("\n[9/9] Evaluating ...")
    results = test_df.copy()
    results['pred_label']   = decisions
    results['reason']       = reasons
    results['confidence']   = confidences
    results['latency_ms']   = [l*1000 for l in latencies]
    results['from_memory']  = from_mem
    _hits = int(sum(1 for x in from_mem if x))
    print(f"\n  cache: {_hits:,} of {len(from_mem):,} decisions served "
          f"({_hits/max(len(from_mem),1):.1%})"
          + ("   [cache off: the full decision path ran for every event]"
             if CACHE_DISABLED else
             "   [--cache: these decisions bypassed scoring, thresholds and escalation]"))
    results['fusion_score'] = fs_all
    metrics = evaluate(results['final_label'], results['pred_label'], results['fusion_score'],
                       y_type=results['type'] if 'type' in results.columns else None)
    _b5_report(results, train_df)
    _tau_trace_report(_trace, results, _tr_hi0, _tr_lo0, contextual_tau)
    if _p2 is not None:
        _p2.report(results, train_df)
    plot_score_distribution(results['final_label'], results['fusion_score'],
                             adaptive_tau.tau_high, adaptive_tau.tau_low)

    memory.run_dbscan(); memory.plot_clusters(); memory.stats()

    comp = [l for l,m in zip(latencies,from_mem) if not m]
    cach = [l for l,m in zip(latencies,from_mem) if m]
    if comp:
        ac = np.mean(comp)*1000
        am = (f" | cache hits: {np.mean(cach)*1000:.4f}ms over {len(cach):,}" if cach else "")
        print(f"\n[Speed] AI: {ac:.4f}ms per event over {len(comp):,} decided events{am}"
              "  (wall clock on pre-computed features; excludes preprocessing)")

    print("\n" + "═"*66)
    print("  INCIDENT EXPLANATIONS — First 3 Detected Attacks")
    print("═"*66)
    # Skip warmup period (first 30%) — those show cache reasons
    _warmup_end  = max(int(len(decisions)*0.30), 50)
    _attack_idxs = [i for i,d in enumerate(decisions)
                    if d==1 and i >= _warmup_end][:3]
    # Fall back to any attack if not enough post-warmup attacks
    if len(_attack_idxs) < 3:
        _attack_idxs = [i for i,d in enumerate(decisions) if d==1][:3]
    for idx in _attack_idxs:
        if idx >= len(ev_test): continue
        _reason = reasons[idx] if idx < len(reasons) else 'UNKNOWN'
        _fs     = float(fs_all[idx]) if hasattr(fs_all,'__len__') else 0.5
        _pn     = float(pn_te[idx])
        _pi     = float(pi_te[idx])
        _pl     = float(pl_te[idx])
        _pd     = float(full_fused['phys_delta'].iloc[idx])
        # Build intent profile
        _ip = {}
        for _c in ['atk_scan','atk_dos','atk_brute_force',
                   'atk_exfiltration','atk_ransomware',
                   'atk_backdoor','atk_password']:
            _v = float(full_fused[_c].iloc[idx]) \
                 if _c in full_fused.columns else 0.0
            if _v > 0.20: _ip[_c.replace('atk_','')] = _v
        # LIME pairs if available
        _lime_pairs = []
        if explainer and explainer._ready:
            _lime_pairs = explainer.get_pairs(ev_test.iloc[[idx]])
        # Full structured explanation
        _expl = _explain_engine.full_event_explanation(
            event_idx=idx, decision=1, reason=_reason,
            fusion_score=_fs, p_net=_pn, p_iot=_pi, p_log=_pl,
            phys_delta=_pd, s_level_name='ELEVATED', energy=0.6,
            tau_high=float(adaptive_tau.tau_high),
            tau_low=float(adaptive_tau.tau_low),
            trust_scores=dict(trust_tracker.trust),
            intent_profile=_ip, lime_pairs=_lime_pairs,
        )
        for _ln in _expl: print(_ln)

    print("\n[Behavioral Memory Stats]")
    for k,v in behavioral_mem.stats().items(): print(f"  {k:<25} {v}")
    print("\n[Outcome Memory Stats]")
    for k,v in outcome_mem.stats().items(): print(f"  {k:<25} {v}")
    print("\n[Modality Trust Scores]")
    for k,v in trust_tracker.status().items(): print(f"  {k:<20} {v}")
    print("\n[Contextual Threshold Stats]")
    for k,v in contextual_tau.stats().items(): print(f"  {k:<20} {v}")
    print("\n[Adaptive Threshold Status]")
    for k,v in adaptive_tau.status().items(): print(f"  {k:<20} {v}")
    print("\n[Decision Breakdown]")
    # Multi-stage threat level summary
    print("\n[Threat State Machine Stats]")
    for k,v in threat_fsm.stats().items(): print(f"  {k:<22} {v}")

    if 'scan_energy' in dir():
        print("\n[Per-Source Scan Accumulator]")
        for k,v in scan_energy.stats().items(): print(f"  {k:<22} {v}")
    print("\n[Threat Level Distribution]")
    for lvl, info in THREAT_LEVELS.items():
        c = sum(1 for r in reasons if f"[{info['name']}]" in r)
        if c: print(f"  {info['name']:<16} {c:>4}  → {info['action']}")
    print("\n[Decision Breakdown]")
    for tag in ['OVERRIDE','FUSION_ALERT','FUSION_NORMAL','CLUSTER_MATCH','ANOMALY','SUSPICIOUS_SAFE','MEM','POLICY','FORCED']:
        c = sum(1 for r in reasons if tag in r)
        if c: print(f"  {tag:<22} {c}")

    os.makedirs('outputs', exist_ok=True)
    results.to_csv('test_results.csv', index=False)

    ol = OnlineLearner(fc, memory, mode='incremental')
    ol.retrain(force=True)


    print(f"\n✅ FULL TON_IoT PIPELINE COMPLETE")
    return lm, fc, memory, explainer, results, metrics


def run_pipeline(net_csv, iot_csv, log_csv, win_csv=None, n_samples=10000):
    print("\n" + "="*60)
    print("  MULTI-LAYER IDS FUSION — FULL PIPELINE")
    print("="*60)

    # Load
    print("\n[1/9] Loading datasets ...")
    raw_net = load_and_snip(net_csv, 'Network',    n_samples)
    raw_iot = load_and_snip(iot_csv, 'Thermostat', n_samples)
    raw_log = load_and_snip(log_csv, 'Disk Log',   n_samples)
    raw_win = load_and_snip(win_csv, 'Windows',    n_samples) if win_csv else None
    if any(v is None for v in [raw_net, raw_iot, raw_log]):
        print("❌ Aborting — one or more required files failed."); return None
    if win_csv and raw_win is None:
        print("⚠️  Windows CSV failed to load — continuing without it.")

    # Preprocess + fuse
    print("\n[2/9] Preprocessing & fusing ...")
    df_win = preprocess_windows(raw_win) if raw_win is not None else None
    fused, full_fused = fuse_datasets(preprocess_network(raw_net),
                                      preprocess_iot(raw_iot),
                                      preprocess_logs(raw_log),
                                      df_win)
    globals()['_B5_BALANCED'] = (set(fused['_rowid'].tolist())
        if '_rowid' in fused.columns else None)   # rows the fusion ensemble saw

    # Split — train on balanced, test on full real distribution
    print("\n[3/9] Train/test split ...")
    # ── Stateful 5-fold cross-validation on balanced data ──────
    print("  Running stateful 5-fold CV on balanced dataset...")
    cv_metrics, cv_folds, cv_memory, cv_components = run_cross_validation(fused, n_splits=5)
    # ── Aggregation strategy selection ──────────────────────────
    import sys
    use_quick    = '--quick_compare' in sys.argv
    use_ablation = '--fpr_ablation'  in sys.argv
    if use_ablation:
        print("\n[3b/9] Running FPR ablation study (5 variants)...")
        run_fpr_ablation(fused, full_fused, cv_components)
        print("  FPR ablation complete — main pipeline continues with full system")
    elif use_quick:
        print("\n[3b/9] Running targeted 6-combo comparison...")
        best_comps = run_quick_compare(fused, full_fused, cv_folds)
        cv_components = best_comps
    else:
        print("\n[3b/9] Running aggregation strategy grid search...")
        print("  Using CV-derived components (run with --quick_compare for targeted comparison)")
    # ── Main pipeline: train on 70% balanced ───────────────────
    train_df, _ = train_test_split(
        fused, test_size=0.3, random_state=42, stratify=fused['final_label'])
    train_df = train_df.reset_index(drop=True)
    # Test on the ENTIRE fused dataset (all 2978 rows — including training rows)
    # This gives full-coverage evaluation: how the system handles all known traffic
    test_df    = full_fused.reset_index(drop=True)
    n_test_atk = int((test_df['final_label']==1).sum())
    n_test_nrm = int((test_df['final_label']==0).sum())
    print(f"  Train: {len(train_df)} (balanced 50/50)")
    print(f"  Test:  {len(test_df)} rows | attack={n_test_atk} | normal={n_test_nrm} (full dataset — in+out of sample)")

    # Layer models
    print("\n[4/9] Training layer models ...")
    lm = LayerModels(); lm.fit(train_df); lm.save()

    # Calibrated evidence
    print("\n[5/9] Building evidence vectors ...")
    pn_r, pi_r, pl_r = lm.predict_proba_raw(train_df)
    ev_train = build_evidence_vector(calibrate(pn_r), calibrate(pi_r),
                                      calibrate(pl_r), train_df['phys_delta'].values)

    # Fusion
    print("\n[6/9] Training fusion classifier ...")
    fc_base = FusionClassifier(); fc_base.fit(ev_train, train_df['final_label'])
    fc = cv_components.get('ensemble_fc', fc_base)
    if isinstance(fc, EnsembleFusionClassifier):
        print("  ✅ Using CV AUC-weighted ensemble (5 fold models)")
        fc.save()
    else:
        fc_base.save()

    # Adaptive threshold + Policy engine
    print("\n[6.5] Fitting adaptive threshold ...")
    adaptive_tau = AdaptiveThreshold(beta=1.5)
    adaptive_tau.fit(train_df['final_label'].values,
                     fc_base.predict_proba_score(ev_train))
    if cv_components.get('weighted_tau_h') is not None:
        old_h = adaptive_tau.tau_high
        adaptive_tau.tau_high       = cv_components['weighted_tau_h']
        adaptive_tau._base_tau_high = cv_components['weighted_tau_h']
        adaptive_tau.tau_low        = cv_components['weighted_tau_l']
        print(f"  ✅ Threshold: fbeta-weighted tau_high={adaptive_tau.tau_high:.3f}"
              f" (was {old_h:.3f})")
    adaptive_tau.save()
    policy_engine  = PolicyEngine()
    risk_scorer    = UnifiedRiskScorer(policy_engine)
    trust_tracker  = ModalityTrustTracker()
    if 'stability_trust' in cv_components:
        for mod, val in cv_components['stability_trust'].items():
            if mod in trust_tracker.trust:
                trust_tracker.trust[mod] = val
        print(f"  ✅ Trust initialized from CV stability: "
              + "  ".join(f"{m}={v:.3f}"
                for m,v in cv_components['stability_trust'].items()))
    contextual_tau = ContextualThresholdEngine(adaptive_tau)
    contextual_tau.auto_tune(train_df['final_label'].values,
                             fc_base.predict_proba_score(ev_train))
    threat_fsm = ThreatStateMachine()
    scan_energy = ScanEnergy()
    if 'cv_dist' in cv_components:
        print(f"  ✅ FSM reference distribution: "
              + "  ".join(f"{s}={v:.1%}"
                for s,v in cv_components['cv_dist'].items()))

    # LIME
    print("\n[7/9] Setting up LIME ...")
    explainer = IDSExplainer(fc, ev_train)

    # Inference
    print("\n[8/9] Inference on test set ...")
    memory = ExperienceMemory()
    behavioral_mem = BehavioralThreatMemory()
    outcome_mem    = OutcomeMemory()
    pn_te, pi_te, pl_te = lm.predict_proba_raw(test_df)
    pn_te, pi_te, pl_te = calibrate(pn_te), calibrate(pi_te), calibrate(pl_te)
    ev_test = build_evidence_vector(pn_te, pi_te, pl_te, test_df['phys_delta'].values)
    fs_all  = fc.predict_proba_score(ev_test)

    # Warm-up memory
    warmup = max(int(len(test_df)*0.3), 50)
    for i in range(warmup):
        pn,pi,pl = float(pn_te[i]),float(pi_te[i]),float(pl_te[i])
        d = float(test_df['phys_delta'].iloc[i]); fs = float(fs_all[i])
        ev_v = [float(ev_test['L_net'].iloc[i]),float(ev_test['L_iot'].iloc[i]),
                float(ev_test['L_log'].iloc[i]),d]
        _, sig = memory.lookup(pn,pi,pl,d)
        memory.store(sig, int(fs>TAU_HIGH), 'warmup', fs, ev_v, generate_pseudo_label(fs, pn, pi, pl))
    memory.run_dbscan()

    # Main pass
    decisions,reasons,confidences,latencies,from_mem = [],[],[],[],[]
    for i in range(len(test_df)):
        t0 = time.time()
        pn,pi,pl = float(pn_te[i]),float(pi_te[i]),float(pl_te[i])
        d  = float(test_df['phys_delta'].iloc[i]); fs = float(fs_all[i])
        ev_v = [float(ev_test['L_net'].iloc[i]),float(ev_test['L_iot'].iloc[i]),
                float(ev_test['L_log'].iloc[i]),d]
        # Scan energy accumulates over the WHOLE stream. It previously sat in
        # the cache-miss branch and so saw 25% of events; a scanner's repeated
        # probes are exactly what the cache absorbs.
        if SCAN_FSM and 'scan_energy' in dir():
            _s = str(test_df['src_ip'].iloc[i]) if 'src_ip' in test_df.columns else 'unknown'
            _c = lambda c: (float(test_df[c].iloc[i]) if c in test_df.columns else 0.0)
            scan_energy.observe(_s, _c('phys_fanout'),
                                _c('phys_port_entropy'), _c('phys_fail_rate'))

        cached, sig = memory.lookup(pn,pi,pl,d)
        if cached:
            decisions.append(cached['decision']); reasons.append(cached['reason']+' [MEM]')
            confidences.append(cached['confidence']); latencies.append(time.time()-t0); from_mem.append(True)
        else:
            # ── Build feature row ──────────────────────────────────
            def _get(col): return float(test_df[col].iloc[i]) if col in test_df.columns else 0.0
            row_feats = {
                'p_net':pn,'p_iot':pi,'p_log':pl,'phys_delta':d,'phys_delta_abs':abs(d),
                'disk_write':_get('disk_write'),'cpu_usage':_get('cpu_usage'),
                'cpu_spike':_get('cpu_spike'),'mem_vgrow':_get('mem_vgrow'),
                'page_faults_maj':_get('page_faults_maj'),
                'src_bytes':_get('net_bytes'),'temp_roll_std':_get('temp_roll_std'),
            }

            # ── FIX 5: Forced policy check first ───────────────────
            forced, force_reason = check_forced_policies(row_feats)
            if forced and not NO_FORCED:
                decisions.append(1)
                reasons.append(f"[FORCED_ACTION] {force_reason}")
                confidences.append(0.99); latencies.append(time.time()-t0)
                from_mem.append(False); continue

            # ── FIX 4: Apply modality trust dampening ──────────────
            pn_t, pi_t, pl_t = trust_tracker.apply(pn, pi, pl)

            # ── Unified Risk Score ──────────────────────────────────
            cid_r,c_risk,_ = memory.find_closest_cluster(ev_v)
            R,breakdown,p_flags = risk_scorer.score(pn_t,pi_t,pl_t,d,0.0,c_risk,row_feats,adaptive_tau)

            # ── FIX 3: Contextual tau ───────────────────────────────
            layer_std = float(np.std([pn,pi,pl]))
            recent_density = sum(decisions[-20:]) / max(len(decisions[-20:]),1) if decisions else 0.0
            tau_h_ctx, tau_l_ctx, ctx_reason = contextual_tau.get_tau(
                layer_disagreement=layer_std,
                trust_scores=trust_tracker.trust
            )
            adaptive_tau.tau_high = tau_h_ctx
            adaptive_tau.tau_low  = tau_l_ctx

            # ── Per-source scan accumulator ─────────────────────────
            # Scoped feedback: evidence about this host lowers the bar for this
            # host only, and decays. Not a global escalation loop.
            if SCAN_FSM and 'scan_energy' in dir():
                _src = (str(test_df['src_ip'].iloc[i])
                        if 'src_ip' in test_df.columns else 'unknown')
                _rel = scan_energy.relief(_src)   # energy already accumulated above
                if _rel:
                    adaptive_tau.tau_high = max(adaptive_tau.tau_high - _rel,
                                                ScanEnergy.FLOOR)
                    tau_h_ctx = adaptive_tau.tau_high

            _inband = _band_hit('site3_secondary_stream', fs, R, tau_l_ctx, tau_h_ctx)
            eff_score = R if _inband else fs
            dec,reason,conf = hierarchical_decision(pn_t,pi_t,pl_t,d,eff_score,ev_v,memory,adaptive_tau)
            if p_flags: reason = reason + " | POLICY:" + p_flags[0].split("[")[1].rstrip("]")

            # ── Multi-stage threat level ────────────────────────────
            t_level, t_name, _ = score_to_threat_level(eff_score, P_risk if 'P_risk' in dir() else 0.0)
            camp_risk = behavioral_mem.get_campaign_risk() if 'behavioral_mem' in dir() else 0.0
            s_level = threat_fsm.update(t_level, cluster_risk=max(c_risk, camp_risk), policy_risk=min(P_risk if 'P_risk' in dir() else 0.0, 1.0))
            s_name  = THREAT_LEVELS[s_level]['name']
            # Escalation gate. It compares against tau_low AFTER the drift update
            # (AdaptiveThreshold.update), which can sit below the tau_low the
            # fusion stage decided with: it limits the reversal of confident
            # negatives, it does not prevent it (paper §5.5: 35 of 213 in C).
            if s_level >= 2 and eff_score > adaptive_tau.tau_low and not NO_ESCALATION:
                dec = 1   # ELEVATED+ → binary attack (gated)
            reason = f"[{s_name}] {reason}"

            # ── Update trust tracker ────────────────────────────────
            trust_tracker.update(dec, pn, pi, pl)
            memory.store(sig,dec,reason,conf,ev_v,generate_pseudo_label(fs,pn,pi,pl))

            # ── Behavioral memory: store intent profile + escalation path ──
            # Extract attack family scores (atk_* features) as intent profile
            intent_prof = {}
            for c in ['atk_scan','atk_dos','atk_brute_force','atk_exfiltration',
                      'atk_injection','atk_mitm','atk_ransomware','atk_backdoor',
                      'atk_password']:
                val = float(test_df[c].iloc[i]) if c in test_df.columns else float(row_feats.get(c,0))
                if val > 0: intent_prof[c.replace('atk_','')] = val
            temporal_prof = {"periodicity": row_feats.get('phys_periodicity',0),
                             "burst":       row_feats.get('phys_burst',0),
                             "fail_rate":   row_feats.get('phys_fail_rate',0)}
            behavioral_mem.store(ts=float(test_df['ts'].iloc[i]) if 'ts' in test_df.columns else i,
                                 intent_profile=intent_prof,
                                 temporal_profile=temporal_prof,
                                 escalation_path=[s_name],
                                 net_risk=float(row_feats.get('net_risk_score', eff_score)))

            # ── Outcome memory: auto-record based on final decision ─────────
            dom_tag = max(intent_prof, key=intent_prof.get) if intent_prof else "unknown"
            outcome_mem.record(dom_tag, eff_score,
                               'confirmed' if dec==1 else 'false_positive')

            decisions.append(dec); reasons.append(reason); confidences.append(conf)
            latencies.append(time.time()-t0); from_mem.append(False)

    # Results
    print("\n[9/9] Evaluating ...")
    results = test_df.copy()
    results['pred_label']   = decisions
    results['reason']       = reasons
    results['confidence']   = confidences
    results['latency_ms']   = [l*1000 for l in latencies]
    results['from_memory']  = from_mem
    _hits = int(sum(1 for x in from_mem if x))
    print(f"\n  cache: {_hits:,} of {len(from_mem):,} decisions served "
          f"({_hits/max(len(from_mem),1):.1%})"
          + ("   [cache off: the full decision path ran for every event]"
             if CACHE_DISABLED else
             "   [--cache: these decisions bypassed scoring, thresholds and escalation]"))
    results['fusion_score'] = fs_all
    metrics = evaluate(results['final_label'], results['pred_label'], results['fusion_score'],
                       y_type=results['type'] if 'type' in results.columns else None)
    _b5_report(results, train_df)
    plot_score_distribution(results['final_label'], results['fusion_score'],
                             adaptive_tau.tau_high, adaptive_tau.tau_low)

    memory.run_dbscan(); memory.plot_clusters(); memory.stats()

    # Speed
    comp = [l for l,m in zip(latencies,from_mem) if not m]
    cach = [l for l,m in zip(latencies,from_mem) if m]
    if comp:
        ac = np.mean(comp)*1000
        am = (f" | cache hits: {np.mean(cach)*1000:.4f}ms over {len(cach):,}" if cach else "")
        print(f"\n[Speed] AI: {ac:.4f}ms per event over {len(comp):,} decided events{am}"
              "  (wall clock on pre-computed features; excludes preprocessing)")

    # LIME sample
    print("\n" + "═"*66)
    print("  INCIDENT EXPLANATIONS — First 3 Detected Attacks")
    print("═"*66)
    # Skip warmup period (first 30%) — those show cache reasons
    _warmup_end  = max(int(len(decisions)*0.30), 50)
    _attack_idxs = [i for i,d in enumerate(decisions)
                    if d==1 and i >= _warmup_end][:3]
    # Fall back to any attack if not enough post-warmup attacks
    if len(_attack_idxs) < 3:
        _attack_idxs = [i for i,d in enumerate(decisions) if d==1][:3]
    for idx in _attack_idxs:
        if idx >= len(ev_test): continue
        _reason = reasons[idx] if idx < len(reasons) else 'UNKNOWN'
        _fs     = float(fs_all[idx]) if hasattr(fs_all,'__len__') else 0.5
        _pn     = float(pn_te[idx])
        _pi     = float(pi_te[idx])
        _pl     = float(pl_te[idx])
        _pd     = float(full_fused['phys_delta'].iloc[idx])
        # Build intent profile
        _ip = {}
        for _c in ['atk_scan','atk_dos','atk_brute_force',
                   'atk_exfiltration','atk_ransomware',
                   'atk_backdoor','atk_password']:
            _v = float(full_fused[_c].iloc[idx]) \
                 if _c in full_fused.columns else 0.0
            if _v > 0.20: _ip[_c.replace('atk_','')] = _v
        # LIME pairs if available
        _lime_pairs = []
        if explainer and explainer._ready:
            _lime_pairs = explainer.get_pairs(ev_test.iloc[[idx]])
        # Full structured explanation
        _expl = _explain_engine.full_event_explanation(
            event_idx=idx, decision=1, reason=_reason,
            fusion_score=_fs, p_net=_pn, p_iot=_pi, p_log=_pl,
            phys_delta=_pd, s_level_name='ELEVATED', energy=0.6,
            tau_high=float(adaptive_tau.tau_high),
            tau_low=float(adaptive_tau.tau_low),
            trust_scores=dict(trust_tracker.trust),
            intent_profile=_ip, lime_pairs=_lime_pairs,
        )
        for _ln in _expl: print(_ln)

    # Adaptive threshold status
    print("\n[Policy Engine Stats]")
    for k,v in policy_engine.stats().items(): print(f"  {k:<20} {v}")
    print("\n[Modality Trust Scores]")
    for k,v in trust_tracker.status().items(): print(f"  {k:<20} {v}")
    print("\n[Contextual Threshold Stats]")
    for k,v in contextual_tau.stats().items(): print(f"  {k:<20} {v}")
    print("\n[Adaptive Threshold Status]")
    for k,v in adaptive_tau.status().items(): print(f"  {k:<20} {v}")

    # Decision breakdown
    print("\n[Decision Breakdown]")
    # Multi-stage threat level summary
    print("\n[Threat Level Distribution]")
    for lvl, info in THREAT_LEVELS.items():
        c = sum(1 for r in reasons if f"[{info['name']}]" in r)
        if c: print(f"  {info['name']:<16} {c:>4}  → {info['action']}")
    print("\n[Decision Breakdown]")
    for tag in ['OVERRIDE','FUSION_ALERT','FUSION_NORMAL','CLUSTER_MATCH','ANOMALY','SUSPICIOUS_SAFE','MEM','POLICY','FORCED']:
        c = sum(1 for r in reasons if tag in r)
        if c: print(f"  {tag:<22} {c}")

    os.makedirs('outputs', exist_ok=True)
    results.to_csv('test_results.csv', index=False)
    print(f"\n✅ Results → test_results.csv")

    # Online learner demo
    print("\n[OnlineLearner] Testing retraining loop ...")
    ol = OnlineLearner(fc, memory, mode='batch')
    ol_result = ol.retrain(force=True)
    print(f"  Retrain result: {ol_result}")


    print("\n✅ FULL PIPELINE COMPLETE\n")
    return lm, fc, memory, explainer, results, metrics


# ════════════════════════════════════════════════════════════════════
# SECTION 13 — FASTAPI SERVER
# ════════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════════
# SOC-ANALYST-FRIENDLY EXPLANATION LAYER
#
# Translates the technical decision reason + raw layer scores into a
# plain-English summary, a severity/action recommendation, and a
# named "top contributing signal" — the fields a SOC analyst actually
# needs at a glance, without needing to know the internal reason codes.
# ════════════════════════════════════════════════════════════════════

MODALITY_LABELS = {
    "net": "Network Traffic",
    "iot": "IoT / Physical Sensors",
    "log": "Linux System Logs",
}

REASON_SUMMARY = {
    "OVERRIDE":        "A single signal was so extreme it triggered an automatic override, bypassing the normal scoring process.",
    "FUSION_ALERT":     "Combined evidence across modalities exceeded the trained detection threshold.",
    "FUSION_NORMAL":    "Combined evidence is confidently within normal operating range.",
    "CLUSTER_MATCH":    "This event closely resembles a known attack pattern seen before.",
    "ANOMALY":          "This event doesn't match any known pattern — a novel anomaly that may be a zero-day or unusual legitimate behavior.",
    "SUSPICIOUS_SAFE":  "Evidence is inconclusive; the system is defaulting to normal but this event sits in the uncertain zone.",
}

def soc_friendly_explanation(reason, p_net, p_iot, p_log, fusion_score,
                              tau_high, tau_low, dec):
    """
    Build the analyst-facing fields for one /analyze response:
      severity, recommended_action, summary, top_signal
    """
    reason_key = reason.split(":")[0].strip()
    summary = REASON_SUMMARY.get(reason_key,
        "Decision made from combined modality evidence.")

    # Severity + recommended action reuse the same 5-level scale as the
    # main pipeline's ThreatStateMachine, computed from the fusion score
    # alone (no policy/cluster risk available in the lightweight API path).
    level, level_name, action = score_to_threat_level(fusion_score, 0.0)
    if dec == 1 and level < 2:
        # Ensure a positive detection is never reported as a low severity
        level, level_name, action = 2, "ELEVATED", "alert SOC"

    scores = {"net": p_net, "iot": p_iot, "log": p_log}
    top_mod = max(scores, key=scores.get)
    top_label = MODALITY_LABELS[top_mod]

    position = ("above the attack threshold" if fusion_score > tau_high else
                "below the confident-normal floor" if fusion_score < tau_low else
                "in the uncertain zone between thresholds")

    return {
        "severity": level_name,
        "recommended_action": action,
        "summary": summary,
        "top_signal": {"modality": top_label, "score": round(scores[top_mod], 4)},
        "score_position": position,
    }


def start_api():
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import StreamingResponse
        from pydantic import BaseModel, Field
        import uvicorn
    except ImportError:
        print("❌ FastAPI/uvicorn not installed.")
        print("   pip install fastapi uvicorn")
        return

    app = FastAPI(title="IDS Fusion API v3",
                  description="Context-Aware Multi-Layer IDS", version="3.0.0")
    app.add_middleware(CORSMiddleware, allow_origins=["*"],
                       allow_methods=["*"], allow_headers=["*"])

    class State:
        def __init__(self):
            self.lm      = None; self.fc    = None
            self.memory  = ExperienceMemory()
            self.explainer = None; self.ol = None
            self.adaptive_tau = None
            self.ctx     = default_context()
            self.history = []; self.counter = 0
            self.score_history = []   # [(fusion_score, decision), ...] this session
    st = State()

    @app.on_event("startup")
    def load():
        if os.path.exists('models/net_model.pkl'):
            try:
                st.lm = LayerModels(); st.lm.load()
                st.fc = FusionClassifier(); st.fc.load()
                st.ol = OnlineLearner(st.fc, st.memory, mode='incremental', auto=True)
                if os.path.exists('models/adaptive_threshold.pkl'):
                    st.adaptive_tau = AdaptiveThreshold.load()
                    print(f"✅ Adaptive threshold loaded "
                          f"(tau_high={st.adaptive_tau.tau_high}, "
                          f"tau_low={st.adaptive_tau.tau_low})")
                else:
                    st.adaptive_tau = None
                    print("⚠️  No saved adaptive threshold found — "
                          "/analyze will fall back to hardcoded TAU_HIGH/TAU_LOW "
                          "defaults (0.65/0.30). Retrain with this updated file "
                          "to fix this.")
                print("✅ Models loaded")
            except Exception as e:
                print(f"⚠️  {e}")

    class AnalyzeReq(BaseModel):
        net_duration: float=0.0; net_bytes: float=0.0; net_proto_enc: float=0.0
        current_temp: float=20.0; phys_delta: float=0.0; phys_delta_abs: float=0.0
        disk_write: float=0.0; cpu_usage: float=0.0

    class IntentReq(BaseModel):
        policy: str

    class RetrainReq(BaseModel):
        force: bool=False; mode: str='incremental'

    @app.get("/health")
    def health():
        return {"status":"online","models_ready":st.lm is not None,
                "cache":len(st.memory.db),"context":st.ctx,
                "incidents":st.counter}

    @app.post("/analyze")
    def analyze(req: AnalyzeReq):
        if st.lm is None: raise HTTPException(503,"Run pipeline first")
        st.counter += 1; inc = f"INC-{st.counter:05d}"; t0=time.time()

        # Context transform
        delta, net_dur, disk_w = apply_context(req.phys_delta, req.net_duration,
                                                req.disk_write, st.ctx)
        row = pd.DataFrame([{
            'net_duration':net_dur,'net_bytes':req.net_bytes,'net_proto_enc':req.net_proto_enc,
            'current_temp':req.current_temp,'phys_delta':delta,'phys_delta_abs':abs(delta),
            'disk_write':disk_w,'cpu_usage':req.cpu_usage,
            'net_label':0,'iot_label':0,'log_label':0,'final_label':0}])

        pn_r,pi_r,pl_r = st.lm.predict_proba_raw(row)
        pn,pi,pl = float(calibrate(pn_r)[0]),float(calibrate(pi_r)[0]),float(calibrate(pl_r)[0])
        cached,sig = st.memory.lookup(pn,pi,pl,delta)

        if cached:
            dec=cached['decision']; reason=cached['reason']+' [MEM]'; conf=cached['confidence']
            fs=float(conf if dec==1 else 1-conf); mem=True
        else:
            ev = build_evidence_vector(np.array([pn]),np.array([pi]),np.array([pl]),np.array([delta]))
            fs = float(st.fc.predict_proba_score(ev)[0])
            ev_v=[float(ev['L_net'].iloc[0]),float(ev['L_iot'].iloc[0]),
                  float(ev['L_log'].iloc[0]),delta]
            dec,reason,conf = hierarchical_decision(pn,pi,pl,delta,fs,ev_v,st.memory,st.adaptive_tau)
            st.memory.store(sig,dec,reason,float(conf),ev_v,generate_pseudo_label(fs, pn, pi, pl))
            mem=False

        verdict=("ATTACK" if dec==1 else ("SUSPICIOUS" if "SUSPICIOUS" in reason else "NORMAL"))
        st.score_history.append((float(fs), int(dec)))
        tau_h = st.adaptive_tau.tau_high if st.adaptive_tau else TAU_HIGH
        tau_l = st.adaptive_tau.tau_low  if st.adaptive_tau else TAU_LOW
        soc = soc_friendly_explanation(reason, pn, pi, pl, fs, tau_h, tau_l, dec)
        return {"incident_id":inc,"verdict":verdict,
                "severity":soc["severity"],"recommended_action":soc["recommended_action"],
                "summary":soc["summary"],
                "confidence":f"{float(conf):.1%}","fusion_score":round(fs,4),
                "score_position":soc["score_position"],
                "top_signal":soc["top_signal"],
                "reason":reason,"from_memory":mem,
                "latency_ms":round((time.time()-t0)*1000,3),
                "layer_scores":{
                    "network": {"label": MODALITY_LABELS["net"], "score": round(pn,4)},
                    "iot":     {"label": MODALITY_LABELS["iot"], "score": round(pi,4)},
                    "log":     {"label": MODALITY_LABELS["log"], "score": round(pl,4)},
                },
                "context_applied":{"delta_used":round(delta,4),"net_dur_used":round(net_dur,4)},
                "threshold_used":{"tau_high":round(tau_h,4),"tau_low":round(tau_l,4),
                                   "source":"trained" if st.adaptive_tau else "hardcoded_fallback"}}

    @app.post("/intent/apply")
    def intent(req: IntentReq):
        old = dict(st.ctx)
        st.ctx, desc = parse_intent(req.policy, st.ctx)
        st.history.append({"policy":req.policy,"action":desc})
        return {"matched": st.ctx!=old,"action":desc,
                "context_before":old,"context_after":dict(st.ctx),
                "note":"Signals scaled before model — thresholds unchanged"}

    @app.get("/intent/context")
    def ctx(): return {"active_context":st.ctx,"history":st.history[-5:]}

    @app.get("/memory/stats")
    def mem_stats():
        lbl=st.memory.cluster_labels
        return {"cache":len(st.memory.db),"evidence":len(st.memory.evidence_log),
                "pseudo_labels":len([x for x in st.memory.pseudo_labels if x is not None]),
                "clusters":len(st.memory.cluster_profiles),
                "novel_anomalies":int(list(lbl).count(-1)) if lbl is not None else 0}

    @app.get("/memory/clusters")
    def clusters():
        if not st.memory.cluster_profiles:
            st.memory.run_dbscan()
        return {"clusters":[{"id":cid,"size":p['size'],"risk":round(p['risk'],3)}
                             for cid,p in st.memory.cluster_profiles.items()]}

    @app.post("/memory/run_dbscan")
    def run_db():
        lbl=st.memory.run_dbscan()
        return {"clusters":len(st.memory.cluster_profiles),
                "novel":int(list(lbl).count(-1)) if lbl is not None else 0}

    @app.post("/learn/retrain")
    def retrain(req: RetrainReq):
        if st.ol is None:
            if st.fc is None: raise HTTPException(503,"No model")
            st.ol = OnlineLearner(st.fc, st.memory, mode=req.mode)
        st.ol.mode = req.mode
        return st.ol.retrain(force=req.force)

    @app.get("/learn/status")
    def learn_status():
        if st.ol is None: return {"active":False}
        return {"active":True, **st.ol.status()}

    @app.get("/learn/history")
    def learn_hist():
        if st.ol is None: return {"retrains":[],"total":0}
        return {"retrains":st.ol._log,"total":st.ol._count}

    @app.get("/threshold/status")
    def threshold_status():
        if st.adaptive_tau is None:
            return {"source": "hardcoded_fallback",
                    "tau_high": TAU_HIGH, "tau_low": TAU_LOW,
                    "note": "No trained AdaptiveThreshold found on disk — "
                            "retrain and restart the server."}
        return {"source": "trained", **st.adaptive_tau.status()}

    @app.get("/analyze/distribution")
    def analyze_distribution():
        """
        Live PNG plot of this session's fusion scores (normal vs attack,
        by decision), with the current tau_high/tau_low marked — the
        same view a SOC analyst would use to see where live traffic sits
        relative to the detection boundary.
        """
        import io

        tau_h = st.adaptive_tau.tau_high if st.adaptive_tau else TAU_HIGH
        tau_l = st.adaptive_tau.tau_low  if st.adaptive_tau else TAU_LOW

        fig, ax = plt.subplots(figsize=(8, 4.5))
        if not st.score_history:
            ax.text(0.5, 0.5, "No events analyzed yet this session",
                    ha='center', va='center', fontsize=13, color='#666')
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.axis('off')
        else:
            scores = np.array([s for s, d in st.score_history])
            decs   = np.array([d for s, d in st.score_history])
            bins = np.linspace(0, 1, 31)
            if (decs == 0).any():
                ax.hist(scores[decs==0], bins=bins, alpha=0.65, color='#2E7D32',
                        label=f'Classified Normal (n={int((decs==0).sum())})')
            if (decs == 1).any():
                ax.hist(scores[decs==1], bins=bins, alpha=0.65, color='#C62828',
                        label=f'Classified Attack (n={int((decs==1).sum())})')
            ax.axvline(tau_h, color='#1F4D78', linestyle='--', linewidth=2,
                       label=f'tau_high = {tau_h:.3f}')
            ax.axvline(tau_l, color='#2E75B6', linestyle='--', linewidth=2,
                       label=f'tau_low = {tau_l:.3f}')
            ax.set_xlabel('Fusion score')
            ax.set_ylabel('Event count (this session)')
            ax.legend(fontsize=9)
        ax.set_title('Live Fusion Score Distribution')
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=140)
        plt.close(fig)
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")

    print("\n🚀 Starting API server at http://127.0.0.1:8000")
    print("   Interactive docs: http://127.0.0.1:8000/docs\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)


# ════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    # ═══════════════════════════════════════════════════════════════
    # DATASET PATHS — Full TON_IoT dataset
    # ═══════════════════════════════════════════════════════════════
    # Folder that holds the four Processed_*_dataset folders of TON_IoT.
    # Set it with --base "<folder>" or the CATF_DATA environment variable.
    BASE = os.environ.get('CATF_DATA', os.path.join('..', 'data', 'TON_IoT', 'Processed_datasets'))
    if '--base' in sys.argv:
        BASE = sys.argv[sys.argv.index('--base') + 1]
    _j = os.path.join

    # Network: all 23 files concatenated
    NET_FILES = [_j(BASE, 'Processed_Network_dataset', f'Network_dataset_{i}.csv')
                 for i in range(1, 24)]

    # IoT: all 7 device types concatenated
    IOT_FILES = [_j(BASE, 'Processed_IoT_dataset', f) for f in (
        'IoT_Fridge.csv', 'IoT_Garage_Door.csv', 'IoT_GPS_Tracker.csv', 'IoT_Modbus.csv',
        'IoT_Motion_Light.csv', 'IoT_Thermostat.csv', 'IoT_Weather.csv')]

    # Linux: all 6 log files concatenated
    LOG_FILES = [_j(BASE, 'Processed_Linux_dataset', f) for f in (
        'linux_disk_1.csv', 'linux_disk_2.csv', 'linux_memory1.csv', 'linux_memory2.csv',
        'Linux_process_1.csv', 'Linux_process_2.csv')]

    # Windows: both versions concatenated
    WIN_FILES = [_j(BASE, 'Processed_Windows_dataset', f) for f in (
        'windows7_dataset.csv', 'windows10_dataset.csv')]

    N_ROWS = 46000   # --n: 46000 // 23 network files = 2,000 records per file (the paper)
    # ═══════════════════════════════════════════════════════════════

    parser = argparse.ArgumentParser(
        description='CATF-IDS. No switches = the paper\'s final configuration (D); --config_c = configuration C.')
    parser.add_argument('--fpr_ablation', action='store_true',
                        help='Run FPR ablation study (5 variants)')
    parser.add_argument('--quick_compare', action='store_true',
                        help='Run targeted 6-combo comparison instead of full grid')
    parser.add_argument('--test',  action='store_true',
                        help='Run self-test with synthetic data (no CSVs needed)')
    parser.add_argument('--serve', action='store_true',
                        help='Start API server (requires trained models)')
    parser.add_argument('--all', action='store_true',
                        help='One-shot: run on synthetic data, then start the API server')
    parser.add_argument('--n',     type=int, default=N_ROWS)
    parser.add_argument('--base', default=BASE,
                        help='folder holding the Processed_*_dataset folders of TON_IoT')
    parser.add_argument('--scan_window', type=int, default=60,
                        help='session-physics window in seconds (0 = off)')
    parser.add_argument('--no_window',   action='store_true',
                        help='pool session physics per src_ip, pre-windowing behaviour')
    parser.add_argument('--scan_fsm',    action='store_true',
                        help='enable the per-source scan accumulator')
    parser.add_argument('--scan_trigger', type=float, default=0.60,
                        help='scan energy required before threshold relief')
    parser.add_argument('--cache',          action='store_true',
                        help='the experience cache serves decisions (demonstration M)')
    parser.add_argument('--forced',         action='store_true',
                        help='fixed conditions force alerts (demonstration F)')
    parser.add_argument('--no_escalation',  action='store_true',
                        help='escalation state does not raise a verdict')
    parser.add_argument('--tau_floor',      type=float, default=0.35,
                        help='floor on tau_high (default 0.35)')
    parser.add_argument('--legacy_cv_rule', action='store_true',
                        help='CV verdicts from the escalation stage alone (pre-correction loop)')
    parser.add_argument('--p2_recal',       action='store_true',
                        help='label-free adaptation: recalibrate the drift reference from confident normals')
    parser.add_argument('--p2_pseudo',      action='store_true',
                        help='label-free adaptation: refit fusion on replay + consistent pseudo-labels')
    parser.add_argument('--p2_replay_only', action='store_true',
                        help='control for --p2_pseudo: same refit schedule, replay rows only')
    args = parser.parse_args()

    if args.all:
        print("\n🧪 ONE-SHOT MODE — synthetic run, then API server")
        net_p, iot_p, log_p = save_synthetic_csvs('synthetic_data')
        run_pipeline(net_p, iot_p, log_p, n_samples=args.n)
        print("\n" + "="*60)
        print("  TRAINING COMPLETE — STARTING API SERVER")
        print("="*60)
        start_api()
    elif args.serve:
        start_api()
    elif args.test:
        print("\n🧪 SELF-TEST MODE — Synthetic ToN_IoT data")
        net_p, iot_p, log_p = save_synthetic_csvs('synthetic_data')
        run_pipeline(net_p, iot_p, log_p, n_samples=args.n)
    else:
        print(f"\n📂 Running on full TON_IoT dataset:")
        print(f"   Network : {len(NET_FILES)} files")
        print(f"   IoT     : {len(IOT_FILES)} devices")
        print(f"   Linux   : {len(LOG_FILES)} log files")
        print(f"   Windows : {len(WIN_FILES)} files")
        run_pipeline_multi(NET_FILES, IOT_FILES, LOG_FILES, WIN_FILES, n_samples=args.n)