#!/usr/bin/env python3
"""
final_run.py -- configuration D (all nine stages acting in the evaluation),
checked against configuration C and attributed change by change.

The four changes in catf_ids.py (each has a switch; --config_c turns all off):
  trust      stage 4 as its own specification says: an exponential average of
             each modality's agreement with the verdict (the legacy rule sat at
             the 1.0 cap, so trust changed nothing)
  envelopes  stage 5, hard part: limits set from the site's own normal
             training records (99.9th percentile, per field, per device)
             force an alert, replacing the paper's five fixed conditions
  limits     stage 5, soft part: the policy pressure reads site-calibrated
             limits (99th / 99.9th percentiles) instead of POLICY_THRESHOLDS
  frozen     the stream is scored by the five validated fold fusion models
             (before, each was refit in place by a one-pass SGD on its fold's
             pseudo-labels after the fold was scored, and the refits scored
             the stream)
  one path   cross-validation decides every event exactly as the stream does
             (trust damping, contextual thresholds, band, gated escalation),
             and the stream feeds the policy pressure to the energy state
             machine as cross-validation already did

Runs, all from one data preparation (about 15 minutes):
  C              must reproduce the paper (checks below)
  C+envelopes    must reproduce policy_study.py's envelope arm (cross-check)
  C+frozen       C with the ensemble the paper describes (the fair baseline)
  D              the final configuration
  D-trust, D-envelopes, D-limits, D-one_path, D-frozen   D with one change
                 removed (D-frozen must reproduce the previous D run)

Usage (in the folder that holds catf_ids.py and threshold_stress.py):
  python final_run.py
  python final_run.py --base "<folder with the Processed_*_dataset folders>"
Then, for every report and figure of the final configuration:
  python catf_ids.py

Writes ./final_run/final_run_summary.txt, final_run.csv and logs/.
"""
import argparse
import contextlib
import copy
import importlib.util
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import threshold_stress as TS          # noqa: E402  (file lists, data preparation, metrics)

OUT = os.path.join(HERE, "final_run")
FLAGS = ("TRUST_EMA", "SITE_ENVELOPES", "SITE_LIMITS", "ONE_PATH", "FROZEN_ENSEMBLE")
CONFIGS = [("C", (0, 0, 0, 0, 0)), ("C+envelopes", (0, 1, 0, 0, 0)),
           ("C+frozen", (0, 0, 0, 0, 1)), ("D", (1, 1, 1, 1, 1)),
           ("D-trust", (0, 1, 1, 1, 1)), ("D-envelopes", (1, 0, 1, 1, 1)),
           ("D-limits", (1, 1, 0, 1, 1)), ("D-one_path", (1, 1, 1, 0, 1)),
           ("D-frozen", (1, 1, 1, 1, 0))]
CHANGE = {"D-trust": "trust", "D-envelopes": "envelopes", "D-limits": "limits",
          "D-one_path": "one path", "D-frozen": "frozen ens."}
PREVIOUS_D = {"TP": 15636, "FP": 292}     # D before the ensemble fix (= D-frozen)
PAPER_C_CV = {"TP": 6416, "FP": 1027}                  # pooled over the five folds
PAPER_C_STREAM = dict(TS.PAPER_STREAM)
POLICY_STUDY_ENV = {"cv": {"TP": 6419, "FP": 1102}, "stream": {"TP": 16832, "FP": 699}}


class _Stop(Exception):
    pass


def load_catf():
    path = os.path.join(HERE, "catf_ids.py")
    if not os.path.exists(path):
        sys.exit(f"catf_ids.py not found next to this script ({HERE})")
    argv, sys.argv = sys.argv, [path]
    spec = importlib.util.spec_from_file_location("catf_ids", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["catf_ids"] = mod
    spec.loader.exec_module(mod)
    sys.argv = argv
    if not all(hasattr(mod, f) for f in FLAGS):
        sys.exit("this catf_ids.py has no configuration D: apply _build/patch_configD.py first")
    try:
        import atexit
        atexit.unregister(mod._band_report)
    except Exception:                                             # noqa: BLE001
        pass
    return mod


def conf(y, p):
    y, p = np.asarray(y).astype(int), np.asarray(p).astype(int)
    TP = int(((p == 1) & (y == 1)).sum()); FP = int(((p == 1) & (y == 0)).sum())
    FN = int(((p == 0) & (y == 1)).sum()); TN = int(((p == 0) & (y == 0)).sum())
    P = TP / (TP + FP) if TP + FP else 0.0
    R = TP / (TP + FN) if TP + FN else 0.0
    return {"TP": TP, "FP": FP, "FN": FN, "TN": TN, "precision": P, "recall": R,
            "F1": 2 * P * R / (P + R) if P + R else 0.0, "FPR": FP / (FP + TN) if FP + TN else 0.0}


def band_in(catf, site):
    return int(catf._BAND.get(site, {}).get("in", 0))


def run_config(catf, prep, name, flags):
    for f, v in zip(FLAGS, flags):
        setattr(catf, f, bool(v))
    out = {"config": name, **{f: int(v) for f, v in zip(FLAGS, flags)}}

    # ── cross-validation: the pipeline's own fold decisions ──
    import sklearn.metrics as skm
    got, o_f1 = [], skm.f1_score

    def f1_rec(y_true, y_pred, *a, **k):
        got.append((np.asarray(y_true).copy(), np.asarray(y_pred).copy()))
        return o_f1(y_true, y_pred, *a, **k)
    b0 = band_in(catf, "cv_folds")
    skm.f1_score = f1_rec
    try:
        with open(os.path.join(OUT, "logs", f"{name}_cv.log"), "w", encoding="utf-8") as lg, \
                contextlib.redirect_stdout(lg):
            res = catf.run_cross_validation(prep.fused.copy(), n_splits=5)
    finally:
        skm.f1_score = o_f1
    folds = [fr.metrics for fr in res[1]]
    for k in ("f1", "precision", "recall", "fpr", "roc_auc"):
        v = np.array([m[k] for m in folds], dtype=float)
        out[f"cv_{k}"], out[f"cv_{k}_std"] = float(v.mean()), float(v.std())
    out["cv_fold_auc"] = [round(float(m["roc_auc"]), 4) for m in folds]
    pooled = conf(np.concatenate([t for t, _ in got]), np.concatenate([p for _, p in got]))
    out.update({f"cv_{k}_pooled": v for k, v in pooled.items()})
    out["cv_band"] = band_in(catf, "cv_folds") - b0
    out["cv_trust"] = {m: round(float(np.mean([fr.trust[m] for fr in res[1]])), 3)
                       for m in ("net", "iot", "log")}

    # ── the stream, with this configuration's own cross-validation result ──
    names = ("load_multi", "preprocess_network", "preprocess_iot_multi",
             "preprocess_linux_multi", "preprocess_windows", "fuse_datasets",
             "run_cross_validation", "_tau_trace_report")
    saved = {n: getattr(catf, n) for n in names}
    sentinel, cap = pd.DataFrame({"_": [0]}), {}
    for n in names[:5]:
        setattr(catf, n, lambda *a, **k: sentinel)
    catf.fuse_datasets = lambda *a, **k: (prep.fused.copy(), prep.full.copy())
    catf.run_cross_validation = lambda *a, **k: copy.deepcopy(res)

    def ttr(trace, results, *a, **k):
        cols = [c for c in ("final_label", "pred_label", "reason", "type") if c in results.columns]
        cap["r"] = results[cols].copy()
        raise _Stop()
    catf._tau_trace_report = ttr
    b0 = band_in(catf, "site2_full_stream")
    try:
        with open(os.path.join(OUT, "logs", f"{name}_stream.log"), "w", encoding="utf-8") as lg, \
                contextlib.redirect_stdout(lg):
            catf.run_pipeline_multi(*prep.files, n_samples=TS.N_SAMPLES)
    except _Stop:
        pass
    finally:
        for n, f in saved.items():
            setattr(catf, n, f)
    if "r" not in cap:
        raise RuntimeError(f"{name}: the stream ended early; see logs/{name}_stream.log")
    r = cap["r"]
    out.update({f"stream_{k}": v for k, v in conf(r["final_label"], r["pred_label"]).items()})
    out["stream_forced"] = int(r["reason"].astype(str).str.startswith("[FORCED_ACTION]").sum())
    out["stream_band"] = band_in(catf, "site2_full_stream") - b0
    if name == "D" and "type" in r.columns:
        g = r.assign(hit=r["pred_label"].astype(int))
        out["stream_per_class"] = {str(k): [int(len(h)), round(float(h["hit"].mean()), 4)]
                                   for k, h in g.groupby("type")}
    return out


def report(rows):
    L = []
    W = lambda s="": L.append(s)
    by = {r["config"]: r for r in rows}
    W("FINAL RUN -- configuration D (all nine stages), checked against C")
    W("")
    W("CHECKS")
    c = by.get("C")
    if c:
        ok = c["cv_TP_pooled"] == PAPER_C_CV["TP"] and c["cv_FP_pooled"] == PAPER_C_CV["FP"]
        W(f"  [{'OK' if ok else 'FAIL'}] C cv pooled TP {c['cv_TP_pooled']:,} FP {c['cv_FP_pooled']:,}"
          f" = paper {PAPER_C_CV['TP']:,} / {PAPER_C_CV['FP']:,}   (mean fold F1 {c['cv_f1']:.4f})")
        ok = c["cv_fold_auc"] == TS.PAPER_FOLD_AUC
        W(f"  [{'OK' if ok else 'FAIL'}] C cv fold AUCs {c['cv_fold_auc']} = paper")
        got = {k: c[f"stream_{k}"] for k in PAPER_C_STREAM}
        W(f"  [{'OK' if got == PAPER_C_STREAM else 'FAIL'}] C stream {got} = paper")
    e = by.get("C+envelopes")
    if e:
        for p, key in (("cv", "cv_{}_pooled"), ("stream", "stream_{}")):
            got = {k: e[key.format(k)] for k in ("TP", "FP")}
            W(f"  [{'OK' if got == POLICY_STUDY_ENV[p] else 'FAIL'}] C+envelopes {p} {got} "
              f"= policy_study.py {POLICY_STUDY_ENV[p]}")
    x = by.get("D-frozen")
    if x:
        got = {k: x[f"stream_{k}"] for k in ("TP", "FP")}
        W(f"  [{'OK' if got == PREVIOUS_D else 'FAIL'}] D-frozen stream {got} = the previous D run "
          f"{PREVIOUS_D}")
    d = by.get("D")
    if d:
        ok = d["cv_fold_auc"] == TS.PAPER_FOLD_AUC
        W(f"  [{'OK' if ok else 'FAIL'}] D cv fold AUCs unchanged (the models are the paper's)")
    W("")
    W("RESULTS")
    W(f"  {'config':<13}{'cv F1':>15}{'cv FPR':>8}{'cv TP':>7}{'cv FP':>7} |"
      f"{'str F1':>8}{'str FPR':>8}{'TP':>8}{'FP':>6}{'FN':>7}{'TN':>7}{'forced':>7}{'band':>6}"
      f"   cv trust net/iot/log")
    for r in rows:
        t = r["cv_trust"]
        W(f"  {r['config']:<13}{r['cv_f1']:>8.4f}±{r['cv_f1_std']:.4f}{r['cv_fpr']:>8.4f}"
          f"{r['cv_TP_pooled']:>7,}{r['cv_FP_pooled']:>7,} |{r['stream_F1']:>8.4f}{r['stream_FPR']:>8.4f}"
          f"{r['stream_TP']:>8,}{r['stream_FP']:>6,}{r['stream_FN']:>7,}{r['stream_TN']:>7,}"
          f"{r['stream_forced']:>7,}{r['stream_band']:>6,}   {t['net']:.3f}/{t['iot']:.3f}/{t['log']:.3f}")
    if d:
        W("")
        W("WHAT EACH CHANGE CONTRIBUTES (D minus D without it)")
        W(f"  {'change':<11}{'cv dF1':>9}{'cv dTP':>8}{'cv dFP':>8} |{'str dF1':>9}{'str dTP':>9}{'str dFP':>9}")
        for name, ch in CHANGE.items():
            x = by.get(name)
            if x:
                W(f"  {ch:<11}{d['cv_f1'] - x['cv_f1']:>+9.4f}{d['cv_TP_pooled'] - x['cv_TP_pooled']:>+8,}"
                  f"{d['cv_FP_pooled'] - x['cv_FP_pooled']:>+8,} |{d['stream_F1'] - x['stream_F1']:>+9.4f}"
                  f"{d['stream_TP'] - x['stream_TP']:>+9,}{d['stream_FP'] - x['stream_FP']:>+9,}")
        cf = by.get("C+frozen")
        if cf:
            W(f"  {'D - C+frozen':<11}{d['cv_f1'] - cf['cv_f1']:>+9.4f}{d['cv_TP_pooled'] - cf['cv_TP_pooled']:>+8,}"
              f"{d['cv_FP_pooled'] - cf['cv_FP_pooled']:>+8,} |{d['stream_F1'] - cf['stream_F1']:>+9.4f}"
              f"{d['stream_TP'] - cf['stream_TP']:>+9,}{d['stream_FP'] - cf['stream_FP']:>+9,}")
        if c:
            W(f"  {'all (D-C)':<11}{d['cv_f1'] - c['cv_f1']:>+9.4f}{d['cv_TP_pooled'] - c['cv_TP_pooled']:>+8,}"
              f"{d['cv_FP_pooled'] - c['cv_FP_pooled']:>+8,} |{d['stream_F1'] - c['stream_F1']:>+9.4f}"
              f"{d['stream_TP'] - c['stream_TP']:>+9,}{d['stream_FP'] - c['stream_FP']:>+9,}")
        pc = d.get("stream_per_class")
        if pc:
            W("")
            W("D, STREAM, SHARE FLAGGED BY THE SOURCE RECORDS' TYPE FIELD (support in brackets)")
            W("  " + "   ".join(f"{k} {v[1]:.3f} ({v[0]:,})" for k, v in sorted(pc.items())))
    txt = "\n".join(L)
    with open(os.path.join(OUT, "final_run_summary.txt"), "w", encoding="utf-8") as fh:
        fh.write(txt + "\n")
    print(txt)


def main():
    ap = argparse.ArgumentParser(description="configuration D, checked and attributed")
    ap.add_argument("--base", default=TS.DEFAULT_BASE)
    args = ap.parse_args()
    os.makedirs(os.path.join(OUT, "logs"), exist_ok=True)
    catf = load_catf()
    cwd = os.getcwd()
    os.chdir(OUT)                       # models/ and plots/ saved by the pipeline land here
    for d in ("models", "plots"):
        os.makedirs(d, exist_ok=True)
    try:
        t0 = time.time()
        print("preparing the data (the paper's loading, preprocessing and fusion) ...", flush=True)
        with open(os.path.join(OUT, "logs", "prep.log"), "w", encoding="utf-8") as lg, \
                contextlib.redirect_stdout(lg):
            prep = TS.Prep(catf, args.base)
        print(f"  balanced set {len(prep.fused):,}, stream {len(prep.full):,} "
              f"({time.time()-t0:.0f}s)", flush=True)
        rows = []
        for name, flags in CONFIGS:
            t = time.time()
            r = run_config(catf, prep, name, flags)
            rows.append(r)
            print(f"  {name:<13} cv F1 {r['cv_f1']:.4f}  stream F1 {r['stream_F1']:.4f}  "
                  f"TP {r['stream_TP']:,}  FP {r['stream_FP']:,}  ({time.time()-t:.0f}s)", flush=True)
            pd.DataFrame(rows).to_csv(os.path.join(OUT, "final_run.csv"), index=False)
        json.dump(rows, open(os.path.join(OUT, "final_run.json"), "w"), indent=1, default=str)
        print("")
        report(rows)
        print(f"\nwritten to {OUT}  (total {time.time()-t0:.0f}s)")
        print("next: python catf_ids.py   (configuration D, every report and figure)")
    finally:
        os.chdir(cwd)


if __name__ == "__main__":
    main()
