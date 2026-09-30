#!/usr/bin/env python3
"""
threshold_stress.py -- does the adaptive threshold of CATF-IDS react when the
share of attacks in the stream changes over time?   (configuration D; --config c for C)

Real TON_IoT records only. A schedule reorders the records of a test stream so
that the attack share rises and falls; every record is used exactly once and no
feature value is touched. Each schedule runs twice on the identical stream:

  adaptive   the thresholds exactly as in the paper
  frozen     both thresholds held at their starting point for the whole stream
             (tau_high at its base with the 0.35 floor applied, tau_low at its
             fitted value); everything else unchanged

The difference adaptive - frozen is the effect of threshold adaptation.

Two decision paths, as in the paper:
  cv       the stateful 5-fold cross-validation on the balanced set (D: F1 0.9050;
           C: 0.9120). Each fold's test part is reordered; training is
           unaffected, so the models are the paper's fold models. In D the
           folds decide exactly as the stream does; in C only the drift update
           moved the thresholds there.
  stream   the full stream (24,934 events; D: TP 17,649 / FP 1,172 / FN 657 /
           TN 5,456). The stream is reordered; the contextual adjustment and
           band substitution act, as deployed.

Schedules
  original      the paper's order (built-in check against the paper's figures)
  shuffled      random order, attack share constant on average (control)
  waves_short   attack share alternating 0.9 / 0.1, period 20 events
  waves_medium  the same, period 100 events
  waves_long    the same, period 1,000 events
  ramp          attack share rising slowly to 0.9 by mid-stream, then falling
  When a stream holds more attacks than normals, attack-heavy blocks are made
  longer than normal-heavy ones so every record is used and the shape holds.

Usage (in the folder that holds catf_ids.py; catf_ids.py is not modified):
  python threshold_stress.py --quick     original + waves_medium, both arms, both paths
  python threshold_stress.py             all six schedules (about an hour)
  python threshold_stress.py --path cv   one path only (cv | stream | both)
  python threshold_stress.py --replot    summary and figure again from saved events
  python threshold_stress.py --config c  configuration C, as first reported
  python threshold_stress.py --base "<folder with the Processed_*_dataset folders>"

Writes into ./threshold_stress_D/ (./threshold_stress/ with --config c; the
models/ and plots/ the pipeline saves go
there too, so the paper's own models/ folder is left alone):
  stress_summary.txt    checks, threshold behaviour, adaptive - frozen effect
  stress_summary.csv    one row per path x schedule x arm x phase
  stress_events.csv.gz  one row per event
  stress_trace.png/pdf  thresholds over the medium waves, adaptive vs frozen
  logs/                 the pipeline's own output for every run
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
OUT = os.path.join(HERE, "threshold_stress")
DEFAULT_BASE = os.environ.get("CATF_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "TON_IoT", "Processed_datasets"))
N_SAMPLES = 46000                                  # 2,000 records per network file
PAPER_FOLD_AUC = [0.9752, 0.9748, 0.9764, 0.9752, 0.9733]
PAPER_CV_F1 = 0.9120
PAPER_STREAM = {"TP": 16810, "FP": 625, "FN": 1496, "TN": 6003}
# what the original order must reproduce, per configuration (--config)
EXPECT = {"c": {"cv_f1": 0.9120, "cv_pooled": {"TP": 6416, "FP": 1027},
                "stream": dict(PAPER_STREAM)},
          "d": {"cv_f1": 0.9050, "cv_pooled": {"TP": 6415, "FP": 1133},
                "stream": {"TP": 17649, "FP": 1172, "FN": 657, "TN": 5456}}}
CONFIG = "d"
_OUT_OVERRIDE = None                               # tests only
HI, LO = 0.9, 0.1                                  # attack share in the two kinds of block
EDGE = 20                                          # events after a switch = transition
                                                   # (1/alpha of the score EWMA)
SCHEDULES_FULL = ["original", "shuffled", "waves_short", "waves_medium", "waves_long", "ramp"]
SCHEDULES_QUICK = ["original", "waves_medium"]
PERIOD = {"waves_short": 20, "waves_medium": 100, "waves_long": 1000}
ARMS = ["adaptive", "frozen"]


class _Stop(Exception):
    """Raised once the stream's per-event trace is captured: the reporting that
    follows (plots, explanations) is not needed here."""


# ── loading ───────────────────────────────────────────────────────────
def load_catf(config=None):
    """catf_ids as configuration D (default) or C (config='c')."""
    config = (config or CONFIG).lower()
    path = os.path.join(HERE, "catf_ids.py")
    if not os.path.exists(path):
        sys.exit(f"catf_ids.py not found next to this script ({HERE})")
    argv, sys.argv = sys.argv, [path] + (["--config_c"] if config == "c" else [])
    spec = importlib.util.spec_from_file_location("catf_ids", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["catf_ids"] = mod
    spec.loader.exec_module(mod)
    sys.argv = argv
    want = "_IS_C" if config == "c" else "_IS_D"
    if not getattr(mod, want, config == "c"):
        sys.exit(f"catf_ids.py did not load as configuration {config.upper()}")
    return mod


def file_lists(base):
    net = [os.path.join(base, "Processed_Network_dataset", f"Network_dataset_{i}.csv")
           for i in range(1, 24)]
    iot = [os.path.join(base, "Processed_IoT_dataset", f) for f in (
        "IoT_Fridge.csv", "IoT_Garage_Door.csv", "IoT_GPS_Tracker.csv", "IoT_Modbus.csv",
        "IoT_Motion_Light.csv", "IoT_Thermostat.csv", "IoT_Weather.csv")]
    log = [os.path.join(base, "Processed_Linux_dataset", f) for f in (
        "linux_disk_1.csv", "linux_disk_2.csv", "linux_memory1.csv", "linux_memory2.csv",
        "Linux_process_1.csv", "Linux_process_2.csv")]
    win = [os.path.join(base, "Processed_Windows_dataset", f) for f in (
        "windows7_dataset.csv", "windows10_dataset.csv")]
    return net, iot, log, win


class Prep:
    """The paper's loading, preprocessing and fusion, done once."""

    def __init__(self, catf, base):
        self.files = file_lists(base)
        missing = [f for grp in self.files for f in grp if not os.path.exists(f)]
        if missing:
            sys.exit(f"{len(missing)} data files not found, e.g. {missing[0]}\n"
                     f"Pass the dataset folder with --base.")
        net, iot, log, win = self.files
        n_per_file = max(500, N_SAMPLES // len(net))          # as run_pipeline_multi
        raw_net = catf.load_multi(net, 'Network', n_per_file=n_per_file)
        raw_iot = catf.load_multi(iot, 'IoT', n_per_file=n_per_file)
        raw_log = catf.load_multi(log, 'Linux Log', n_per_file=n_per_file)
        raw_win = catf.load_multi(win, 'Windows', n_per_file=n_per_file)
        self.fused, self.full = catf.fuse_datasets(
            catf.preprocess_network(raw_net), catf.preprocess_iot_multi(raw_iot),
            catf.preprocess_linux_multi(raw_log), catf.preprocess_windows(raw_win))

    @classmethod
    def from_frames(cls, fused, full, files):
        self = cls.__new__(cls)
        self.fused, self.full, self.files = fused, full, files
        return self


# ── schedules ─────────────────────────────────────────────────────────
def _place(y, share, rng):
    """Order the records so the cumulative attack count follows the share curve.
    Every record is used once; within a class the order is random."""
    att = rng.permutation(np.flatnonzero(y == 1))
    nor = rng.permutation(np.flatnonzero(y == 0))
    target = np.cumsum(share)
    target = target * (len(att) / target[-1])
    out = np.empty(len(y), dtype=int)
    a = b = 0
    for k in range(len(y)):
        take_attack = a < len(att) and (b >= len(nor) or a + 1 <= target[k] + 0.5)
        if take_attack:
            out[k] = att[a]; a += 1
        else:
            out[k] = nor[b]; b += 1
    return out


def schedule(name, y, seed):
    """(order, target share, phase, events since the last switch) for labels y."""
    y = np.asarray(y).astype(int)
    n = len(y)
    i = np.arange(n)
    rng = np.random.default_rng(seed)
    p = float(y.mean())
    flat = np.full(n, p)
    if name == "original":
        return i, flat, np.full(n, "all", dtype=object), i
    if name == "shuffled":
        return rng.permutation(n), flat, np.full(n, "all", dtype=object), i
    pc = float(np.clip(p, LO + 0.01, HI - 0.01))
    if name in PERIOD:
        P = PERIOD[name]
        r = (HI - pc) / (pc - LO)                  # normal-heavy : attack-heavy length
        L_hi = int(np.clip(round(P / (1 + r)), 1, P - 1))
        pos = i % P
        hi = pos < L_hi
        share = np.where(hi, HI, LO).astype(float)
        phase = np.where(hi, "attack-heavy", "normal-heavy").astype(object)
        since = np.where(hi, pos, pos - L_hi)
    elif name == "ramp":
        s0 = max(0.02, 2 * pc - HI)
        t = i / max(n - 1, 1)
        share = s0 + (HI - s0) * (1 - np.abs(2 * t - 1))
        phase = np.where(t < 0.5, "rising", "falling").astype(object)
        since = i
    else:
        raise ValueError(f"unknown schedule {name}")
    return _place(y, share, rng), share, phase, since


# ── the harness ───────────────────────────────────────────────────────
def _clone(obj):
    try:
        return copy.deepcopy(obj)
    except Exception:
        return obj


class Harness:
    """Wraps a few catf_ids functions so every decision can be recorded. The
    wrappers are passive (they only observe) except where stated: the fold
    splitter reorders the test part, and frozen() holds the thresholds."""

    def __init__(self, catf):
        self.c = catf
        self.floor = float(catf.TAU_FLOOR)
        self.rec_on = False
        self.path = self.sched = self.arm = None
        self.runs = []                      # one DataFrame per run
        self.checks = []                    # (label, ok, detail)
        self.cv_ref = None                  # CV result at the paper's order, for the stream
        self._folds = []                    # per fold: order, labels, schedule, events
        self._fold = None
        self._cap = []                      # (y_true, y_pred) and (y_true, score) per fold
        self._lm_cache = {}
        self._install()
        try:                                # catf_ids' exit report would sum all runs
            import atexit
            atexit.unregister(catf._band_report)
        except Exception:                   # noqa: BLE001
            pass

    # -- wrappers -------------------------------------------------------
    def _install(self):
        c, H = self.c, self

        o_hd = c.hierarchical_decision

        def hierarchical_decision(*a, **k):
            e = H._event()
            if e is None:
                return o_hd(*a, **k)
            tau = a[7] if len(a) > 7 else k.get("adaptive_tau")
            th, tl = float(tau.tau_high), float(tau.tau_low)
            out = o_hd(*a, **k)
            e.update({"score_used": float(a[4]), "tau_h": th, "tau_l": tl,
                      "gate": float(tau.tau_low), "dec_fusion": int(out[0]),
                      "kind": str(out[1]).split(":")[0]})
            return out
        c.hierarchical_decision = hierarchical_decision

        o_fsm = c.ThreatStateMachine.update

        def fsm_update(self_, *a, **k):
            lvl = o_fsm(self_, *a, **k)
            e = H._event()
            if e is not None:
                e["s_level"] = int(lvl)
            return lvl
        c.ThreatStateMachine.update = fsm_update

        o_band = c._band_hit                # band decision (in D the CV folds make it too)

        def band_hit(site, fs, R, lo, hi):
            r = o_band(site, fs, R, lo, hi)
            e = H._event()
            if e is not None:
                e["inband"] = int(r)
            return r
        c._band_hit = band_hit

        # event index: every scored event consults the experience cache exactly
        # once, after the warm-up pass (which consults it once per warm-up row)
        o_lk = c.ExperienceMemory.lookup

        def lookup(self_, *a, **k):
            f = H._fold
            if H.rec_on and H.path == "cv" and f is not None:
                f["lookups"] += 1
            return o_lk(self_, *a, **k)
        c.ExperienceMemory.lookup = lookup

        import sklearn.metrics as skm       # the pipeline's own fold decisions and scores
        o_f1, o_auc = skm.f1_score, skm.roc_auc_score

        def f1_rec(y_true, y_pred, *a, **k):
            if H.rec_on and H.path == "cv":
                H._cap.append(("pred", np.asarray(y_true).copy(), np.asarray(y_pred).copy()))
            return o_f1(y_true, y_pred, *a, **k)

        def auc_rec(y_true, y_score, *a, **k):
            if H.rec_on and H.path == "cv":
                H._cap.append(("score", np.asarray(y_true).copy(), np.asarray(y_score).copy()))
            return o_auc(y_true, y_score, *a, **k)
        skm.f1_score, skm.roc_auc_score = f1_rec, auc_rec

        o_ttr = c._tau_trace_report

        def tau_trace_report(trace, results, *a, **k):
            if H.rec_on and H.path == "stream":
                cols = [x for x in ("final_label", "pred_label", "type", "reason", "fusion_score")
                        if x in results.columns]
                H._stream_capture = (list(trace), results[cols].reset_index(drop=True).copy())
                raise _Stop()
            return o_ttr(trace, results, *a, **k)
        c._tau_trace_report = tau_trace_report

        o_fit = c.LayerModels.fit            # deterministic: fit once per training set

        def lm_fit(self_, df, *a, **k):
            cols = [x for x in ("final_label", "phys_delta", "net_bytes") if x in df.columns]
            key = (len(df), int(pd.util.hash_pandas_object(df[cols], index=False).sum()))
            if key in H._lm_cache:
                state, ret, first = H._lm_cache[key]
                self_.__dict__.update(state)
                return self_ if ret is first else ret
            ret = o_fit(self_, df, *a, **k)
            H._lm_cache[key] = (dict(self_.__dict__), ret, self_)
            return ret
        c.LayerModels.fit = lm_fit

        SK = c.StratifiedKFold

        class OrderedSKF(SK):
            def split(self_, X, y=None, groups=None):
                for k, (tr, te) in enumerate(SK.split(self_, X, y, groups), 1):
                    if not (H.rec_on and H.path == "cv"):
                        yield tr, te
                        continue
                    yl = np.asarray(y)[te].astype(int)
                    order, share, phase, since = schedule(H.sched, yl, seed=1000 + k)
                    H._begin_fold(k, yl[order], share, phase, since)
                    yield tr, te[order]
        c.StratifiedKFold = OrderedSKF

    @contextlib.contextmanager
    def frozen(self):
        """Both thresholds held at their starting point. The EWMA still advances
        (it is read elsewhere); nothing that moves a threshold runs."""
        AT, CT = self.c.AdaptiveThreshold, self.c.ContextualThresholdEngine
        o_up, o_get, o_fit = AT.update, CT.get_tau, AT.fit
        floor = self.floor

        def update(self_, fusion_score):
            if self_._ewma_score is None:
                self_._ewma_score = fusion_score
                return
            self_._ewma_score = (self_.ewma_alpha * fusion_score
                                 + (1 - self_.ewma_alpha) * self_._ewma_score)

        def get_tau(self_, *a, **k):
            if not hasattr(self_, "_frozen_pair"):
                h = max(float(self_.base._base_tau_high), floor)
                l = float(np.clip(self_.base.tau_low, 0.05, h - 0.12))
                self_._frozen_pair = (round(h, 3), round(l, 3))
            return self_._frozen_pair[0], self_._frozen_pair[1], "frozen"

        def fit(self_, *a, **k):
            r = o_fit(self_, *a, **k)
            self_.tau_high = max(float(self_.tau_high), floor)
            self_.tau_low = min(float(self_.tau_low), self_.tau_high - 0.12)
            return r

        AT.update, CT.get_tau, AT.fit = update, get_tau, fit
        try:
            yield
        finally:
            AT.update, CT.get_tau, AT.fit = o_up, o_get, o_fit

    # -- CV path --------------------------------------------------------
    def _begin_fold(self, k, y, share, phase, since):
        n = len(y)
        self._fold = {"k": k, "y": y, "share": share, "phase": phase, "since": since,
                      "n": n, "warmup": min(max(int(n * 0.3), 30), n), "lookups": 0,
                      "events": {}}
        self._folds.append(self._fold)

    def _event(self):
        """The record of the event being scored in the CV main pass, or None."""
        f = self._fold
        if not (self.rec_on and self.path == "cv" and f is not None):
            return None
        i = f["lookups"] - f["warmup"] - 1
        if i < 0 or i >= f["n"]:
            return None
        return f["events"].setdefault(i, {})

    def _cv_rows(self):
        preds = [(yt, yp) for kind, yt, yp in self._cap if kind == "pred"]
        scores = [(yt, s) for kind, yt, s in self._cap if kind == "score"]
        if len(preds) != len(self._folds) or len(scores) != len(self._folds):
            raise RuntimeError(f"captured {len(preds)} fold decisions and {len(scores)} fold "
                               f"scores for {len(self._folds)} folds")
        rows, aligned = [], True
        for f, (yt, yp), (_, sc) in zip(self._folds, preds, scores):
            aligned &= bool(np.array_equal(np.asarray(yt).astype(int), f["y"]))
            for i in range(f["n"]):
                e = f["events"].get(i, {})
                forced = int(yp[i] == 1 and "dec_fusion" not in e)
                rows.append((f["k"], i, int(f["y"][i]), float(f["share"][i]), f["phase"][i],
                             int(f["since"][i]), float(sc[i]), e.get("score_used", np.nan),
                             e.get("inband", 0),
                             e.get("tau_h", np.nan), e.get("tau_l", np.nan), e.get("gate", np.nan),
                             e.get("s_level", -1), e.get("kind", "FORCED" if forced else ""),
                             e.get("dec_fusion", np.nan), int(yp[i]), forced))
        return rows, aligned

    def run_cv(self, fused, sched, arm):
        self.path, self.sched, self.arm = "cv", sched, arm
        self._folds, self._fold, self._cap = [], None, []
        ctx = self.frozen() if arm == "frozen" else contextlib.nullcontext()
        with open(os.path.join(OUT, "logs", f"cv_{sched}_{arm}.log"), "w",
                  encoding="utf-8") as lg, ctx, contextlib.redirect_stdout(lg):
            self.rec_on = True
            try:
                res = self.c.run_cross_validation(fused, n_splits=5)
            finally:
                self.rec_on = False
                self._fold = None
        if sched == "original" and arm == "adaptive":
            self.cv_ref = res
        rows, aligned = self._cv_rows()
        df = self._frame(rows)
        self.runs.append(df)
        try:
            own = [round(float(fr.metrics["f1"]), 4) for fr in res[1]]
            mine = [round(confusion(g)["F1"], 4) for _, g in df.groupby("fold")]
            ok = own == mine and aligned
            self.checks.append((f"cv {sched}/{arm}: every event recorded, labels in the "
                                f"reordered order, fold F1 = pipeline's", ok, f"{mine} vs {own}"))
        except Exception as e:                                   # noqa: BLE001
            self.checks.append((f"cv {sched}/{arm}: fold comparison", False, repr(e)))
        return df

    def reference_cv(self, fused):
        """The CV at the paper's order, unrecorded (needed by the stream path)."""
        self.path, self.sched, self.arm = "cv_ref", "original", "adaptive"
        with open(os.path.join(OUT, "logs", "cv_reference.log"), "w",
                  encoding="utf-8") as lg, contextlib.redirect_stdout(lg):
            self.cv_ref = self.c.run_cross_validation(fused, n_splits=5)

    # -- stream path ----------------------------------------------------
    def run_stream(self, prep, sched, arm):
        c = self.c
        full = prep.full.reset_index(drop=True)
        y = full["final_label"].values.astype(int)
        order, share, phase, since = schedule(sched, y, seed=7)
        full_o = full.iloc[order].reset_index(drop=True)
        sentinel = pd.DataFrame({"_": [0]})
        names = ("load_multi", "preprocess_network", "preprocess_iot_multi",
                 "preprocess_linux_multi", "preprocess_windows", "fuse_datasets",
                 "run_cross_validation")
        saved = {n: getattr(c, n) for n in names}
        cv_ref = self.cv_ref
        c.load_multi = lambda *a, **k: sentinel
        for n in names[1:5]:
            setattr(c, n, lambda *a, **k: sentinel)
        c.fuse_datasets = lambda *a, **k: (prep.fused.copy(), full_o.copy())
        c.run_cross_validation = lambda *a, **k: _clone(cv_ref)
        self.path, self.sched, self.arm = "stream", sched, arm
        self._stream_capture = None
        ctx = self.frozen() if arm == "frozen" else contextlib.nullcontext()
        with open(os.path.join(OUT, "logs", f"stream_{sched}_{arm}.log"), "w",
                  encoding="utf-8") as lg, ctx, contextlib.redirect_stdout(lg):
            self.rec_on = True
            try:
                c.run_pipeline_multi(*prep.files, n_samples=N_SAMPLES)
            except _Stop:
                pass
            finally:
                self.rec_on = False
                for n, f in saved.items():
                    setattr(c, n, f)
        if self._stream_capture is None:
            raise RuntimeError(f"stream {sched}/{arm} ended before the per-event trace; "
                               f"see logs/stream_{sched}_{arm}.log")
        trace, res = self._stream_capture
        # tuple layout in run_pipeline_multi's _trace:
        # (i, fs, eff, inband, tau_h, tau_l, gate, lo_prev, ewma, base, D, std, s_level,
        #  kind, dec_fusion, dec)
        by_i = {int(t[0]): t for t in trace}
        reason = res["reason"].astype(str) if "reason" in res.columns else pd.Series([""] * len(res))
        fsc = res["fusion_score"].to_numpy(float) if "fusion_score" in res.columns else None
        rows, same = [], True
        for i in range(len(res)):
            dec = int(res["pred_label"].iloc[i])
            t = by_i.get(i)
            if t is not None:
                same &= int(t[15]) == dec
                rows.append((0, i, int(res["final_label"].iloc[i]), float(share[i]), phase[i],
                             int(since[i]), float(t[1]), float(t[2]), int(t[3]), float(t[4]),
                             float(t[5]), float(t[6]), int(t[12]), str(t[13]), int(t[14]),
                             dec, 0))
            else:                                  # forced alert: it skips the rest of the loop
                forced = int(reason.iloc[i].startswith("[FORCED_ACTION]"))
                rows.append((0, i, int(res["final_label"].iloc[i]), float(share[i]), phase[i],
                             int(since[i]), float(fsc[i]) if fsc is not None else np.nan, np.nan,
                             0, np.nan, np.nan, np.nan, -1, "FORCED" if forced else "", np.nan,
                             dec, forced))
        df = self._frame(rows)
        self.runs.append(df)
        n_forced = int(df["forced"].sum())
        ok = bool(same) and len(by_i) + n_forced == len(res)
        self.checks.append((f"stream {sched}/{arm}: {len(by_i):,} traced + {n_forced:,} forced "
                            f"= {len(res):,} events, decisions = pipeline's", ok, ""))
        return df

    def _frame(self, rows):
        df = pd.DataFrame(rows, columns=[
            "fold", "pos", "label", "share", "phase", "since", "score", "score_used",
            "inband", "tau_h", "tau_l", "gate", "s_level", "kind", "dec_fusion", "dec",
            "forced"])
        df.insert(0, "arm", self.arm)
        df.insert(0, "schedule", self.sched)
        df.insert(0, "path", self.path)
        return df


# ── reporting ─────────────────────────────────────────────────────────
def confusion(df):
    y, d = df["label"].values, df["dec"].values
    TP = int(((d == 1) & (y == 1)).sum()); FP = int(((d == 1) & (y == 0)).sum())
    FN = int(((d == 0) & (y == 1)).sum()); TN = int(((d == 0) & (y == 0)).sum())
    P = TP / (TP + FP) if TP + FP else 0.0
    R = TP / (TP + FN) if TP + FN else 0.0
    return {"n": len(df), "TP": TP, "FP": FP, "FN": FN, "TN": TN, "precision": P,
            "recall": R, "F1": 2 * P * R / (P + R) if P + R else 0.0,
            "FPR": FP / (FP + TN) if FP + TN else 0.0}


def thresholds(df, floor):
    esc = (df["dec"] == 1) & (df["dec_fusion"] == 0)
    forced = int(df["forced"].sum()) if "forced" in df.columns else 0
    df = df[df["tau_h"].notna()]           # forced alerts never reach the thresholds
    if not len(df):
        return {"forced": forced}
    q = lambda s, v: float(np.nanpercentile(s, v))
    return {"forced": forced,
            "tau_h_at_floor": float((np.abs(df["tau_h"] - floor) < 5e-4).mean()),
            "tau_h_below_floor": float((df["tau_h"] < floor - 5e-4).mean()),
            "tau_h_max": float(df["tau_h"].max()),
            "tau_l_p5": q(df["tau_l"], 5), "tau_l_p50": q(df["tau_l"], 50),
            "tau_l_p95": q(df["tau_l"], 95),
            "gate_p5": q(df["gate"], 5), "gate_p50": q(df["gate"], 50),
            "gate_p95": q(df["gate"], 95),
            "gate_le_0.08": float((df["gate"] <= 0.08).mean()),
            "escalations": int(esc.sum()),
            "escalated_attacks": int((esc & (df["label"] == 1)).sum()),
            "escalated_normals": int((esc & (df["label"] == 0)).sum()),
            "band_substituted": int(df["inband"].sum())}


def phases(df):
    yield "ALL", df
    for ph, g in df.groupby("phase", sort=False):
        if ph != "all":
            yield ph, g
    if df["phase"].isin(["attack-heavy", "normal-heavy"]).any():
        yield f"transition (first {EDGE} after a switch)", df[df["since"] < EDGE]
        yield "settled", df[df["since"] >= EDGE]


def report(ev, floor):
    from sklearn.metrics import roc_auc_score
    rows, lines = [], []
    for (path, sched, arm), g in ev.groupby(["path", "schedule", "arm"], sort=False):
        for ph, h in phases(g):
            r = {"path": path, "schedule": sched, "arm": arm, "phase": ph}
            r.update(confusion(h)); r.update(thresholds(h, floor))
            if path == "cv" and ph == "ALL":
                f1s = [confusion(x)["F1"] for _, x in h.groupby("fold")]
                r["mean_fold_F1"] = float(np.mean(f1s)); r["std_fold_F1"] = float(np.std(f1s))
            rows.append(r)
    S = pd.DataFrame(rows)
    S.to_csv(os.path.join(OUT, "stress_summary.csv"), index=False)

    W = lambda s="": lines.append(s)
    W(f"THRESHOLD STRESS TEST -- configuration {CONFIG.upper()}, real TON_IoT records reordered")
    W("adaptive = thresholds as in the paper; frozen = held at their starting point")
    W("")
    W("CHECKS")
    meta_path = os.path.join(OUT, "stress_checks.json")
    checks = json.load(open(meta_path)) if os.path.exists(meta_path) else []
    for label, ok, detail in checks:
        W(f"  [{'OK' if ok else 'FAIL'}] {label}" + (f"  {detail}" if detail and not ok else ""))
    cv = ev[ev["path"] == "cv"]
    for (sched, arm), g in cv.groupby(["schedule", "arm"], sort=False):
        aucs = [round(float(roc_auc_score(x["label"], x["score"])), 4)
                for _, x in g.groupby("fold")]
        ok = np.allclose(aucs, PAPER_FOLD_AUC, atol=6e-5)
        W(f"  [{'OK' if ok else 'FAIL'}] cv {sched}/{arm}: fold AUCs {aucs} "
          f"= paper {PAPER_FOLD_AUC}  (order does not change the models)")
    X = EXPECT[CONFIG]
    o = S[(S.path == "cv") & (S.schedule == "original") & (S.arm == "adaptive") & (S.phase == "ALL")]
    if len(o):
        m = float(o["mean_fold_F1"].iloc[0])
        got = {k: int(o[k].iloc[0]) for k in X["cv_pooled"]}
        ok = abs(m - X["cv_f1"]) < 6e-5 and got == X["cv_pooled"]
        W(f"  [{'OK' if ok else 'FAIL'}] cv original/adaptive: mean fold F1 {m:.4f}, pooled "
          f"{got} = configuration {CONFIG.upper()} {X['cv_f1']:.4f}, {X['cv_pooled']}")
    o = S[(S.path == "stream") & (S.schedule == "original") & (S.arm == "adaptive") & (S.phase == "ALL")]
    if len(o):
        got = {k: int(o[k].iloc[0]) for k in X["stream"]}
        W(f"  [{'OK' if got == X['stream'] else 'FAIL'}] stream original/adaptive: "
          f"{got} = configuration {CONFIG.upper()} {X['stream']}")

    W("")
    W("THRESHOLD BEHAVIOUR (adaptive arm; forced alerts never reach the thresholds)")
    W(f"  {'path':<7}{'schedule':<14}{'tau_h@floor':>12}{'<floor':>8}{'max':>7}"
      f"{'tau_l p5/50/95':>19}{'gate<=.08':>10}{'escal. atk/nrm':>16}{'band':>7}{'forced':>8}")
    for _, r in S[(S.arm == "adaptive") & (S.phase == "ALL")].iterrows():
        W(f"  {r.path:<7}{r.schedule:<14}{r.tau_h_at_floor:>12.1%}{r.tau_h_below_floor:>8.1%}"
          f"{r.tau_h_max:>7.3f}   {r.tau_l_p5:.3f}/{r.tau_l_p50:.3f}/{r.tau_l_p95:.3f}"
          f"{r['gate_le_0.08']:>10.1%}{r.escalated_attacks:>9,}/{r.escalated_normals:<6,}"
          f"{r.band_substituted:>7,}{int(r.get('forced', 0) or 0):>8,}")

    W("")
    W("EFFECT OF ADAPTATION (adaptive - frozen, same records in the same order)")
    W("  +TP = detections gained, +FP = false alarms added")
    W(f"  {'path':<7}{'schedule':<14}{'phase':<34}{'events':>8}{'dTP':>7}{'dFP':>7}"
      f"{'dF1':>9}{'dFPR':>9}   {'F1 adapt/frozen':>16}")
    key = ["path", "schedule", "phase"]
    A = S[S.arm == "adaptive"].set_index(key); F = S[S.arm == "frozen"].set_index(key)
    for idx in A.index:
        if idx not in F.index:
            continue
        a, f = A.loc[idx], F.loc[idx]
        W(f"  {idx[0]:<7}{idx[1]:<14}{idx[2]:<34}{int(a.n):>8,}{int(a.TP - f.TP):>+7,}"
          f"{int(a.FP - f.FP):>+7,}{a.F1 - f.F1:>+9.4f}{a.FPR - f.FPR:>+9.4f}"
          f"   {a.F1:.4f}/{f.F1:.4f}")
    txt = "\n".join(lines)
    with open(os.path.join(OUT, "stress_summary.txt"), "w", encoding="utf-8") as fh:
        fh.write(txt + "\n")
    print(txt)
    return S


def plot(ev, floor, n_show=600):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sched = next((s for s in ("waves_medium", "waves_short", "waves_long", "ramp")
                  if (ev["schedule"] == s).any()), None)
    if sched is None:
        return
    panels = [(p, "fold 1" if p == "cv" else "stream") for p in ("cv", "stream")
              if ((ev["path"] == p) & (ev["schedule"] == sched)).any()]
    fig, axes = plt.subplots(len(panels), 1, figsize=(10, 3.8 * len(panels)), squeeze=False)
    for ax, (path, name) in zip(axes[:, 0], panels):
        g = ev[(ev["path"] == path) & (ev["schedule"] == sched)]
        if path == "cv":
            g = g[g["fold"] == 1]
        for arm, ls in (("adaptive", "-"), ("frozen", "--")):
            h = g[g["arm"] == arm].sort_values("pos").head(n_show)
            if not len(h):
                continue
            if arm == "adaptive":
                ax.fill_between(h["pos"], 0, h["share"], step="post", color="#DCE6F1",
                                label="target attack share")
            ax.plot(h["pos"], h["tau_h"], ls, color="#1F4D78", lw=1.3,
                    label=f"tau_high ({arm})")
            ax.plot(h["pos"], h["gate"], ls, color="#C0504D", lw=1.1,
                    label=f"tau_low at the escalation gate ({arm})")
        ax.axhline(floor, color="#7F7F7F", lw=0.8, ls=":")
        ax.set_xlim(0, n_show); ax.set_ylim(0, 1)
        ax.set_ylabel("threshold / share")
        ax.set_title(f"{path} path, {name}: {sched} (first {n_show} events)", fontsize=10)
        ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3,
                  frameon=False)
    axes[-1, 0].set_xlabel("event position in the reordered stream")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"stress_trace.{ext}"), dpi=200)
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────
def main(argv=None, prep_override=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--path", choices=["cv", "stream", "both"], default="both")
    ap.add_argument("--replot", action="store_true")
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--config", choices=["c", "d"], default="d",
                    help="d = the final system (default); c = configuration C, as first reported")
    args = ap.parse_args(argv)
    global CONFIG, OUT
    CONFIG = args.config
    OUT = _OUT_OVERRIDE or os.path.join(HERE, "threshold_stress" + ("_D" if CONFIG == "d" else ""))
    os.makedirs(os.path.join(OUT, "logs"), exist_ok=True)
    ev_path = os.path.join(OUT, "stress_events.csv.gz")

    if args.replot:
        if not os.path.exists(ev_path):
            sys.exit(f"{ev_path} not found: run without --replot first")
        ev = pd.read_csv(ev_path)
        floor = json.load(open(os.path.join(OUT, "stress_meta.json")))["floor"]
        report(ev, floor); plot(ev, floor)
        return

    catf = load_catf(CONFIG)
    cwd = os.getcwd()
    os.chdir(OUT)                      # models/ and plots/ saved by the pipeline land here
    for d in ("models", "plots"):
        os.makedirs(d, exist_ok=True)
    try:
        H = Harness(catf)
        t0 = time.time()
        print("preparing the data (the paper's loading, preprocessing and fusion) ...", flush=True)
        if prep_override is not None:
            prep = prep_override(catf)
        else:
            with open(os.path.join(OUT, "logs", "prep.log"), "w", encoding="utf-8") as lg, \
                    contextlib.redirect_stdout(lg):
                prep = Prep(catf, args.base)
        print(f"  balanced set {len(prep.fused):,}, stream {len(prep.full):,} "
              f"({time.time()-t0:.0f}s)", flush=True)
        scheds = SCHEDULES_QUICK if args.quick else SCHEDULES_FULL
        paths = ["cv", "stream"] if args.path == "both" else [args.path]
        if "cv" in paths:
            for s in scheds:
                for arm in ARMS:
                    t = time.time()
                    df = H.run_cv(prep.fused, s, arm)
                    m = confusion(df)
                    print(f"  cv     {s:<13}{arm:<9} F1 {m['F1']:.4f}  TP {m['TP']:,}  "
                          f"FP {m['FP']:,}  ({time.time()-t:.0f}s)", flush=True)
        if "stream" in paths:
            if H.cv_ref is None:
                t = time.time()
                H.reference_cv(prep.fused)
                print(f"  cv at the paper's order, for the stream ({time.time()-t:.0f}s)",
                      flush=True)
            for s in scheds:
                for arm in ARMS:
                    t = time.time()
                    df = H.run_stream(prep, s, arm)
                    m = confusion(df)
                    print(f"  stream {s:<13}{arm:<9} F1 {m['F1']:.4f}  TP {m['TP']:,}  "
                          f"FP {m['FP']:,}  ({time.time()-t:.0f}s)", flush=True)
        ev = pd.concat(H.runs, ignore_index=True)
        checks = H.checks
        ck_path = os.path.join(OUT, "stress_checks.json")
        if args.path != "both" and os.path.exists(ev_path):   # keep the other path's results
            old = pd.read_csv(ev_path)
            ev = pd.concat([old[old["path"] != args.path], ev], ignore_index=True)
            if os.path.exists(ck_path):
                checks = [c for c in json.load(open(ck_path))
                          if not c[0].startswith(args.path + " ")] + checks
        ev.to_csv(ev_path, index=False)
        json.dump({"floor": H.floor}, open(os.path.join(OUT, "stress_meta.json"), "w"))
        json.dump(checks, open(ck_path, "w"))
        print("")
        report(ev, H.floor)
        plot(ev, H.floor)
        print(f"\nwritten to {OUT}  (total {time.time()-t0:.0f}s)")
    finally:
        os.chdir(cwd)


if __name__ == "__main__":
    main()
