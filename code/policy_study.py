#!/usr/bin/env python3
"""
policy_study.py -- can the policy engine (stage 5) contribute once its
conditions are written in the data's own units?   (configuration D otherwise)

A. SITE-CALIBRATED ENVELOPES, on the unaltered data
   The five fixed conditions fail on TON_IoT for reasons in the data (Sec. 8):
   R4's threshold is in degrees but most device signals are not temperatures,
   R2/R3 combine fields no record carries together, R1 is scored on another
   modality's record. Here each condition becomes an envelope in its own
   field's units, set the way a site would set it: from the NORMAL records of
   the training part only (normal by that modality's own label), at quantile q.
     host    page_faults_maj, mem_vgrow, cpu_usage, disk_write above the
             q-quantile of their nonzero normal values          (R1, R2, R3)
     device  per device: |change| above the q-quantile, or reading outside
             the [1-q, q] quantiles of that device's normal readings   (R4)
     net     source bytes above the q-quantile                          (R5)
   A record that breaks any envelope is forced to alert, exactly as the fixed
   conditions are. CV: refitted per fold on that fold's training part.
   Stream: fitted on the stream's training split.

B. INJECTED PHYSICAL VIOLATIONS (a pre-deployment rehearsal), stream path
   On a copy of the data, 5% of each device's normal readings, in episodes of
   5 consecutive readings, are pushed out of their normal behaviour:
     jump   + m x the device's normal range (1st-99th percentile)
     drift  ramping up to + m x the range across the episode
     stuck  held at the reading before the episode (a frozen sensor)
   for m in 0.5, 1, 2, 5; the same readings are chosen at every m. The
   derived features are recomputed with the pipeline's own formulas (checked
   on every device). Models, envelopes and thresholds are fitted on the clean
   data; only the scored stream carries the injections. No label is changed:
   the ground truth of an injected record is "carries an injected reading".

Arms (configuration D otherwise, each with its own cross-validation):
  off               D without forced alerts            (= final_run.py's D-envelopes)
  fixed             off + the paper's five conditions
  envelope q=0.999  the site-calibrated envelopes      (= D, the final system)
  envelope q=0.9999 the same at the stricter quantile

Usage (in the folder that holds catf_ids.py and threshold_stress.py):
  python policy_study.py --quick      A on both paths + B at m = 1 and 5
  python policy_study.py              A + B at m = 0.5, 1, 2, 5
  python policy_study.py --skip_cv    A on the stream path only
  python policy_study.py --replot     summary and figure again from saved events
  python policy_study.py --base "<folder with the Processed_*_dataset folders>"

Writes into ./policy_study_D/ (policy_study_configC.py kept the C study):
  policy_summary.txt    checks, A per arm and per envelope, B per magnitude
  policy_summary.csv    the same numbers, one row each
  policy_events.csv.gz  one row per event and run
  policy_envelopes.json the envelope values fitted for the stream (q = 0.999)
  policy_injection.png/pdf   B: injected records caught, by type and magnitude
  logs/                 the pipeline's own output for every run
"""
import argparse
import contextlib
import json
import os
import sys
import time
import zlib

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import threshold_stress as TS          # noqa: E402  (loader, file lists, metrics)

OUT = os.path.join(HERE, "policy_study_D")
_OUT_OVERRIDE = None                                     # tests only
HOST = ["page_faults_maj", "mem_vgrow", "cpu_usage", "disk_write"]
QS = (0.999, 0.9999)
ARMS = ["off", "fixed"] + [f"envelope_q{q}" for q in QS]
MAGS_FULL, MAGS_QUICK = (0.5, 1, 2, 5), (1, 5)
RATE, EP_LEN, AFTER = 0.05, 5, 5       # share injected, episode length, rows after it the rolling features still carry it
TYPES = {1: "jump", 2: "drift", 3: "stuck"}
TAGS = ["_dev", "_inj", "_aff"]
OWN = {"host": "log_label", "device": "iot_label", "net": "net_label"}


class _Stop(Exception):
    pass


class AlignmentError(RuntimeError):
    pass


# ── the derived device features, exactly as normalize_iot_device computes them ──
def _derived(ct):
    s = pd.Series(np.asarray(ct, dtype=float))
    sm = s.rolling(window=5, min_periods=1).mean()
    d = sm.diff().fillna(0)
    return {"phys_delta": d.to_numpy(), "phys_delta_abs": d.abs().to_numpy(),
            "temp_roll_mean": sm.to_numpy(),
            "temp_roll_std": s.rolling(5, min_periods=1).std().fillna(0).to_numpy()}


class Tagger:
    """Wraps normalize_iot_device: tags each reading with its device and, in
    inject mode, alters the chosen readings and recomputes the derived ones."""

    def __init__(self, catf):
        self.c, self.orig = catf, catf.normalize_iot_device
        self.mode, self.m = None, 0.0
        self.stats, self.recompute_ok = [], True
        catf.normalize_iot_device = self._wrap

    def run(self, raw_iot, mode, m=0.0):
        self.mode, self.m, self.stats = mode, float(m), []
        try:
            return self.c.preprocess_iot_multi(raw_iot)
        finally:
            self.mode = None

    def _wrap(self, df, source_name):
        out = self.orig(df, source_name)
        if self.mode is None:
            return out
        dev = next((k for k in self.c.IOT_DEVICE_MAP if k in str(source_name).lower()),
                   str(source_name))
        out = out.copy()
        for col, v in _derived(out["current_temp"]).items():       # self-check
            if not np.allclose(out[col].to_numpy(float), v, atol=1e-9):
                self.recompute_ok = False
        out["_dev"], out["_inj"], out["_aff"] = dev, 0, 0
        return self._inject(out, dev) if self.mode == "inject" else out

    def _inject(self, out, dev):
        rng = np.random.default_rng(zlib.crc32(dev.encode()))    # same readings at every m
        n = len(out)
        ok = (out["label"].to_numpy() == 0) & (out["iot_present"].to_numpy(float) == 1)
        ct0 = out["current_temp"].to_numpy(float)
        ct = ct0.copy()
        vals = ct0[ok]
        if len(vals) < 50:
            return out
        R = float(np.quantile(vals, 0.99) - np.quantile(vals, 0.01))
        if R <= 0:
            R = float(vals.std()) or 1.0
        span = EP_LEN + AFTER
        full_ok = np.convolve(ok.astype(int), np.ones(EP_LEN, int), "valid") == EP_LEN
        cand = [s for s in range(1, n - span) if full_ok[s]]
        rng.shuffle(cand)
        n_ep = int(round(RATE * ok.sum() / EP_LEN))
        taken, starts = np.zeros(n, bool), []
        for s in cand:
            if len(starts) >= n_ep:
                break
            if taken[s - 1: s + span + 1].any():
                continue
            starts.append(s)
            taken[s - 1: s + span + 1] = True
        starts.sort()
        inj, aff = np.zeros(n, int), np.zeros(n, int)
        for j, s in enumerate(starts):
            t, e = j % 3 + 1, np.arange(s, s + EP_LEN)
            if t == 1:
                ct[e] = ct0[e] + self.m * R
            elif t == 2:
                ct[e] = ct0[e] + self.m * R * np.arange(1, EP_LEN + 1) / EP_LEN
            else:
                ct[e] = ct0[s - 1]
            changed = e[ct[e] != ct0[e]]
            inj[changed] = t
            aff[s: s + span] = 1
        out["current_temp"] = ct
        for col, v in _derived(ct).items():
            out[col] = v
        out["_inj"], out["_aff"] = inj, aff
        self.stats.append((dev, len(starts), int((inj > 0).sum()), R))
        return out


# ── data ──────────────────────────────────────────────────────────────
class PData:
    """The paper's loading, preprocessing and fusion, done once, with every
    fused record tagged by its device; the pipeline itself never sees a tag."""

    def __init__(self, catf, tagger, base):
        self.c, self.tagger = catf, tagger
        self.files = TS.file_lists(base)
        missing = [f for grp in self.files for f in grp if not os.path.exists(f)]
        if missing:
            sys.exit(f"{len(missing)} data files not found, e.g. {missing[0]}\n"
                     f"Pass the dataset folder with --base.")
        net, iot, log, win = self.files
        npf = max(500, TS.N_SAMPLES // len(net))                 # as run_pipeline_multi
        raw_net = catf.load_multi(net, 'Network', n_per_file=npf)
        self.raw_iot = catf.load_multi(iot, 'IoT', n_per_file=npf)
        raw_log = catf.load_multi(log, 'Linux Log', n_per_file=npf)
        raw_win = catf.load_multi(win, 'Windows', n_per_file=npf)
        self.proc_net = catf.preprocess_network(raw_net)
        proc_iot = tagger.run(self.raw_iot, "tag")
        self.proc_log = catf.preprocess_linux_multi(raw_log)
        self.proc_win = catf.preprocess_windows(raw_win)
        self.fused_t, self.full_t = catf.fuse_datasets(self.proc_net, proc_iot,
                                                       self.proc_log, self.proc_win)
        self._injector = None

    @classmethod
    def from_frames(cls, fused_t, full_t, injector, files):
        self = cls.__new__(cls)
        self.fused_t, self.full_t, self._injector, self.files = fused_t, full_t, injector, files
        return self

    @property
    def fused(self):
        return self.fused_t.drop(columns=TAGS)

    def injected(self, m):
        if self._injector is not None:
            full_t = self._injector(m)
        else:
            proc_iot = self.tagger.run(self.raw_iot, "inject", m)
            _, full_t = self.c.fuse_datasets(self.proc_net, proc_iot, self.proc_log, self.proc_win)
        same = (len(full_t) == len(self.full_t)
                and np.array_equal(full_t["ts"].to_numpy(), self.full_t["ts"].to_numpy())
                and np.array_equal(full_t["final_label"].to_numpy(),
                                   self.full_t["final_label"].to_numpy())
                and np.array_equal(full_t["net_bytes"].to_numpy(),
                                   self.full_t["net_bytes"].to_numpy()))
        if not same:
            raise AlignmentError("the injected stream does not align with the clean one")
        return full_t


# ── the study ─────────────────────────────────────────────────────────
# Each arm is configuration D with the hard part of stage 5 set by catf's own
# switches; everything else (trust, policy limits, one decision path, frozen
# ensemble) stays as in D.
ARM_SET = {"off":              dict(SITE_ENVELOPES=False, NO_FORCED=True),
           "fixed":            dict(SITE_ENVELOPES=False, NO_FORCED=False),
           "envelope_q0.999":  dict(SITE_ENVELOPES=True, NO_FORCED=True, ENVELOPE_Q=0.999),
           "envelope_q0.9999": dict(SITE_ENVELOPES=True, NO_FORCED=True, ENVELOPE_Q=0.9999)}
EXPECT = {"envelope_q0.999": ("D", {"TP": 6415, "FP": 1133}, {"TP": 17649, "FP": 1172}),
          "off": ("D-envelopes", {"TP": 6414, "FP": 1068}, {"TP": 17653, "FP": 1112})}


class Study:
    def __init__(self, catf):
        self.c = catf
        self.rec, self.path, self.arm, self.scen, self.m = False, None, None, None, np.nan
        self.runs, self.checks, self.cv_ref = [], [], {}
        self._folds, self._fold, self._cap, self._lm_cache = [], None, [], {}
        self._install()
        try:
            import atexit
            atexit.unregister(catf._band_report)
        except Exception:                                         # noqa: BLE001
            pass

    def _install(self):
        c, S = self.c, self

        o_env = c.SiteCalibration.envelope_hits

        def envelope_hits(self_, df):
            out = o_env(self_, df)
            if S.rec and S.path == "cv" and S._fold is not None:
                S._fold["env"] = out
            return out
        c.SiteCalibration.envelope_hits = envelope_hits

        o_fp = c.check_forced_policies           # the paper's five fixed conditions

        def check_forced_policies(row):
            r = o_fp(row)
            if S.rec and S.path == "cv" and S._fold is not None:
                S._fold["fixed"].append(bool(r[0]))
            return r
        c.check_forced_policies = check_forced_policies

        import sklearn.metrics as skm            # the pipeline's own fold decisions
        o_f1 = skm.f1_score

        def f1_rec(y_true, y_pred, *a, **k):
            if S.rec and S.path == "cv":
                S._cap.append((np.asarray(y_true).copy(), np.asarray(y_pred).copy()))
            return o_f1(y_true, y_pred, *a, **k)
        skm.f1_score = f1_rec

        o_ttr = c._tau_trace_report

        def tau_trace_report(trace, results, *a, **k):
            if S.rec and S.path == "stream":
                cols = [x for x in ("final_label", "pred_label", "reason") if x in results.columns]
                S._capture = results[cols].reset_index(drop=True).copy()
                raise _Stop()
            return o_ttr(trace, results, *a, **k)
        c._tau_trace_report = tau_trace_report

        o_fit = c.LayerModels.fit

        def lm_fit(self_, df, *a, **k):
            cols = [x for x in ("final_label", "phys_delta", "net_bytes") if x in df.columns]
            key = (len(df), int(pd.util.hash_pandas_object(df[cols], index=False).sum()))
            if key in S._lm_cache:
                state, ret, first = S._lm_cache[key]
                self_.__dict__.update(state)
                return self_ if ret is first else ret
            ret = o_fit(self_, df, *a, **k)
            S._lm_cache[key] = (dict(self_.__dict__), ret, self_)
            return ret
        c.LayerModels.fit = lm_fit

        SK = c.StratifiedKFold

        class FoldHook(SK):
            def split(self_, X, y=None, groups=None):
                for k, (tr, te) in enumerate(SK.split(self_, X, y, groups), 1):
                    if S.rec and S.path == "cv":
                        S._fold = {"k": k, "te": np.asarray(te), "env": None, "fixed": []}
                        S._folds.append(S._fold)
                    yield tr, te
        c.StratifiedKFold = FoldHook

    @contextlib.contextmanager
    def arm_ctx(self, arm):
        c = self.c
        saved = {k: getattr(c, k) for k in ("SITE_ENVELOPES", "NO_FORCED", "ENVELOPE_Q")}
        self.arm = arm
        for k, v in ARM_SET[arm].items():
            setattr(c, k, v)
        try:
            yield
        finally:
            for k, v in saved.items():
                setattr(c, k, v)

    # -- CV path --------------------------------------------------------
    def run_cv(self, P, arm, record=True):
        self.path, self.scen, self.m = ("cv" if record else "cv_ref"), "clean", np.nan
        self._folds, self._fold, self._cap = [], None, []
        tag = "cv" if record else "cv_reference"
        with open(os.path.join(OUT, "logs", f"{tag}_{arm}.log"), "w", encoding="utf-8") as lg, \
                self.arm_ctx(arm), contextlib.redirect_stdout(lg):
            self.rec = record
            try:
                res = self.c.run_cross_validation(P.fused, n_splits=5)
            finally:
                self.rec, self._fold = False, None
        self.cv_ref[arm] = res                     # each arm's stream starts from its own CV
        if not record:
            return None
        A = ARM_SET[arm]
        if len(self._cap) != len(self._folds):
            raise RuntimeError(f"{len(self._cap)} fold decisions for {len(self._folds)} folds")
        rows, aligned, complete = [], True, True
        lab = P.fused_t["final_label"].to_numpy().astype(int)
        for f, (yt, yp) in zip(self._folds, self._cap):
            n = len(f["te"])
            aligned &= bool(np.array_equal(np.asarray(yt).astype(int), lab[f["te"]]))
            complete &= len(f["fixed"]) == n
            fixed = f["fixed"] + [False] * (n - len(f["fixed"]))
            hits, rules = (f["env"] if f["env"] is not None
                           else (np.zeros(n, bool), np.full(n, "", dtype=object)))
            for i in range(n):
                fx = fixed[i] and not A["NO_FORCED"]
                ev = A["SITE_ENVELOPES"] and bool(hits[i])
                rows.append((f["k"], i, int(f["te"][i]), int(yp[i]), int(fx or ev),
                             str(rules[i]) if ev else ("fixed" if fx else "")))
        df = pd.DataFrame(rows, columns=["fold", "pos", "row", "dec", "forced", "rule"])
        df = self._label(df, P.fused_t)
        self.runs.append(df)
        try:
            own = [round(float(fr.metrics["f1"]), 4) for fr in res[1]]
            mine = [round(TS.confusion(g)["F1"], 4) for _, g in df.groupby("fold")]
            ok = own == mine and aligned and complete
            self.checks.append((f"cv {arm}: every event recorded in fold order, fold F1 = "
                                f"pipeline's", ok, f"{mine} vs {own}"))
        except Exception as e:                                   # noqa: BLE001
            self.checks.append((f"cv {arm}: fold comparison", False, repr(e)))
        return df

    # -- stream path ----------------------------------------------------
    def run_stream(self, P, arm, full_t, scen="clean", m=np.nan):
        c = self.c
        full_t = full_t.reset_index(drop=True)
        full_pipe = full_t.drop(columns=TAGS)
        sentinel = pd.DataFrame({"_": [0]})
        names = ("load_multi", "preprocess_network", "preprocess_iot_multi",
                 "preprocess_linux_multi", "preprocess_windows", "fuse_datasets",
                 "run_cross_validation")
        saved = {n: getattr(c, n) for n in names}
        fused_pipe, cv_ref = P.fused, self.cv_ref[arm]
        for n in names[:5]:
            setattr(c, n, lambda *a, **k: sentinel)
        c.fuse_datasets = lambda *a, **k: (fused_pipe.copy(), full_pipe.copy())
        c.run_cross_validation = lambda *a, **k: TS._clone(cv_ref)
        self.path, self.scen, self.m, self._capture = "stream", scen, m, None
        tag = scen if scen == "clean" else f"{scen}_m{m:g}"
        with open(os.path.join(OUT, "logs", f"stream_{tag}_{arm}.log"), "w",
                  encoding="utf-8") as lg, self.arm_ctx(arm), contextlib.redirect_stdout(lg):
            self.rec = True
            try:
                c.run_pipeline_multi(*P.files, n_samples=TS.N_SAMPLES)
            except _Stop:
                pass
            finally:
                self.rec = False
                for n, f in saved.items():
                    setattr(c, n, f)
        if self._capture is None:
            raise RuntimeError(f"stream {tag}/{arm} ended early; see its log")
        res = self._capture
        reason = res["reason"].astype(str)
        forced = reason.str.startswith("[FORCED_ACTION]").to_numpy()
        env_rule = reason.str.extract(r"FORCED: envelope (\S+)")[0]
        fix_rule = reason.str.extract(r"FORCED: ([a-z_]+)")[0]
        rule = env_rule.fillna(fix_rule).fillna("forced").to_numpy(dtype=object)
        df = pd.DataFrame({"fold": 0, "pos": np.arange(len(res)), "row": np.arange(len(res)),
                           "dec": res["pred_label"].astype(int).to_numpy(),
                           "forced": forced.astype(int),
                           "rule": np.where(forced, rule, "")})
        df = self._label(df, full_t)
        self.runs.append(df)
        ok = (len(res) == len(full_t) and np.array_equal(
            res["final_label"].to_numpy().astype(int), full_t["final_label"].to_numpy().astype(int)))
        self.checks.append((f"stream {tag}/{arm}: {len(res):,} events decided, in the stream's "
                            f"order ({int(forced.sum()):,} forced)", bool(ok), ""))
        return df

    def _label(self, df, frame):
        cols = ["final_label", "iot_label", "log_label", "net_label", "_dev", "_inj", "_aff"]
        cols += [x for x in ("type",) if x in frame.columns]
        lab = frame.iloc[df["row"].to_numpy()][cols].reset_index(drop=True)
        df = pd.concat([df.reset_index(drop=True), lab], axis=1).rename(columns={"final_label": "label"})
        df.insert(0, "m", self.m)
        df.insert(0, "scenario", self.scen)
        df.insert(0, "arm", self.arm)
        df.insert(0, "path", "cv" if self.path == "cv" else "stream")
        return df

# ── reporting ─────────────────────────────────────────────────────────
def _family(rule):
    return str(rule).split(":")[0] if rule else ""


def report(ev):
    lines, rows = [], []
    W = lambda s="": lines.append(s)
    ck = os.path.join(OUT, "policy_checks.json")
    W("POLICY ENGINE STUDY -- configuration D, envelopes set from normal training records")
    W("")
    W("CHECKS")
    for label, ok, detail in (json.load(open(ck)) if os.path.exists(ck) else []):
        W(f"  [{'OK' if ok else 'FAIL'}] {label}" + (f"  {detail}" if detail and not ok else ""))
    clean = ev[ev["scenario"] == "clean"]
    for arm, (name, want_cv, want_st) in EXPECT.items():
        for path, want in (("cv", want_cv), ("stream", want_st)):
            g = clean[(clean.path == path) & (clean.arm == arm)]
            if len(g):
                got = {k: TS.confusion(g)[k] for k in want}
                W(f"  [{'OK' if got == want else 'FAIL'}] {path} {arm}: {got} = "
                  f"final_run.py's {name} {want}")

    W("")
    W("A. ON THE UNALTERED DATA (configuration D with each form of forced alert)")
    W(f"  {'path':<7}{'arm':<17}{'F1':>8}{'FPR':>8}{'TP':>8}{'FP':>7}{'forced':>8}"
      f"{'on normal':>10}{'+TP':>7}{'+FP':>7}")
    for path in ("cv", "stream"):
        base = clean[(clean.path == path) & (clean.arm == "off")].set_index("row")
        for arm in ARMS:
            g = clean[(clean.path == path) & (clean.arm == arm)]
            if not len(g):
                continue
            m = TS.confusion(g)
            b = TS.confusion(base) if len(base) else m
            f = g[g.forced == 1]
            W(f"  {path:<7}{arm:<17}{m['F1']:>8.4f}{m['FPR']:>8.4f}{m['TP']:>8,}{m['FP']:>7,}"
              f"{len(f):>8,}{int((f.label == 0).sum()):>10,}{m['TP'] - b['TP']:>+7,}"
              f"{m['FP'] - b['FP']:>+7,}")
            rows.append({"part": "A", "path": path, "arm": arm, **m, "forced": len(f),
                         "forced_on_normal": int((f.label == 0).sum()),
                         "dTP": m["TP"] - b["TP"], "dFP": m["FP"] - b["FP"]})
    W("")
    W("  Per envelope (stream, q = 0.999 and 0.9999): the record's own-modality label vs")
    W("  the consensus label it is scored on; +TP/+FP = alerts the off arm did not raise")
    W(f"  {'arm':<17}{'envelope':<30}{'fires':>7}{'own atk':>9}{'cons atk':>9}{'+TP':>6}{'+FP':>6}")
    for arm in ARMS[2:]:
        g = clean[(clean.path == "stream") & (clean.arm == arm)]
        base = clean[(clean.path == "stream") & (clean.arm == "off")].set_index("row")["dec"]
        if not len(g):
            continue
        g = g.assign(dec_off=base.reindex(g["row"]).to_numpy())
        for rule, h in g[g.forced == 1].groupby("rule"):
            own = OWN.get(_family(rule))
            new = h[h.dec_off == 0]
            W(f"  {arm:<17}{rule:<30}{len(h):>7,}"
              f"{(h[own] == 1).mean() if own else np.nan:>9.1%}{(h.label == 1).mean():>9.1%}"
              f"{int((new.label == 1).sum()):>6,}{int((new.label == 0).sum()):>6,}")
            rows.append({"part": "A-rule", "path": "stream", "arm": arm, "rule": rule,
                         "fires": len(h), "own_attack": float((h[own] == 1).mean()) if own else None,
                         "consensus_attack": float((h.label == 1).mean()),
                         "dTP": int((new.label == 1).sum()), "dFP": int((new.label == 0).sum())})

    inj = ev[ev["scenario"] == "inject"]
    if len(inj):
        W("")
        W("B. INJECTED PHYSICAL VIOLATIONS (stream; models, envelopes, thresholds fitted clean)")
        W("  caught = flagged in the injected run but not in the clean run, among injected")
        W("  records the consensus label calls normal (nothing else would flag them)")
        cl = clean[clean.path == "stream"]
        W(f"  {'m':>4} {'type':<7}{'records':>9}{'normal':>8}   "
          + "".join(f"{a:>17}" for a in ARMS))
        for m in sorted(inj["m"].unique()):
            for t, name in TYPES.items():
                cells, n_all, n_nrm = [], 0, 0
                for arm in ARMS:
                    g = inj[(inj.m == m) & (inj.arm == arm)]
                    if not len(g):
                        cells.append(f"{'-':>17}")
                        continue
                    h = g[(g["_inj"] == t)]
                    hn = h[h.label == 0]
                    c0 = cl[cl.arm == arm].set_index("row")["dec"].reindex(hn["row"]).to_numpy()
                    caught = int(((hn["dec"].to_numpy() == 1) & (c0 == 0)).sum())
                    n_all, n_nrm = len(h), len(hn)
                    cells.append(f"{caught:>8,} ({caught / max(len(hn), 1):>5.1%})")
                    rows.append({"part": "B", "m": m, "type": name, "arm": arm,
                                 "records": len(h), "normal": len(hn), "caught": caught})
                W(f"  {m:>4g} {name:<7}{n_all:>9,}{n_nrm:>8,}   " + "".join(cells))
        W("")
        W("  False alarms on untouched normal records (not within an injected episode or")
        W("  the 5 readings after it): injected run vs the clean run, same records")
        for m in sorted(inj["m"].unique()):
            cells = []
            for arm in ARMS:
                g = inj[(inj.m == m) & (inj.arm == arm)]
                if not len(g):
                    continue
                u = g[(g["_aff"] == 0) & (g.label == 0)]
                c0 = cl[cl.arm == arm].set_index("row")["dec"].reindex(u["row"]).to_numpy()
                cells.append(f"{arm} {int((u['dec'] == 1).sum()):,} vs {int((c0 == 1).sum()):,}")
            W(f"  m={m:g}: " + "   ".join(cells))
    txt = "\n".join(lines)
    with open(os.path.join(OUT, "policy_summary.txt"), "w", encoding="utf-8") as fh:
        fh.write(txt + "\n")
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "policy_summary.csv"), index=False)
    print(txt)
    return rows


def plot(rows):
    B = pd.DataFrame([r for r in rows if r.get("part") == "B"])
    if not len(B):
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    style = {"off": ("#7F7F7F", "o", "off (C)"), "fixed": ("#C0504D", "s", "fixed conditions"),
             f"envelope_q{QS[0]}": ("#1F4D78", "^", f"envelope q={QS[0]}"),
             f"envelope_q{QS[1]}": ("#4F81BD", "v", f"envelope q={QS[1]}")}
    for ax, name in zip(axes, TYPES.values()):
        for arm, (col, mk, lab) in style.items():
            h = B[(B.type == name) & (B.arm == arm)].sort_values("m")
            if len(h):
                ax.plot(h.m, h.caught / h.normal.clip(lower=1), marker=mk, color=col, label=lab,
                        ls="--" if arm == "off" else "-", ms=7 if arm == "off" else 5, mfc="none" if arm == "off" else col)
        ax.set_title(name, fontsize=10)
        ax.set_xscale("log")
        ax.minorticks_off()
        ax.set_xticks(sorted(B.m.unique()))
        ax.set_xticklabels([f"{v:g}" for v in sorted(B.m.unique())])
        ax.set_xlabel("size (x the device's normal range)")
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("injected records caught")
    axes[-1].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"policy_injection.{ext}"), dpi=200)
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────
def main(argv=None, data_override=None):
    ap = argparse.ArgumentParser(description="policy engine study (A calibrated, B injected)")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip_cv", action="store_true")
    ap.add_argument("--replot", action="store_true")
    ap.add_argument("--base", default=TS.DEFAULT_BASE)
    args = ap.parse_args(argv)
    global OUT
    OUT = _OUT_OVERRIDE or OUT
    os.makedirs(os.path.join(OUT, "logs"), exist_ok=True)
    ev_path = os.path.join(OUT, "policy_events.csv.gz")
    if args.replot:
        if not os.path.exists(ev_path):
            sys.exit(f"{ev_path} not found: run without --replot first")
        plot(report(pd.read_csv(ev_path)))
        return

    catf = TS.load_catf("d")
    cwd = os.getcwd()
    os.chdir(OUT)                   # models/ and plots/ saved by the pipeline land here
    for d in ("models", "plots"):
        os.makedirs(d, exist_ok=True)
    try:
        S = Study(catf)
        tagger = Tagger(catf)
        t0 = time.time()
        print("preparing the data (the paper's loading, preprocessing and fusion) ...", flush=True)
        if data_override is not None:
            P = data_override(catf)
        else:
            with open(os.path.join(OUT, "logs", "prep.log"), "w", encoding="utf-8") as lg, \
                    contextlib.redirect_stdout(lg):
                P = PData(catf, tagger, args.base)
            S.checks.append(("device features recomputed with the pipeline's formulas "
                             "reproduce its own, on every device", tagger.recompute_ok, ""))
        print(f"  balanced set {len(P.fused_t):,}, stream {len(P.full_t):,} "
              f"({time.time()-t0:.0f}s)", flush=True)

        def say(tag, arm, df, t):
            mm = TS.confusion(df)
            print(f"  {tag:<22}{arm:<17} F1 {mm['F1']:.4f}  TP {mm['TP']:,}  FP {mm['FP']:,}  "
                  f"forced {int(df.forced.sum()):,}  ({time.time()-t:.0f}s)", flush=True)
        if not args.skip_cv:
            for arm in ARMS:
                t = time.time()
                say("A cv", arm, S.run_cv(P, arm), t)
        for arm in ARMS:                      # each arm's stream starts from its own CV
            if arm not in S.cv_ref:
                t = time.time()
                S.run_cv(P, arm, record=False)
                print(f"  cv reference for the stream, {arm} ({time.time()-t:.0f}s)", flush=True)
        for arm in ARMS:
            t = time.time()
            say("A stream", arm, S.run_stream(P, arm, P.full_t), t)
        for m in (MAGS_QUICK if args.quick else MAGS_FULL):
            t = time.time()
            with open(os.path.join(OUT, "logs", f"inject_m{m:g}.log"), "w",
                      encoding="utf-8") as lg, contextlib.redirect_stdout(lg):
                full_inj = P.injected(m)
            n_inj = int((full_inj["_inj"] > 0).sum())
            print(f"  injected m={m:g}: {n_inj:,} stream records carry an altered reading "
                  f"({time.time()-t:.0f}s)", flush=True)
            for arm in ARMS:
                t = time.time()
                say(f"B stream m={m:g}", arm, S.run_stream(P, arm, full_inj, "inject", m), t)
        ev = pd.concat(S.runs, ignore_index=True)
        ev.to_csv(ev_path, index=False)
        json.dump(S.checks, open(os.path.join(OUT, "policy_checks.json"), "w"))
        print("")
        plot(report(ev))
        print(f"\nwritten to {OUT}  (total {time.time()-t0:.0f}s)")
    finally:
        os.chdir(cwd)


if __name__ == "__main__":
    main()
