#!/usr/bin/env python3
"""
export_processed_data.py -- write the processed, fused dataset the paper evaluates.

It runs exactly the paper's data preparation (catf_ids.py's loaders, per-modality
preprocessing and timestamp fusion, 2,000 records per network file) and saves:

  data/processed/catf_fused_stream.csv.gz    the full fused stream (24,934 records)
  data/processed/catf_fused_balanced.csv.gz  the class-balanced training set (13,256)
  data/processed/manifest.json               row and class counts, column groups,
                                             SHA-256 checksums, package versions

Nothing is trained. Expect a few minutes.

    python export_processed_data.py --base "<folder with the Processed_*_dataset folders>"
"""
import argparse
import contextlib
import datetime
import hashlib
import json
import os
import platform
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import threshold_stress as TS                              # noqa: E402  (the paper's Prep)

OUT = os.path.join(HERE, "..", "data", "processed")
EXPECT = {"stream": (24934, 18306, 6628), "balanced": (13256, 6628, 6628)}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def counts(df):
    y = df["final_label"].astype(int)
    return len(df), int(y.sum()), int((y == 0).sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--base", default=TS.DEFAULT_BASE,
                    help="folder holding the Processed_*_dataset folders of TON_IoT")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    catf = TS.load_catf("d")
    print("preparing the data (the paper's loading, preprocessing and fusion) ...", flush=True)
    log = os.path.join(OUT, "export_prep.log")
    with open(log, "w", encoding="utf-8") as lg, contextlib.redirect_stdout(lg):
        prep = TS.Prep(catf, args.base)

    frames = {"stream": prep.full, "balanced": prep.fused}
    files, summary, ok = {}, {}, True
    for name, df in frames.items():
        n, atk, nrm = counts(df)
        match = (n, atk, nrm) == EXPECT[name]
        ok &= match
        summary[name] = {"records": n, "attack": atk, "normal": nrm,
                         "matches_paper": match, "paper": dict(zip(("records", "attack", "normal"), EXPECT[name]))}
        path = os.path.join(OUT, f"catf_fused_{name}.csv.gz")
        df.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})
        files[os.path.basename(path)] = {"sha256": sha256(path), "bytes": os.path.getsize(path)}
        print(f"  {name:<9} {n:>6,} records  attack {atk:,}  normal {nrm:,}  "
              f"{'matches the paper' if match else 'DOES NOT MATCH the paper ' + str(EXPECT[name])}")

    lm = catf.LayerModels
    used = set(lm.NET_FEATURES) | set(lm.IOT_FEATURES) | set(lm.LOG_FEATURES)
    labels = [c for c in ("net_label", "iot_label", "log_label", "win_label",
                          "fusion_vote", "final_label") if c in prep.full.columns]
    cols = list(prep.full.columns)
    manifest = {
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "source": "TON_IoT (UNSW Canberra), Processed_datasets; see data/README.md for terms of use",
        "preparation": {"records_per_network_file": max(500, TS.N_SAMPLES // 23),
                        "join": "pandas.merge_asof on ts, network as base, direction='nearest', "
                                "tolerance 300 s; records without all partners dropped",
                        "fused_label": "vote net 0.45, iot 0.20, linux 0.20, windows 0.15; attack if >= 0.45",
                        "balanced_set": "random equal-size sample per class (random_state=42), sorted by ts"},
        "counts": summary,
        "files": files,
        "column_groups": {
            "network_features": lm.NET_FEATURES,
            "device_features": lm.IOT_FEATURES,
            "linux_features": lm.LOG_FEATURES,
            "labels": labels,
            "row_id_and_time": [c for c in ("_rowid", "ts") if c in cols],
            "other_columns_not_used_by_the_models": [c for c in cols if c not in used
                                                     and c not in labels and c not in ("_rowid", "ts")],
        },
        "versions": {"python": platform.python_version(),
                     **{m: __import__(m).__version__ for m in ("numpy", "pandas", "sklearn", "joblib")}},
    }
    with open(os.path.join(OUT, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    print(f"\nwritten to {os.path.abspath(OUT)}  (preparation log: export_prep.log)")
    if not ok:
        print("WARNING: counts differ from the paper; check the dataset folder and package versions.")
        sys.exit(1)


if __name__ == "__main__":
    main()
