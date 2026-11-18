"""exp7: does case-mix (resident acuity) add signal? y_g12, gap-h splits, 133 gold features.

Case-mix columns are dropped before silver (column_map `ignored`). Here they are read straight from bronze
provider_info at vintage == t (point-in-time, same vintage the gold provider features use; checked below by
matching bronze reported total HPRD against gold hprd_total).

Nursing Case-Mix Index / Ratio only exist from 2024-07 (after the train window), so the experiment uses the
case-mix staffing hours available since 2021-01 (expected HPRD given acuity = national HPRD x case-mix ratio).
Added features:
  cm_hprd_total, cm_hprd_rn, cm_hprd_lpn, cm_hprd_aide        (acuity / expected staffing)
  staff_vs_cm_total, staff_vs_cm_rn                           (reported / expected: staffing gap)
Models: LightGBM, same params, 3 seeds, early stopping on calib PR-AUC; base vs base + case-mix.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import lightgbm as lgb
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp_common as E  # noqa: E402

MIN_FREE_GB = 0.8
SEEDS = (42, 7, 2026)
PARAMS = {"objective": "binary", "metric": "average_precision", "learning_rate": 0.05, "num_leaves": 31,
          "min_child_samples": 100, "feature_fraction": 0.8, "num_threads": 4, "verbosity": -1}
RAW = {
    "cm_hprd_aide": "Case-Mix Nurse Aide Staffing Hours per Resident per Day",
    "cm_hprd_lpn": "Case-Mix LPN Staffing Hours per Resident per Day",
    "cm_hprd_rn": "Case-Mix RN Staffing Hours per Resident per Day",
    "cm_hprd_total": "Case-Mix Total Nurse Staffing Hours per Resident per Day",
}
CCN_RAW = ("Federal Provider Number", "CMS Certification Number (CCN)")
REPORTED_TOTAL = "Reported Total Nurse Staffing Hours per Resident per Day"


def guard(where):
    f = E.free_ram_gb()
    print(f"[ram] {where}: {f:.2f} GB free", flush=True)
    if f < MIN_FREE_GB:
        sys.exit(f"abort: free RAM {f:.2f} GB < {MIN_FREE_GB} GB at {where}")


def load_casemix(months):
    parts = []
    for v in sorted(months):
        f = E.ROOT / "data" / "bronze" / "provider_info" / f"vintage={v}" / "part-0.parquet"
        names = pq.read_schema(f).names
        ccn = next(c for c in CCN_RAW if c in names)
        cols = [ccn, *RAW.values()] + ([REPORTED_TOTAL] if REPORTED_TOTAL in names else [])
        d = pd.read_parquet(f, columns=cols).rename(columns={ccn: "ccn", **{r: k for k, r in RAW.items()}})
        d["ccn"] = d["ccn"].astype(str).str.strip().str.zfill(6)
        for c in [*RAW, REPORTED_TOTAL]:
            if c in d:
                d[c] = pd.to_numeric(d[c], errors="coerce").astype("float32")
        d["t"] = v
        parts.append(d.drop_duplicates("ccn"))
    return pd.concat(parts, ignore_index=True)


def fit_predict(Xtr, ytr, Xca, yca, Xte, cat, seed):
    dtr = lgb.Dataset(Xtr, ytr, categorical_feature=cat)
    dca = lgb.Dataset(Xca, yca, reference=dtr, categorical_feature=cat)
    bst = lgb.train({**PARAMS, "seed": seed}, dtr, 3000, valid_sets=[dca],
                    callbacks=[lgb.early_stopping(100, first_metric_only=True, verbose=False)])
    return bst.predict(Xte, num_iteration=bst.best_iteration), bst


def boot_diff(y, p1, p2, groups, n=500, seed=0):
    """Facility-clustered paired bootstrap of PR-AUC(p2) - PR-AUC(p1)."""
    rng = np.random.default_rng(seed)
    codes, uniq = pd.factorize(groups)
    rows = [np.flatnonzero(codes == i) for i in range(len(uniq))]
    d = []
    for _ in range(n):
        s = np.concatenate([rows[i] for i in rng.integers(0, len(uniq), len(uniq))])
        d.append(average_precision_score(y[s], p2[s]) - average_precision_score(y[s], p1[s]))
    return float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    t0 = time.time()
    guard("start")
    df, num, cat = E.load("y_g12")
    cm = load_casemix(df["t"].unique())
    n0 = len(df)
    df = df.merge(cm, on=["ccn", "t"], how="left")
    assert len(df) == n0
    # point-in-time check: bronze vintage-t reported total HPRD should equal gold hprd_total
    if REPORTED_TOTAL in df and "hprd_total" in df:
        both = df[[REPORTED_TOTAL, "hprd_total"]].dropna()
        match = float((np.abs(both[REPORTED_TOTAL] - both["hprd_total"]) < 1e-3).mean())
        print(f"vintage alignment: bronze reported HPRD == gold hprd_total on {match:.1%} of {len(both):,} rows")
        assert match > 0.95, "case-mix not aligned to the gold vintage"
        df = df.drop(columns=[REPORTED_TOTAL])
    df["staff_vs_cm_total"] = (df["hprd_total"] / df["cm_hprd_total"]).astype("float32")
    df["staff_vs_cm_rn"] = (df["hprd_rn"] / df["cm_hprd_rn"]).astype("float32")
    new = [*RAW, "staff_vs_cm_total", "staff_vs_cm_rn"]
    df[new] = df[new].replace([np.inf, -np.inf], np.nan)
    guard("loaded")

    tr, ca, te = E.split_frames(df)
    cov = {k: round(float(v["cm_hprd_total"].notna().mean()), 4) for k, v in
           {"train": tr, "calib": ca, "test": te}.items()}
    print("case-mix coverage:", cov, flush=True)
    # single-feature signal on train (sanity)
    sig = {}
    for c in new:
        x = tr[c].fillna(tr[c].median())
        sig[c] = round(max(average_precision_score(tr.y, x), average_precision_score(tr.y, -x)), 4)
    print("single-feature train PR-AUC (base rate %.3f):" % tr.y.mean(), sig, flush=True)

    base_f, cm_f = num + cat, num + new + cat
    P = {"base": [], "casemix": []}
    gain = {}
    for seed in SEEDS:
        p, _ = fit_predict(tr[base_f], tr.y, ca[base_f], ca.y, te[base_f], cat, seed)
        P["base"].append(p)
        p, bst = fit_predict(tr[cm_f], tr.y, ca[cm_f], ca.y, te[cm_f], cat, seed)
        P["casemix"].append(p)
        imp = pd.Series(bst.feature_importance("gain"), index=bst.feature_name())
        rank = imp.rank(ascending=False)
        gain[seed] = {c: int(rank[c]) for c in new}
        guard(f"seed {seed}")

    y = te.y.values
    ap = {k: [average_precision_score(y, p) for p in v] for k, v in P.items()}
    mean = {k: np.mean(v, axis=0) for k, v in P.items()}
    rep = {
        "coverage": cov,
        "single_feature_train_prauc": sig,
        "test_prauc_mean_std": {k: (float(np.mean(v)), float(np.std(v))) for k, v in ap.items()},
        "test_metrics_seedmean": {k: E.metrics(y, p) for k, p in mean.items()},
        "boot_casemix_minus_base_facility_clustered": boot_diff(y, mean["base"], mean["casemix"], te.ccn.values),
        "gain_rank_of_new_features_out_of_%d" % len(cm_f): gain,
    }
    for k, p in mean.items():
        E.save_result("exp7", k, {}, E.metrics(y, p), {"seeds": list(SEEDS), "params": PARAMS, "new": new})
    out = E.RESULTS / "exp7__summary.json"
    out.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps(rep, indent=2))
    print(f"done in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
