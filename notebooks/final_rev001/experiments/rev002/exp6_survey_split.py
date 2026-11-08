"""exp6: survey x harm split test (y_g12, gap-h splits, 133 gold features).

surveyed = any HEALTH survey (standard/complaint/infection control, from ledger_surveys) or any citation
survey date (ledger_citations) strictly after cutoff_date and within 12 months.

Models (same LightGBM params, 3 seeds, early stopping on calib PR-AUC of the model's own target):
  base : y_g12 on all rows                       (today's setup)
  B    : y_g12 on surveyed rows only             (P(harm | survey))
  A    : surveyed on all rows                    (P(survey))
  AxB  : P(A) * P(B), scored on all rows         (comparable with base)
Evaluations:
  all-rows test      : base vs AxB vs B-alone
  surveyed-only test : base vs B   (same rows, so PR-AUC is directly comparable)
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp_common as E  # noqa: E402

MIN_FREE_GB = 0.8
SEEDS = (42, 7, 2026)
PARAMS = {"objective": "binary", "metric": "average_precision", "learning_rate": 0.05, "num_leaves": 31,
          "min_child_samples": 100, "feature_fraction": 0.8, "num_threads": 4, "verbosity": -1}


def guard(where):
    f = E.free_ram_gb()
    print(f"[ram] {where}: {f:.2f} GB free", flush=True)
    if f < MIN_FREE_GB:
        sys.exit(f"abort: free RAM {f:.2f} GB < {MIN_FREE_GB} GB at {where}")


def surveyed_flag(df):
    sp = pd.read_parquet(E.ROOT / "data/gold/spine", columns=["ccn", "t", "cutoff_date"])
    d = df[["ccn", "t"]].merge(sp, on=["ccn", "t"], how="left")
    assert d.cutoff_date.notna().all()
    sv = pd.read_parquet(E.ROOT / "data/silver/ledger_surveys", columns=["ccn", "survey_date", "survey_type"])
    sv = sv[sv.survey_type.str.startswith("HEALTH") | (sv.survey_type == "INFECTION_CONTROL")]
    ct = pd.read_parquet(E.ROOT / "data/silver/ledger_citations", columns=["ccn", "survey_date"])
    ev = pd.concat([sv[["ccn", "survey_date"]], ct]).drop_duplicates()
    ev["sd"] = pd.to_datetime(ev.survey_date).values.astype("datetime64[D]")
    g = {c: np.sort(s.values) for c, s in ev.groupby("ccn").sd}
    a = pd.to_datetime(d.cutoff_date).values.astype("datetime64[D]")
    b = (pd.to_datetime(d.cutoff_date) + pd.DateOffset(months=12)).values.astype("datetime64[D]")
    out = np.zeros(len(d), dtype="int8")
    for i, c in enumerate(d.ccn.values):
        s = g.get(c)
        if s is not None:
            j = np.searchsorted(s, a[i], side="right")
            out[i] = j < len(s) and s[j] <= b[i]
    return out


def fit_predict(Xtr, ytr, Xca, yca, Xte_list, cat, seed):
    p = {**PARAMS, "seed": seed}
    dtr = lgb.Dataset(Xtr, ytr, categorical_feature=cat)
    dca = lgb.Dataset(Xca, yca, reference=dtr, categorical_feature=cat)
    bst = lgb.train(p, dtr, 3000, valid_sets=[dca],
                    callbacks=[lgb.early_stopping(100, first_metric_only=True, verbose=False)])
    return [bst.predict(X, num_iteration=bst.best_iteration) for X in Xte_list], bst.best_iteration


def boot_diff(y, p1, p2, n=500, seed=0):
    """Paired bootstrap of PR-AUC(p2) - PR-AUC(p1), resampling facilities would be stricter; rows used here."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    d = []
    for _ in range(n):
        s = rng.choice(idx, len(idx))
        d.append(average_precision_score(y[s], p2[s]) - average_precision_score(y[s], p1[s]))
    return float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    t0 = time.time()
    guard("start")
    df, num, cat = E.load("y_g12")
    feats = num + cat
    df["surv"] = surveyed_flag(df)
    guard("loaded")
    tr, ca, te = E.split_frames(df)
    print({k: (len(v), round(v.y.mean(), 4), round(v.surv.mean(), 4)) for k, v in
           {"train": tr, "calib": ca, "test": te}.items()}, flush=True)

    trS, caS = tr[tr.surv == 1], ca[ca.surv == 1]
    teS_mask = te.surv.values == 1
    y_te, s_te = te.y.values, te.surv.values
    P = {k: [] for k in ("base", "B", "A")}
    iters = {k: [] for k in P}
    for seed in SEEDS:
        (pb,), it = fit_predict(tr[feats], tr.y, ca[feats], ca.y, [te[feats]], cat, seed)
        P["base"].append(pb); iters["base"].append(it)
        (pB,), it = fit_predict(trS[feats], trS.y, caS[feats], caS.y, [te[feats]], cat, seed)
        P["B"].append(pB); iters["B"].append(it)
        (pA,), it = fit_predict(tr[feats], tr.surv, ca[feats], ca.surv, [te[feats]], cat, seed)
        P["A"].append(pA); iters["A"].append(it)
        guard(f"seed {seed}")

    def per_seed(fn):
        v = [fn(i) for i in range(len(SEEDS))]
        return float(np.mean(v)), float(np.std(v))

    ap = average_precision_score
    rep = {}
    rep["A_survey_test_prauc"] = per_seed(lambda i: ap(s_te, P["A"][i]))
    rep["A_survey_base_rate"] = float(s_te.mean())
    rep["all_base"] = per_seed(lambda i: ap(y_te, P["base"][i]))
    rep["all_AxB"] = per_seed(lambda i: ap(y_te, P["A"][i] * P["B"][i]))
    rep["all_B_alone"] = per_seed(lambda i: ap(y_te, P["B"][i]))
    rep["all_base_rate"] = float(y_te.mean())
    yS = y_te[teS_mask]
    rep["surv_base"] = per_seed(lambda i: ap(yS, P["base"][i][teS_mask]))
    rep["surv_B"] = per_seed(lambda i: ap(yS, P["B"][i][teS_mask]))
    rep["surv_base_rate"] = float(yS.mean())
    mean = {k: np.mean(v, axis=0) for k, v in P.items()}
    rep["boot_surv_B_minus_base"] = boot_diff(yS, mean["base"][teS_mask], mean["B"][teS_mask])
    rep["boot_all_AxB_minus_base"] = boot_diff(y_te, mean["base"], mean["A"] * mean["B"])
    rep["iters"] = iters
    rep["n"] = {"train": len(tr), "train_surveyed": len(trS), "test": len(te), "test_surveyed": int(teS_mask.sum())}

    # standard result JSONs (seed-mean predictions)
    for name, y, p in [("base_all", y_te, mean["base"]), ("AxB_all", y_te, mean["A"] * mean["B"]),
                       ("base_surveyed", yS, mean["base"][teS_mask]), ("B_surveyed", yS, mean["B"][teS_mask])]:
        E.save_result("exp6", name, {}, E.metrics(y, p), {"seeds": list(SEEDS), "params": PARAMS})

    import json
    out = E.RESULTS / "exp6__summary.json"
    out.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps(rep, indent=2))
    print(f"done in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
