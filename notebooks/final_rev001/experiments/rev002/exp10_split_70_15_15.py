"""exp10: 70 / 15 / 15 train / validation / test on every labelled gold row (y_g12), three ways.

  random   : rows shuffled (stratified). The same facility sits in all three parts, and neighbouring months share
             label events -> leaky, optimistic; shown because it was asked for.
  facility : facilities split 70/15/15 (a facility is in one part only) -> no facility leakage.
  time     : rows ordered by month; earliest 70 % train, next 15 % validation, latest 15 % test -> a forecast.
Validation = early stopping + accuracy / F1 threshold; test scored once. Old 133 vs all 202 features.
"""

from __future__ import annotations

import json
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split

import exp_common as ec
from exp8_lr import fit_lr

T0 = time.time()
OUT = ec.EXP_DIR / "outputs" / "exp10_split_70_15_15"
OUT.mkdir(parents=True, exist_ok=True)
ec.REGISTRY = ec.ROOT / "configs" / "feature_registry.yaml"
df, num, cat = ec.load("y_g12", splits=("train", "purge_calib", "calib", "purge", "test"))
old_reg = yaml.safe_load((ec.EXP_DIR / "feature_registry_rev002.yaml").read_text(encoding="utf-8"))["features"]
old_names = set(old_reg) if isinstance(old_reg, dict) else {f["name"] for f in old_reg}
ALL = num + cat
OLD = [f for f in ALL if f in old_names]
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_child_samples=400, subsample=0.7,
              subsample_freq=1, colsample_bytree=0.5, reg_lambda=5.0, n_estimators=3000, verbose=-1,
              random_state=ec.SEED, n_jobs=int(os.environ.get("EXP_JOBS", "2")))
print(f"labelled rows {len(df):,} ({df.t.min()}..{df.t.max()}), positive rate {df.y.mean():.3f}, "
      f"features all {len(ALL)} old {len(OLD)}", flush=True)


def split(kind: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.arange(len(df))
    if kind == "random":
        tr, rest = train_test_split(idx, test_size=0.30, random_state=ec.SEED, stratify=df.y)
        va, te = train_test_split(rest, test_size=0.50, random_state=ec.SEED, stratify=df.y.iloc[rest])
    elif kind == "facility":
        g = df.ccn.values
        tr, rest = next(GroupShuffleSplit(1, test_size=0.30, random_state=ec.SEED).split(idx, groups=g))
        va_i, te_i = next(GroupShuffleSplit(1, test_size=0.50, random_state=ec.SEED).split(rest, groups=g[rest]))
        va, te = rest[va_i], rest[te_i]
    else:  # time: df is sorted by (t, ccn); cut at month boundaries closest to 70 % / 85 %
        months = df.t.values
        cum = pd.Series(months).value_counts().sort_index().cumsum() / len(df)
        m70, m85 = cum[cum >= 0.70].index[0], cum[cum >= 0.85].index[0]
        tr, va, te = idx[months <= m70], idx[(months > m70) & (months <= m85)], idx[months > m85]
    return tr, va, te


def thr_best(y, p, score):
    grid = np.unique(np.quantile(p, np.linspace(0.01, 0.99, 197)))
    return float(max(grid, key=lambda g: score(y, p >= g)))


res = {}
KINDS = os.environ.get("EXP_KINDS", "random,facility,time").split(",")
for kind in KINDS:
    tr, va, te = split(kind)
    A, V, T = df.iloc[tr], df.iloc[va], df.iloc[te]
    info = {s: {"rows": len(f), "share": round(len(f) / len(df), 3), "pos_rate": round(float(f.y.mean()), 4),
                "months": f"{f.t.min()}..{f.t.max()}", "facilities": int(f.ccn.nunique())}
            for s, f in (("train", A), ("validation", V), ("test", T))}
    shared = len(set(A.ccn) & set(T.ccn))
    print(f"\n== {kind}: " + "; ".join(f"{s} {v['rows']:,} ({v['share']:.0%}, {v['months']})" for s, v in info.items())
          + f"; facilities in both train and test: {shared:,}", flush=True)
    res[kind] = {"split": info, "facilities_in_train_and_test": shared, "models": {}}
    for fname, cols in (("old_133", OLD), ("all_202", ALL)):
        lg = lgb.LGBMClassifier(**PARAMS).fit(A[cols], A.y.values, eval_set=[(V[cols], V.y.values)],
                                              eval_metric="average_precision",
                                              callbacks=[lgb.early_stopping(100, verbose=False)])
        # LightGBM only (logistic regression on 350k rows was too slow for this run)
        pv = lg.predict_proba(V[cols])[:, 1]
        pt = lg.predict_proba(T[cols])[:, 1]
        yv, yt = V.y.values, T.y.values
        ta = thr_best(yv, pv, lambda y, h: np.mean(h == y))
        tf = thr_best(yv, pv, f1_score)
        m = ec.metrics(yt, pt)
        m.update(p_at_2=ec.precision_at(yt, pt, 0.02), accuracy=float(np.mean((pt >= ta) == yt)),
                 majority_accuracy=float(max(yt.mean(), 1 - yt.mean())), f1=float(f1_score(yt, pt >= tf)),
                 recall_at_f1_thr=float((pt >= tf)[yt == 1].mean()), trees=int(lg.best_iteration_ or 0))
        res[kind]["models"][fname] = m
        print(f"   {fname}: ROC {m['roc_auc']:.4f} PR-AUC {m['pr_auc']:.4f} acc {m['accuracy']:.4f} "
              f"(majority {m['majority_accuracy']:.4f}) F1 {m['f1']:.3f} P@2 {m['p_at_2']:.3f} P@5 {m['p_at_5']:.3f} "
              f"P@10 {m['p_at_10']:.3f}  [{time.time() - T0:.0f}s]", flush=True)
(OUT / f"summary_{'_'.join(KINDS)}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
print("done", flush=True)
