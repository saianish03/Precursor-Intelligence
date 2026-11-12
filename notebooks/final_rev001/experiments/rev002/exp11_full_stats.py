"""exp11: full statistics for the honest 70/15/15 splits (LightGBM, y_g12), old 133 vs all 202 features.

Splits (every labelled gold row):
  facility      : facilities 70/15/15, all months mixed (no facility in two parts)
  time          : months in order; earliest ~70 % train, next ~15 % validation, latest ~15 % test
  facility_time : the time split, but validation/test keep only facilities that are NOT in train (strictest)
Validation = early stopping + thresholds. Test scored once. Predictions saved; report.md written.
"""

from __future__ import annotations

import json
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix, f1_score, log_loss,
                             matthews_corrcoef, roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit

import exp_common as ec

T0 = time.time()
OUT = ec.EXP_DIR / "outputs" / "exp11_full_stats"
OUT.mkdir(parents=True, exist_ok=True)
ec.REGISTRY = ec.ROOT / "configs" / "feature_registry.yaml"
df, num, cat = ec.load("y_g12", splits=("train", "purge_calib", "calib", "purge", "test"))
old_reg = yaml.safe_load((ec.EXP_DIR / "feature_registry_rev002.yaml").read_text(encoding="utf-8"))["features"]
old_names = set(old_reg) if isinstance(old_reg, dict) else {f["name"] for f in old_reg}
ALL = num + cat
OLD = [f for f in ALL if f in old_names]
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_child_samples=400, subsample=0.7,
              subsample_freq=1, colsample_bytree=0.5, reg_lambda=5.0, n_estimators=3000, verbose=-1,
              random_state=ec.SEED, n_jobs=int(os.environ.get("EXP_JOBS", "4")))
KINDS = os.environ.get("EXP_KINDS", "facility,time,facility_time").split(",")
rng = np.random.default_rng(ec.SEED)


def log(m: str) -> None:
    print(f"[{time.time() - T0:6.0f}s | {ec.free_ram_gb():.1f} GB] {m}", flush=True)


def split(kind: str):
    idx = np.arange(len(df))
    g = df.ccn.values
    if kind == "facility":
        tr, rest = next(GroupShuffleSplit(1, test_size=0.30, random_state=ec.SEED).split(idx, groups=g))
        a, b = next(GroupShuffleSplit(1, test_size=0.50, random_state=ec.SEED).split(rest, groups=g[rest]))
        return tr, rest[a], rest[b]
    months = df.t.values
    cum = pd.Series(months).value_counts().sort_index().cumsum() / len(df)
    m70, m85 = cum[cum >= 0.70].index[0], cum[cum >= 0.85].index[0]
    tr, va, te = idx[months <= m70], idx[(months > m70) & (months <= m85)], idx[months > m85]
    if kind == "facility_time":
        fac = np.unique(g)
        keep = set(rng.choice(fac, int(len(fac) * 0.70), replace=False))
        in_tr = np.isin(g, list(keep))
        tr, va, te = tr[in_tr[tr]], va[~in_tr[va]], te[~in_tr[te]]
    return tr, va, te


def best_thr(y, p, score):
    grid = np.unique(np.quantile(p, np.linspace(0.005, 0.995, 199)))
    return float(max(grid, key=lambda h: score(y, (p >= h).astype(int))))


def at_threshold(y, p, h) -> dict:
    pred = (p >= h).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec, spec = tp / (tp + fn), tn / (tn + fp)
    return {"threshold": round(float(h), 4), "accuracy": (tp + tn) / len(y), "balanced_accuracy": (rec + spec) / 2,
            "precision": prec, "recall": rec, "specificity": spec, "f1": f1_score(y, pred),
            "mcc": matthews_corrcoef(y, pred), "flagged_share": pred.mean(),
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)}


def topk(y, p) -> dict:
    o = np.argsort(-p, kind="stable")
    out = {}
    for f in (0.01, 0.02, 0.05, 0.10, 0.20, 0.30):
        k = max(1, int(round(len(y) * f)))
        hits = y[o[:k]].sum()
        out[f"{int(f * 100)}%"] = {"n": k, "precision": hits / k, "recall": hits / y.sum(), "lift": (hits / k) / y.mean()}
    return out


def calibration(y, p, bins: int = 10) -> tuple[float, list]:
    q = pd.qcut(p, bins, labels=False, duplicates="drop")
    t = pd.DataFrame({"q": q, "y": y, "p": p}).groupby("q").agg(n=("y", "size"), predicted=("p", "mean"), actual=("y", "mean"))
    ece = float((t.n * (t.predicted - t.actual).abs()).sum() / len(y))
    return ece, t.round(4).reset_index().to_dict("records")


def boot_ci(y, p, groups, n=200) -> dict:
    uniq = np.unique(groups)
    idx = pd.Series(np.arange(len(groups))).groupby(groups).apply(np.array).to_dict()
    roc, pr = [], []
    for _ in range(n):
        pick = np.concatenate([idx[g] for g in rng.choice(uniq, len(uniq), replace=True)])
        roc.append(roc_auc_score(y[pick], p[pick])); pr.append(average_precision_score(y[pick], p[pick]))
    return {"roc_auc_95": [float(np.percentile(roc, 2.5)), float(np.percentile(roc, 97.5))],
            "pr_auc_95": [float(np.percentile(pr, 2.5)), float(np.percentile(pr, 97.5))]}


res = {"labelled_rows": len(df), "positive_rate": float(df.y.mean()), "features": {"old": len(OLD), "all": len(ALL)}}
for kind in KINDS:
    tr, va, te = split(kind)
    A, V, T = df.iloc[tr], df.iloc[va], df.iloc[te]
    res[kind] = {"split": {s: {"rows": len(f), "share": len(f) / (len(A) + len(V) + len(T)), "months": f"{f.t.min()}..{f.t.max()}",
                               "facilities": int(f.ccn.nunique()), "positive_rate": float(f.y.mean())}
                           for s, f in (("train", A), ("validation", V), ("test", T))},
                 "facilities_in_train_and_test": len(set(A.ccn) & set(T.ccn)), "models": {}}
    log(f"== {kind}: " + "; ".join(f"{s} {v['rows']:,} ({v['months']}, pos {v['positive_rate']:.3f})"
                                   for s, v in res[kind]["split"].items()))
    preds = {"ccn": T.ccn.values, "t": T.t.values, "y": T.y.values}
    for fname, cols in (("old_133", OLD), ("all_202", ALL)):
        m = lgb.LGBMClassifier(**PARAMS).fit(A[cols], A.y.values, eval_set=[(V[cols], V.y.values)],
                                             eval_metric="average_precision",
                                             callbacks=[lgb.early_stopping(100, verbose=False)])
        pv, pt = m.predict_proba(V[cols])[:, 1], m.predict_proba(T[cols])[:, 1]
        yv, yt = V.y.values, T.y.values
        ece, calib_tab = calibration(yt, pt)
        r = {"trees": int(m.best_iteration_ or 0), "roc_auc": roc_auc_score(yt, pt), "pr_auc": average_precision_score(yt, pt),
             "brier": brier_score_loss(yt, pt), "brier_baseline": brier_score_loss(yt, np.full_like(pt, A.y.mean())),
             "log_loss": log_loss(yt, pt), "ece": ece, "majority_accuracy": float(max(yt.mean(), 1 - yt.mean())),
             "at_validation_best_accuracy": at_threshold(yt, pt, best_thr(yv, pv, lambda y, h: np.mean(h == y))),
             "at_validation_best_f1": at_threshold(yt, pt, best_thr(yv, pv, f1_score)),
             "at_0.5": at_threshold(yt, pt, 0.5),
             "best_possible_accuracy_on_test": float(max(np.mean((pt >= h) == yt) for h in np.quantile(pt, np.linspace(.005, .995, 199)))),
             "top_k": topk(yt, pt), "calibration_deciles": calib_tab, "ci": boot_ci(yt, pt, T.ccn.values),
             "validation": {"roc_auc": roc_auc_score(yv, pv), "pr_auc": average_precision_score(yv, pv)},
             "top_features": pd.Series(m.booster_.feature_importance("gain"), index=cols).nlargest(15).round(0).to_dict()}
        if kind != "facility":
            r["by_test_month"] = {t: {"n": int((T.t == t).sum()), "pos": float(yt[T.t.values == t].mean()),
                                      "roc_auc": roc_auc_score(yt[T.t.values == t], pt[T.t.values == t]),
                                      "pr_auc": average_precision_score(yt[T.t.values == t], pt[T.t.values == t])}
                                  for t in sorted(T.t.unique())}
        res[kind]["models"][fname] = r
        preds[f"p_{fname}"] = pt
        a = r["at_validation_best_accuracy"]
        log(f"   {kind}/{fname}: ROC {r['roc_auc']:.4f} {r['ci']['roc_auc_95']} PR-AUC {r['pr_auc']:.4f} acc {a['accuracy']:.4f} "
            f"(majority {r['majority_accuracy']:.4f}) F1@best {r['at_validation_best_f1']['f1']:.3f} "
            f"P@2 {r['top_k']['2%']['precision']:.3f} P@5 {r['top_k']['5%']['precision']:.3f} P@10 {r['top_k']['10%']['precision']:.3f} "
            f"Brier {r['brier']:.4f} ECE {ece:.4f}")
        (OUT / "summary.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    pd.DataFrame(preds).to_parquet(OUT / f"test_predictions_{kind}.parquet", index=False)
log("done")
