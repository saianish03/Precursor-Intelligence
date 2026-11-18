"""exp8: does bringing every CMS table into gold help? (feat/all-tables, 2026-11-06)

Same gold, same splits for every arm (y_g12, train -> purge gap h -> calib -> test, from configs/splits.yaml):
  A  OLD   : the 133-feature set used in rev002 (feature_registry_rev002.yaml)
  B  ALL   : every model feature in configs/feature_registry.yaml (old + 7 new families)
  C  A + one new family at a time (calib ΔPR-AUC)  -> which tables carry signal
  D  feature selection on B: LightGBM gain ranking on TRAIN, top-k chosen on CALIB
Final models on the selected set (LightGBM tuned params, logistic regression, mean ensemble) are scored once on TEST.
Thresholds for accuracy / F1 are chosen on CALIB, never on test. Test differences get a facility-clustered bootstrap CI.

Run: uv run --no-sync python exp8_all_tables.py   (from this folder; ~30-60 min, ~2 GB RAM)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import exp_common as ec

T0 = time.time()
OUT = ec.EXP_DIR / "outputs" / "exp8_all_tables"
OUT.mkdir(parents=True, exist_ok=True)
NEW_FAMILIES = ["fire_safety", "survey_summary", "ownership", "snf_qrp", "vbp", "covid", "state_context"]
FAST = dict(objective="binary", learning_rate=0.05, num_leaves=31, max_depth=-1, min_child_samples=400,
            subsample=0.7, subsample_freq=1, colsample_bytree=0.5, reg_lambda=5.0, n_estimators=3000,
            verbose=-1, random_state=ec.SEED, n_jobs=2)
TUNED = json.loads((ec.ROOT.parent / "feature engineering run/spine2022/boost/lgbm_best_params.json")
                   .read_text(encoding="utf-8"))["params"] if (ec.ROOT.parent / "feature engineering run/spine2022/boost/lgbm_best_params.json").exists() else FAST


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.0f}s | {ec.free_ram_gb():.1f} GB free] {msg}", flush=True)


# ---------------------------------------------------------------- data
ec.REGISTRY = ec.ROOT / "configs" / "feature_registry.yaml"   # the full registry (old + new families)
ec.write_gold_manifest()                                       # gold was just rebuilt
df, num, cat = ec.load("y_g12")
feats = num + cat
reg = {f["name"]: f for f in yaml.safe_load(ec.REGISTRY.read_text(encoding="utf-8"))["features"]}
old_reg = yaml.safe_load((ec.EXP_DIR / "feature_registry_rev002.yaml").read_text(encoding="utf-8"))["features"]
old_names = set(old_reg) if isinstance(old_reg, dict) else {f["name"] for f in old_reg}
OLD = [f for f in feats if f in old_names]
FAM = {fam: [f for f in feats if reg.get(f, {}).get("family") == fam] for fam in NEW_FAMILIES}
tr, ca, te = ec.split_frames(df)
log(f"rows train {len(tr):,} ({tr.t.min()}..{tr.t.max()}), calib {len(ca):,} ({ca.t.min()}..{ca.t.max()}), "
    f"test {len(te):,} ({te.t.min()}..{te.t.max()}); features all {len(feats)}, old {len(OLD)}, "
    + ", ".join(f"{k} {len(v)}" for k, v in FAM.items()))
ytr, yca, yte = tr.y.values, ca.y.values, te.y.values


# ---------------------------------------------------------------- models
def fit_lgb(cols: list[str], params: dict, early: bool = True) -> lgb.LGBMClassifier:
    m = lgb.LGBMClassifier(**params)
    kw = dict(eval_set=[(ca[cols], yca)], eval_metric="average_precision",
              callbacks=[lgb.early_stopping(100, verbose=False)]) if early else {}
    m.fit(tr[cols], ytr, **kw)
    return m


def fit_lr(cols: list[str]) -> Pipeline:
    n, c = [x for x in cols if x in num], [x for x in cols if x in cat]
    pre = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), n),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=50), c),
    ])
    m = Pipeline([("pre", pre), ("lr", LogisticRegression(C=0.3, max_iter=3000))])
    m.fit(tr[cols], ytr)
    return m


def proba(m, frame: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return m.predict_proba(frame[cols])[:, 1]


def calib_pr(cols: list[str]) -> tuple[float, int, np.ndarray]:
    m = fit_lgb(cols, FAST)
    return ec.metrics(yca, proba(m, ca, cols))["pr_auc"], m.best_iteration_ or FAST["n_estimators"], m


def thresholds(y: np.ndarray, p: np.ndarray) -> dict:
    grid = np.unique(np.quantile(p, np.linspace(0.01, 0.99, 197)))
    acc = [(np.mean((p >= g) == y), g) for g in grid]
    f1 = [(f1_score(y, p >= g), g) for g in grid]
    return {"acc_threshold": float(max(acc)[1]), "f1_threshold": float(max(f1)[1])}


def full_metrics(y: np.ndarray, p: np.ndarray, thr: dict) -> dict:
    m = ec.metrics(y, p)
    pred_a, pred_f = p >= thr["acc_threshold"], p >= thr["f1_threshold"]
    m.update(accuracy_at_calib_thr=float(np.mean(pred_a == y)), accuracy_at_0_5=float(np.mean((p >= 0.5) == y)),
             majority_accuracy=float(max(y.mean(), 1 - y.mean())), f1_at_calib_thr=float(f1_score(y, pred_f)),
             recall_at_f1_thr=float(pred_f[y == 1].mean()), precision_at_f1_thr=float(y[pred_f].mean() if pred_f.any() else 0))
    return m


def cluster_boot(y: np.ndarray, pa: np.ndarray, pb: np.ndarray, groups: np.ndarray, n: int = 300) -> list[float]:
    """95% CI of PR-AUC(b) - PR-AUC(a), resampling facilities."""
    rng = np.random.default_rng(ec.SEED)
    uniq = np.unique(groups)
    idx_by = pd.Series(np.arange(len(groups))).groupby(groups).apply(np.array).to_dict()
    diffs = []
    for _ in range(n):
        pick = np.concatenate([idx_by[g] for g in rng.choice(uniq, len(uniq), replace=True)])
        yy = y[pick]
        if yy.min() == yy.max():
            continue
        diffs.append(ec.metrics(yy, pb[pick])["pr_auc"] - ec.metrics(yy, pa[pick])["pr_auc"])
    return [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]


res: dict = {"rows": {"train": len(tr), "calib": len(ca), "test": len(te)},
             "windows": {s: [f.t.min(), f.t.max()] for s, f in (("train", tr), ("calib", ca), ("test", te))},
             "base_rate_test": float(yte.mean()), "n_features": {"all": len(feats), "old": len(OLD),
             **{k: len(v) for k, v in FAM.items()}}}

# ---------------------------------------------------------------- A / B / C (fast LightGBM, calib)
pr_old, it_old, m_old = calib_pr(OLD)
log(f"A old {len(OLD)} feats: calib PR-AUC {pr_old:.4f} ({it_old} trees)")
pr_all, it_all, m_all = calib_pr(feats)
log(f"B all {len(feats)} feats: calib PR-AUC {pr_all:.4f} ({it_all} trees)")
res["calib_fast"] = {"old": pr_old, "all": pr_all}
res["family_add"] = {}
for fam, cols in FAM.items():
    if not cols:
        continue
    pr, it, _ = calib_pr(OLD + cols)
    res["family_add"][fam] = {"n": len(cols), "calib_pr_auc": pr, "delta": pr - pr_old}
    log(f"C old + {fam} ({len(cols)}): calib {pr:.4f}  Δ {pr - pr_old:+.4f}")

# ---------------------------------------------------------------- D selection (ranking on train only)
gain = pd.Series(m_all.booster_.feature_importance("gain"), index=feats).sort_values(ascending=False)
gain.to_csv(OUT / "gain_all_features.csv", header=["gain"])
ks = sorted({k for k in (30, 50, 75, 100, 130, 160, len(feats)) if k <= len(feats)})
res["topk"] = {}
for k in ks:
    cols = list(gain.index[:k])
    pr, it, _ = calib_pr(cols)
    res["topk"][k] = pr
    log(f"D top-{k}: calib {pr:.4f}")
best_pr = max(res["topk"].values())
# smallest k within 0.002 of the best (fewer features, same signal)
k_sel = min(k for k, v in res["topk"].items() if v >= best_pr - 0.002)
SEL = list(gain.index[:k_sel])
res["selected_k"] = k_sel
(OUT / "selected_features.txt").write_text("\n".join(SEL), encoding="utf-8")
log(f"selected k={k_sel} ({sum(f not in old_names for f in SEL)} from new tables)")

# ---------------------------------------------------------------- final models -> test (once)
final = {}
for name, cols in (("old", OLD), ("selected", SEL)):
    lg = fit_lgb(cols, TUNED, early=False)
    lr = fit_lr(cols)
    for mname, m in (("lgbm", lg), ("logreg", lr)):
        final[f"{name}_{mname}"] = (proba(m, ca, cols), proba(m, te, cols))
    final[f"{name}_ensemble"] = tuple((final[f"{name}_lgbm"][i] + final[f"{name}_logreg"][i]) / 2 for i in (0, 1))
    log(f"final models on {name} ({len(cols)} feats) fitted")
    if name == "selected":
        imp = pd.Series(lg.booster_.feature_importance("gain"), index=cols).sort_values(ascending=False)
        imp.to_csv(OUT / "gain_selected_tuned.csv", header=["gain"])
res["test"] = {}
for key, (pc, pt) in final.items():
    thr = thresholds(yca, pc)
    res["test"][key] = {"calib": ec.metrics(yca, pc), "test": full_metrics(yte, pt, thr), "thresholds": thr}
    t = res["test"][key]["test"]
    log(f"{key:20s} test PR-AUC {t['pr_auc']:.4f} ROC {t['roc_auc']:.4f} acc {t['accuracy_at_calib_thr']:.4f} "
        f"(majority {t['majority_accuracy']:.4f}) F1 {t['f1_at_calib_thr']:.3f}")
groups = te.ccn.values
res["test_ci_selected_vs_old"] = {
    m: cluster_boot(yte, final[f"old_{m}"][1], final[f"selected_{m}"][1], groups) for m in ("lgbm", "logreg", "ensemble")}
log(f"CI ΔPR-AUC selected-old: {res['test_ci_selected_vs_old']}")
res["top_features"] = list(imp.index[:25])
res["minutes"] = (time.time() - T0) / 60
(OUT / "summary.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
log(f"done -> {OUT / 'summary.json'}")
