"""Reproduce the recommended model (exp9 arm R4: LightGBM + LR mean ensemble, 133 gold features, trained on a
seeded sample of 2023-01..2023-11) and compute its complete test-month performance (2025-02).

Run from the repo root:
    uv run --no-sync python notebooks/final_rev001/final_model/final_model_metrics.py
Needs >= 2.5 GB free RAM. Writes final_model_metrics.json, test_predictions.csv and reliability.csv next to this file.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parents[1] / "experiments_rev002"
sys.path.insert(0, str(EXP))

import psutil  # noqa: E402

free = psutil.virtual_memory().available / 1e9
if free < 2.5:
    sys.exit(f"only {free:.2f} GB free RAM (< 2.5 GB); not running")

import lightgbm as lgb  # noqa: E402
import yaml  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score  # noqa: E402

import exp_common as ec  # noqa: E402
from exp8_lr import fit_lr  # noqa: E402

# --- identical to exp9_recency.py, arm "R4 2023 only, size of R1" ---
OUT8 = ec.EXP_DIR / "outputs" / "exp8_all_tables"
ec.REGISTRY = ec.ROOT / "configs" / "feature_registry.yaml"
df, num, cat = ec.load("y_g12", splits=("train", "calib", "test", "purge_calib"))
old_reg = yaml.safe_load((ec.EXP_DIR / "feature_registry_rev002.yaml").read_text(encoding="utf-8"))["features"]
old_names = set(old_reg) if isinstance(old_reg, dict) else {f["name"] for f in old_reg}
OLD = [f for f in num + cat if f in old_names]
TUNED = json.loads((ec.ROOT.parent / "feature engineering run/spine2022/boost/lgbm_best_params.json")
                   .read_text(encoding="utf-8"))["params"]
THR = json.loads((OUT8 / "summary.json").read_text(encoding="utf-8"))["test"]["old_ensemble"]["thresholds"]["acc_threshold"]
lab = df[df.split.isin(["train", "purge_calib", "calib"])]
te = df[df.split == "test"]
r1 = lab[(lab.t >= "2022-01") & (lab.t <= "2022-08")]
r3 = lab[(lab.t >= "2023-01") & (lab.t <= "2023-11")]
tr = r3.sample(n=min(len(r1), len(r3)), random_state=ec.SEED)
del df
print(f"train {len(tr):,} rows {tr.t.min()}..{tr.t.max()}; test {len(te):,} ({te.t.min()}); {len(OLD)} features; "
      f"threshold {THR:.4f}; free RAM {free:.2f} GB", flush=True)

lg = lgb.LGBMClassifier(**TUNED).fit(tr[OLD], tr.y.values)
lr = fit_lr(tr, OLD, num, cat)
p_lgb = lg.predict_proba(te[OLD])[:, 1]
p_lr = lr.predict_proba(te[OLD])[:, 1]
p = (p_lgb + p_lr) / 2
y = te.y.values.astype(int)

ref = json.loads((OUT8 / "summary_recency.json").read_text(encoding="utf-8"))["R4 2023 only, size of R1"]
pr = float(average_precision_score(y, p))
print(f"reproduction: PR-AUC {pr:.6f} vs saved {ref['pr_auc']:.6f}", flush=True)
assert abs(pr - ref["pr_auc"]) < 1e-6, "reproduction does not match exp9 R4"


def at_threshold(y: np.ndarray, flag: np.ndarray) -> dict:
    tp = int(((flag == 1) & (y == 1)).sum()); fp = int(((flag == 1) & (y == 0)).sum())
    fn = int(((flag == 0) & (y == 1)).sum()); tn = int(((flag == 0) & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn); spec = tn / (tn + fp); npv = tn / (tn + fn) if tn + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    mcc_d = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, flagged=tp + fp, flag_rate=(tp + fp) / len(y),
                accuracy=(tp + tn) / len(y), precision=prec, recall=rec, specificity=spec, npv=npv, f1=f1,
                fpr=fp / (fp + tn), fnr=fn / (fn + tp), balanced_accuracy=(rec + spec) / 2,
                mcc=(tp * tn - fp * fn) / mcc_d if mcc_d else 0.0)


def top_k(y: np.ndarray, p: np.ndarray, frac: float) -> dict:
    k = max(1, int(round(len(y) * frac)))
    flag = np.zeros(len(y), int); flag[np.argsort(-p, kind="stable")[:k]] = 1
    d = at_threshold(y, flag)
    d.update(k=k, lift=d["precision"] / y.mean(), score_cutoff=float(np.sort(p)[::-1][k - 1]))
    return d


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> tuple[float, list[dict]]:
    order = np.argsort(p, kind="stable"); groups = np.array_split(order, bins)
    rows, e = [], 0.0
    for i, g in enumerate(groups):
        mp, oy = float(p[g].mean()), float(y[g].mean())
        e += len(g) / len(y) * abs(oy - mp)
        rows.append(dict(bin=i + 1, n=len(g), p_min=float(p[g].min()), p_max=float(p[g].max()),
                         mean_predicted=mp, observed_rate=oy, gap=oy - mp))
    return e, rows


def core(y: np.ndarray, p: np.ndarray) -> dict:
    d = dict(pr_auc=float(average_precision_score(y, p)), roc_auc=float(roc_auc_score(y, p)),
             brier=float(brier_score_loss(y, p)), accuracy=float(np.mean((p >= THR) == y)),
             ece=ece(y, p)[0])
    for f in (0.02, 0.05, 0.10):
        t = top_k(y, p, f)
        d[f"p_at_{int(f * 100)}"] = t["precision"]; d[f"recall_at_{int(f * 100)}"] = t["recall"]
    return d


res: dict = {"model": "LightGBM (tuned) + logistic regression, mean of probabilities; 133 gold features",
             "train": {"rows": len(tr), "months": f"{tr.t.min()}..{tr.t.max()}", "base_rate": float(tr.y.mean())},
             "test": {"month": str(te.t.min()), "rows": len(y), "positives": int(y.sum()), "base_rate": float(y.mean())},
             "threshold": THR, "reproduction_pr_auc_matches_exp9": True}
res["threshold_free"] = core(y, p)
bs_ref = float(y.mean() * (1 - y.mean()))
res["threshold_free"].update(brier_reference_base_rate=bs_ref, brier_skill_score=1 - res["threshold_free"]["brier"] / bs_ref,
                             majority_accuracy=float(max(y.mean(), 1 - y.mean())))
res["components"] = {n: {"pr_auc": float(average_precision_score(y, q)), "roc_auc": float(roc_auc_score(y, q)),
                         "brier": float(brier_score_loss(y, q))} for n, q in (("lightgbm", p_lgb), ("logistic_regression", p_lr))}
res["at_threshold"] = at_threshold(y, (p >= THR).astype(int))
res["top_k"] = {f"top_{int(f * 100)}pct": top_k(y, p, f) for f in (0.01, 0.02, 0.05, 0.10, 0.20)}
e, rel = ece(y, p)
lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
cal = LogisticRegression(C=1e6, max_iter=1000).fit(lp.reshape(-1, 1), y)
res["calibration"] = {"ece_10_equal_mass_bins": e, "mean_predicted": float(p.mean()), "observed_rate": float(y.mean()),
                      "calibration_slope": float(cal.coef_[0][0]), "calibration_intercept": float(cal.intercept_[0])}

# bootstrap CIs (one row per facility in the test month, so facility-clustered == row resampling)
rng = np.random.default_rng(ec.SEED)
B = 1000
draws = {k: [] for k in core(y, p)}
for _ in range(B):
    i = rng.integers(0, len(y), len(y))
    for k, v in core(y[i], p[i]).items():
        draws[k].append(v)
res["bootstrap_95ci"] = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in draws.items()}
res["bootstrap_replicates"] = B

(HERE / "final_model_metrics.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
import pandas as pd  # noqa: E402

pd.DataFrame(rel).to_csv(HERE / "reliability.csv", index=False)
pd.DataFrame({"ccn": te.ccn.values, "t": te.t.values, "y": y, "p_lightgbm": p_lgb, "p_logreg": p_lr,
              "p_ensemble": p}).to_csv(HERE / "test_predictions.csv", index=False, float_format="%.6f")
print(json.dumps({k: res[k] for k in ("threshold_free", "at_threshold", "calibration")}, indent=1), flush=True)
print("done", flush=True)
