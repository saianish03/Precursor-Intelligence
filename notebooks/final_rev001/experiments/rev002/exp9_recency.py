"""exp9: recency vs row count. Same 133 old features, same tuned LightGBM + logistic regression ensemble, same
test month (2025-02); only the training window changes. Accuracy uses one fixed threshold (exp8 old_ensemble,
chosen on calib) for every arm. 95 % CIs: facility-clustered bootstrap of (arm - R1) on the test month.
"""

from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml

import exp_common as ec
from exp8_lr import fit_lr

OUT = ec.EXP_DIR / "outputs" / "exp8_all_tables"
ec.REGISTRY = ec.ROOT / "configs" / "feature_registry.yaml"
df, num, cat = ec.load("y_g12", splits=("train", "calib", "test", "purge_calib"))
old_reg = yaml.safe_load((ec.EXP_DIR / "feature_registry_rev002.yaml").read_text(encoding="utf-8"))["features"]
old_names = set(old_reg) if isinstance(old_reg, dict) else {f["name"] for f in old_reg}
OLD = [f for f in num + cat if f in old_names]
TUNED = json.loads((ec.ROOT.parent / "feature engineering run/spine2022/boost/lgbm_best_params.json")
                   .read_text(encoding="utf-8"))["params"]
THR = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))["test"]["old_ensemble"]["thresholds"]["acc_threshold"]
lab = df[df.split.isin(["train", "purge_calib", "calib"])]
te = df[df.split == "test"]
yte, groups = te.y.values, te.ccn.values
rng = np.random.default_rng(ec.SEED)

r1 = lab[(lab.t >= "2022-01") & (lab.t <= "2022-08")]
r3 = lab[(lab.t >= "2023-01") & (lab.t <= "2023-11")]
y2022 = lab[(lab.t >= "2022-01") & (lab.t <= "2022-12")]
arms = {
    "R1 2022-01..08 (original)": r1,
    "R2 2022-01..2023-11 (all)": lab,
    "R3 2023 only": r3,
    "R4 2023 only, size of R1": r3.sample(n=min(len(r1), len(r3)), random_state=ec.SEED),
    "R5 2022 only, size of R3": y2022.sample(n=min(len(r3), len(y2022)), random_state=ec.SEED),
}


def boot(pa: np.ndarray, pb: np.ndarray, n: int = 300) -> dict:
    uniq = np.unique(groups)
    idx = pd.Series(np.arange(len(groups))).groupby(groups).apply(np.array).to_dict()
    d_pr, d_acc = [], []
    for _ in range(n):
        pick = np.concatenate([idx[g] for g in rng.choice(uniq, len(uniq), replace=True)])
        yy = yte[pick]
        d_pr.append(ec.metrics(yy, pb[pick])["pr_auc"] - ec.metrics(yy, pa[pick])["pr_auc"])
        d_acc.append(np.mean((pb[pick] >= THR) == yy) - np.mean((pa[pick] >= THR) == yy))
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]  # noqa: E731
    return {"d_pr_auc_ci": q(d_pr), "d_acc_ci": q(d_acc)}


res, preds = {}, {}
for name, tr in arms.items():
    lg = lgb.LGBMClassifier(**TUNED).fit(tr[OLD], tr.y.values)
    lr = fit_lr(tr, OLD, num, cat)
    p = (lg.predict_proba(te[OLD])[:, 1] + lr.predict_proba(te[OLD])[:, 1]) / 2
    preds[name] = p
    m = ec.metrics(yte, p)
    m.update(rows=len(tr), months=f"{tr.t.min()}..{tr.t.max()}", train_base_rate=float(tr.y.mean()),
             accuracy=float(np.mean((p >= THR) == yte)),
             p_at_2=ec.precision_at(yte, p, 0.02))
    res[name] = m
    print(f"{name:28s} rows {len(tr):>7,} PR-AUC {m['pr_auc']:.4f} ROC {m['roc_auc']:.4f} acc {m['accuracy']:.4f} "
          f"P@2 {m['p_at_2']:.3f} P@5 {m['p_at_5']:.3f} P@10 {m['p_at_10']:.3f}", flush=True)
base = preds["R1 2022-01..08 (original)"]
for name in list(arms)[1:]:
    res[name] |= boot(base, preds[name])
    print(f"{name:28s} vs R1: ΔPR-AUC CI {res[name]['d_pr_auc_ci']}, Δacc CI {res[name]['d_acc_ci']}", flush=True)
res["_meta"] = {"test": te.t.min(), "test_rows": len(te), "base_rate": float(yte.mean()), "threshold": THR}
(OUT / "summary_recency.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
print("done", flush=True)
