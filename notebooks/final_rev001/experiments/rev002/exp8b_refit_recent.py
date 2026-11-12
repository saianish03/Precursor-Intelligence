"""exp8b: same feature sets as exp8, final models refit on every labelled row up to the end of calib.

exp8 fits on train only (2022-01..2022-08, after the purge gap before calib). Hyper-parameters and the feature
selection stay exactly as chosen on calib in exp8; here the final models are refit on train + gap + calib
(2022-01..2023-11). Their labels close by 2024-11, before the test month, so there is no leakage.
Thresholds (accuracy/F1) still come from exp8's calib predictions of the train-only models.
"""

from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import pandas as pd

import exp_common as ec

OUT = ec.EXP_DIR / "outputs" / "exp8_all_tables"
ec.REGISTRY = ec.ROOT / "configs" / "feature_registry.yaml"
df, num, cat = ec.load("y_g12", splits=("train", "calib", "test", "purge_calib"))
fit = df[df.split.isin(["train", "purge_calib", "calib"])]
te = df[df.split == "test"]
print(f"refit rows {len(fit):,} ({fit.t.min()}..{fit.t.max()}), test {len(te):,} ({te.t.min()})", flush=True)
s8 = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
SEL = (OUT / "selected_features.txt").read_text(encoding="utf-8").split("\n")
import yaml  # noqa: E402

old_reg = yaml.safe_load((ec.EXP_DIR / "feature_registry_rev002.yaml").read_text(encoding="utf-8"))["features"]
old_names = set(old_reg) if isinstance(old_reg, dict) else {f["name"] for f in old_reg}
OLD = [f for f in num + cat if f in old_names]
TUNED = json.loads((ec.ROOT.parent / "feature engineering run/spine2022/boost/lgbm_best_params.json")
                   .read_text(encoding="utf-8"))["params"]
from exp8_lr import fit_lr  # noqa: E402

res = {}
yte = te.y.values
for name, cols in (("old", OLD), ("selected", SEL)):
    lg = lgb.LGBMClassifier(**TUNED).fit(fit[cols], fit.y.values)
    lr = fit_lr(fit, cols, num, cat)
    p = {"lgbm": lg.predict_proba(te[cols])[:, 1], "logreg": lr.predict_proba(te[cols])[:, 1]}
    p["ensemble"] = (p["lgbm"] + p["logreg"]) / 2
    for m, pt in p.items():
        thr = s8["test"][f"{name}_{m}"]["thresholds"]["acc_threshold"]
        r = ec.metrics(yte, pt)
        r["accuracy_at_calib_thr"] = float(np.mean((pt >= thr) == yte))
        r["best_possible_accuracy_on_test"] = float(max(np.mean((pt >= g) == yte) for g in np.quantile(pt, np.linspace(.01, .99, 99))))
        r["majority_accuracy"] = float(max(yte.mean(), 1 - yte.mean()))
        res[f"{name}_{m}"] = r
        print(f"{name}_{m:9s} PR-AUC {r['pr_auc']:.4f} ROC {r['roc_auc']:.4f} P@10 {r['p_at_10']:.3f} "
              f"acc {r['accuracy_at_calib_thr']:.4f} (oracle {r['best_possible_accuracy_on_test']:.4f}, "
              f"majority {r['majority_accuracy']:.4f})", flush=True)
    res[f"{name}_pred"] = p
pd.DataFrame({"ccn": te.ccn.values, "y": yte, **{f"{k}": v for k, v in res.pop("selected_pred").items()},
              **{f"old_{k}": v for k, v in res.pop("old_pred").items()}}).to_csv(OUT / "test_predictions_refit.csv", index=False)
(OUT / "summary_refit.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
print("done", flush=True)
