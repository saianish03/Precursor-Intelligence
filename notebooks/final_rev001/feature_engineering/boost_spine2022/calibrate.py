"""Calibrate the saved y_g12 ensemble (tuned LightGBM + CatBoost + LR) for Gold 2022.

usage: python calibrate.py <selected_csv> <feature_set_yaml> <boost_out_dir> <models_dir>

1. Out-of-fold (3 folds grouped by facility) predictions on all DEV rows, same model recipes as save_models.py
   (CatBoost uses the saved iteration count, no inner early stopping).
2. Ensemble score = mean of the 3 member probabilities (batch-independent, unlike the rank ensemble).
3. Fit isotonic and Platt (logistic on logit) calibrators on the OOF scores.
4. Evaluate on TEST with the saved full-DEV models (models_dir/test_predictions.csv): Brier, log loss, ECE, PR-AUC.
Writes to <models_dir>: calibrator.joblib, calibration_test.csv, calibration_reliability.csv, calibration_reliability.png
"""
import os, sys, time, json, warnings
import numpy as np, pandas as pd, yaml, joblib
warnings.filterwarnings("ignore")
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, log_loss
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from catboost import CatBoostClassifier
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_ensemble as se
from score_ensemble import num, cb_frame, lr_matrix

CSV, FS, BOOST_OUT, MODELS = sys.argv[1:5]
LABEL, STATUS, SPLIT, SEED = "y_g12", "status_g12", "split_g12", 42
NJ = int(os.environ.get("N_JOBS", os.cpu_count() or 1))
t0 = time.time()
def log(m): print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)

FEATS = [f["name"] for f in yaml.safe_load(open(FS, encoding="utf-8"))["features"]]
need_set = set(["ccn", "t", SPLIT, LABEL, STATUS] + FEATS)
parts = []
for ch in pd.read_csv(CSV, dtype={"ccn": str, "t": str}, usecols=lambda c: c in need_set, chunksize=50000, low_memory=False):
    ch = ch[ch[STATUS].isin(["positive", "negative"]) & ch[SPLIT].isin(["train", "calib"])]
    if not len(ch): continue
    f64 = ch.select_dtypes("float64").columns
    if len(f64): ch[f64] = ch[f64].astype("float32")
    parts.append(ch[["ccn", "t", SPLIT, LABEL] + FEATS])
ds = pd.concat(parts, ignore_index=True).sort_values(["ccn", "t"]).reset_index(drop=True); del parts

meta = json.load(open(os.path.join(MODELS, "ensemble_meta.json"), encoding="utf-8"))
CAT, NUMF, LEVELS = meta["categorical_features"], meta["numeric_features"], meta["categorical_levels"]
assert meta["features"] == FEATS, "feature set differs from the saved models"
X = se.model_frame(ds, FEATS, CAT, LEVELS)
y = ds[LABEL].astype(float).astype(int).values
g = ds["ccn"].values
lgb_params = dict(meta["lgbm_params"], n_jobs=NJ)
CB_IT, cat_idx = meta["catboost_iterations"], [FEATS.index(c) for c in CAT]
log(f"DEV {len(ds):,} rows, base rate {y.mean():.3f}; 3-fold grouped OOF, CatBoost {CB_IT} iterations")

oof = np.zeros((len(ds), 3))
for k, (tr, va) in enumerate(GroupKFold(n_splits=3).split(X, y, groups=g), 1):
    m = lgb.LGBMClassifier(**lgb_params).fit(X.iloc[tr], y[tr]); oof[va, 0] = m.predict_proba(X.iloc[va])[:, 1]; del m
    m = CatBoostClassifier(iterations=CB_IT, learning_rate=.05, depth=6, l2_leaf_reg=5, random_seed=SEED, verbose=0,
                           thread_count=NJ, cat_features=cat_idx).fit(cb_frame(X.iloc[tr], CAT), y[tr])
    oof[va, 1] = m.predict_proba(cb_frame(X.iloc[va], CAT))[:, 1]; del m
    Xn = X.iloc[tr][NUMF]; med = Xn.median()
    pre = dict(numf=NUMF, cat=CAT, medians=med, scaler=StandardScaler().fit(Xn.fillna(med)),
               dummy_cols=(pd.get_dummies(ds.iloc[tr][CAT].astype(str)) if CAT else pd.DataFrame(index=Xn.index)).columns.tolist())
    pre["model"] = LogisticRegression(max_iter=3000, C=.5).fit(lr_matrix(X.iloc[tr], ds.iloc[tr], pre), y[tr])
    oof[va, 2] = pre["model"].predict_proba(lr_matrix(X.iloc[va], ds.iloc[va], pre))[:, 1]
    log(f"fold {k}/3 done (OOF PR-AUC lgbm {average_precision_score(y[va], oof[va, 0]):.4f})")

s_oof = oof.mean(axis=1)
eps = 1e-6
logit = lambda p: np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(s_oof, y)
platt = LogisticRegression(C=1e6, max_iter=1000).fit(logit(s_oof)[:, None], y)
cal = dict(method_default="isotonic", isotonic=iso, platt=platt,
           input="mean of p_lgbm, p_catboost, p_logreg (raw member probabilities)",
           fitted_on=dict(rows=int(len(ds)), oof_folds=3, grouped_by="ccn", positive_rate=float(y.mean())))
joblib.dump(cal, os.path.join(MODELS, "calibrator.joblib"))
log("calibrator saved")

# ---- TEST evaluation with the saved full-DEV models ----
tp = pd.read_csv(os.path.join(MODELS, "test_predictions.csv"))
yt = tp[LABEL].astype(int).values
s_test = tp[["p_lgbm", "p_catboost", "p_logreg"]].values.mean(axis=1)
S = {"raw ensemble mean": s_test, "isotonic": iso.predict(s_test), "platt": platt.predict_proba(logit(s_test)[:, None])[:, 1],
     "raw LightGBM": tp["p_lgbm"].values}
def ece(yy, p, bins=10):
    b = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum(abs(yy[b == i].mean() - p[b == i].mean()) * (b == i).mean() for i in range(bins) if (b == i).any()))
def p_at(yy, s, f): o = np.argsort(-s, kind="stable"); return float(yy[o[:max(1, int(len(yy) * f))]].mean())
res = pd.DataFrame({k: dict(brier=brier_score_loss(yt, p), log_loss=log_loss(yt, np.clip(p, eps, 1 - eps)), ece=ece(yt, p),
                            mean_pred=p.mean(), pr_auc=average_precision_score(yt, p), roc_auc=roc_auc_score(yt, p),
                            p_at_5=p_at(yt, p, .05)) for k, p in S.items()}).T
res.loc["base rate (no model)", ["brier", "mean_pred"]] = [brier_score_loss(yt, np.full(len(yt), y.mean())), y.mean()]
res.index.name = "scores"
res.to_csv(os.path.join(MODELS, "calibration_test.csv"))
log(f"TEST ({len(yt):,} rows, base rate {yt.mean():.3f}):\n{res.round(4).to_string()}")

rel = []
fig, ax = plt.subplots(figsize=(6, 6)); ax.plot([0, 1], [0, 1], ls="--", c="grey")
for k in ["raw ensemble mean", "isotonic", "platt"]:
    p = S[k]; q = pd.qcut(p, 10, labels=False, duplicates="drop")
    r = pd.DataFrame({"p": p, "y": yt, "q": q}).groupby("q").agg(pred=("p", "mean"), obs=("y", "mean"), rows=("y", "size"))
    rel += [dict(scores=k, decile=int(i), **row) for i, row in r.iterrows()]
    ax.plot(r.pred, r.obs, marker="o", label=k)
ax.set_xlabel("mean predicted"); ax.set_ylabel("observed rate"); ax.set_title("Reliability on TEST (deciles)"); ax.legend()
plt.tight_layout(); plt.savefig(os.path.join(MODELS, "calibration_reliability.png"), dpi=120)
pd.DataFrame(rel).to_csv(os.path.join(MODELS, "calibration_reliability.csv"), index=False)
log("DONE")
