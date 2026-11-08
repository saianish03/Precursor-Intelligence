"""Refit the y_g12 ensemble (tuned LightGBM + CatBoost + LR) on ALL DEV rows (train+calib) and save it for scoring.
Data loading / feature frame / model recipes are identical to boost_models.py.
usage: python save_models.py <selected_or_engineered_csv> <feature_set_yaml> <boost_out_dir> <models_dir>
Writes to <models_dir>: lgbm_tuned.joblib, catboost.cbm, logreg.joblib, ensemble_meta.json, test_check.csv, test_predictions.csv
"""
import os, sys, time, json, warnings, platform
from datetime import datetime, timezone
import numpy as np, pandas as pd, yaml, joblib
warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import sklearn, scipy, lightgbm as lgb, catboost
from catboost import CatBoostClassifier
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score_ensemble as se
from score_ensemble import num, cb_frame, lr_matrix, rank

CSV, FS, BOOST_OUT, MODELS = sys.argv[1:5]
LABEL, STATUS, SPLIT, SEED = "y_g12", "status_g12", "split_g12", 42
NJ = os.cpu_count() or 1
os.makedirs(MODELS, exist_ok=True)
t0 = time.time()
def log(m): print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)

FEATS = [f["name"] for f in yaml.safe_load(open(FS, encoding="utf-8"))["features"]]
need = ["ccn", "t", SPLIT, LABEL, STATUS] + FEATS
need_set = set(need)
parts = []
for ch in pd.read_csv(CSV, dtype={"ccn": str, "t": str}, usecols=lambda c: c in need_set, chunksize=50000, low_memory=False):
    ch = ch[ch[STATUS].isin(["positive", "negative"]) & ch[SPLIT].isin(["train", "calib", "test"])]
    if not len(ch): continue
    f64 = ch.select_dtypes("float64").columns
    if len(f64): ch[f64] = ch[f64].astype("float32")
    parts.append(ch[["ccn", "t", SPLIT, LABEL] + FEATS])
ds = pd.concat(parts, ignore_index=True).sort_values(["ccn", "t"]).reset_index(drop=True); del parts

CAT = [c for c in FEATS if num(ds[c]) is None]
NUMF = [c for c in FEATS if c not in CAT]
dev_mask = ds[SPLIT].isin(["train", "calib"]).values
LEVELS = {c: sorted(ds.loc[dev_mask, c].dropna().astype(str).unique().tolist()) for c in CAT}
X = se.model_frame(ds, FEATS, CAT, LEVELS)
y = ds[LABEL].astype(float).astype(int).values
DEV, TEST = np.where(dev_mask)[0], np.where(ds[SPLIT].values == "test")[0]
g = ds["ccn"].values
log(f"DEV {len(DEV):,}, TEST {len(TEST):,}; {len(FEATS)} features ({len(CAT)} categorical: {CAT})")

# ---- tuned LightGBM on all DEV ----
lgb_params = dict(json.load(open(os.path.join(BOOST_OUT, "lgbm_best_params.json")))["params"], n_jobs=NJ)
lgbm = lgb.LGBMClassifier(**lgb_params).fit(X.iloc[DEV], y[DEV])
log(f"LightGBM fitted ({lgb_params['n_estimators']} trees)")

# ---- CatBoost: iterations from an inner facility-grouped 15% split of DEV, then refit on all DEV ----
cat_idx = [FEATS.index(c) for c in CAT]
a, b = next(GroupShuffleSplit(n_splits=1, test_size=.15, random_state=SEED).split(DEV, groups=g[DEV]))
m = CatBoostClassifier(iterations=3000, learning_rate=.05, depth=6, l2_leaf_reg=5, eval_metric="PRAUC", od_type="Iter", od_wait=150,
                       random_seed=SEED, verbose=0, thread_count=-1, cat_features=cat_idx)
m.fit(cb_frame(X.iloc[DEV[a]], CAT), y[DEV[a]], eval_set=(cb_frame(X.iloc[DEV[b]], CAT), y[DEV[b]]))
CB_IT = max(100, m.get_best_iteration()); del m
cb = CatBoostClassifier(iterations=CB_IT, learning_rate=.05, depth=6, l2_leaf_reg=5, random_seed=SEED, verbose=0, thread_count=-1,
                        cat_features=cat_idx).fit(cb_frame(X.iloc[DEV], CAT), y[DEV])
log(f"CatBoost fitted ({CB_IT} iterations)")

# ---- logistic regression with saved preprocessing ----
Xn = X.iloc[DEV][NUMF]
dummy_cols = (pd.get_dummies(ds.iloc[DEV][CAT].astype(str)) if CAT else pd.DataFrame(index=Xn.index)).columns.tolist()
med = Xn.median()
lr_pre = dict(numf=NUMF, cat=CAT, medians=med, scaler=StandardScaler().fit(Xn.fillna(med)), dummy_cols=dummy_cols)
lr_pre["model"] = LogisticRegression(max_iter=3000, C=.5).fit(lr_matrix(X.iloc[DEV], ds.iloc[DEV], lr_pre), y[DEV])
log(f"LR fitted ({len(NUMF)} numeric + {len(dummy_cols)} dummy columns)")

# in-memory TEST predictions (to verify the reload)
P_mem = dict(p_lgbm=lgbm.predict_proba(X.iloc[TEST])[:, 1], p_catboost=cb.predict_proba(cb_frame(X.iloc[TEST], CAT))[:, 1],
             p_logreg=lr_pre["model"].predict_proba(lr_matrix(X.iloc[TEST], ds.iloc[TEST], lr_pre))[:, 1])

# ---- save ----
joblib.dump(lgbm, os.path.join(MODELS, "lgbm_tuned.joblib"))
cb.save_model(os.path.join(MODELS, "catboost.cbm"))
joblib.dump(lr_pre, os.path.join(MODELS, "logreg.joblib"))
dev_t = ds[SPLIT].values
meta = dict(label=LABEL, created=datetime.now(timezone.utc).isoformat(timespec="seconds"), source_csv=os.path.abspath(CSV),
            feature_set=os.path.abspath(FS),
            fitted_on=dict(splits=sorted(set(dev_t[DEV].tolist())), months=sorted(ds["t"].iloc[DEV].dropna().unique().tolist()),
                           rows=int(len(DEV)), positive_rate=float(y[DEV].mean())),
            features=FEATS, categorical_features=CAT, numeric_features=NUMF, categorical_levels=LEVELS,
            lgbm_params=lgb_params, catboost_iterations=int(CB_IT),
            catboost_params=dict(learning_rate=.05, depth=6, l2_leaf_reg=5, random_seed=SEED),
            logreg=dict(C=.5, max_iter=3000, n_dummy_columns=len(dummy_cols)),
            ensemble=dict(method="mean of the 3 models' percentile ranks (rankdata/n) within the scored batch",
                          members=["p_lgbm", "p_catboost", "p_logreg"]),
            versions=dict(python=platform.python_version(), pandas=pd.__version__, numpy=np.__version__, sklearn=sklearn.__version__,
                          scipy=scipy.__version__, lightgbm=lgb.__version__, catboost=catboost.__version__))
json.dump(meta, open(os.path.join(MODELS, "ensemble_meta.json"), "w", encoding="utf-8"), indent=2, default=str)
log(f"saved models to {MODELS}")
del lgbm, cb, lr_pre

# ---- reload from disk and score TEST ----
bundle = se.load(MODELS)
P = se.score(bundle, ds.iloc[TEST].reset_index(drop=True))
for k, v in P_mem.items():
    d = float(np.max(np.abs(P[k].values - v))) if len(v) else 0.0
    log(f"reload check {k}: max |disk - memory| = {d:.3g}")
    if d > 1e-6: raise SystemExit(f"reload mismatch for {k}: {d}")
yy = y[TEST]
def p_at(yy, s, f): o = np.argsort(-s, kind="stable"); return float(yy[o[:max(1, int(len(yy) * f))]].mean())
def mets(yy, s): return dict(roc_auc=roc_auc_score(yy, s), pr_auc=average_precision_score(yy, s), p_at_5=p_at(yy, s, .05), p_at_10=p_at(yy, s, .10))
NAMES = {"p_lgbm": "LightGBM tuned", "p_catboost": "CatBoost", "p_logreg": "Logistic regression",
         "ensemble_rank": "Ensemble (tuned LGBM + CatBoost + LR)"}
chk = pd.DataFrame({k: mets(yy, P[k].values) for k in NAMES}).T
chk.index.name = "model"
ref_path = os.path.join(BOOST_OUT, "boost_test.csv")
if os.path.exists(ref_path):
    ref = pd.read_csv(ref_path, index_col=0)["pr_auc"]
    chk["pr_auc_reference"] = [ref.get(NAMES[k], np.nan) for k in chk.index]
    chk["abs_diff_pr_auc"] = (chk["pr_auc"] - chk["pr_auc_reference"]).abs()
chk.to_csv(os.path.join(MODELS, "test_check.csv"))
pd.concat([ds[["ccn", "t"]].iloc[TEST].reset_index(drop=True), pd.Series(yy, name=LABEL), P], axis=1) \
  .to_csv(os.path.join(MODELS, "test_predictions.csv"), index=False)
log(f"TEST ({len(TEST):,} rows, base rate {yy.mean():.3f}):\n{chk.round(4).to_string()}")
log("DONE")
