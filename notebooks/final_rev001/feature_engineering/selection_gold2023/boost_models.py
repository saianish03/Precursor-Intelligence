"""Push y_g12 performance on the 75 selected features without leakage: Optuna-tuned LightGBM, CatBoost, LR, rank ensemble.
Same DEV / 20 % facility holdout / test months as select_full_gold.py. usage: python boost_models.py <selected_csv> <feature_set_yaml> <out_dir>
"""
import os, sys, time, json, warnings
import numpy as np, pandas as pd, yaml
warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import rankdata
import lightgbm as lgb, optuna
from catboost import CatBoostClassifier

CSV, FS, OUT = sys.argv[1:4]
LABEL, STATUS, SPLIT, SEED, N_TRIALS = "y_g12", "status_g12", "split_g12", 42, int(os.environ.get("N_TRIALS", 30))
t0 = time.time()
def log(m): print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)

FEATS = [f["name"] for f in yaml.safe_load(open(FS, encoding="utf-8"))["features"]]
parts = []
for ch in pd.read_csv(CSV, dtype={"ccn": str, "t": str}, chunksize=50000, low_memory=False):
    ch = ch[ch[STATUS].isin(["positive", "negative"]) & ch[SPLIT].isin(["train", "calib", "test"])]
    if len(ch): parts.append(ch[["ccn", "t", SPLIT, LABEL] + FEATS])
ds = pd.concat(parts, ignore_index=True).sort_values(["ccn", "t"]).reset_index(drop=True); del parts

def num(s):
    if pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s): return pd.to_numeric(s, errors="coerce").astype(float)
    v = set(s.dropna().unique().tolist())
    if v and v <= {True, False, "True", "False"}: return s.map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0}).astype(float)
    return None
CAT = [c for c in FEATS if num(ds[c]) is None]
X = pd.DataFrame(index=ds.index)
dev_mask = ds[SPLIT].isin(["train", "calib"]).values
for c in FEATS:
    n = num(ds[c])
    X[c] = n if n is not None else pd.Categorical(ds[c].astype(str), categories=sorted(ds.loc[dev_mask, c].dropna().astype(str).unique()))
y = ds[LABEL].astype(float).astype(int).values
DEV, TEST = np.where(dev_mask)[0], np.where(ds[SPLIT].values == "test")[0]
g = ds["ccn"].values
sel_i, hold_i = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=SEED).split(DEV, y[DEV], groups=g[DEV]))
SEL, HOLD = DEV[sel_i], DEV[hold_i]
log(f"DEV {len(DEV):,} (SEL {len(SEL):,} / HOLD {len(HOLD):,}), TEST {len(TEST):,}; {len(FEATS)} features ({len(CAT)} categorical: {CAT})")

def p_at(yy, s, f): o = np.argsort(-s, kind="stable"); return float(yy[o[:max(1, int(len(yy) * f))]].mean())
def mets(yy, s): return dict(roc_auc=roc_auc_score(yy, s), pr_auc=average_precision_score(yy, s), p_at_5=p_at(yy, s, .05), p_at_10=p_at(yy, s, .10))
FOLDS = list(GroupKFold(n_splits=3).split(SEL, y[SEL], groups=g[SEL]))

# ---- 1. Optuna LightGBM ----
def lgb_params(tr):
    return dict(objective="binary", learning_rate=tr.suggest_float("learning_rate", .01, .1, log=True),
                num_leaves=tr.suggest_int("num_leaves", 8, 128, log=True), max_depth=tr.suggest_int("max_depth", 3, 12),
                min_child_samples=tr.suggest_int("min_child_samples", 20, 1000, log=True),
                subsample=tr.suggest_float("subsample", .5, 1.0), subsample_freq=1, colsample_bytree=tr.suggest_float("colsample_bytree", .3, 1.0),
                reg_alpha=tr.suggest_float("reg_alpha", 1e-3, 10, log=True), reg_lambda=tr.suggest_float("reg_lambda", 1e-3, 30, log=True),
                min_split_gain=tr.suggest_float("min_split_gain", 0, 1), cat_smooth=tr.suggest_float("cat_smooth", 1, 100, log=True),
                n_estimators=3000, verbose=-1, random_state=SEED, n_jobs=2)
def objective(tr):
    p = lgb_params(tr); scores, iters = [], []
    for tr_i, va_i in FOLDS:
        a, b = SEL[tr_i], SEL[va_i]
        m = lgb.LGBMClassifier(**p).fit(X.iloc[a], y[a], eval_set=[(X.iloc[b], y[b])], eval_metric="average_precision",
                                        callbacks=[lgb.early_stopping(100, verbose=False)])
        scores.append(average_precision_score(y[b], m.predict_proba(X.iloc[b], num_iteration=m.best_iteration_)[:, 1])); iters.append(m.best_iteration_)
    tr.set_user_attr("n_estimators", int(np.mean(iters))); return float(np.mean(scores))
optuna.logging.set_verbosity(optuna.logging.WARNING)
if os.environ.get("REUSE_TUNING") and os.path.exists(f"{OUT}/lgbm_best_params.json"):
    best = json.load(open(f"{OUT}/lgbm_best_params.json"))["params"]
    log(f"reusing saved tuned params: {best}")
    N_TRIALS = 0
if N_TRIALS:
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.enqueue_trial(dict(learning_rate=.05, num_leaves=31, max_depth=12, min_child_samples=100, subsample=.8, colsample_bytree=.8,
                             reg_alpha=1e-3, reg_lambda=1e-3, min_split_gain=0, cat_smooth=10))           # trial 0 ~ the current default model
    study.optimize(objective, n_trials=N_TRIALS, callbacks=[lambda s, t: log(f"trial {t.number + 1}/{N_TRIALS}: CV PR-AUC {t.value:.4f} (best {s.best_value:.4f})")])
    best = {**lgb_params(optuna.trial.FixedTrial(study.best_params)), "n_estimators": study.best_trial.user_attrs["n_estimators"]}
    json.dump({"cv_pr_auc": study.best_value, "params": best}, open(f"{OUT}/lgbm_best_params.json", "w"), indent=2, default=str)
    study.trials_dataframe().to_csv(f"{OUT}/optuna_trials.csv", index=False)
    log(f"best CV PR-AUC {study.best_value:.4f} (default-like trial 0: {study.trials[0].value:.4f}); params {study.best_params}")

# ---- models: fit on SEL -> HOLD (decision), fit on DEV -> TEST (report) ----
def fit_lgb(tr_idx, p): return lgb.LGBMClassifier(**p).fit(X.iloc[tr_idx], y[tr_idx])
DEFAULT = dict(n_estimators=400, learning_rate=.05, num_leaves=31, min_child_samples=100, subsample=.8, subsample_freq=1,
               colsample_bytree=.8, verbose=-1, random_state=SEED, n_jobs=2)
cat_idx = [FEATS.index(c) for c in CAT]
def cb_frame(idx):
    Z = X.iloc[idx].copy()
    for c in CAT: Z[c] = Z[c].astype(object).fillna("missing").astype(str)
    return Z
def fit_cb(tr_idx):
    # iterations chosen by an inner facility-grouped split of the training rows (never the evaluation rows)
    a, b = next(GroupShuffleSplit(n_splits=1, test_size=.15, random_state=SEED).split(tr_idx, groups=g[tr_idx]))
    m = CatBoostClassifier(iterations=3000, learning_rate=.05, depth=6, l2_leaf_reg=5, eval_metric="PRAUC", od_type="Iter", od_wait=150,
                           random_seed=SEED, verbose=0, thread_count=2, cat_features=cat_idx)
    m.fit(cb_frame(tr_idx[a]), y[tr_idx[a]], eval_set=(cb_frame(tr_idx[b]), y[tr_idx[b]]))
    it = max(100, m.get_best_iteration())
    return CatBoostClassifier(iterations=it, learning_rate=.05, depth=6, l2_leaf_reg=5, random_seed=SEED, verbose=0, thread_count=2,
                              cat_features=cat_idx).fit(cb_frame(tr_idx), y[tr_idx]), it
NUMF = [c for c in FEATS if c not in CAT]
def lr_fit_predict(tr_idx, te_idx):
    def mk(idx, fit=None):
        Xn = X.iloc[idx][NUMF]; Xc = pd.get_dummies(ds.iloc[idx][CAT].astype(str)) if CAT else pd.DataFrame(index=Xn.index)
        if fit is None: fit = dict(med=Xn.median(), cols=Xc.columns); fit["sc"] = StandardScaler().fit(Xn.fillna(fit["med"]))
        return np.c_[fit["sc"].transform(Xn.fillna(fit["med"])), Xc.reindex(columns=fit["cols"], fill_value=0).values], fit
    A, f = mk(tr_idx); B, _ = mk(te_idx, f)
    return LogisticRegression(max_iter=3000, C=.5).fit(A, y[tr_idx]).predict_proba(B)[:, 1]
def rank(s): return rankdata(s) / len(s)

results = {}
STAGES = [("holdout", SEL, HOLD), ("test", DEV, TEST)]
if os.environ.get("TEST_ONLY"): STAGES = STAGES[1:]
for stage, tr_idx, te_idx in STAGES:
    yy = y[te_idx]; P = {}
    P["GBM default (current)"] = fit_lgb(tr_idx, DEFAULT).predict_proba(X.iloc[te_idx])[:, 1]
    P["LightGBM tuned"] = fit_lgb(tr_idx, best).predict_proba(X.iloc[te_idx])[:, 1]
    cb, it = fit_cb(tr_idx); P[f"CatBoost"] = cb.predict_proba(cb_frame(te_idx))[:, 1]
    P["Logistic regression"] = lr_fit_predict(tr_idx, te_idx)
    P["Ensemble (tuned LGBM + CatBoost + LR)"] = rank(P["LightGBM tuned"]) + rank(P["CatBoost"]) + rank(P["Logistic regression"])
    P["Ensemble (tuned LGBM + CatBoost)"] = rank(P["LightGBM tuned"]) + rank(P["CatBoost"])
    if os.environ.get("SAVE_PREDS"):
        pd.DataFrame({"ccn": ds["ccn"].values[te_idx], "t": ds["t"].values[te_idx], "y": yy, **P}).to_csv(f"{OUT}/preds_{stage}.csv", index=False)
    results[stage] = pd.DataFrame({k: mets(yy, v) for k, v in P.items()}).T
    results[stage].to_csv(f"{OUT}/boost_{stage}.csv")
    log(f"{stage} ({len(te_idx):,} rows, base rate {yy.mean():.3f}; CatBoost iters {it}):\n{results[stage].round(4)}")
log("DONE")
