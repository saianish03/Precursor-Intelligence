"""Rerun the notebook's B9 feature selection + Part C decision on the FULL-gold engineered CSV (270 cols).

Same method as Feature_Engineering_EDA.ipynb cells 22/24: hold out 20 % of DEV facilities; on the rest, drop near-constant and
|rho|>=.95 duplicates, then keep features whose LightGBM permutation importance (delta PR-AUC) is positive in >=2 of 3
facility-grouped folds. Then v1 (133 pipeline features) vs selected on the holdout, and a test-month report.
Writes reports + a slim model-ready CSV (ccn, t, label/split columns, selected features) for all rows.
usage: python select_full_gold.py <engineered_csv> <feature_registry.yaml> <out_dir>
"""
import os, sys, time, warnings
import numpy as np, pandas as pd, yaml
warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

CSV, REGISTRY, OUT = sys.argv[1:4]
os.makedirs(OUT, exist_ok=True)
LABEL, SEED = "y_g12", 42
STATUS, SPLIT = "status_g12", "split_g12"
t0 = time.time()
def log_(m): print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)

def as_num(s):
    if pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").astype(float)
    vals = set(s.dropna().unique().tolist())
    if vals and vals <= {True, False}:
        return s.map({True: 1.0, False: 0.0}).astype(float)
    if vals and vals <= {"True", "False"}:
        return s.map({"True": 1.0, "False": 0.0}).astype(float)
    return None

ALL_COLS = list(pd.read_csv(CSV, nrows=0).columns)
GOLD_COLS, NEW = ALL_COLS[:150], ALL_COLS[150:]
_reg = yaml.safe_load(open(REGISTRY, encoding="utf-8"))["features"]
reg = _reg if isinstance(_reg, dict) else {f["name"]: f for f in _reg}
MODEL_FEATS = [c for c, v in reg.items() if v["model_use"] == "model" and c in GOLD_COLS]
LABEL_LIKE = [c for c in GOLD_COLS if c.startswith(("y_", "status_", "split_")) or c.endswith("_fwd") or c == "exited_within_12m"]

# ---- load only labeled DEV + TEST rows, chunked ----
parts = []
for ch in pd.read_csv(CSV, dtype={"ccn": str, "t": str}, chunksize=40000, low_memory=False):
    ch = ch[ch[STATUS].isin(["positive", "negative"]) & ch[SPLIT].isin(["train", "calib", "test"])]
    if len(ch): parts.append(ch)
ds = pd.concat(parts, ignore_index=True).sort_values(["ccn", "t"]).reset_index(drop=True); del parts
DEV, TEST = ds[ds[SPLIT].isin(["train", "calib"])], ds[ds[SPLIT] == "test"]
y_dev = DEV[LABEL].astype(float).astype(int).values
log_(f"loaded DEV {len(DEV):,} ({DEV.t.min()}..{DEV.t.max()}), TEST {len(TEST):,} ({sorted(TEST.t.unique())}); {len(MODEL_FEATS)} v1 feats, {len(NEW)} engineered")
FULL = ds

def GBM(): return lgb.LGBMClassifier(n_estimators=400, learning_rate=.05, num_leaves=31, min_child_samples=100, subsample=.8,
                                     subsample_freq=1, colsample_bytree=.8, verbose=-1, random_state=SEED, n_jobs=2)

sel_i, hold_i = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=SEED).split(DEV, y_dev, groups=DEV["ccn"]))
SEL, HOLD = DEV.index[sel_i], DEV.index[hold_i]
y_sel, y_hold = y_dev[sel_i], y_dev[hold_i]
log_(f"selection rows {len(SEL):,} | holdout {len(HOLD):,} ({DEV.loc[HOLD, 'ccn'].nunique():,} facilities, {y_hold.mean():.1%} positive)")

DROP_FROM_V1 = ["state"]
cand = [c for c in MODEL_FEATS if c not in DROP_FROM_V1] + [c for c in NEW if c not in MODEL_FEATS]
assert not {"state", "state_v"} & set(cand)
N_CAND = len(cand)

NUMCACHE = {}
def frame(rows_idx, cols, levels=None):
    X = pd.DataFrame(index=rows_idx); levels = {} if levels is None else levels
    for c in cols:
        if c not in NUMCACHE: NUMCACHE[c] = as_num(FULL[c])
        n = NUMCACHE[c]
        if n is not None:
            X[c] = n.loc[rows_idx]
        else:
            levels.setdefault(c, sorted(FULL.loc[DEV.index, c].dropna().astype(str).unique()))
            X[c] = pd.Categorical(FULL.loc[rows_idx, c].astype(str), categories=levels[c])
    return X, levels

log = []
Xd, LV = frame(SEL, cand)
const = [c for c in cand if Xd[c].value_counts(normalize=True, dropna=False).iloc[0] > .995]
log += [dict(feature=c, step="filter", decision="drop",
             reason="all/most missing in DEV" if Xd[c].isna().mean() > .995 else "near-constant (>99.5% one value)") for c in const]
cand = [c for c in cand if c not in const]
numc = [c for c in cand if pd.api.types.is_numeric_dtype(Xd[c])]
cc = Xd[numc].corr(method="spearman").abs()
def single_ap(c):
    x = Xd[c].astype(float); x = x.fillna(x.min() - 1 if x.notna().any() else 0)
    return max(average_precision_score(y_sel, x), average_precision_score(y_sel, -x))
ap = {c: single_ap(c) for c in numc}
dup = set()
for i, a in enumerate(numc):
    for b in numc[i + 1:]:
        if a not in dup and b not in dup and cc.loc[a, b] >= .95:
            loser = a if ap[a] < ap[b] else b; dup.add(loser)
            log.append(dict(feature=loser, step="filter", decision="drop", reason=f"|rho|>=.95 with {b if loser == a else a} (lower single-feature PR-AUC)"))
cand = [c for c in cand if c not in dup]
del Xd, cc
log_(f"filter: {N_CAND} -> {len(cand)} ({len(const)} near-constant, {len(dup)} duplicates)")

imp_rows = []
for k, (tr_i, va_i) in enumerate(GroupKFold(n_splits=3).split(SEL, y_sel, groups=DEV.loc[SEL, "ccn"])):
    Xtr, lv = frame(SEL[tr_i], cand); Xva, _ = frame(SEL[va_i], cand, lv)
    m = GBM().fit(Xtr, y_sel[tr_i])
    pi = permutation_importance(m, Xva, y_sel[va_i], scoring="average_precision", n_repeats=3, random_state=SEED, n_jobs=1)
    gain = m.booster_.feature_importance("gain")
    imp_rows += [dict(fold=k, feature=c, perm=pi.importances_mean[j], gain=gain[j]) for j, c in enumerate(cand)]
    log_(f"fold {k + 1}/3 permutation importance done")
IMP = pd.DataFrame(imp_rows)
stab = IMP.groupby("feature").agg(perm_mean=("perm", "mean"), perm_min=("perm", "min"),
                                  folds_positive=("perm", lambda s: int((s > 0).sum())), gain_mean=("gain", "mean")).sort_values("perm_mean", ascending=False)
keep = stab[(stab.folds_positive >= 2) & (stab.perm_mean > 0)].index.tolist()
for c in cand:
    if c not in keep:
        log.append(dict(feature=c, step="permutation", decision="drop", reason=f"not useful in >=2 of 3 folds (mean dPR-AUC {stab.loc[c, 'perm_mean']:.4f})"))
for c in keep:
    log.append(dict(feature=c, step="final", decision="keep", reason=f"mean dPR-AUC {stab.loc[c, 'perm_mean']:.4f}, positive in {stab.loc[c, 'folds_positive']}/3 folds"))
SELECTED = keep
pd.DataFrame(log).to_csv(f"{OUT}/B9_feature_decisions_log_full.csv", index=False)
stab.to_csv(f"{OUT}/B9_importance_stability_full.csv")
log_(f"candidates {N_CAND} -> after filter {len(cand)} -> selected {len(SELECTED)} "
     f"({sum(c in NEW for c in SELECTED)} engineered, {sum(c not in NEW for c in SELECTED)} original)")

# ---- Part C ----
def p_at(y, s, f):
    o = np.argsort(-s, kind="stable"); return float(y[o[:max(1, int(len(y) * f))]].mean())
V1 = MODEL_FEATS
dec = {}
for name, cols in [("v1", V1), ("selected", SELECTED)]:
    Xtr, lv = frame(SEL, cols); Xho, _ = frame(HOLD, cols, lv)
    s = GBM().fit(Xtr, y_sel).predict_proba(Xho)[:, 1]
    dec[name] = dict(n_features=len(cols), pr_auc=average_precision_score(y_hold, s), p_at_5=p_at(y_hold, s, .05))
DEC = pd.DataFrame(dec).T
ACCEPT = bool(DEC.loc["selected", "pr_auc"] > DEC.loc["v1", "pr_auc"] and DEC.loc["selected", "p_at_5"] > DEC.loc["v1", "p_at_5"])
DEC.assign(accepted_selected=ACCEPT).to_csv(f"{OUT}/C_decision_holdout_full.csv")
log_(f"holdout decision:\n{DEC.round(4)}\naccept selected: {ACCEPT}")

y_te = TEST[LABEL].astype(float).astype(int).values
def metrics(name, s):
    s = np.asarray(s, float); ok = ~np.isnan(s)
    return dict(model=name, roc_auc=roc_auc_score(y_te[ok], s[ok]), pr_auc=average_precision_score(y_te[ok], s[ok]),
                p_at_5=p_at(y_te[ok], s[ok], .05), p_at_10=p_at(y_te[ok], s[ok], .10))
res = []
if "overall_rating" in TEST:
    res.append(metrics("baseline: overall star rating", -as_num(TEST["overall_rating"]).fillna(3)))
res.append(metrics("baseline: severity rule", (as_num(TEST["n_cit_GI_24m"]) + 2 * as_num(TEST["n_cit_JL_24m"]) + .1 * as_num(TEST["n_cit_24m"])).values))
def lr_frame(idx, cols, fit=None):
    num = [c for c in cols if NUMCACHE.get(c) is not None]; cat = [c for c in cols if c not in num]
    Xn = pd.DataFrame({c: NUMCACHE[c].loc[idx] for c in num})
    Xc_ = pd.get_dummies(FULL.loc[idx, cat].astype(str), dummy_na=False) if cat else pd.DataFrame(index=idx)
    if fit is None:
        fit = dict(med=Xn.median(), cols=Xc_.columns); fit["sc"] = StandardScaler().fit(Xn.fillna(fit["med"]))
    Xn = fit["sc"].transform(Xn.fillna(fit["med"])); Xc_ = Xc_.reindex(columns=fit["cols"], fill_value=0)
    return np.c_[Xn, Xc_.values], fit
Xl, fit = lr_frame(DEV.index, SELECTED); Xlt, _ = lr_frame(TEST.index, SELECTED, fit)
res.append(metrics("logistic regression - selected", LogisticRegression(max_iter=2000, C=.5).fit(Xl, y_dev).predict_proba(Xlt)[:, 1]))
del Xl, Xlt
for name, cols in [("GBM - v1 (133 pipeline features)", V1), (f"GBM - selected ({len(SELECTED)})", SELECTED)]:
    Xtr, lv = frame(DEV.index, cols); Xte, _ = frame(TEST.index, cols, lv)
    res.append(metrics(name, GBM().fit(Xtr, y_dev).predict_proba(Xte)[:, 1]))
R = pd.DataFrame(res).set_index("model"); R.to_csv(f"{OUT}/C_test_comparison_full.csv")
log_(f"test months {sorted(TEST.t.unique())} (base rate {y_te.mean():.3f}):\n{R.round(3)}")

yaml.safe_dump({"label": LABEL, "source_csv": os.path.basename(CSV), "fitted_on_months": sorted(DEV["t"].unique()),
                "n_candidates": N_CAND, "n_after_filter": len(cand), "n_selected": len(SELECTED), "accepted_over_v1": ACCEPT,
                "decision_holdout": DEC.round(4).to_dict(orient="index"),
                "features": [{"name": c, "origin": "engineered" if c in NEW else "pipeline",
                              "perm_mean": round(float(stab.loc[c, "perm_mean"]), 5), "folds_positive": int(stab.loc[c, "folds_positive"])}
                             for c in SELECTED]},
               open(f"{OUT}/feature_set_selected_full.yaml", "w", encoding="utf-8"), sort_keys=False)

# ---- slim model-ready CSV for all rows ----
del ds, DEV, TEST, FULL, NUMCACHE
keep_cols = ["ccn", "t"] + [c for c in LABEL_LIKE if c not in ("ccn", "t")] + SELECTED
slim = os.path.join(OUT, "facility_month_dataset_selected.csv")
if os.path.exists(slim): os.remove(slim)
n = 0
for i, ch in enumerate(pd.read_csv(CSV, dtype={"ccn": str, "t": str}, usecols=keep_cols, chunksize=50000, low_memory=False)):
    ch[keep_cols].to_csv(slim, mode="a", header=(i == 0), index=False); n += len(ch)
log_(f"DONE slim CSV {n:,} rows x {len(keep_cols)} cols -> {slim} ({os.path.getsize(slim) / 2**20:.0f} MB)")
