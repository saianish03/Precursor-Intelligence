"""Feature selection on the spine-2022 engineered gold (same method as Feature_Engineering_EDA.ipynb cells 22 and 24).

usage: python select_spine2022.py <engineered_csv> <feature_registry_yaml> <out_dir>

- DEV = labeled g12 train + calib (2022-01 -> 2023-09); TEST = labeled g12 test (2024-10, 2024-11), reported only.
- 20 % of DEV facilities are held out (GroupShuffleSplit, seed 42) for the v1-vs-selected decision; selection sees only SEL.
- Candidates = v1 model features minus raw `state` (bias mitigation) + every engineered column.
- 1) filter: near-constant (> 99.5 % one value) and near-duplicate (|Spearman rho| >= .95, keep the higher single-feature PR-AUC)
  2) LightGBM + permutation importance (Delta PR-AUC) on 3 facility-grouped folds of SEL; keep if positive in >= 2 of 3 folds.
- Memory changes vs the notebook (8 GB laptop): floats as float32, Spearman correlation on a 100k-row SEL sample,
  permutation importance on at most 40k validation rows per fold, parallel in threads (N_JOBS, default all cores).
"""
import sys, time, warnings
import numpy as np, pandas as pd, yaml
warnings.filterwarnings("ignore")
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb, os
from joblib import parallel_backend
from concurrent.futures import ThreadPoolExecutor
NJ = int(os.environ.get("N_JOBS", os.cpu_count() or 4))   # threads share memory (no data copies)

CSV, REG, OUT = sys.argv[1:4]
OUT = Path(OUT); OUT.mkdir(parents=True, exist_ok=True)
LABEL, STATUS, SPLIT, SEED = "y_g12", "status_g12", "split_g12", 42
CORR_SAMPLE, PERM_VAL_CAP = 100_000, 40_000
t0 = time.time()
def log(m): print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)

# ---- columns ----
reg = yaml.safe_load(open(REG, encoding="utf-8"))["features"]
reg = reg if isinstance(reg, dict) else {f["name"]: f for f in reg}
header = pd.read_csv(CSV, nrows=0).columns.tolist()
MODEL_FEATS = [c for c, v in reg.items() if v["model_use"] == "model" and c in header]
GOLD_NONFEAT = {"ccn", "t", "cutoff_date", "chain_id", "y_g12", "status_g12", "y_g6", "status_g6", "y_pen12", "status_pen12",
                "split_g12", "split_g6", "split_pen12"}
NEW = [c for c in header if c not in reg and c not in GOLD_NONFEAT]
BASELINE_COLS = ["overall_rating", "n_cit_GI_24m", "n_cit_JL_24m", "n_cit_24m"]
cand0 = [c for c in MODEL_FEATS if c != "state"] + [c for c in NEW if c not in MODEL_FEATS]
assert not {"state", "state_v"} & set(cand0), "raw state must stay out of the candidates (bias mitigation)"
use = list(dict.fromkeys(["ccn", "t", SPLIT, STATUS, LABEL] + MODEL_FEATS + NEW + BASELINE_COLS))
log(f"{len(MODEL_FEATS)} v1 features, {len(NEW)} engineered, {len(cand0)} candidates")

# ---- load labeled train / calib / test rows only ----
parts = []
for ch in pd.read_csv(CSV, usecols=use, dtype={"ccn": str, "t": str}, chunksize=40_000, low_memory=False):
    ch = ch[ch[STATUS].isin(["positive", "negative"]) & ch[SPLIT].isin(["train", "calib", "test"])]
    if len(ch):
        for c in ch.columns:
            if ch[c].dtype == "float64": ch[c] = ch[c].astype("float32")
        parts.append(ch)
ds = pd.concat(parts, ignore_index=True); del parts
ds = ds.sort_values(["ccn", "t"]).reset_index(drop=True)
y = ds[LABEL].astype(float).astype(int).values
g = ds["ccn"].values
DEV = np.where(ds[SPLIT].isin(["train", "calib"]).values)[0]
TEST = np.where((ds[SPLIT] == "test").values)[0]
log(f"loaded {len(ds):,} rows: DEV {len(DEV):,} ({ds.t.iloc[DEV].min()}..{ds.t.iloc[DEV].max()}), TEST {len(TEST):,}; "
    f"{ds.memory_usage(deep=True).sum() / 1e9:.2f} GB")

def as_num(s):
    if pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s): return pd.to_numeric(s, errors="coerce").astype("float32")
    v = set(s.dropna().unique().tolist())
    if v and v <= {True, False, "True", "False"}: return s.map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0}).astype("float32")
    return None

# one model-ready frame for all columns: numerics float32, categoricals with DEV-fitted levels
X = pd.DataFrame(index=ds.index)
for c in dict.fromkeys(MODEL_FEATS + cand0):
    n = as_num(ds[c])
    X[c] = n if n is not None else pd.Categorical(ds[c].astype(str), categories=sorted(ds[c].iloc[DEV].dropna().astype(str).unique()))
CATS = [c for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype)]

sel_i, hold_i = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=SEED).split(DEV, y[DEV], groups=g[DEV]))
SEL, HOLD = DEV[sel_i], DEV[hold_i]
log(f"SEL {len(SEL):,} rows | HOLD {len(HOLD):,} rows ({len(set(g[HOLD])):,} facilities, {y[HOLD].mean():.1%} positive)")

def GBM(): return lgb.LGBMClassifier(n_estimators=400, learning_rate=.05, num_leaves=31, min_child_samples=100, subsample=.8,
                                     subsample_freq=1, colsample_bytree=.8, verbose=-1, random_state=SEED, n_jobs=NJ)

# ---- 1) filter ----
dlog, cand = [], list(cand0)
Xs = X.iloc[SEL]; ys = y[SEL]
const = []
for c in cand:
    top = Xs[c].value_counts(normalize=True, dropna=False)
    if (top.iloc[0] if len(top) else 1) > .995: const.append(c)
for c in const:
    dlog.append(dict(feature=c, step="filter", decision="drop",
                     reason="all/most missing in DEV" if Xs[c].isna().mean() > .995 else "near-constant (>99.5% one value)"))
cand = [c for c in cand if c not in const]
numc = [c for c in cand if c not in CATS]
samp = np.random.RandomState(SEED).choice(len(SEL), min(CORR_SAMPLE, len(SEL)), replace=False)
cc = Xs.iloc[samp][numc].astype("float64").corr(method="spearman").abs()
def single_ap(c):
    x = Xs[c].astype("float64"); x = x.fillna(x.min() - 1 if x.notna().any() else 0)
    return max(average_precision_score(ys, x), average_precision_score(ys, -x))
with ThreadPoolExecutor(NJ) as ex: ap = dict(zip(numc, ex.map(single_ap, numc)))
dup = set()
for i, a in enumerate(numc):
    for b in numc[i + 1:]:
        if a not in dup and b not in dup and cc.loc[a, b] >= .95:
            loser = a if ap[a] < ap[b] else b; dup.add(loser)
            dlog.append(dict(feature=loser, step="filter", decision="drop",
                             reason=f"|rho|>=.95 with {b if loser == a else a} (lower single-feature PR-AUC)"))
cand = [c for c in cand if c not in dup]
del cc
log(f"filter: {len(cand0)} -> {len(cand)} ({len(const)} near-constant, {len(dup)} near-duplicate)")

# ---- 2) LightGBM + permutation importance on 3 facility-grouped folds ----
rows = []
for k, (tr_i, va_i) in enumerate(GroupKFold(n_splits=3).split(SEL, ys, groups=g[SEL])):
    m = GBM().fit(Xs.iloc[tr_i][cand], ys[tr_i])
    va = va_i if len(va_i) <= PERM_VAL_CAP else np.sort(np.random.RandomState(SEED + k).choice(va_i, PERM_VAL_CAP, replace=False))
    m.set_params(n_jobs=1)   # one LightGBM thread per permutation worker; NJ workers in threads
    with parallel_backend("threading", n_jobs=NJ):
        pi = permutation_importance(m, Xs.iloc[va][cand], ys[va], scoring="average_precision", n_repeats=3, random_state=SEED, n_jobs=NJ)
    gain = m.booster_.feature_importance("gain")
    rows += [dict(fold=k, feature=c, perm=pi.importances_mean[j], gain=gain[j]) for j, c in enumerate(cand)]
    log(f"fold {k + 1}/3 done (train {len(tr_i):,}, perm rows {len(va):,})")
IMP = pd.DataFrame(rows)
stab = IMP.groupby("feature").agg(perm_mean=("perm", "mean"), perm_min=("perm", "min"),
                                  folds_positive=("perm", lambda s: int((s > 0).sum())), gain_mean=("gain", "mean"))
stab = stab.sort_values("perm_mean", ascending=False)
keep = stab[(stab.folds_positive >= 2) & (stab.perm_mean > 0)].index.tolist()
for c in cand:
    if c not in keep:
        dlog.append(dict(feature=c, step="permutation", decision="drop",
                         reason=f"not useful in >=2 of 3 folds (mean dPR-AUC {stab.loc[c, 'perm_mean']:.4f})"))
for c in keep:
    dlog.append(dict(feature=c, step="final", decision="keep",
                     reason=f"mean dPR-AUC {stab.loc[c, 'perm_mean']:.4f}, positive in {stab.loc[c, 'folds_positive']}/3 folds"))
SELECTED = keep
pd.DataFrame(dlog).to_csv(OUT / "B9_feature_decisions_log_spine2022.csv", index=False)
stab.to_csv(OUT / "B9_importance_stability_spine2022.csv")
n_eng = sum(c in NEW for c in SELECTED)
log(f"selection: {len(cand0)} -> {len(cand)} -> {len(SELECTED)} ({len(SELECTED) - n_eng} pipeline + {n_eng} engineered)")
log("top 10 by dPR-AUC:\n" + stab.head(10)[["perm_mean", "folds_positive"]].round(4).to_string())

# ---- 3) decision: v1 vs selected on HOLD ----
def p_at(yy, s, f): o = np.argsort(-s, kind="stable"); return float(yy[o[:max(1, int(len(yy) * f))]].mean())
V1 = MODEL_FEATS
dec = {}
for name, cols in [("v1", V1), ("selected", SELECTED)]:
    s = GBM().fit(X.iloc[SEL][cols], y[SEL]).predict_proba(X.iloc[HOLD][cols])[:, 1]
    dec[name] = dict(n_features=len(cols), pr_auc=average_precision_score(y[HOLD], s), p_at_5=p_at(y[HOLD], s, .05))
DEC = pd.DataFrame(dec).T
ACCEPT = bool(DEC.loc["selected", "pr_auc"] > DEC.loc["v1", "pr_auc"] and DEC.loc["selected", "p_at_5"] > DEC.loc["v1", "p_at_5"])
DEC.assign(accepted_selected=ACCEPT).to_csv(OUT / "C_decision_holdout_spine2022.csv")
log(f"holdout decision:\n{DEC.round(4)}\n-> {'accept selected' if ACCEPT else 'keep v1'}")

# ---- 4) test (reported only) ----
yt = y[TEST]
def mets(name, s):
    s = np.asarray(s, float); ok = ~np.isnan(s)
    return dict(model=name, roc_auc=roc_auc_score(yt[ok], s[ok]), pr_auc=average_precision_score(yt[ok], s[ok]),
                p_at_5=p_at(yt[ok], s[ok], .05), p_at_10=p_at(yt[ok], s[ok], .10))
T = ds.iloc[TEST]
res = [mets("baseline: overall star rating", -as_num(T["overall_rating"]).fillna(3)),
       mets("baseline: severity rule", (as_num(T["n_cit_GI_24m"]) + 2 * as_num(T["n_cit_JL_24m"]) + .1 * as_num(T["n_cit_24m"])).values)]
num_sel = [c for c in SELECTED if c not in CATS]; cat_sel = [c for c in SELECTED if c in CATS]
def lr_mk(idx, fit=None):
    Xn = X.iloc[idx][num_sel].astype("float64")
    Xc = pd.get_dummies(ds.iloc[idx][cat_sel].astype(str)) if cat_sel else pd.DataFrame(index=Xn.index)
    if fit is None: fit = dict(med=Xn.median(), cols=Xc.columns); fit["sc"] = StandardScaler().fit(Xn.fillna(fit["med"]))
    return np.c_[fit["sc"].transform(Xn.fillna(fit["med"])), Xc.reindex(columns=fit["cols"], fill_value=0).values], fit
A, f = lr_mk(DEV); B, _ = lr_mk(TEST, f)
res.append(mets(f"logistic regression - selected ({len(SELECTED)})", LogisticRegression(max_iter=3000, C=.5).fit(A, y[DEV]).predict_proba(B)[:, 1]))
del A, B
for name, cols in [(f"GBM - v1 ({len(V1)} pipeline features)", V1), (f"GBM - selected ({len(SELECTED)})", SELECTED)]:
    res.append(mets(name, GBM().fit(X.iloc[DEV][cols], y[DEV]).predict_proba(X.iloc[TEST][cols])[:, 1]))
R = pd.DataFrame(res).set_index("model"); R.to_csv(OUT / "C_test_comparison_spine2022.csv")
log(f"test ({len(TEST):,} rows, base rate {yt.mean():.3f}):\n{R.round(4)}")

# ---- 5) feature set record (format read by boost/boost_models.py) ----
yaml.safe_dump({"label": LABEL, "data": "spine 2022 engineered gold", "dev_months": f"{ds.t.iloc[DEV].min()}..{ds.t.iloc[DEV].max()}",
                "dev_rows": int(len(DEV)), "candidates": len(cand0), "after_filter": len(cand), "selected": len(SELECTED),
                "decision": "accept selected" if ACCEPT else "keep v1",
                "features": [dict(name=c, source="engineered" if c in NEW else "pipeline", kind="categorical" if c in CATS else "numeric",
                                  perm_mean=float(stab.loc[c, "perm_mean"])) for c in SELECTED]},
               open(OUT / "feature_set_selected_spine2022.yaml", "w", encoding="utf-8"), sort_keys=False, allow_unicode=True)
log("DONE")
