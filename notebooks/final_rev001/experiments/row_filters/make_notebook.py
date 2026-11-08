"""Writes finetune_rows.ipynb: row-curation experiment on the spine-2022 gold (see report.md, FE_Rows.md).

Run:  python make_notebook.py
Env:  FT_SMOKE=1 runs on 10 % of facilities with a small model (smoke test).
"""
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# rev002: fine-tune the training rows (spine-2022 gold, `y_g12`)

Each filter **removes rows from training only** (one change per experiment, per `FE_Rows.md`). Validation and test rows are never filtered.
The reasons for each filter are in `report.md`, section 3.

- **CV (decides):** 3 folds grouped by facility. Train = fold-train facilities, months 2022-01 → 2023-06 (filtered).
  Validate = fold-held-out facilities, months 2023-07 → 2023-09 (unfiltered, later in time).
- **Test (reported only):** train on all DEV 2022-01 → 2023-09 (filtered), score 2024-10 → 11 (unfiltered).
- **Model:** tuned LightGBM with fixed parameters on the fixed 75 selected features.
- **Keep rule:** mean Δ PR-AUC > 0 and > 1 std of the per-fold Δ.
""")

code(r"""
import os, sys, json, time, warnings, glob
from pathlib import Path
import numpy as np, pandas as pd, yaml
import pyarrow.parquet as pq
import lightgbm as lgb
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, f1_score, accuracy_score, precision_recall_curve
from sklearn.model_selection import GroupKFold
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

HERE = Path.cwd()
REPO = HERE.parents[1]
FER = REPO.parent / "feature engineering run" / "spine2022"
CSV = Path(os.environ.get("FT_CSV", FER / "facility_month_dataset_selected_spine2022.csv"))
FS = FER / "feature_set_selected75.yaml"
PARAMS = FER / "boost" / "lgbm_best_params.json"
GOLD = REPO / "data" / "gold" / "facility_month_dataset"
SMOKE = os.environ.get("FT_SMOKE") == "1"
OUT = HERE / ("outputs_smoke" if SMOKE else "outputs"); (OUT / "figures").mkdir(parents=True, exist_ok=True)
LABEL, STATUS, SPLIT, SEED, NFOLD = "y_g12", "status_g12", "split_g12", 42, 3
TRAIN_END, VAL_START = "2023-06", "2023-07"
sys.path.insert(0, str(FER / "boost"))
from score_ensemble import num, model_frame
t0 = time.time()
def log(m): print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)
log(f"csv={CSV.name} smoke={SMOKE} out={OUT.name}")
""")

code(r"""
# ---- load: 75 features + labels (selected CSV), joined with the gold columns the filters need ----
FEATS = [f["name"] for f in yaml.safe_load(open(FS, encoding="utf-8"))["features"]]
need = set(["ccn", "t", SPLIT, LABEL, STATUS] + FEATS)
parts = []
for ch in pd.read_csv(CSV, dtype={"ccn": str, "t": str}, usecols=lambda c: c in need, chunksize=50000, low_memory=False):
    ch = ch[ch[STATUS].isin(["positive", "negative"]) & ch[SPLIT].isin(["train", "calib", "test"])]
    f64 = ch.select_dtypes("float64").columns
    if len(f64): ch[f64] = ch[f64].astype("float32")
    parts.append(ch[["ccn", "t", SPLIT, LABEL] + FEATS])
ds = pd.concat(parts, ignore_index=True); del parts

AUX = ["no_std_on_record", "most_recent_inspection_over_2y", "qm_missing_count", "hprd_total",
       "certified_beds", "provider_in_hospital", "facility_age_years"]
aux = pd.concat([pq.read_table(f, columns=["ccn", "t"] + AUX).to_pandas() for f in sorted(glob.glob(f"{GOLD}/t=*/*.parquet"))],
                ignore_index=True)
aux["t"] = aux["t"].astype(str)
aux = aux.rename(columns={c: f"aux_{c}" for c in AUX})
ds = ds.merge(aux, on=["ccn", "t"], how="left", validate="one_to_one"); del aux
assert ds["aux_certified_beds"].notna().mean() > .95, "gold join failed"
ds = ds.sort_values(["ccn", "t"]).reset_index(drop=True)
if SMOKE:
    keep = pd.Series(ds["ccn"].unique()).sample(frac=.1, random_state=SEED)
    ds = ds[ds["ccn"].isin(keep)].reset_index(drop=True)

CAT = [c for c in FEATS if num(ds[c]) is None]
t = ds["t"].values
dev_mask = ds[SPLIT].isin(["train", "calib"]).values
LEVELS = {c: sorted(ds.loc[dev_mask, c].dropna().astype(str).unique().tolist()) for c in CAT}
X = model_frame(ds, FEATS, CAT, LEVELS)
y = ds[LABEL].astype(float).astype(int).values
g = ds["ccn"].values
DEV = np.where(dev_mask)[0]; TEST = np.where(ds[SPLIT].values == "test")[0]
FOLDS = []
for tr_i, va_i in GroupKFold(n_splits=NFOLD).split(DEV, groups=g[DEV]):
    tr, va = DEV[tr_i], DEV[va_i]
    FOLDS.append((tr[t[tr] <= TRAIN_END], va[t[va] >= VAL_START]))
P0 = dict(json.load(open(PARAMS, encoding="utf-8"))["params"]); P0["n_jobs"] = os.cpu_count()
if SMOKE: P0["n_estimators"] = 150
PN = dict(P0, n_estimators=300, learning_rate=.04)   # cheaper model, only to score label noise
if SMOKE: PN["n_estimators"] = 80
log(f"DEV {len(DEV):,}, TEST {len(TEST):,}; {len(FEATS)} features ({len(CAT)} categorical); folds (train/val rows): "
    + ", ".join(f"{len(a):,}/{len(b):,}" for a, b in FOLDS))
""")

code(r"""
# ---- row filters: each returns a boolean mask over ds (True = remove from training). Reasons: report.md section 3 ----
A = lambda c: ds[f"aux_{c}"]
def is_true(s): return s.astype(str).str.lower().isin(["true", "1", "1.0", "y", "yes"]).values
mi = pd.PeriodIndex(ds["t"], freq="M").astype("int64")

STATIC = {
    "R1 drop_2022H1": (t >= "2022-01") & (t <= "2022-06"),
    "R2 drop_2022": t < "2023-01",
    # R3 drop_no_std removed: no labeled row lacks a standard survey (report.md section 3)
    "R4 drop_stale_insp": is_true(A("most_recent_inspection_over_2y")),
    "R5 drop_missing_staffing": A("hprd_total").isna().values,
    "R6 drop_missing_qm": (pd.to_numeric(A("qm_missing_count"), errors="coerce").fillna(0) >= 10).values,
    "R7 drop_tiny_or_hospital": (pd.to_numeric(A("certified_beds"), errors="coerce") < 20).values | is_true(A("provider_in_hospital")),
    "R8 drop_new_facility": (pd.to_numeric(A("facility_age_years"), errors="coerce") < 2).values,
    "R9 quarterly_thin": (mi % 3) != 0,
}
NOISE = {"R10 noise_2pct": 2, "R11 noise_5pct": 5}
FILTERS = ["R0 baseline"] + list(STATIC) + list(NOISE)

def noise_mask(train_idx, pct):
    # confident learning: out-of-fold loss inside the training rows only (3 inner folds grouped by facility)
    oof = np.empty(len(train_idx))
    for a, b in GroupKFold(n_splits=3).split(train_idx, groups=g[train_idx]):
        m = lgb.LGBMClassifier(**PN).fit(X.iloc[train_idx[a]], y[train_idx[a]])
        oof[b] = m.predict_proba(X.iloc[train_idx[b]])[:, 1]
    yy = y[train_idx]
    loss = -(yy * np.log(np.clip(oof, 1e-6, 1)) + (1 - yy) * np.log(np.clip(1 - oof, 1e-6, 1)))
    out = np.zeros(len(ds), bool); out[train_idx[loss >= np.percentile(loss, 100 - pct)]] = True
    return out

_noise_cache = {}
def removed(name, train_idx, key):
    if name == "R0 baseline": return np.zeros(len(train_idx), bool)
    if name in STATIC: return STATIC[name][train_idx]
    if key not in _noise_cache:               # one inner-OOF scoring per training set, shared by 2 % and 5 %
        _noise_cache[key] = {p: noise_mask(train_idx, p) for p in NOISE.values()}
    return _noise_cache[key][NOISE[name]][train_idx]

# coverage on all DEV rows
cov = []
for k in FILTERS[1:]:
    if k in STATIC:
        m = STATIC[k][DEV]
        cov.append(dict(filter=k, removed_dev_rows=int(m.sum()), removed_pct=m.mean(), removed_pos_rate=y[DEV][m].mean() if m.any() else np.nan))
cov = pd.DataFrame(cov); cov.to_csv(OUT / "filter_coverage.csv", index=False)
print(f"DEV positive rate {y[DEV].mean():.3f}"); print(cov.round(3).to_string(index=False))
""")

code(r"""
# ---- metrics (FE_Rows.md section 1) ----
def recall_at(yy, s, f):
    o = np.argsort(-s, kind="stable"); return float(yy[o[:max(1, int(len(yy) * f))]].sum() / max(1, yy.sum()))
def best_threshold(yy, s):
    p, r, th = precision_recall_curve(yy, s); f = 2 * p * r / np.clip(p + r, 1e-12, None)
    return float(th[np.nanargmax(f[:-1])])
def mets(yy, s, thr):
    pred = (s >= thr).astype(int)
    return dict(pr_auc=average_precision_score(yy, s), roc_auc=roc_auc_score(yy, s), recall_at_5=recall_at(yy, s, .05),
                recall_at_10=recall_at(yy, s, .10), f1=f1_score(yy, pred), brier=brier_score_loss(yy, s),
                accuracy=accuracy_score(yy, pred), threshold=thr)

# ---- CV: one LightGBM per (filter, fold) ----
cv_rows, VALPRED = [], {}
for fi, (tr, va) in enumerate(FOLDS):
    for k in FILTERS:
        rm = removed(k, tr, ("fold", fi))
        keep = tr[~rm]
        mdl = lgb.LGBMClassifier(**P0).fit(X.iloc[keep], y[keep])
        s = mdl.predict_proba(X.iloc[va])[:, 1]
        VALPRED[(k, fi)] = s
        r = dict(filter=k, fold=fi, train_rows=len(keep), removed_rows=int(rm.sum()), **mets(y[va], s, best_threshold(y[va], s)))
        cv_rows.append(r)
    log(f"fold {fi + 1}/{NFOLD} done")
cv = pd.DataFrame(cv_rows); cv.to_csv(OUT / "cv_by_fold.csv", index=False)
""")

code(r"""
# ---- CV summary: mean ± std, paired Δ vs baseline on the same fold, keep rule ----
base = cv[cv["filter"] == "R0 baseline"].set_index("fold")["pr_auc"]
cv["delta_pr_auc"] = cv["pr_auc"] - cv["fold"].map(base)
M = ["pr_auc", "roc_auc", "recall_at_5", "recall_at_10", "f1", "brier", "accuracy"]
agg = cv.groupby("filter", sort=False).agg(train_rows=("train_rows", "mean"), removed_rows=("removed_rows", "mean"),
                                           **{f"{m}_mean": (m, "mean") for m in M}, pr_auc_std=("pr_auc", "std"),
                                           delta_mean=("delta_pr_auc", "mean"), delta_std=("delta_pr_auc", "std"),
                                           delta_min=("delta_pr_auc", "min"))
agg["keep"] = (agg["delta_mean"] > 0) & (agg["delta_mean"] > agg["delta_std"])
agg.loc["R0 baseline", "keep"] = False
agg = agg.reset_index(); agg.to_csv(OUT / "cv_summary.csv", index=False)
print(agg[["filter", "train_rows", "pr_auc_mean", "pr_auc_std", "delta_mean", "delta_std", "roc_auc_mean", "recall_at_10_mean",
           "f1_mean", "brier_mean", "keep"]].round(4).to_string(index=False))

fig, ax = plt.subplots(figsize=(9, 5)); rr = agg[agg["filter"] != "R0 baseline"].sort_values("delta_mean")
ax.barh(rr["filter"], rr["delta_mean"], xerr=rr["delta_std"], color=["tab:green" if k else "tab:grey" for k in rr["keep"]])
ax.axvline(0, c="k", lw=.8); ax.set_xlabel("Δ PR-AUC vs baseline (mean ± std over 3 facility-grouped folds)")
ax.set_title("Removing training rows: effect on later-month validation"); plt.tight_layout()
plt.savefig(OUT / "figures" / "cv_delta_pr_auc.png", dpi=120)
""")

code(r"""
# ---- TEST (reported only): train on all DEV with each filter, score 2024-10/11; threshold from pooled CV validation ----
test_rows = []
for k in FILTERS:
    rm = removed(k, DEV, ("dev",))
    keep = DEV[~rm]
    mdl = lgb.LGBMClassifier(**P0).fit(X.iloc[keep], y[keep])
    s = mdl.predict_proba(X.iloc[TEST])[:, 1]
    yv = np.concatenate([y[va] for _, va in FOLDS]); sv = np.concatenate([VALPRED[(k, fi)] for fi in range(NFOLD)])
    test_rows.append(dict(filter=k, train_rows=len(keep), **mets(y[TEST], s, best_threshold(yv, sv))))
    log(f"test {k}: PR-AUC {test_rows[-1]['pr_auc']:.4f}")
test = pd.DataFrame(test_rows)
test["delta_pr_auc"] = test["pr_auc"] - test.loc[test["filter"] == "R0 baseline", "pr_auc"].iloc[0]
test.to_csv(OUT / "test_results.csv", index=False)
print(test.round(4).to_string(index=False))
""")

code(r"""
# ---- experiment log rows (FE_Rows.md section 9) ----
lg = agg.merge(test[["filter", "pr_auc", "delta_pr_auc"]].rename(columns={"pr_auc": "test_pr_auc", "delta_pr_auc": "test_delta"}), on="filter")
lines = ["| ID | Change (one only) | Model | #Features | PR-AUC (CV mean ± std) | ROC-AUC | Recall@10% | Brier | Δ PR-AUC vs baseline (± std) | Test PR-AUC (Δ) | Keep? |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
for _, r in lg.iterrows():
    fid, name = r["filter"].split(" ", 1)
    lines.append(f"| {fid} | {name} (train rows {r['train_rows']:,.0f}) | LightGBM tuned | {len(FEATS)} | {r['pr_auc_mean']:.4f} ± {r['pr_auc_std']:.4f} | "
                 f"{r['roc_auc_mean']:.4f} | {r['recall_at_10_mean']:.4f} | {r['brier_mean']:.4f} | "
                 f"{r['delta_mean']:+.4f} ± {0 if pd.isna(r['delta_std']) else r['delta_std']:.4f} | {r['test_pr_auc']:.4f} ({r['test_delta']:+.4f}) | "
                 f"{'baseline' if fid == 'R0' else ('yes' if r['keep'] else 'no')} |")
(OUT / "experiment_log.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines)); log("DONE")
""")

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "precursor-exp", "display_name": "precursor-exp", "language": "python"}})
nbf.write(nb, HERE / "finetune_rows.ipynb")
print("wrote", HERE / "finetune_rows.ipynb")
