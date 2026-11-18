"""Apply the notebook's FeatureTransformer (Part B) to the full gold dataset, month by month, and write one CSV.

Same fitting rules as Feature_Engineering_EDA.ipynb: fit on DEV (g12 train + calib, labeled) only, DEV rows get
the out-of-fold state encoding, panel lags use each facility's earlier months over the whole panel.
Output = 150 gold columns + 120 engineered columns.
"""
import glob, os, sys, time
import numpy as np, pandas as pd, yaml

GOLD, REGISTRY, ART, OUT_CSV = sys.argv[1:5]
sys.path.insert(0, ART)
import feature_transform_v2 as ftm

t0 = time.time()
def log(msg):
    print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)

files = sorted(glob.glob(os.path.join(GOLD, "t=*", "part-0.parquet")))
def read(f, columns=None):
    d = pd.read_parquet(f, columns=columns)
    d["t"] = d["t"].astype(str) if "t" in d.columns else f.split("t=")[1][:7]
    return d

# ---- DEV (same definition as the notebook) ----
parts = []
for f in files:
    d = read(f)
    d = d[d["status_g12"].isin(["positive", "negative"]) & d["split_g12"].isin(["train", "calib"])]
    if len(d):
        parts.append(d)
DEV = pd.concat(parts, ignore_index=True).sort_values(["ccn", "t"]).reset_index(drop=True); del parts
_reg = yaml.safe_load(open(REGISTRY, encoding="utf-8"))["features"]
reg = _reg if isinstance(_reg, dict) else {f["name"]: f for f in _reg}
MODEL_FEATS = [c for c, v in reg.items() if v["model_use"] == "model" and c in DEV.columns]
NUM = [c for c in MODEL_FEATS if ftm.as_num(DEV[c]) is not None]
CAT = [c for c in MODEL_FEATS if c not in NUM]
skewed = [c for c in NUM if abs(ftm.as_num(DEV[c]).skew()) > 2 and ftm.as_num(DEV[c]).nunique() > 10]
y_dev = DEV["y_g12"].astype(int).values
log(f"DEV {len(DEV):,} rows, months {DEV['t'].min()}..{DEV['t'].max()} | {len(NUM)} numeric, {len(CAT)} categorical, {len(skewed)} skewed")

ft = ftm.FeatureTransformer(NUM, CAT, skewed, rare_min=30, te_smooth=50, seed=42).fit(DEV, y_dev)
oof = ft.oof_state_encoding(DEV, y_dev, DEV["ccn"])
OOF = pd.Series(oof.values, index=pd.MultiIndex.from_arrays([DEV["ccn"], DEV["t"]]))
del DEV
log("transformer fitted; out-of-fold state encoding done")

# ---- panel lags over the whole panel (slim frame) ----
# same math as ft._panel_lags (shift(3) and rolling(4, min_periods=3).std() per facility), vectorised
ORDER = list(ft.transform(read(files[0]).head(200)).columns)        # engineered column order of the notebook
slim = pd.concat([read(f, ["ccn", "t"] + ft.lag_cols) for f in files], ignore_index=True).sort_values(["ccn", "t"]).reset_index(drop=True)
LAGS = pd.DataFrame(index=slim.index)
for c in ft.lag_cols:
    v = ftm.as_num(slim[c]); g = v.groupby(slim["ccn"])
    LAGS[f"lag3_diff_{c}"] = v - g.shift(3)
    LAGS[f"vol4_{c}"] = g.rolling(4, min_periods=3).std().reset_index(level=0, drop=True)
LAGS.index = pd.MultiIndex.from_arrays([slim["ccn"], slim["t"]]); del slim
ft.lag_cols = []                                                     # month-by-month transform skips the lags; filled from LAGS
log(f"panel lags done: {list(LAGS.columns)}")

# ---- transform month by month and append ----
if os.path.exists(OUT_CSV):
    os.remove(OUT_CSV)
rows, ncols = 0, None
for i, f in enumerate(files):
    d = read(f).reset_index(drop=True)
    X = ft.transform(d)
    key = pd.MultiIndex.from_arrays([d["ccn"], d["t"]])
    X[LAGS.columns] = LAGS.reindex(key).values
    is_dev = key.isin(OOF.index)
    X.loc[is_dev, "state_te"] = OOF.reindex(key[is_dev]).values
    X = X[ORDER]
    out = pd.concat([d, X], axis=1)
    assert not out.columns.duplicated().any()
    ncols = ncols or out.shape[1]
    assert out.shape[1] == ncols
    out.to_csv(OUT_CSV, mode="a", header=(i == 0), index=False)
    rows += len(out)
    log(f"{d['t'].iloc[0]}: {len(out):,} rows (dev rows {int(is_dev.sum()):,}) | total {rows:,}")
log(f"DONE {rows:,} rows x {ncols} columns -> {OUT_CSV} ({os.path.getsize(OUT_CSV) / 2**30:.2f} GB)")
