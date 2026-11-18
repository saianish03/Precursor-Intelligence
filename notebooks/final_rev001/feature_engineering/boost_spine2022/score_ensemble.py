"""Score rows with the saved y_g12 ensemble (tuned LightGBM + CatBoost + LR) written by save_models.py.
usage: python score_ensemble.py <models_dir> <input_csv> <output_csv>
As a library: bundle = load(models_dir); out = score(bundle, df)
ensemble_rank = mean of the 3 models' percentile ranks (rankdata / n) within the scored batch.
"""
import os, sys, json, warnings
import numpy as np, pandas as pd, joblib
from scipy.stats import rankdata
from catboost import CatBoostClassifier
warnings.filterwarnings("ignore")


def num(s):
    """Same numeric detection as boost_models.py (None -> not numeric)."""
    if pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s): return pd.to_numeric(s, errors="coerce").astype(float)
    v = set(s.dropna().unique().tolist())
    if v and v <= {True, False, "True", "False"}: return s.map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0}).astype(float)
    return None


def to_num(s):
    """Numeric conversion for a feature that was numeric at fit time (falls back to coercion, e.g. an all-NaN object column)."""
    n = num(s)
    return n if n is not None else pd.to_numeric(s, errors="coerce").astype(float)


def model_frame(df, feats, cat, levels):
    """Feature frame as in boost_models.py: numerics as float, categoricals as pd.Categorical with the DEV levels."""
    X = pd.DataFrame(index=df.index)
    for c in feats:
        X[c] = pd.Categorical(df[c].astype(str), categories=levels[c]) if c in cat else to_num(df[c])
    return X


def cb_frame(X, cat):
    Z = X.copy()
    for c in cat: Z[c] = Z[c].astype(object).fillna("missing").astype(str)
    return Z


def lr_matrix(X, raw, pre):
    """LR design matrix: median impute + scaler on numerics, get_dummies (raw values as str) on categoricals."""
    Xn = X[pre["numf"]]
    Xc = pd.get_dummies(raw[pre["cat"]].astype(str)) if pre["cat"] else pd.DataFrame(index=Xn.index)
    return np.c_[pre["scaler"].transform(Xn.fillna(pre["medians"])), Xc.reindex(columns=pre["dummy_cols"], fill_value=0).values]


def rank(s): return rankdata(s) / len(s)


def load(models_dir):
    meta = json.load(open(os.path.join(models_dir, "ensemble_meta.json"), encoding="utf-8"))
    cb = CatBoostClassifier(); cb.load_model(os.path.join(models_dir, "catboost.cbm"))
    return dict(meta=meta, lgbm=joblib.load(os.path.join(models_dir, "lgbm_tuned.joblib")), catboost=cb,
                logreg=joblib.load(os.path.join(models_dir, "logreg.joblib")))


def score(bundle, df):
    meta = bundle["meta"]; feats, cat, levels = meta["features"], meta["categorical_features"], meta["categorical_levels"]
    miss = [c for c in feats if c not in df.columns]
    if miss: raise KeyError(f"input is missing {len(miss)} features: {miss[:10]}")
    X = model_frame(df, feats, cat, levels)
    lr = bundle["logreg"]
    out = pd.DataFrame(index=df.index)
    out["p_lgbm"] = bundle["lgbm"].predict_proba(X)[:, 1]
    out["p_catboost"] = bundle["catboost"].predict_proba(cb_frame(X, cat))[:, 1]
    out["p_logreg"] = lr["model"].predict_proba(lr_matrix(X, df, lr))[:, 1]
    out["ensemble_rank"] = (rank(out["p_lgbm"].values) + rank(out["p_catboost"].values) + rank(out["p_logreg"].values)) / 3
    return out


if __name__ == "__main__":
    MODELS, INP, OUTP = sys.argv[1:4]
    b = load(MODELS)
    keep = [c for c in ("ccn", "t") if c in pd.read_csv(INP, nrows=0).columns]
    need = set(keep + b["meta"]["features"])
    df = pd.read_csv(INP, usecols=lambda c: c in need, dtype={"ccn": str, "t": str}, low_memory=False)
    res = pd.concat([df[keep], score(b, df)], axis=1)
    res.to_csv(OUTP, index=False)
    print(f"scored {len(res):,} rows -> {OUTP}")
