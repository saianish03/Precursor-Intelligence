"""Feature engineering v2 transformer, written by Feature_Engineering_EDA.ipynb and shipped next to its joblib.

Scoring:
    import sys, joblib; sys.path.insert(0, "<OUT>/artifacts")
    bundle = joblib.load("<OUT>/artifacts/feature_transform_v2.joblib")
    X = bundle["transformer"].transform(panel)   # panel = the month to score + each facility's earlier months (for lag3/vol4)

Every learned parameter and every column list lives on the object, so nothing depends on notebook globals.
"""
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold

LOG_CANDIDATES  = ["sev_points_12m", "sev_points_24m", "n_cit_12m", "n_cit_24m", "n_cit_DF_24m", "max_fine_24m", "denial_days_24m",
                   "n_complaint_cit_12m", "certified_beds", "days_since_last_penalty", "days_since_last_gplus"]
RANK_CANDIDATES = ["hprd_total", "hprd_rn", "turnover_total", "sev_points_24m", "n_complaint_surveys_cited_12m", "n_cit_24m", "qm_ls_falls_major_injury"]
LAG_CANDIDATES  = ["hprd_total", "n_cit_12m", "n_complaint_surveys_cited_12m", "turnover_total"]
ISNA_CANDIDATES = ["hprd_total", "turnover_total", "days_since_last_penalty", "days_since_last_gplus", "qm_ls_falls_major_injury"]
REQUIRED = ["ccn", "t", "state", "certified_beds", "n_cit_12m", "n_cit_24m", "n_complaint_surveys_cited_12m", "n_complaint_surveys_cited_24m",
            "fine_dollars_24m", "days_since_last_std", "gplus_last_std", "std_overdue", "ownership_group", "chain_id", "turnover_total", "hprd_rn"]
BED_BINS, BED_LABELS = [0, 50, 100, 150, 200, 1e6], ["<50", "50-99", "100-149", "150-199", "200+"]
STD_BINS, STD_LABELS = [-1, 180, 365, 456, 730, 1e6], ["<6m", "6-12m", "12-15m", "15-24m", "24m+"]   # inspections due at ~15 months (456 days)


def as_num(s):
    """numeric view: numbers and booleans as float; True/False objects mapped; None if categorical"""
    if pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").astype(float)
    vals = set(s.dropna().unique().tolist())
    if vals and vals <= {True, False}:
        return s.map({True: 1.0, False: 0.0}).astype(float)
    return None


class FeatureTransformer:
    """Fit on DEV rows only; transform any month (Lecture 6: full-pass transforms, training/serving consistency)."""

    def __init__(self, num, cat, skewed, rare_min=30, te_smooth=50, no_vocab=("state",), seed=42):
        self.num, self.cat, self.skewed = list(num), list(cat), list(skewed)
        self.rare_min, self.te_smooth, self.no_vocab, self.seed = rare_min, te_smooth, tuple(no_vocab), seed

    def fit(self, dev, y):
        missing = [c for c in REQUIRED if c not in dev.columns]
        if missing:
            raise KeyError(f"columns the transformer needs are missing from the dataset: {missing}")
        cols = set(dev.columns)
        self.log_cols = [c for c in LOG_CANDIDATES if c in cols]
        self.clip_cols = [c for c in self.num if c in self.skewed or c in self.log_cols]
        self.rank_cols = [c for c in RANK_CANDIDATES if c in cols]
        self.lag_cols = [c for c in LAG_CANDIDATES if c in cols]
        self.isna_cols = [c for c in ISNA_CANDIDATES if c in cols]
        self.qm_cols = [c for c in self.num if c.startswith(("qm_ls_", "qm_ss_")) and as_num(dev[c]).notna().any()]
        self.vocab_cols = [c for c in self.cat if c not in self.no_vocab]      # raw state never gets a _v copy (bias mitigation)
        self.clip = {c: tuple(as_num(dev[c]).quantile([.01, .99])) for c in self.clip_cols}
        self.vocab = {c: set(dev[c].astype(str).value_counts().loc[lambda v: v >= self.rare_min].index) for c in self.vocab_cols}
        self.prior = float(np.mean(y))
        self.te_state = self._te(dev["state"], y, self.prior)
        Q = self._qm_matrix(dev)
        self.qm_med = Q.median(); Qf = Q.fillna(self.qm_med)
        self.qm_mu, self.qm_sd = Qf.mean(), Qf.std().replace(0, 1)
        self.pca = PCA(n_components=min(3, Q.shape[1]), random_state=self.seed).fit(((Qf - self.qm_mu) / self.qm_sd).values) if Q.shape[1] else None
        return self

    def _te(self, s, y, prior):
        g = pd.DataFrame({"k": s.astype(str).values, "y": np.asarray(y)}).groupby("k")["y"].agg(["sum", "count"])
        return ((g["sum"] + self.te_smooth * prior) / (g["count"] + self.te_smooth)).to_dict()

    def oof_state_encoding(self, dev, y, groups, n_splits=5):
        """state_te for the DEV rows themselves: out-of-fold, folds grouped by facility, so a home never sees its own label."""
        y = np.asarray(y); out = pd.Series(np.nan, index=dev.index)
        for tr, va in GroupKFold(n_splits=n_splits).split(dev, y, groups=groups):
            prior = float(y[tr].mean()); te = self._te(dev["state"].iloc[tr], y[tr], prior)
            out.iloc[va] = dev["state"].iloc[va].astype(str).map(te).fillna(prior).values
        return out

    def _qm_matrix(self, d):
        return pd.DataFrame({c: as_num(d[c]) for c in self.qm_cols}, index=d.index)

    def _panel_lags(self, d):
        """B6 lags: change vs three snapshots earlier and 4-snapshot volatility, from the same facility's earlier rows only."""
        f = d.sort_values(["ccn", "t"]); out = pd.DataFrame(index=f.index)
        for c in self.lag_cols:
            v = as_num(f[c]); g = v.groupby(f["ccn"])
            out[f"lag3_diff_{c}"] = v - g.shift(3)
            out[f"vol4_{c}"] = g.transform(lambda s: s.rolling(4, min_periods=3).std())
        return out.reindex(d.index)

    def transform(self, d):
        X = pd.DataFrame(index=d.index)
        peer = [d["t"], d["state"]]
        # B1 clip + log + per-bed normalisation
        for c in self.clip_cols:
            lo, hi = self.clip[c]; X[f"{c}_clip"] = as_num(d[c]).clip(lo, hi)
        for c in self.log_cols:
            X[f"log_{c}"] = np.log1p(as_num(d[c]).clip(lower=0))
        beds = as_num(d["certified_beds"]).where(lambda b: b > 0)
        compl12 = as_num(d["n_complaint_surveys_cited_12m"])
        X["cit_per_bed_24m"] = as_num(d["n_cit_24m"]) / beds
        X["complaints_per_100beds_12m"] = 100 * compl12 / beds
        X["fine_dollars_per_bed_24m"] = np.expm1(as_num(d["fine_dollars_24m"])) / beds      # fine_dollars_24m is stored as log1p
        # B2 bins
        X["bed_band"] = pd.cut(as_num(d["certified_beds"]), BED_BINS, labels=BED_LABELS, right=False).astype(str)
        X["std_recency_band"] = pd.cut(as_num(d["days_since_last_std"]), STD_BINS, labels=STD_LABELS).astype(str)
        # B3 rare-level grouping + target encoding of state (the notebook overwrites DEV rows with the out-of-fold version)
        for c in self.vocab_cols:
            s = d[c].astype(str); X[f"{c}_v"] = s.where(s.isin(self.vocab[c]), "other")
        X["state_te"] = d["state"].astype(str).map(self.te_state).fillna(self.prior)
        # B4 crosses
        rn_rank = as_num(d["hprd_rn"]).groupby(peer).rank(pct=True)
        X["x_complaints_x_low_rn"] = compl12 * (1 - rn_rank)          # complaints × RN staffing below state peers (same month)
        X["x_harm_last_std_x_overdue"] = as_num(d["gplus_last_std"]).fillna(0) * as_num(d["std_overdue"]).fillna(0)
        X["x_size_ownership"] = X["bed_band"] + "|" + d["ownership_group"].astype(str)
        X["x_chain_x_turnover"] = np.where(d["chain_id"].notna(), 1.0, 0.0) * as_num(d["turnover_total"])
        # B5 peer ranks within state-month (same-month data only: no leakage)
        for c in self.rank_cols:
            X[f"pct_state_{c}"] = as_num(d[c]).groupby(peer).rank(pct=True)
        # B6 momentum (from counts already in the row) + panel lags (earlier months of the same facility only)
        X["cit_momentum_12m"] = as_num(d["n_cit_12m"]) - (as_num(d["n_cit_24m"]) - as_num(d["n_cit_12m"]))
        X["complaint_momentum_12m"] = compl12 - (as_num(d["n_complaint_surveys_cited_24m"]) - compl12)
        X = X.join(self._panel_lags(d))
        # B7 missingness indicators (where missing has meaning)
        for c in self.isna_cols:
            X[f"isna_{c}"] = d[c].isna().astype(float)
        # B8 PCA quality index
        if self.pca is not None:
            Z = self.pca.transform(((self._qm_matrix(d).fillna(self.qm_med) - self.qm_mu) / self.qm_sd).values)
            for i in range(Z.shape[1]):
                X[f"qm_pca_{i + 1}"] = Z[:, i]
        return X
