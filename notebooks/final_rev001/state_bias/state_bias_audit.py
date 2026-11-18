"""State-wise bias / fairness audit of the main Precursor model (old_ens = LightGBM + LR mean ensemble, 133 features,
trained 2023-01..2023-11), scored on test month 2025-02 (primary) and 2024-11 (stability check).

Two stages (run from the repo root):
  uv run --no-sync python notebooks/final_rev001/state_bias/state_bias_audit.py --stage retrain
      Retrains LightGBM (same params / window as exp12) three ways: all 133 features, without `state`, and without
      all state-derived features; saves predictions + per-state SHAP-style contributions (pred_contrib).
      Refuses to run if free RAM < 2.5 GB.
  uv run --no-sync python notebooks/final_rev001/state_bias/state_bias_audit.py            (= --stage audit)
      All audit analyses from the saved predictions; uses the retrain outputs if they exist.

Outputs (this folder only): tables/*.csv, results.json, figures/*.png.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "pyproject.toml").exists())
GOLD = ROOT / "data" / "gold" / "facility_month_dataset"
EXP12 = ROOT / "notebooks" / "experiments_rev003" / "outputs" / "exp12_pbj"
PARAMS = ROOT.parent / "feature engineering run" / "spine2022" / "boost" / "lgbm_best_params.json"
TAB = HERE / "tables"
FIG = HERE / "figures"
MONTHS = {"2025-02": "preds_test_2025-02.csv", "2024-11": "preds_extra_2024-11.csv"}
PRIMARY, SECOND = "2025-02", "2024-11"
PAST = "2024-01"     # last month whose 12-month label window (2024-02..2025-01) is complete by 2025-02
TRAIN = ("2023-01", "2023-11")
SEED = 42
B = 500             # bootstrap replicates
NPERM = 1000        # permutation replicates
MIN_N, MIN_POS, MIN_NEG = 100, 20, 20   # eligibility for per-state gap summaries
SLICE_MIN_N, SLICE_MIN_POS, SLICE_MIN_NEG = 100, 15, 15
STATE_FEATS = ["state", "state_gplus_rate_12m", "state_median_hprd_total", "rel_hprd_total", "rel_sev_points_24m"]
SSA = {  # SSA state code (first 2 CCN digits) -> postal code; fallback only
    "01": "AL", "02": "AK", "03": "AZ", "04": "AR", "05": "CA", "06": "CO", "07": "CT", "08": "DE", "09": "DC",
    "10": "FL", "11": "GA", "12": "HI", "13": "ID", "14": "IL", "15": "IN", "16": "IA", "17": "KS", "18": "KY",
    "19": "LA", "20": "ME", "21": "MD", "22": "MA", "23": "MI", "24": "MN", "25": "MS", "26": "MO", "27": "MT",
    "28": "NE", "29": "NV", "30": "NH", "31": "NJ", "32": "NM", "33": "NY", "34": "NC", "35": "ND", "36": "OH",
    "37": "OK", "38": "OR", "39": "PA", "40": "PR", "41": "RI", "42": "SC", "43": "SD", "44": "TN", "45": "TX",
    "46": "UT", "47": "VT", "48": "VI", "49": "VA", "50": "WA", "51": "WV", "52": "WI", "53": "WY", "64": "AS",
    "65": "GU", "66": "MP", "67": "TX", "68": "FL", "74": "TX"}
INK, INK2, GRID, BLUE, ORANGE, AQUA, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#eb6834", "#1baf7a", "#fcfcfb"
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def free_ram_gb() -> float:
    import psutil
    return psutil.virtual_memory().available / 1e9


# ----------------------------------------------------------------------------------------------- data
def load_month(m: str) -> pd.DataFrame:
    p = pd.read_csv(EXP12 / MONTHS[m], dtype={"ccn": str})
    g = pd.read_parquet(GOLD / f"t={m}" / "part-0.parquet",
                        columns=["ccn", "state", "ownership_group", "certified_beds", "state_gplus_rate_12m"])
    d = p.merge(g, on="ccn", how="left")
    miss = d.state.isna()
    d.loc[miss, "state"] = d.loc[miss, "ccn"].str[:2].map(SSA)
    d["state_source_fallback"] = miss
    d["p"] = d["old_ens"].clip(1e-6, 1 - 1e-6)
    d["ownership_group"] = d["ownership_group"].fillna("unknown")
    return d


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def expit(x):
    return 1 / (1 + np.exp(-x))


# ----------------------------------------------------------------------------------------------- metric helpers
def auc_fast(y, s):
    y = np.asarray(y)
    npos, nneg = y.sum(), len(y) - y.sum()
    if npos == 0 or nneg == 0:
        return np.nan
    r = stats.rankdata(s)
    return (r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def safe_div(a, b):
    return a / b if b else np.nan


def conf_rates(y, f):
    tp = int(((f == 1) & (y == 1)).sum()); fp = int(((f == 1) & (y == 0)).sum())
    fn = int(((f == 0) & (y == 1)).sum()); tn = int(((f == 0) & (y == 0)).sum())
    return {"flag_rate": (tp + fp) / len(y), "tpr": safe_div(tp, tp + fn), "fpr": safe_div(fp, fp + tn),
            "fnr": safe_div(fn, tp + fn), "tnr": safe_div(tn, fp + tn), "precision": safe_div(tp, tp + fp),
            "for": safe_div(fn, fn + tn), "accuracy": (tp + tn) / len(y),
            "fn_fp_ratio": safe_div(fn, fp), "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def ece(y, p, nb=10):
    """Equal-mass ECE (adaptive bins)."""
    nb = int(max(2, min(nb, len(y) // 30)))
    order = np.argsort(p, kind="stable")
    tot = 0.0
    for b in np.array_split(order, nb):
        tot += len(b) * abs(y[b].mean() - p[b].mean())
    return tot / len(y)


def hosmer_lemeshow(y, p, g=10):
    """External-validation HL: equal-mass groups, df = g (model not fitted on these data)."""
    g = int(max(2, min(g, len(y) // 30)))
    order = np.argsort(p, kind="stable")
    hl = 0.0
    for b in np.array_split(order, g):
        o, e, n = y[b].sum(), p[b].sum(), len(b)
        v = e * (1 - e / n)
        hl += (o - e) ** 2 / v if v > 0 else 0
    return hl, g, float(stats.chi2.sf(hl, g))


def calib_slope_intercept(y, p):
    """Logistic recalibration: logit P(y) = a + b*logit(p) (slope) and a0 with offset (calibration-in-the-large)."""
    import statsmodels.api as sm
    x = logit(p)
    out = {"cal_slope": np.nan, "cal_slope_se": np.nan, "cal_intercept_citl": np.nan, "cal_intercept_citl_se": np.nan}
    if y.sum() < 3 or (1 - y).sum() < 3:
        return out
    try:
        r = sm.GLM(y, sm.add_constant(x), family=sm.families.Binomial()).fit()
        out["cal_slope"], out["cal_slope_se"] = float(r.params[1]), float(r.bse[1])
        r0 = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=x).fit()
        out["cal_intercept_citl"], out["cal_intercept_citl_se"] = float(r0.params[0]), float(r0.bse[0])
    except Exception:  # noqa: BLE001
        pass
    return out


def group_metrics(y, p, thr, c10, c20):
    m = {"n": len(y), "positives": int(y.sum()), "base_rate": y.mean(), "mean_pred": p.mean()}
    m["o_minus_e"] = m["base_rate"] - m["mean_pred"]
    m["o_over_e"] = safe_div(m["base_rate"], m["mean_pred"])
    m["oe_z"] = (y.sum() - p.sum()) / np.sqrt((p * (1 - p)).sum())
    m["oe_p"] = float(2 * stats.norm.sf(abs(m["oe_z"])))
    m["roc_auc"] = auc_fast(y, p)
    m["pr_auc"] = float(average_precision_score(y, p)) if 0 < y.sum() < len(y) else np.nan
    m["pr_auc_lift"] = safe_div(m["pr_auc"], m["base_rate"])
    m["brier"] = float(np.mean((p - y) ** 2))
    m["ece"] = ece(y, p)
    for tag, cut in (("thr", thr), ("top10", c10), ("top20", c20)):
        for k, v in conf_rates(y, (p >= cut).astype(int)).items():
            m[f"{k}_{tag}"] = v
    m.update(calib_slope_intercept(y, p))
    hl, g, hp = hosmer_lemeshow(y, p)
    m.update({"hl_stat": hl, "hl_groups": g, "hl_p": hp})
    return m


# ----------------------------------------------------------------------------------------------- statistics helpers
def bh(pvals):
    p = np.asarray(pvals, float)
    q = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    if ok.sum() == 0:
        return q
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order] * len(pv) / (np.arange(len(pv)) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(pv)
    out[order] = np.minimum(ranked, 1)
    q[ok] = out
    return q


def two_prop(k1, n1, k2, n2):
    if min(n1, n2) == 0:
        return np.nan
    p1, p2, pp = k1 / n1, k2 / n2, (k1 + k2) / (n1 + n2)
    se = np.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
    return float(2 * stats.norm.sf(abs(p1 - p2) / se)) if se > 0 else np.nan


def dersimonian_laird(est, var):
    est, var = np.asarray(est, float), np.asarray(var, float)
    ok = np.isfinite(est) & np.isfinite(var) & (var > 0)
    w = 1 / var[ok]
    mu_fe = (w * est[ok]).sum() / w.sum()
    q = (w * (est[ok] - mu_fe) ** 2).sum()
    k = ok.sum()
    tau2 = max(0.0, (q - (k - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
    wr = 1 / (var[ok] + tau2)
    mu = (wr * est[ok]).sum() / wr.sum()
    shrunk = np.full_like(est, np.nan)
    shrunk[ok] = mu + tau2 / (tau2 + var[ok]) * (est[ok] - mu)
    i2 = max(0.0, (q - (k - 1)) / q) if q > 0 else 0.0
    return {"tau2": tau2, "mu": mu, "Q": q, "k": int(k), "Q_p": float(stats.chi2.sf(q, k - 1)), "I2": i2,
            "shrunk": shrunk}


# ----------------------------------------------------------------------------------------------- analyses
def per_state_table(d, thr, c10, c20):
    rows = []
    for s, g in d.groupby("state"):
        m = group_metrics(g.y.values, g.p.values, thr, c10, c20)
        m["state"] = s
        m["mean_state_gplus_rate_12m"] = g.state_gplus_rate_12m.mean()
        rows.append(m)
    t = pd.DataFrame(rows)
    t["eligible"] = (t.n >= MIN_N) & (t.positives >= MIN_POS) & (t.n - t.positives >= MIN_NEG)
    return t.set_index("state")


def rate_tests(d, t, tag="thr", cut=None):
    """State vs rest-of-nation two-proportion tests for base rate, flag rate, TPR, FPR, precision, FOR; O-E z."""
    f = (d.p >= cut).astype(int).values
    y = d.y.values
    st = d.state.values
    out = {}
    for s in t.index:
        a = st == s
        b = ~a
        r = {}
        r["base_rate_p"] = two_prop(y[a].sum(), a.sum(), y[b].sum(), b.sum())
        r["flag_rate_p"] = two_prop(f[a].sum(), a.sum(), f[b].sum(), b.sum())
        pa, pb = a & (y == 1), b & (y == 1)
        r["tpr_p"] = two_prop(f[pa].sum(), pa.sum(), f[pb].sum(), pb.sum())
        na, nb_ = a & (y == 0), b & (y == 0)
        r["fpr_p"] = two_prop(f[na].sum(), na.sum(), f[nb_].sum(), nb_.sum())
        fa, fb = a & (f == 1), b & (f == 1)
        r["precision_p"] = two_prop(y[fa].sum(), fa.sum(), y[fb].sum(), fb.sum())
        ua, ub = a & (f == 0), b & (f == 0)
        r["for_p"] = two_prop(y[ua].sum(), ua.sum(), y[ub].sum(), ub.sum())
        out[s] = r
    r = pd.DataFrame(out).T
    r["oe_p"] = t["oe_p"]
    el = t["eligible"].values
    for c in list(r.columns):
        q = np.full(len(r), np.nan)
        q[el] = bh(r[c].values[el])  # BH across eligible states, per family
        r[c.replace("_p", "_q")] = q
    return r


def fairness_summary(t, nat, tags=("thr", "top10", "top20")):
    e = t[t.eligible]
    rows = []
    for tag in tags:
        spec = {"demographic_parity (flag rate)": f"flag_rate_{tag}", "equal_opportunity (TPR)": f"tpr_{tag}",
                "FPR": f"fpr_{tag}", "predictive_parity (precision)": f"precision_{tag}", "FNR": f"fnr_{tag}",
                "FOR (false omission rate)": f"for_{tag}", "accuracy": f"accuracy_{tag}",
                "treatment_equality (FN/FP)": f"fn_fp_ratio_{tag}"}
        for name, col in spec.items():
            v = e[col].replace([np.inf], np.nan).dropna()
            rows.append({"cutoff": tag, "criterion": name, "national": nat[col], "min": v.min(), "argmin": v.idxmin(),
                         "max": v.max(), "argmax": v.idxmax(), "max_minus_min": v.max() - v.min(),
                         "max_abs_diff_vs_national": (v - nat[col]).abs().max(),
                         "ratio_min_over_max": v.min() / v.max() if v.max() > 0 else np.nan})
        fr = e[f"flag_rate_{tag}"]
        rows.append({"cutoff": tag, "criterion": "disparate_impact (4/5ths): states with flag rate < 0.8 x max state",
                     "national": nat[f"flag_rate_{tag}"], "min": (fr / fr.max()).min(), "argmin": fr.idxmin(),
                     "max": 1.0, "argmax": fr.idxmax(), "max_minus_min": np.nan,
                     "max_abs_diff_vs_national": int((fr < 0.8 * fr.max()).sum()),
                     "ratio_min_over_max": fr.min() / fr.max()})
        rows.append({"cutoff": tag, "criterion": "disparate_impact vs national: states with flag rate < 0.8 x national",
                     "national": nat[f"flag_rate_{tag}"], "min": np.nan, "argmin": "", "max": np.nan, "argmax": "",
                     "max_minus_min": np.nan,
                     "max_abs_diff_vs_national": int((fr < 0.8 * nat[f"flag_rate_{tag}"]).sum()),
                     "ratio_min_over_max": np.nan})
        eo = np.maximum((e[f"tpr_{tag}"] - nat[f"tpr_{tag}"]).abs(), (e[f"fpr_{tag}"] - nat[f"fpr_{tag}"]).abs())
        rows.append({"cutoff": tag, "criterion": "equalized_odds (max of |dTPR|,|dFPR| vs national)",
                     "national": 0.0, "min": eo.min(), "argmin": eo.idxmin(), "max": eo.max(), "argmax": eo.idxmax(),
                     "max_minus_min": max(e[f"tpr_{tag}"].max() - e[f"tpr_{tag}"].min(),
                                          e[f"fpr_{tag}"].max() - e[f"fpr_{tag}"].min()),
                     "max_abs_diff_vs_national": eo.max(), "ratio_min_over_max": np.nan})
        # base-rate-adjusted demographic parity: flag rate / base rate
        adj = e[f"flag_rate_{tag}"] / e["base_rate"]
        rows.append({"cutoff": tag, "criterion": "flag rate / base rate (base-rate-adjusted parity)",
                     "national": nat[f"flag_rate_{tag}"] / nat["base_rate"], "min": adj.min(), "argmin": adj.idxmin(),
                     "max": adj.max(), "argmax": adj.idxmax(), "max_minus_min": adj.max() - adj.min(),
                     "max_abs_diff_vs_national": (adj - nat[f"flag_rate_{tag}"] / nat["base_rate"]).abs().max(),
                     "ratio_min_over_max": adj.min() / adj.max()})
    for name, col in {"calibration-in-the-large (O-E)": "o_minus_e", "ECE": "ece", "ROC-AUC": "roc_auc",
                      "PR-AUC": "pr_auc", "PR-AUC lift": "pr_auc_lift", "Brier": "brier",
                      "calibration slope": "cal_slope", "base rate (label)": "base_rate",
                      "mean predicted": "mean_pred"}.items():
        v = e[col].dropna()
        rows.append({"cutoff": "-", "criterion": name, "national": nat[col], "min": v.min(), "argmin": v.idxmin(),
                     "max": v.max(), "argmax": v.idxmax(), "max_minus_min": v.max() - v.min(),
                     "max_abs_diff_vs_national": (v - nat[col]).abs().max(),
                     "ratio_min_over_max": v.min() / v.max() if v.max() > 0 else np.nan})
    return pd.DataFrame(rows)


def bootstrap_states(d, thr, rng):
    """Per-state CIs. Within one month each facility has one row, so resampling facilities = resampling rows."""
    keys = ["o_minus_e", "tpr", "fpr", "precision", "flag_rate", "roc_auc", "pr_auc", "ece"]
    out = {}
    for s, g in d.groupby("state"):
        y, p = g.y.values, g.p.values
        n = len(y)
        res = {k: [] for k in keys}
        for _ in range(B):
            i = rng.integers(0, n, n)
            yy, pp = y[i], p[i]
            f = pp >= thr
            res["o_minus_e"].append(yy.mean() - pp.mean())
            res["flag_rate"].append(f.mean())
            res["tpr"].append(f[yy == 1].mean() if yy.sum() else np.nan)
            res["fpr"].append(f[yy == 0].mean() if (yy == 0).sum() else np.nan)
            res["precision"].append(yy[f].mean() if f.sum() else np.nan)
            res["roc_auc"].append(auc_fast(yy, pp))
            res["pr_auc"].append(average_precision_score(yy, pp) if 0 < yy.sum() < n else np.nan)
            res["ece"].append(ece(yy, pp))
        row = {}
        for k, v in res.items():
            v = np.asarray(v, float)
            row[f"{k}_lo"], row[f"{k}_hi"] = np.nanpercentile(v, 2.5), np.nanpercentile(v, 97.5)
        out[s] = row
    return pd.DataFrame(out).T


def pooled_cluster_bootstrap(d2, rng):
    """O-E per state pooling both months, resampling facilities (clusters with 2 rows each)."""
    out = {}
    for s, g in d2.groupby("state"):
        agg = g.groupby("ccn").agg(o=("y", "sum"), e=("p", "sum"), n=("y", "size"))
        o, e, n = agg.o.values, agg.e.values, agg.n.values
        k = len(agg)
        vals = []
        for _ in range(B):
            i = rng.integers(0, k, k)
            vals.append((o[i].sum() - e[i].sum()) / n[i].sum())
        out[s] = {"n_rows": int(n.sum()), "n_facilities": k, "o_minus_e_pooled": (o.sum() - e.sum()) / n.sum(),
                  "lo": np.percentile(vals, 2.5), "hi": np.percentile(vals, 97.5)}
    return pd.DataFrame(out).T


def permutation_tests(d, thr, rng):
    """Global tests: is the between-state dispersion bigger than random state labels would give?"""
    codes, uniq = pd.factorize(d.state)
    k = len(uniq)
    y = d.y.values.astype(float)
    p = d.p.values
    f = (p >= thr).astype(float)
    r = y - p

    def between_ss(vals, cds):
        n = np.bincount(cds, minlength=k).astype(float)
        s = np.bincount(cds, weights=vals, minlength=k)
        m = vals.mean()
        ok = n > 0
        return float((n[ok] * (s[ok] / n[ok] - m) ** 2).sum())

    tests = {"residual y-p (calibration)": (r, np.ones(len(y), bool)),
             "flag rate (demographic parity)": (f, np.ones(len(y), bool)),
             "TPR (equal opportunity, among y=1)": (f, y == 1),
             "FPR (among y=0)": (f, y == 0),
             "precision (among flagged)": (y, f == 1),
             "base rate (label, reference)": (y, np.ones(len(y), bool))}
    out = []
    for name, (v, mask) in tests.items():
        vv, cc = v[mask], codes[mask]
        obs = between_ss(vv, cc)
        null = np.array([between_ss(vv, rng.permutation(cc)) for _ in range(NPERM)])
        out.append({"test": name, "observed_between_state_SS": obs, "null_mean": null.mean(),
                    "ratio_obs_over_null": obs / null.mean(), "perm_p": (1 + (null >= obs).sum()) / (NPERM + 1)})
    return pd.DataFrame(out)


def multilevel(d):
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    res = {}
    dd = d[["y", "p", "state"]].copy()
    dd["resid"] = dd.y - dd.p
    dd["lp"] = logit(dd.p.values)
    m = smf.mixedlm("resid ~ 1", dd, groups=dd["state"]).fit(reml=True)
    tau2 = float(m.cov_re.iloc[0, 0])
    sig2 = float(m.scale)
    res["lmm_residual"] = {"intercept": float(m.params["Intercept"]), "tau2_state": tau2, "sigma2_within": sig2,
                           "tau_state_pp": 100 * np.sqrt(tau2), "ICC": tau2 / (tau2 + sig2)}
    blup = pd.Series({k: float(v.iloc[0]) for k, v in m.random_effects.items()}, name="lmm_blup")
    # logit-scale: LRT for state fixed intercepts on top of offset(logit p) [+ slope]
    X0 = np.ones((len(dd), 1))
    base = sm.GLM(dd.y.values, X0, family=sm.families.Binomial(), offset=dd.lp.values).fit()
    S = pd.get_dummies(dd.state, drop_first=True).astype(float).values
    full = sm.GLM(dd.y.values, np.column_stack([X0, S]), family=sm.families.Binomial(), offset=dd.lp.values).fit()
    lr = 2 * (full.llf - base.llf)
    res["lrt_state_intercepts_logit"] = {"LR": float(lr), "df": int(S.shape[1]),
                                        "p": float(stats.chi2.sf(lr, S.shape[1]))}
    # random-intercept logistic (variational Bayes) with logit(p) as covariate
    try:
        from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
        g = BinomialBayesMixedGLM.from_formula("y ~ lp", {"state": "0 + C(state)"}, dd).fit_vb()
        sd = float(np.exp(g.vcp_mean[0]))
        res["glmm_logit_vb"] = {"fixed_intercept": float(g.fe_mean[0]), "fixed_slope_logit_p": float(g.fe_mean[1]),
                                "sd_state_intercept_logit": sd,
                                "ICC_latent": sd ** 2 / (sd ** 2 + np.pi ** 2 / 3)}
        names = [n.split("[")[1].rstrip("]") for n in g.model.vc_names]
        blup_logit = pd.Series(g.vc_mean, index=[n.replace("C(state)", "").replace("[", "").replace("]", "")
                                                 for n in names], name="glmm_blup_logit")
    except Exception as ex:  # noqa: BLE001
        res["glmm_logit_vb"] = {"error": str(ex)}
        blup_logit = pd.Series(dtype=float, name="glmm_blup_logit")
    return res, blup, blup_logit


def auc_decomposition(d, prev_state_rate):
    y, p, st = d.y.values, d.p.values, d.state.values
    total_auc = auc_fast(y, p)
    npos, nneg = y.sum(), len(y) - y.sum()
    tot_pairs = npos * nneg
    within_pairs, within_conc = 0.0, 0.0
    for s in np.unique(st):
        a = st == s
        ps, ns = y[a].sum(), a.sum() - y[a].sum()
        if ps and ns:
            au = auc_fast(y[a], p[a])
            within_pairs += ps * ns
            within_conc += au * ps * ns
    between_pairs = tot_pairs - within_pairs
    auc_within = within_conc / within_pairs
    auc_between = (total_auc * tot_pairs - within_conc) / between_pairs
    sm = pd.Series(p).groupby(st).transform("mean").values
    lp = logit(p)
    lsm = pd.Series(lp).groupby(st).transform("mean").values
    return {"overall_auc": total_auc, "share_pairs_within_state": within_pairs / tot_pairs,
            "auc_within_state_pairs": auc_within, "auc_between_state_pairs": auc_between,
            "auc_state_mean_score_only": auc_fast(y, sm),
            "auc_score_centred_within_state (logit)": auc_fast(y, lp - lsm),
            "auc_prev_month_state_base_rate_only": auc_fast(y, pd.Series(st).map(prev_state_rate).fillna(
                np.nanmean(list(prev_state_rate.values()))).values),
            "auc_state_gplus_rate_12m_feature_only": auc_fast(y, d.state_gplus_rate_12m.fillna(
                d.state_gplus_rate_12m.mean()).values),
            "pr_auc_overall": float(average_precision_score(y, p)),
            "pr_auc_score_centred_within_state": float(average_precision_score(y, lp - lsm))}


def slice_discovery(d, nat_rates, thr):
    d = d.copy()
    d["flag"] = (d.p >= thr).astype(int)
    attrs = ["state", "beds_tercile", "ownership_group"]
    rows = []
    for r in (1, 2, 3):
        for combo in itertools.combinations(attrs, r):
            for key, g in d.groupby(list(combo), observed=True):
                key = key if isinstance(key, tuple) else (key,)
                n, pos = len(g), int(g.y.sum())
                if n < SLICE_MIN_N or pos < SLICE_MIN_POS or n - pos < SLICE_MIN_NEG:
                    continue
                y, p, f = g.y.values, g.p.values, g.flag.values
                oe = y.mean() - p.mean()
                z = (y.sum() - p.sum()) / np.sqrt((p * (1 - p)).sum())
                fnr = 1 - f[y == 1].mean()
                fpr = f[y == 0].mean()
                pos_all, neg_all = nat_rates["pos"], nat_rates["neg"]
                rows.append({"slice": " & ".join(f"{a}={v}" for a, v in zip(combo, key)), "order": r, "n": n,
                             "positives": pos, "base_rate": y.mean(), "mean_pred": p.mean(), "o_minus_e": oe,
                             "oe_p": 2 * stats.norm.sf(abs(z)), "fnr": fnr,
                             "fnr_p": two_prop(int((f[y == 1] == 0).sum()), pos, nat_rates["fn"], pos_all),
                             "fpr": fpr, "fpr_p": two_prop(int(f[y == 0].sum()), n - pos, nat_rates["fp"], neg_all),
                             "roc_auc": auc_fast(y, p)})
    s = pd.DataFrame(rows)
    allp = np.concatenate([s.oe_p.values, s.fnr_p.values, s.fpr_p.values])
    q = bh(allp)
    k = len(s)
    s["oe_q"], s["fnr_q"], s["fpr_q"] = q[:k], q[k:2 * k], q[2 * k:]
    return s


def slice_replication(s, d_other, thr):
    """Recompute the same slices on the other month (replication check)."""
    d_other = d_other.copy()
    d_other["flag"] = (d_other.p >= thr).astype(int)
    out = []
    for sl in s.slice:
        m = np.ones(len(d_other), bool)
        for part in sl.split(" & "):
            a, v = part.split("=", 1)
            m &= (d_other[a].astype(str) == v).values
        g = d_other[m]
        if len(g) == 0:
            out.append((np.nan, np.nan, np.nan))
            continue
        y, p, f = g.y.values, g.p.values, g.flag.values
        out.append((y.mean() - p.mean(), 1 - f[y == 1].mean() if y.sum() else np.nan,
                    f[y == 0].mean() if (y == 0).sum() else np.nan))
    o = np.array(out)
    s = s.copy()
    s["o_minus_e_other_month"], s["fnr_other_month"], s["fpr_other_month"] = o[:, 0], o[:, 1], o[:, 2]
    return s


def fit_state_intercepts(d):
    """Per-state logit intercept shift on top of offset logit(p), with its SE (for EB shrinkage)."""
    import statsmodels.api as sm
    rows = {}
    for s, g in d.groupby("state"):
        y, lp = g.y.values, logit(g.p.values)
        if y.sum() == 0 or y.sum() == len(y):
            # one-class state: use a +/-0.5-count correction
            o, e = y.sum() + 0.5, g.p.sum() + 0.5
            rows[s] = (np.log(o / e), 1 / max(o, 0.5) + 1 / max(len(y) - o, 0.5))
            continue
        r = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=lp).fit()
        rows[s] = (float(r.params[0]), float(r.bse[0]) ** 2)
    t = pd.DataFrame(rows, index=["delta", "var"]).T
    dl = dersimonian_laird(t.delta.values, t["var"].values)
    t["delta_shrunk"] = dl["shrunk"]
    return t, dl


def global_platt(d):
    import statsmodels.api as sm
    r = sm.GLM(d.y.values, sm.add_constant(logit(d.p.values)), family=sm.families.Binomial()).fit()
    return float(r.params[0]), float(r.params[1])


def acc_threshold(y, p):
    grid = np.unique(np.quantile(p, np.linspace(0.5, 0.995, 200)))
    return float(grid[np.argmax([np.mean((p >= g) == y) for g in grid])])


def summarise_model(d, pcol, thr, label):
    """Overall + between-state calibration / error-rate dispersion for one score column."""
    dd = d.assign(p=d[pcol].clip(1e-6, 1 - 1e-6))
    y, p = dd.y.values, dd.p.values
    c10, c20 = np.quantile(p, 0.9), np.quantile(p, 0.8)
    t = per_state_table(dd, thr, c10, c20)
    e = t[t.eligible]
    t_te = rate_tests(dd, t, cut=thr)
    w = e.n / e.n.sum()
    return {"model": label, "roc_auc": auc_fast(y, p), "pr_auc": float(average_precision_score(y, p)),
            "brier": float(np.mean((p - y) ** 2)), "ece_overall": ece(y, p, 20),
            "threshold": thr, "accuracy": float(np.mean((p >= thr) == y)), "flag_rate": float(np.mean(p >= thr)),
            "state_oe_sd_pp (eligible, unweighted)": 100 * e.o_minus_e.std(),
            "state_mean_abs_oe_pp (eligible, n-weighted)": 100 * (w * e.o_minus_e.abs()).sum(),
            "state_ece_mean (eligible, n-weighted)": float((w * e.ece).sum()),
            "n_states_oe_sig_BH": int((t_te.oe_q < 0.05).sum()),
            "tpr_gap_max_min (thr)": e.tpr_thr.max() - e.tpr_thr.min(),
            "fpr_gap_max_min (thr)": e.fpr_thr.max() - e.fpr_thr.min(),
            "precision_gap_max_min (thr)": e.precision_thr.max() - e.precision_thr.min(),
            "tpr_gap_max_min (top10)": e.tpr_top10.max() - e.tpr_top10.min(),
            "within_state_auc_mean (eligible, n-weighted)": float((w * e.roc_auc).sum()),
            "corr(state mean pred, state base rate)": float(np.corrcoef(e.mean_pred, e.base_rate)[0, 1])}, t


def mitigation(d_fit, d_eval, thr, fit_label=SECOND):
    out, tabs = [], {}
    r0, t0 = summarise_model(d_eval, "p", thr, "M0 original (no recalibration)")
    out.append(r0); tabs["M0"] = t0
    # M1 unshrunk per-state intercepts, M2 EB-shrunk per-state intercepts, M3 global Platt + shrunk state intercepts
    si, dl = fit_state_intercepts(d_fit)
    lp = logit(d_eval.p.values)
    st = d_eval.state
    raw = st.map(si.delta).fillna(0).values
    shr = st.map(si.delta_shrunk).fillna(0).values
    a, b = global_platt(d_fit)
    lp_fit = a + b * logit(d_fit.p.values)
    si2, dl2 = fit_state_intercepts(d_fit.assign(p=expit(lp_fit)))
    shr2 = st.map(si2.delta_shrunk).fillna(0).values
    variants = {"M1 per-state intercept (unshrunk)": expit(lp + raw),
                "M2 per-state intercept (EB-shrunk)": expit(lp + shr),
                "M3 global Platt + EB-shrunk state intercept": expit(a + b * lp + shr2),
                "M4 global Platt only (no state term)": expit(a + b * lp)}
    fit_lp = logit(d_fit.p.values)
    fit_st = d_fit.state
    fit_variants = {"M1 per-state intercept (unshrunk)": expit(fit_lp + fit_st.map(si.delta).fillna(0).values),
                    "M2 per-state intercept (EB-shrunk)": expit(fit_lp + fit_st.map(si.delta_shrunk).fillna(0).values),
                    "M3 global Platt + EB-shrunk state intercept": expit(lp_fit + fit_st.map(si2.delta_shrunk)
                                                                         .fillna(0).values),
                    "M4 global Platt only (no state term)": expit(lp_fit)}
    for name, pv in variants.items():
        de = d_eval.assign(pm=pv)
        r, t = summarise_model(de, "pm", thr, name + " @ original threshold")
        out.append(r); tabs[name[:2]] = t
        thr2 = acc_threshold(d_fit.y.values, fit_variants[name])
        r2, _ = summarise_model(de, "pm", thr2, name + f" @ threshold re-chosen on {fit_label}")
        out.append(r2)
    # M5 per-state thresholds equalising TPR (fit on 2024-11), scores unchanged
    nat_tpr = ((d_fit.p >= thr) & (d_fit.y == 1)).sum() / d_fit.y.sum()
    st_thr = {}
    for s, g in d_fit.groupby("state"):
        pos = np.sort(g.p[g.y == 1].values)
        if len(g) >= MIN_N and len(pos) >= MIN_POS:
            st_thr[s] = float(np.quantile(pos, 1 - nat_tpr))
    tv = d_eval.state.map(st_thr).fillna(thr).values
    y, p = d_eval.y.values, d_eval.p.values
    f = (p >= tv).astype(int)
    rows = []
    for s, g in d_eval.assign(f=f).groupby("state"):
        rows.append({"state": s, "n": len(g), "positives": g.y.sum(), **conf_rates(g.y.values, g.f.values)})
    tt = pd.DataFrame(rows).set_index("state")
    el = tabs["M0"].eligible
    e = tt[el.reindex(tt.index).fillna(False)]
    out.append({"model": f"M5 per-state thresholds equalising TPR (fit {fit_label})", "roc_auc": np.nan, "pr_auc": np.nan,
                "accuracy": float(np.mean(f == y)), "flag_rate": float(f.mean()),
                "tpr_gap_max_min (thr)": e.tpr.max() - e.tpr.min(), "fpr_gap_max_min (thr)": e.fpr.max() - e.fpr.min(),
                "precision_gap_max_min (thr)": e.precision.max() - e.precision.min(),
                "n_state_thresholds": len(st_thr)})
    return pd.DataFrame(out), tabs, si, dl


# ----------------------------------------------------------------------------------------------- retrain stage
def stage_retrain():
    free = free_ram_gb()
    if free < 2.5:
        sys.exit(f"free RAM {free:.2f} GB < 2.5 GB: retrain skipped (close other apps and rerun)")
    import lightgbm as lgb
    sys.path.insert(0, str(ROOT / "notebooks" / "experiments_rev002"))
    import exp_common as ec  # noqa: E402
    num, cat = ec.feature_lists()
    params = json.loads(PARAMS.read_text(encoding="utf-8"))["params"]
    rmonths = [*MONTHS, PAST]
    want = {f"t={m}" for m in rmonths}
    parts = []
    for f in ec.month_files():
        t = f.parent.name[2:]
        if not (TRAIN[0] <= t <= TRAIN[1] or f.parent.name in want):
            continue
        d = pd.read_parquet(f, columns=["ccn", "t", "y_g12", "status_g12", *num, *cat])
        d = d[d.status_g12.isin(["positive", "negative"])]
        for c in num:
            d[c] = pd.to_numeric(d[c], errors="coerce").astype("float32")
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    del parts
    df["t"] = df["t"].astype(str)
    df["y"] = df["y_g12"].astype("int8")
    df = df.drop(columns=["y_g12", "status_g12"])
    for c in cat:
        df[c] = df[c].astype("string").fillna("__missing__").astype("category")
    feats = num + cat
    tr = df[(df.t >= TRAIN[0]) & (df.t <= TRAIN[1])]
    tes = {m: df[df.t == m].reset_index(drop=True) for m in rmonths}
    log(f"retrain: {len(feats)} features, train {len(tr):,} rows, free RAM {free_ram_gb():.2f} GB")
    arms = {"lgb_full": feats, "lgb_no_state": [c for c in feats if c != "state"],
            "lgb_no_state_info": [c for c in feats if c not in STATE_FEATS]}
    preds = {m: tes[m][["ccn", "t", "y"]].copy() for m in rmonths}
    contrib_rows = []
    gain_rows = []
    for arm, cols in arms.items():
        mdl = lgb.LGBMClassifier(**params).fit(tr[cols], tr.y.values)
        log(f"  fitted {arm} ({len(cols)} features); free RAM {free_ram_gb():.2f} GB")
        imp = pd.Series(mdl.booster_.feature_importance("gain"), index=cols)
        imp = imp / imp.sum()
        for c in STATE_FEATS:
            if c in cols:
                gain_rows.append({"arm": arm, "feature": c, "gain_share": float(imp[c]),
                                  "gain_rank": int((imp > imp[c]).sum() + 1)})
        for m, te in tes.items():
            preds[m][arm] = mdl.predict_proba(te[cols])[:, 1]
            if arm == "lgb_full" and m in MONTHS:
                raw = mdl.predict_proba(te[cols], pred_contrib=True)  # TreeSHAP values on the logit scale
                pc = np.asarray(raw)[:, :-1]
                ix = [cols.index(c) for c in STATE_FEATS]
                ctx = [cols.index(c) for c in STATE_FEATS if c != "state"]
                g = pd.DataFrame({"st": te["state"].astype(str).values,
                                  "shap_state": pc[:, cols.index("state")],
                                  "shap_state_context": pc[:, ctx].sum(axis=1),
                                  "abs_state_feats": np.abs(pc[:, ix]).sum(axis=1),
                                  "abs_shap_total": np.abs(pc).sum(axis=1),
                                  "total_logit": np.asarray(raw).sum(axis=1)})
                g["shap_all_state_feats"] = g.shap_state + g.shap_state_context
                agg = g.groupby("st").agg(n=("total_logit", "size"), shap_state=("shap_state", "mean"),
                                          shap_state_context=("shap_state_context", "mean"),
                                          shap_all_state_feats=("shap_all_state_feats", "mean"),
                                          abs_state_feats=("abs_state_feats", "mean"),
                                          abs_shap_total=("abs_shap_total", "mean"),
                                          mean_logit=("total_logit", "mean"))
                agg["abs_share_state_feats"] = agg.abs_state_feats / agg.abs_shap_total
                agg["month"] = m
                agg.index.name = "state"
                contrib_rows.append(agg.reset_index())
                del raw, pc
        del mdl
    # logistic-regression half of the ensemble (all features) -> retrained old_ens, also scored on PAST
    from exp8_lr import fit_lr  # noqa: E402
    lr = fit_lr(tr, feats, num, cat)
    log(f"  fitted LR; free RAM {free_ram_gb():.2f} GB")
    for m, te in tes.items():
        preds[m]["ens_full"] = (preds[m]["lgb_full"] + lr.predict_proba(te[feats])[:, 1]) / 2
    out = pd.concat([preds[m] for m in rmonths], ignore_index=True)
    out.to_csv(TAB / "retrain_preds.csv.gz", index=False, compression="gzip", float_format="%.5f")
    pd.concat(contrib_rows).to_csv(TAB / "shap_state_contrib.csv", index=False, float_format="%.5f")
    pd.DataFrame(gain_rows).to_csv(TAB / "retrain_state_feature_gain.csv", index=False, float_format="%.5f")
    log("retrain stage done")


# ----------------------------------------------------------------------------------------------- plots
def _style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def plots(st, boot, eb, d, d_prev, st_prev, rel_states, thr, mit_tabs, retrain=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(exist_ok=True)
    st = st[st.n >= 5]  # territories with < 5 rows (e.g. GU, n=1) would only stretch the axes
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "figure.facecolor": SURF, "text.color": INK,
                         "axes.labelcolor": INK})

    # 1 funnel plot
    fig, ax = plt.subplots(figsize=(7, 4.6))
    _style(ax)
    v = (d.p * (1 - d.p)).mean()
    ns = np.linspace(5, st.n.max() * 1.05, 300)
    for z, ls in ((1.96, "--"), (3.09, ":")):
        ax.plot(ns, 100 * z * np.sqrt(v / ns), color=INK2, lw=1, ls=ls)
        ax.plot(ns, -100 * z * np.sqrt(v / ns), color=INK2, lw=1, ls=ls)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.scatter(st.n, 100 * st.o_minus_e, s=22, color=BLUE, edgecolor=SURF, lw=0.8, zorder=3)
    lab = st[(st.o_minus_e.abs() * np.sqrt(st.n) / np.sqrt(v) > 3.09) | (st.n > 800)].index
    for s in lab:
        ax.annotate(s, (st.n[s], 100 * st.o_minus_e[s]), fontsize=7.5, color=INK, xytext=(3, 2),
                    textcoords="offset points")
    ax.set_xlabel("facility rows in state (2025-02)")
    ax.set_ylabel("observed - expected G+ rate (points)")
    ax.set_title("Funnel plot: per-state calibration gap vs state size (states >= 5 rows)\n(dashed 95 %, dotted 99.8 % limits under "
                 "perfect calibration)", loc="left")
    fig.tight_layout(); fig.savefig(FIG / "funnel_oe_2025-02.png", dpi=150); plt.close(fig)

    # 2 forest plot raw O-E with bootstrap CI + EB-shrunk point
    s2 = st.join(boot).join(eb).sort_values("o_minus_e")
    fig, ax = plt.subplots(figsize=(6.5, 9.5))
    _style(ax)
    yy = np.arange(len(s2))
    ax.hlines(yy, 100 * s2.o_minus_e_lo, 100 * s2.o_minus_e_hi, color=BLUE, lw=1.2, alpha=0.6)
    ax.scatter(100 * s2.o_minus_e, yy, s=16, color=BLUE, zorder=3, label="raw O-E (95 % bootstrap CI)")
    ax.scatter(100 * s2.oe_eb, yy, s=22, marker="D", color=ORANGE, edgecolor=SURF, lw=0.6, zorder=4,
               label="empirical-Bayes shrunk")
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(yy); ax.set_yticklabels([f"{s} ({int(n)})" for s, n in zip(s2.index, s2.n)], fontsize=6.5)
    ax.set_xlabel("observed - expected G+ rate (points); >0 = model under-predicts")
    ax.set_title("Per-state calibration-in-the-large, 2025-02", loc="left")
    ax.legend(loc="lower right", fontsize=7.5, frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "forest_oe_by_state_2025-02.png", dpi=150); plt.close(fig)

    # 3 reliability plots for most over/under-predicted eligible states
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.4), sharex=True, sharey=True)
    for ax, s in zip(axes.ravel(), rel_states):
        _style(ax)
        g = d[d.state == s]
        nb = int(max(3, min(8, len(g) // 60)))
        order = np.argsort(g.p.values)
        xs, ys, lo, hi = [], [], [], []
        for b in np.array_split(order, nb):
            yb = g.y.values[b]
            xs.append(g.p.values[b].mean()); ys.append(yb.mean())
            se = np.sqrt(max(yb.mean() * (1 - yb.mean()), 1e-4) / len(b))
            lo.append(yb.mean() - 1.96 * se); hi.append(yb.mean() + 1.96 * se)
        ax.plot([0, 1], [0, 1], color=INK2, lw=0.8, ls="--")
        ax.vlines(xs, lo, hi, color=BLUE, lw=1, alpha=0.6)
        ax.plot(xs, ys, color=BLUE, lw=2, marker="o", ms=5)
        ax.set_title(f"{s}: n={len(g)}, O-E {100 * (g.y.mean() - g.p.mean()):+.1f} pts", loc="left", fontsize=9)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    for ax in axes[1]:
        ax.set_xlabel("mean predicted")
    for ax in axes[:, 0]:
        ax.set_ylabel("observed G+ rate")
    fig.suptitle("Reliability (equal-mass bins, 95 % CI) - top: most under-predicted, bottom: most over-predicted "
                 "states (2025-02)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "reliability_extreme_states.png", dpi=150); plt.close(fig)

    # 4 base rate vs mean predicted
    fig, ax = plt.subplots(figsize=(6, 5))
    _style(ax)
    e = st[st.eligible]
    lim = [0, max(e.base_rate.max(), e.mean_pred.max()) * 1.08]
    ax.plot(lim, lim, color=INK2, lw=0.8, ls="--")
    ax.scatter(e.mean_pred, e.base_rate, s=e.n / 8, color=BLUE, alpha=0.75, edgecolor=SURF, lw=0.8)
    for s in e.index:
        ax.annotate(s, (e.mean_pred[s], e.base_rate[s]), fontsize=6.5, color=INK2, xytext=(2, 2),
                    textcoords="offset points")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("mean predicted P(G+)"); ax.set_ylabel("observed G+ rate")
    ax.set_title("State base rate vs mean prediction (eligible states, 2025-02;\nbubble = state size; "
                 "on the line = calibrated)", loc="left")
    fig.tight_layout(); fig.savefig(FIG / "base_rate_vs_mean_pred.png", dpi=150); plt.close(fig)

    # 5 TPR vs FPR
    fig, ax = plt.subplots(figsize=(6, 5))
    _style(ax)
    ax.scatter(e.fpr_thr, e.tpr_thr, s=e.n / 8, color=BLUE, alpha=0.75, edgecolor=SURF, lw=0.8)
    for s in e.index:
        ax.annotate(s, (e.fpr_thr[s], e.tpr_thr[s]), fontsize=6.5, color=INK2, xytext=(2, 2),
                    textcoords="offset points")
    nat = d.assign(f=d.p >= thr)
    ax.scatter([nat.f[nat.y == 0].mean()], [nat.f[nat.y == 1].mean()], marker="X", s=90, color=ORANGE,
               edgecolor=SURF, zorder=4, label="national")
    ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate (recall)")
    ax.set_title("Error rates by state at the accuracy threshold (2025-02)", loc="left")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "tpr_fpr_by_state.png", dpi=150); plt.close(fig)

    # 6 stability
    j = st[["o_minus_e", "eligible", "n"]].join(st_prev[["o_minus_e"]], rsuffix="_prev")
    j = j[j.eligible]
    fig, ax = plt.subplots(figsize=(5.6, 5))
    _style(ax)
    lim = [100 * min(j.o_minus_e.min(), j.o_minus_e_prev.min()) - 2, 100 * max(j.o_minus_e.max(),
                                                                              j.o_minus_e_prev.max()) + 2]
    ax.plot(lim, lim, color=INK2, lw=0.8, ls="--")
    ax.axhline(0, color=GRID, lw=1); ax.axvline(0, color=GRID, lw=1)
    ax.scatter(100 * j.o_minus_e_prev, 100 * j.o_minus_e, s=j.n / 8, color=BLUE, alpha=0.75, edgecolor=SURF)
    for s in j.index:
        ax.annotate(s, (100 * j.o_minus_e_prev[s], 100 * j.o_minus_e[s]), fontsize=6.5, color=INK2,
                    xytext=(2, 2), textcoords="offset points")
    rho = stats.spearmanr(j.o_minus_e_prev, j.o_minus_e)[0]
    ax.set_xlabel("O-E 2024-11 (points)"); ax.set_ylabel("O-E 2025-02 (points)")
    ax.set_title(f"Is the state calibration gap stable? Spearman rho = {rho:.2f}", loc="left")
    fig.tight_layout(); fig.savefig(FIG / "stability_oe_2024-11_vs_2025-02.png", dpi=150); plt.close(fig)

    # 7 mitigation before/after
    m0, m2 = mit_tabs["M0"], mit_tabs["M2"]
    jj = m0[["o_minus_e", "eligible"]].join(m2[["o_minus_e"]], rsuffix="_m2")
    jj = jj[jj.eligible].sort_values("o_minus_e")
    fig, ax = plt.subplots(figsize=(6.5, 8))
    _style(ax)
    yy = np.arange(len(jj))
    ax.hlines(yy, 100 * jj.o_minus_e, 100 * jj.o_minus_e_m2, color=GRID, lw=1.5)
    ax.scatter(100 * jj.o_minus_e, yy, s=18, color=BLUE, zorder=3, label="original")
    ax.scatter(100 * jj.o_minus_e_m2, yy, s=18, color=ORANGE, zorder=3,
               label="after EB-shrunk state recalibration (fit on 2024-11)")
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(yy); ax.set_yticklabels(jj.index, fontsize=7)
    ax.set_xlabel("observed - expected on 2025-02 (points)")
    ax.set_title("Mitigation: per-state calibration gap before / after", loc="left")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.tight_layout(); fig.savefig(FIG / "mitigation_recalibration_oe.png", dpi=150); plt.close(fig)

    # 8 retrain comparison
    if retrain is not None:
        fig, ax = plt.subplots(figsize=(6, 5))
        _style(ax)
        ax.plot([-25, 25], [-25, 25], color=INK2, lw=0.8, ls="--")
        ax.axhline(0, color=GRID, lw=1); ax.axvline(0, color=GRID, lw=1)
        r = retrain
        ax.scatter(100 * r.oe_full, 100 * r.oe_no_info, s=r.n / 8, color=BLUE, alpha=0.75, edgecolor=SURF)
        for s in r.index:
            ax.annotate(s, (100 * r.oe_full[s], 100 * r.oe_no_info[s]), fontsize=6.5, color=INK2, xytext=(2, 2),
                        textcoords="offset points")
        lo = 100 * min(r.oe_full.min(), r.oe_no_info.min()) - 2
        hi = 100 * max(r.oe_full.max(), r.oe_no_info.max()) + 2
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("O-E, LightGBM with state features (points)")
        ax.set_ylabel("O-E, LightGBM without any state features")
        ax.set_title("Counterfactual: removing state information (2025-02)", loc="left")
        fig.tight_layout(); fig.savefig(FIG / "counterfactual_no_state_oe.png", dpi=150); plt.close(fig)


# ----------------------------------------------------------------------------------------------- audit stage
def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 6)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def stage_audit():
    TAB.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    summ = json.loads((EXP12 / "summary.json").read_text(encoding="utf-8"))
    thr = summ["test"]["test_2025-02"]["old_ens"]["threshold"]
    data = {m: load_month(m) for m in MONTHS}
    beds_cuts = np.nanquantile(data[PRIMARY].certified_beds, [1 / 3, 2 / 3])
    for m, d in data.items():
        d["beds_tercile"] = pd.cut(d.certified_beds, [-np.inf, *beds_cuts, np.inf],
                                   labels=["small", "medium", "large"]).astype(str).replace("nan", "unknown")
    R = {"meta": {"threshold_accuracy": thr, "min_support": {"n": MIN_N, "pos": MIN_POS, "neg": MIN_NEG},
                  "slice_min_support": {"n": SLICE_MIN_N, "pos": SLICE_MIN_POS, "neg": SLICE_MIN_NEG},
                  "bootstrap_B": B, "permutations": NPERM, "beds_tercile_cuts": beds_cuts.tolist()}}
    st, nat, tests = {}, {}, {}
    for m, d in data.items():
        R["meta"][f"rows_{m}"] = len(d)
        R["meta"][f"state_fallback_rows_{m}"] = int(d.state_source_fallback.sum())
        c10, c20 = np.quantile(d.p, 0.9), np.quantile(d.p, 0.8)
        R["meta"][f"cutoffs_{m}"] = {"accuracy_thr": thr, "top10": c10, "top20": c20}
        st[m] = per_state_table(d, thr, c10, c20)
        nat[m] = pd.Series(group_metrics(d.y.values, d.p.values, thr, c10, c20))
        tests[m] = rate_tests(d, st[m], cut=thr)
        log(f"{m}: {len(d)} rows, {st[m].eligible.sum()} eligible states of {len(st[m])}")
    d, dp = data[PRIMARY], data[SECOND]
    R["national"] = {m: nat[m].to_dict() for m in MONTHS}
    R["meta"]["n_states"] = {m: int(len(st[m])) for m in MONTHS}
    R["meta"]["n_eligible_states"] = {m: int(st[m].eligible.sum()) for m in MONTHS}

    # fairness summaries
    fs = {m: fairness_summary(st[m], nat[m]) for m in MONTHS}
    for m in MONTHS:
        fs[m].insert(0, "month", m)
    pd.concat(fs.values()).to_csv(TAB / "fairness_gap_summary.csv", index=False, float_format="%.4f")

    # bootstrap
    log("bootstrap per state ...")
    boot = bootstrap_states(d, thr, rng)
    pooled = pooled_cluster_bootstrap(pd.concat([d, dp]), rng)
    pooled.to_csv(TAB / "oe_pooled_two_months_cluster_bootstrap.csv", float_format="%.4f")

    # EB shrinkage (O-E on probability scale and TPR)
    var_oe = d.groupby("state").apply(lambda g: (g.p * (1 - g.p)).sum() / len(g) ** 2)
    dl_oe = dersimonian_laird(st[PRIMARY].o_minus_e.reindex(var_oe.index).values, var_oe.values)
    eb = pd.DataFrame({"oe_eb": dl_oe["shrunk"]}, index=var_oe.index)
    tp = st[PRIMARY].tpr_thr
    npos = st[PRIMARY].positives
    tpr_s = (tp * npos + 0.5) / (npos + 1)
    dl_tpr = dersimonian_laird(tpr_s.values, (tpr_s * (1 - tpr_s) / (npos + 1)).values)
    eb["tpr_eb"] = dl_tpr["shrunk"]
    fp_ = st[PRIMARY].fpr_thr
    nneg = st[PRIMARY].n - npos
    fpr_s = (fp_ * nneg + 0.5) / (nneg + 1)
    dl_fpr = dersimonian_laird(fpr_s.values, (fpr_s * (1 - fpr_s) / (nneg + 1)).values)
    eb["fpr_eb"] = dl_fpr["shrunk"]
    R["empirical_bayes"] = {k: {kk: vv for kk, vv in v.items() if kk != "shrunk"}
                            for k, v in {"o_minus_e": dl_oe, "tpr": dl_tpr, "fpr": dl_fpr}.items()}
    R["empirical_bayes"]["o_minus_e"]["tau_pp"] = 100 * np.sqrt(dl_oe["tau2"])

    # permutation tests
    log("permutation tests ...")
    perm = permutation_tests(d, thr, rng)
    perm.to_csv(TAB / "permutation_tests_2025-02.csv", index=False, float_format="%.5g")
    R["permutation_tests"] = perm.to_dict(orient="records")

    # multilevel
    log("multilevel models ...")
    R["multilevel"] = {}
    blups = {}
    for m in MONTHS:
        res, blup, blup_logit = multilevel(data[m])
        R["multilevel"][m] = res
        blups[m] = (blup, blup_logit)

    # per-state master table (primary month)
    master = st[PRIMARY].join(tests[PRIMARY][[c for c in tests[PRIMARY].columns if c.endswith("_q")]]) \
        .join(boot).join(eb).join(blups[PRIMARY][0]).join(blups[PRIMARY][1])
    master = master.join(st[SECOND][["n", "base_rate", "mean_pred", "o_minus_e", "tpr_thr", "fpr_thr",
                                      "roc_auc"]].add_suffix("_2024_11"))
    master.index.name = "state"
    master.sort_values("o_minus_e").to_csv(TAB / "per_state_metrics_2025-02.csv", float_format="%.4f")
    st[SECOND].join(tests[SECOND][[c for c in tests[SECOND].columns if c.endswith("_q")]]) \
        .sort_values("o_minus_e").to_csv(TAB / "per_state_metrics_2024-11.csv", float_format="%.4f")

    # significance counts
    R["significance_2025-02"] = {c: int((tests[PRIMARY][c] < 0.05).sum()) for c in tests[PRIMARY].columns
                                 if c.endswith("_q")}
    R["significance_2024-11"] = {c: int((tests[SECOND][c] < 0.05).sum()) for c in tests[SECOND].columns
                                 if c.endswith("_q")}
    e = master[master.eligible]
    R["calibration_dispersion"] = {
        "oe_sd_pp_all_states_unweighted": 100 * master.o_minus_e.std(),
        "oe_sd_pp_eligible": 100 * e.o_minus_e.std(),
        "oe_sd_pp_eligible_EB": 100 * e.oe_eb.std(),
        "oe_sd_pp_all_EB": 100 * master.oe_eb.std(),
        "expected_sd_pp_from_sampling_only_eligible": 100 * np.sqrt((var_oe[e.index]).mean()),
        "states_hl_p_lt_0.05_eligible": int((e.hl_p < 0.05).sum()),
        "states_hl_q_lt_0.05_eligible": int((bh(e.hl_p.values) < 0.05).sum()),
        "cal_slope_median_eligible": float(e.cal_slope.median()),
        "cal_slope_range_eligible": [float(e.cal_slope.min()), float(e.cal_slope.max())],
        "corr_state_gplus_feature_vs_base_rate": float(np.corrcoef(e.mean_state_gplus_rate_12m, e.base_rate)[0, 1]),
        "corr_state_gplus_feature_vs_mean_pred": float(np.corrcoef(e.mean_state_gplus_rate_12m, e.mean_pred)[0, 1]),
        "corr_mean_pred_vs_base_rate": float(np.corrcoef(e.mean_pred, e.base_rate)[0, 1]),
        "corr_base_rate_vs_oe": float(np.corrcoef(e.base_rate, e.o_minus_e)[0, 1]),
        "corr_oe_vs_(base_rate - state_gplus_rate_12m)": float(np.corrcoef(
            e.o_minus_e, e.base_rate - e.mean_state_gplus_rate_12m)[0, 1]),
        "slope_mean_pred_on_base_rate": float(np.polyfit(e.base_rate, e.mean_pred, 1)[0]),
        "within_state_auc_range_eligible": [float(e.roc_auc.min()), float(e.roc_auc.max())],
        "within_state_auc_n_weighted": float((e.n * e.roc_auc).sum() / e.n.sum()),
    }

    # heterogeneity of within-state ROC-AUC (Hanley-McNeil variance, DerSimonian-Laird)
    a_ = e.roc_auc.values
    n1, n0 = e.positives.values, (e.n - e.positives).values
    q1, q2 = a_ / (2 - a_), 2 * a_ ** 2 / (1 + a_)
    v_auc = (a_ * (1 - a_) + (n1 - 1) * (q1 - a_ ** 2) + (n0 - 1) * (q2 - a_ ** 2)) / (n1 * n0)
    dl_auc = dersimonian_laird(a_, v_auc)
    R["within_state_auc_heterogeneity"] = {k: v for k, v in dl_auc.items() if k != "shrunk"}
    R["within_state_auc_heterogeneity"]["tau"] = float(np.sqrt(dl_auc["tau2"]))
    R["within_state_auc_heterogeneity"]["eb_range"] = [float(np.nanmin(dl_auc["shrunk"])),
                                                       float(np.nanmax(dl_auc["shrunk"]))]

    # stability
    j = st[PRIMARY].join(st[SECOND], rsuffix="_prev")
    je = j[j.eligible & j.eligible_prev]
    def _stab(a, b):
        ok = a.notna() & b.notna()
        a, b = a[ok], b[ok]
        return {"spearman": float(stats.spearmanr(a, b)[0]), "pearson": float(np.corrcoef(a, b)[0, 1]),
                "n": int(ok.sum())}
    R["stability"] = {k: _stab(je[k], je[k + "_prev"])
                      for k in ("o_minus_e", "base_rate", "mean_pred", "tpr_thr", "fpr_thr", "precision_thr",
                                "roc_auc", "flag_rate_thr")}
    R["stability"]["n_states"] = int(len(je))
    R["stability"]["oe_sig_both_months_same_sign"] = sorted(
        set(tests[PRIMARY].index[(tests[PRIMARY].oe_q < 0.05)]) & set(tests[SECOND].index[(tests[SECOND].oe_q < 0.05)])
        & set(je.index[np.sign(je.o_minus_e) == np.sign(je.o_minus_e_prev)]))
    R["stability"]["blup_logit_corr"] = float(blups[PRIMARY][1].corr(blups[SECOND][1], method="spearman")) \
        if len(blups[PRIMARY][1]) else None

    # AUC decomposition
    prev_rate = dp.groupby("state").y.mean().to_dict()
    R["auc_decomposition"] = {PRIMARY: auc_decomposition(d, prev_rate)}
    R["auc_decomposition"][SECOND] = auc_decomposition(dp, d.groupby("state").y.mean().to_dict())
    R["auc_decomposition"][SECOND]["note"] = "state-base-rate-only score for 2024-11 uses 2025-02 rates (reference)"

    # intersectional slices
    log("slice discovery ...")
    f = (d.p >= thr)
    nat_rates = {"pos": int(d.y.sum()), "neg": int((1 - d.y).sum()), "fn": int(((~f) & (d.y == 1)).sum()),
                 "fp": int((f & (d.y == 0)).sum())}
    sl = slice_discovery(d, nat_rates, thr)
    sl = slice_replication(sl, dp, thr)
    # does the intersection add miscalibration beyond its state? residual net of the state's mean residual
    dr = d.assign(r=(d.y - d.p) - (d.y - d.p).groupby(d.state).transform("mean"), v=d.p * (1 - d.p))
    exc, exc_p = [], []
    for sname in sl.slice:
        m_ = np.ones(len(dr), bool)
        for part in sname.split(" & "):
            a, v = part.split("=", 1)
            m_ &= (dr[a].astype(str) == v).values
        g = dr[m_]
        exc.append(g.r.mean())
        exc_p.append(2 * stats.norm.sf(abs(g.r.sum() / np.sqrt(g.v.sum()))))
    sl["oe_excess_over_state"], sl["oe_excess_p"] = exc, exc_p
    cand = sl.slice.str.contains("state=") & (sl.order > 1)
    sl["oe_excess_q"] = np.nan
    sl.loc[cand, "oe_excess_q"] = bh(sl.loc[cand, "oe_excess_p"].values)
    sl["oe_replicates"] = (np.sign(sl.o_minus_e) == np.sign(sl.o_minus_e_other_month)) & (sl.oe_q < 0.05)
    sl.sort_values("oe_p").to_csv(TAB / "slices_state_beds_ownership_2025-02.csv", index=False,
                                   float_format="%.4f")
    R["slices"] = {"n_slices_tested": int(len(sl)), "n_tests": int(3 * len(sl)),
                   "n_oe_sig_BH": int((sl.oe_q < 0.05).sum()), "n_fnr_sig_BH": int((sl.fnr_q < 0.05).sum()),
                   "n_fpr_sig_BH": int((sl.fpr_q < 0.05).sum()),
                   "n_oe_sig_and_same_sign_2024_11": int(sl.oe_replicates.sum()),
                   "by_order_oe_sig": sl[sl.oe_q < 0.05].groupby("order").size().to_dict(),
                   "n_state_intersections": int(cand.sum()),
                   "n_state_intersections_excess_sig_BH": int((sl.oe_excess_q < 0.05).sum()),
                   "state_intersections_excess_sig": sl[sl.oe_excess_q < 0.05][
                       ["slice", "n", "o_minus_e", "oe_excess_over_state", "oe_excess_q",
                        "o_minus_e_other_month"]].to_dict(orient="records"),
                   "non_state_slices": sl[~sl.slice.str.contains("state=")][
                       ["slice", "n", "base_rate", "mean_pred", "o_minus_e", "oe_q", "fnr", "fnr_q", "fpr",
                        "fpr_q"]].to_dict(orient="records")}

    # mitigation
    log("mitigation ...")
    mit, mit_tabs, si, dl_logit = mitigation(dp, d, thr)
    mit.to_csv(TAB / "mitigation_recalibration.csv", index=False, float_format="%.4f")
    si.to_csv(TAB / "state_logit_intercepts_fit_2024-11.csv", float_format="%.4f")
    R["mitigation"] = mit.to_dict(orient="records")
    R["logit_intercepts_2024_11_DL"] = {k: v for k, v in dl_logit.items() if k != "shrunk"}

    # retrain outputs (if present)
    rp = TAB / "retrain_preds.csv.gz"
    retr = None
    if rp.exists():
        log("retrain comparison ...")
        rr = pd.read_csv(rp, dtype={"ccn": str, "t": str})
        R["retrain"] = {}
        rows = []
        for m in MONTHS:
            dm = data[m][["ccn", "state", "state_gplus_rate_12m", "y"]].merge(
                rr[rr.t == m].drop(columns=["y"]), on="ccn")
            rf_thr = None
            for arm in ("lgb_full", "lgb_no_state", "lgb_no_state_info"):
                # same national flag rate as the main model at its accuracy threshold
                fr = float((data[m].p >= thr).mean())
                a_thr = float(np.quantile(dm[arm], 1 - fr))
                r, t = summarise_model(dm, arm, a_thr, arm)
                r["month"] = m
                rows.append(r)
                if m == PRIMARY:
                    t.to_csv(TAB / f"retrain_per_state_{arm}_2025-02.csv", float_format="%.4f")
                    if arm == "lgb_full":
                        rf_thr = t
                    if arm == "lgb_no_state_info":
                        retr = pd.DataFrame({"oe_full": rf_thr.o_minus_e, "oe_no_info": t.o_minus_e,
                                             "n": t.n})[rf_thr.eligible]
            # bootstrap CI of PR-AUC delta (facility = row within a month)
            y = dm.y.values
            dd = []
            for _ in range(300):
                i = rng.integers(0, len(y), len(y))
                dd.append([average_precision_score(y[i], dm.lgb_no_state_info.values[i])
                           - average_precision_score(y[i], dm.lgb_full.values[i]),
                           auc_fast(y[i], dm.lgb_no_state_info.values[i]) - auc_fast(y[i], dm.lgb_full.values[i])])
            dd = np.array(dd)
            R["retrain"][f"delta_no_state_info_minus_full_{m}"] = {
                "pr_auc_ci": np.percentile(dd[:, 0], [2.5, 97.5]).tolist(),
                "roc_auc_ci": np.percentile(dd[:, 1], [2.5, 97.5]).tolist()}
        # reproduction check + deployable recalibration (fit on PAST, labels complete before 2025-02)
        if "ens_full" in rr.columns and (rr.t == PAST).any():
            dm = data[PRIMARY].merge(rr[rr.t == PRIMARY][["ccn", "ens_full"]], on="ccn")
            R["retrain"]["ens_reproduction_max_abs_diff_vs_old_ens"] = float((dm.ens_full - dm.old_ens).abs().max())
            pastg = pd.read_parquet(GOLD / f"t={PAST}" / "part-0.parquet", columns=["ccn", "state"])
            dpast = rr[rr.t == PAST][["ccn", "y", "ens_full"]].merge(pastg, on="ccn", how="left")
            dpast["p"] = dpast.ens_full.clip(1e-6, 1 - 1e-6)
            dpast["state"] = dpast.state.fillna(dpast.ccn.str[:2].map(SSA))
            de = dm.assign(p=dm.ens_full.clip(1e-6, 1 - 1e-6))
            mit2, mt2, si2, _ = mitigation(dpast, de, thr, PAST)
            mit2.to_csv(TAB / "mitigation_recalibration_fit_2024-01.csv", index=False, float_format="%.4f")
            si2.to_csv(TAB / "state_logit_intercepts_fit_2024-01.csv", float_format="%.4f")
            R["mitigation_deployable_fit_2024_01"] = mit2.to_dict(orient="records")
            R["meta"]["rows_2024-01"] = int(len(dpast))
        rt = pd.DataFrame(rows)
        rt.to_csv(TAB / "retrain_counterfactual_summary.csv", index=False, float_format="%.4f")
        R["retrain"]["summary"] = rt.to_dict(orient="records")
        sc = TAB / "shap_state_contrib.csv"
        if sc.exists():
            c = pd.read_csv(sc)
            cp = c[c.month == PRIMARY].set_index("state").join(st[PRIMARY][["o_minus_e", "base_rate", "eligible"]])
            ce = cp[cp.eligible]
            R["retrain"]["shap"] = {
                "mean_abs_share_state_feats_national": float((cp.n * cp.abs_share_state_feats).sum() / cp.n.sum()),
                "shap_state_range_logit": [float(ce.shap_all_state_feats.min()), float(ce.shap_all_state_feats.max())],
                "corr_shap_state_feats_vs_base_rate": float(np.corrcoef(ce.shap_all_state_feats, ce.base_rate)[0, 1]),
                "corr_shap_state_feats_vs_oe": float(np.corrcoef(ce.shap_all_state_feats, ce.o_minus_e)[0, 1])}
    else:
        R["retrain"] = "not run (retrain_preds.csv.gz missing; run --stage retrain with >= 2.5 GB free RAM)"

    # plots
    log("plots ...")
    e = master[master.eligible]
    under = e.sort_values("oe_eb", ascending=False).index[:3].tolist()
    over = e.sort_values("oe_eb").index[:3].tolist()
    plots(st[PRIMARY], boot, eb, d, dp, st[SECOND], under + over, thr, mit_tabs, retr)
    R["reliability_states"] = {"most_under_predicted": under, "most_over_predicted": over}
    (HERE / "results.json").write_text(json.dumps(jsonable(R), indent=1), encoding="utf-8")
    log("audit done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["audit", "retrain"], default="audit")
    a = ap.parse_args()
    TAB.mkdir(parents=True, exist_ok=True)
    stage_retrain() if a.stage == "retrain" else stage_audit()
