"""exp12: does PBJ daily nurse staffing add signal? One change only: 133 old features vs 133 + PBJ features.

Setup = the recommended model (exp9 R3): tuned LightGBM + logistic regression, mean of probabilities, trained on
all labelled rows 2023-01..2023-11. Scored once on:
  - test 2025-02 (official test month), and
  - 2024-11 (extra check: labels of the training rows end by 2024-11, this month's label window starts 2024-12).
Keep rule (FE_Rows.md): 3 facility-grouped folds on the training window, keep if mean Δ PR-AUC > 1 std.
Accuracy threshold: the one that maximises accuracy on the out-of-fold training predictions (per arm).

Usage: uv run --no-sync python exp12_pbj.py      (needs outputs/pbj_monthly.parquet from pbj_features.py)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "experiments_rev002"))
import exp_common as ec  # noqa: E402
from exp8_lr import fit_lr  # noqa: E402

OUT = HERE / "outputs" / "exp12_pbj"
TRAIN = ("2023-01", "2023-11")
TESTS = {"test_2025-02": "2025-02", "extra_2024-11": "2024-11"}
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def load_rows(num: list[str], cat: list[str]) -> pd.DataFrame:
    """Labelled gold rows of the training window and the two test months, old features only."""
    want = {f"t={m}" for m in TESTS.values()}
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
    df["t"] = df["t"].astype(str)
    df["y"] = df["y_g12"].astype("int8")
    df = df.drop(columns=["y_g12", "status_g12"])
    for c in cat:
        df[c] = df[c].astype("string").fillna("__missing__").astype("category")
    return df


def fit_predict(tr: pd.DataFrame, tes: list[pd.DataFrame], cols: list[str], num: list[str], cat: list[str],
                params: dict) -> tuple[list[dict], lgb.LGBMClassifier]:
    lg = lgb.LGBMClassifier(**params).fit(tr[cols], tr.y.values)
    lr = fit_lr(tr, cols, num, cat)
    out = []
    for te in tes:
        a, b = lg.predict_proba(te[cols])[:, 1], lr.predict_proba(te[cols])[:, 1]
        out.append({"lgbm": a, "lr": b, "ens": (a + b) / 2})
    return out, lg


def acc_threshold(y: np.ndarray, p: np.ndarray) -> float:
    grid = np.unique(np.quantile(p, np.linspace(0.5, 0.995, 200)))
    return float(grid[np.argmax([np.mean((p >= g) == y) for g in grid])])


def boot(y: np.ndarray, groups: np.ndarray, pa: np.ndarray, pb: np.ndarray, n: int = 300) -> dict:
    rng = np.random.default_rng(ec.SEED)
    idx = pd.Series(np.arange(len(groups))).groupby(groups).apply(np.array).to_dict()
    uniq = np.array(list(idx))
    d_pr, d_roc = [], []
    for _ in range(n):
        pick = np.concatenate([idx[g] for g in rng.choice(uniq, len(uniq), replace=True)])
        ma, mb = ec.metrics(y[pick], pa[pick]), ec.metrics(y[pick], pb[pick])
        d_pr.append(mb["pr_auc"] - ma["pr_auc"])
        d_roc.append(mb["roc_auc"] - ma["roc_auc"])
    q = lambda v: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]  # noqa: E731
    return {"d_pr_auc_ci": q(d_pr), "d_roc_auc_ci": q(d_roc)}


def main() -> None:
    free = ec.free_ram_gb()
    if free < 2.5:
        sys.exit(f"only {free:.2f} GB RAM free (< 2.5 GB); close other apps and rerun")
    OUT.mkdir(parents=True, exist_ok=True)
    num, cat = ec.feature_lists()  # rev002 registry = the old 133 features
    old = num + cat
    df = load_rows(num, cat)
    pbj = pd.read_parquet(HERE / "outputs" / "pbj_monthly.parquet").drop(columns=["pbj_quarter"])
    pcols = [c for c in pbj.columns if c.startswith("pbj_")]
    df = df.merge(pbj, on=["ccn", "t"], how="left")
    numx = num + pcols
    new = old + pcols
    params = json.loads((ec.ROOT.parent / "feature engineering run/spine2022/boost/lgbm_best_params.json")
                        .read_text(encoding="utf-8"))["params"]
    tr = df[(df.t >= TRAIN[0]) & (df.t <= TRAIN[1])].reset_index(drop=True)
    tes = {k: df[df.t == m].reset_index(drop=True) for k, m in TESTS.items()}
    cover = {k: float(v[pcols[0]].notna().mean()) for k, v in {"train": tr, **tes}.items()}
    log(f"{len(old)} old + {len(pcols)} PBJ features; train {len(tr):,} rows, "
        + ", ".join(f"{k} {len(v):,}" for k, v in tes.items()) + f"; PBJ coverage {cover}")
    res: dict = {"meta": {"train": TRAIN, "train_rows": len(tr), "pbj_features": pcols, "pbj_coverage": cover,
                          "params": params, "free_ram_gb_start": round(free, 2)}}

    # 1) keep rule: 3 facility-grouped folds on the training window
    folds = list(GroupKFold(n_splits=3).split(tr, groups=tr.ccn.values))
    oof = {arm: {m: np.zeros(len(tr)) for m in ("lgbm", "lr", "ens")} for arm in ("old", "pbj")}
    cv = {arm: {m: [] for m in ("lgbm", "lr", "ens")} for arm in ("old", "pbj")}
    for i, (a, b) in enumerate(folds):
        for arm, cols, n_ in (("old", old, num), ("pbj", new, numx)):
            (pr,), _ = fit_predict(tr.iloc[a], [tr.iloc[b]], cols, n_, cat, params)
            for m, p in pr.items():
                oof[arm][m][b] = p
                cv[arm][m].append(ec.metrics(tr.y.values[b], p)["pr_auc"])
        log(f"fold {i + 1}/3: ens PR-AUC old {cv['old']['ens'][-1]:.4f} pbj {cv['pbj']['ens'][-1]:.4f} "
            f"(free RAM {ec.free_ram_gb():.2f} GB)")
    res["cv"] = {}
    for m in ("lgbm", "lr", "ens"):
        d = np.array(cv["pbj"][m]) - np.array(cv["old"][m])
        res["cv"][m] = {"old": cv["old"][m], "pbj": cv["pbj"][m], "delta_mean": float(d.mean()),
                        "delta_std": float(d.std(ddof=1)), "keep": bool(d.mean() > d.std(ddof=1))}
        log(f"CV {m:4s}: Δ PR-AUC {d.mean():+.4f} ± {d.std(ddof=1):.4f} -> keep {res['cv'][m]['keep']}")
    thr = {arm: acc_threshold(tr.y.values, oof[arm]["ens"]) for arm in oof}

    # 2) final fit on the whole window, scored once on each test month
    preds, models = {}, {}
    for arm, cols, n_ in (("old", old, num), ("pbj", new, numx)):
        preds[arm], models[arm] = fit_predict(tr, list(tes.values()), cols, n_, cat, params)
    res["test"] = {}
    for j, (k, te) in enumerate(tes.items()):
        y, g = te.y.values, te.ccn.values
        r = {}
        for arm in ("old", "pbj"):
            for m, p in preds[arm][j].items():
                mm = ec.metrics(y, p)
                mm["p_at_2"] = ec.precision_at(y, p, 0.02)
                if m == "ens":
                    mm["accuracy"] = float(np.mean((p >= thr[arm]) == y))
                    mm["threshold"] = thr[arm]
                r[f"{arm}_{m}"] = mm
        for m in ("lgbm", "lr", "ens"):
            r[f"delta_{m}"] = boot(y, g, preds["old"][j][m], preds["pbj"][j][m])
        r["majority_accuracy"] = float(max(y.mean(), 1 - y.mean()))
        res["test"][k] = r
        log(f"{k}: ens PR-AUC old {r['old_ens']['pr_auc']:.4f} pbj {r['pbj_ens']['pr_auc']:.4f} "
            f"CI {r['delta_ens']['d_pr_auc_ci']}; acc old {r['old_ens']['accuracy']:.4f} "
            f"pbj {r['pbj_ens']['accuracy']:.4f}")
        pd.DataFrame({"ccn": g, "t": te.t.values, "y": y, "old_ens": preds["old"][j]["ens"],
                      "pbj_ens": preds["pbj"][j]["ens"]}).to_csv(OUT / f"preds_{k}.csv", index=False)

    imp = pd.DataFrame({"feature": new, "gain": models["pbj"].booster_.feature_importance("gain")})
    imp = imp.sort_values("gain", ascending=False).reset_index(drop=True)
    imp["rank"] = imp.index + 1
    imp.to_csv(OUT / "gain_pbj_model.csv", index=False)
    res["pbj_gain_ranks"] = imp[imp.feature.isin(pcols)][["feature", "rank"]].head(15).values.tolist()
    log("top PBJ features by gain rank: " + ", ".join(f"{f} #{r}" for f, r in res["pbj_gain_ranks"][:8]))
    (OUT / "summary.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    log("done")


if __name__ == "__main__":
    main()
