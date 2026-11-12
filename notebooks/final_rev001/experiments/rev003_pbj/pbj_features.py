"""PBJ daily nurse staffing -> facility-quarter features -> point-in-time facility-month features.

Step 1 (per quarter, chunked, low RAM): daily rows -> one row per facility-quarter (outputs/pbj_quarterly.parquet).
Step 2: for each month t, use the newest quarter q with published(q) < t (published = the month CMS first put the
file online, from pbj_catalog.csv), plus q-1 and q-4 for change features (outputs/pbj_monthly.parquet, keys ccn, t).

Provider Info (already in gold) carries PBJ quarterly *averages*; what is new here is the daily detail: weekend
dips, day-to-day variability, days below staffing floors, days with no RN, contract (agency) share.

Usage: uv run --no-sync python pbj_features.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RAW = ROOT / "data" / "incoming_pbj"
OUT = HERE / "outputs"
QUARTERLY = OUT / "pbj_quarterly.parquet"
MONTHLY = OUT / "pbj_monthly.parquet"

RN = ["Hrs_RNDON", "Hrs_RNadmin", "Hrs_RN"]
LPN = ["Hrs_LPNadmin", "Hrs_LPN"]
AIDE = ["Hrs_CNA", "Hrs_NAtrn", "Hrs_MedAide"]
ALL = RN + LPN + AIDE
CTR = [f"{c}_ctr" for c in ALL]
USE = ["PROVNUM", "WorkDate", "MDScensus", *ALL, *CTR]


def daily(path: Path) -> pd.DataFrame:
    """Read one quarter in chunks and keep only the per-day derived columns (float32)."""
    parts = []
    dtypes = {c: "float32" for c in USE[2:]} | {"PROVNUM": "string", "WorkDate": "string"}
    for ch in pd.read_csv(path, usecols=USE, dtype=dtypes, encoding="latin-1", chunksize=300_000):
        d = pd.DataFrame({
            "ccn": ch["PROVNUM"].str.strip().str.zfill(6),
            "date": pd.to_datetime(ch["WorkDate"], format="%Y%m%d"),
            "census": ch["MDScensus"],
            "rn": ch[RN].sum(axis=1),
            "lpn": ch[LPN].sum(axis=1),
            "aide": ch[AIDE].sum(axis=1),
            "ctr": ch[CTR].sum(axis=1),
            "ctr_rn": ch[[f"{c}_ctr" for c in RN]].sum(axis=1),
        })
        d["total"] = d["rn"] + d["lpn"] + d["aide"]
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def quarter_features(d: pd.DataFrame, quarter: str) -> pd.DataFrame:
    n_days = d["date"].nunique()
    d = d[d["census"] > 0].copy()
    d["wkend"] = d["date"].dt.dayofweek >= 5
    d["hprd"] = d["total"] / d["census"]
    d["hprd_rn"] = d["rn"] / d["census"]
    d["month"] = d["date"].dt.month
    g = d.groupby("ccn", sort=False)
    s = g[["census", "rn", "lpn", "aide", "total", "ctr", "ctr_rn"]].sum()
    f = pd.DataFrame(index=s.index)
    f["pbj_census_mean"] = g["census"].mean()
    for k in ("total", "rn", "lpn", "aide"):
        f[f"pbj_hprd_{k}"] = s[k] / s["census"]
    f["pbj_ctr_share_total"] = s["ctr"] / s["total"].replace(0, np.nan)
    f["pbj_ctr_share_rn"] = s["ctr_rn"] / s["rn"].replace(0, np.nan)
    for k, col in (("total", "total"), ("rn", "rn")):
        we = d[d.wkend].groupby("ccn")[[col, "census"]].sum()
        wd = d[~d.wkend].groupby("ccn")[[col, "census"]].sum()
        f[f"pbj_wkend_ratio_{k}"] = (we[col] / we["census"]) / (wd[col] / wd["census"]).replace(0, np.nan)
    f["pbj_cv_hprd_total"] = g["hprd"].std() / g["hprd"].mean()
    f["pbj_p05_hprd_total"] = g["hprd"].quantile(0.05)
    flags = pd.DataFrame({"ccn": d["ccn"], "z": d["rn"] <= 0, "b": d["hprd"] < 3.48, "r": d["hprd_rn"] < 0.55})
    fm = flags.groupby("ccn", sort=False)[["z", "b", "r"]].mean()
    f["pbj_frac_days_zero_rn"] = fm["z"]
    f["pbj_frac_days_below_348"] = fm["b"]
    f["pbj_frac_days_rn_below_055"] = fm["r"]
    f["pbj_frac_days_reported"] = g.size() / n_days
    mon = d.groupby(["ccn", "month"])["census"].mean().unstack()
    f["pbj_census_trend"] = mon.iloc[:, -1] / mon.iloc[:, 0].replace(0, np.nan) - 1
    f = f.replace([np.inf, -np.inf], np.nan).astype("float32").reset_index()
    f.insert(1, "quarter", quarter)
    return f


def q_shift(q: str, k: int) -> str:
    y, n = int(q[:4]), int(q[-1]) - 1 + k
    return f"{y + n // 4}Q{n % 4 + 1}"


def step1(catalog: list[dict]) -> pd.DataFrame:
    have = pd.read_parquet(QUARTERLY) if QUARTERLY.exists() else pd.DataFrame(columns=["quarter"])
    done = set(have["quarter"])
    out = [have] if len(have) else []
    for r in catalog:
        q, path = r["quarter"], RAW / f"pbj_nurse_{r['quarter']}.csv"
        if q in done or not path.exists():
            continue
        f = quarter_features(daily(path), q)
        out.append(f)
        pd.concat(out, ignore_index=True).to_parquet(QUARTERLY, index=False)  # checkpoint after every quarter
        print(f"{q}: {len(f):,} facilities", flush=True)
    return pd.read_parquet(QUARTERLY)


def step2(qf: pd.DataFrame, catalog: list[dict], months: list[str]) -> pd.DataFrame:
    pub = {r["quarter"]: r["published"] for r in catalog}
    feats = [c for c in qf.columns if c.startswith("pbj_")]
    by_q = {q: g.set_index("ccn")[feats] for q, g in qf.groupby("quarter")}
    rows = []
    for t in months:
        avail = sorted(q for q, p in pub.items() if p < t and q in by_q)  # published strictly before month t
        if not avail:
            continue
        q = avail[-1]
        cur = by_q[q].copy()
        for lag in (1, 4):
            prev = by_q.get(q_shift(q, -lag))
            if prev is None:
                continue
            prev = prev.reindex(cur.index)
            cur[f"pbj_d{lag}_hprd_total"] = cur["pbj_hprd_total"] - prev["pbj_hprd_total"]
            cur[f"pbj_d{lag}_hprd_rn"] = cur["pbj_hprd_rn"] - prev["pbj_hprd_rn"]
            cur[f"pbj_d{lag}_ctr_share_total"] = cur["pbj_ctr_share_total"] - prev["pbj_ctr_share_total"]
            cur[f"pbj_d{lag}_census_pct"] = cur["pbj_census_mean"] / prev["pbj_census_mean"].replace(0, np.nan) - 1
        cur = cur.reset_index()
        cur.insert(1, "t", t)
        cur.insert(2, "pbj_quarter", q)
        rows.append(cur)
    m = pd.concat(rows, ignore_index=True)
    num = [c for c in m.columns if c.startswith("pbj_") and c != "pbj_quarter"]
    m[num] = m[num].replace([np.inf, -np.inf], np.nan).astype("float32")
    return m


def main() -> None:
    OUT.mkdir(exist_ok=True)
    with (HERE / "pbj_catalog.csv").open(encoding="utf-8") as fh:
        catalog = list(csv.DictReader(fh))
    qf = step1(catalog)
    gold = ROOT / "data" / "gold" / "facility_month_dataset"
    months = sorted(p.name[2:] for p in gold.glob("t=*"))
    m = step2(qf, catalog, months)
    m.to_parquet(MONTHLY, index=False)
    print(f"monthly: {len(m):,} rows, {m.t.nunique()} months, quarters used per month:", flush=True)
    print(m.groupby("t")["pbj_quarter"].first().to_string(), flush=True)


if __name__ == "__main__":
    main()
