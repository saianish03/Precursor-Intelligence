# experiments_rev003: PBJ daily nurse staffing (exp12)

Question: do CMS Payroll-Based Journal (PBJ) **daily** nurse staffing features add signal to the 133-feature model?
One change only (FE_Rows.md rules).

## Status (2026-11-07 ~11:00)

| Step | Script | Status |
|---|---|---|
| 1. Download 22 quarters 2021Q1–2026Q2 (~5.5 GB) | `pbj_download.py` → `data/incoming_pbj/`, `pbj_catalog.csv` | **done** (`pbj_download.log`) |
| 2. Features: facility-quarter → point-in-time facility-month | `pbj_features.py` → `outputs/pbj_quarterly.parquet`, `outputs/pbj_monthly.parquet` | **done** (759,985 rows, 52 months; `pbj_features.log`) |
| 3. Experiment: 133 old vs 133 + PBJ | `exp12_pbj.py` → `outputs/exp12_pbj/` | **done** (16 min; `exp12_pbj.log`) |
| 4. Write `report.md` here with results | `report.md` | **done** |

Resume: `cd notebooks/experiments_rev003 && uv run --no-sync python exp12_pbj.py > exp12_pbj.log 2>&1`
(the script exits by itself if < 2.5 GB RAM is free).

## Point-in-time rule
Published month = the upload folder in the CMS file URL (first time the file was online). Month t uses the newest
quarter published strictly before t, plus q-1 and q-4 for change features. E.g. 2023-01 → 2022Q2; test 2025-02 → 2024Q3.
Caveat: we download the latest revision of each quarter; revisions after first upload are small.

## Features (24)
Per quarter: census mean; HPRD total / RN / LPN / aide; contract share total / RN; weekend÷weekday HPRD (total, RN);
CV and 5th percentile of daily HPRD; share of days with zero RN, below 3.48 HPRD, RN below 0.55; share of days
reported; census trend. Changes vs q-1 and q-4: HPRD total, HPRD RN, contract share, census %.
Provider Info in gold already has PBJ quarterly averages; the new information is the daily detail.

## Experiment design (exp12)
- Train: all labelled rows 2023-01..2023-11 (exp9 R3 = recommended window); tuned LightGBM + LR, mean ensemble.
- Keep rule: 3 facility-grouped folds on the training window; keep if mean Δ PR-AUC > 1 std.
- Test (once): 2025-02 (official) and 2024-11 (extra; training labels end by 2024-11, its window starts 2024-12).
- Facility-clustered bootstrap CIs (300) for Δ; accuracy threshold picked on out-of-fold training predictions.
