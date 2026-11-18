# exp12: PBJ daily nurse staffing features — report (2026-11-07)

**Question:** do CMS Payroll-Based Journal (PBJ) *daily* staffing features add signal on top of the 133 gold features?
**One change only** (FE_Rows.md): arm `old` = 133 features, arm `pbj` = 133 + 24 PBJ features. Same rows, same params.

**Answer: real but tiny.** PBJ passes the keep rule for LightGBM and the ensemble (Δ PR-AUC ≈ +0.001 to +0.002),
does nothing for LR, and does **not** move accuracy (≈ 75.5–75.8 %, both arms). It does not break the ~76 % accuracy ceiling.

## Setup

| Item | Value |
|---|---|
| Train | all labelled rows 2023-01..2023-11 — 162,276 rows |
| Tests (scored once) | 2025-02 official (14,643 rows, base rate 25.6 %); 2024-11 extra (14,659 rows, base rate 26.2 %) |
| Models | tuned LightGBM (`spine2022/boost/lgbm_best_params.json`), LR, mean ensemble |
| Keep rule | 3 facility-grouped folds on train; keep if mean Δ PR-AUC > 1 std |
| CIs | facility-clustered bootstrap (300) on test Δ |
| Accuracy threshold | maximises accuracy on out-of-fold training predictions (per arm) |
| PBJ coverage | train 97.8 %, test 98.5 %, extra 98.4 % (missing = no PBJ quarter published yet / facility not reporting) |
| Point-in-time | month t uses the newest PBJ quarter published strictly before t (e.g. test 2025-02 → 2024Q3) |
| Runtime | 16 min, free RAM 2.45–2.83 GB |

**PBJ features (24):** census mean; HPRD total / RN / LPN / aide; contract share total / RN; weekend÷weekday HPRD
(total, RN); CV and 5th percentile of daily HPRD; share of days with zero RN, below 3.48 HPRD, RN below 0.55;
share of days reported; census trend; changes vs q-1 and q-4 of HPRD total, HPRD RN, contract share, census %.
(Earlier notes said 31; the built set is 24.)

## Keep rule (3-fold CV on training window, PR-AUC)

| Model | old (folds) | pbj (folds) | Δ mean ± std | Keep |
|---|---|---|---|---|
| LightGBM | 0.5269 / 0.5193 / 0.5214 | 0.5277 / 0.5219 / 0.5240 | **+0.0020 ± 0.0011** | yes |
| LR | 0.5131 / 0.5115 / 0.5157 | 0.5116 / 0.5110 / 0.5165 | −0.0004 ± 0.0012 | no |
| Ensemble | 0.5236 / 0.5197 / 0.5217 | 0.5240 / 0.5211 / 0.5234 | **+0.0011 ± 0.0007** | yes |

## Test results

### 2025-02 (official test)

| Model | PR-AUC | ROC-AUC | P@2 % | P@5 % | P@10 % | Brier | Accuracy |
|---|---|---|---|---|---|---|---|
| old LightGBM | 0.4773 | 0.7080 | 79.9 | 67.8 | 57.6 | 0.1703 | |
| pbj LightGBM | 0.4785 | 0.7087 | 79.9 | 66.7 | 57.5 | 0.1700 | |
| old LR | 0.4776 | 0.7073 | 77.8 | 67.8 | 57.7 | 0.1694 | |
| pbj LR | 0.4776 | 0.7068 | 79.2 | 68.9 | 57.9 | 0.1695 | |
| **old ensemble** | **0.4802** | 0.7092 | 80.2 | 67.8 | 57.5 | 0.1690 | **75.80 %** |
| **pbj ensemble** | **0.4810** | 0.7095 | 80.9 | 69.1 | 57.4 | 0.1689 | **75.72 %** |
| majority class | | | | | | | 74.39 % |

Δ PR-AUC 95 % CI: LightGBM [−0.0001, +0.0027], LR [−0.0016, +0.0014], ensemble [−0.0004, +0.0018] — all include 0.

### 2024-11 (extra test)

| Model | PR-AUC | ROC-AUC | P@2 % | P@5 % | P@10 % | Brier | Accuracy |
|---|---|---|---|---|---|---|---|
| old LightGBM | 0.4881 | 0.7097 | 82.3 | 68.4 | 58.5 | 0.1719 | |
| pbj LightGBM | 0.4895 | 0.7108 | 81.6 | 68.9 | 58.5 | 0.1715 | |
| old LR | 0.4844 | 0.7085 | 78.8 | 68.6 | 58.3 | 0.1715 | |
| pbj LR | 0.4853 | 0.7083 | 79.5 | 68.9 | 58.5 | 0.1715 | |
| **old ensemble** | **0.4892** | 0.7107 | 81.6 | 68.6 | 59.0 | 0.1709 | **75.46 %** |
| **pbj ensemble** | **0.4904** | 0.7112 | 83.6 | 69.3 | 58.8 | 0.1706 | **75.46 %** |
| majority class | | | | | | | 73.80 % |

Δ PR-AUC 95 % CI: LightGBM [0.0000, +0.0027], LR [−0.0005, +0.0024], ensemble [**+0.0001, +0.0023**] (just excludes 0).

## Which PBJ features the model used (gain rank among 157)

pbj_census_mean #13, pbj_ctr_share_total #15, pbj_p05_hprd_total #31, pbj_hprd_total #44, pbj_ctr_share_rn #47,
pbj_hprd_rn #51, pbj_hprd_lpn #52, pbj_cv_hprd_total #60. Full list: `outputs/exp12_pbj/gain_pbj_model.csv`.
Census size and contract (agency) staff share carry most of it; the "bad-day" features (5th percentile, CV) help a bit.
Much of the average-HPRD signal was already in gold via Provider Info.

## Conclusion

- PBJ daily staffing is **consistent but very small** signal: +0.001–0.002 PR-AUC, same direction on CV and both test
  months, CI excludes 0 only on the 2024-11 ensemble.
- **Accuracy does not move** (75.7 % vs 75.8 % on 2025-02; majority 74.4 %). PBJ does not break the ~76 % ceiling.
- Same picture as exp7 (case-mix, +0.0017): staffing-type features are close to saturated. The ceiling is P(harm | survey);
  the next gains need different information (e.g. inspection/complaint text, complaint volume, ownership-group history).
- Recommendation: adopt only if the extra pipeline cost is acceptable (5.5 GB download, quarterly refresh); otherwise skip.
  If adopted, keep ~8 features (census, contract shares, p05/CV, HPRD) rather than all 24.

## Files

- `pbj_download.py`, `pbj_catalog.csv`, `pbj_download.log` — 22 quarters 2021Q1–2026Q2 → `data/incoming_pbj/`
- `pbj_features.py`, `pbj_features.log` → `outputs/pbj_quarterly.parquet`, `outputs/pbj_monthly.parquet`
- `exp12_pbj.py`, `exp12_pbj.log` → `outputs/exp12_pbj/` (`summary.json`, `preds_*.csv`, `gain_pbj_model.csv`)
