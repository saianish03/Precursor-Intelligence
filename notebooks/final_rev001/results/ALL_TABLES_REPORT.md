# All CMS tables → gold → model (local run, 2026-11-06)

Branch `feat/all-tables` (local only, nothing pushed). Contract: `Precursor-Intelligence/docs/ALL_TABLES_CONTRACT.md`.

## 1. What was built

| Layer | Before | Now |
|---|---|---|
| Tables ingested per release | 7 | **21** (every data CSV; only `yf87-tv6k`, a list of state-agency contacts, is skipped) |
| Bronze monthly files | 441 | **1,207** (0 failures) |
| Silver tables | 15 | **29** (14 new `*_by_vintage` tables) |
| Gold feature families | 5 on the old branch (quality missing) | **13** (+ quality merged, + 7 new) |
| Gold columns | 133 model features | **202 model features** (219 columns in the merged dataset) |
| Gold rows | — | 775,400 facility-months, 52 months (2022-01 → 2026-10) |

Not every table exists every month (CMS changed what it publishes):

| Table | Monthly files | Note |
|---|---|---|
| provider_info, citations, surveys, penalties, MDS/claims QM, citation lookup, ownership, survey summary, state averages, cut points, SNF QRP | 63 of 70 months | the 7 empty months (2021-12, 2022-12, 2023-12, 2024-12, 2025-01, 2025-08, 2026-01) |
| fire_citations | 62 | 2026-08 release had no fire-safety file |
| snf_qrp_national, swing_bed | 53 | not in every release |
| vbp_facility / vbp_aggregate | 64 zips | once a year, fiscal-year file repeated monthly |
| data_collection_intervals | 59 | from 2021-05 |
| covid_vax (provider/state) | 46 | CMS stopped after 2025-07 |
| qm_complaint_periods | 4 | only 2021-01..04 |

New gold families (all point-in-time: vintage t only, Δ at t-12; star ratings never used):
fire_safety (13), survey_summary (13), ownership (11), snf_qrp (10), vbp (10), covid (3), state_context (9).

Fixes made on the way: Windows MAX_PATH in the local store's temp files (long SNF QRP names); inventory header
decoding now matches bronze (2021 VBP files are cp1252); 5-digit CCNs in Swing Bed / VBP 2023 padded back;
fire-safety month missing → features null (unknown), not 0.

## 2. Model results (label y_g12: serious G+ citation within 12 months)

Splits from `configs/splits.yaml` on the new gold: train 2022-01..2022-08 (after the 12-month purge gap before
calib), calib 2023-09..2023-11, **test = one month, 2025-02 (14,643 facilities, base rate 25.6 %)**.
The neighbouring months are CMS's empty months, so the test window is a single month — numbers are noisier
than the earlier 2-month test.

### Does each new table add signal? (LightGBM, calib PR-AUC, old 133 features = 0.5045)

| Added family | Δ calib PR-AUC |
|---|---|
| survey_summary | +0.0023 |
| state_context | +0.0011 |
| vbp | +0.0010 |
| ownership | +0.0006 |
| fire_safety | +0.0004 |
| covid | 0.0000 |
| snf_qrp | −0.0002 |
| **all 202 features** | **+0.0021** |

Feature selection (gain ranking on train, k chosen on calib): **100 features** (39 from the new tables).

### Test (scored once). Final models refit on all labelled rows up to 2023-11 (325,920 rows)

| Model | Features | PR-AUC | ROC-AUC | P@10 % | Accuracy* |
|---|---|---|---|---|---|
| Predict "no citation" for all | — | 0.256 | 0.500 | — | **74.4 %** |
| LightGBM | old 133 | 0.4745 | 0.708 | 0.573 | 75.8 % |
| Logistic regression | old 133 | 0.4716 | 0.705 | 0.571 | 75.9 % |
| **Ensemble (LGBM + LR)** | **old 133** | **0.4754** | **0.708** | 0.572 | **76.0 %** |
| LightGBM | selected 100 | 0.4715 | 0.708 | 0.567 | 75.7 % |
| Logistic regression | selected 100 | 0.4694 | 0.703 | 0.576 | 76.0 % |
| Ensemble | selected 100 | 0.4733 | 0.707 | 0.574 | 75.9 % |

### Precision among the top-flagged facilities (test 2025-02, 14,643 facilities, base rate 25.6 %)

| Model | P@2 % (293 fac.) | P@5 % (732) | P@10 % (1,464) |
|---|---|---|---|
| **Old 133, ensemble** | **79.9 % (3.12×)** | **66.5 % (2.60×)** | 57.2 % (2.23×) |
| Old 133, LightGBM | 77.5 % (3.03×) | 65.6 % (2.56×) | 57.3 % (2.24×) |
| Old 133, LogReg | 79.5 % (3.11×) | 65.7 % (2.57×) | 57.1 % (2.23×) |
| Selected 100, ensemble | 79.5 % (3.11×) | 65.8 % (2.57×) | 57.4 % (2.24×) |
| Selected 100, LightGBM | 79.2 % (3.09×) | 65.6 % (2.56×) | 56.7 % (2.21×) |
| Selected 100, LogReg | 77.5 % (3.03×) | 64.9 % (2.53×) | **57.6 % (2.25×)** |

(lift = precision / base rate)

\* threshold chosen on calib. The best accuracy any threshold could reach on test (an upper bound, picked on
test) is 76.1 %.

Train-only fit (exp8, before refit): old ensemble 0.4651, selected ensemble 0.4613; facility-clustered 95 % CI of
selected − old is −0.0066..−0.0009 (LGBM −0.0056..−0.0004).

### Recency vs row count (exp9; old 133 features, LGBM + LR ensemble, test 2025-02, fixed calib threshold)

| Training window | Rows | PR-AUC | ROC | Accuracy | P@2 % | P@5 % | P@10 % | ΔPR-AUC vs R1 (95 % CI) |
|---|---|---|---|---|---|---|---|---|
| R1 2022-01..08 (original) | 119,165 | 0.4651 | 0.700 | 75.6 % | 78.8 | 64.5 | 56.2 | — |
| R2 2022-01..2023-11 (all) | 325,920 | 0.4754 | 0.708 | 76.0 % | 79.9 | 66.5 | 57.2 | +0.007..+0.013 |
| R3 2023 only | 162,276 | 0.4802 | 0.709 | 76.4 % | 80.2 | 67.8 | 57.5 | +0.010..+0.021 |
| **R4 2023 only, same size as R1** | 119,165 | **0.4813** | **0.709** | **76.4 %** | **80.5** | **68.2** | **57.9** | **+0.011..+0.022** |
| R5 2022 only, same size as R3 | 162,276 | 0.4660 | 0.702 | 75.7 % | 78.5 | 64.6 | 56.0 | −0.000..+0.002 |

Same size, newer months (R4 vs R1): +0.016 PR-AUC. More rows, same period (R5 vs R1): nothing. Adding the old
2022 rows to 2023 (R2 vs R3) makes it slightly worse. **The gain is recency, not row count.**

### 70 / 15 / 15 splits on all 502,102 labelled rows (exp10 / exp11, LightGBM, validation = early stop + thresholds)

| Split | Features | Test rows | ROC-AUC (95 % CI) | PR-AUC | Acc. | Majority | Bal. acc | F1 (best) | P@1 / 2 / 5 / 10 / 20 / 30 % | Brier (base) | ECE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| random rows (**leaky**) | old 133 | 75,316 | 0.926 | 0.824 | 87.3 % | 73.8 % | — | 0.757 | — / 96.2 / 94.5 / 91.2 / — / — | — | — |
| facility | old 133 | 75,367 | 0.733 (0.720–0.747) | 0.503 | 76.4 % | 74.4 % | 0.604 | 0.511 | 80.5 / 78.0 / 70.5 / 61.3 / 52.0 / 46.3 | 0.164 (0.191) | 0.007 |
| facility | all 202 | 75,367 | 0.732 (0.720–0.743) | 0.502 | 76.7 % | 74.4 % | 0.589 | 0.511 | 80.4 / 78.5 / 68.8 / 61.3 / 52.2 / 46.4 | 0.164 (0.191) | 0.006 |
| time (no purge gap) | old 133 | 73,309 | 0.714 (0.706–0.721) | 0.488 | 74.5 % | 73.6 % | 0.613 | 0.486 | 87.2 / 79.3 / 66.0 / 58.3 / 50.3 / 45.1 | 0.175 (0.194) | 0.055 |
| time (no purge gap) | all 202 | 73,309 | 0.727 (0.721–0.736) | 0.505 | 74.8 % | 73.6 % | 0.626 | 0.498 | 86.5 / 79.7 / 68.6 / 60.9 / 52.1 / 46.7 | 0.171 (0.194) | 0.047 |
| facility + time (strictest) | old 133 | 21,934 | 0.704 (0.689–0.721) | 0.486 | 74.8 % | 73.3 % | 0.593 | 0.491 | 88.6 / 82.0 / 66.1 / 58.2 / 49.7 / 45.1 | 0.174 (0.196) | 0.022 |
| facility + time (strictest) | all 202 | 21,934 | 0.706 (0.692–0.723) | 0.485 | 74.8 % | 73.3 % | 0.591 | 0.494 | 87.2 / 80.9 / 64.9 / 58.5 / 50.2 / 45.0 | 0.174 (0.196) | 0.026 |

Paired difference all 202 − old 133 (same test rows, facility-clustered bootstrap, 300 resamples):

| Split | ΔROC-AUC (95 % CI) | ΔPR-AUC (95 % CI) |
|---|---|---|
| facility | −0.001 (−0.004..+0.003) | −0.001 (−0.005..+0.003) |
| time (no purge gap) | **+0.014 (+0.011..+0.017)** | **+0.018 (+0.013..+0.023)** |
| facility + time | +0.002 (−0.001..+0.005) | −0.001 (−0.005..+0.004) |

**Why the time-split gain is not trusted:** this split has no purge gap, so the 12-month labels of the last training
months (to 2024-02) are the same citations as the validation/test labels. Signs of that leak: validation ROC 0.82–0.85
vs test 0.71–0.73, and early stopping never stopped (3,000 trees). The new tables add facility-identifying features
(e.g. own_months_since_oldest_assoc, vbp_snfrm_performance_rate, days_c1_to_c2_health_survey — all in the new top-10),
which let the model recognise facilities whose label events it already saw. When the test facilities are new
(facility + time) the gain disappears (+0.002, CI includes 0), and with the proper purge gap (exp8/exp9) it was zero.

Confusion matrices (threshold that maximised accuracy on validation):

| Split / features | TP | FP | TN | FN | Precision | Recall | Specificity | Flagged |
|---|---|---|---|---|---|---|---|---|
| facility / old 133 | 5,312 | 3,769 | 52,296 | 13,990 | 58.5 % | 27.5 % | 93.3 % | 12.0 % |
| time / old 133 | 6,458 | 5,816 | 48,129 | 12,906 | 52.6 % | 33.4 % | 89.2 % | 16.7 % |
| facility + time / old 133 | 1,523 | 1,199 | 14,881 | 4,331 | 56.0 % | 26.0 % | 92.5 % | 12.4 % |

At the F1-best threshold (flag ~30–46 %): recall 54–70 %, precision 39–46 %, F1 0.49–0.51.
Per test month (time split, old 133): ROC 0.733 (2024-08) → 0.722 → 0.713 → 0.704 → 0.695 (2025-02): performance decays
as the forecast gets further from the training data — an argument for frequent retraining (see exp9).
Calibration: facility split ECE 0.6–0.7 pts (well calibrated); time split 4.7–5.5 pts (top decile predicted 0.71, actual
0.58) → recalibrate on recent months before showing probabilities.

## 3. Conclusions

1. **The 14 extra tables give no real lift.** +0.002 on calib, slightly negative on test. They mostly re-describe
   what the 7 core tables already say (survey summary = citations; state averages ≈ state context; fire safety
   correlates with health citations). Ownership, VBP and QRP carry a little signal but not enough.
2. **80 % accuracy is not reachable with this data.** The base rate already gives 74.4 %; the best model gives
   76.0 %, and no threshold on test does better than 76.1 %. The ranking metrics (PR-AUC 0.475, ROC 0.71, top-10 %
   precision 57 % vs 26 % base) are the meaningful ones for an insurer, and they are where they were before.
3. **Training on more recent rows helps more than any feature** (+0.010 PR-AUC from refitting up to 2023-11).
4. Recommendation: keep the 133-feature model (simpler, equal or better); keep the new tables in silver for the
   product/agents (ownership, fire safety are useful to show), and do not add them to the model. Raise with the
   team: the test window is now one month because of CMS's empty months.

Files: `notebooks/experiments_rev002/exp8_all_tables.py`, `exp8b_refit_recent.py`, outputs in
`notebooks/experiments_rev002/outputs/exp8_all_tables/` (summary.json, summary_refit.json, selected_features.txt,
gain_*.csv, test_predictions_refit.csv).
