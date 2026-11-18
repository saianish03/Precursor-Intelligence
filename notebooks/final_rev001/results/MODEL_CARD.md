# Model card — Precursor Intelligence G+ citation risk model (local, 2026-11-06)

## 1. Purpose
Rank US nursing homes by the probability of a **serious CMS health citation (severity G–L: actual harm or immediate
jeopardy) within the next 12 months**, for liability insurers. Output: a calibrated risk score per facility per month,
used as a ranked watch-list (top 2 / 5 / 10 %), not a yes/no verdict.

## 2. Target
- `y_g12` = 1 if the facility receives a new G+ health citation in the 12 months after month t (configs/labels.yaml:
  gplus_severities G–L, horizon 12). Status `positive / negative`; `censored` and `immature` rows are excluded.
- Base rate: 26.2 % of labelled facility-months (131,371 positive / 370,731 negative).
- Side labels in gold (not modelled here): `y_g6` (6 months), `y_pen12` (penalty within 12 months).

## 3. Data
- Source: CMS Nursing Home Provider Data Catalog monthly archives 2021-01 → 2026-10 (68 zips).
- Pipeline: ingest → bronze (1,207 monthly files, 21 tables) → silver (29 tables) → gold
  (run `20261107T014704Z-c299c0`, branch `feat/all-tables`).
- Gold: **775,400 facility-months**, 15,452 facilities, 52 monthly snapshots (2022-01 → 2026-10), 219 columns.
  2021 is history only (12-month look-back). 7 CMS months are empty by design (vintage = Processing Date).
- Point-in-time: every feature for (facility, month t) uses only data published at t (Δ features also t-6 / t-12);
  labels use only events after the cutoff. Star ratings are never features (except per the registry's model_use).

## 4. Features
Recommended model: the **133-feature set** (the extra 69 from the new tables gave no honest lift).

| Family | # (model) | Source | Examples |
|---|---|---|---|
| survey_timing | 9 | Inspection Dates | days since last standard survey, days to due, complaint surveys 12/24 m |
| citations | 26 | Health Citations | citations 12/24 m, by severity band, severity points, G+ surveys 36 m, days since last G+ |
| enforcement | 7 | Penalties | fines 24 m, max fine, payment denials, days since last penalty |
| facility_staffing | 42 | Provider Info (t, t-6, t-12) | beds, HPRD by role, turnover, Δ staffing, ownership type, chain |
| quality | 43 | MDS + Claims QMs | pressure ulcers, falls, antipsychotics, rehospitalisation, Δ6 |
| context | 6 | derived within t | state, state G+ rate 12 m, state median HPRD, relative severity |
| *(new, not in recommended set)* | 69 | 7 new tables | fire_safety 13, survey_summary 13, ownership 11, snf_qrp 10, vbp 10, covid 3, state_context 9 |

Most important (LightGBM gain): state, complaint surveys with citations (24 m), G+ surveys (36 m), days since last G+,
certified beds, state G+ rate, citations 24 m, severity points 24 m, quality-of-life/care citations, complaint
citations 12 m, fines, staffing vs state.

## 5. Algorithm
- **Ensemble = mean of LightGBM + logistic regression** probabilities.
- LightGBM (Optuna-tuned, `feature engineering run/spine2022/boost/lgbm_best_params.json`): learning_rate 0.0102,
  num_leaves 34, max_depth 3, min_child_samples 870, subsample 0.63, colsample_bytree 0.33, reg_lambda 9.48,
  reg_alpha 0.0068, min_split_gain 0.435, n_estimators 1,265.
- Logistic regression: median imputation + missing indicators, standardisation, one-hot categoricals (min frequency 50),
  C = 0.3.
- Recommended training window: **the most recent 12 labelled months** (exp9: 2023 only beat all of 2022-01..2023-11).
- Threshold: chosen on the calibration/validation months; probabilities should be recalibrated on recent months.

## 6. Evaluation (all leak-free unless marked)

### Main forecast test (team split: train → 12-month purge → calib → test 2025-02, 14,643 facilities)
| Model | ROC-AUC | PR-AUC | Accuracy | P@2 % | P@5 % | P@10 % |
|---|---|---|---|---|---|---|
| Predict "no citation" | 0.500 | 0.256 | 74.4 % | 25.6 | 25.6 | 25.6 |
| Ensemble, trained 2022-01..08 | 0.700 | 0.465 | 75.6 % | 78.8 | 64.5 | 56.2 |
| Ensemble, trained 2022-01..2023-11 | 0.708 | 0.475 | 76.0 % | 79.9 | 66.5 | 57.2 |
| **Ensemble, trained 2023 only (recommended)** | **0.709** | **0.481** | **76.4 %** | **80.5** | **68.2** | **57.9** |

### Earlier report (test 2024-10/11, 29,324 facilities) — baselines
| Model | ROC-AUC | PR-AUC | P@5 % | P@10 % |
|---|---|---|---|---|
| CMS star rating | 0.628 | 0.341 | 41.2 | 38.4 |
| Severity rule (prior severe citation) | 0.661 | 0.411 | 57.6 | 50.3 |
| Ensemble (LGBM + CatBoost + LR) | 0.708 | 0.488 | 68.3 | 58.6 |

### 70 / 15 / 15 robustness splits (LightGBM, old 133)
| Split | ROC-AUC (95 % CI) | PR-AUC | Accuracy | P@2 % | ECE |
|---|---|---|---|---|---|
| random rows (**leaky**, not valid) | 0.926 | 0.824 | 87.3 % | 96.2 | — |
| by facility | 0.733 (0.720–0.747) | 0.503 | 76.4 % | 78.0 | 0.7 pts |
| by time (no purge) | 0.714 (0.706–0.721) | 0.488 | 74.5 % | 79.3 | 5.5 pts |
| by facility + time (strictest) | 0.704 (0.689–0.721) | 0.486 | 74.8 % | 82.0 | 2.2 pts |

Confusion matrix, facility split, validation-best-accuracy threshold (flags 12 %): TP 5,312 · FP 3,769 · TN 52,296 ·
FN 13,990 → precision 58.5 %, recall 27.5 %, specificity 93.3 %. At the F1-best threshold (flags ~45 %): recall 70 %,
precision 40 %, F1 0.51.

**Honest summary:** ROC-AUC ≈ 0.70–0.71, accuracy ≈ 75–76 % (vs 74 % for "no citation"), ≈ 80 % precision in the
top 2 % flagged (3× the base rate). 80 % overall accuracy is not reachable with this data (best possible threshold on
test ≈ 76 %).

## 7. Calibration
Well calibrated within a period (ECE 0.6–0.7 pts, facility split; earlier report ECE 2.2 pts, Brier 0.172). Drifts when
forecasting further ahead (time split ECE 5.5 pts; top decile predicted 0.71 vs actual 0.58) → recalibrate on the most
recent labelled months before publishing probabilities.

## 8. Limitations
- Partly unpredictable target: a citation needs an inspection; 15–22 % of negative rows had no health survey in the
  window (survey timing is part random).
- Performance decays with distance from training: ROC 0.733 → 0.695 over 7 months ahead.
- Current test window is one month (2025-02) because of CMS's empty months; numbers have ±0.7 pt (accuracy) noise.
- The 14 extra CMS tables add no lift; the remaining ceiling needs new information (inspection text, complaints,
  detailed staffing), not more CMS tables.
- State is the strongest feature: performance and calibration differ by state; check bias slices before release.

## 9. Operating recommendations
1. Retrain monthly (or at least quarterly) on the latest 12 labelled months; monitor ROC/PR-AUC per month.
2. Recalibrate (Platt/isotonic) on the latest labelled months.
3. Use as a ranked watch-list: top 2 % (~300 facilities, ~80 % hit rate), top 5 % (~730, ~67 %), top 10 % (~1,460, ~58 %).
4. Keep the new tables in silver for the product/agents (ownership, fire safety are useful context), not in the model.

## 10. Artifacts
- Saved models (previous gold, 75 selected features): `feature engineering run/spine2022/boost/models/`
  (lgbm_tuned.joblib, logreg.joblib, catboost.cbm, calibrator.joblib, ensemble_meta.json).
- This session's experiments (no model files saved; predictions saved):
  `Precursor-Intelligence/notebooks/experiments_rev002/exp8_all_tables.py, exp8b_refit_recent.py, exp9_recency.py,
  exp10_split_70_15_15.py, exp11_full_stats.py`; outputs in `outputs/exp8_all_tables/`, `outputs/exp11_full_stats/`.
- Full results: `ALL_TABLES_REPORT.md`. Pipeline contract: `Precursor-Intelligence/docs/ALL_TABLES_CONTRACT.md`.
