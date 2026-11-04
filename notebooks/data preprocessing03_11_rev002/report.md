# Feature Engineering & EDA: run report rev002 (full gold, spine from 2022)

**Pipeline:** the same steps as `Feature_Engineering_EDA.ipynb` (Parts A, B, C), rerun on the full monthly gold with the spine from 2022, followed by boosting, saved models and calibration
**Run date:** 2026-10-31 (gold rebuild, engineering, boosting) and 2026-11-03 (Part A, saved models, calibration) · Python 3.11 · pandas 3.0 · scikit-learn · LightGBM · CatBoost
**Label analysed:** `y_g12` (new G+ harm citation within 12 months)
**Outputs:** `feature engineering run/spine2022/`: `eda/reports/*.csv` (7 reports), `eda/figures/*.png` (7 figures), engineered and selected CSVs, `boost/` (model comparison), `boost/models/` (saved models, calibrator, test predictions)
**Compared with:** the quarterly **subset** run (`notebooks/data preprocessing31_10_rev001/report.md`)

---

## 0. Data used

The data is the **full local gold dataset** (`data/gold/facility_month_dataset`, run `20261031T200700Z-9312f1`, V = 2026-08), rebuilt with `spine_start = 2022-01`. It has **all 150 columns** and **50 monthly snapshots**.

| | Value |
|---|---|
| Rows / columns | 746,024 / 150 |
| Months | 2022-01 → 2026-08, monthly (50 snapshots) |
| Model features (registry `model_use: model`) | 133 (101 numeric, 32 categorical) |
| Labeled rows (positive/negative, not score) | 487,459 |
| DEV = train + calib (used for all fitting) | 296,476 rows: 2022-01 → 2023-06 (train, 252,271), 2023-07 → 2023-09 (calib, 44,205) |
| TEST (used only in Part C) | 29,324 rows: 2024-10, 2024-11 |
| Purge (labeled but never used for fitting) | 161,659 rows: 2023-10 → 2024-09 |
| Score / immature | 2025-02 → 2026-08 (250,392 rows, excluded) |

**Subset vs Gold 2022:**

| | Subset | Gold 2022 |
|---|---|---|
| Months | 15 quarterly snapshots | **50 monthly (2022-01 → 2026-08)** |
| Rows | 222,677 | **746,024** |
| DEV rows | 44,295 | **296,476** (6.7×) |
| DEV months | 2023-01, 04, 07 | 2022-01 → 2023-09 |
| TEST rows | 14,665 (2024-10) | 29,324 (2024-10, 11) |
| `lag3_diff_*` lags usable | no (quarterly gaps) | yes |

---

## Part A: Data analysis

### A1. Target: base rate, imbalance, label status
- **26.2 % positive overall.** The rate is moderately imbalanced and stable across splits: train 25.2 %, calib 27.5 %, purge 27.4 %, test 26.4 %.
- Label status by month follows the design. Months from 2025-04 onward are all `immature`, so they are correctly excluded.
- **Large geographic spread:** among states with ≥ 200 rows, the positive rate ranges from **6.9 % to 62.1 %**.
- Files: `A1_target.csv`, `A1_target_by_month.png`

| | Subset | Gold 2022 |
|---|---|---|
| Positive overall | 27.0 % | 26.2 % |
| Train / calib / purge / test | 25.9 / 27.3 / 27.5 / 26.5 % | 25.2 / 27.5 / 27.4 / 26.4 % |
| State range (≥ 200 rows) | 6.5 % → 62.9 % | 6.9 % → 62.1 % |

### A2. Univariate profile (distribution, zeros, outliers, missingness, cardinality)
- Profiled all 133 features on DEV (296,476 rows).
- **42 numeric features are heavily skewed** (|skew| > 2, > 10 distinct values; subset 41). The most skewed are `sev_points_per_bed_24m` (skew 78, max 326), `qm_ls_restraints` (41, 91 % zeros), `denial_days_24m` (9.8, 90 % zeros), `n_uncorrected` (9.1), `n_fines_24m` (7.2) and `max_fine_24m` (6.1, max $2.1 M).
- **Near-constant categoricals:** `sprinkler_status` (99.8 % YES), `avg_residents_per_day_fn` (99.7 % none), `qm_rating_fn` (98.3 %), and most `qm_*_fn` footnotes (~97 % none). `era` has only one level in DEV.
- Files: `A2_univariate_profile.csv`, `A2_skewed_histograms.png`, `A2_missingness.png`

### A3. Feature vs target: single-feature AUC, PR-AUC, mutual information, WOE/IV, binned rates
Each feature was scored alone against `y_g12` on DEV. Categoricals use an out-of-fold label-rate encoding (5 folds grouped by facility).

| Rank | Feature | AUC | PR-AUC | IV | MI |
|---|---|---|---|---|---|
| 1 | n_cit_24m | 0.667 | 0.409 | 0.360 | 0.041 |
| 2 | sev_points_24m | 0.671 | 0.407 | 0.373 | 0.044 |
| 3 | state | 0.662 | 0.404 | 0.365 | 0.040 |
| 4 | state_gplus_rate_12m | 0.651 | 0.402 | 0.313 | 0.041 |
| 5 | n_cat_qol_care_24m | 0.658 | 0.401 | 0.339 | 0.034 |
| 6 | n_cit_DF_24m | 0.661 | 0.400 | 0.333 | 0.033 |
| 7 | n_complaint_surveys_cited_24m | 0.665 | 0.396 | 0.365 | 0.038 |
| 8 | n_gplus_surveys_36m | 0.645 | 0.383 | 0.335 | 0.032 |

- **Main finding (same as the subset):** the strongest signal comes from **inspection history**, **complaints** and **state**. The top-8 list contains the same 8 features as the subset, in a slightly different order.
- **Quality measures are weak on their own.** The best is `qm_ls_depressive` (AUC 0.55, IV 0.053). **57 of 133 features have IV < 0.02** (subset: 56).
- Single-feature PR-AUC is about 0.01 lower than on the subset because DEV now includes 2022, which has a slightly lower positive rate.
- Files: `A3_feature_signal.csv`, `A3_binned_target_rate_top8.png`

### A4. Redundancy: Spearman correlation and VIF (top-40 numeric by PR-AUC)
- **15 pairs with |ρ| ≥ 0.85** (subset: 15), for example `denial_days_24m`↔`n_denials_24m` 0.998, `fine_dollars_24m`↔`max_fine_24m` 0.991, `n_cit_24m`↔`n_cit_DF_24m` 0.990, `n_complaint_cit_12m`↔`n_complaint_surveys_cited_12m` 0.967, `sev_points_12m`↔`n_cit_12m` 0.959.
- **VIF (top 25):** `n_cit_24m` 365, `n_cit_DF_24m` 262, `sev_points_24m` 53, `rel_sev_points_24m` 38. These are practically identical to the subset (354 / 252 / 56 / 39).
- Files: `A4_high_correlation_pairs.csv`, `A4_vif.csv`, `A4_correlation.png`

### A5. Temporal stability: PSI by month vs 2022-01
- **35 features have PSI > 0.25 (major shift) in at least one month** (subset: 22, with 2023-01 as the reference).
  - `era` (PSI 18.4): expected, since it is a regime flag.
  - **CMS methodology / reporting changes:** `qm_ls_incontinence` 5.4, `qm_ls_ed_per_1000` 1.3, and the staffing footnotes `turnover_rn_fn` 2.7, `admin_departures_fn` 2.6, `turnover_total_fn` 1.8.
  - **COVID-era inspection backlog (new with 2022):** `n_ic_surveys_12m` 5.2, `days_since_last_std` 1.5, `days_to_std_due` 1.4, `days_since_last_penalty` 1.5.
  - **Enforcement and state-level drift:** `state_gplus_rate_12m` 2.4, `state_median_hprd_total` 1.7, `max_fine_24m` 1.3, `fine_dollars_24m` 1.3, `d6_hprd_total` 1.1.
- **Implication:** adding 2022 brings in a different inspection regime (catch-up surveys after COVID). Older months are less like the test period, which argues for drift monitoring and a rolling or recency-weighted training window.
- Files: `A5_psi_by_month.csv`, `A5_psi_heatmap.png`

### A6. Coverage and rare levels
- **1 rare categorical level** (< 100 DEV rows): state GU (20 rows). The subset had 18. The larger DEV removes nearly all of the coverage problems.
- File: `A6_rare_levels.csv`

### A7. Slices (label rate by subgroup, all labeled rows)
| Slice | Subset | Gold 2022 |
|---|---|---|
| Bed size | < 50: 17.5 % · 50–99: 24.6 % · 100–149: 29.4 % · 150–199: 33.7 % · 200+: 34.3 % | < 50: 16.6 % · 50–99: 23.9 % · 100–149: 28.5 % · 150–199: 33.1 % · 200+: 34.0 % |
| Ownership | for-profit 28.6 % · government 25.3 % · non-profit 21.9 % | for-profit 28.0 % · government 23.6 % · non-profit 21.2 % |
| Chain | chain 28.8 % · independent 25.2 % | chain 28.8 % · independent 24.9 % |

Same gradients: risk rises with size, and for-profit and chain homes are higher. The full bias analysis belongs to the bias-slicing ticket.
- File: `A7_slices.csv`

### A8. Leakage scan
- **No feature has single-feature AUC > 0.80 or IV > 0.5**, so no leakage was detected. The strongest is IV 0.37.
- An assertion confirmed that **no label-like column** is registered as a model feature.

### A9. PCA and t-SNE (3,000-row DEV sample, standardized numeric features)
- Same picture as the subset: positives and negatives overlap heavily, with no clean separation in two dimensions. The signal is spread across many weak features.
- File: `A9_pca_tsne.png`

---

## Part B: Feature engineering (fitted on DEV only, applied to every month)

### Why engineer features: each step answers a problem found in Part A
The 133 pipeline features are raw facts (counts, rates, flags). Part A on Gold 2022 found the same problems as on the subset, and raw features handle them badly:

| Problem found in Part A (Gold 2022) | Engineering step that addresses it |
|---|---|
| 42 heavily skewed features with extreme outliers, e.g. `sev_points_per_bed_24m` skew 78, `max_fine_24m` up to $2.1 M (A2) | **B1:** clip to the 1st–99th percentile, log transform |
| Big homes get more citations just because they're bigger: the rate rises from 16.6 % (< 50 beds) to 34.0 % (200+) (A7) | **B1:** per-bed rates (citations per bed, complaints per 100 beds) |
| Size, ownership and survey timing affect risk in steps rather than smoothly (A7, A5 survey-timing drift) | **B2:** bed-size bands and survey-recency bands |
| `state` has 53 levels, ranges from 6.9 % to 62.1 % positive, is the #3 single feature and is a bias concern (A1, A3) | **B3:** group rare levels into "other"; replace raw state with a smoothed target encoding (`state_te`) |
| Single features are weak (best AUC 0.67), and positives and negatives overlap in PCA/t-SNE (A3, A9) | **B4:** crosses, e.g. many complaints *and* low RN staffing |
| State-level drift (`state_gplus_rate_12m` PSI 2.4, `state_median_hprd_total` 1.7): a facility should be judged against its peers, not against the whole country (A5) | **B5:** percentile rank within the same state and month |
| A worsening trend may matter more than the level; monthly data now gives each facility a usable history | **B6:** momentum (last 12 months vs the 12 before) and lags/volatility from earlier months |
| Missing values often mean "never happened", e.g. `days_since_last_gplus` 48 % non-null (never had a G+), `days_since_last_penalty` 70 %, `admin_departures` 73 %, `turnover_rn` 78 % (A2) | **B7:** `isna_*` flags |
| Quality measures are weak and correlated with each other; best is `qm_ls_depressive` IV 0.053, and 57 of 133 features have IV < 0.02 (A3) | **B8:** PCA to compress them into a 3-component index |
| 15 near-duplicate pairs (\|ρ\| ≥ 0.85) and VIF up to 365 in the citation family (A4) | **B9:** selection drops near-duplicates and features that don't help on held-out data |

### What was built
The same `FeatureTransformer` (`feature_transform_v2.py`) was refitted on the Gold 2022 DEV rows (**302,427 rows, 296,476 labeled, 2022-01 → 2023-09**) and applied to all 50 months. The steps are the same as in the subset report (B1 transforms, B2 bins, B3 categoricals + `state_te`, B4 crosses, B5 peer ranks, B6 trends, B7 missingness flags, B8 quality PCA).
- **No label leakage on DEV rows:** they get the out-of-fold `state_te` (5 folds grouped by facility).
- **Panel lags** (`lag3_diff_*`, `vol4_*`) use each facility's earlier months only. They are filled on monthly data, unlike the subset.
- Log: `spine2022/run.log` (about 4.5 min).

### B9. Feature selection
Same method as the subset: filter, then LightGBM permutation importance on 3 facility-grouped folds (80 % of DEV facilities; 20 % held out).

**252 candidates → 176 after filter → 75 selected.**

| Step | Subset | Gold 2022 |
|---|---|---|
| Candidates | 252 | 252 |
| Filter: near-constant | 10 | 10 |
| Filter: empty lags | 4 | 0 |
| Filter: near-duplicates (\|ρ\| ≥ 0.95) | ~65 | 66 |
| After filter | 173 | 176 |
| **Selected** | **66** (15 engineered + 51 original) | **75** (27 engineered + 48 original) |

- More engineered features survive on monthly data (27 vs 15) because the lag and volatility features now have values, for example `lag3_diff_turnover_total` and `vol4_n_complaint_surveys_cited_12m`.
- 5 of the 75 are categorical: `x_size_ownership`, `provider_type`, `qm_ss_new_antipsychotic_fn`, `resident_family_council`, `std_recency_band`.
- `state_te` is again the top feature.
- File: `spine2022/feature_set_selected75.yaml`

---

## Column count: before and after feature engineering

| Stage | Columns | What's in it |
|---|---|---|
| Gold dataset (input) | **150** | 133 model features + 9 label/split columns + 8 others |
| + engineered features | **+121** | what B1–B8 add |
| **Everything after engineering** | **271** | `facility_month_dataset_engineered_full_spine2022.csv`, 746,024 rows, 1.5 GB |
| Final selected feature set | **75** | 48 original + 27 engineered |
| Selected dataset | **86** | `ccn` + `t` + 9 label/split columns + 75 features: `facility_month_dataset_selected_spine2022.csv`, 0.5 GB |

Both CSVs are kept outside the repo because of their size.

---

## Part C: Validation

### C1. v1 vs selected (subset holdout: 2,964 facilities, 8,856 rows)
| Set | PR-AUC | P@5 % |
|---|---|---|
| **v1** (133 pipeline features) | **0.512** | **0.710** |
| Selected (66) | 0.509 | 0.701 |

v1 and the selected set are effectively tied, and the selected set uses half the features. Gold 2022 uses the **75-feature selected set** for all models below.

### C2. Test months 2024-10 and 2024-11 (reported only; 29,324 rows, base rate 26.4 %)
Gold 2022, 75 selected features, models fitted on all 296,476 DEV rows:

| Model | ROC-AUC | PR-AUC | P@5% | P@10% |
|---|---|---|---|---|
| Baseline: star rating | 0.628 | 0.341 | 0.412 | 0.384 |
| Baseline: severity rule | 0.661 | 0.411 | 0.576 | 0.503 |
| Logistic regression, 75 features | 0.703 | 0.482 | **0.684** | 0.583 |
| GBM default, 75 features | 0.705 | 0.480 | 0.666 | 0.576 |
| LightGBM, tuned with Optuna | **0.708** | 0.487 | 0.676 | 0.582 |
| CatBoost | 0.707 | 0.487 | 0.675 | 0.585 |
| **Ensemble (LGBM + CatBoost + LR)** | **0.708** | **0.488** | 0.683 | **0.586** |

Every model clearly beats both baselines. PR-AUC rises from 0.34–0.41 to 0.48–0.49.

### C3. Subset vs Gold 2022: one comparison table (test)

| Model | Subset PR-AUC | Gold 2022 PR-AUC | Subset ROC-AUC | Gold 2022 ROC-AUC | Subset P@5% | Gold 2022 P@5% | Subset P@10% | Gold 2022 P@10% |
|---|---|---|---|---|---|---|---|---|
| Baseline: star rating | 0.344 | 0.341 | 0.629 | 0.628 | 0.422 | 0.412 | 0.389 | 0.384 |
| Baseline: severity rule | 0.415 | 0.411 | 0.663 | 0.661 | 0.576 | 0.576 | 0.514 | 0.503 |
| Logistic regression, selected features* | **0.492** | 0.482 | **0.708** | 0.703 | **0.694** | **0.684** | **0.598** | 0.583 |
| GBM, selected features* | 0.484 | 0.480 | 0.703 | 0.705 | 0.679 | 0.666 | 0.582 | 0.576 |
| GBM, v1 (133 pipeline features) | 0.482 | — | 0.706 | — | 0.649 | — | 0.579 | — |
| LightGBM, tuned with Optuna | — | 0.487 | — | **0.708** | — | 0.676 | — | 0.582 |
| CatBoost | — | 0.487 | — | 0.707 | — | 0.675 | — | 0.585 |
| **Ensemble (LGBM + CatBoost + LR)** | — | **0.488** | — | **0.708** | — | 0.683 | — | **0.586** |

\* Selected features: 66 on the subset, 75 on Gold 2022. The subset tests on 2024-10 only (14,665 rows); Gold 2022 tests on 2024-10 and 2024-11 (29,324 rows).

**Reading:** Gold 2022 trains on 6.7× more rows than the subset (296,476 vs 44,295) but **does not raise test performance**: the best Gold 2022 model is 0.488, against 0.492 on the subset, which is within noise. Every model sits within about 0.01 PR-AUC of the others, so the features, not the model, set the ceiling.

### C4. Holdout (20 % of DEV facilities, 59,436 rows, base rate 26.3 %)
| Model | PR-AUC | ROC-AUC | P@5% | P@10% |
|---|---|---|---|---|
| GBM default | 0.514 | 0.732 | 0.725 | 0.643 |
| LightGBM tuned | 0.519 | 0.736 | **0.726** | 0.656 |
| CatBoost | 0.519 | 0.736 | 0.724 | 0.655 |
| Logistic regression | 0.510 | 0.732 | 0.722 | 0.641 |
| **Ensemble (LGBM + CatBoost + LR)** | **0.520** | **0.737** | 0.722 | **0.659** |

Optuna (3-fold facility-grouped CV) best PR-AUC 0.5112 vs 0.5087 for the default-like trial. The tuned LightGBM is shallow: depth 3, 34 leaves, min_child_samples 870, 1,265 trees, learning rate 0.010.

### C5. How precise the ensemble is when you flag the top-ranked homes (test, 26.4 % positive)
| Top flagged | Precision | Lift over random |
|---|---|---|
| 0.5% (146 homes) | **0.92** | **3.5×** |
| 1% (293 homes) | 0.86 | 3.2× |
| 5% (1,466 homes) | 0.68 | 2.6× |
| 10% (2,932 homes) | 0.59 | 2.2× |

Of the 146 homes the ensemble ranks highest, **134 (92 %) get a harm citation within 12 months**, against 26 % for a random pick and 41 % for the top 5 % by star rating.

### C6. Calibration (Gold 2022 ensemble)
The rank ensemble gives a ranking, not a probability. For probabilities, the ensemble score is the **mean of the three members' probabilities**, which doesn't depend on which other homes are scored in the same batch. Calibrators were fitted on **out-of-fold DEV predictions** (3 folds grouped by facility, all 296,476 rows) and checked on test with the saved full-DEV models.

| Scores (test) | Brier | Log loss | ECE | Mean predicted | PR-AUC | ROC-AUC | P@5% |
|---|---|---|---|---|---|---|---|
| Base rate only (no model) | 0.194 | — | — | 0.255 | — | — | — |
| **Raw ensemble mean** | **0.172** | **0.521** | **0.022** | 0.272 | **0.488** | 0.708 | **0.684** |
| Platt (logistic on logit) | 0.172 | 0.521 | 0.023 | 0.273 | 0.488 | 0.708 | 0.684 |
| Isotonic | 0.172 | 0.522 | 0.025 | 0.273 | 0.482 | 0.708 | 0.684 |
| Raw LightGBM alone | 0.173 | 0.522 | 0.027 | 0.279 | 0.487 | 0.708 | 0.675 |

(Observed test rate 0.264.) Reliability by decile of the raw ensemble mean:

| Decile | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Mean predicted | 0.07 | 0.11 | 0.14 | 0.17 | 0.21 | 0.25 | 0.30 | 0.37 | 0.46 | 0.65 |
| Observed | 0.08 | 0.12 | 0.14 | 0.21 | 0.21 | 0.24 | 0.29 | 0.35 | 0.41 | 0.59 |

- **The ensemble mean is already well calibrated:** expected calibration error (ECE) is 2.2 percentage points, and Brier improves 11 % over predicting the base rate.
- **Neither calibrator helps on test.** Platt is identical, and isotonic is slightly worse (its step function creates ties and lowers PR-AUC to 0.482).
- **The top two deciles are over-predicted by 5–6 points** (0.65 predicted vs 0.59 observed). The calibrators can't fix this because it doesn't appear in DEV. It is drift between 2022–23 and 2024, consistent with A5.
- **Decision:** report the raw ensemble mean as the probability. `calibrator.joblib` (Platt + isotonic) is saved for monthly recalibration once newer labeled months exist.
- Files: `boost/models/calibrator.joblib`, `calibration_test.csv`, `calibration_reliability.csv`, `calibration_reliability.png`, log `spine2022/calibrate.log` (8.7 min)

---

## Artifacts saved (Gold 2022)

All in `feature engineering run/spine2022/boost/models/`, fitted on all 296,476 DEV rows (log `spine2022/save_models.log`, 7.7 min):

| File | What it is |
|---|---|
| `lgbm_tuned.joblib` | tuned LightGBM, 1,265 trees |
| `catboost.cbm` | CatBoost, 300 iterations |
| `logreg.joblib` | logistic regression + its preprocessing (medians, scaler, dummy columns) |
| `ensemble_meta.json` | features, categorical levels, parameters, fit months, library versions |
| `calibrator.joblib` | isotonic + Platt calibrators for the ensemble mean |
| `test_predictions.csv`, `test_check.csv` | test scores and the reload check |

**Reload check:** the saved models give the same predictions as the in-memory models (max difference 0). The test scores match the comparison run: LightGBM 0.487 vs 0.487, CatBoost 0.485 vs 0.487, LR 0.482 vs 0.482, ensemble 0.487 vs 0.488.

Scoring new data: `spine2022/boost/score_ensemble.py <models_dir> <input_csv> <output_csv>`.

---

## Gold 2022: rev002 experiments (133 v1 features, 12-month gap before calib)

These run on the same Gold 2022 data with a stricter setup: train only 2022-01 → 06 (89,459 rows), so no train label window overlaps calib.

| Experiment | Model | Test PR-AUC | ROC-AUC | P@5% | P@10% |
|---|---|---|---|---|---|
| exp4 | **Logistic regression, one-hot** | **0.481** | 0.699 | **0.681** | **0.577** |
| exp3 | LightGBM tuned | 0.474 | **0.700** | 0.640 | 0.563 |
| exp5 | Logistic regression + 120 engineered | 0.474 | 0.700 | 0.657 | 0.565 |
| exp5 | LightGBM tuned + 120 engineered | 0.474 | 0.699 | 0.648 | 0.560 |
| exp4 | LightGBM, target encoding | 0.470 | 0.695 | 0.647 | 0.562 |
| exp1 | FT-Transformer | 0.464 | 0.694 | 0.623 | 0.545 |
| exp2 | MLP | 0.462 | 0.689 | 0.647 | 0.551 |
| exp3 | LightGBM default | 0.462 | 0.689 | 0.624 | 0.555 |
| exp1 | TabNet, pre-trained | 0.461 | 0.682 | 0.642 | 0.558 |
| exp2 | DeepFM | 0.452 | 0.678 | 0.626 | 0.544 |
| exp2 | DCN-v2 | 0.451 | 0.674 | 0.645 | 0.543 |
| exp1 | TabNet | 0.436 | 0.675 | 0.591 | 0.526 |

They score lower than C2 because they train on 6 months instead of 21. Deep models trail the classical ones by 0.02–0.03.

---

## Key takeaways and open points
1. **Signal lives in inspection and complaint history plus geography.** This holds on both datasets, with the same top-8 features. Quality measures add little on their own.
2. **More history does not raise the ceiling.** Test PR-AUC is about 0.48–0.49 on both the subset and Gold 2022. The top 0.5 % of homes are 92 % positive (3.5× random).
3. **Gold 2022 brings more drift:** 35 features have PSI > 0.25 (vs 22), led by COVID-era inspection-backlog features. Add them to drift monitoring and consider a rolling training window.
4. **Feature selection ties v1 with about half the features.** The 75-feature selected set is the one in use on Gold 2022.
5. **State dominates.** `state_te` is the top feature again. Raw `state` was removed for bias reasons, but its target encoding brings the same information back. The team needs to decide whether that is acceptable (bias-slicing ticket).
6. **Probabilities are usable as-is:** the ensemble mean has ECE 0.022 on test. It over-predicts the riskiest decile by about 6 points because of drift, so recalibrate on recent labeled months in production.
7. **Open:** run B9 directly on Gold 2022 if the team wants it (≈ 1.5–2 h; not done by decision). Decide the `state_te` question.
