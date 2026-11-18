# Feature Engineering & EDA: run report

**Notebook:** `Precursor-Intelligence/notebooks/Feature_Engineering_EDA.ipynb` (all 26 cells ran without errors, exit 0)
**Run date:** 2026-10-31 · Python 3.11.16 · pandas 3.0.6 · scikit-learn 1.9.1 · LightGBM
**Label analysed:** `y_g12` (new G+ harm citation within 12 months)
**Outputs in this folder:** `outputs/reports/*.csv` (13 reports), `outputs/figures/*.png` (9 figures), `outputs/artifacts/` (fitted transformer, `feature_set_v2.yaml`), and the executed notebook.

---

## 0. Data used

The data is a **subset of the local gold dataset** (`data/gold/facility_month_dataset`, run `20261028T203532Z-0abcdf`), with **all 150 columns** and 15 quarterly snapshot months. These are the same months as `facility_month_dataset_subset.csv`, except 2026-09, which is not in local gold.

| | Value |
|---|---|
| Rows / columns | 222,677 / 150 |
| Facilities | 15,272 |
| Months | 2023-01, 04, 07, 10 · 2024-01, 04, 07, 10 · 2025-02, 04, 07, 10, 12 · 2026-04, 08 |
| Model features (registry `model_use: model`) | 133 (101 numeric, 32 categorical) |
| Labeled rows (positive/negative, not score) | 117,782 |
| DEV = train + calib (used for all fitting) | 44,295 rows: 2023-01, 2023-04 (train), 2023-07 (calib) |
| TEST (used only in Part C) | 14,665 rows: 2024-10 |
| Purge (labeled but never used for fitting) | 58,822 rows: 2023-10 → 2024-07 |
| Score / immature | 2025-02 → 2026-08 (excluded) |

> **Caveat on the subset.** Because the snapshots are quarterly, DEV has only 3 months per facility. The "3 snapshots earlier" lags (`lag3_diff_*`) are therefore empty in DEV and were dropped by the filter, and the 4-snapshot volatilities (`vol4_*`) cover about one year instead of four months. Re-run on the full monthly gold for final numbers.

---

## Part A: Data analysis

### A1. Target: base rate, imbalance, label status
- **27.0 % positive overall.** The rate is moderately imbalanced and stable across splits: train 25.9 %, calib 27.3 %, purge 27.5 %, test 26.5 %.
- Label status by month follows the design. Months from 2025-04 onward are all `immature` (outcome window not closed yet), so they are correctly excluded.
- **Large geographic spread:** among states with ≥ 200 rows, the positive rate ranges from **6.5 % to 62.9 %**. State is the single biggest source of variation in the label (see A3 and B9).
- Files: `A1_target.csv`, `A1_target_by_month.png`

### A2. Univariate profile (distribution, zeros, outliers, missingness, cardinality)
- Profiled all 133 features on DEV: non-null share, zero share, mean/sd, skew, p01/p50/p99, max, and cardinality. Categoricals were profiled by top level and top share.
- **41 numeric features are heavily skewed** (|skew| > 2, > 10 distinct values). These are almost all count and dollar features: citation counts, severity points, fines, denial days and complaint counts. The most skewed are `qm_ls_restraints` (skew 43, 91 % zeros), `denial_days_24m` (10.4, 89 % zeros), `n_uncorrected` (8.3) and `max_fine_24m` (6.8, max $2.1 M).
- **Lowest coverage:** `days_since_last_gplus` 49 % non-null (missing means "never had a G+"), `admin_departures` 71 %, `days_since_last_penalty` 73 %, `d12_turnover_total` 75 %, `days_since_last_complaint_survey_cited` 77 %, `turnover_rn` 78 %. Most of these gaps carry meaning ("event never happened"), so missingness indicators were engineered in B7.
- **Near-constant categoricals:** `sprinkler_status` (99.8 % YES), `avg_residents_per_day_fn` (99.7 % none), `qm_rating_fn` (98.3 %), and most `qm_*_fn` footnotes (~97 % none).
- Files: `A2_univariate_profile.csv`, `A2_skewed_histograms.png`, `A2_missingness.png`

### A3. Feature vs target: single-feature AUC, PR-AUC, mutual information, WOE/IV, binned rates
Each feature was scored alone against `y_g12` on DEV. Categoricals use an out-of-fold label-rate encoding (5 folds grouped by facility), so a level never scores its own rows.

| Rank | Feature | AUC | PR-AUC | IV | MI |
|---|---|---|---|---|---|
| 1 | n_cit_24m | 0.671 | 0.423 | 0.381 | 0.038 |
| 2 | state | 0.669 | 0.420 | 0.385 | 0.042 |
| 3 | sev_points_24m | 0.674 | 0.419 | 0.383 | 0.037 |
| 4 | n_cat_qol_care_24m | 0.661 | 0.415 | 0.357 | 0.038 |
| 5 | n_cit_DF_24m | 0.665 | 0.413 | 0.351 | 0.035 |
| 6 | state_gplus_rate_12m | 0.654 | 0.410 | 0.329 | 0.046 |
| 7 | n_complaint_surveys_cited_24m | 0.668 | 0.408 | 0.387 | 0.034 |
| 8 | n_gplus_surveys_36m | 0.647 | 0.395 | 0.361 | 0.035 |

- **Main finding:** the strongest signal comes from **inspection history** (24-month citation counts and severity), **complaints**, and **state**. All of the top-20 features have a positive direction: more past problems mean more future harm.
- **Quality measures are weak on their own.** The best is `qm_ls_depressive` (AUC 0.54, IV 0.046). **56 of 133 features have IV < 0.02** ("not useful" by the WOE rule).
- Decile plots of the top 8 numeric features show a monotone increase in the positive rate.
- Files: `A3_feature_signal.csv`, `A3_binned_target_rate_top8.png`

### A4. Redundancy: Spearman correlation and VIF (top-40 numeric by PR-AUC)
- **15 pairs with |ρ| ≥ 0.85**, for example `denial_days_24m`↔`n_denials_24m` 0.998, `fine_dollars_24m`↔`max_fine_24m` 0.992, `n_cit_24m`↔`n_cit_DF_24m` 0.991, `n_complaint_cit_12m`↔`n_complaint_surveys_cited_12m` 0.963, `hprd_total`↔`hprd_weekend_total` 0.956.
- **VIF (top 25):** `n_cit_24m` 354, `n_cit_DF_24m` 252, `sev_points_24m` 56, `rel_sev_points_24m` 39. The citation family is strongly collinear. This matters for the logistic regression and not for tree models. The near-duplicates were removed in B9.
- Files: `A4_high_correlation_pairs.csv`, `A4_vif.csv`, `A4_correlation.png`

### A5. Temporal stability: PSI by month vs 2023-01
- **22 features have PSI > 0.25 (major shift) in at least one month.**
  - `era` (PSI 18.4): expected, since it is a regime flag by design.
  - **CMS methodology / reporting changes:** `qm_ls_incontinence` 5.3, `qm_ls_pressure_ulcer` 0.90, `qm_ls_ed_per_1000` 0.71, and the staffing footnotes `admin_departures_fn` 3.4, `turnover_rn_fn` 2.8, `turnover_total_fn` 1.8, `staffing_rating_fn` 0.91.
  - **Real drift in enforcement and inspection activity:** `max_fine_24m` 1.6, `fine_dollars_24m` 1.3, `n_ic_surveys_12m` 2.4, `n_std_36m` 0.62, `n_fines_24m` 0.56, and the state-level aggregates `state_median_hprd_total` 1.7 and `state_gplus_rate_12m` 1.3.
- **Implication:** these features need drift monitoring in production, and models fitted on 2023 may decay. This supports the planned retraining and regime flags.
- Files: `A5_psi_by_month.csv`, `A5_psi_heatmap.png`

### A6. Coverage and rare levels
- **18 rare categorical levels** (< 100 DEV rows): states AK (60), DC (51), PR (18), GU (3), `sprinkler_status = PARTIAL` (88), and `missing_not_submitted` (48 rows each) on the `qm_*_fn` footnotes. These are grouped into `other` in B3.
- File: `A6_rare_levels.csv`

### A7. Slices (label rate by subgroup, all labeled rows)
| Slice | Values (rate) |
|---|---|
| Bed size | < 50: 17.5 % · 50–99: 24.6 % · 100–149: 29.4 % · 150–199: 33.7 % · 200+: 34.3 % |
| Ownership | for-profit 28.6 % · government 25.3 % · non-profit 21.9 % |
| Chain | chain 28.8 % · independent 25.2 % |

The risk rises steadily with size: the largest homes have about twice the rate of the smallest. For-profit and chain homes are higher. The full bias analysis (FDR-controlled) belongs to the separate bias-slicing ticket.
- File: `A7_slices.csv`

### A8. Leakage scan
- **No feature has single-feature AUC > 0.80 or IV > 0.5**, so no leakage was detected. The strongest is IV 0.39.
- An assertion confirmed that **no label-like column** (`y_*`, `status_*`, `split_*`, `*_fwd`) is registered as a model feature.

### A9. PCA and t-SNE (3,000-row DEV sample, standardized numeric features)
- PCA cumulative explained variance, PC1 vs PC2 and t-SNE scatter, colored by the label. Positives and negatives overlap heavily, with no clean separation in two dimensions. This fits the moderate single-feature AUCs (≤ 0.67): the signal is spread across many weak features.
- File: `A9_pca_tsne.png`

---

## Part B: Feature engineering (fitted on DEV only, applied to every month)

### Why engineer features: each step answers a problem found in Part A
The 133 pipeline features are raw facts (counts, rates, flags). Part A found problems that raw features handle badly:

| Problem found in Part A | Engineering step that addresses it |
|---|---|
| 41 heavily skewed features with extreme outliers (A2) | **B1:** clip to the 1st–99th percentile, log transform |
| Big homes get more citations just because they're bigger (A7 size gradient) | **B1:** per-bed rates (citations per bed, complaints per 100 beds) |
| Size, ownership and survey timing affect risk in steps rather than smoothly | **B2:** bed-size bands and survey-recency bands |
| `state` has 53 levels, 4 of them rare (A6), and is a bias concern | **B3:** group rare levels into "other"; replace raw state with a smoothed target encoding (`state_te`) |
| Risk may come from combinations, e.g. many complaints *and* low RN staffing | **B4:** crosses |
| A facility should be judged against its peers, not against the whole country | **B5:** percentile rank within the same state and month |
| A worsening trend may matter more than the level | **B6:** momentum (last 12 months vs the 12 before) and lags/volatility from earlier months |
| Missing values often mean "never happened" (A2) | **B7:** `isna_*` flags |
| 18 weak quality measures that are correlated with each other (A3) | **B8:** PCA to compress them into a 3-component index |

### What was built
One `FeatureTransformer` (`feature_transform_v2.py` + `.joblib`) learns every parameter on DEV and transforms all months, which keeps training and serving consistent. It produced **120 engineered features**:

| Step | What was built |
|---|---|
| B1 Transforms | 1–99 % clipping of all skewed and count features (`*_clip`); `log_*` for 11 count/dollar features; per-bed rates (`cit_per_bed_24m`, `complaints_per_100beds_12m`, `fine_dollars_per_bed_24m`) |
| B2 Bins | `bed_band` (5 size bands); `std_recency_band` (time since the last standard survey, around the 15-month due date) |
| B3 Categoricals | rare levels (< 30 rows) grouped into `other` (`*_v`); **`state_te`** = smoothed target encoding of state (out-of-fold on DEV). Raw `state` was removed from v2 as bias mitigation |
| B4 Crosses | complaints × low-RN-staffing rank, harm-at-last-survey × overdue, size × ownership, chain × turnover |
| B5 Peer ranks | percentile within state-month for staffing, turnover, citations, complaints, falls |
| B6 Trends | citation and complaint momentum (last 12 m vs prior 12 m); `lag3_diff_*` and `vol4_*` per facility from earlier snapshots only |
| B7 Missingness | `isna_*` flags for staffing, turnover, days-since-penalty / G+, falls |
| B8 PCA | 3-component quality index from all `qm_ls_*` / `qm_ss_*` measures |

### B9. Feature selection (on 80 % of DEV facilities; 20 % held out for the decision)
**252 candidates → 173 after filter → 66 selected.**

1. **Filter (79 dropped):**
   - 10 near-constant features.
   - 4 all-missing lag features (`lag3_diff_*`, a result of using quarterly snapshots).
   - About 65 near-duplicates (|ρ| ≥ 0.95). The feature with the lower single-feature PR-AUC was dropped. This was usually a raw/clip/log pair, e.g. the raw versions of `sev_points_12m`, `n_cit_24m` and the fine features.
2. **Embedded + permutation (107 dropped):** LightGBM on 3 facility-grouped folds. A feature was kept only if its permutation importance (Δ PR-AUC) was positive in ≥ 2 of the 3 folds.
3. **Kept: 66 features.** 15 are engineered and 51 are original pipeline features.

**Top permutation importances (Δ PR-AUC, mean of 3 folds; all positive in 3/3 folds):**

| Feature | Δ PR-AUC |
|---|---|
| **state_te** | **0.070** |
| certified_beds_clip | 0.011 |
| state_gplus_rate_12m | 0.010 |
| n_complaint_surveys_cited_24m | 0.009 |
| n_cat_qol_care_24m | 0.004 |
| n_cit_24m | 0.003 |
| days_since_last_complaint_survey_cited | 0.003 |
| days_since_last_std | 0.003 |

**Engineered features that survived:** `state_te`, `certified_beds_clip`, `pct_state_hprd_rn`, `pct_state_turnover_total`, `pct_state_n_cit_24m`, `x_size_ownership`, `x_complaints_x_low_rn`, `complaint_momentum_12m`, `denial_days_24m_clip`, `n_fines_24m_clip`, `qm_ls_catheter_clip`, `qm_ls_restraints_clip`, `vol4_hprd_total`, `vol4_n_complaint_surveys_cited_12m`, `isna_days_since_last_gplus`.

- Files: `B9_feature_decisions_log.csv` (one row per feature with the reason), `B9_importance_stability.csv`, `B9_permutation_importance.png`

---

## Column count: before and after feature engineering

| Stage | Columns | What's in it |
|---|---|---|
| Gold dataset (input) | **150** | 133 model features + 9 label/split columns + 8 others (IDs such as `ccn` and `t`, `cutoff_date`, non-model registry columns) |
| + engineered features | **+120** | what B1–B8 add |
| **Everything after engineering** | **270** | 150 original + 120 new |
| Candidates for selection | 252 | 132 original model features (raw `state` dropped) + 120 engineered |
| After filtering | 173 | near-constant, empty and duplicate features removed |
| **Final selected feature set (v2)** | **66** | 51 original + 15 engineered |
| v2 dataset, if saved | **77** | `ccn` + `t` + 9 label/split columns + 66 features |

The v2 dataset was **not written** in this run (`SAVE_DATASET = False`); only the transformer was saved, and it can produce those columns on demand. The decision **kept v1** (133 features), so the model-ready dataset is still the original 150-column gold until the full monthly run confirms or reverses it.

---

## Part C: Validation

### C1. v1 vs v2 decision (DEV holdout: 2,964 facilities, 8,856 rows, 26.4 % positive; never seen by B9)
| Set | PR-AUC | Precision@5 % |
|---|---|---|
| **v1** (133 pipeline features) | **0.512** | **0.710** |
| v2 (66 selected) | 0.509 | 0.701 |

**Decision: keep v1.** v2 did not beat v1 on both metrics. The two are effectively tied, and v2 does it with half the features.

### C2. Test month 2024-10 (reported only; does not change the decision)
| Model | ROC-AUC | PR-AUC | P@5 % | P@10 % |
|---|---|---|---|---|
| Baseline: overall star rating | 0.629 | 0.344 | 0.422 | 0.389 |
| Baseline: severity rule | 0.663 | 0.415 | 0.576 | 0.514 |
| Logistic regression · v2 | **0.708** | **0.492** | **0.694** | **0.598** |
| GBM · v1 | 0.706 | 0.482 | 0.649 | 0.579 |
| GBM · v2 | 0.703 | 0.484 | 0.679 | 0.582 |

(Base rate 26.5 %.) Every ML model clearly beats both baselines. PR-AUC rises from 0.34–0.42 to 0.48–0.49, and in the top 5 % flagged homes, 65–69 % have a harm citation within 12 months versus 42 % for the star rating. These use default settings and are not tuned or calibrated.

### Artifacts saved
`feature_transform_v2.joblib` + `.py`, `feature_set_v2.yaml` (66 features, decision record). The engineered dataset was not written (`SAVE_DATASET = False`).

---

## Full-gold engineered dataset (CSV)

After the notebook run, the same Part B feature engineering was applied to the **full local gold dataset** and written to one CSV.

**File:** `MLOPS project/feature engineering run/facility_month_dataset_engineered_full.csv` (1.09 GB). It is kept outside the repo on purpose, because a file that size should not go into git.

| Check | Result |
|---|---|
| Rows | **578,975**, all 39 months of gold (2023-01 → 2026-08) |
| Columns | **270**: 150 original gold + 120 engineered, no duplicates |
| `ccn` | kept as 6-character text, leading zeros intact |
| Engineered columns | none entirely empty, none more than 50 % empty (the `lag3_*` features that were empty on the quarterly subset now have values) |

**How it was built** (script `build_full_csv.py`, run time about 4.5 minutes, month by month to fit the 8 GB laptop):
1. **Same `FeatureTransformer` as the notebook**, fitted only on the full-gold development rows: g12 train + calib, labeled, **132,832 rows, 2023-01 → 2023-09**.
2. **No label leakage on development rows:** they get the out-of-fold `state_te` (5 folds grouped by facility), so a home never sees its own label.
3. **Panel lags computed over the whole panel:** `lag3_diff_*` and `vol4_*` use each facility's earlier months only. The math is the same as the notebook's, vectorised.
4. **Each month is transformed and appended** with the frozen parameters, in the notebook's column order.

**Things to know before using it:**
- **The transformer was refitted on full gold.** Its clip points, encodings and PCA come from 9 development months, so they differ slightly from the subset run's saved `outputs/artifacts/feature_transform_v2.joblib`.
- **The CSV has all 270 columns.** The 66-feature selection (B9) and the "keep v1" decision (Part C) came from the quarterly subset; they have **not** been rerun on full gold yet. That rerun needs about 2.5 GB of free RAM.

---

## Key takeaways and open points
1. **Signal lives in inspection and complaint history plus geography.** Quality measures add little on their own.
2. **State dominates.** `state_te` has 7× the importance of the next feature. Raw `state` was removed for bias reasons, but its target encoding brings the same information back. The team needs to decide whether that is acceptable (bias-slicing ticket).
3. **Drift is real:** 22 features have PSI > 0.25, many caused by CMS methodology changes. Add these to drift monitoring.
4. **Feature engineering did not beat the pipeline features** on this subset (v1 kept). It does reach the same performance with 66 instead of 133 features. Logistic regression on v2 matched or beat GBM on the test month.
5. **This is a quarterly-snapshot subset.** The trend features are handicapped and DEV has only 3 months. **Next step: re-run on the full monthly gold** (578,975 rows) once ≥ 2.5 GB RAM is free, for final numbers and the final v1/v2 decision.
