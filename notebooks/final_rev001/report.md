# Precursor Intelligence: final modelling report (rev001)

**Project:** IE7275 MLOps, "Precursor Intelligence".
**Question:** For every US nursing home and every month *t*, what is the probability of a serious CMS health citation (severity G or worse) in the next 12 months? The label is `y_g12`, and the users are liability insurers.
**Status:** modelling is closed. This report explains what the final model is, how we got there, every score along the way, and why we stopped short of the original 80 % accuracy target.
**Date:** 2026-11-07.

Every number in this report comes from a file in this folder or from the project's run logs. Section 12 indexes the files. Where two sources disagree, the report says so, and the main disagreements are collected in §12.3.

---

## Contents

1. Executive summary
2. The problem from first principles
3. Data and pipeline
4. Timeline of every experiment and score
5. Feature engineering
6. Diagnosing the ceiling
7. External evidence
8. The decision, as a chain of reasoning
9. What we deliver and how to read it
10. What would change the conclusion
11. State-wise bias
12. Appendix: glossary, discrepancies, file index
13. Final model: description and complete performance

---

## 1. Executive summary

**Target and result.** The team set a target of **80 % accuracy**. The best honest model reaches **76.4 % accuracy**. Predicting "no citation" for every facility already gives **74.4 %**, because only 25.6 % of facility-months in the test month are positive. Measured by ranking quality, the model is useful:

- PR-AUC 0.481, against 0.256 for random ordering.
- ROC-AUC 0.709.
- About **80 % precision among the top 2 %** of flagged facilities, a 3.1× lift over the base rate.

**Why we stopped.** We tried the following, and none of them moved the result by more than about 0.002 PR-AUC:

- 6 dataset builds
- 12 numbered experiments
- 11 row filters
- 14 extra CMS tables
- daily payroll staffing (PBJ)
- deep tabular models

Error analysis then showed where the missing information is. **68 % of the positives come from complaint surveys only.** These surveys are unscheduled, and our data can't see them coming. Even an *oracle* that is told how many complaint surveys each facility will get in the next 12 months reaches only **78.2 % accuracy** (ROC 0.819, PR-AUC 0.584). That oracle cheats by looking at the future, and it still misses 80 %. The published literature agrees: comparable inspector-judgement labels top out around ROC 0.75–0.78. We therefore stopped. 80 % accuracy can't be reached honestly with public data, and the team decided not to chase it further.

### Key numbers

| Quantity | Value | Source |
|---|---|---|
| Base rate of `y_g12` (test month 2025-02) | 25.6 % (14,643 facilities) | `results/ALL_TABLES_REPORT.md` |
| Accuracy of "predict no citation" | **74.4 %** | same |
| First baselines (test 2024-10): CMS star rating / severity rule | PR-AUC 0.344 / 0.415; ROC 0.629 / 0.663 | `results/ALL_RESULTS_TABLE.md` |
| First learned model (subset, LR on 66 features) | PR-AUC 0.492, ROC 0.708 (test 2024-10) | same |
| Fake score from a leaky random-row split | ROC 0.926, PR-AUC 0.824, **accuracy 87.3 %** (invalid) | `results/summaries/exp11__summary.json` |
| **Recommended model** (LightGBM + LR mean ensemble, 133 gold features, trained on 2023 rows only) | **PR-AUC 0.481, ROC 0.709, accuracy 76.4 %, P@2 % 80.5 %, P@5 % 68.2 %, P@10 % 57.9 %** | `results/summaries/exp9__summary_recency.json` |
| Highest accuracy any threshold reached on test (picked on test, so an upper bound) | 76.1 % (2022-01..2023-11 ensemble) | `results/summaries/exp8b__summary_refit.json` |
| Oracle with future complaint-survey counts | ROC 0.819, PR-AUC 0.584, best accuracy **78.2 %** | `research/DEEP_RESEARCH.md` §1 |
| Published ceiling for comparable labels | ROC ~0.75–0.78, PR-AUC ~0.52–0.56 | `research/DEEP_RESEARCH.md` §2 |

---

## 2. The problem from first principles

### 2.1 How a label is generated

A G+ citation is not a direct measurement of harm. It is the end of a chain of events, and every link in the chain must happen inside the 12-month window:

```
facility condition ──► a survey happens ──► surveyors find a deficiency ──► they judge its severity ≥ G ──► CMS publishes it
                        (timing)              (detection)                    (judgement)                    (posting lag)
```

1. **A survey must happen.** CMS writes a citation only during an inspection, and there are two kinds:
   - **Standard surveys** recur on a cycle of roughly 9–15 months. The gold feature `days_to_std_due` uses a 456-day (15-month) due date.
   - **Complaint surveys** are unscheduled. A complaint or a facility-reported incident triggers one, and nothing in the public data says when the next one will arrive.

   Infection-control surveys exist as well. Our project log records that 21.7 % of `y_g12` negatives had no health survey in their window when infection-control surveys are excluded. Counting those surveys, 14.5 % of the test rows were unsurveyed. On the 2025-02 test month, 2,261 of 14,643 rows (15.4 %) had no health survey at all (`research/DEEP_RESEARCH.md`). Those rows are negative by construction, whatever the quality of care.
2. **Surveyors must find a deficiency, and they choose its severity.** CMS's scope × severity grid runs from A to L:
   - G, H and I mean actual harm.
   - J, K and L mean immediate jeopardy.

   Severity is a human judgement, and §7.2 shows how much it varies.
3. **The citation must be published and not removed.** Gold counts a G+ event only if `removed_by_cms` is false. It also waits a **posting buffer of 8 months** (`configs/labels.yaml`, `buffers_months.citations: 8`) before declaring a label final.

So, in probability terms:

> P(y = 1 | x) = P(survey in window | x) × P(G+ found | survey, x)

The first factor depends on scheduling and on complaints. The second depends on the facility's real condition and on the surveyor. Our features describe the facility as of month *t*. They can't observe future complaints, and they can't observe which surveyor team will be sent.

### 2.2 How gold computes the label

Gold uses `src/precursorintelligence/gold/labels.py` (the feature-family modules are copied as reference in `feature_engineering/gold_features/`). For a spine row (facility *ccn*, month *t*, cutoff date) with horizon H = 12 and buffer B = 8, it decides the status in this order:

1. `immature`: t + H + B > V, the newest vintage. The window has not been fully observed, so there is no label.
2. `positive`: there is a G–L citation with a survey date in (cutoff, cutoff + 12 months] that CMS has not removed.
3. `censored`: there is no event, and the facility disappears from the data before t + H + B.
4. `negative`: everything else.

Only `positive` and `negative` rows are used for training and evaluation.

### 2.3 What the 74.4 % majority baseline means for accuracy

Because 74.4 % of test rows are negative, the constant answer "no citation" is already 74.4 % accurate. A model raises accuracy only through the rows it flags as positive. Each flagged row adds +1 correct answer if it is a true positive and −1 if it is a false positive, compared with saying "no". If a model flags a fraction *f* of the rows with precision *p*:

> accuracy = majority accuracy + f · (2p − 1)

The rest of this subsection is arithmetic on reported numbers.

- **A flag helps accuracy only if p > 50 %.** Most of the useful risk ranking happens below that precision, so accuracy throws it away.
- **The facility-split model shows the formula works.** At its accuracy-best threshold it flags 12.0 % of rows at 58.5 % precision: 0.120 × 0.170 = +2.0 points, which turns 74.4 % into 76.4 %. That is the measured value.
- **Reaching 80 % needs +5.6 points.** That means, for example, flagging 10 % of facilities at about 78 % precision, or 9.3 % at 80 %. Our model reaches 80 % precision only in the top 2 %, which is worth 0.02 × 0.61 ≈ +1.2 points. In the top 10 % its precision is 57.9 %, worth about +1.6 points.

So accuracy has at most 25.6 points of headroom, and the model's precision falls below the level that would move it long before enough facilities are flagged.

### 2.4 Why we use PR-AUC, ROC-AUC, precision@k and lift

| Metric | What it answers | Value with no skill |
|---|---|---|
| **PR-AUC** (primary) | How clean is the ranking among the positives? It is sensitive to the 26 % minority class. | the base rate, about 0.26 |
| ROC-AUC | Given one cited and one uncited facility, how often does the model rank the cited one higher? | 0.50 |
| **Precision@k %** | Of the top k % flagged, what share are cited? This is how an insurer with a fixed review budget uses the score. | the base rate |
| **Lift@k %** | Precision@k divided by the base rate | 1.0× |
| Brier score, ECE | Are the probabilities numerically right? (needed for pricing) | Brier of the base rate, 0.191–0.196 |
| Accuracy | Reported for completeness, always next to the 74.4 % majority | 74.4 % |

The project's experiment rules (the team's feature-engineering playbook) put PR-AUC first, then Recall@k, F1, Brier and accuracy.

### 2.5 Point-in-time and leakage rules

- **Features use only what CMS had published by month *t*.** Gold reads silver only through `FeatureReader` (`feature_engineering/gold_features/reader.py`):
  - It reads `*_by_vintage` tables at vintage *t*, and also at *t−6* and *t−12* for change (Δ) features.
  - It refuses ledgers, other month offsets, and any vintage after *t*.
  - It records every read, so the gold tests can rebuild features from a reader limited to *t* and check that they are equal ("time-travel test").
- **Events are counted only when `survey_date ≤ cutoff`** (citations, penalties, surveys). Labels count only events after the cutoff.
- **The vintage is the CMS "Processing Date" inside the file,** not the zip date (`results/MISSING_MONTHS_REPORT.md`).
- **Star ratings:**
  - `health_inspection_rating` and `overall_rating` are `ablation_only`. They are never features, because they are computed from the citation history that the label also uses.
  - `staffing_rating` and `qm_rating` are `model` features under the column contract (`experiments/rev002/feature_registry_rev002.yaml` header).
  - The 7 newer families never use star ratings.
- **The A8 leakage scan** found no feature with single-feature AUC > 0.80 or IV > 0.5. The strongest feature has IV 0.37 (`reports/rev002_gold2022_fe_eda_report.md`).
- **Splits are by time, with a purge gap** (§3.4 and §4.3). Folds that decide keep or drop are grouped by facility.
- **Thresholds are chosen** on calibration months, validation data or out-of-fold training predictions, never on test. The one exception, the "best possible accuracy on test", is labelled as an upper bound.

---

## 3. Data and pipeline

### 3.1 Source

The source is the CMS Nursing Home Provider Data Catalog monthly archives, 2021-01 → 2026-10: 68 zips plus 2021 history (`results/MODEL_CARD.md`). The 7 core tables are:

- Provider Information
- Health Citations
- Inspection Dates (surveys)
- Penalties
- MDS quality measures
- Claims quality measures
- Citation lookup

The "all tables" build added 14 more, for 21 in total (`results/ALL_TABLES_REPORT.md`).

### 3.2 Medallion layers

| Layer | What it holds | Size (latest build) |
|---|---|---|
| Landing / ingest | raw zips, CRC-checked, inventoried by Processing Date | 68 zips; ingest run 0 failures |
| **Bronze** | one partition per table per data month, as published | 1,207 monthly files, 21 tables, 0 failures |
| **Silver** | cleaned, typed `*_by_vintage` tables plus event ledgers | 29 tables. Ledgers from the 7-table rebuild: citations 993,199, surveys 491,981, penalties 112,460 (project log 2026-11-06) |
| **Gold** | one row per facility × month (spine), features + labels + split columns | 775,400 facility-months, 15,452 facilities, 52 months (2022-01 → 2026-10), 202 model features (`results/MODEL_CARD.md`) |

**Empty months.** 7 data months have no CMS file at all:

- 2021-12, 2022-12, 2023-12, 2024-12 and 2026-01 are lost to a year-end shift: CMS re-issues October data and skips a December stamp.
- 2025-01 and 2025-08: CMS published nothing.

These are by design (`results/MISSING_MONTHS_REPORT.md`), and they are why the latest test window is a single month, 2025-02.

### 3.3 Dataset vintages used during the project

| ID | Dataset | Rows | Train rows used | Test (rows) | Features |
|---|---|---|---|---|---|
| D1 | Subset, 15 quarterly snapshots | 222,677 | 44,295 (2023-01/04/07) | 2024-10 (14,665) | 133 v1 / 66 selected |
| D2 | Gold 2023 (spine 2023) | 578,975 | 132,832 (2023-01 → 09) | 2024-10/11 (29,324) | 75 selected |
| D3 | Gold 2022 (spine 2022) | 746,024 | 296,476 (2022-01 → 2023-09) | 2024-10/11 (29,324) | 75 selected |
| D4 | Gold 2022, with 12-month purge gap before calib | 746,024 | 89,459 (2022-01 → 06) | 2024-10/11 (29,324) | 133 (+120 engineered in exp5) |
| D5 | All-tables gold (21 tables) | 775,400 | 119k–326k (varies by run) | 2025-02 (14,643) | 133 / 202 / 100 selected |
| D6 | All-tables gold, 70/15/15 splits | 502,102 labelled | ~351k | ~75k (strictest 21,934) | 133 / 202 |

Source: `results/ALL_RESULTS_TABLE.md` §1.

### 3.4 Splits

The team split is defined in `configs/splits.yaml` and recomputed from the newest vintage V on every run:

- **Test:** the last labelled months. The label window must be closed: t + 12 + 8 ≤ V.
- **Purge:** the 12 months before test. These rows are labelled but never used, because their label windows overlap the test windows.
- **Calibration:** 3 months before the purge.
- **Train:** from `spine_start` (2022-01) up to calibration.
- **Train → calib gap** (added in experiments, `experiments/rev002/exp_common.py`, `EXP_CALIB_GAP = h`): gold originally had no purge between train and calibration. The experiments relabel the last 12 train months as `purge_calib` and drop them. §4.3 explains why.

For D5 (`results/ALL_TABLES_REPORT.md`) the windows are: train 2022-01..2022-08, 12-month gap, calib 2023-09..2023-11, test 2025-02.

---

## 4. Timeline of every experiment and score

All scores are on **test months, scored once**, unless marked. Test months differ between datasets (D1: 2024-10; D2–D4: 2024-10/11; D5: 2025-02), so compare across dataset blocks only loosely. "—" means not measured.

### 4.1 Master table

| # | Step | Data | One change | PR-AUC | ROC | Acc. | Decision |
|---|---|---|---|---|---|---|---|
| 0 | Majority class | D5 | predict "no citation" | 0.256 | 0.500 | 74.4 % | the floor for accuracy |
| 1 | **Baseline: CMS star rating** | D1 | rank by overall star rating | **0.344** | 0.629 | — | first baseline |
| 2 | **Baseline: severity rule** | D1 | rank by prior severe citation | **0.415** | 0.663 | — | stronger baseline |
| 3 | GBM, 133 pipeline features | D1 | learned model | 0.482 | 0.706 | — | beats baselines by +0.07 |
| 4 | GBM, 66 selected | D1 | B1–B9 feature engineering + selection | 0.484 | 0.703 | — | tie with 133 |
| 5 | **LR, 66 selected** | D1 | linear model | **0.492** | 0.708 | — | best on subset |
| 6 | Baselines on full gold | D2 | star / severity | 0.341 / 0.411 | 0.628 / 0.661 | — | same as subset |
| 7 | GBM default / GBM 133 / LR / CatBoost / LGBM tuned | D2 | model zoo on 75 selected | 0.479 / 0.483 / 0.486 / 0.486 / 0.489 | 0.702–0.708 | — | all within 0.01 |
| 8 | **Ensemble LGBM + CatBoost + LR** | D2 | ensemble | **0.490** | 0.709 | — | best D2 |
| 9 | Ensemble, 6.7× more rows (2022 spine) | D3 | add 2022 to training | 0.488 | 0.708 | — | **more rows ≠ better** |
| 10 | Calibration of D3 ensemble | D3 | Platt / isotonic | 0.488 (raw) / 0.482 (iso) | 0.708 | — | raw mean already calibrated (ECE 0.022, Brier 0.172) |
| 11 | rev002 exp1–exp4 without a purge gap | D4 (no gap) | DL, DCN/DeepFM, LGBM GOSS/EFB, target encoding | 0.3725 – 0.4778 | 0.632–0.699 | — | **calib/test disagree, so leak found** |
| 12 | **Add 12-month purge gap** | D4 | `EXP_CALIB_GAP=h` | — | — | — | calib and test now agree; adopted |
| 13 | exp4 LR one-hot | D4 | linear, gap | **0.481** | 0.699 | — | best D4 |
| 14 | exp3 LGBM tuned / default / GOSS / EFB | D4 | boosting variants | 0.474 / 0.462 / 0.462 / 0.462 | 0.689–0.700 | — | EFB is a no-op |
| 15 | exp4 LGBM target encoding / native cat | D4 | categorical encodings | 0.470 / 0.465 | 0.695 / 0.693 | — | no gain |
| 16 | exp1 FT-Transformer / TabNet / TabNet-pretrained | D4 | tabular deep learning | 0.464 / 0.436 / 0.461 | 0.675–0.694 | — | below LR |
| 17 | exp2 MLP / DeepFM / DCN-v2 | D4 | interaction networks | 0.462 / 0.452 / 0.451 | 0.674–0.689 | — | below LR |
| 18 | **exp5 +120 engineered features** | D4 | 133 → 253 features | LR 0.4811 → 0.4738; LGBM 0.4739 → 0.4735 | 0.699–0.700 | — | **dropped** (−0.007 for LR) |
| 19 | Row filters R1–R11 | D3 (75 f.) | remove training rows | 0.4860 – 0.4879 (base 0.4870) | 0.706–0.708 | — | **none adopted** (§5.6) |
| 20 | exp6 survey × harm split | D4 | P(survey) × P(harm \| survey) | 0.4671 vs 0.4675 | — | — | **no gain** |
| 21 | exp7 case-mix staffing | D4 | +6 case-mix features | 0.4675 → 0.4691 | 0.696 → 0.697 | — | real but tiny; not adopted |
| 22 | exp8 14 extra CMS tables | D5 | 133 → 202 / 100 selected | 0.4754 → 0.4733 (refit) | 0.708 → 0.707 | 76.0 → 75.9 % | **dropped** |
| 23 | **exp9 recency** | D5 | train window only | 0.4651 → **0.4813** | 0.700 → **0.709** | 75.6 → **76.4 %** | **adopted: train on recent months** |
| 24 | exp10/11 random-row split | D6 | shuffle rows 70/15/15 | 0.824 | 0.926 | **87.3 %** | **invalid (leak)** |
| 25 | exp10/11 facility split | D6 | split by facility | 0.503 | 0.733 | 76.4 % | honest, same months |
| 26 | exp10/11 time split (no purge) | D6 | split by time | 0.488 (133) / 0.505 (202) | 0.714 / 0.727 | 74.5 / 74.8 % | 202 gain judged a leak |
| 27 | exp10/11 facility + time | D6 | strictest | 0.486 / 0.485 | 0.704 / 0.706 | 74.8 % | ceiling confirmed |
| 28 | **exp12 PBJ daily staffing** | D5 | +24 PBJ features | 0.4802 → 0.4810 | 0.7092 → 0.7095 | 75.80 → 75.72 % | real but tiny; not adopted |
| 29 | Error analysis + oracle | D5 test | diagnosis, not a model | oracle 0.584 | oracle 0.819 | oracle 78.2 % | **80 % not reachable** |
| 30 | Observable proxies (FRI/complaints, state strictness, sibling owners) | D5 test | 5-fold stacking diagnostic | 0.4800 → ≤ 0.4834 | 0.7092 → ≤ 0.7120 | — | +0.001–0.003 ROC each |
| 31 | **Decision** | — | stop modelling toward 80 % | — | — | — | final (2026-11-07) |

### 4.2 Phase 1: baselines and the first models (D1 subset, then D2 / D3 full gold)

- **Subset (D1).** The first run used the subset `rev001`, 15 quarterly snapshots (`reports/rev001_subset_fe_eda_report.md`). The two rule baselines gave PR-AUC 0.344 (star rating) and 0.415 (severity rule). Learned models immediately reached 0.48–0.49, and the best was LR on 66 selected features: 0.492 / 0.708 / P@5 % 0.694. On the holdout, v1 (133 features) and the selected set tied: 0.512 vs 0.509 PR-AUC.
- **Gold 2023 (D2).** Feature selection on DEV 2023-01..09 (`feature_engineering/selection_gold2023/select.log`) went 252 candidates → 176 after the filter → **75 selected** (27 engineered, 48 original).
  - On the facility holdout, v1 beat the selected set (0.5207 vs 0.5167, `accept selected: False`). The team kept the 75-feature set anyway for the boosting runs, as a simpler set within noise.
  - The tuned ensemble scored PR-AUC **0.490**, ROC 0.709 and P@5 % 0.703 (`feature_engineering/selection_gold2023/boost.log`).
- **Gold 2022 (D3)** trained on 6.7× more rows (296,476 vs 44,295 on the subset) and did not improve: the ensemble scored 0.488.
  - Holdout (20 % of DEV facilities, same months): 0.520 / 0.737.
  - The top 0.5 % of homes were 92 % positive (3.5× lift).
  - The raw ensemble mean was already calibrated: ECE 0.022, Brier 0.172 against 0.194 for the base rate.
  - The top decile was over-predicted by 5–6 points (0.65 predicted vs 0.59 observed), the first sign of drift (`reports/rev002_gold2022_fe_eda_report.md` C2–C6).

**Lesson 1:** the model family and the data volume barely matter. Every model sat within about 0.01 PR-AUC of every other.

### 4.3 Phase 2: the purge-gap discovery (rev002 experiments, D4)

The first rev002 runs (exp1–exp4, gold split as-is) ranked models differently on calib and test (`results/comparison.md`, `calib_gap = 0` rows):

- LightGBM default: calib PR-AUC **0.7485**, test **0.4565**.
- Logistic regression with target encoding: calib 0.518, test 0.4778.

A calib score 0.29 above test is not drift. It is a leak.

**Mechanism.** The label of a row at month *t* covers months (t, t+12]. A training row from 2023-06 and a calib row from 2023-07 have label windows that overlap by 11 months, and they belong to the same facility with almost the same features. The model therefore "sees" calib citations during training. Early stopping on calib rewards memorising those citations, and LightGBM early stopping ran to about 4,758 trees.

**Fix.** Drop the last *h* = 12 training months before the first calib month (`experiments/rev002/exp_common.py`, `apply_calib_gap`). Train became 2022-01..06 (89,459 rows). With the gap:

- LightGBM early stopping fell from about 4,758 to 18 trees (project log).
- Calib PR-AUC fell to 0.508.
- Calib and test rankings agreed.

All later experiments keep this gap or an equivalent purge.

Gap results (`results/comparison.md`, `calib_gap = h`):

- LR one-hot **0.4811** > LGBM tuned 0.4739 > LGBM target-encoded 0.4703 > FT-Transformer 0.4636 ≈ MLP 0.4620 ≈ LGBM default 0.4618 > TabNet-pretrained 0.4613 > DeepFM 0.4521 > DCN-v2 0.4509 > TabNet 0.4357.
- EFB variants equal the default: with 133 dense features there is nothing to bundle.

**Lesson 2:** an evaluation window that sits next to the training window inflates scores. Every honest split needs a purge at least as long as the label horizon.

### 4.4 Phase 3: features, rows and label structure (exp5, row filters, exp6, exp7)

- **exp5:** +120 engineered features gave no lift (§5.5).
- **Row filters (R1–R11):** no filter mattered. Every CV change was within ±0.003, and the baseline fold standard deviation was 0.011 (§5.6).
- **exp6 survey × harm** (`results/summaries/exp6__summary.json`, 3 seeds). Arm A modelled P(survey in window) and scored test PR-AUC **0.947** against a base rate of 0.855, so survey timing is very predictable.
  - On all test rows: A × B = 0.4671 vs base 0.4675. The clustered bootstrap difference was −0.0001 (CI −0.0020..+0.0016).
  - On surveyed rows only: B (trained only on surveyed rows) = 0.482 vs base 0.492 (CI −0.012..−0.007).
  - **Conclusion:** the base model already learns survey timing. The ceiling is P(harm | survey), and restructuring the label can't fix that.
- **exp7 case-mix** (`results/summaries/exp7__summary.json`). Six case-mix HPRD and staffing-to-case-mix features moved test PR-AUC from 0.4675 to 0.4691. The facility-clustered CI was +0.0001..+0.0036, and P@5 and P@10 were slightly lower. The best new feature was `staff_vs_cm_total`, at gain rank 22–32 of 139. The effect is real but tiny.

### 4.5 Phase 4: all CMS tables, recency and split robustness (exp8–exp11, D5 / D6)

- **exp8: 14 extra tables, 69 new features** (`results/ALL_TABLES_REPORT.md`).
  - Calibration PR-AUC went from 0.5045 to 0.5066 (+0.0021). The best family was survey_summary at +0.0023.
  - On test, the 100-feature selected set was *worse*: −0.0066..−0.0009 against the old 133 in the train-only fit.
  - After refitting to 2023-11: old 133 ensemble **0.4754** / 0.708 / 76.0 %, selected 100 ensemble 0.4733 / 0.707 / 75.9 %.
  - No threshold, even one picked on test, exceeded **76.1 %** accuracy.
- **exp9 recency vs row count** (`results/summaries/exp9__summary_recency.json`). Only the training window changes. Test is 2025-02, and all arms share one fixed threshold chosen on calib.

| Arm | Train rows | PR-AUC | ROC | Acc. | P@2 % | P@5 % | P@10 % | Δ PR-AUC vs R1 (95 % CI) |
|---|---|---|---|---|---|---|---|---|
| R1 2022-01..08 | 119,165 | 0.4651 | 0.700 | 75.6 % | 78.8 | 64.5 | 56.2 | — |
| R2 2022-01..2023-11 | 325,920 | 0.4754 | 0.708 | 76.0 % | 79.9 | 66.5 | 57.2 | +0.007..+0.013 |
| R3 2023 only | 162,276 | 0.4802 | 0.709 | 76.4 % | 80.2 | 67.8 | 57.5 | +0.010..+0.021 |
| **R4 2023 only, R1's size** | 119,165 | **0.4813** | **0.709** | **76.4 %** | **80.5** | **68.2** | **57.9** | **+0.011..+0.022** |
| R5 2022 only, R3's size | 162,276 | 0.4660 | 0.702 | 75.7 % | 78.5 | 64.6 | 56.0 | −0.000..+0.002 |

  Newer months at the same size (R4 vs R1) gain +0.016 PR-AUC. More rows from the same period (R5 vs R1) gain nothing. Adding 2022 to 2023 (R2 vs R3) makes the model slightly worse. **The gain comes from recency, not from row count.**

- **exp10 / exp11: 70/15/15 splits on all 502,102 labelled rows** (`results/summaries/exp11__summary.json`).

| Split | Features | ROC (95 % CI) | PR-AUC | Acc. | Majority | Verdict |
|---|---|---|---|---|---|---|
| random rows | 133 | 0.926 | 0.824 | **87.3 %** | 73.8 % | **leaky, invalid** |
| facility | 133 | 0.733 (0.720–0.747) | 0.503 | 76.4 % | 74.4 % | honest, but shares months with train |
| facility | 202 | 0.732 (0.720–0.743) | 0.502 | 76.7 % | 74.4 % | no gain from new tables |
| time (no purge) | 133 | 0.714 (0.706–0.721) | 0.488 | 74.5 % | 73.6 % | forecast |
| time (no purge) | 202 | 0.727 (0.721–0.736) | 0.505 | 74.8 % | 73.6 % | +0.014 ROC, judged a leak |
| facility + time | 133 | 0.704 (0.689–0.721) | 0.486 | 74.8 % | 73.3 % | strictest |
| facility + time | 202 | 0.706 (0.692–0.723) | 0.485 | 74.8 % | 73.3 % | gain vanishes (CI includes 0) |

**Why the random-row split gives a fake 87 %.**

- Shuffling rows puts the same facility in train and test, often in adjacent months.
- Adjacent months of one facility have near-identical features. Their label windows overlap by 11 of 12 months, so they usually share the very same future citation.
- The model learns "this facility → this label". It does not learn "these conditions → harm".
- Splitting by facility removes the identity leak: ROC drops from 0.926 to 0.733. Splitting by facility and time also removes the shared-window leak: ROC 0.704.

**Why the 202-feature gain on the time split is not trusted.**

- The time split has no purge, so the 12-month labels of the last training months (to 2024-02) are the same citations as the validation and test labels.
- Validation ROC was 0.82–0.85 against a test ROC of 0.71–0.73, and early stopping never stopped (3,000 trees).
- The new tables add facility-identifying features such as `own_months_since_oldest_assoc`, `vbp_snfrm_performance_rate` and `days_c1_to_c2_health_survey`.
- With new facilities (facility + time split) or with a proper purge (exp8/exp9), the gain disappears.

**Drift.** Per test month (time split, 133 features), ROC decays 0.733 (2024-08) → 0.722 → 0.713 → 0.704 → 0.695 (2025-02). Calibration ECE is 4.7–5.5 points on the time split against 0.6–0.7 points on the facility split. In the top decile, the model predicts 0.71 and the actual rate is 0.58.

### 4.6 Phase 5: the last lever (exp12 PBJ daily staffing)

Source: `reports/exp12_pbj_report.md` and `results/summaries/exp12__summary.json`.

- **Setup:** train 2023-01..11 (162,276 rows). Test 2025-02 (official) and 2024-11 (extra).
- **Features:** 24 PBJ features built point-in-time. Month *t* uses the newest PBJ quarter published strictly before *t*.
- **Keep rule** (3 facility-grouped folds):
  - LightGBM +0.0020 ± 0.0011: keep.
  - Ensemble +0.0011 ± 0.0007: keep.
  - LR −0.0004 ± 0.0012: no.
- **Test 2025-02:** ensemble PR-AUC 0.4802 → 0.4810 (CI −0.0004..+0.0018). Accuracy 75.80 % → 75.72 %.
- **Test 2024-11:** 0.4892 → 0.4904 (CI +0.0001..+0.0023).
- **Most useful PBJ features:** census size and contract-staff share, at gain ranks 13 and 15.

**Lesson 3:** staffing-type information is saturated. Case-mix gave +0.0017, PBJ +0.001–0.002, and the 14 tables nothing.

### 4.7 Phase 6: diagnosis and decision

The error analysis and the oracle are in §6. The decision is in §8.

---

## 5. Feature engineering

### 5.1 Principles applied to every feature

1. **Point in time.** A feature for (facility, *t*) is computed only from silver tables at vintage *t*. Δ features also read *t−6* or *t−12*. Event features count only events with date ≤ cutoff (§2.5).
2. **Look-back windows of 12, 24 and 36 months.** Short windows capture current trouble, long ones capture chronic patterns. The 36-month window matches the CMS survey-history horizon.
3. **Δ at t−6 and t−12.** These capture direction. A Δ is null (never 0) when the earlier month was never published, when the facility is absent from it, or (for quality measures) when the measure definition changed or the value was not refreshed. A null is "unknown", which LightGBM can treat differently from "no change".
4. **Missing means "never happened" where that is true.** Examples: `days_since_last_gplus` is null if the facility never had a G+ citation, and so are the `days_since_last_*` features. The engineered `isna_*` flags make this explicit.
5. **No star ratings derived from inspections** (§2.5), and no label-like columns. Gold asserts this.

### 5.2 The 133 gold features: the recommended model's input

These are the six families in `experiments/rev002/feature_registry_rev002.yaml` (all `model_use: model`). The code is in `feature_engineering/gold_features/`.

| Family (#) | Source table at *t* | What it measures | Representative features (windows) |
|---|---|---|---|
| **survey_timing** (9) | Inspection Dates | When the facility was last inspected and how often. Complaint surveys appear only if cited. | `days_since_last_std`, `days_to_std_due` (456-day due), `std_overdue`, `n_std_36m`, `n_complaint_surveys_cited_12m/_24m`, `n_ic_surveys_12m`, `days_since_last_complaint_survey_cited`, `no_std_on_record` |
| **citations** (26) | Health Citations (+ surveys, beds) | How many citations, how serious, what kind | `n_cit_12m/_24m`; by severity band `n_cit_AC/DF/GI/JL_24m`; `sev_points_12m/_24m`, `sev_points_per_bed_24m`; `max_sev_last_std`, `gplus_last_std`, `max_sev_prev_std`, `gplus_prev_std`; `n_gplus_surveys_36m`, `days_since_last_gplus`; `n_complaint_cit_12m`, `n_complaint_gplus_24m`; `n_ic_cit_24m`; 6 category counts (abuse, quality of life/care, assessment, nursing/physician, resident rights, administration; 24 m); `n_repeat_tags_last_std`, `n_uncorrected` |
| **enforcement** (7) | Penalties | Fines and payment denials | `n_fines_24m`, `fine_dollars_24m`, `max_fine_24m`, `n_denials_24m`, `denial_days_24m`, `any_penalty_36m`, `days_since_last_penalty` |
| **facility_staffing** (42) | Provider Info at *t*, *t−6*, *t−12* | Profile and staffing | beds, occupancy, ownership type, provider type, `state`, council, sprinkler, in-hospital, CCRC, age, SFF, abuse icon, ownership change 12 m, inspection > 2 y; HPRD aide/LPN/RN/total/weekend/PT, case-mix-adjusted HPRD, RN share, weekend gap; turnover total/RN, admin departures; 9 footnote reason codes; `staffing_rating`, `qm_rating`; Δ: `d6_hprd_total`, `d6_hprd_rn`, `d12_hprd_total`, `d12_turnover_total` |
| **quality** (43) | MDS QM (four-quarter avg) + Claims QM (adjusted) | Clinical outcomes | 15 long-stay measures, 3 short-stay measures, their footnotes, and Δ6 for antipsychotics, falls with major injury, pressure ulcers and rehospitalisation; `qm_missing_count`; data age `qm_age_months_mds/_claims`; definition-change flags `qm_def_post_2025_02/_2026_02` |
| **context** (6) | derived within *t* | The facility against its state, plus regime flags | `state_gplus_rate_12m`, `state_median_hprd_total`, `rel_hprd_total`, `rel_sev_points_24m`, `era` (pre/post 2025-07), `penalty_regime_2025_10` |

Single-feature signal is strongest for inspection history, complaints and state (A3 on Gold 2022, single-feature AUC, top 8):

| Feature | AUC |
|---|---|
| `n_cit_24m` | 0.667 |
| `sev_points_24m` | 0.671 |
| `state` | 0.662 |
| `state_gplus_rate_12m` | 0.651 |
| `n_cat_qol_care_24m` | 0.658 |
| `n_cit_DF_24m` | 0.661 |
| `n_complaint_surveys_cited_24m` | 0.665 |
| `n_gplus_surveys_36m` | 0.645 |

Quality measures are weak on their own: the best, `qm_ls_depressive`, has AUC 0.55. 57 of the 133 features have IV < 0.02.

LightGBM gain on the 133-feature model ranks these highest (`results/MODEL_CARD.md` §4):

- state
- complaint surveys with citations (24 m)
- G+ surveys (36 m)
- days since last G+
- certified beds
- state G+ rate
- citations (24 m)
- severity points (24 m)
- quality-of-life/care citations
- complaint citations (12 m)
- fines
- staffing vs state

### 5.3 The recommended model: final feature set and algorithm

This was confirmed from `results/MODEL_CARD.md`, `results/ALL_RESULTS_TABLE.md` (row "Ensemble, train 2023 only (119k) — recommended") and `experiments/rev002/exp9_recency.py`.

| Item | Value |
|---|---|
| Features | the **133 gold features** of §5.2, raw (not transformed by the B1–B8 transformer) |
| LightGBM | Optuna-tuned parameters from `feature_engineering/boost_spine2022/lgbm_best_params.json`: learning rate 0.0102, 34 leaves, max depth 3, min_child_samples 870, subsample 0.63, colsample_bytree 0.33, reg_lambda 9.48, reg_alpha 0.0068, min_split_gain 0.435, 1,265 trees. CV PR-AUC at tuning: 0.5112 |
| Logistic regression | median imputation + missing indicators, standard scaling, one-hot categoricals (min frequency 50), C = 0.3 (`experiments/rev002/exp8_lr.py`) |
| Ensemble | mean of the two probabilities |
| Training window | labelled rows from 2023 only: exp9 R4 = 119,165 rows sampled from 2023-01..2023-11. R3, all 162,276 rows of the same window, scores 0.4802, the same within noise. The operational rule in the model card is to retrain on the most recent 12 labelled months. |
| Threshold for accuracy | 0.5665, chosen on calib months in exp8 |
| Test 2025-02 | PR-AUC 0.4813, ROC 0.709, accuracy 76.4 %, P@2 % 80.5 %, P@5 % 68.2 %, P@10 % 57.9 %, Brier 0.1689 |

### 5.4 The engineered layer and the 75-feature selected set

Sources: `reports/rev001_subset_fe_eda_report.md` and `reports/rev002_gold2022_fe_eda_report.md` Part B.

Part A of the EDA found several problems. One `FeatureTransformer` (`experiments/rev002/feature_transform_v2.py`), fitted on DEV only, addresses each of them:

| Step | What it builds | Problem it answers |
|---|---|---|
| B1 | 1–99 % clipping (`*_clip`), `log_*` for 11 count/dollar features, per-bed rates (`cit_per_bed_24m`, `complaints_per_100beds_12m`, `fine_dollars_per_bed_24m`) | 42 heavily skewed features (e.g. `sev_points_per_bed_24m` skew 78); risk rising with size from 16.6 % (< 50 beds) to 34.0 % (200+) |
| B2 | `bed_band`, `std_recency_band` | stepwise effects |
| B3 | rare levels → `other`; **`state_te`**, a smoothed out-of-fold target encoding of state. Raw `state` is removed from v2 | 53 state levels with positive rates from 6.9 % to 62.1 % |
| B4 | crosses: complaints × low-RN rank, harm-at-last-survey × overdue, size × ownership, chain × turnover | weak single features |
| B5 | percentile within state-month (staffing, turnover, citations, complaints, falls) | state drift |
| B6 | momentum (last 12 m vs prior 12 m), `lag3_diff_*`, `vol4_*` from earlier months of the same facility only | trends |
| B7 | `isna_*` flags | "never happened" missingness |
| B8 | 3-component PCA quality index | weak, correlated quality measures |
| B9 | selection: filter (near-constant, empty lags, \|ρ\| ≥ 0.95 duplicates), then LightGBM permutation importance on 3 facility-grouped folds. A feature is kept if its importance is positive in ≥ 2 of 3 folds | redundancy (15 pairs with \|ρ\| ≥ 0.85; VIF up to 365) |

| Selection run | Candidates | After filter | Selected |
|---|---|---|---|
| Subset (D1) | 252 | 173 | **66** (15 engineered + 51 original), `feature_engineering/feature_set_v2_subset66.yaml` |
| Gold 2023 (D2), reused for Gold 2022 (D3) | 252 | 176 | **75** (27 engineered + 48 original), `feature_engineering/feature_set_selected75.yaml` |

The **75 selected features** were used for all D2 and D3 models and for the saved model artifacts. In order of selection importance, they are:

`state_te`, `certified_beds_clip`, `state_gplus_rate_12m`, `n_complaint_surveys_cited_24m`, `n_cit_24m`, `days_since_last_std`, `n_complaint_cit_12m`, `hprd_lpn`, `sev_points_12m`, `x_size_ownership`, `n_cat_resident_rights_24m_clip`, `qm_ls_pressure_ulcer`, `days_since_last_gplus`, `qm_ls_depressive`, `n_gplus_surveys_36m`, `state_median_hprd_total_clip`, `days_since_last_complaint_survey_cited`, `pct_state_hprd_rn`, `n_cat_qol_care_24m`, `sev_points_24m`, `rel_hprd_total`, `hprd_pt`, `rel_sev_points_24m`, `facility_age_years`, `days_since_last_penalty_clip`, `adj_hprd_total`, `qm_ls_uti`, `n_cit_GI_24m_clip`, `x_complaints_x_low_rn`, `qm_ls_falls_major_injury`, `n_fines_24m_clip`, `d12_turnover_total`, `qm_ls_mobility_decline`, `n_cat_administration_24m`, `hprd_weekend_rn_clip`, `n_ic_surveys_12m`, `occupancy`, `qm_ss_ed_visit`, `qm_pca_2`, `pct_state_n_complaint_surveys_cited_12m`, `n_cat_abuse_24m`, `n_std_36m`, `pct_state_turnover_total`, `rn_share`, `n_cat_assessment_24m`, `n_ic_cit_24m`, `d6_qm_ls_antipsychotic`, `max_sev_last_std`, `lag3_diff_turnover_total`, `n_repeat_tags_last_std_clip`, `qm_ls_restraints_clip`, `qm_ls_adl_decline`, `vol4_n_complaint_surveys_cited_12m`, `admin_departures`, `n_uncorrected`, `d6_hprd_rn`, `n_cit_JL_24m`, `provider_type`, `staffing_rating`, `qm_ss_new_antipsychotic_fn`, `vol4_turnover_total`, `isna_days_since_last_gplus`, `n_cit_AC_24m`, `complaints_per_100beds_12m`, `hprd_total`, `lag3_diff_n_cit_12m`, `lag3_diff_n_complaint_surveys_cited_12m`, `x_harm_last_std_x_overdue`, `qm_pca_1`, `vol4_hprd_total`, `resident_family_council`, `most_recent_inspection_over_2y`, `std_recency_band`, `n_complaint_gplus_24m`, `provider_in_hospital`.

`state_te` was the top permutation feature: Δ PR-AUC 0.070 on the subset, about 6× the next feature.

**Why the final model is not the 75-feature model.** Engineered and selected sets never beat the raw 133 on an honest test:

- Gold 2023 holdout: 0.5207 (133) vs 0.5167 (75).
- exp5: −0.007 for LR.

The recommended model also had to be refit on the newest gold build (D5), and only the 133 raw features are produced by the production pipeline. So the 133 raw set is the simpler and equal-or-better choice.

### 5.5 Everything tried and dropped

**The keep rule** (from the team's feature-engineering playbook, applied from the row-filter experiment onwards):

- Change one thing at a time.
- Compare against the baseline on the same 3 facility-grouped folds.
- **Keep the change only if mean Δ PR-AUC > 0 and > 1 standard deviation of the per-fold Δ.**
- Confirm on test with a **facility-clustered bootstrap CI**, 300 resamples, that resamples whole facilities.

Earlier experiments (exp5, exp6, exp7) used test Δ with seeds and clustered CIs. exp8 used calib Δ.

| Tried | Data | Δ PR-AUC | Evidence | Verdict |
|---|---|---|---|---|
| B1–B9 engineered + selection (66 / 75 features) | D1, D2 | v1 0.512 vs 66-set 0.509 (D1 holdout); v1 0.5207 vs 75-set 0.5167 (D2 holdout) | `reports/rev001_*`, `selection_gold2023/select.log` | tie or worse; used for D2/D3 only |
| **exp5 +120 engineered** (B1–B8 transformer refit on gap train) | D4 | LR 0.4811 → 0.4738 (−0.007); LGBM fixed 0.4739 → 0.4735; LGBM retuned 0.4723 (Brier 0.216, miscalibrated) | `results/summaries/exp5__*.json` | dropped |
| exp4 target encoding (empirical Bayes / double validation) | D4 | LGBM 0.4703 / 0.4683 vs LR one-hot 0.4811 | `results/comparison.md` | dropped |
| **exp7 case-mix** (6 features: case-mix HPRD aide/LPN/RN/total, staff-vs-case-mix total/RN) | D4 | +0.0017 (CI +0.0001..+0.0036) | `exp7__summary.json` | real but tiny; not adopted |
| **exp8 14 extra CMS tables** (69 features: fire_safety 13, survey_summary 13, ownership 11, snf_qrp 10, vbp 10, covid 3, state_context 9) | D5 | calib +0.0021; test 0.4754 → 0.4733 (selected 100); train-only CI −0.0066..−0.0009 | `exp8__summary.json`, `exp8b__summary_refit.json` | dropped from the model (kept in silver for product context) |
| **exp12 PBJ** (24 features: census, HPRD by role, contract share, weekend ratios, CV and 5th percentile of daily HPRD, bad-day shares, reporting share, census trend, Δ vs q−1 / q−4) | D5 | CV: LGBM +0.0020 ± 0.0011 (keep), ensemble +0.0011 ± 0.0007 (keep), LR −0.0004 (no); test +0.0008 (CI includes 0) / +0.0012 (2024-11, CI +0.0001..+0.0023) | `exp12__summary.json` | passes rule, but accuracy is flat and the pipeline cost (5.5 GB, quarterly) is high; not adopted |
| Row filters R1–R11 | D3 | all within ±0.003 CV, ±0.0013 test | §5.6 | none adopted |
| Observable proxies (FRI + substantiated complaints; sibling-owner G+; state strictness) | D5 test, stacking diagnostic | ROC +0.0028 / +0.0001 / +0.0020 | `research/DEEP_RESEARCH.md` §1 | not built (§8) |

### 5.6 Row filters (training rows only; validation and test never filtered)

Source: `reports/row_filters_report.md`. Model: LightGBM tuned, 75 features. Baseline CV PR-AUC 0.5246 ± 0.0112, test 0.4870.

| ID | Filter | Reason | Rows removed | CV Δ (mean ± std) | Test Δ | Keep |
|---|---|---|---|---|---|---|
| R1 | drop 2022-H1 | COVID backlog, most unlike 2024 | 30.2 % | +0.0011 ± 0.0017 | +0.0002 | no |
| R2 | drop 2022 | same, stronger | 55.2 % | +0.0004 ± 0.0032 | +0.0009 | no |
| R3 | no standard survey on record | empty history | 0 rows | not run | — | — |
| R4 | last inspection > 2 y | stale history | 21.7 % | −0.0003 ± 0.0002 | −0.0013 | no |
| R5 | staffing missing | imputed noise | 3.2 % | −0.0005 ± 0.0005 | +0.0006 | no |
| R6 | ≥ 10 quality measures missing | unreported block | 2.6 % | **+0.0002 ± 0.0001** | +0.0001 | formally yes; recommended not to adopt |
| R7 | < 20 beds or hospital-based | different population | 3.9 % | +0.0006 ± 0.0011 | −0.0002 | no |
| R8 | facility < 2 years old | short history | 0.6 % | +0.0002 ± 0.0005 | −0.0005 | no |
| R9 | keep 1 month in 3 | redundancy | 65.0 % | −0.0003 ± 0.0007 | −0.0003 | no |
| R10 | 2 % most confidently wrong | label noise | 2 % | −0.0007 ± 0.0006 | −0.0004 | no |
| R11 | 5 % most confidently wrong | label noise | 5 % | −0.0023 ± 0.0010 | −0.0011 (Brier 0.1726 → 0.1780) | no (hurts) |

What the row filters show:

- The model saturates at about 60–100k rows.
- The "confidently wrong" rows are real hard cases (clean-looking homes that do get cited), not label noise to remove.
- Rows are not what limits the model.

---

## 6. Diagnosing the ceiling

Source: `research/DEEP_RESEARCH.md` §1 and Addendum. The test month is 2025-02 (14,643 rows, 3,750 positives), scored with exp12's old ensemble. The label was reconstructed from the silver citation ledger and agrees with `y_g12` on 99.85 % of rows.

### 6.1 Where the positives come from

| G+ came from | Share of positives | Recall in top 20 % | Mean score |
|---|---|---|---|
| **complaint survey only** | **68 %** | 39 % | 0.36 |
| both complaint and standard | 18 % | 45 % | 0.40 |
| standard survey only | 14 % | 30 % | 0.32 |

| Prior G+ in 36 months | Share of positives | Recall in top 20 % |
|---|---|---|
| **none ("first-timers")** | **32 %** | **6 %** |
| 1–2 | 33 % | 34 % |
| 3+ | 35 % | 74 % |

The model finds repeat offenders: 74 % of the 3+ group are in the top 20 %. It almost never finds first-timers. A third of all positives have no G+ history the model could use.

### 6.2 How well the model ranks within each kind of row

| Rows | n | Base rate | ROC | PR-AUC |
|---|---|---|---|---|
| had a standard survey in the window | 10,126 | 0.280 | 0.702 | 0.497 |
| had ≥ 1 complaint survey in the window | 8,770 | 0.406 | **0.632** | 0.557 |
| no health survey at all | 2,261 | 0.000 | — | — |

Among the facilities that did get complaint surveys, the model separates harm from no harm worst (ROC 0.632).

### 6.3 The oracle argument

The oracle adds to the current score a feature that **looks into the future**. It can never be used in production. Its job is to show how much information is missing and where it is.

| Added to the current score | ROC | PR-AUC | Best accuracy |
|---|---|---|---|
| nothing | 0.707 | 0.472 | 76.1 % |
| knew **whether a standard survey would happen** | 0.709 | 0.475 | 76.1 % |
| knew **how many complaint surveys would happen** | **0.819** | **0.584** | **78.2 %** |

1. Knowing standard-survey timing adds nothing. The model already knows it: exp6's P(survey) has PR-AUC 0.947.
2. Complaint-survey arrival is the missing information, worth about +0.11 ROC.
3. **Even with perfect knowledge of future complaint arrivals, accuracy reaches only 78.2 %.** The remaining error comes from surveyor judgement and from the 74 % negative base rate. The 80 % target lies above what even this oracle can reach.

### 6.4 Observable proxies for complaint arrival

The diagnostic is 5-fold logistic stacking on the test month. It is indicative, not a formal experiment.

| Added to the current score | ROC | PR-AUC |
|---|---|---|
| nothing | 0.7092 | 0.4800 |
| facility-reported incidents + substantiated complaints (Provider Info, not used in silver) | 0.7120 | 0.4834 |
| sibling homes' G+ rate through shared owners (87 % coverage) | 0.7093 | 0.4803 |
| state strictness (G+ share of health surveys, last 3 / 6 / 12 months) | 0.7112 | 0.4816 |
| sibling + state strictness | 0.7114 | 0.4820 |

Each proxy adds +0.001 to +0.003 ROC, the same scale as PBJ and case-mix. None of them closes any real part of the 0.11 ROC gap to the oracle.

### 6.5 Why complaint arrival can't be observed in public data

This is the verified addendum in `research/DEEP_RESEARCH.md`.

- **Uncited complaint surveys are invisible.** CMS Inspection Dates lists a complaint inspection "only if the inspection resulted in one or more citations". Only 0.1 % of complaint surveys in 2024–26 have no citation. Complaint *volume* can't be rebuilt from survey data.
- **The two Provider Info counts look like intake data but are not.** "Number of Facility Reported Incidents" and "Number of Substantiated Complaints" count incidents and complaints **in the past 3 years that resulted in a citation**. They are derived from citation history, which is why they add only +0.003 ROC.
  - Their definition broke at the iQIES cutover on 2025-07-14, when they began counting all allegations, including unverified ones.
  - Both columns were **deleted in the 202602 release**.
- **Complaint intake itself is not public.** CMS ACTS / iQIES intake is not published. New York publishes record-level ombudsman complaints (one state only, with a crosswalk needed). No usable 2022–26 intake data was found for CA, TX, IL, WA or OH.
- **Conclusion:** no public, facility-level complaint-intake measure exists. The information the oracle shows to be decisive is not available to us.

### 6.6 Drift: why recent months beat more data

The evidence that the data-generating process moves over time:

- PSI > 0.25 for 35 features against 2022-01 (`reports/rev002_*` A5).
- Per-month ROC decay of 0.733 → 0.695 over 7 months.
- The exp9 recency result (§4.5).
- The over-predicted top decile.

The causes found (`research/DEEP_RESEARCH.md` §2):

- **States catching up on the COVID survey backlog.** 56 % of facilities had a standard survey in 2022 and 65 % in 2024. California went from 38 % to 72 %.
- **New complaint-investigation deadlines** from 2023-10-01 (QSO-22-19-NH).
- **Surveyor guidance:** phase-3 guidance from 2022-10-24 and revised guidance from 2025-04-28.
- **The CMP rule change** of 2024-08-06.
- **The risk-based survey** (QSO-26-14-NH) from 2026-09-08, a new regime break that comes after all our labelled data.

---

## 7. External evidence

Source: `research/DEEP_RESEARCH.md` §2 and `research/RESEARCH_SOLUTIONS.md`.

### 7.1 Published ceilings

No published ML model predicts CMS G+ citations. Comparable inspector-judgement labels score low:

| Study | Result |
|---|---|
| Ontario nursing homes, next inspection (3 classes) | 40 % accuracy, called "weak" |
| England CQC (GP practices) | κ ≈ 0.1; accuracy at or below the majority class |
| England CQC (acute hospitals) | 24 % out of sample |
| US nursing-home COVID-outbreak GBM | AUC 0.72 |

High AUCs (0.93–0.97, in customs and tax) come from labels determined by documents, not by an inspector's judgement. **Our ROC 0.71 is at or above the published state of the art.** The estimated ceiling for this label is **ROC ~0.75–0.78, PR-AUC ~0.52–0.56**.

### 7.2 Label noise (GAO)

- In federal comparison surveys, federal teams found G+ deficiencies that state surveyors had missed in about 12–15 % of surveys, and in over 25 % in 9 states (GAO-10-70, GAO-08-517).
- In FY2007, 20 % of disputed deficiencies were deleted and 7 % downgraded.

### 7.3 State variance

- The G+ share of deficiencies, 2022–24, was 17.1 % in Kentucky, 15.3 % in Tennessee and 1.5 % in Nevada.
- Alabama went from 5.3 % to 18.3 % between 2023 and 2024.
- Our own data shows positive rates by state from 6.9 % to 62.1 % (states with ≥ 200 rows). The per-state calibration gap has a standard deviation of 7.4 points (§11).

### 7.4 What has worked elsewhere

- **Inspector-side and entity features.** The Chicago food-inspection model's top feature was the inspector cluster (odds ratio ~4.7). Customs-fraud graph features (DATE, GraphFC) added the most lift. CMS publishes no surveyor IDs, and our state-strictness proxy gave only +0.002 ROC.
- **Field deployments are judged by hit rate at a fixed budget, not by accuracy:**

| Domain | Hit-rate gain |
|---|---|
| EPA hazardous waste (randomized field test) | +79–82 % |
| Clean Water Act | 2×–7× |
| Boston restaurants (Yelp) | +25 % more violations found |

  Our top 2 % is 80 % precise against a 26 % base rate, a 3.1× lift.

- **Competition techniques** (AmEx, IEEE-CIS, Home Credit): tuning, neural networks and ensembling add +0.001–0.01. Big jumps come only from new context about the entity. That matches what we observed.

---

## 8. The decision, as a chain of reasoning

1. **The goal** was 80 % accuracy on `y_g12` with no leakage (the team's target, 2026-11-06). We agreed to report it honestly next to PR-AUC.
2. **Accuracy has a floor of 74.4 %** (majority class). To reach 80 %, the model must flag about 10 % of facilities at ≈ 78 % precision (§2.3).
3. **Our honest model reaches 76.4 %.** Its precision is 80 % in the top 2 %, 68 % in the top 5 % and 58 % in the top 10 %. No threshold, even one picked on test, exceeds about 76 %.
4. **The only score above 80 % was a leak.** Random-row splits gave 87.3 %, because the same facility in adjacent months shares both features and future citations. Splits by facility, by time with a purge, and by both all land at ROC 0.70–0.73 and accuracy 74.5–76.7 %.
5. **Models are not the bottleneck.** LR, LightGBM, CatBoost, ensembles, FT-Transformer, TabNet, MLP, DCN-v2 and DeepFM all fall within about 0.03 PR-AUC, and the simplest (LR, ensemble) are on top.
6. **Rows are not the bottleneck.** 6.7× more rows gave nothing (D3 vs D2/D1). None of the 11 row filters mattered. The model saturates at about 100k rows. Only *recency* helped, by +0.016 PR-AUC (exp9), and it is already in the final model.
7. **More features of the same kind are not the bottleneck.** Every addition was dropped or showed only a tiny real gain:
   - +120 engineered features: −0.007
   - 14 extra CMS tables (69 features): −0.002 on test
   - case-mix: +0.0017
   - PBJ daily staffing: +0.001
   - observable complaint, state and sibling proxies: +0.001–0.003 ROC each
8. **The label is driven by events our features can't observe.** 68 % of positives come only from complaint surveys, which are unscheduled. 32 % are first-timers with no history, and the model's recall on them is 6 %. Standard-survey timing is already learned (P(survey) PR-AUC 0.947).
9. **Even the missing information is not enough.** An oracle told the future number of complaint surveys reaches only 78.2 % accuracy (ROC 0.819, PR-AUC 0.584). The rest is surveyor judgement and label noise, documented by GAO.
10. **The missing information is not publicly available.** Uncited complaint surveys are not published. The Provider Info incident and complaint counts are citation-derived, changed definition in 2025-07 and were deleted in 2026-02. CMS complaint intake (ACTS) is not public.
11. **The literature agrees.** Comparable labels top out at ROC ~0.75–0.78. We are within about 0.05 of that, and no published model does better on a comparable target.
12. **Therefore** 80 % accuracy is not reachable honestly with public data. Further experiments (exp13 complaint/incident intake, exp14 NHSN, peer aggregates) could add about +0.01 ROC at best and would not change accuracy materially. **The team decided on 2026-11-07 not to run them, and closed modelling.**

---

## 9. What we deliver and how to read it

### 9.1 The deliverable

- **Model:** a mean ensemble of tuned LightGBM and logistic regression on the 133 gold features, retrained on the most recent 12 labelled months (§5.3).
- **Code:**
  - Experiment code: `experiments/rev002/exp8_all_tables.py`, `exp8b_refit_recent.py`, `exp9_recency.py`, `exp8_lr.py`.
  - Saved-model scripts for the earlier 75-feature ensemble: `feature_engineering/boost_spine2022/save_models.py`, `score_ensemble.py`, `calibrate.py`.
- **Model binaries are not included.** No binary was saved for the D5 recommended model; the D5 runs saved predictions only. The D3 binaries live outside this folder.

### 9.2 How the score should be read and used

| Use | How | Expected quality (test 2025-02, 14,643 facilities, base rate 25.6 %) |
|---|---|---|
| **Watch list, top 2 %** | ~293 facilities | **80.5 % cited within 12 months (3.1× lift)** |
| Watch list, top 5 % | ~732 facilities | 68.2 % (2.7×) |
| Watch list, top 10 % | ~1,464 facilities | 57.9 % (2.3×) |
| Ranking | sort facilities by score | ROC 0.709, PR-AUC 0.481 (random 0.256) |
| Probability | ensemble mean | well calibrated within a period (facility-split ECE 0.6–0.7 pts; D3 ECE 2.2 pts, Brier 0.172). Over-predicts the top decile when forecasting further ahead (predicted 0.71 vs actual 0.58), so recalibrate on the latest labelled months |
| Yes/no | threshold 0.5665 | accuracy 76.4 % vs 74.4 % for "no citation"; not the recommended use |

**Do not** present the score as an 87 % accurate classifier. That number comes from a leaky split.

**Do** present it as a ranked watch-list with honest lift, alongside the 74.4 % majority baseline.

### 9.3 Operating rules

These come from `results/MODEL_CARD.md` §9.

1. Retrain monthly, or at least quarterly, on the latest 12 labelled months.
2. Monitor ROC and PR-AUC per month. From the per-month decay in §4.5 (0.733 → 0.695 over 7 months), expect roughly 0.005 ROC lost per month ahead.
3. Recalibrate on the latest labelled months.
4. Check state slices before release (§11).
5. Keep the extra CMS tables (ownership, fire safety) in silver as product context, not as model inputs.

---

## 10. What would change the conclusion

| New information | Why it could matter | Status |
|---|---|---|
| **Complaint-intake data** (CMS ACTS / iQIES, state complaint logs) | It is exactly the oracle's missing variable: complaint arrival is worth about +0.11 ROC | Not public. NY ombudsman data covers one state only. Would need a FOIA request or a state pilot |
| **CDC NHSN weekly long-term-care data** (staff and supply shortage flags, outbreaks, reporting gaps; CCN-keyed, weekly, free) | The most promising untested public source of *leading* signals for complaints | Feasibility check (exp14) designed, not run |
| Deficiency narrative text (monthly CMS statement-of-deficiencies zips, ~600 MB each) | Near-harm wording in D/E citations | Expected gain modest; effort high |
| Chain / affiliated-entity history (Provider Info `Chain ID` from 202507, `Affiliated Entity ID` from 202306; PECOS from 2022-09) | Entity-level priors helped in other domains | Sibling-owner proxy gave +0.000 ROC; chain ID too late for 2023 training |
| Surveyor or team identity | The top feature in comparable inspection models | Not published by CMS |
| A new survey regime (risk-based survey QSO-26-14-NH from 2026-09-08) | Could change both base rate and predictability | Needs labelled months after 2026-09, which will not exist before about 2028 because of the 12 + 8 month label lag |

**Team decision (2026-11-07):** the team will not pursue complaint-intake or NHSN data within this project. The evidence says that even perfect complaint information would not reach 80 % accuracy (78.2 % oracle), so these sources could not meet the stated target. They remain the right next step if the target is reframed as lift at a fixed review budget.

---

## 11. State-wise bias

State is the strongest single variable in this problem:

- **Positive rates** range from 6.9 % to 62.1 % across states with ≥ 200 rows (`reports/rev002_gold2022_fe_eda_report.md` A1).
- **Feature importance:** `state` is the top LightGBM gain feature of the recommended 133-feature model, and its target encoding `state_te` is the top permutation feature of the 75-feature set.
- **Calibration error is large at state level.** The per-state gap between actual and predicted rate has a standard deviation of 7.4 points. One example state is under-predicted by 17 points and another over-predicted by 19 (`research/DEEP_RESEARCH.md` §1).
- **Part of this is genuine.** GAO and KFF document real differences between state survey agencies in how often they cite G+ (§7.3). Part of it is a fairness risk: insurers would price facilities partly by where the surveyor works.

The full state-wise bias audit is in **[`state_bias/STATE_BIAS_REPORT.md`](state_bias/STATE_BIAS_REPORT.md)**: basic metrics, group fairness, calibration, statistical tests, multilevel models, slice discovery, a counterfactual retrain and mitigation. Its findings:

- **The bias is in calibration, not ranking or precision.** The per-state O−E standard deviation is 7.6 points across 34 eligible states, and 18 states are significantly miscalibrated after FDR correction. Precision among flagged facilities is similar across states. Facility size and ownership add no bias beyond state.
- **Cause:** the model anchors on each state's *past* surveyor strictness (`state` is the #1 gain feature, `state_gplus_rate_12m` #3). It misses states whose regime changed (r = 0.91 between the error and that drift).
- **Removing state information does not help.** PR-AUC falls 0.477 → 0.463, and 17 states are still miscalibrated.
- **Honest recalibration helps only a little.** Fitted on matured 2024-01 labels, it narrows the spread from 7.6 to 6.5 points. Fitting on 2024-11 (2.5 points) would need labels not available at scoring time.
- **Recommendation:** monitor per-state calibration monthly and treat absolute probabilities by state with caution. Error-rate parity across states cannot be reached with one national threshold, because state base rates range from 9 % to 58 %.

Read it before any release that shows scores or probabilities by state.

---

## 12. Appendix

### 12.1 Glossary

| Term | Meaning |
|---|---|
| **G+ citation** | A CMS health deficiency at scope/severity G–L: G–I is actual harm, J–L is immediate jeopardy |
| `y_g12` | 1 if a new G+ citation (survey date after the cutoff, not removed by CMS) falls within 12 months of month *t*; `y_g6` is the 6-month version, `y_pen12` the penalty version |
| CCN | CMS Certification Number, the facility ID |
| Standard survey | the scheduled recertification inspection (due within ~15 months) |
| Complaint survey | an unscheduled inspection triggered by a complaint or facility-reported incident; published only if cited |
| FRI | facility-reported incident |
| Vintage | the CMS data month, taken from the Processing Date inside the file |
| Spine | the facility × month grid gold is built on (from 2022-01; 2021 is look-back only) |
| Bronze / silver / gold | raw as published / cleaned per vintage + event ledgers / model-ready facility-month table |
| Posting buffer B | months to wait before a label is final (8 for citations) |
| Immature / censored | label window not yet fully observed / facility left the data before t + H + B |
| Purge gap | months dropped between training and evaluation so that their label windows do not overlap |
| PR-AUC | area under the precision–recall curve; no-skill value = base rate |
| ROC-AUC | probability that a random positive is ranked above a random negative; no skill = 0.5 |
| P@k %, lift | precision among the top k % ranked; lift = P@k ÷ base rate |
| Brier, ECE | mean squared error of probabilities; expected calibration error (mean gap between predicted and observed rate per bin) |
| PSI | population stability index; > 0.25 means a major distribution shift |
| HPRD | nurse staffing hours per resident day |
| PBJ | Payroll-Based Journal, CMS's daily staffing submissions |
| Keep rule | adopt a change only if mean Δ PR-AUC across 3 facility-grouped folds is > 0 and > 1 std |
| Facility-clustered bootstrap | confidence interval from resampling whole facilities (300 times) |
| Oracle | a diagnostic model given future information; it shows the value of missing information and is never deployable |
| NHSN | CDC National Healthcare Safety Network (weekly long-term-care reporting) |
| ACTS / iQIES | CMS complaint-tracking and survey systems (not public) |

### 12.2 Recommended model vs other "best" numbers

| Setting | PR-AUC | ROC | Acc. | Why it differs |
|---|---|---|---|---|
| D1 LR, subset (test 2024-10) | 0.492 | 0.708 | — | different test month, quarterly data |
| D2 ensemble, Gold 2023 (test 2024-10/11) | 0.490 | 0.709 | — | 2-month test; no train→calib gap |
| D6 facility split | 0.503 | 0.733 | 76.4 % | test facilities share months with train (not a forecast) |
| **D5 recommended (test 2025-02)** | **0.481** | **0.709** | **76.4 %** | honest forecast, newest gold |

### 12.3 Discrepancies between sources (reported, not resolved)

1. **Accuracy of the 2023-trained ensemble.** exp9 R3 (2023-01..11, PR-AUC 0.4802) gives 76.4 % with the exp8 calib threshold (0.5665). exp12's "old ensemble" on the same window and the same PR-AUC gives 75.80 % with a threshold from out-of-fold training predictions (0.5046). The model is the same; the threshold rule differs.
2. **exp7 test PR-AUC.** The seed-mean of per-seed test PR-AUC is 0.4675 → 0.4691 (`test_prauc_mean_std`). The metric on seed-averaged scores is 0.4686 → 0.4705 (`test_metrics_seedmean`). The bootstrap Δ is +0.0018 in the JSON and +0.0017 in the reports.
3. **Unsurveyed rows.** The project log reports 21.7 % of negatives without a health survey (excluding infection-control) and 14.5 % of test rows unsurveyed including infection-control (2024-10/11 test). DEEP_RESEARCH reports 2,261 of 14,643 = 15.4 % (2025-02). The model card summarises this as "15–22 %". The definitions and months differ.
4. **Oracle "nothing" row vs diagnostics "nothing" row.** The oracle baseline is ROC 0.707 / PR-AUC 0.472 / 76.1 %, while the proxy diagnostics baseline is 0.7092 / 0.4800. Both are described as built on exp12 old-ensemble scores of 2025-02. The source does not explain the gap; it is probably the different stacking procedures.
5. **Star ratings.** `results/MODEL_CARD.md` and `results/ALL_TABLES_REPORT.md` say star ratings are never features. The 133-feature registry does include `staffing_rating` and `qm_rating` as model features. Only `health_inspection_rating` and `overall_rating` are excluded. The model card qualifies its statement with "except per the registry's model_use".
6. **75-feature selection.** The gold-2023 holdout preferred v1 (133) over the 75 selected (`accept selected: False`), but the 75-feature set was used for the D2/D3 models. The 75-feature YAML says the selection came from the 2023-01..2023-09 DEV.
7. **PBJ feature count.** Early notes say 31 PBJ features; the built and evaluated set has 24.
8. **Empty months.** The project log lists 6 empty months from 2022 on. `results/MISSING_MONTHS_REPORT.md` lists 7, adding 2021-12, which is history only.
9. **Test windows differ across datasets** (2024-10; 2024-10/11; 2025-02). D5 numbers are not directly comparable with D1–D4.

### 12.4 Index of every file in this folder

All supporting files are **copies**. The originals are left in place. Copied markdown reports are **historical copies**: their internal file references point to their original locations, some of which are outside the repository (`MLOPS project/...`). The logs `experiments/rev003_pbj/exp12_pbj.log` and `feature_engineering/selection_gold2023/select.log` contain machine-specific absolute paths. They are left as-is.

**Top level**

| File | What it is |
|---|---|
| `README.md` | folder layout |
| `report.md` | this report |
| `final_model/` | re-run of the recommended model with complete test metrics (`final_model_metrics.py`, `final_model_metrics.json`, `reliability.csv`, `test_predictions.csv`); see §13 |
| `state_bias/` | state-wise bias audit, produced separately (`STATE_BIAS_REPORT.md`, `state_bias_audit.py`, `results.json`, `figures/`, `tables/`) |

**`research/`**

| File | What it is |
|---|---|
| `DEEP_RESEARCH.md` | error analysis, oracle, proxy diagnostics, literature, verified data-source addendum |
| `RESEARCH_SOLUTIONS.md` | ranked shortlist of ceiling-breaking ideas (competitions, domain, label reformulation) |

**`results/`**

| File | What it is |
|---|---|
| `ALL_RESULTS_TABLE.md` | every model × dataset in one table (historical copy) |
| `RESULTS_REPORT.md` | condensed version of the same (historical copy) |
| `ALL_TABLES_REPORT.md` | exp8–exp11: all CMS tables, recency, 70/15/15 splits (historical copy) |
| `MODEL_CARD.md` | model card of the recommended model (historical copy) |
| `MISSING_MONTHS_REPORT.md` | why 7 CMS months are empty (historical copy) |
| `comparison.md`, `comparison.csv` | rev002 exp1–exp5 ranking, with and without the purge gap |
| `summaries/exp3__*.json`, `exp4__*.json`, `exp5__*.json` | per-model test/calib metrics, gap runs (D4) |
| `summaries/exp6__*.json` | survey × harm split (base, A×B, B surveyed, summary) |
| `summaries/exp7__*.json` | case-mix (base, casemix, summary with CIs and gain ranks) |
| `summaries/exp8__summary.json` | all-tables run: family deltas, top-k, train-only test |
| `summaries/exp8b__summary_refit.json` | refit to 2023-11; accuracy, best possible accuracy |
| `summaries/exp9__summary_recency.json` | recency arms R1–R5 with CIs (source of the recommended model's numbers) |
| `summaries/exp11__summary.json` | 70/15/15 splits: full statistics, CIs, confusion matrices, calibration |
| `summaries/exp12__summary.json`, `exp12__gain_pbj_model.csv` | PBJ experiment: CV keep rule, tests, CIs; PBJ gain ranks |
| `summaries/gain_all_features.csv`, `gain_selected_tuned.csv`, `selected_features.txt` | exp8 gain rankings and the 100 selected features |
| `row_filters/*.csv`, `row_filters/experiment_log.md` | row-filter coverage, CV per fold, CV summary, test results, log |

**`reports/`** (all historical copies)

| File | What it is |
|---|---|
| `rev001_subset_fe_eda_report.md` | EDA + feature engineering + validation on the subset (D1) |
| `rev002_gold2022_fe_eda_report.md` | the same on Gold 2022 (D3) with calibration and rev002 gap experiments (identical to `MLOPS project/report_rev002.md`) |
| `row_filters_report.md` | row-filter experiment R0–R11 |
| `exp12_pbj_report.md`, `exp12_pbj_README.md` | PBJ experiment report and status |

**`experiments/`**

| File | What it is |
|---|---|
| `rev002/exp_common.py` | data loading, split, purge gap (`apply_calib_gap`), metrics, gold manifest check |
| `rev002/feature_registry_rev002.yaml` | the 133-feature registry used by all experiments |
| `rev002/feature_transform_v2.py` | B1–B8 FeatureTransformer (exp5) |
| `rev002/exp6_survey_split.py` … `exp11_full_stats.py`, `exp8_lr.py`, `exp8b_refit_recent.py` | experiment scripts behind §4.4–§4.5 |
| `rev002/compare_results.py`, `run_nb.py`, `gold_manifest.json` | ranking builder, RAM-guarded notebook runner, gold completeness manifest. The exp1–exp5 notebooks are not copied: they are large executed notebooks, and their results are in `results/` |
| `rev003_pbj/exp12_pbj.py`, `pbj_features.py`, `pbj_download.py`, `pbj_catalog.csv`, `exp12_pbj.log` | PBJ download, point-in-time feature build, experiment, run log |
| `row_filters/make_notebook.py` | builds the row-filter notebook |

**`feature_engineering/`**

| File | What it is |
|---|---|
| `gold_features/*.py` | gold feature-family code: citations, survey_timing, enforcement, facility_staffing, quality, context, plus the 7 all-tables families, `reader.py` (point-in-time reader), `common.py` |
| `feature_registry_202.yaml` | current registry: 133 + 69 features, with formula and first vintage |
| `feature_set_selected75.yaml` | the 75-feature selected set (D2/D3) |
| `feature_set_v2_subset66.yaml` | the 66-feature subset selection, with holdout decision |
| `select_spine2022.py`, `calibrate.log`, `save_models.log` | Gold 2022 selection script; calibration and model-save logs |
| `selection_gold2023/` | `select_full_gold.py`, `select.log`, `topk.log`, `boost_models.py`, `boost.log`: where the 75 features and the D2 scores come from |
| `boost_spine2022/` | `lgbm_best_params.json` (the tuned parameters of the final LightGBM), `optuna_trials.csv`, `boost_models.py`, `boost_test.csv`, `boost_holdout.csv`, `calibrate.py`, `save_models.py`, `score_ensemble.py` |
| `eda_spine2022/*.csv` | Part A EDA reports on Gold 2022 (target, profile, signal, correlation, VIF, PSI, rare levels, slices) |

**Not included, by design:** data files, parquet, model binaries, prediction CSVs, executed notebooks, `data/`, environment configuration files, internal session notes, and the team playbook file `FE_Rows.md`. Its rules are summarised in §5.5.

---

## 13. Final model: description and complete performance

### 13.1 What the model is

| Item | Value |
|---|---|
| Name | Precursor G+ risk model, recommended configuration (exp9 arm **R4**, "2023 only") |
| Task | P(facility receives a G-or-higher CMS health citation within the next 12 months), scored per facility per month |
| Model | **Mean ensemble** of two models, each scoring a probability: (1) tuned **LightGBM** and (2) **logistic regression**. Final score = (p_LightGBM + p_LR) / 2 |
| Features | the **133 gold features** (registry `experiments/rev002/feature_registry_rev002.yaml`; families in §5): citation history, survey timing, enforcement, facility and staffing, quality measures, state context. Point-in-time at month t. No star ratings |
| LightGBM hyperparameters (Optuna, `feature_engineering/boost_spine2022/lgbm_best_params.json`) | objective binary, learning_rate 0.0102, n_estimators 1,265, num_leaves 34, max_depth 3, min_child_samples 870, subsample 0.634 (freq 1), colsample_bytree 0.335, reg_alpha 0.0068, reg_lambda 9.48, min_split_gain 0.435, cat_smooth 2.71, random_state 42 |
| Logistic regression | numeric: median imputation + missing-indicator + standard scaling; categorical: one-hot (min frequency 50); L2, C = 0.3, max_iter 3,000 |
| Training data | 119,165 labelled facility-months, sampled (seed fixed) from 2023-01..2023-11 (162,276 rows); training base rate 26.7 % |
| Test data | one future month, **2025-02**: 14,643 facilities, 3,750 positives (base rate **25.6 %**); scored once |
| Decision threshold (for accuracy) | 0.5665, chosen on the calibration split before testing (exp8), never on test |
| Source of numbers | `final_model/final_model_metrics.json` (re-run of exp9 arm "R4 2023 only, size of R1", reproduction exact); original run `results/summaries/exp9__summary_recency.json`, code `experiments/rev002/exp9_recency.py` |

### 13.2 Complete performance on the test month (2025-02)

Measured by `final_model/final_model_metrics.py`, a deterministic re-run of exp9 arm R4. The reproduction check passed
exactly (PR-AUC 0.481310 = saved value). All 95 % CIs come from 1,000 bootstrap replicates over facilities; the test
month has one row per facility. Outputs: `final_model/final_model_metrics.json`, `final_model/reliability.csv`,
`final_model/test_predictions.csv`.

**Ranking and probability quality (threshold-free)**

| Metric | Value | 95 % CI | Reference | Reading |
|---|---|---|---|---|
| PR-AUC (average precision) | **0.4813** | 0.465–0.499 | random = 0.2561 | 1.88× random |
| ROC-AUC | **0.7092** | 0.700–0.719 | random = 0.50 | a random positive outranks a random negative 71 % of the time |
| Brier score | **0.1689** | 0.166–0.173 | 0.1905 for always predicting the base rate | Brier skill score **+0.113** |
| ECE (10 equal-mass bins) | **0.027** | 0.022–0.035 | 0 = perfect | mean predicted 26.3 % vs observed 25.6 % |
| Calibration slope / intercept | 0.84 / −0.19 | — | 1 / 0 | slightly over-confident at the extremes |
| Δ PR-AUC vs original 2022 window (R1) | +0.011 to +0.022 | — | — | recency gain is significant |

The two members on their own: LightGBM PR-AUC 0.4791 / ROC 0.7080 / Brier 0.1701; logistic regression 0.4781 /
0.7073 / 0.1693. The ensemble beats both.

**Precision at the top of the ranking (the watch-list view)**

| Flagged group | Facilities flagged (k) | Score cut-off | True positives | False positives | **Precision (P@k)** | 95 % CI | Lift | Recall | F1 | Specificity |
|---|---|---|---|---|---|---|---|---|---|---|
| Top 1 % | 146 | 0.787 | 126 | 20 | **86.3 %** | — | 3.37× | 3.4 % | 0.065 | 99.8 % |
| **Top 2 %** | 293 | 0.719 | 236 | 57 | **80.5 %** | 76.1–85.0 % | **3.15×** | 6.3 % | 0.117 | 99.5 % |
| **Top 5 %** | 732 | 0.602 | 499 | 233 | **68.2 %** | 64.6–71.6 % | **2.66×** | 13.3 % | 0.223 | 97.9 % |
| **Top 10 %** | 1,464 | 0.511 | 848 | 616 | **57.9 %** | 55.3–60.7 % | **2.26×** | 22.6 % | 0.325 | 94.3 % |
| Top 20 % | 2,929 | 0.409 | 1,462 | 1,467 | 49.9 % | — | 1.95× | 39.0 % | 0.438 | 86.5 % |
| All facilities (no model) | 14,643 | — | 3,750 | 10,893 | 25.6 % | — | 1.00× | 100 % | — | — |

**At the decision threshold (0.5665, chosen on the calibration split, never on test)**

| | Predicted cited | Predicted not cited | Total |
|---|---|---|---|
| **Actually cited** | 630 (TP) | 3,120 (FN) | 3,750 |
| **Not cited** | 337 (FP) | 10,556 (TN) | 10,893 |
| **Total** | 967 (6.6 % flagged) | 13,676 | 14,643 |

**How the confusion matrix is built.** A facility counts as "predicted cited" when its ensemble score (mean of the
LightGBM and logistic-regression probabilities) is at least 0.5665. Each of the 14,643 facilities in the test month is
then compared with its actual label (a G+ health citation in the next 12 months, or not). The cut-off was chosen in
exp8 (`experiments_rev002/exp8_all_tables.py`, `thresholds()`):

1. Score the calibration months (2023-09..2023-11), not the test month.
2. Build 197 candidate cut-offs from the 1st to the 99th percentile of those scores.
3. Keep the cut-off with the highest **accuracy** on the calibration months: 0.5665. (exp8 also records an F1-optimal
   cut-off, which is not used here.)
4. Apply it once, unchanged, to the test month (2025-02), so these test metrics are not tuned on test.

Two consequences:

- **Accuracy-optimal means conservative.** With 74 % negatives, the cut-off that maximises accuracy sits high. Only
  6.6 % of facilities are flagged, so precision (65 %) and specificity (97 %) are high and recall is low (17 %). A
  different goal (F1, a recall target, or a fixed review budget) would need a different cut-off. That is why the
  top-k table above is the main way to read the model.
- **The cut-off was not re-tuned for this model.** 0.5665 was selected for the exp8 ensemble, which was trained on
  2022-01..2022-08. The final model (exp9 R4) is trained on a sample of 2023-01..2023-11, a window that includes the
  calibration months. Re-tuning on those months would reuse training rows, so the exp8 cut-off was kept as is. It is
  still a fair, pre-registered cut-off for the test month, but not one optimised for R4. A cut-off tuned on clean
  held-out months for R4 could give a slightly different accuracy (see also §12.3). The top-k precision figures
  (P@2/5/10 %) use no threshold and are not affected.

| Metric | Value |
|---|---|
| Accuracy | **76.39 %** (95 % CI 75.7–77.1 %); majority-class baseline 74.39 % → **+2.0 points** |
| Precision (PPV) | 65.1 % |
| Recall (sensitivity, TPR) | 16.8 % |
| Specificity (TNR) | 96.9 % |
| Negative predictive value | 77.2 % |
| F1 | 0.267 |
| False-positive rate / false-negative rate | 3.1 % / 83.2 % |
| Balanced accuracy | 56.9 % |
| Matthews correlation coefficient | 0.241 |

**Reliability (10 equal-mass bins, about 1,465 facilities each)**

| Bin | Mean predicted | Observed rate | Gap (obs − pred) |
|---|---|---|---|
| 1 (lowest) | 5.9 % | 6.4 % | +0.5 |
| 2 | 9.3 % | 12.5 % | +3.2 |
| 3 | 12.4 % | 14.7 % | +2.2 |
| 4 | 15.8 % | 18.0 % | +2.2 |
| 5 | 19.8 % | 21.6 % | +1.8 |
| 6 | 24.3 % | 24.5 % | +0.2 |
| 7 | 29.8 % | 27.5 % | −2.4 |
| 8 | 36.7 % | 31.1 % | −5.6 |
| 9 | 45.7 % | 41.9 % | −3.8 |
| 10 (highest) | 63.0 % | 57.9 % | −5.1 |

Low-risk facilities are slightly under-predicted and high-risk ones over-predicted by about 4–6 points. This matches
the calibration slope of 0.84 and the earlier finding that the top decile is over-predicted when forecasting further
ahead (§9).

### 13.3 How to read it

- **This is a ranking and watch-list tool, not an accurate yes/no classifier.**
  - Accuracy is only 2 points above "always no". At its accuracy-optimal threshold the model flags just 6.6 % of
    facilities and catches 16.8 % of positives (F1 0.27).
  - The reasons are in §2 and §6: the label depends on unscheduled complaint surveys and surveyor judgement.
- **Its value is at the top.**
  - Of the 293 highest-scored facilities, 4 in 5 (80.5 %, CI 76–85 %) receive a G+ citation within 12 months, 3.1× the
    base rate.
  - The top 1 % is 86 % precise, and the top 10 % still has more than double the base rate.
- **The top lists catch a minority of all positives** (22.6 % within the top 10 %, 39 % within the top 20 %). About a
  third of positives are first-time offenders with clean histories, which no public signal anticipates (§6).
- **Probabilities are usable nationally** (ECE 2.7 points, Brier skill +0.11), with mild over-confidence at the top.
  They are off by about ±6 points at **state** level (§11 and `state_bias/STATE_BIAS_REPORT.md`). Use national rank
  cut-offs for lists.
- **Choose the operating point by use case:**

  | Use case | Cut-off | Precision | Recall |
  |---|---|---|---|
  | High-confidence watch list | top 2 % | 80 % | 6 % |
  | Balanced review queue | top 10 % | 58 % | 23 % |
  | Broad screening | top 20 % | 50 % | 39 % |
