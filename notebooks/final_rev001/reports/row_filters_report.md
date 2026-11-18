# rev002: fine-tuning the training rows (spine-2022 gold, `y_g12`)

**Goal:** find training rows that hurt the model and test whether removing them improves performance.
**Playbook followed:** `FE_Rows.md` in this folder (one change per experiment, leakage checklist, metric set, "Δ PR-AUC > 1 std" keep rule, experiment log format).
**Status:** complete. Full run 2026-11-03 19:59 → 20:30 (31 min), no errors. Notebook: `finetune_rows.ipynb` (built by `make_notebook.py`); outputs in `outputs/`.

---

## 1. Starting point (summary of the repo state)

| Item | Value |
|---|---|
| Target | `y_g12`: new G+ (harm) citation within 12 months of the cutoff. **Not changed.** |
| Data | gold, spine from 2022 (746,024 rows, 2022-01 → 2026-08); DEV = train + calib, 296,476 labeled rows (2022-01 → 2023-09) |
| Gold split | train 2022-01 → 2023-06, calib 2023-07 → 09, purge 2023-10 → 2024-09, test 2024-10 → 11. **Not changed.** |
| Features | the 75 selected features (`feature engineering run/spine2022/feature_set_selected75.yaml`). **Fixed** in this experiment. |
| Model | tuned LightGBM (`spine2022/boost/lgbm_best_params.json`: depth 3, 34 leaves, 1,265 trees, lr 0.010), parameters fixed |
| Current test result | PR-AUC 0.487, ROC-AUC 0.708 (LightGBM alone); ensemble 0.488 |

---

## 2. Why remove rows at all

The rev002 Part A analysis (`notebooks/data preprocessing03_11_rev002/report.md`) found several kinds of training rows that may teach the model the wrong thing for the period it will score:

1. **Drift:** 35 features have PSI > 0.25 against 2022-01, led by the COVID-era inspection backlog. These rows describe a regime that no longer exists.
2. **Empty or stale history:** rows where the inspection-history features are empty or years old carry little signal and may add noise.
3. **Missing blocks:** some rows have the whole staffing or quality-measure block unreported.
4. **Different populations:** very small or hospital-based homes behave differently and are few.
5. **Redundancy:** consecutive months of the same facility are near-copies (the 24-month counts change slowly), so the model sees the same home about 21 times.
6. **Label noise:** some labels may be wrong or unpredictable (e.g. a citation from a random complaint survey at an otherwise clean home).

Each reason becomes **one filter, tested on its own** (playbook rule 0.3).

---

## 3. Row filters and the reason for each

Every filter removes rows **from training only**. Validation and test rows are never filtered, so a filter cannot score better by making the evaluation easier.

| ID | Filter (rows removed from training) | Reason | Evidence |
|---|---|---|---|
| R0 | none (baseline) | reference | — |
| R1 | `drop_2022H1`: months 2022-01 → 2022-06 | Peak COVID inspection backlog. These months are the least like 2024. | A5: `n_ic_surveys_12m` PSI 5.2; `days_since_last_std` 1.5; `days_to_std_due` 1.4 vs 2022-01 |
| R2 | `drop_2022`: all of 2022 | Same reason, stronger; trains on 2023 only. Tests whether more history helps at all. | rev002: 296k DEV rows gave the same test PR-AUC (0.488) as 133k (2023-only) |
| R3 | `drop_no_std`: no standard survey on record | The history features (citations, severity, timing) are empty, so the row has almost no signal. | A3: inspection history is the strongest signal block ; **not run:** the gold has no such labeled rows (0 of 487,459; only 1 unlabeled score row), so the data is already clean on this point |
| R4 | `drop_stale_insp`: last inspection more than 2 years old | Stale history, mostly a backlog artifact; the features describe the past, not current care. | A5: survey-timing features drift strongly |
| R5 | `drop_missing_staffing`: `hprd_total` missing | The staffing block is unreported, so imputed values add noise. | A2: staffing coverage ~97 %; turnover 78 % |
| R6 | `drop_missing_qm`: ≥ 10 quality measures missing | The quality block is unreported. | A2: QM footnotes `missing_not_submitted` |
| R7 | `drop_tiny_or_hospital`: < 20 beds or hospital-based | A different population with few rows; rates per bed are unstable. | A7: risk depends strongly on size (16.6 % for < 50 beds vs 34.0 % for 200+) |
| R8 | `drop_new_facility`: facility younger than 2 years | Little history; 24- and 36-month windows are incomplete. | B6: trend and window features need history |
| R9 | `quarterly_thin`: keep 1 month in 3 per facility | Consecutive months are near-copies; tests whether the redundancy hurts or helps. | subset run (quarterly) scored about the same as monthly |
| R10 | `noise_2pct`: the 2 % of training rows the model gets most confidently wrong | Possible label noise ("confident learning"). | — |
| R11 | `noise_5pct`: the same, 5 % | Stronger version of R10. | — |

The noise score of R10/R11 is computed **inside each training fold only** (inner 3-fold out-of-fold predictions), so the validation facilities never influence which rows are removed.

---

## 4. Evaluation design and leakage checklist

**Cross-validation (decides keep or drop):** 3 folds grouped by facility (CCN).
- **Train:** the fold's training facilities, months 2022-01 → 2023-06, with the filter applied.
- **Validate:** the fold's held-out facilities, months 2023-07 → 2023-09 (later than all training months), unfiltered.

**Test (reported only):** train on all DEV (2022-01 → 2023-09, filtered), score 2024-10 → 11 (29,324 rows, unfiltered).

**Metrics (playbook section 1, primary first):** PR-AUC, ROC-AUC, Recall@5 %, Recall@10 %, F1 at a threshold tuned on validation (for test: tuned on the pooled CV validation predictions, never on test), Brier score, accuracy.

**Keep rule (playbook section 9):** a filter is kept only if the mean Δ PR-AUC across folds is > 0 **and** larger than 1 std of the per-fold Δ (paired against the baseline on the same fold).

| Leakage check | Status |
|---|---|
| Group by facility | ✅ CV folds are grouped by CCN; no facility is in both train and validation |
| Respect time | ✅ validation months (2023-07 → 09) are later than all training months; test is 2024 |
| No target-derived columns | ✅ star ratings excluded; the 75 features passed the A8 leakage scan |
| Transforms fit on train only | ⚠️ The feature transformer (clips, `state_te`, PCA) was fitted once on all of DEV, which includes the CV validation facilities. It is identical for every filter, so the **comparison between filters stays fair**, but absolute CV scores are slightly optimistic. The test months are outside DEV, so test is clean. |
| State aggregates from training data only | ✅ `state_gplus_rate_12m` uses only past months (pipeline feature); `state_te` is out-of-fold on DEV |

---

## 5. Results

### 5.1 How many rows each filter removes (all DEV rows, 296,476; positive rate 25.5 %)
| ID | Filter | Rows removed | Share of DEV | Positive rate of removed rows |
|---|---|---|---|---|
| R1 | drop_2022H1 | 89,459 | 30.2 % | 24.3 % |
| R2 | drop_2022 | 163,644 | 55.2 % | 24.6 % |
| R3 | drop_no_std | 0 | 0 % | — (not run) |
| R4 | drop_stale_insp | 64,195 | 21.7 % | 21.2 % |
| R5 | drop_missing_staffing | 9,389 | 3.2 % | 23.2 % |
| R6 | drop_missing_qm | 7,786 | 2.6 % | **9.1 %** |
| R7 | drop_tiny_or_hospital | 11,430 | 3.9 % | **12.7 %** |
| R8 | drop_new_facility | 1,911 | 0.6 % | 20.8 % |
| R9 | quarterly_thin | 192,647 | 65.0 % | 25.6 % |
| R10 / R11 | noise 2 % / 5 % | 2 % / 5 % of each training set | — | — |

R6 and R7 remove groups with a much lower risk than average (homes that don't report quality measures; tiny and hospital-based homes). They are different populations, as expected.

### 5.2 Cross-validation (decides): Δ PR-AUC vs baseline on the same fold
Baseline fold PR-AUC: 0.5216 / 0.5371 / 0.5153 (mean **0.5246 ± 0.0112**). About 168k training rows and about 14.7k validation rows (2023-07 → 09) per fold.

| ID | Filter | Fold 1 | Fold 2 | Fold 3 | Mean Δ ± std | Keep rule (mean > 0 and > 1 std) |
|---|---|---|---|---|---|---|
| R1 | drop_2022H1 | +0.0013 | +0.0027 | −0.0006 | +0.0011 ± 0.0017 | no |
| R2 | drop_2022 | +0.0018 | +0.0027 | −0.0032 | +0.0004 ± 0.0032 | no |
| R4 | drop_stale_insp | −0.0004 | −0.0001 | −0.0003 | −0.0003 ± 0.0002 | no |
| R5 | drop_missing_staffing | −0.0010 | −0.0001 | −0.0002 | −0.0005 ± 0.0005 | no |
| **R6** | **drop_missing_qm** | **+0.0001** | **+0.0001** | **+0.0003** | **+0.0002 ± 0.0001** | **yes (formally)** |
| R7 | drop_tiny_or_hospital | +0.0005 | −0.0004 | +0.0017 | +0.0006 ± 0.0011 | no |
| R8 | drop_new_facility | −0.0004 | +0.0002 | +0.0007 | +0.0002 ± 0.0005 | no |
| R9 | quarterly_thin | −0.0011 | +0.0003 | −0.0001 | −0.0003 ± 0.0007 | no |
| R10 | noise_2pct | −0.0014 | −0.0003 | −0.0003 | −0.0007 ± 0.0006 | no |
| R11 | noise_5pct | −0.0019 | −0.0035 | −0.0017 | −0.0023 ± 0.0010 | no (hurts) |

### 5.3 Test 2024-10 → 11 (reported only; 29,324 rows, base rate 26.4 %)
| ID | Filter | Train rows | PR-AUC | Δ | ROC-AUC | Recall@5% | Recall@10% | F1 | Brier | Accuracy |
|---|---|---|---|---|---|---|---|---|---|---|
| R0 | baseline | 296,476 | 0.4870 | — | 0.7082 | 0.128 | 0.221 | 0.491 | 0.1726 | 0.635 |
| R1 | drop_2022H1 | 207,017 | 0.4873 | +0.0002 | 0.7076 | 0.127 | 0.223 | 0.492 | 0.1727 | 0.628 |
| R2 | drop_2022 | 132,832 | **0.4879** | +0.0009 | 0.7070 | 0.130 | 0.221 | 0.492 | 0.1728 | 0.654 |
| R4 | drop_stale_insp | 232,281 | 0.4857 | −0.0013 | 0.7074 | 0.126 | 0.221 | 0.490 | 0.1729 | 0.626 |
| R5 | drop_missing_staffing | 287,087 | 0.4876 | +0.0006 | 0.7082 | 0.128 | 0.221 | 0.491 | 0.1727 | 0.624 |
| R6 | drop_missing_qm | 288,690 | 0.4872 | +0.0001 | 0.7082 | 0.128 | 0.221 | 0.492 | 0.1726 | 0.628 |
| R7 | drop_tiny_or_hospital | 285,046 | 0.4868 | −0.0002 | 0.7077 | 0.128 | 0.222 | 0.491 | 0.1728 | 0.623 |
| R8 | drop_new_facility | 294,565 | 0.4866 | −0.0005 | 0.7079 | 0.128 | 0.220 | 0.490 | 0.1726 | 0.635 |
| R9 | quarterly_thin | 103,829 | 0.4867 | −0.0003 | 0.7078 | 0.127 | 0.220 | 0.491 | 0.1727 | 0.632 |
| R10 | noise_2pct | 290,546 | 0.4866 | −0.0004 | 0.7073 | 0.129 | 0.222 | 0.492 | 0.1741 | 0.631 |
| R11 | noise_5pct | 281,652 | 0.4860 | −0.0011 | 0.7060 | 0.128 | 0.222 | 0.491 | **0.1780** | 0.620 |

F1 and accuracy use the threshold that maximizes F1 on the pooled CV validation predictions, never on test. Recall@k% is the share of all positives found in the top k %: the top 10 % can hold at most 10 / 26.4 = 38 % of the positives.

### 5.4 What the results say
1. **No row filter changes performance in a meaningful way.** Every CV change is within ±0.003 PR-AUC, and every test change within ±0.0013. The baseline's own fold-to-fold std is 0.011, about 4× larger than any effect.
2. **More rows don't help, and fewer rows don't hurt.** Dropping all of 2022 (R2, 55 % of the rows) and keeping one month in three (R9, 65 % removed) both score the same as the baseline. The model saturates at about 60–100k rows; beyond that, extra months of the same facilities are near-copies and add no new information. This explains why the spine-2022 gold (296k rows) didn't beat the 2023-only gold (133k rows).
3. **The 2022 drift does not hurt the model either.** R1/R2 are slightly positive on 2 of 3 folds and on test but negative on the third fold, so the effect is not stable.
4. **Removing "confidently wrong" rows hurts** (R11: negative in every fold, −0.0023 on average) and damages calibration (test Brier 0.1726 → 0.1780). These rows are not label noise; they are real hard cases (clean-looking homes that do get cited), and the model needs them to keep its probabilities honest.
5. **Removing incomplete or unusual rows (R4–R8) does nothing.** LightGBM already handles missing values and small groups well.
6. **R6 (drop rows with ≥ 10 quality measures missing) is the only filter that passes the keep rule:** +0.0001 / +0.0001 / +0.0003 on the three folds, +0.0001 on test. It passes because it is very consistent, not because it is large: the gain is about 0.04 % of the PR-AUC.

## 6. Experiment log (playbook section 9 format)

| ID | Change (one only) | Model | #Features | PR-AUC (CV mean ± std) | ROC-AUC | Recall@10% | Brier | Δ PR-AUC vs baseline (± std) | Test PR-AUC (Δ) | Keep? |
|---|---|---|---|---|---|---|---|---|---|---|
| R0 | baseline (train rows 168,181) | LightGBM tuned | 75 | 0.5246 ± 0.0112 | 0.7319 | 0.2299 | 0.1705 | +0.0000 ± 0.0000 | 0.4870 (+0.0000) | baseline |
| R1 | drop_2022H1 (train rows 108,541) | LightGBM tuned | 75 | 0.5258 ± 0.0128 | 0.7320 | 0.2302 | 0.1705 | +0.0011 ± 0.0017 | 0.4873 (+0.0002) | no |
| R2 | drop_2022 (train rows 59,085) | LightGBM tuned | 75 | 0.5251 ± 0.0139 | 0.7309 | 0.2300 | 0.1707 | +0.0004 ± 0.0032 | 0.4879 (+0.0009) | no |
| R4 | drop_stale_insp (train rows 128,711) | LightGBM tuned | 75 | 0.5244 ± 0.0113 | 0.7315 | 0.2294 | 0.1707 | -0.0003 ± 0.0002 | 0.4857 (-0.0013) | no |
| R5 | drop_missing_staffing (train rows 162,629) | LightGBM tuned | 75 | 0.5242 ± 0.0114 | 0.7318 | 0.2294 | 0.1706 | -0.0005 ± 0.0005 | 0.4876 (+0.0006) | no |
| R6 | drop_missing_qm (train rows 163,709) | LightGBM tuned | 75 | 0.5248 ± 0.0111 | 0.7322 | 0.2303 | 0.1705 | +0.0002 ± 0.0001 | 0.4872 (+0.0001) | yes |
| R7 | drop_tiny_or_hospital (train rows 161,657) | LightGBM tuned | 75 | 0.5252 ± 0.0102 | 0.7325 | 0.2301 | 0.1704 | +0.0006 ± 0.0011 | 0.4868 (-0.0002) | no |
| R8 | drop_new_facility (train rows 167,060) | LightGBM tuned | 75 | 0.5248 ± 0.0111 | 0.7318 | 0.2304 | 0.1706 | +0.0002 ± 0.0005 | 0.4866 (-0.0005) | no |
| R9 | quarterly_thin (train rows 59,389) | LightGBM tuned | 75 | 0.5244 ± 0.0116 | 0.7319 | 0.2292 | 0.1706 | -0.0003 ± 0.0007 | 0.4867 (-0.0003) | no |
| R10 | noise_2pct (train rows 164,817) | LightGBM tuned | 75 | 0.5240 ± 0.0114 | 0.7298 | 0.2305 | 0.1721 | -0.0007 ± 0.0006 | 0.4866 (-0.0004) | no |
| R11 | noise_5pct (train rows 159,771) | LightGBM tuned | 75 | 0.5223 ± 0.0102 | 0.7287 | 0.2294 | 0.1756 | -0.0023 ± 0.0010 | 0.4860 (-0.0011) | no |

## 7. Decision

**Recommendation: keep all training rows (no row filter).** The training rows are already clean enough for this model; the data is not what limits it.

- **R6 passes the rule formally, but I recommend not adopting it.** Its gain (+0.0002 CV, +0.0001 test) is far below anything that matters, and it would add a data rule to maintain in the pipeline. Those homes still have to be scored in production, and training on them keeps the model familiar with them. If you prefer to follow the rule strictly, R6 is safe to adopt: it removes 2.6 % of the rows, and every metric stays the same or slightly better.
- **Optional efficiency win:** R2 and R9 show the model saturates at about 100k rows, so training on 2023 only or on quarterly snapshots would cut training time by 55–65 % with no loss. That is a cost decision, not an accuracy one, and it means dropping data, so it needs your approval.
- **Where the gains are not:** rows. Together with the rev002 finding that more history and more model complexity don't help either, the ceiling (test PR-AUC about 0.49) comes from the information in the features. The next improvements should come from new information (for example owner/chain-level history or deficiency text), not from reshaping the rows.

**Not done, by design:** no filter was applied to the gold data or the saved models; nothing was committed.

## 8. Files
| File | What it is |
|---|---|
| `FE_Rows.md` | the playbook this experiment follows |
| `make_notebook.py` | builds `finetune_rows.ipynb` (`FT_SMOKE=1` for a 10 % smoke run) |
| `finetune_rows.ipynb` | executed notebook (full run) |
| `outputs/filter_coverage.csv` | rows removed by each filter |
| `outputs/cv_by_fold.csv`, `outputs/cv_summary.csv` | CV metrics per fold and mean ± std |
| `outputs/test_results.csv` | test metrics per filter |
| `outputs/experiment_log.md` | the section 9 log table |
| `outputs/figures/cv_delta_pr_auc.png` | Δ PR-AUC per filter with ± 1 std |
