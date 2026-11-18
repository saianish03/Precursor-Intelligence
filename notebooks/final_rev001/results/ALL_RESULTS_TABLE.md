# Precursor Intelligence: all results in one table (as of 2026-11-07)

Label `y_g12`: a new G+ (harm) citation within 12 months. Base rate ≈ 26 %. Every number is on the **test** months
(scored once), unless marked. "—" = not measured in that run.

Sources: `report_rev001.md`, `report_rev002.md`, `feature engineering run/gold2023_logs/boost.log`,
`ALL_TABLES_REPORT.md`, `MODEL_CARD.md`, `notebook_rev002_finetune_rows/report.md`, exp6/exp7 summaries, `experiments_rev003/report.md` (exp12 PBJ).

## 1. The datasets

| # | Dataset | Rows | Months | Train rows used | Test window (rows) | Features |
|---|---|---|---|---|---|---|
| D1 | Subset (quarterly CSV) | 222,677 | 15 quarterly snapshots | 44,295 (2023-01/04/07) | 2024-10 (14,665) | 133 v1 / 66 selected |
| D2 | Gold 2023 (spine 2023) | 578,975 | monthly 2023-01 → 2026-08 | 132,832 (2023-01 → 09) | 2024-10/11 (29,324) | 75 selected |
| D3 | Gold 2022 (spine 2022) | 746,024 | 50 monthly, 2022-01 → 2026-08 | 296,476 (2022-01 → 2023-09) | 2024-10/11 (29,324) | 75 selected |
| D4 | Gold 2022, 12-month purge gap | 746,024 | same | 89,459 (2022-01 → 06) | 2024-10/11 (29,324) | 133 v1 (+120 engineered in exp5) |
| D5 | All-tables gold (21 CMS tables) | 775,400 | 52 monthly, 2022-01 → 2026-10 | 119k–326k (see rows) | 2025-02 (14,643) | 133 old / 202 all / 100 selected |
| D6 | All-tables gold, 70/15/15 splits | 502,102 labelled | same | ~351k | ~75k (strictest: 21,934) | 133 old / 202 all |

## 2. Every model, every dataset

| Dataset | Model | Features | PR-AUC | ROC-AUC | P@5 % | P@10 % | Accuracy |
|---|---|---|---|---|---|---|---|
| **D1 Subset** | Baseline: star rating | — | 0.344 | 0.629 | 0.422 | 0.389 | — |
| D1 Subset | Baseline: severity rule | — | 0.415 | 0.663 | 0.576 | 0.514 | — |
| D1 Subset | GBM | 133 v1 | 0.482 | 0.706 | 0.649 | 0.579 | — |
| D1 Subset | GBM | 66 selected | 0.484 | 0.703 | 0.679 | 0.582 | — |
| D1 Subset | **Logistic regression** | 66 selected | **0.492** | **0.708** | **0.694** | **0.598** | — |
| **D2 Gold 2023** | Baseline: star rating | — | 0.341 | 0.628 | 0.412 | 0.384 | — |
| D2 Gold 2023 | Baseline: severity rule | — | 0.411 | 0.661 | 0.576 | 0.503 | — |
| D2 Gold 2023 | GBM default | 75 selected | 0.479 | 0.702 | 0.672 | 0.568 | — |
| D2 Gold 2023 | GBM | 133 v1 | 0.483 | 0.706 | 0.685 | 0.578 | — |
| D2 Gold 2023 | Logistic regression | 75 selected | 0.486 | 0.707 | 0.687 | 0.588 | — |
| D2 Gold 2023 | CatBoost | 75 selected | 0.486 | 0.706 | 0.686 | 0.581 | — |
| D2 Gold 2023 | LightGBM tuned (Optuna) | 75 selected | 0.489 | 0.708 | 0.690 | 0.584 | — |
| D2 Gold 2023 | **Ensemble (LGBM + CatBoost + LR)** | 75 selected | **0.490** | **0.709** | **0.703** | **0.588** | — |
| **D3 Gold 2022** | Baseline: star rating | — | 0.341 | 0.628 | 0.412 | 0.384 | — |
| D3 Gold 2022 | Baseline: severity rule | — | 0.411 | 0.661 | 0.576 | 0.503 | — |
| D3 Gold 2022 | GBM default | 75 selected | 0.480 | 0.705 | 0.666 | 0.576 | — |
| D3 Gold 2022 | Logistic regression | 75 selected | 0.482 | 0.703 | 0.684 | 0.583 | — |
| D3 Gold 2022 | LightGBM tuned (Optuna) | 75 selected | 0.487 | 0.708 | 0.676 | 0.582 | — |
| D3 Gold 2022 | CatBoost | 75 selected | 0.487 | 0.707 | 0.675 | 0.585 | — |
| D3 Gold 2022 | **Ensemble (LGBM + CatBoost + LR)** | 75 selected | **0.488** | **0.708** | 0.683 | **0.586** | — |
| **D4 Gold 2022 + gap** | exp4 **Logistic regression, one-hot** | 133 v1 | **0.481** | 0.699 | **0.681** | **0.577** | — |
| D4 Gold 2022 + gap | exp3 LightGBM tuned | 133 v1 | 0.474 | 0.700 | 0.640 | 0.563 | — |
| D4 Gold 2022 + gap | exp5 Logistic regression | 133 + 120 engineered | 0.474 | 0.700 | 0.657 | 0.565 | — |
| D4 Gold 2022 + gap | exp5 LightGBM tuned | 133 + 120 engineered | 0.474 | 0.699 | 0.648 | 0.560 | — |
| D4 Gold 2022 + gap | exp4 LightGBM, target encoding | 133 v1 | 0.470 | 0.695 | 0.647 | 0.562 | — |
| D4 Gold 2022 + gap | exp1 FT-Transformer | 133 v1 | 0.464 | 0.694 | 0.623 | 0.545 | — |
| D4 Gold 2022 + gap | exp2 MLP | 133 v1 | 0.462 | 0.689 | 0.647 | 0.551 | — |
| D4 Gold 2022 + gap | exp3 LightGBM default | 133 v1 | 0.462 | 0.689 | 0.624 | 0.555 | — |
| D4 Gold 2022 + gap | exp1 TabNet, pre-trained | 133 v1 | 0.461 | 0.682 | 0.642 | 0.558 | — |
| D4 Gold 2022 + gap | exp2 DeepFM | 133 v1 | 0.452 | 0.678 | 0.626 | 0.544 | — |
| D4 Gold 2022 + gap | exp2 DCN-v2 | 133 v1 | 0.451 | 0.674 | 0.645 | 0.543 | — |
| D4 Gold 2022 + gap | exp1 TabNet | 133 v1 | 0.436 | 0.675 | 0.591 | 0.526 | — |
| **D5 All-tables** | Predict "no citation" | — | 0.256 | 0.500 | 0.256 | 0.256 | 74.4 % |
| D5 All-tables | LightGBM (train 2022-01 → 2023-11) | 133 old | 0.475 | 0.708 | 0.656 | 0.573 | 75.8 % |
| D5 All-tables | Logistic regression (2022-01 → 2023-11) | 133 old | 0.472 | 0.705 | 0.657 | 0.571 | 75.9 % |
| D5 All-tables | Ensemble LGBM + LR (2022-01 → 2023-11) | 133 old | 0.475 | 0.708 | 0.665 | 0.572 | 76.0 % |
| D5 All-tables | LightGBM (2022-01 → 2023-11) | 100 selected (39 new) | 0.472 | 0.708 | 0.656 | 0.567 | 75.7 % |
| D5 All-tables | Logistic regression (2022-01 → 2023-11) | 100 selected | 0.469 | 0.703 | 0.649 | 0.576 | 76.0 % |
| D5 All-tables | Ensemble (2022-01 → 2023-11) | 100 selected | 0.473 | 0.707 | 0.658 | 0.574 | 75.9 % |
| D5 All-tables | Ensemble, train 2022-01 → 08 (119k) | 133 old | 0.465 | 0.700 | 0.645 | 0.562 | 75.6 % |
| D5 All-tables | Ensemble, train 2022 only (162k) | 133 old | 0.466 | 0.702 | 0.646 | 0.560 | 75.7 % |
| D5 All-tables | Ensemble, train 2023 only (162k) | 133 old | 0.480 | 0.709 | 0.678 | 0.575 | 76.4 % |
| D5 All-tables | **Ensemble, train 2023 only (119k) — recommended** | 133 old | **0.481** | **0.709** | **0.682** | **0.579** | **76.4 %** |
| D5 All-tables | exp12 Ensemble, train 2023-01 → 11 (162k) | 133 old | 0.480 | 0.709 | 0.678 | 0.575 | 75.8 % |
| D5 All-tables | exp12 LightGBM tuned, train 2023-01 → 11 | 133 + 24 PBJ | 0.479 | 0.709 | 0.667 | 0.575 | — |
| D5 All-tables | exp12 Logistic regression, train 2023-01 → 11 | 133 + 24 PBJ | 0.478 | 0.707 | 0.689 | 0.579 | — |
| D5 All-tables | exp12 Ensemble LGBM + LR, train 2023-01 → 11 | 133 + 24 PBJ | 0.481 | 0.710 | 0.691 | 0.574 | 75.7 % |
| **D6 70/15/15** | LightGBM, random rows (**leaky, invalid**) | 133 old | 0.824 | 0.926 | 0.945 | 0.912 | 87.3 % |
| D6 70/15/15 | LightGBM, by facility | 133 old | 0.503 | 0.733 | 0.705 | 0.613 | 76.4 % |
| D6 70/15/15 | LightGBM, by facility | 202 all | 0.502 | 0.732 | 0.688 | 0.613 | 76.7 % |
| D6 70/15/15 | LightGBM, by time (no purge, leak suspected) | 133 old | 0.488 | 0.714 | 0.660 | 0.583 | 74.5 % |
| D6 70/15/15 | LightGBM, by time (no purge, leak suspected) | 202 all | 0.505 | 0.727 | 0.686 | 0.609 | 74.8 % |
| D6 70/15/15 | LightGBM, facility + time (strictest) | 133 old | 0.486 | 0.704 | 0.661 | 0.582 | 74.8 % |
| D6 70/15/15 | LightGBM, facility + time (strictest) | 202 all | 0.485 | 0.706 | 0.649 | 0.585 | 74.8 % |

Holdout (not test) for reference: D2 ensemble 0.527 PR-AUC / 0.737 ROC; D3 ensemble 0.520 / 0.737
(20 % of DEV facilities — same months as training, so higher than test).

## 3. Experiments that changed one thing (test PR-AUC)

| Experiment | Baseline | With change | Δ | Kept? |
|---|---|---|---|---|
| Row filters R1–R11 (D3, 75 feats) | 0.487 | best/worst within ±0.0013 | ≈ 0 | No (none matters) |
| exp6 survey × harm split (D4) | 0.4675 | 0.4671 | −0.0004 | No |
| exp7 case-mix HPRD features (D4) | 0.4675 | 0.4691 | +0.0017 (CI +0.0001..+0.0036) | Real but tiny |
| exp5 +120 engineered features (D4, LR) | 0.4811 | 0.4738 | −0.007 | No |
| 14 extra CMS tables (D5, ensemble) | 0.4754 | 0.4733 | −0.002 | No |
| exp12 PBJ daily staffing, 24 feats (D5, ensemble, test 2025-02) | 0.4802 | 0.4810 | +0.0008 (CI −0.0004..+0.0018); 2024-11: +0.0012 (CI +0.0001..+0.0023) | Real but tiny (passes keep rule; accuracy flat) |

## 4. Best model per dataset

| Dataset | Best model | PR-AUC | ROC-AUC | P@5 % | P@10 % | Accuracy |
|---|---|---|---|---|---|---|
| D1 Subset | Logistic regression, 66 selected | 0.492 | 0.708 | 0.694 | 0.598 | — |
| D2 Gold 2023 | Ensemble, 75 selected | 0.490 | 0.709 | 0.703 | 0.588 | — |
| D3 Gold 2022 | Ensemble, 75 selected | 0.488 | 0.708 | 0.683 | 0.586 | — |
| D4 Gold 2022 + gap | Logistic regression, 133 | 0.481 | 0.699 | 0.681 | 0.577 | — |
| D5 All-tables | Ensemble, 133, train 2023 only | 0.481 | 0.709 | 0.682 | 0.579 | 76.4 % |
| D6 70/15/15 (strictest) | LightGBM, 202 | 0.485 | 0.706 | 0.649 | 0.585 | 74.8 % |

**Reading:** every honest setup lands at PR-AUC 0.48–0.49 and ROC-AUC ≈ 0.70–0.71 (about 2.6× the base rate in the
top 5 %). Data size, extra tables, engineered features, PBJ daily staffing and deep models don't move it; recent training months do.
Caveats: D1 tests on 2024-10 only; D2–D4 on 2024-10/11; D5 on a single month (2025-02), so D5 is not directly
comparable with D1–D4. D6 tests on pooled rows from many months.
