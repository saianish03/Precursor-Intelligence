# Precursor Intelligence: results across all datasets

Label `y_g12`: a new G+ (harm) citation within 12 months. Base rate ≈ 26 %. All numbers are on the **test** months,
scored once.

**Datasets**

| # | Dataset | Rows |
|---|---|---|
| D1 | Subset (quarterly) | 222,677 |
| D2 | Gold 2023 | 578,975 |
| D3 | Gold 2022 | 746,024 |
| D4 | Gold 2022 with the 12-month purge gap | 746,024 (trained on 89,459) |
| D5 | All-tables gold (21 CMS tables) | 775,400 |
| D6 | All-tables gold, 70/15/15 splits | 502,102 labelled |

## Results

| Dataset | Model | Features | PR-AUC | ROC-AUC | P@5% | P@10% | Acc. |
|---|---|---|---|---|---|---|---|
| D1 Subset | Star rating (baseline) | — | 0.344 | 0.629 | 0.422 | 0.389 | — |
| D1 Subset | Severity rule (baseline) | — | 0.415 | 0.663 | 0.576 | 0.514 | — |
| D1 Subset | GBM | 133 | 0.482 | 0.706 | 0.649 | 0.579 | — |
| D1 Subset | GBM | 66 sel | 0.484 | 0.703 | 0.679 | 0.582 | — |
| D1 Subset | **Logistic regression** | 66 sel | **0.492** | 0.708 | 0.694 | 0.598 | — |
| D2 Gold 2023 | Star rating / severity rule | — | 0.341 / 0.411 | 0.628 / 0.661 | 0.412 / 0.576 | 0.384 / 0.503 | — |
| D2 Gold 2023 | GBM default | 75 sel | 0.479 | 0.702 | 0.672 | 0.568 | — |
| D2 Gold 2023 | GBM | 133 | 0.483 | 0.706 | 0.685 | 0.578 | — |
| D2 Gold 2023 | Logistic regression | 75 sel | 0.486 | 0.707 | 0.687 | 0.588 | — |
| D2 Gold 2023 | CatBoost | 75 sel | 0.486 | 0.706 | 0.686 | 0.581 | — |
| D2 Gold 2023 | LightGBM tuned | 75 sel | 0.489 | 0.708 | 0.690 | 0.584 | — |
| D2 Gold 2023 | **Ensemble (LGBM+CB+LR)** | 75 sel | **0.490** | 0.709 | 0.703 | 0.588 | — |
| D3 Gold 2022 | Star rating / severity rule | — | 0.341 / 0.411 | 0.628 / 0.661 | 0.412 / 0.576 | 0.384 / 0.503 | — |
| D3 Gold 2022 | GBM default | 75 sel | 0.480 | 0.705 | 0.666 | 0.576 | — |
| D3 Gold 2022 | Logistic regression | 75 sel | 0.482 | 0.703 | 0.684 | 0.583 | — |
| D3 Gold 2022 | LightGBM tuned | 75 sel | 0.487 | 0.708 | 0.676 | 0.582 | — |
| D3 Gold 2022 | CatBoost | 75 sel | 0.487 | 0.707 | 0.675 | 0.585 | — |
| D3 Gold 2022 | **Ensemble** | 75 sel | **0.488** | 0.708 | 0.683 | 0.586 | — |
| D4 + gap | **LogReg one-hot (exp4)** | 133 | **0.481** | 0.699 | 0.681 | 0.577 | — |
| D4 + gap | LightGBM tuned (exp3) | 133 | 0.474 | 0.700 | 0.640 | 0.563 | — |
| D4 + gap | LR / LGBM + 120 engineered (exp5) | 253 | 0.474 / 0.474 | 0.700 / 0.699 | 0.657 / 0.648 | 0.565 / 0.560 | — |
| D4 + gap | LightGBM target-enc (exp4) | 133 | 0.470 | 0.695 | 0.647 | 0.562 | — |
| D4 + gap | FT-Transformer / MLP | 133 | 0.464 / 0.462 | 0.694 / 0.689 | 0.623 / 0.647 | 0.545 / 0.551 | — |
| D4 + gap | LightGBM default / TabNet-pre | 133 | 0.462 / 0.461 | 0.689 / 0.682 | 0.624 / 0.642 | 0.555 / 0.558 | — |
| D4 + gap | DeepFM / DCN-v2 / TabNet | 133 | 0.452 / 0.451 / 0.436 | 0.678 / 0.674 / 0.675 | 0.626 / 0.645 / 0.591 | 0.544 / 0.543 / 0.526 | — |
| D5 All-tables | Predict "no citation" | — | 0.256 | 0.500 | 0.256 | 0.256 | 74.4% |
| D5 All-tables | LGBM / LR / Ensemble (train 22-01→23-11) | 133 | 0.475 / 0.472 / 0.475 | 0.708 / 0.705 / 0.708 | 0.656 / 0.657 / 0.665 | 0.573 / 0.571 / 0.572 | 75.8 / 75.9 / 76.0% |
| D5 All-tables | LGBM / LR / Ensemble (train 22-01→23-11) | 100 sel | 0.472 / 0.469 / 0.473 | 0.708 / 0.703 / 0.707 | 0.656 / 0.649 / 0.658 | 0.567 / 0.576 / 0.574 | 75.7 / 76.0 / 75.9% |
| D5 All-tables | Ensemble, train 2022-01→08 | 133 | 0.465 | 0.700 | 0.645 | 0.562 | 75.6% |
| D5 All-tables | Ensemble, train 2022 only | 133 | 0.466 | 0.702 | 0.646 | 0.560 | 75.7% |
| D5 All-tables | **Ensemble, train 2023 only (recommended)** | 133 | **0.481** | 0.709 | 0.682 | 0.579 | **76.4%** |
| D6 70/15/15 | Random rows (**leaky, invalid**) | 133 | 0.824 | 0.926 | 0.945 | 0.912 | 87.3% |
| D6 70/15/15 | By facility | 133 / 202 | 0.503 / 0.502 | 0.733 / 0.732 | 0.705 / 0.688 | 0.613 / 0.613 | 76.4 / 76.7% |
| D6 70/15/15 | By time (no purge gap, leak suspected) | 133 / 202 | 0.488 / 0.505 | 0.714 / 0.727 | 0.660 / 0.686 | 0.583 / 0.609 | 74.5 / 74.8% |
| D6 70/15/15 | Facility + time (strictest) | 133 / 202 | 0.486 / 0.485 | 0.704 / 0.706 | 0.661 / 0.649 | 0.582 / 0.585 | 74.8 / 74.8% |

## What the table shows

- **Every honest setup scores about the same:** PR-AUC 0.48–0.49 and ROC-AUC about 0.71.
- **What doesn't help:** more rows, the extra CMS tables, engineered features and the deep models.
- **What does help:** training on more recent months.

## Comparing across datasets

- **Different test months:** D1 tests on 2024-10 only, D2–D4 on 2024-10/11, and D5 on one month (2025-02). Compare D5
  with D1–D4 loosely.
- **No accuracy for D1–D4:** those runs never measured it, so those cells are blank.
- **Two sources:** the D2 baselines, GBM-133 and CatBoost rows come from the earlier subset-vs-full comparison table.
  The rest of D2 comes from `feature engineering run/gold2023_logs/boost.log`.
