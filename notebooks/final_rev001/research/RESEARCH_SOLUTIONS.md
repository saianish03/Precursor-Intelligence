# Breaking the ~0.48 PR-AUC / ~76 % accuracy ceiling — research shortlist (2026-11-07)

Three parallel research passes: (A) winning ML-competition techniques (AmEx, IEEE-CIS, Home Credit, Amazon ML
Challenge, Kaggle Playground), (B) nursing-home citation literature and public data, (C) target / label
reformulation (survival, PU learning, food-inspection forecasting). Sources at the end.

## What the research says about the ceiling

- **No published model predicts CMS G+ citations with a reported AUC.** The closest facility-level model (COVID
  outbreaks in nursing homes, GBM on public data) got AUC 0.72–0.73. Ontario found public quality indicators are "poor
  predictors" of inspection results. Our ROC 0.71 is in line with that, so expect **+0.01–0.03 ROC / +0.02–0.04 PR-AUC
  at most** from everything below combined. 80 % accuracy is very unlikely with public data (majority = 74.4 %).
- **Amazon ML Challenge does not transfer:** 2021 browse-node classification, 2024 image entity extraction and 2025
  multimodal price prediction are text/image problems. Santander "magic" was a dataset artifact. The transferable
  competition ideas come from **AmEx Default Prediction** (monthly panel, binary label: closest to us), **IEEE-CIS
  fraud** (entity aggregation), **Home Credit** (neighbour target mean) and the **Chicago / Boston food-inspection**
  forecasting work (same structure: the outcome only exists when an inspector visits).
- Every competition write-up agrees: model tuning, NNs and ensembling add +0.001–0.01. **Big jumps came from features
  that give context the row does not have**: the entity's group, its peers, its position in the cross-section.
  That matches what we saw: tuning, deep models, engineered ratios, case-mix and PBJ all ≈ 0.

## Already covered (no need to redo)
State G+ rate (`state_gplus_rate_12m`), survey timing (P(survey) PR-AUC 0.947), two-stage survey × harm (exp6, no
gain), recent-window training (exp9: 2023-only is best), case-mix (exp7), PBJ (exp12), 14 extra CMS tables, Optuna,
CatBoost, target encoding, FT-Transformer/TabNet/MLP/DCN/DeepFM, +120 trend/ratio features, row filters.

## Ranked candidates

| # | Idea | Where it comes from | Expected Δ PR-AUC | Effort | Notes |
|---|---|---|---|---|---|
| ★1 | **Peer / entity aggregates**: leave-one-out sibling-facility (same owner / chain) G+ rate 12 m; county and CMS-region G+ rate **per survey**; state survey-agency propensity (G+ share of completed surveys, overdue share, facility's survey gap vs its state's cadence); out-of-fold 500-NN target mean | IEEE-CIS 1st place "UID" aggregates; Home Credit 1st place `neighbors_target_mean_500`; Chicago inspector-cluster effect (odds ratio ~4.7, the top feature); OIG/GAO state variation (2.9–11.2 deficiencies per home) | +0.005–0.02 | 6–10 h | Highest upside. Must use only labels matured by t. Our ownership family counts owners but has no sibling-citation feature |
| ★2 | **Complaint / facility-reported-incident intake**: `Number of Facility Reported Incidents`, `Number of Substantiated Complaints` from Provider Info (mapped in `configs/column_map.yaml:490-493`, **not used in gold**); levels + 6/12-month change | GAO-11-280 (1 in 5 complaints cited), OIG complaint-trend reports; complaint surveys produce most abuse citations | +0.003–0.015 | 2–4 h | Intake drives unscheduled complaint surveys, the part of timing we model worst. Definitions change in 2025-07 (iQIES) → regime flag; CMS drops them from Care Compare 2026-02 |
| ★3 | **Cross-sectional month ranks**: percentile rank + z-score of each numeric feature within the same month (and within state-month) | AmEx 1st place `ym_rank_` features; Tschalzev et al. 2024: test-time feature engineering helps on time-ordered Kaggle data | +0.005–0.015 | 2–4 h | Targets the 2023 → 2025 drift directly ("worse than peers" instead of absolute levels) |
| 4 | Exponential recency sample weights (τ ∈ 6/12/24 months, pick τ on a 2024 month) | common practice; generalises our exp9 finding | +0.002–0.005 | 1 h | Cheap; can run alongside |
| 5 | Lagged out-of-fold score trajectory (model's own score at t-1 … t-12) + full-history per-facility aggregates | AmEx 1st place (S4 lag predictions) | +0.002–0.01 | 6–10 h | Skeptical: trend features already failed; the score trajectory is the new part |
| 6 | Next-survey target: P(standard survey) · P(G+ \| standard) + P(complaint survey) · P(G+ \| complaint) | Chicago / Boston predict the next inspection's outcome, not a window | +0.0–0.01 | 6–8 h | Removes censoring; exp6 used a windowed target |
| 7 | Discrete-time hazard / survival stacking (monthly hazard, aggregate 1 − ∏(1 − h)) | Craig, Zhong & Tibshirani 2021 "Survival stacking" | +0.0–0.005 | 6–8 h | No evidence it beats direct fixed-horizon classification |
| 8 | Partial labels / pseudo-labels for the newest 12 months (an already-observed G+ is a sure positive; 3/6-month-horizon model as a feature) | AmEx 15th place (Deotte) | +0.003–0.01 | 5–8 h | Uses the most recent data, which exp9 says matters most |
| 9 | Adversarial validation (2023 vs 2025) / time-consistency feature pruning | IEEE-CIS 1st place; Uber AutoML | 0–0.005 | 3–5 h | Mostly stability |
| 10 | Text of prior deficiencies (TF-IDF → SVD; harm-adjacent wording in D/E citations) | Boston "Keeping it Fresh" winner (review text); Kang et al. EMNLP 2013 | uncertain | 10+ h | Needs dated full-text deficiency source (ProPublica / QCOR); high effort |
| 11 | DART LightGBM, multi-seed, hill-climb stacking | AmEx 1st place, NVIDIA grandmaster playbook | +0.001–0.005 | 2–3 h | Final polish only |
| 12 | Financial distress (HCRIS cost reports: margin, Medicaid share); change-of-ownership timing; ADI / rurality | Zhao 2010, Harrington 2012, GAO-11-571 | small | 4–8 h | Cost reports need simulated publication lag (FY end + 12–18 m) |

## Recommendation

Run **exp13 = ★2 complaint/incident intake** first. The data is already in silver, so it's the cheapest real new
information. Then **exp14 = ★1 peer/entity aggregates**, which has the highest upside, and **exp15 = ★3
cross-sectional ranks**. Use exp12's design: one change per experiment, train 2023-01..11, test 2025-02 + 2024-11,
3-fold facility-grouped keep rule, facility-clustered bootstrap CIs. If all three pass, run a combined model with
recency weights (#4) and a DART / seed ensemble (#11) as the final candidate.

Honest target: PR-AUC ~0.49–0.51, ROC ~0.72–0.73. Accuracy will stay ~76–77 %, because the label is driven by
inspection timing and surveyor behaviour that public data only partly captures. Report P@k / lift alongside it.

## Sources
- AmEx 1st place code: https://github.com/jxzly/Kaggle-American-Express-Default-Prediction-1st-solution
- AmEx 15th place (Deotte): https://www.kaggle.com/competitions/amex-default-prediction/discussion/347641
- IEEE-CIS 1st place: https://www.kaggle.com/c/ieee-fraud-detection/discussion/111284
- Home Credit 1st place: https://www.kaggle.com/c/home-credit-default-risk/discussion/64821
- Tschalzev et al. 2024: https://arxiv.org/html/2407.02112v1
- NVIDIA grandmaster playbook: https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/
- Chicago food-inspection hindsight analysis: https://arxiv.org/abs/1910.04906
- Boston "Keeping it Fresh": https://blog.drivendata.org/blog/keeping-it-fresh-results ; https://www.nber.org/papers/w22124
- Kang et al. 2013: https://aclanthology.org/D13-1150/
- Survival stacking: https://arxiv.org/pdf/2107.13480
- Selective labels: https://pmc.ncbi.nlm.nih.gov/articles/PMC5958915
- COVID outbreak model in nursing homes (AUC 0.72): https://doi.org/10.1016/j.jamda.2020.08.030
- GAO-11-280 (complaints): https://www.gao.gov/products/gao-11-280
- OIG complaint investigations 2016–2018: https://oig.hhs.gov/reports/all/2020/states-continued-to-fall-short-in-meeting-required-timeframes-for-investigating-nursing-home-complaints-2016-2018
- CMS QSAM-26-01 (FRI/allegation counts removed from Care Compare): https://www.cms.gov/files/document/qssam-26-01-nh.pdf
- Harrington 2012 (chains): https://pmc.ncbi.nlm.nih.gov/articles/PMC3447240 ; GAO-11-571: https://www.gao.gov/assets/a321072.html
- Affiliated Entity Performance Measures: https://catalog.data.gov/dataset/nursing-home-affiliated-entity-performance-measures
- Senate Aging, survey-agency vacancies: https://www.aging.senate.gov/imo/media/doc/UNINSPECTED%20&%20NEGLECTED%20-%20FINAL%20REPORT.pdf
- Gandhi, Olenski & Shi, inspection timing: https://www.nber.org/bh/20262/how-inspection-timing-affects-care-quality-us-nursing-homes
- HCRIS SNF cost reports: https://data.cms.gov/provider-compliance/cost-reports/skilled-nursing-facility-cost-report
