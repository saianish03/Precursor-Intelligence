# The ML pipeline: what is built and what it gives

> **Scope.** This is the model-development deliverable: how the ML pipeline meets each point of the course
> guidelines (`Model_pipeline.pdf`), with the results on the real gold and the evidence behind them.
> **The design and every rule:** [`ML_PIPELINE_SPEC.md`](ML_PIPELINE_SPEC.md) (§ numbers below refer to it).
> **Operating it in Airflow:** [`runbooks/airflow-local.md`](runbooks/airflow-local.md).
> **Status (November 2026):** Phase 1 is built (work packages W0 to W6). Its training, evaluation and tuning have
> run on the real gold on the laptop integration stack (Docker Compose: Airflow 3, MLflow, Postgres) against the GCS
> bucket's `ml-dev/` prefix; the first full DAG run, with Checkpoint ML-A and the first champion, is next. Phase 2
> moves it to the GCP VM, adds BigQuery and connects the API, the agents and the dashboard.

---

## Contents

1. [In one page](#1-in-one-page)
2. [Architecture](#2-architecture)
3. [Course requirements and where they are met](#3-course-requirements-and-where-they-are-met)
4. [Loading the data from the data pipeline](#4-loading-the-data-from-the-data-pipeline)
5. [Training and selecting the model](#5-training-and-selecting-the-model)
6. [Validation on held-out data](#6-validation-on-held-out-data)
7. [Bias detection and mitigation](#7-bias-detection-and-mitigation)
8. [Hyperparameter tuning](#8-hyperparameter-tuning)
9. [Sensitivity analysis](#9-sensitivity-analysis)
10. [Experiment tracking and the model registry](#10-experiment-tracking-and-the-model-registry)
11. [Gates, promotion and rollback](#11-gates-promotion-and-rollback)
12. [CI/CD, notifications and containers](#12-cicd-notifications-and-containers)
13. [How to run it](#13-how-to-run-it)
14. [The runs behind the numbers](#14-the-runs-behind-the-numbers)
15. [Known limitations and Phase 2](#15-known-limitations-and-phase-2)

---

## 1. In one page

- **The task.** For every U.S. nursing home and month `t`, the probability of a serious health citation (scope and
  severity G to L) in the next 12 months (`y_g12`; base rate about 26 %). Batch, monthly: CMS publishes once a month,
  so scores are computed once a month and stored. There is no model server.
- **The model.** LightGBM, logistic regression (LR) and their ensemble (the mean of the two members' log-odds) are
  trained on 11 recent months, each calibrated with Platt scaling on out-of-fold predictions. One is chosen on
  training data only, after a bias check. 119 features from the feature registry (the runs below used 120:
  `staffing_rating_fn` was excluded since, §14); the base recipe is the best model
  of the team's exploration (PR #49).
- **The result.** On the held-out test month 2025-02 (14,643 homes, 3,750 events), in the PR #49 setting, the
  ensemble reaches **PR-AUC 0.481 (95 % CI 0.465 to 0.498)**, 1.9 times the base rate, ROC-AUC 0.710, Brier 0.169 and
  ECE 0.026. This reproduces PR #49 (0.481, CI 0.465 to 0.499) with 120 features instead of 133. All three
  candidates beat the three baselines (prior-G+ rule 0.311, 24-month severity 0.395, star rating 0.330).
- **Bias.** No gated group is clearly miscalibrated (no O−E 95 % CI lies wholly outside ±0.05). Reweighting by ownership costs
  ranking (OOF PR-AUC 0.523 to 0.517) and helps no group, so it is available (`bias.mitigation: reweight`) but off.
- **Tuning.** 30 Optuna trials on 3 facility-grouped folds: every trial scores between 0.509 and 0.511; LR's `C`
  (importance 0.41) and LightGBM's `max_depth` (0.23) matter most. The surface is flat, so PR #49's parameters stay.
- **MLOps.** One Airflow DAG (`cms_nh_ml`), every step a CLI command in the same Docker image; MLflow tracking and a
  model registry with `@champion` and `@challenger`; seven deterministic gates; rollback by moving the alias; an
  email for every run; the whole chain runs on synthetic data in CI on every pull request.

---

## 2. Architecture

![ML pipeline architecture](img/ml_pipeline_architecture.svg)

The DAG `cms_nh_ml` (`dags/cms_nh_ml.py`) starts when the data pipeline publishes gold (asset `cms_nh_gold`) or by
hand. Every task is one `precursor-pipeline ml <command>` in the `precursor-pipeline` image, with the run's id, so all
of a run's files live in `ml/runs/<run id>/` (GCS, `ml-dev/` prefix). The numbers match the diagram.

| Step | Stage | Command | Built in | Writes |
|---|---|---|---|---|
| 1 | Decide and pin | `ml decide` | W6, W0 | `decision.json`: RETRAIN, SCORE, SKIP (or by hand RETUNE, PROMOTE); the pinned gold run |
| 2 | Prepare | `ml prepare` | W0 | `input/` (pinned gold copy, manifest with sha256s), `windows.json`, `frames/` |
| 3 | Train and select | `ml train` | W1 | `models/eval/<candidate>/`, `models/refit/<chosen>/`, `oof/`, `selection.json`, `train.json`; one MLflow run |
| 4 | Evaluate | `ml evaluate` | W2 | `evaluation.json`, `report/*.png`, `slices.csv`, `shap_mean_abs.csv`; one MLflow run |
| 5 | Gate and promote | `ml gate` / `ml promote` | W2 | `report/gates.json`, `report/model_card.md`, a registry version and alias, an `ml_runs` row |
| 6 | Score and explain | `ml score` | W3 | `scores/v=YYYY-MM/`: scores, top reasons per home, SHAP values, chain roll-up |
| 7 | Publish and monitor | `ml publish`, `ml monitor` | W3, W6 | the scores manifest and asset `cms_nh_scores`; PSI drift, `status/latest.json`; the email (`notify`) |
| (side) | Re-tune | `ml tune` | W1 | `tuning/`: proposal, trials, importances, plots; one MLflow run |

**Paths through the DAG** (`decide` chooses, the branches only read its output):

```
decide ─► prepare ─┬─ RETUNE ─► tune ───────────────────────────────────────────────────┐
   │               ├─ RETRAIN ─► train ─► evaluate ─► gate ─┐                           │
   │               ├─ PROMOTE ─► promote ───────────────────┤                           ▼
   │               └─ SCORE ────────────────────────────────┴─► score ─► publish ─► monitor ─► notify
   └─ SKIP ─────────────────────────────────────────────────────────────────────────────────────┘
```

| Mode (DAG param) | What `decide` does |
|---|---|
| `auto` (the asset trigger) | **RETRAIN** when new matured labels or a new recipe (code or `ml.yaml`) arrive; **SCORE** when only a new month of features arrives; **SKIP** when nothing is new. Never re-tunes or promotes |
| `retrain`, `score_only` | forces that path |
| `retune` | runs `tune` alone (a proposal and the sensitivity plots); nothing is trained, gated or promoted in Phase 1 |
| `promote` + `version` + `reason` | moves `@champion` to that version: Checkpoint ML-A, or a rollback to any earlier version |

`train`, `tune` and `evaluate` share the Airflow pool `heavy` (one slot, also taken by gold's build) with timeouts
of 45, 90 and 60 minutes and no retry. Behaviour lives in `configs/ml.yaml`; the code is in
`src/precursorintelligence/{training,evaluation,serving,monitoring}`.

---

## 3. Course requirements and where they are met

| Guideline (`Model_pipeline.pdf`) | Where | Evidence |
|---|---|---|
| §2.1, §8.2 Load versioned data from the data pipeline | `ml prepare` pins gold's `_current.json`, copies the merged file with its sha256, records `dataset_version` (§1.2) | [section 4](#4-loading-the-data-from-the-data-pipeline) |
| §2.2, §8.3 Train, compare configurations, select the best | `ml train`: 3 candidates, OOF Platt, selection on OOF PR-AUC (§4, §5) | [section 5](#5-training-and-selecting-the-model) |
| §2.3, §8.4 Validate on a hold-out set with task metrics | `ml evaluate` on the test month (never trained or tuned on): PR-AUC, ROC-AUC, precision/recall/F1/accuracy at an OOF threshold, precision@k, Brier, ECE, bootstrap CIs (§6) | [section 6](#6-validation-on-held-out-data) |
| §2.4, §6.1, §6.2 Bias detection with slicing (Fairlearn) | `evaluation/slices.py`: Fairlearn `MetricFrame` per slice value (§7) | [section 7](#7-bias-detection-and-mitigation) |
| §2.5, §8.5 Bias checks, reports and visualizations | `slices.csv`, the model card's bias section, `report/bias_slices.png` | [section 7](#7-bias-detection-and-mitigation) |
| §6.3, §6.4 Mitigation, its steps and trade-offs | reweighting measured on every run (`evaluation/mitigation.py`) and applied by `bias.mitigation: reweight` | [section 7.3](#73-mitigation-and-its-trade-off) |
| §8.6 Selection after the bias check | a candidate that fails the OOF bias check is never selected; gate G7 checks again | [sections 5](#5-training-and-selecting-the-model), [11](#11-gates-promotion-and-rollback) |
| §3 Hyperparameter tuning, search space and process | `ml tune`: Optuna TPE (Bayesian), facility-grouped folds; space in `configs/ml.yaml: tuning.space` (§13) | [section 8](#8-hyperparameter-tuning) |
| §4 Experiment tracking; result visualizations | MLflow: params, metrics, tags, models, plots per run; charts in `report/` | [sections 6](#6-validation-on-held-out-data), [10](#10-experiment-tracking-and-the-model-registry) |
| §5 Sensitivity: features (SHAP) and hyperparameters | exact TreeSHAP/linear contributions; Optuna fANOVA importances (§8) | [section 9](#9-sensitivity-analysis) |
| §2.6, §8.7 Push to a model registry | `ml gate` registers `precursor_y_g12` in the MLflow registry (artifacts in the GCS bucket `precursor-intel-mlops-mlflow`) | [section 10](#10-experiment-tracking-and-the-model-registry) |
| §7.1, §7.2 CI trains and validates on every code change | `.github/workflows/ci.yml`: the `toy_ml` DVC stage runs the whole chain on every PR | [section 12](#12-cicd-notifications-and-containers) |
| §7.3 Bias detection blocks deployment | G7 holds the model; in CI a hold fails the check, so the PR cannot merge | [sections 11](#11-gates-promotion-and-rollback), [12](#12-cicd-notifications-and-containers) |
| §7.4 Registry push after validation | `gate`: all gates pass → registered and promoted; otherwise `@challenger` | [section 11](#11-gates-promotion-and-rollback) |
| §7.5 Notifications | every run emails its summary (`notify`); a failed run emails the failure | [section 12](#12-cicd-notifications-and-containers) |
| §7.6 Rollback | G5 stops a worse model; `mode=promote` with an earlier version rolls back (audited) | [section 11](#11-gates-promotion-and-rollback) |
| §8.1 Docker | every step runs in the `precursor-pipeline` image; MLflow has its own image | [section 12](#12-cicd-notifications-and-containers) |

---

## 4. Loading the data from the data pipeline

- **Input:** the data pipeline's gold table `gold/facility_month_dataset` (one row per home and month), read only
  through `gold/_current.json`. `prepare` copies the merged file into the run folder, checks its sha256 and records
  the gold run, `dataset_version`, `V` (newest feature month) and `V_L` (newest month whose labels are known). Every
  later step reads only the run folder, so a new gold mid-run cannot change a run's input.
- **The real gold** (`ml-dev/`): gold run `20261112T020010Z-543cdc`, `dataset_version` `3f9bac9367ae18ec`,
  775,400 home-months × 148 columns, 52 months (2022-01 to 2026-10; CMS published no data for 6 of them).
- **Features:** the registry (`configs/feature_registry.yaml`) lists 131 model features in 6 families. 12 are
  excluded in `configs/ml.yaml` (regime flags and month keys that are constant in training, four quality measures whose
  definition changed,
  `d12_turnover_total`, and `staffing_rating_fn`, whose distribution shifted, PSI 0.405 > 0.25), leaving **119**:
  facility and staffing 40, quality 33, citations 26, survey timing 9, enforcement 7, context 4. The overall and
  health-inspection star ratings are never features.
- **Windows** (§3): labels mature 20 months after `t` (12-month horizon + 8-month posting buffer), so the evaluation
  leaves the same 20-month gap between training and test as production has. Windows count months that have data.

| Frame | Production setting (`eval_gap: production_lag`) | PR #49 reproduction (`eval_gap: purge`) |
|---|---|---|
| `eval_train` (fits that are evaluated) | 2022-05 to 2023-04, 11 months, 163,066 rows | 2023-01 to 2023-11, 11 months, 162,276 rows |
| `test` (held out) | 2025-02, 14,643 homes, 3,750 events | same |
| `refit_train` (the model that ships) | 2024-02 to 2025-02, 11 months, 161,460 rows | same |
| `score` | 2026-10, 14,686 homes | same |

- **Facility-grouped folds:** 5 folds by a stable hash of the CCN, so a home is never on both sides of a split.

---

## 5. Training and selecting the model

- **Candidates** (`training/members.py`, `ensemble.py`): LightGBM (PR #49's tuned parameters: learning rate 0.0102,
  34 leaves, depth 3, 1,265 trees, strong regularisation), LR (median impute + missing indicators, scaling, one-hot,
  `C` = 0.3) and their ensemble, the mean of the two members' log-odds (exact SHAP, same accuracy as averaging
  probabilities: measured 0.4813 either way on PR #49's predictions).
- **Calibration:** each candidate's out-of-fold (OOF) predictions get a Platt fit on the newest 3 months of the
  window, so calibration never sees a facility's own fit.
- **Selection** (`training/select.py`): highest OOF PR-AUC among the candidates that pass the OOF bias check (no
  gated group with O−E clearly beyond ±0.05); within 0.002 the simpler one wins. Never on the test month.
- **Eval fit, then refit:** the recipe is fitted on the evaluation window (the evidence for the gates) and, once
  selected, refitted on the newest 11 matured months (the model that scores).

| Candidate (reproduction run) | OOF PR-AUC | OOF Brier | OOF bias check |
|---|---|---|---|
| **lgbm (selected)** | **0.5231** | 0.1666 | passed (0 of 12 gated slice values clearly outside ±0.05) |
| ensemble | 0.5218 | 0.1669 | passed |
| lr | 0.5142 | 0.1682 | passed |

---

## 6. Validation on held-out data

`ml evaluate` scores every candidate's eval fit on the test month, which no step trains, tunes or selects on. CIs are
95 % facility-bootstrap intervals (1,000 replicates). The classification threshold (0.254) is the OOF max-F1 point,
never chosen on test.

**Test 2025-02, reproduction run** (14,643 homes, 3,750 events, base rate 25.6 %):

| | PR-AUC (95 % CI) | ROC-AUC | Brier | ECE | P@5 % | P@10 % | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| PR #49 ensemble (133 features) | 0.481 (0.465-0.499) | 0.709 | 0.169 | 0.027 | 0.682 | 0.579 | | | | |
| **ensemble** | **0.481 (0.465-0.498)** | 0.710 | 0.169 | 0.026 | 0.676 | 0.580 | 0.642 | 0.384 | 0.660 | 0.485 |
| lgbm (selected on OOF) | 0.478 (0.461-0.494) | 0.709 | 0.170 | 0.034 | 0.660 | 0.572 | 0.626 | 0.375 | 0.689 | 0.485 |
| lr | 0.479 (0.463-0.496) | 0.708 | 0.169 | 0.020 | 0.682 | 0.582 | 0.654 | 0.392 | 0.637 | 0.486 |
| prior-G+ rule (baseline) | 0.311 (0.301-0.320) | | | | | | | | | |
| 24-month severity (baseline) | 0.395 (0.380-0.410) | | | | | | | | | |
| star rating (baseline) | 0.330 (0.318-0.341) | | | | | | | | | |

- The three candidates are within each other's CIs; selection on OOF picked lgbm (by 0.0013 over the ensemble).
- Within a state, PR-AUC is 0.397 and precision@10 % is 0.425: the model ranks homes inside a state, not only states.
- **2025-02 is a development month** (PR #49 chose its window and parameters on it). It is reported, never used for
  a decision; Checkpoint ML-A reads 2025-03 or later once the late-November release brings it into the test window.
- **Production setting:** with the 20-month gap the eval fit is older. W2's first full chain on the same gold
  (PR #62) gave lgbm PR-AUC 0.470 (0.454-0.486) on 2025-02, every gate passing.

| | |
|---|---|
| ![Candidates and baselines](img/ml/candidates.png) | ![Precision-recall](img/ml/pr_curve.png) |
| ![ROC](img/ml/roc_curve.png) | ![Reliability before and after Platt](img/ml/reliability.png) |
| ![Lift](img/ml/lift.png) | ![Confusion matrix at the OOF threshold](img/ml/confusion_matrix.png) |

---

## 7. Bias detection and mitigation

### 7.1 Slices and metrics
CMS facility data has no person-level demographics, so the slices are facility attributes (`configs/bias_slices.yaml`):
state, CMS region, ownership group, bed band, chain status, in-hospital, CCRC, SFF status and star band. Per slice
value (`evaluation/slices.py`, Fairlearn `MetricFrame`): homes, events, mean predicted p, **O−E** (observed minus
expected rate, the calibration gap that matters to an insurer) with a facility-clustered CI, PR-AUC, and at the global
top 10 % the flag share, precision, recall, false-positive and false-negative rates. Written to `slices.csv`, the
model card and MLflow. Data-side bias (label rates, inspector effects) is analysed before any model exists:
[`bias/findings.md`](bias/findings.md).

### 7.2 The gate and the result
Gate G7 (§9.1) judges `bias.gate_slices` without the geographic ones: ownership group, bed band, chain status and
in-hospital (`state` joins with the Benjamini-Hochberg correction in Phase 2, ML-27), on values with at least 200 homes
and 50 events: (a) against the champion, no group's |O−E| grows by more than 0.03 or its PR-AUC
drops by more than 0.05; (b) no group's O−E CI lies entirely outside ±0.05. A failure holds the model and emails.

**Result (reproduction run, lgbm, test 2025-02):** 0 calibration failures: no gated group's O−E 95 % CI lies wholly
outside ±0.05. Across all reported non-geographic values, gated or not, the largest gaps are SFF candidates (−0.050,
433 homes, CI −0.094 to −0.006; SFF status is reported, not gated), homes under 50 beds (−0.045) and homes with 200+
beds (+0.040). Most of each group's gap is one overall shift: on 2025-02 the model over-predicts by
2.6 points (mean p 0.282 vs an observed 0.256; in-hospital = false, 14,182 of the 14,643 homes, is −0.027), as an
evaluation fit whose data ends 22 months before the test month can. So −0.045 for small homes is mostly that shift,
not a small-home problem.
Ranking is weaker for small homes (PR-AUC 0.28) and 4- and 5-star homes (0.33, 0.23), where events are rarer. States vary most (as the
data analysis found); they are reported, not gated, and per-state shrinkage is Phase 2.

![Bias by slice](img/ml/bias_slices.png)

*Per slice value: O−E with its 95 % CI against ±0.05 (red would mean G7(b) fails), PR-AUC against all homes, and the
false-negative rate at the top 10 %. Homes in brackets.*

### 7.3 Mitigation and its trade-off
Every `evaluate` refits the selected candidate fold by fold with **inverse-frequency sample weights on ownership
group** (clipped 0.2 to 5, fit on the window's own rows) and compares it with the unweighted model on OOF predictions,
never on test. Setting `bias.mitigation: reweight` makes `train` fit every member on those weights; it is a recipe
change, so it goes through the gates. Other remedies the guidelines name (re-sampling, per-group thresholds) are not
used: the score is a calibrated probability, and per-group thresholds would break calibration across groups.

| Variant (eval-window OOF, lgbm, 162,276 rows) | PR-AUC | Brier | ECE | largest gated \|O−E\| | miscalibrated groups |
|---|---|---|---|---|---|
| none (in use) | 0.523 | 0.1666 | 0.0050 | 0.018 | 0 |
| reweight (ESS 71,987) | 0.517 | 0.1678 | 0.0044 | 0.013 | 0 |

**Decision:** keep `none`. No group is miscalibrated, so there is nothing to fix; reweighting costs 0.006 PR-AUC to
move the largest gap from 0.018 to 0.013. The trade-off is recomputed and printed in every model card, so the
decision is revisited every month.

---

## 8. Hyperparameter tuning

- **Method:** Optuna, TPE sampler (Bayesian optimisation), seed 42, 30 trials, 60-minute cap, one at a time.
- **Objective:** mean held-out PR-AUC of the ensemble over **3 facility-grouped folds of `eval_train`**, with early
  stopping (100 rounds) per fold; the final tree count is the median best iteration. The test month is never used.
- **Search space** (`configs/ml.yaml: tuning.space`, from PR #49's `boost_models.py`):

| Parameter | Range | Parameter | Range |
|---|---|---|---|
| learning_rate | 0.005 to 0.1, log | subsample | 0.5 to 1.0 |
| num_leaves | 8 to 128 | colsample_bytree | 0.3 to 1.0 |
| max_depth | 3 to 12 | reg_alpha | 0.001 to 10, log |
| min_child_samples | 20 to 1,000 | reg_lambda | 0.001 to 30, log |
| min_split_gain | 0 to 1 | cat_smooth | 1 to 100 |
| LR `C` | 0.03, 0.1, 0.3, 1.0 | | |

- **Result** (run `tune-phase1-1`, production window, 889 s): best CV PR-AUC **0.5112** (trial 19); every trial is
  between 0.509 and 0.511 (median 0.5106). Best: learning rate 0.070, 99 leaves, depth 4, 102 trees, LR `C` 1.0.
- **Output and decision:** `tuning/tuning_proposal.yaml`, ready to paste into `ml.yaml`, but applied only through a
  reviewed PR (ML-19). The gain over the current parameters is within noise, so PR #49's parameters stay. In Phase 1 a
  re-tune runs only by hand (`mode=retune`); a drift flag emails a request for one.

---

## 9. Sensitivity analysis

### 9.1 Features (SHAP)
Contributions are exact: TreeSHAP for LightGBM, `w·(z − mean)` for LR, the weighted mean for the ensemble, scaled by
the Platt slope, so for every home base + Σ contributions = the calibrated log-odds (checked to 1e-14 by gate G2).

![Top 20 features by mean |SHAP|](img/ml/shap_top20.png)

- **Location and history dominate:** the home's state and the state's recent G+ rate (inspection practice differs by
  state), size, and the home's own complaint and citation history (complaint surveys cited, citations, recent G+
  surveys, days since the last G+).
- **Staffing and quality add the rest:** turnover, hours per resident day, the QM rating and individual measures.
- **Per home:** `ml score` writes the top 5 risk-raising and top 3 risk-lowering features of every home, with the
  value and a plain-language label (`configs/reason_labels.yaml`), plus the full SHAP matrix and a beeswarm plot per
  month. `scripts/ml_checkpoint_reasons.py` prints the same reasons for any registered version (Checkpoint ML-A).

### 9.2 Hyperparameters
fANOVA importances from the tuning run: **LR `C` 0.41, `max_depth` 0.23**, `cat_smooth` 0.07, `min_split_gain` 0.07,
`colsample_bytree` 0.07, the rest 0.04 or less (learning rate 0.01). Higher `C` (weaker LR regularisation, 22 of 30
trials) and shallow trees (depth 3 to 4) score best; the spread across all trials is only 0.002 PR-AUC.

| | |
|---|---|
| ![Hyperparameter importances](img/ml/param_importances.png) | ![Optimisation history](img/ml/optimization_history.png) |

![Parallel coordinates of the 30 trials](img/ml/parallel_coordinate.png)

---

## 10. Experiment tracking and the model registry

- **MLflow 3** runs as a compose service (backend: Postgres database `mlflow`; artifacts proxied to
  `gs://precursor-intel-mlops-mlflow/`). Tasks log over HTTP; the UI is local only.
- **One MLflow run per task** (`train`, `evaluate`, `gate`, `tune`, `score`) in experiment `precursor_y_g12`. Every run
  is tagged with the ML run id, gold run, `V`, `V_L`, `dataset_version`, gold verified, git sha, image, `recipe_hash`,
  decision and mode, and logs:
  - `train`: every hyperparameter, Platt parameters, windows, feature count and hash, excluded features, OOF metrics,
    PSI, timings; the input dataset; one model per candidate (MLflow pyfunc); selection, windows and input manifest;
  - `evaluate`: every metric of section 6 per candidate and baseline, the mitigation trade-off, the plots;
  - `gate`: G1 to G7 (`gates.json`) and the model card;
  - `tune`: the trials table, the proposal, importances and the three plots.
- **Registry:** model `precursor_y_g12`; `ml gate` registers the refit package as a new version, tagged with its run, gold and recipe, and
  aliases it `@challenger` or `@champion`. Scoring and rollback load the package from the run folder after checking
  its sha256 against `train.json`; MLflow's copy is never unpickled.
- **Audit:** every decision, gate result, promotion and publish is a row in the Postgres table `ml_runs`.

---

## 11. Gates, promotion and rollback

All seven gates are deterministic code (`evaluation/gates.py`, values in `configs/ml.yaml: gates`). Comparisons with
the champion are paired on the same test rows: "worse" means beyond the tolerance and the bootstrap CI of the
difference excludes zero.

| Gate | Passes when |
|---|---|
| G1 Data | the input checks passed; gold verified or `allow_unverified` |
| G2 Sanity | no uncleared single-feature leakage flag; SHAP additive; a model trained on shuffled labels scores ROC-AUC 0.48 to 0.52 |
| G3 Beats the rule | PR-AUC better than the prior-G+ rule (paired CI above 0) |
| G4 Calibration | test ECE ≤ 0.05 and calibration slope in 0.7 to 1.3 |
| G5 Not worse than the champion | PR-AUC not worse by more than 0.01, Brier not worse by more than 0.005 (skipped without a champion) |
| G6 Refit sanity | the refit's OOF lift ≥ 0.85 × the eval fit's; its mean score within ±0.05 of the recent observed rate; Spearman ≥ 0.9 with the eval fit on the score month |
| G7 Bias | section 7.2 |

- **Promotion:** all gates pass and a champion exists → the new version becomes `@champion`; otherwise it is
  registered as `@challenger`, the champion keeps scoring and an email says which gate held.
- **The first version** is never promoted automatically (`gates.first_run: challenger`): a person reads the model
  card, `gates.json`, the slice table and 20 homes' reasons (**Checkpoint ML-A**) and promotes it with
  `mode=promote`, `version` and `reason`.
- **Rollback:** the same `mode=promote` with an earlier version. Only a version that passed G1 and G2 can be promoted;
  every promotion and rollback is an `ml_runs` row, and `score` refuses a champion whose alias moved outside it.

---

## 12. CI/CD, notifications and containers

- **CI on every pull request** (`.github/workflows/ci.yml`): ruff and the offline test suite (1,500+ tests, with a
  leakage suite) on Python 3.11 and 3.12, and `dvc repro` of the **`toy_ml` stage: the whole chain**, prepare →
  train → evaluate → gate → register (a SQLite MLflow store) → score → publish, **on a synthetic gold**
  (`tests/ml/make_ml_gold.py`). A failed step or a gate that holds the model (validation or bias) fails the check, and
  the `ci-ok` check is required to merge. The images job builds the three images, runs the DAG tests in the Airflow
  image and starts MLflow in the compose bundle.
- **CD:** `.github/workflows/cd.yml` pushes the images to Artifact Registry on every merge to `dev` or `main` that changes code, configs or images; `deploy.yml` points the VM
  at a release (the VM runs the data pipeline today; the ML DAG moves there in Phase 2).
- **Notifications** (`dags/cms_nh_ml.py: notify`, `precursorintelligence.alerts`): every finished run emails its
  summary (decision, gate outcome, champion version, test PR-AUC and Brier with CIs, drift flags, a re-tune request);
  a failed run emails the failure (`on_dag_failure`).
- **Containers:** every step runs in `precursor-pipeline` (built from `uv.lock`, with the `ml` and `bias` extras); the
  same image runs in CI, in Airflow and by hand. MLflow has its own image (`docker/mlflow.Dockerfile`).
- **Reproducibility:** fits are seeded and deterministic; a run is keyed on gold's `dataset_version` and the
  `recipe_hash` (code + `ml.yaml` + feature list), and its pinned input stays in its run folder.

---

## 13. How to run it

**Offline, on synthetic data (no credentials, a few minutes):** the same chain CI runs.
```bash
uv sync --all-extras
uv run dvc repro --force --single-item toy_ml      # outputs in data/toy_ml/, summary in metrics/toy_ml.json
```

**On the real gold, step by step** (a GCS-rooted store at `ml-dev/`; MLflow is optional up to `evaluate`, but `gate`'s
registration and the reasons script need it):
```bash
export PI_BUCKET_URI=gs://precursor-intel-mlops-data/ml-dev PI_DATABASE_URL=sqlite:///data/ops/ml-local.db
export MLFLOW_TRACKING_URI=http://localhost:5000      # needed from gate on; unset skips tracking up to evaluate
for step in prepare train evaluate gate; do
  uv run precursor-pipeline ml $step --env prod --run-id my-run-1 || break
done
uv run python scripts/ml_checkpoint_reasons.py --run-id my-run-1 --version 1 --out reasons.csv
```

**In Airflow** (the laptop stack, `docs/runbooks/airflow-local.md`):
```bash
docker compose exec airflow-scheduler airflow dags unpause cms_nh_ml
docker compose exec airflow-scheduler airflow dags trigger cms_nh_ml --conf '{"mode": "retrain"}'
docker compose exec airflow-scheduler airflow dags trigger cms_nh_ml \
  --conf '{"mode": "promote", "version": 1, "reason": "ML-A approved by <name> after reading model_card.md"}'
```
`retrain` → train, evaluate and gate (the first version becomes `@challenger`); `promote` → the champion scores,
publishes and is monitored. A rollback is the same `promote` with an earlier version; `retune` runs the tuning alone.

---

## 14. The runs behind the numbers

All on the laptop integration stack `ml_dev` (Airflow 3 + MLflow + Postgres in Docker Compose, task containers on the
pipeline image built from `dev` at `a7cbf98`), reading the gold run `20261112T020010Z-543cdc`
(`dataset_version` `3f9bac9367ae18ec`), on 2026-11-16. Each step ran in a 7 GB container.

| Run (`ml/runs/<id>/`) | Config | Steps | Time, peak memory | MLflow |
|---|---|---|---|---|
| `repro-purge-1` | `ml.yaml` with `eval_gap: purge` (PR #49's setting), 120 features | prepare, train, evaluate (no gate: nothing registered) | 175 s / 3.0 GB, 390 s / 2.3 GB, 160 s / 2.0 GB | experiment `precursor_y_g12_reproduction`; train `3497311c`, evaluate `c5fb373a` |
| `tune-phase1-1` | `ml.yaml` (production window) | prepare, tune | 154 s / 3.1 GB, 889 s / 2.3 GB | experiment `precursor_y_g12`; tune `9fce913e` |

- The charts in `img/ml/` come from these runs; `bias_slices.png` and `shap_top20.png` were drawn from
  `repro-purge-1`'s `evaluation.json` and `shap_mean_abs.csv` with the pipeline's own plotting code (the bias chart
  was added in PR #69, after these runs).
- Since these runs, `staffing_rating_fn` is excluded (PR #69): the production recipe has 119 features.
- W2's full chain on the same gold (PR #62, a teammate's laptop): prepare → train → evaluate → gate in about 15
  minutes, every gate passing, registered as `@challenger`.

---

## 15. Known limitations and Phase 2

- **Labels lag 20 months**, so the newest test month is 2025-02 at `V_L` = 2026-10, and it is a development month.
  Checkpoint ML-A and threshold reviews use 2025-03 or later, from the late-November release.
- **Unverified gold is allowed** (`allow_unverified: true`, tagged on every run) until the data spec's §11
  verification gate exists.
- **Phase 2:** MLflow and the ML DAG on the GCP VM (the registry starts fresh there); BigQuery publish; the tuned
  challenger and an automatic re-tune; the performance trend and score drift in monitoring; per-state empirical-Bayes
  shrinkage; MLflow authentication (the UI is local only today); the API, the four agents and the dashboard.
- **States differ most** in calibration; they are reported, not gated, until per-state shrinkage lands.
