# Precursor Intelligence: ML Pipeline Specification (v1)

> **Purpose.** This is the single reference for building the Precursor Intelligence ML pipeline. It turns a published gold dataset into a calibrated risk score and a per-facility explanation for every US nursing home, every month. Runs are tracked and registered in MLflow and orchestrated by one Airflow DAG. It is written for the five engineers building it and for whoever operates it.
>
> **What it builds on:**
> - `docs/DATA_PIPELINE_SPEC.md` (the data spec): its gold dataset, labels, splits and leakage rules are this pipeline's input. This file never redefines a gold rule; where it needs one, it cites the data spec.
> - PRs #38 and #49 (feature engineering, EDA, model experiments): **the baseline for features, transforms, models and hyperparameters.** The best model in PR #49 is the base model here (§4, Appendix A).
> - The course brief "Model Development Guidelines" (`../Model_pipeline.pdf`): every requirement maps to a section (Appendix D).
>
> **Precedence when sources disagree:**
> 1. measured facts (counts, metrics, timelines), with their source;
> 2. the decisions in this file (§19) and, for anything about gold, the data spec;
> 3. the team skills (`ml-engineering`, `testing-evaluation`, `mlops-deployment`), `CLAUDE.md` and the team documents, as background. Known conflicts where this file wins are listed in §0.8.
>
> **The goal is a correct, complete, well-measured system with decent model performance, not a top-scoring model.** The target is PR #49's level (test PR-AUC ≈ 0.45–0.50 at a base rate of ≈ 0.26); model optimization comes after the system works (ML-22).
>
> **Working rules:**
> - At each 🛑 **CHECKPOINT**, a human reads the report and approves before the next step goes to production. Routine monthly runs have no human checkpoint: the gates in §9 stop them instead.
> - Every value that can change lives in `configs/ml.yaml` (W0, #54; Appendix B) or an existing config. Never hard-code a feature list, a threshold or a window.
> - Tuning, selection and every "keep or drop" choice use training-window data only. **The test window is looked at once per model, to report and gate, never to choose** (§3, §20).
>
> **Status:** v1.0, 2026-11-10. Design only; no ML code exists yet (`src/precursorintelligence/{training,evaluation,serving,monitoring}` are empty). Reviewed by an outside ML-systems reviewer, the leakage reviewer and the infra reviewer; their findings are applied. **Phase 1, everything the course asks for, running on a laptop stack against GCS, is due Sun 2026-11-15; Phase 2 (VM, BigQuery, hardening) follows** (§18).

---

## 0. Context

### 0.1 What the ML pipeline does
Every month, after the data pipeline publishes gold (Airflow asset `cms_nh_gold`), one Airflow DAG, `cms_nh_ml`:
1. **decides** what to do: new matured labels or new code → retrain; new features only → score; nothing new → skip (§15.2);
2. **pins** the gold run it reads, so a later data run can't change its input mid-run (§1.2);
3. **trains** three candidates (LightGBM, logistic regression, their ensemble), calibrates them, **selects** one after a bias check on training data, **evaluates** it on the newest matured months it never saw, checks **bias** across facility slices and **explains** it;
4. **gates** it: registers it in MLflow and promotes it to `champion` only if every deterministic gate passes; otherwise it stays a `challenger`, the current champion keeps scoring, and an email goes out;
5. **scores** every facility for the newest month with the champion, with its top reasons (SHAP), and rolls the scores up to chains;
6. **publishes** the scores to GCS (source of truth; BigQuery in Phase 2);
7. **monitors** drift and performance, raises a re-tune request when needed, writes `status/latest.json` and emails a summary.

There is no online model server: CMS publishes once a month and every score changes only then, so batch is the right design (§0.4).

### 0.2 Problem framing
| Item | Value | Source |
|---|---|---|
| Unit | facility (CCN, 6-character text) × prediction month `t` | data spec §0.3 |
| Target | `y_g12`: any health citation with scope/severity G–L in `(cutoff, cutoff + 12 months]` | data spec §0.2, §9.1 |
| Rows trained and evaluated | `status_g12 ∈ {positive, negative}` only; immature and censored rows never | data spec §9.2 |
| Rows scored | every facility at `t = V`, the newest vintage | data spec §9.5 (`score` rows) |
| Base rate | ≈ 25.6 % (PR #49 test month 2025-02, 14,643 facilities) | Appendix A |
| What the product needs | a **calibrated probability** per facility (insurers price on it), a **ranking**, and **reasons** per facility | product brief, `ml-engineering` skill |
| Headline metrics | PR-AUC vs the base rate, precision@5 % / @10 %, Brier, ECE; ROC-AUC, accuracy, precision, recall, F1 for the course | §6 |
| Other labels | `y_g6`, `y_pen12` exist in gold; out of scope in v1; nothing in the code is specific to `g12` (ML-1) | §19 |

### 0.3 Notation
| Term | Meaning |
|---|---|
| `V` | gold's newest vintage (`_current.json: v`) |
| `V_L` | the vintage maturity is measured from (`_current.json: v_labels`; equals `V` unless `V` is a stale refresh, data spec §9.2) |
| H, B | label horizon 12 months; posting buffer `B_cit` = 8 months (`configs/labels.yaml`) |
| `L` | last labelled month: `V_L − (H + B)` = 20 months before `V_L` (`gold/splits.py: last_labeled_t`) |
| test window | gold's `split_g12 == "test"` months: the last `test_months` calendar months up to `L` |
| eval window | the `train_months` most recent non-empty months ending no later than H + B months before the first test month: production's lag (§3.2) |
| refit window | the `train_months` most recent non-empty months up to `L` (§3.3) |
| recipe | everything that defines how a model is built: feature list, member hyperparameters, ensemble rule, calibration, windows, and the ML code and `configs/ml.yaml` (`recipe_hash`, §14.3) |
| champion / challenger | MLflow registry aliases on model `precursor_y_g12` (§9.2) |
| OOF | out-of-fold: a prediction for a row from a model that did not see that row's facility |

### 0.4 Design principles
1. **Batch, not online.** Scores change once a month, so they are computed once a month and stored. No endpoint, no autoscaling. "Deploying" a model means moving the `champion` alias; the next scoring uses it.
2. **Point-in-time, inherited and extended.** Gold already guarantees that features at `t` use only vintage `t` (plus `t−6`, `t−12`) and labels only events after the cutoff (data spec §0.4). The ML pipeline adds: an evaluation gap as long as the production lag (H + B); facility-grouped folds; no tuning, selection or mitigation choice on the test window.
3. **Deterministic gates, auditable promotion.** Promotion is code (`evaluation/gates.py`), not judgment. Every promotion, hold, manual promotion and rollback is a row in `ml_runs` and a tag in MLflow.
4. **Registry-driven features.** The feature list is `configs/feature_registry.yaml` entries with `model_use: model`, minus `ml.yaml: features.exclude` (§1.4). New features (e.g. the quality family) change the model by config, not code.
5. **Idempotent.** Runs are keyed on gold's `dataset_version` and the `recipe_hash`; a retry or rerun gives the same models (seeded) and overwrites the same score partition.
6. **One image, one DAG.** ML tasks run in the existing `precursor-pipeline` image as `precursor-pipeline ml <command>`; one DAG, `cms_nh_ml`, mirroring the data pipeline's one-DAG decision (data spec §14 #19).
7. **Simple enough to finish.** Three candidates already fitted for the ensemble, one calibration method, PSI-based monitoring. Other challengers (CatBoost, EBM, monotone constraints) are later experiments.

### 0.5 Defaults
| Setting | Default | Config key |
|---|---|---|
| Label | `g12` | `label` |
| Window length | 11 non-empty months (eval and refit) | `windows.train_months` |
| Evaluation gap | production lag (H + B = 20 months) before the first test month | `windows.eval_gap: production_lag` |
| Minimum training months | 6 | `windows.min_train_months` |
| Folds | 5 facility-grouped folds by a stable hash of `ccn` (3 for tuning) | `folds` |
| LightGBM / LR hyperparameters | PR #49's values (§4.2) | `members` |
| Ensemble | mean of the members' log-odds | `ensemble` |
| Calibration | Platt on the OOF predictions of the window's newest 3 months | `calibration` |
| Excluded features | `era`, `penalty_regime_2025_10` (§1.4) | `features.exclude` |
| Gates | §9.1 | `gates` |
| Bias slices | `configs/bias_slices.yaml` (shared with the data-level bias report) | `bias` |
| Reasons per facility | top 5 raising + top 3 lowering | `explain` |
| Drift flag | PSI > 0.25 on a key feature | `monitor` |
| Optuna | 30 trials, 3 folds, early stopping, 60 minutes | `tuning` |
| Unverified gold | allowed and tagged, until the data spec's §11 gate exists | `allow_unverified` |
| Output prefix | `""` (bucket root) on the VM; `ml-dev/` for Phase 1 laptop runs | `outputs.root_prefix` |

### 0.6 Architecture
```
 cms_nh_pipeline ──(asset cms_nh_gold)──► cms_nh_ml  (Airflow, LocalExecutor, one container per task)
                                            │
  <prefix>gold/ (GCS) ──pin──► <prefix>ml/runs/<run_id>/     decide → prepare → [tune] → train → evaluate
                               frames, run JSON, input         → gate | promote → score → publish → monitor → notify
                                            │
             ┌──────────────────────────────┼──────────────────────────────┐
             ▼                              ▼                              ▼
   MLflow server (compose service)   <prefix>scores/v=YYYY-MM/      <prefix>status/latest.json
   ├─ backend: Postgres db mlflow    facility_scores, reasons,       read by the API / UI
   ├─ artifacts: gs://precursor-     shap_values, chain_scores
   │  intel-mlops-mlflow (proxied)          │ Phase 2: batch load
   └─ registry: precursor_y_g12             ▼
      @champion / @challenger      BigQuery precursorintelligence_scores (API, agents, dashboard)
```

| Component | Where | Notes |
|---|---|---|
| Code | `src/precursorintelligence/{training,evaluation,serving,monitoring}` | roles as in `src/precursorintelligence/README.md`; modules per work package in §18.3 |
| CLI | `precursor-pipeline ml <command>` | a typer sub-app in `training/cli.py`, registered in `harmonize/cli.py` like `gold`, `bias-report`, `validate` |
| Config | `configs/ml.yaml` → `MlConfig` / `load_ml_config` in `common/config.py` | the existing loader pattern |
| Run state | `precursor_meta.ml_runs` (one row per decision, promotion, hold or rollback) | `common/io/schemas.py`; tasks reach Postgres only as `pi_meta` (L11 rule) |
| Tracking and registry | MLflow 3.17 server in the compose stack (§16.1) | tasks talk to it over HTTP |
| Scores | `scores/` under the output prefix | §10 |

### 0.7 The ML DAG's steps
| # | Step | Task(s) | Writes |
|---|---|---|---|
| M1 | Decide and pin gold | `decide`, `prepare` | decision JSON; `ml/runs/<id>/input/` |
| M2 | Windows and folds | `prepare` | `frames/{eval_train,test,refit_train,score,prev_score}.parquet`, `windows.json` |
| M3 | Train three candidates | `train` | MLflow runs; model packages |
| M4 | Calibrate (Platt, OOF) | `train` | the calibrator inside each package |
| M5 | Select the candidate after a bias check | `train` | `selection.json` |
| M6 | Evaluate | `evaluate` | metrics with CIs, baselines, plots, model card |
| M7 | Bias and slices | `evaluate` | slice report, gate inputs |
| M8 | Explain and sensitivity | `evaluate`, `tune` | global SHAP; hyperparameter importance |
| M9 | Gates, registry, promotion, rollback | `gate`, `promote` | registry version + alias; `ml_runs` row |
| M10 | Score, reasons, roll-up | `score` | `scores/v=YYYY-MM/…` |
| M11 | Publish | `publish` | asset `cms_nh_scores`; BigQuery in Phase 2 |
| M12 | Monitor and notify | `monitor`, `notify` | drift report, re-tune request, `status/latest.json`, email |
| M13 | Tune (a proposal) | `tune` | tuned challenger + `tuning_proposal.yaml` |

### 0.8 Deviations from the team skills, documents, data spec and course examples
| Source says | This spec | Why |
|---|---|---|
| `ml-engineering`: about 129 features; isotonic calibration on held-out facilities | 90 registry features (131 with the quality family; PR #49's 133 included the incontinence bridge W5 dropped), minus excluded regime flags; Platt on OOF predictions | measured registry; isotonic always wins on its own fitting rows and creates rank ties; PR #49 found it worsened test ECE (0.0246 vs 0.0218) |
| `ml-engineering`: champion LightGBM with monotone constraints; CatBoost/EBM challengers | base model = LightGBM + LR ensemble from PR #49 (ML-8), with selection among the three fitted candidates | the user chose PR #49's best model; LightGBM alone barely beats LR there (0.4791 vs 0.4781) |
| `ml-engineering`: tuning with time-based, facility-grouped folds | facility-grouped folds; rolling-origin later (O3) | a facility-grouped fold is the simple first step; time-based folds need more history inside the eval window |
| Skills and data spec §15.1: separate DAGs `train_evaluate`, `batch_score`, `monitor` | one DAG `cms_nh_ml` | one run = one pinned gold, one decision, one audit trail (data spec §14 #19 reasoning) |
| `ml-engineering`: `promote-model.yml` for manual override | `cms_nh_ml` run with `mode=promote` (§9.4) | GitHub Actions can't reach the VM (IAP only) |
| `mlops-deployment`: e2-standard-2, instance schedule, nightly dump | e2-standard-4, started by hand, dump before every stop | data spec §14 #26, #27 |
| `CLAUDE.md` (older copies): "star ratings are never features" | `staffing_rating`, `qm_rating` may be features (subject to the 2025-07 check, §1.4); `overall_rating`, `health_inspection_rating` never | ML-17: registry and data spec §8 decide; the wording is fixed (O6) |
| Data spec §9.5: purge rows "are not used at all"; gold's purge is H | the refit trains on purge and test months; the ML evaluation gap is H + B | the refit is never evaluated on them (§3.3); the production lag is H + B (§3.1) |
| Data spec §11.5: the dataset is frozen at Checkpoint 5 (`training_dataset_vN`) | every RETRAIN trains on the newest gold; the pinned input of each **registered** model version is kept and tagged as its `training_dataset` | monthly retraining (ML-2) needs each month's data; pinning gives the same reproducibility per version |
| `GOLD_HANDOFF.md`, `AIRFLOW_PIPELINE.md`: ML must refuse `verified: false` | allowed and tagged until the §11 gate exists (ML-9) | every gold is unverified today |
| Course: push to GCP Artifact Registry or a model registry | MLflow Model Registry (artifacts in GCS) | the brief allows "a model registry" |
| PR #49: probability average, uncalibrated, window chosen on test | log-odds average (ML-16), calibrated (§5), windows fixed in config | exact SHAP; probabilities are the product; no selection on test |

---

## 1. Inputs: the gold contract

### 1.1 What ML reads
| Object | Content | Used for |
|---|---|---|
| `gold/_current.json` | `{state, run_id, v, v_labels, verified, sha256: {file → hash}}` | which run to read; hash checks |
| `gold/dataset_manifest.json` | `dataset_version`, `config_sha256`, `git_commit`, `rows_per_split`, `merged_file {path, sha256}`, `flags` (`stale_refresh`, …) | run key, provenance, the decide step |
| `gold/facility_month_dataset.parquet` | every month in one file: `ccn, t, cutoff_date`, the feature columns, `y_*`, `status_*`, `split_*` | the only data file ML reads |

Never read: the per-family partitions (the merged file has them), `gold/spine`, `gold/labels`, and **`gold/dim_facility_latest`** (data spec §8: "ML training must never join `dim_facility_latest`").

### 1.2 Pinning gold
`decide` records the gold `run_id` it decided on. `prepare` then copies gold into the run folder, using the pattern of `gold/snapshot.py: snapshot_gold` and `bias/io.py: load_gold_store`:
1. read `_current.json`; refuse unless `state == "current"` and `run_id` equals the one `decide` recorded (exit 1, retried; a newer gold triggers a new `cms_nh_ml` run anyway);
2. refuse unless `dataset_manifest.merged_file.sha256 == _current.sha256["facility_month_dataset.parquet"]` (exit 1);
3. copy the merged parquet, the manifest and `_current.json` to `ml/runs/<run_id>/input/`, hashing while copying (exit 1 on a mismatch);
4. re-read `_current.json`; exit 1 if its `run_id` changed during the copy.

Every later task reads only the pinned copy, so `cms_nh_ml` needs no run lock and can run while `cms_nh_pipeline` builds the next gold. The input of a run that registers a model version is kept for good (that version's training set); other run folders may be deleted after 90 days.

### 1.3 The `verified` flag (ML-9)
Gold writes `verified: false` until the data spec's §11 gate and Checkpoint 5 exist (`gold/build.py: VERIFIED = False`). With `allow_unverified: true` (the default until then) the pipeline trains and scores; every MLflow run, model version, score row and `status/latest.json` carries `gold_verified = false`. Once the gate exists, prod sets `allow_unverified: false` and an unverified gold is refused (exit 2, alert).

### 1.4 Features
- **The feature list** = every `configs/feature_registry.yaml` entry with `model_use: model` (`common/config.py: load_feature_registry`), in registry order, minus `ml.yaml: features.exclude`. Today 121 features: 90 − 2 = 88, plus 33 of the quality family's 41 columns (W5, #55); 120 if W1 also excludes `d12_turnover_total` (below). For the quality family, `features.exclude` also takes out its two month keys (`qm_age_months_*`), its two definition flags (`qm_def_post_*`) and the values of the four measures whose scale changed with a new definition.
- **Never features:** `ablation_only` (`overall_rating`, `health_inspection_rating`, their `_fn` footnotes), used only as the star-rating benchmark (§6.2); `product_only` (`chain_id`), used only by the roll-up (§10.3); `label_aux` columns; `y_*`, `status_*`, `split_*`; `ccn`, `t`, `cutoff_date`.
- **Regime flags are excluded until training has seen both values (ML-21).** `era` (2025-07 rating-method change) and `penalty_regime_2025_10` are pure functions of `t` (`gold/features/context.py`). Every eval and refit window until `V_L` ≈ 2027-03 / 2027-06 sees only the old value, while every scored month has the new one: the columns are constant in training, LightGBM never splits on them and LR gives them ~0 weight. They are listed in `features.exclude`. `prepare` runs a **regime-coverage check** on every remaining `t`-derived or regime-sensitive feature (`features.regime_sensitive`, default `[staffing_rating, qm_rating, staffing_rating_fn, qm_rating_fn]` plus the quality family's `qm_def_post_*` flags when they land): any level or range in the score frame that the refit window doesn't cover is a monitoring flag (§12.1) and a model-card line.
- **The two star sub-ratings (ML-17 + ML-21):** CMS changed the rating method in 2025-07, so `staffing_rating` and `qm_rating` are trained on the old method and scored on the new one. W1 measures their PSI between the refit window and the score month; above 0.25 they go into `features.exclude` until the refit window covers the new method. Checkpoint ML-A confirms.
- **A late-starting feature: `d12_turnover_total` (O9; W1 decides).** It is the 12-month change in staff turnover, so it needs turnover at `t − 12`, which CMS publishes from 2022-01 (registry `first_available_vintage: "2023-01"`).
  - **Measured on the P0 gold** (2026-11-11): null in every 2022 month, 74 % filled in 2023-01, rising to 88 % by 2026-10.
  - **Why it matters:** the eval window at `V_L` = 2026-10 (2022-05..2023-04) has it empty in 7 of its 11 months, while the refit window and every scored month have it. The eval fit and the champion would learn it differently, and LR's missing indicator would act as a time flag.
  - **Recommended:** list it in `features.exclude` for Phase 1, like the regime flags (ML-21). Take it back once the eval window starts in 2023-01 or later (about `V_L` = 2027-05). State the choice in the model card.
  - It is the only `model_use: model` feature that starts after 2022-03.
- **Model frame.** One function, `training/data.py: to_model_frame(df, features)`, builds every model input (training, OOF, scoring, signature): numeric and bool columns cast to float64 (bool → 0/1, nulls stay NaN), categoricals to string with null as its own level. The same function everywhere avoids schema mismatches between training and scoring (e.g. an all-non-null sample inferred as `long` that later receives a null).

### 1.5 Input checks (`prepare`)
| Check | Rule | On failure |
|---|---|---|
| Columns | every listed feature is present with its registry dtype; the feature matrix holds no `y_*`, `status_*`, `split_*` or `_fwd` column (the rule of `bias/slices.py: check_no_leakage`) and nothing from `dim_facility_latest` | exit 2 |
| Rows | `ccn, t` unique; `ccn` matches `^[0-9A-Z]{6}$` | exit 2 |
| Labels | eval, test and refit frames each have positive and negative rows; at least `min_train_months` months in the eval window | exit 2 |
| Score rows | at least `checks.min_score_facilities` (10,000 on real gold; the universe is ~14.7–15.3k) | exit 2 |
| Order of `V` | `V` is not earlier than the last scored `V` in `ml_runs` (an older gold would overwrite newer scores) | exit 2 |
| Single-feature leakage flag | on the eval window and on the refit window, each feature's own PR-AUC, taking max(AP(x), AP(−x)), with nulls as their own value and categoricals scored by their out-of-fold target mean (the level's positive rate in the other facility folds, so a row never scores its own label), above min(0.6, 3 × base rate) | **flag**: written to the report; holds promotion at G2 until the feature is listed, with a reason, in `ml.yaml: checks.leakage_cleared` |

---

## 2. Step M1: decide and pin (tasks `decide`, `prepare`)
- `decide` reads `_current.json` and the manifest from the store (cheap, read-only), the last `ml_runs` rows and the champion from MLflow, applies the decision table (§15.2) and prints its result as JSON on its last stdout line (`decision`, `proceed`, `reason`, `gold_run_id`, …). The DAG reads it with `dags/utils/pipeline_operator.py: last_json_object(output, required)` (added by PR #48).
- `prepare` pins gold (§1.2), runs the checks (§1.5), builds the frames (§3) and writes `ml/runs/<run_id>/input/manifest.json` (pinned `run_id`, `v`, `v_labels`, `dataset_version`, `verified`, merged-file sha256, rows per frame, window months, flags).

---

## 3. Step M2: windows and folds (task `prepare`)

### 3.1 Why these windows
- Labels mature 20 months after `t` (H 12 + B 8). In production, the model that scores month `V` has training rows up to `L = V − 20`: **the production lag is 20 months.**
- Models decay with distance in time. PR #49 measured ROC 0.733 → 0.695 and ECE 0.7 → 5.5 points over 7 months out of time. So the evaluation leaves the same 20-month gap between its last training month and the first test month (20–22 months to the test window's three months), or the test metrics, and especially the calibration gate, describe a fresher model than the one that ships (ML-20).
- PR #49 also found that **recent training months predict better** (2023 only beat 2022-01..08 by +0.016 PR-AUC), so windows are short and recent: 11 non-empty months.

### 3.2 Frames
| Frame | Rows | Purpose |
|---|---|---|
| `eval_train` | labelled rows in the eval window: the `train_months` most recent non-empty months ending at or before `test_from − (H + B)` (`eval_gap: production_lag`: its newest labels are known by the first test month, as production's are by V_L), and not before `spine_start` | the eval fit |
| `test` | `split_g12 == "test"`, labelled rows | evaluation and gates |
| `refit_train` | labelled rows in the refit window: the `train_months` most recent non-empty months up to `L` | the production refit |
| `score` | every row with `t = V` | scoring |
| `prev_score` | every row with `t` = the previous scored month | score-drift monitoring with the same model (§12.1) |

- **Non-empty months:** CMS never published some months (2021-12, 2022-12, 2023-12, 2024-12, 2025-01, 2025-08, 2026-01). Windows count months that have rows. When `spine_start` binds, a window can hold fewer than `train_months`; fewer than `min_train_months` exits 2.
- **This needs gold to start in 2022-01** (`configs/splits.yaml: spine_start: "2022-01"`, the data spec's own recommendation for its open item #10; ML-20). With 2023-01 the eval window would have 4 months. `eval_gap: purge` (end before gold's purge window, 13+ months) exists for reproducing PR #49 and as a fallback; the model card states which gap was used.
- **Worked example,** `V_L = 2026-10` (the 2026-10-28 release): `L = 2025-02`; test window 2024-12..2025-02, of which only 2025-02 has rows (14,643 facilities); eval window = 11 non-empty months ending 2023-04 (2022-05..2023-04 without the 2022-12 gap); refit window = 2024-02..2025-02 without 2024-12 and 2025-01. The eval model predicts 2025-02 from data that ends 22 months earlier; the production model predicts 2026-10 from data that ends 20 months earlier.
- **2025-02 is a development month.** PR #49 chose its training window, features and parameters by test score on 2025-02, and ML-16 was measured on it. Metrics are reported **per test month**; months in `evaluation.development_months` (default `["2025-02"]`) are marked as such in `evaluation.json` and the model card. Checkpoint ML-A and the threshold review (O2) use 2025-03 or later, which enters the test window with the late-November release (`V_L = 2026-11`).

### 3.3 Eval fit and refit (ML-15)
Every RETRAIN fits the recipe twice:
1. the **eval fit** on `eval_train` → evaluated on `test` (§6), bias-checked (§7), gated (§9): the evidence;
2. the **refit** on `refit_train` → registered and, if the gates pass, the champion that scores. Its data is about two years newer than the eval fit's (22 months in the example of §3.2).

The refit is never evaluated on the test window (it trained on it), and **the registered champion (a refit) is never scored on a test frame** (`evaluate` refuses any model whose `meta.json` training months reach the test window minus H). Comparisons with the champion use the **champion's eval fit**, logged with every version: it trained on months that end at least H + B before its own test window, which ends before the current one, so scoring it on the current test frame is leak-free (§9.1 G5).

### 3.4 Folds
- `fold = int(sha256(ccn)[:8], 16) % K`: stable across months and runs; every facility always in the same fold.
- K = 5 for OOF calibration and the selection step; K = 3 for tuning (`folds.k`, `tuning.folds`).
- **Why facility-grouped:** a facility's consecutive months share overlapping label windows; grouping keeps all of a facility's rows on one side. This removes the leak behind data open #16 (same-facility shared events).
- **Limit:** folds still share calendar months, so OOF metrics are in-period estimates (state-by-period shocks are shared). The test window is the out-of-time check; the model card records the OOF-minus-test PR-AUC gap. Rolling-origin folds are open item O3.

---

## 4. Step M3: training (task `train`)

### 4.1 The base model (ML-8)
PR #49's best model (`notebooks/final_rev001/`, commit `4a0b7e0a`): an equal-weight ensemble of tuned LightGBM and logistic regression on 133 raw gold features, trained on 2023-01..2023-11, tested on 2025-02 (14,643 facilities, base rate 0.256):

| | PR-AUC | ROC | Brier | ECE | P@5 % | P@10 % |
|---|---|---|---|---|---|---|
| LightGBM alone | 0.4791 | 0.7080 | 0.1701 | 0.0332 | 0.669 | 0.579 |
| LR alone | 0.4781 | 0.7073 | 0.1693 | 0.0271 | 0.671 | 0.580 |
| Ensemble, probability average (PR #49) | 0.4813 (CI 0.465–0.499) | 0.7092 | 0.1689 | 0.0268 | 0.682 | 0.579 |
| **Ensemble, log-odds average (this spec)** | **0.4813** | 0.7091 | 0.1689 | 0.0274 | 0.682 | 0.579 |

The last row was measured on PR #49's `final_model/test_predictions.csv` (2026-11-10): paired-bootstrap 95 % CI of the PR-AUC difference [−0.0003, +0.0003], rank correlation 0.99993, identical top 10 % (ML-16). These numbers are on a development month (§3.2) and a 15-month gap; expect our honest numbers (20-month gap, 121 features) to be somewhat lower.

Not carried over from the PRs (Appendix A): the 7 extra CMS tables and PBJ (no gain); `state_te` and row-count lags (leaky or wrong); batch-relative rank scores; the uncalibrated output; choosing the window on test.

### 4.2 Candidates and members
**Logistic regression** (PR #49 `exp8_lr.py`, one scikit-learn `Pipeline`): numeric columns median-imputed with a missing indicator each, then `StandardScaler`; categoricals `OneHotEncoder(min_frequency=50, handle_unknown="infrequent_if_exist")`, null as its own level; `LogisticRegression(C=0.3, max_iter=2000)`; no class weights.

**LightGBM** (`lgbm_best_params.json` from PR #49, binary objective):
| Param | Value | Param | Value |
|---|---|---|---|
| learning_rate | 0.01024 | subsample, freq 1 | 0.634 |
| num_leaves | 34 | colsample_bytree | 0.335 |
| max_depth | 3 | reg_alpha | 0.0068 |
| min_child_samples | 870 | reg_lambda | 9.48 |
| n_estimators | 1,265 (no early stopping) | min_split_gain | 0.435 |
| cat_smooth | 2.71 | seed, determinism | 42; `deterministic`, `force_col_wise`, `num_threads: 4` |

Categoricals use LightGBM's native handling with levels fixed at fit time (an unseen level becomes missing). No resampling and no class weights: the base rate (~26 %) isn't extreme and the probabilities must stay calibrated. These parameters were tuned on another feature set; re-tuning is a proposal (§13), not a requirement.

**Ensemble (ML-16):** `logit = (logit_lgbm + logit_lr) / 2` (weights in config), then calibration.

### 4.3 Selection after the bias check (M5; course §2.2, §8.6)
All three candidates (ensemble, LightGBM alone, LR alone) are fitted anyway. On the eval window's OOF predictions:
1. drop any candidate that fails the bias check G7(b) on OOF predictions;
2. keep the one with the highest OOF PR-AUC; a simpler candidate wins ties within 0.002 (LR < LightGBM < ensemble in complexity).

The choice and the table go to `selection.json` and the model card. The test frame is not used to select. The refit and registration then use the selected candidate. Expect the ensemble in most months; LR alone is the natural fallback.

### 4.4 What one RETRAIN fits
| Fit | Data | Fits |
|---|---|---|
| eval fit, 3 candidates | `eval_train` | 1 full + 5 fold fits per member (OOF for calibration, selection) |
| refit, selected candidate | `refit_train` | 1 full + 5 fold fits per member (OOF for Platt and G6) |

With these parameters LightGBM is expected to take about a minute per fit on ~130–170k rows and 4 vCPUs, LR seconds; a RETRAIN (≤ 12 LightGBM fits) should finish within ~20 minutes and the 10 GB task memory limit. W1 measures it and sets the `execution_timeout`s.

### 4.5 The model package
One MLflow pyfunc, `training/ensemble.py: PrecursorModel(mlflow.pyfunc.PythonModel)`, logged with:
- artifacts: `lgbm.txt` (LightGBM text format), `lr.joblib`, `calibration.json` (Platt slope and intercept), `meta.json` (`package_format: 1`, candidate kind, feature list and dtypes, categorical levels, member weights, training months of both windows, `recipe_hash`, library versions);
- a signature from `to_model_frame` on a sample, and an input example of 5 rows;
- `predict(df)` → `p` (calibrated), `logit` (calibrated log-odds), `logit_raw`, and the member logits;
- `unwrap_python_model().explain(df)` → per-feature contributions (§8.1).

Every version must stay loadable by later code (`package_format` is checked; a test loads a stored version-1 package). Eval fits are logged as run artifacts; only the refit is registered.

---

## 5. Step M4: calibration (task `train`)
- **Method: Platt scaling** on the out-of-fold ensemble logits: for each of the 5 facility-grouped folds, fit the members on the other folds and predict the held-out fold; fit `p = σ(a · logit + b)` on the OOF rows of the window's **newest `calibration.fit_recent_months` (3) months**, closest to when the model is used.
- **Why Platt only:** two parameters on ~40k rows can't overfit, it keeps the ranking strictly unchanged (no ties), and the reasons stay exactly additive (§8.1). Isotonic is computed as a diagnostic (OOF Brier and ECE in the report), never chosen: on its own fitting rows it always wins by construction.
- If `a ≤ 0` (a degenerate fit), fall back to identity and flag.
- **This resolves data open #16** inside the ML layer: no separate calib window, so gold's calib months simply belong to the training windows.
- **Recorded:** OOF Brier and ECE raw / Platt / isotonic; `a`, `b`; test ECE and Brier before and after.
- The calibrator is fitted on fold models' OOF logits and applied to the full-window model; this is standard and acceptable.

---

## 6. Step M6: evaluation (task `evaluate`)

### 6.1 Metrics (test frame, eval fit; per test month and overall)
| Group | Metrics |
|---|---|
| Ranking | PR-AUC and **lift** (PR-AUC ÷ base rate); precision and recall at the top 1 / 2 / 5 / 10 / 20 %; lift at 2 / 5 / 10 %; ROC-AUC |
| Probability | Brier, Brier skill score vs the base rate, ECE (10 equal-mass bins), calibration slope and intercept, reliability table |
| Classification (course) | accuracy, precision, recall, F1, MCC and the confusion matrix at one threshold chosen on the OOF predictions (max F1), never on test |
| Within-state | PR-AUC and precision@10 % within each state, averaged |
| Candidates | the core metrics for all three candidates (reported; selection used OOF) |
| OOF vs test | the OOF-minus-test PR-AUC gap |

Functions are ported from PR #49 (`exp_common.py`, `final_model_metrics.py`) into `evaluation/metrics.py`, tested against scikit-learn where one exists.

### 6.2 Baselines
| Baseline | Definition |
|---|---|
| Prior-G+ rule (G3) | `gplus_last_std` or `n_gplus_surveys_36m > 0` as a 0/1 score; and `sev_points_24m` as a ranking score |
| LR alone | the LR candidate (reported) |
| Star-rating benchmark | `overall_rating` (lower = riskier) as a ranking score; a benchmark only, never a feature |

### 6.3 Uncertainty
Facility block bootstrap, 1,000 replicates, seed 0 (`evaluation/bootstrap.py`, from PR #49 `exp8::cluster_boot`): a 95 % CI on every headline metric, and **paired** CIs (same resampled facilities) for every comparison used by a gate.

### 6.4 Sanity checks
- **SHAP additivity** on 1,000 test rows (§8.1), every run.
- **Shuffled labels:** when a `recipe_hash` is evaluated for the first time (and in CI), refit LightGBM (300 trees) with labels shuffled within month; test ROC-AUC must be in 0.48–0.52. It catches evaluation bugs, not feature leakage.
- **Single-feature leakage flags** (§1.5).

### 6.5 Report
`evaluation.json` (every number above) and, in MLflow: PR and ROC curves, reliability diagrams (before and after calibration), lift chart, confusion matrix, candidate comparison chart, per-month table, and **`model_card.md`**: data (gold run, windows, evaluation gap, development months), recipe, selected candidate and why, metrics with CIs vs baselines and vs the champion's eval fit, calibration, bias summary, top global features, regime and drift flags, known limitations (§21), gate outcome.

---

## 7. Step M7: bias and slices (task `evaluate`)
- **Slices:** `configs/bias_slices.yaml` (state, cms_region, ownership_group, bed_band, chain_status, in_hospital, ccrc, sff_status, star_band, and its crossings), built with `bias/slices.py: build_slice_columns`, the same definitions as the data-level report (`precursor-pipeline bias-report`). Slices are facility attributes and state inspection practice; the data has no demographic attributes.
- **Tool:** Fairlearn `MetricFrame` (fairlearn 0.15, already used by `bias/disparity.py`).
- **Per slice:** rows, facilities, positives, base rate, mean predicted probability, **O−E** (observed − expected rate, i.e. calibration within the group) with its CI, PR-AUC (≥ 30 positives), and at the global top-10 % cut-off the share flagged, precision, recall, FPR and FNR. Slices below `thresholds.min_slice_facilities` (30) are reported and marked unreliable.
- **Why calibration within groups is the criterion:** the score is a risk probability used for pricing, and real risk differs between groups (for-profit 28.6 % vs non-profit 21.9 %, PR #38). Equal flag rates would be wrong; an accurate probability in every group is the goal. Precision and FNR are secondary disparity measures.
- **States:** PR #49 found per-state O−E with a standard deviation of 7.6 points (18 of 34 eligible states miscalibrated). The report shows per-state O−E shrunk with PR #49's empirical-Bayes method (`state_bias_audit.py`, DerSimonian–Laird) and lists the outliers.
- **Mitigation:** one documented option, **reweighting** with `bias/mitigation.py: sample_weights` (inverse frequency on `ownership_group`, clipped 0.2–5). `fit_rows` there selects `split == "train"`; W2 adds a parameter so the weights are fit on each frame's own rows. The trade-off (overall PR-AUC and Brier vs slice gaps) is measured on the eval window's OOF predictions, not on test, and written to the report. Adopting it is a recipe change (`ml.yaml: bias.mitigation: reweight`) that goes through the gates.
- **The bias gate G7** (ML-13) is in §9.1; the selection step uses its OOF version (§4.3).

---

## 8. Step M8: explainability and sensitivity

### 8.1 Per-facility explanations (computed at scoring, checked in evaluation)
| Member | Method | Exact? |
|---|---|---|
| LightGBM | `booster.predict(X, pred_contrib=True)`: TreeSHAP contributions in log-odds plus the expected value | exact |
| LR | each transformed column's `w_j · (z_j − mean_j)`, with means over the training rows, summed back to its source feature (the scaled column + its missing indicator; all one-hot columns of a categorical) | exact |
| Ensemble | the weighted average of the members' contributions and base values | exact (ML-16) |
| After calibration | contributions and base multiplied by the Platt slope `a` (base also gets `+ b`) | exact: base + Σ contributions = the calibrated log-odds |

- Contributions are in **calibrated log-odds**, never presented as probability points; the calibrated probability is shown next to them.
- A null feature has a contribution too (e.g. "no standard survey on record"); the reason text handles it.
- **Reason codes:** the 5 largest positive ("raises risk") and 3 largest negative ("lowers risk") contributions per facility (ML-14), each with the feature value, the contribution and a plain-language label from **`configs/reason_labels.yaml`**: `feature → {label, value_format, null_text}`, for example `n_cit_GI_24m: {label: "actual-harm (G–I) citations in the last 24 months", value_format: "{:.0f}"}`. The registry schema forbids extra fields (`FeatureSpec`, `extra="forbid"`), so the labels live beside it. A test lists model features without a label; they fall back to the feature name.

### 8.2 Global importance (per model version)
Mean |contribution| per feature on the test frame (eval fit) and on the score frame (champion), as a CSV and a bar chart, plus a SHAP summary (beeswarm) plot; the top 20 go into the model card. The champion's top-15 features are its **key features** for drift monitoring (a model-version tag).

### 8.3 Sensitivity analysis (course §5)
- **Features:** the global SHAP ranking; contributions per feature family (sum of |contribution| per family) in the model card.
- **Hyperparameters:** every tune (§13) logs Optuna's parameter importances (fANOVA), the optimization history and a parallel-coordinate plot.

---

## 9. Step M9: gates, registry, promotion, rollback (tasks `gate`, `promote`)

### 9.1 Gates (`evaluation/gates.py`; deterministic; values in `ml.yaml: gates`)
All comparisons with the champion are **paired on the current test frame**: the champion's logged eval fit is scored on the same rows (§3.3). "Worse" means the point estimate is beyond the tolerance **and** the paired-bootstrap 95 % CI of the difference excludes zero.

| # | Gate | Rule (default) |
|---|---|---|
| G1 | Data | input checks passed (§1.5); gold `verified` or `allow_unverified` |
| G2 | Sanity | no uncleared single-feature leakage flag; SHAP additivity holds; shuffled-label check in range when it ran |
| G3 | Beats the rule | selected candidate's PR-AUC better than the prior-G+ rule's (paired CI of the difference above 0) |
| G4 | Calibration | test ECE ≤ 0.05; calibration slope in [0.7, 1.3] |
| G5 | Not worse than the champion | not worse in PR-AUC by more than 0.01, nor in Brier by more than 0.005 (skipped when there is no champion) |
| G6 | Refit sanity (the model that ships) | refit OOF lift ≥ 0.85 × eval-fit OOF lift; on the score frame, mean `p` within ±0.05 of the observed rate of the refit window's newest 3 months; Spearman correlation of refit and eval-fit scores on the score frame ≥ 0.9 |
| G7 | Bias (ML-13) | gated slices: `bias.gate_slices` (default state, ownership_group, bed_band, chain_status, in_hospital) with ≥ 200 facilities and ≥ 50 positives in the test frame. (a) vs the champion: no slice's \|O−E\| grows by more than 0.03, nor its PR-AUC drops by more than 0.05 ("worse" as above, with Benjamini–Hochberg at 5 % across the gated slices; skipped without a champion). (b) Absolute: no non-geographic gated slice's O−E CI lies entirely outside ±0.05 |

- Any gate fails → **hold**: the version is registered as `challenger`, the champion stays, an email names the failed gates. A hold is exit 0, not a failure.
- **First run (no champion):** G5 and G7(a) are skipped; the version becomes `challenger` and 🛑 **Checkpoint ML-A** applies (§17.3): a human promotes it with `mode=promote`. Until then nothing is scored (§15.2).
- The thresholds are starting values; Checkpoint ML-A reviews them on the first real numbers (O2) and changes them only in `ml.yaml`.

### 9.2 Registry (MLflow Model Registry)
- Model name `precursor_y_g12` (case-sensitive), one per label.
- **Registration happens in `gate`**, idempotently: it first searches for a version tagged with this `ml_run_id` and reuses it (a retried task never creates a second version).
- Version tags: `ml_run_id`, `recipe_hash`, `candidate`, `eval_run_id` (the run holding the eval-fit artifact used for later paired comparisons), `gold_run_id`, `dataset_version` (also as `training_dataset`), `v_labels`, `gold_verified`, `gate_outcome`, `failed_gates`.
- All gates pass → alias `champion` moves to it. Any gate fails → alias `challenger`.

### 9.3 Audit
Every decision, promotion, hold, manual promotion and rollback is a row in `precursor_meta.ml_runs` (§15.2: `action`, `champion_version_before`, `champion_version_after`, `gate_results_json`, `actor`, `reason`). Append-only.

### 9.4 Rollback and manual promotion
- **Automatic:** a model that performs worse never becomes champion (G3–G7): the course's "rollback mechanism in case the newly trained model performs worse".
- **Manual:** trigger `cms_nh_ml` with `mode=promote`, `version=<N>` and a reason. It moves `champion` to version N (any earlier version, so this is also the rollback), writes the `ml_runs` row (`manual_promote` or `rollback`), rescores the current month, publishes and emails. Only versions that passed G1–G2 can be promoted by hand.
- **Integrity check:** `score` refuses (exit 2, alert) unless `@champion` equals the version of the latest `promote`, `manual_promote` or `rollback` row in `ml_runs`. This catches an alias moved by hand in the MLflow UI, outside the gates.

---

## 10. Step M10: scoring and chain roll-up (task `score`)

### 10.1 Scores
- `score` loads the **explicit champion version** recorded in the run's `ml/runs/<id>/champion.json` (written by `decide`, `gate` or `promote`), not whatever `@champion` points to at load time; a retried task scores with the same model.
- **`facility_scores.parquet`:** `ccn, t, score_month (date), state, chain_id, p_g12 (calibrated), logit, risk_percentile, risk_decile (1 = highest), model_name, model_version, candidate, gold_run_id, dataset_version, gold_verified, stale_refresh, ml_run_id, scored_at`.
- Percentiles and deciles are within the month; `p_g12` is the absolute, calibrated number.
- At a stale refresh (`V ≠ V_L`), event-based features are copied from `V_L`; scores are produced and stamped `stale_refresh = true` (also in `status/latest.json`).

### 10.2 Reasons and SHAP
- **`facility_reasons.parquet`** (long): `ccn, t, model_version, direction (raises | lowers), rank, feature, feature_value (text), contribution_logodds, label`; 8 rows per facility.
- **`shap_values.parquet`** (wide): `ccn, t, model_version, base_value`, one column per feature.

### 10.3 Chain roll-up (ML-11)
Per `chain_id` (facilities without a chain get no chain row), in **`chain_scores.parquet`**: `n_facilities`, `beds` (Σ `certified_beds`), `expected_facilities_with_event` (Σ p), `bed_weighted_mean_p`, `max_p` and `worst_ccn`, `share_top_decile`. No model is trained at chain level. A score-band → action mapping, if any, is a fixed rule table outside ML.

### 10.4 Layout and idempotency
All four files go to `<prefix>scores/v=YYYY-MM/` with `scores_manifest.json` (model version, gold run, rows, sha256 per file, `ml_run_id`). A rescoring of the same month overwrites the folder; bucket versioning keeps the previous files.

---

## 11. Step M11: publish (task `publish`)
- **Phase 1:** verifies `scores_manifest.json` against the files and emits the Airflow asset **`cms_nh_scores`**.
- **Phase 2 (BigQuery, ML-4):** dataset `precursorintelligence_scores` (us-central1, Terraform), tables `facility_scores`, `facility_reasons`, `chain_scores`, monthly-partitioned on `score_month`, clustered by `state` where present. Loaded with a **load job** from the parquet files (never streaming inserts; data spec D6), `WRITE_TRUNCATE` on the month's partition (decorator `table$YYYYMM`), so a rerun replaces exactly that month. The SHAP matrix stays in GCS.

---

## 12. Step M12: monitoring and re-tune requests (tasks `monitor`, `notify`)

### 12.1 Signals (each run)
| Signal | How | Flag |
|---|---|---|
| Feature drift, month over month | PSI (10 quantile bins of the reference, nulls as their own bin; categoricals by level) of each key feature, score month vs previous month | any PSI > 0.25 |
| Training vs scoring skew | PSI of key, enforcement and regime-sensitive features (§1.4), refit window vs score month; regime-coverage gaps | any PSI > 0.25 or a coverage gap |
| Score drift | the **current** champion scores both `score` and `prev_score`; PSI of `p` and the change in mean `p` (isolates data change from model change) | PSI > 0.1, or the mean moves by > 0.03 |
| Performance trend | the eval fit's test **lift** per test month vs the median of earlier recorded months | a drop of more than 15 % |
| Stale champion | the champion's `v_labels` vs the current `V_L` | more than 3 months behind (repeated holds) |
| Realized performance (Phase 2) | when published scores mature (20 months later): PR-AUC, Brier, ECE of those exact scores; an early read of decile lift after 3–6 months | reported |

PSI is computed in code (`monitoring/drift.py`, tested). Month over month is the trigger, not a fixed old reference: PR #38 found 22–35 features with PSI > 0.25 against 2022–2023, which would fire every month. An Evidently `DataDriftPreset` HTML report is an optional artifact (Phase 2).

### 12.2 Re-tune requests (ML-2, ML-19)
A feature-drift, score-drift or performance flag sets `retune_requested = true` in this run's `ml_runs` row and emails. The next `decide` with `mode=auto` returns RETUNE: Optuna runs (§13), the tuned model is evaluated through the gates and registered as a **challenger only**, with a report. The tuned parameters become the recipe only through a human PR to `configs/ml.yaml` (CD → new image → new `recipe_hash` → RETRAIN → gates). Drift alone never promotes or demotes a model; the monthly refits already adapt to new data. Training-vs-scoring skew and stale-champion flags alert a human (they are not fixed by re-tuning).

### 12.3 Status file and notifications
- **`<prefix>status/latest.json`** (written last by `monitor`): `{v, v_labels, scored_at, model_name, model_version, candidate, gold_run_id, gold_verified, stale_refresh, decision, gate_outcome, failed_gates, n_facilities, flags, retune_requested, ml_run_id}`. The API and UI read it (data spec §0.6).
- `monitor` also prints a summary JSON on its last stdout line; **`notify`** (a Python task in the Airflow image, which has no MLflow or GCS client) reads it via XCom and emails it with `precursorintelligence.alerts.send`: decision, gate outcome, champion version, headline metrics with CIs, flags. Failures email through `dags/utils/alerts.py: on_dag_failure`; holds and flags also email (course §7.5).

---

## 13. Step M13: tuning (task `tune`; decision RETUNE; ML-19)
- **When:** `mode=retune`, a re-tune request (§12.2), and once in Phase 1 to document the search space and sensitivity (course §3, §5).
- **Search:** Optuna TPE, seed 42, `n_trials` 30, timeout 60 minutes, `n_jobs` 1; objective = mean PR-AUC over 3 facility-grouped folds of `eval_train`, with early stopping (100 rounds) on each validation fold; the final `n_estimators` = the median best iteration. Never the test frame. Space (from PR #49 `boost_models.py`):
  | Param | Range | Param | Range |
  |---|---|---|---|
  | learning_rate | 0.005–0.1, log | subsample | 0.5–1.0 |
  | num_leaves | 8–128 | colsample_bytree | 0.3–1.0 |
  | max_depth | 3–12 | reg_alpha | 1e-3–10, log |
  | min_child_samples | 20–1000 | reg_lambda | 1e-3–30, log |
  | min_split_gain | 0–1 | cat_smooth | 1–100 |
  | LR `C` | {0.03, 0.1, 0.3, 1.0} | | |
- **Output:** `tuning_proposal.yaml` (the best parameters, ready to paste into `ml.yaml: members`), the tuned challenger (gated like any candidate but never promoted automatically; its G5 comparison has no tolerance: it must not be worse than the champion), one MLflow run `tune` with the trials table and the plots of §8.3.
- Runs in the Airflow pool `heavy` (§15.6) with `retries=0` (Optuna has its own timeout).

---

## 14. MLflow design

### 14.1 Server and access
MLflow 3.17 tracking server in the compose stack (§16.1); `MLFLOW_TRACKING_URI=http://mlflow:5000` in task containers; the UI on `127.0.0.1:${PI_MLFLOW_HOST_PORT:-5001}` (5000 is taken by AirPlay on macOS), reached through the IAP tunnel on the VM. Artifacts go to `gs://precursor-intel-mlops-mlflow`, proxied by the server (`--serve-artifacts`, clients use `mlflow-artifacts:/`). Note: every task runs as `sa-pipeline`, which can write to that bucket and to the scores anyway; the integrity check (§9.4) and the audit rows are the safeguards against a moved alias, not network isolation.

### 14.2 Experiments and runs
- Experiment `precursor_y_g12` (one per label). CI and smoke runs use `precursor_y_g12_smoke`.
- **No run spans tasks** (a crashed task would leave it RUNNING): each task opens its own run (`decide` none; `train`, `evaluate`, `gate`, `score`, `monitor`, `tune` one each), all tagged `ml_run_id` (= the Airflow run id) to group them; a run that raises is marked FAILED by MLflow, and a retry opens a new run.
- Tags on every run: `ml_run_id`, `gold_run_id`, `v`, `v_labels`, `dataset_version`, `gold_verified`, `gold_merged_sha256`, `git_sha` (`PI_GIT_SHA`), `image` (`PI_PIPELINE_IMAGE`, passed to tasks by `docker_kwargs`), `recipe_hash`, `decision`, `mode`.
- Params: member hyperparameters, ensemble weights, Platt `a`/`b`, `train_months`, `eval_gap`, window months, `n_features`, feature-list hash, excluded features.
- Metrics: §6.1 (CI bounds as `<metric>_ci_low`/`_ci_high`), baselines, OOF calibration diagnostics, slice summary.
- Inputs: `mlflow.log_input` with the pinned gold file and `dataset_version` as its digest.
- Artifacts: `evaluation.json`, `selection.json`, `slices.csv`, `bias_report.md`, plots, `shap_mean_abs.csv`, `model_card.md`, `feature_list.json`, `windows.json`, the input `manifest.json`, the eval-fit packages.

### 14.3 `recipe_hash`
sha256 over: the relative paths and bytes of the `*.py` files of `precursorintelligence.{training,evaluation,serving}` (not `.pyc`, which change per build), `configs/ml.yaml`, `configs/reason_labels.yaml`, the feature list, and the lightgbm, scikit-learn and numpy versions. A new hash means new code, config, features or libraries.

---

## 15. The DAG `cms_nh_ml`

### 15.1 Shape
- `schedule=[Asset("cms_nh_gold")]` plus manual runs; `catchup=False`, `max_active_runs=1`, `start_date` 2026-10-01, `default_args={"retries": 2, "retry_delay": 5 min}`, `on_failure_callback=on_dag_failure`, created paused. **Unpause it before the pipeline run whose gold should start it:** Airflow does not queue asset events for a paused DAG (verified for the release check).
- Every task except `notify` is one `PipelineDockerOperator` running `precursor-pipeline ml <command> --env <env> --run-id {{ run_id }}` (`docker_kwargs` plus `MLFLOW_TRACKING_URI` and `PI_PIPELINE_IMAGE`).
- `cms_nh_gold` is emitted on every successful pipeline run, even with nothing new (`docs/AIRFLOW_PIPELINE.md`), so `decide` is cheap and usually says SKIP or SCORE.

```
decide ─► branch ─┬─ SKIP ───────────────────────────────────────────────────────────────► notify
                  └─► prepare ─► branch ─┬─ RETUNE ─► tune ─┐
                                         ├─ RETRAIN ────────┴─► train ─► evaluate ─► gate ─┐
                                         ├─ PROMOTE ─► promote ────────────────────────────┤
                                         └─ SCORE ─────────────────────────────────────────┤
                                                                                           ▼
                                                     score ─► publish ─► monitor ─► notify
```
Branches use `@task.branch`; `train` and `score` use `trigger_rule="none_failed_min_one_success"`. `score` skips itself (and so `publish`) when there is no champion; `monitor` and `notify` still run.

### 15.2 The decide step
| Condition (first match wins) | Decision |
|---|---|
| `mode=promote` | PROMOTE: prepare → promote → score → publish → monitor |
| gold not `current` | exit 1 (retried) |
| `V` earlier than the last scored `V`; or unverified with `allow_unverified: false` | REFUSE (exit 2, alert) |
| `mode=retune`, or `mode=auto` and the last run requested a re-tune | RETUNE (then the rules below for scoring) |
| `mode=retrain`, or no completed RETRAIN in `ml_runs` for (`v_labels`, `recipe_hash`, `gold_config_sha`) | RETRAIN |
| `mode=score_only`, or a champion exists and no published scores for (`V`, `dataset_version`, champion version) | SCORE |
| otherwise | SKIP |

- A RETRAIN that holds still counts as completed, so a no-op gold publish doesn't retrain again (no loop).
- After RETRAIN, RETUNE or PROMOTE, the run scores if a champion exists and scores for (`V`, `dataset_version`, champion version) are missing.
- With no champion yet, nothing is scored; `status/latest.json` says "awaiting Checkpoint ML-A".
- **`ml_runs`** (one row per run and per promotion action): `ml_run_id, at, mode, decision, action (retrain | retune | score | skip | refuse | promote | hold | manual_promote | rollback), reason, actor, gold_run_id, v, v_labels, dataset_version, gold_config_sha, gold_verified, recipe_hash, champion_version_before, champion_version_after, candidate_version, gate_outcome, gate_results_json, scores_published (bool), flags_json, retune_requested`.
- **`gold_config_sha`** = sha256 (hex) of the canonical JSON (sorted keys, no spaces) of `{"config_sha256": …, "git_commit": …}`, both taken from gold's `dataset_manifest.json`: gold built with other configs or other code is another gold. One function computes it, `training/data.py: gold_config_sha(manifest)` (W2 adds it for `gate`'s rows; W6's `decide` calls the same one), so the writer and the reader always agree.

### 15.3 Tasks
| Task | Command | Notes |
|---|---|---|
| `decide` | `precursor-pipeline ml decide` | last stdout line: JSON (`decision`, `proceed`, `gold_run_id`, `champion_version`); `do_xcom_push`, `xcom_all` |
| `prepare` | `ml prepare --gold-run-id <from decide>` | §1.2, §1.5, §3 |
| `tune` | `ml tune` | RETUNE; pool `heavy`; `retries=0` |
| `train` | `ml train` | pool `heavy`; `retries=1` |
| `evaluate` | `ml evaluate` | §6–§8 |
| `gate` | `ml gate` | registers (idempotent), aliases, `ml_runs`; writes `champion.json` |
| `promote` | `ml promote` | the version comes from the param through the environment (`PI_ML_VERSION`, `PI_ML_REASON`), never the command line |
| `score` | `ml score` | §10; skips without a champion |
| `publish` | `ml publish` | §11; `outlets=[cms_nh_scores]` |
| `monitor` | `ml monitor` | §12; writes `status/latest.json`; last stdout line: the summary JSON |
| `notify` | Python | emails the summary (XCom from `monitor`, or `decide` on SKIP) |

### 15.4 Params and run modes
| Param | Type | Default |
|---|---|---|
| `mode` | enum: auto, retrain, retune, score_only, promote | auto |
| `version` | integer ≥ 1 (promote only) | — |
| `reason` | string (required for promote) | — |
| `allow_unverified` | boolean (one-run override, logged) | from `ml.yaml` |

| Situation | Decision | What happens |
|---|---|---|
| New CMS release, labels matured one more month | RETRAIN | train → gates → score the new month |
| Stale refresh (new features, no new labels) | SCORE | score with the champion, stamped `stale_refresh` |
| Pipeline rerun, nothing new | SKIP | one `ml_runs` row, a short email |
| New ML code or config deployed | RETRAIN (next gold, or `mode=auto` by hand) | paired comparison with the champion's eval fit |
| Drift or performance flag | RETUNE (next run) | tuned challenger + proposal for a PR |
| A bad model was promoted | PROMOTE (manual) | alias back to an earlier version, rescore |

### 15.5 Exit codes, retries, alerts
The data pipeline's contract (`ingestion/pipeline.py`): 0 ok; 1 transient error, retried (MLflow, GCS, BigQuery unreachable; gold changed underneath); 2 data or validation refusal, not retried; 3 config or usage, not retried; 137 out of memory, not retried. A gate hold is exit 0 with an email.

### 15.6 Running next to `cms_nh_pipeline`
- No shared lock (gold is pinned, §1.2).
- **Airflow pool `heavy`, 1 slot**, taken by `gold.build_gold` (data DAG), `ml train` and `ml tune`, so the two memory-heavy jobs never run together on a 16 GB machine. The pool is created by `airflow-init` (`airflow pools set heavy 1 …`) in the **same PR** that adds `pool="heavy"` to `build_gold` (a task in a missing pool never runs, and it would hold the data lock). A DAG test asserts both DAGs use the same pool constant.
- On the VM, everything that waits for "no pipeline run" (idle shutdown T21, `pi-down`, the deploy) must also count `cms_nh_ml` runs, running or queued, and a queued `cms_nh_gold` event (Phase 2).

---

## 16. Deployment

### 16.1 MLflow in the compose stack (ML-3, ML-12)
- **Image** `docker/mlflow.Dockerfile`: `python:3.12.12-slim-bookworm` and `uv:0.13.0` (as `pipeline.Dockerfile`), `uv sync --frozen --no-dev --no-install-project --extra mlflow-server` (mlflow 3.17, google-cloud-storage, psycopg2-binary), non-root user owning the artifacts directory, `PI_GIT_SHA` build arg and revision label.
- **Service `mlflow`** in `docker/airflow-local/docker-compose.yaml`:
  - **on the `tasks` network only** (not the default network): an unauthenticated server reachable by task containers must not become a bridge into the network where Airflow and the `airflow` Postgres role live (the L11 threat model);
  - Postgres: `init-db.sh` inserts `host mlflow mlflow ${PI_TASKS_SUBNET} scram-sha-256` above the task-subnet reject line (next to the `pi_meta` line). Existing databases (laptops) need it added once (runbook command: edit `pg_hba.conf`, `SELECT pg_reload_conf()`);
  - a one-shot **`postgres-setup`** service (`postgres:16.10`, default network, as `airflow`) creates role and database `mlflow` if missing, sets the password from `MLFLOW_DB_PASSWORD` (so rotation needs no hand step) and revokes `CONNECT` from `PUBLIC`; `mlflow` waits for it (`service_completed_successfully`);
  - the entrypoint runs `mlflow db upgrade "$MLFLOW_BACKEND_STORE_URI"` and then `mlflow server --host 0.0.0.0 --port 5000 --artifacts-destination ${PI_MLFLOW_ARTIFACTS} --serve-artifacts --workers 1 --allowed-hosts "mlflow,mlflow:5000,localhost,localhost:*,127.0.0.1,127.0.0.1:*"` (check the flag on 3.17 with `mlflow server --help`), with the backend URI in `environment`, not the command line;
  - port `127.0.0.1:${PI_MLFLOW_HOST_PORT:-5001}:5000`; a health check in Python (the slim image has no curl) with a `start_period` of ~120 s; `mem_limit: 1g`; `restart: unless-stopped`; the shared logging block.
  - `PI_MLFLOW_ARTIFACTS`: a prefix of `gs://precursor-intel-mlops-mlflow` per environment, `ml-dev/` for the laptop stack (matching the data bucket's `ml-dev/`, D11) and its own on the VM, so one environment's artifacts can be deleted without touching the other's (laptops use the impersonated ADC file like the task containers; the VM its metadata server); the named volume `mlflow-artifacts` when unset.
  - on the VM, `docker-compose.vm.yaml` keeps `mlflow` and `postgres-setup` in the profile `phase2` until Phase 2 (W4).
- **An MLflow version bump** runs `mlflow db upgrade`, which can't be undone: take a database dump first.
- **VM (Phase 2):** secret `mlflow-db-password` (Terraform `vm_secrets`, created in Phase 1, value by hand, hex only: `openssl rand -hex 32`; the VM renders 9 secrets); add `mlflow` to `infra/vm/pi-pg-backup` and `pi-pg-restore` (order airflow → mlflow → precursor_meta; a restore runs before the `mlflow` service starts); the image pinned by `PI_RELEASE_SHA` in `docker-compose.vm.yaml`.

### 16.2 Images, CI and CD (ML-5, ML-6)
- **Extras:** split the `ml` extra: `ml` = lightgbm, shap, scikit-learn, mlflow, optuna; a new `ml-extra` = catboost, interpret, evidently (interpret alone pulls dash, gevent and notebook). `docker/pipeline.Dockerfile` installs `--extra pipeline --extra ml --extra bias`. Update `requirements.txt` and the export check in `ci.yml`.
- **CI** (`ci.yml`, every PR): the test suites (§17.1) and a **smoke run** of the whole chain on a **synthetic ML gold fixture** (`tests/ml/make_ml_gold.py`: ~2,000 facilities × 30 months with a planted signal, all gold columns; the toy gold has 5 facilities and no `y_g12` training month), as DVC stage `toy_ml`, with a toy `data/toy_ml/ml.yaml` written by the fixture (`TOY_ML` merged over `configs/ml.yaml`: short windows, `min_train_months` 2, `min_score_facilities` 100, 50 trees, 2 folds, 200 bootstrap replicates, `gates.first_run: promote` so `score` runs) and `MLFLOW_TRACKING_URI=sqlite:///$RUNNER_TEMP/mlflow.db` (same store code as Postgres). Only byte-stable outputs are DVC `outs`; the MLflow store and anything carrying a run id are not. The `images` job builds the bigger pipeline image and the mlflow image and smoke-tests both. `MLFLOW_DB_PASSWORD` becomes the fifth laptop secret (update the dummy exports in `ci.yml`/`cd.yml`, `.env.example`, the runbook).
- **CD** (Phase 2): add `mlflow` to the image loops and the "already published" check in `cd.yml`; the mlflow merge sha becomes the deploy rollback floor; change the Artifact Registry DELETE rule to 7 days (KEEP 5 stays) to hold image storage near $1–2.5/month.
- **"Training on code change"** (course §7.1): CI runs the full chain on the fixture on every PR (train → validate → bias → gate → register into the ephemeral registry → score). In production, a merged change publishes a new image; the deploy moves the VM to it; the next `cms_nh_ml` run sees a new `recipe_hash` and retrains on real data (§15.2).

### 16.3 Terraform (`infra/terraform/envs/prod/main.tf`)
| Resource | Phase | Details |
|---|---|---|
| Bucket `precursor-intel-mlops-mlflow` | 1 | `modules/gcs`: us-central1, uniform access, public access prevention, versioning, `prevent_destroy`, no lifecycle expiry (models must survive D11's data-bucket wipe); `sa-pipeline` `roles/storage.objectUser` (write, overwrite and delete for `mlflow gc`; no IAM rights on objects) + `legacyBucketReader` (the client checks the bucket, as on the data bucket) |
| BigQuery dataset `precursorintelligence_scores` + 3 tables | 2 | schemas as JSON files read by Terraform and by a test against the parquet writer; explicit field modes; `deletion_protection`; monthly partitioning |
| `sa-pipeline` BigQuery | 2 | `roles/bigquery.dataEditor` on the dataset; `roles/bigquery.jobUser` on the project (`pipeline_runtime` set) |
| Secret `mlflow-db-password` | 1 | `vm_secrets`, empty; the value is added by hand before compose requires `MLFLOW_DB_PASSWORD` (§16.1) |
All changes only add resources (the CI apply guard refuses deletes); `sa-tf-apply` has the rights; the BigQuery API is enabled (`docs/runbooks/gcp-bootstrap.md`).

### 16.4 Phase 1 runtime: a laptop stack against GCS
- Each developer runs the compose stack (with the `mlflow` service) on a laptop, as the data pipeline was tested (T13, `dp_gcp_test`): `PI_PIPELINE_ENV=prod`, the gcloud ADC file of an account allowed to impersonate `sa-pipeline`.
- **Prefix:** Phase 1 reads and writes under `gs://precursor-intel-mlops-data/ml-dev/` (`outputs.root_prefix` for ML; `PI_BUCKET_URI` for the data pipeline), never the bucket root (D11: laptop stacks don't write to the root).
- **Gold for Phase 1** (P0, ML-23): set `spine_start: "2022-01"` in `configs/splits.yaml` and run `cms_nh_pipeline` with `mode=backfill` on one laptop stack (its own compose project and metadata DB) into `ml-dev/`. That run includes the 2026-10-28 release (V = 2026-10). **Done 2026-11-11** (§18.0).
- One shared MLflow is better than five: the person who runs P0 hosts the integration stack; others develop against `file:`/`sqlite:` stores and the fixture, and run real-gold jobs on the integration stack.

### 16.5 Phase 2 on the VM
After the VM exists (VM plan T19–T23) and the real backfill is done (T25): pull and start the mlflow image; T21 idle shutdown and T20 deploys count `cms_nh_ml`; T22 deploys three images; the ML acceptance (§17.2) runs after T25 on real gold (a smoke run on trial gold would pollute the registry and the outputs; if one is needed, it uses the `_smoke` experiment and model name and publishes nothing).

### 16.6 Cost
On top of the data pipeline's ~$10–15/month: MLflow bucket cents; BigQuery loads free, storage in the free tier; one secret ~$0.06; VM time ~20–60 extra minutes per monthly run (≈ $0.05–0.13) and ~$0.25 per re-tune; Artifact Registry +$1–2.5 with the 7-day rule (+$5–10 without it).

---

## 17. Testing and verification

### 17.1 Test suites
| Suite | Path | Proves |
|---|---|---|
| Unit | `tests/training`, `tests/evaluation`, `tests/serving`, `tests/monitoring` | windows (empty months, `eval_gap` modes, `spine_start` binding, `min_train_months`); stable folds; `to_model_frame`; LR preprocessing; the log-odds ensemble; Platt fit and the degenerate fallback; selection rule; metrics vs scikit-learn; paired bootstrap; roll-up arithmetic; reason ranking; PSI; every row of the decide table, including a hold followed by a no-op gold (SKIP, not RETRAIN) and a failed publish (SCORE); pin refusals; registration idempotency; the integrity check |
| Model | same | Platt lowers ECE on a miscalibrated fixture; shuffled labels give ROC ≈ 0.5; **SHAP additivity** for LightGBM, LR, the ensemble and after Platt (base + Σ = calibrated log-odds within 1e-6); the pyfunc round-trips; a stored `package_format: 1` package still loads |
| Gates | `tests/evaluation/test_gates.py` | table-driven: each gate passes and fails at its boundary; "worse" needs both the tolerance and the CI; first-run behaviour; manual-promote rules |
| Leakage | `tests/leakage` | (1) permuting the test labels leaves the model, calibration, selection, threshold, tuned parameters and mitigation weights byte-identical; (2) window arithmetic for g12, g6, pen12 over many `V_L` incl. gap months: max(eval t) + H < min(test t) (no training outcome window reaches the test window), max(eval t) + H + B ≤ the first test month and the eval window ends at the latest month with rows that allows (production's lag, no more), `refit_train` ≤ `L`, every training frame has status only positive/negative; (3) `evaluate` refuses a model whose training months reach the test window minus H; (4) each `ccn` in exactly one fold, OOF predictions from fits without that facility; (5) the model input equals the registry `model` list minus `features.exclude`, in order, and never contains `_fwd`, `y_*`, `slice__*`, `overall_rating`, `chain_id`, `t`, `cutoff_date`; (6) reordered or unseen categorical levels give the same predictions for seen levels; (7) regime-coverage flags; (8) a stale-refresh fixture (`v ≠ v_labels`); (9) mitigation weights fit on each frame's own rows; (10) G5/G7 compare on identical rows and nothing loads `@champion` to compute test metrics; (11) realized performance only for months with `V' + H + B ≤ V_L`, matched to the publishing version; (12) the pinned run equals the run `decide` evaluated |
| DAG | `tests/dags/test_ml_dag.py` (in the airflow image) | `cms_nh_ml` imports; schedule = `cms_nh_gold`; tasks use the pipeline image; branch wiring and trigger rules; the `heavy` pool shared with `build_gold`; typed params; `last_json_object` keys |
| Smoke | DVC stage `toy_ml` (CI) | the whole chain on the synthetic fixture writes every artifact in the contract |

Tests never call paid services or a shared MLflow server; fixtures are synthetic with fake CCNs; a bug fix starts with a failing test.

### 17.2 Acceptance on real gold
Phase 1 (laptop integration stack, gold under `ml-dev/`); repeated in Phase 2 on the VM after T25.
- [ ] `mode=retrain` completes: MLflow shows the `train`, `evaluate` and `gate` runs, a registered version (challenger), the model card.
- [ ] With `eval_gap: purge` and the window 2023-01..2023-11, the eval fit lands within PR #49's CI (0.465–0.499) on 2025-02, or the gap is explained (121 features vs PR #49's 133). The production setting (`production_lag`) is then used for everything else.
- [ ] Selection, calibration (OOF diagnostics, `a`, `b`, ECE before/after) and G1–G7 results are in `evaluation.json`.
- [ ] Checkpoint ML-A done, `mode=promote` run: scores, reasons, SHAP and chain files in `scores/v=2026-10/`; `status/latest.json`; the summary email.
- [ ] `mode=auto` on the same gold gives SKIP; `mode=score_only` rewrites the same partition with identical scores.
- [ ] A forced bad candidate (a broken feature) is held and the champion keeps scoring; `mode=promote` with an earlier version rolls back.
- [ ] One `mode=retune` run: tuned challenger, `tuning_proposal.yaml`, sensitivity plots.

### 17.3 🛑 Checkpoint ML-A: the first champion
Before the first promotion (and after any change to the gates or mitigation), a human reads `model_card.md` and the bias report and checks: metrics vs PR #49 and the baselines, on non-development months where available; calibration; the slice table and state list; the top-20 features are plausible (nothing leakage-like); 20 random facilities' reasons read sensibly; the regime and skew flags (§1.4). Then promotes with `mode=promote`, recording the approval in `reason`.

---

## 18. Work plan (5 people, all using Claude Code)

### 18.0 Before you start (everyone, once)
**What exists already (P0, W0 #54 and W5 #55 merged):**
- `configs/splits.yaml` starts the spine at 2022-01 (#53, merged).
- **The Phase 1 gold** is `gs://precursor-intel-mlops-data/ml-dev/gold/`. The bucket root's `gold/` is an old 16-month test build: never use it.
  - run `20261112T020010Z-543cdc`, `dataset_version` `3f9bac9367ae18ec`, V = `v_labels` = 2026-10 (rebuilt 2026-11-12 with W5's quality family);
  - 52 months (2022-01 → 2026-10), 775,400 rows, 148 columns, all six feature families, `verified: false`;
  - `ml prepare` on it: 121 features; eval 2022-05..2023-04, test 2025-02 (14,643 facilities, base rate 0.256), refit 2024-02..2025-02; no leakage or regime flags.
  - It was built by the `ml_dev` laptop stack, which has its own metadata database. **Nobody else writes to `ml-dev/gold/`.** If gold must change, ask the P0 owner, who rebuilds it.
- **GCP VMs (2026-11-12): both stopped.** `pi-airflow` (e2-standard-4, us-central1-a; T19/T20, used from Phase 2) and Nishanth's old `pi-temp-validate` (2026-10-16, `purpose=temporary`). Phase 1 needs no VM. `pi-airflow` stops itself about an hour after the last run and SSH session (the idle shutdown, T21, #56; its timer starts at the next boot), and `sudo pi-down` on it dumps the databases and stops it at once. `pi-temp-validate` has no idle shutdown: stop it by hand, `gcloud compute instances stop pi-temp-validate --zone us-central1-a`.

**Prerequisites:**
1. **Read** §0, §1, §3, §17, §18 and §20 of this spec, plus the gold contract in the data spec (§9, §11). Every package's names come from here.
2. **Update your repo:** pull `dev`, run `uv sync --frozen --all-extras`, and rebuild `precursor-pipeline` and `precursor-airflow` if you run the laptop stack. They now carry the 2022-01 spine.
3. **Fix your local `CLAUDE.md`** (not in git; O6). The star-rating rule reads: "the overall and health-inspection star ratings are never features (`staffing_rating` and `qm_rating` may be, per `configs/feature_registry.yaml`)".
4. **GCP access as `sa-pipeline`, never your personal account:** `gcloud auth application-default login --impersonate-service-account=sa-pipeline@precursor-intel-mlops.iam.gserviceaccount.com` (the T13 setup; `docs/runbooks/airflow-local.md`, the GCS section).
   - Read gold through its pinned `_current.json` (§1.2).
   - Write ML outputs only under `ml-dev/` (`outputs.root_prefix`), never the bucket root (D11).
5. **Code against W0** (merged, #54).
   - Package code uses W0's config keys, `ml` commands, frames and run-folder JSON contracts.
   - Unit tests use the synthetic fixture `tests/ml/make_ml_gold.py`: no network, no real gold.
6. **Run your package on the real gold locally before its PR,** once its unit tests pass:
   - `PI_BUCKET_URI=gs://precursor-intel-mlops-data/ml-dev PI_DATABASE_URL=sqlite:///data/ops/ml-local.db uv run precursor-pipeline ml prepare --env prod --run-id <name>-<package>`, then your command with the same `--run-id`. With the store rooted at `ml-dev/`, `ml.yaml` stays unchanged and outputs land in `ml-dev/ml/runs/<run id>/`.
   - **Expected results** (PR #49's chosen ensemble, §4.1): with a copy of `ml.yaml` that sets `windows.eval_gap: purge` (pass it with `--ml-config`), the eval window is 2023-01..2023-11, and test 2025-02 gives a PR-AUC within 0.465–0.499 (PR #49: 0.481), ROC ≈ 0.71, Brier ≈ 0.169, ECE ≈ 0.03 and P@5 % ≈ 0.68, or the gap is explained (§17.2). The default `production_lag` window scores somewhat lower.
   - A package without metrics checks that its command exits 0 and writes every file of its contract.
7. **One branch from `dev`, and one PR per package** (§20). Before opening it:
   - run `uv run ruff check .`, `uv run mypy src` and `uv run pytest`;
   - run the `leakage-reviewer` agent on window, fold, feature or label code;
   - run the `infra-reviewer` agent on W4's Docker, Terraform and CI changes.
8. **One shared MLflow:** the P0 owner hosts the integration stack (§16.4).
   - Develop with a local tracking store (`MLFLOW_TRACKING_URI=sqlite:///mlruns.db` or `file:`).
   - Register models only from the integration stack, so the registry has one history; item 6's local runs use the local store.
9. **Memory:** the full gold frame (775k rows × 148 columns) and the gold build need several GB. The W5 gold build peaked at 3.6 GB resident (~10 GB footprint); `ml prepare` at 3.3 GB.
   - Docker Desktop's default 8 GB is too small for real-gold jobs in containers. Raise it to 12 GB or more, or run those jobs on the integration stack.
   - Unit tests on the fixture are unaffected.
10. **Known data caveats to keep in mind:**
   - `d12_turnover_total` is empty for all of 2022 (§1.4, O9);
   - the regime flags are excluded (ML-21);
   - 2025-02 is a development month (§3.2, §21);
   - 2024-12 and 2025-01 are empty CMS months.
11. **The must-nots in §20 apply to every package,** above all: nothing is chosen on the test frame, no secret goes into MLflow, and nothing reads `dim_facility_latest`.

### 18.1 Phases (ML-22)
| Phase | When | Scope |
|---|---|---|
| **1. Course-complete system** | Tue Nov 10 → **Sun Nov 15** | everything in the course brief (Appendix D) working end to end: the `cms_nh_ml` DAG on a laptop compose stack against GCS (`ml-dev/` prefix), MLflow (tracking + registry, artifacts in the MLflow bucket), training with selection, calibration, validation, bias detection and gate, SHAP (global + per facility), tuning with documented sensitivity, scoring with reasons and chain roll-up, monitoring (PSI + status file), email notifications, rollback, CI smoke run |
| **2. Production** | Nov 16 → Dec 1 and later | VM deployment (MLflow on the VM, secret, backups, T19–T26 counting `cms_nh_ml`), BigQuery publish, CD for the mlflow image, the §11 verified gate (W5 part 2), realized and early-read performance, Evidently, rolling-origin folds, Checkpoint ML-A repeated on 2025-03 (the late-November release) |

**Cut line for Phase 1.** If a package runs late, these stay, because the course brief needs them (Appendix D): training with selection, hold-out validation, bias slices and the G7 gate, SHAP (global and per facility), tuning with its sensitivity, MLflow tracking and registry, the CI smoke run, alerts and rollback. These may move to Phase 2: the chain roll-up, per-state empirical-Bayes shrinkage, the Benjamini–Hochberg correction, and score-drift and regime-coverage monitoring. The slowest package never holds up Friday's integration.

### 18.2 Phase 1 order
```
 Tue        Wed                 Thu             Fri              Sat               Sun
 P0 data ── spine 2022 + full backfill to ml-dev/ ──► gold ready ───────────────────────────►
 W0 ───────►(interfaces merged by Wed noon)
 W4 ──────── MLflow service, bucket, image extras, CI smoke ──►
            W1 training ────────────────────────────►
            W2 evaluation, bias, gates ─────────────►
            W3 explain, score, roll-up ─────────────►
                                     W6 DAG, decide, monitor, notify ──► integration ─► acceptance
                                                                         on real gold   + ML-A
```
| Lane | Packages | Notes |
|---|---|---|
| 1 | W0 → W6 | the lead: interfaces first, then the DAG and integration |
| 2 | W1 | |
| 3 | W2 | |
| 4 | W3 | |
| 5 | P0 → W4 | P0 is mostly machine time once started |

**Rules:** W0 is thin and merges first (stubs for every command, the frames, the run-folder JSON contracts) so W1–W4 code against it in parallel with the synthetic fixture; every package merges as soon as its tests pass; the first end-to-end run on real gold happens by Friday, not Sunday.

### 18.3 Packages
| WP | Phase | Scope | Modules / files | Depends on | Done when |
|---|---|---|---|---|---|
| **P0 Data** ✅ done 2026-11-11 (§18.0) | 1 | `spine_start: "2022-01"` (data spec §14 #10, recorded); full `cms_nh_pipeline` backfill on a laptop stack into `ml-dev/`, including the 2026-10-28 release | `configs/splits.yaml` | — | `ml-dev/gold/` current, spine 2022-01, V = 2026-10 |
| **W0 Foundation** ✅ done (#54) | 1 | `configs/ml.yaml` + `MlConfig`/`load_ml_config`; the `ml` typer sub-app with stubs (exit 3 "not built"); `training/data.py` (pin, checks, `to_model_frame`, windows, frames, folds); `ml_runs` in `common/io/schemas.py`; run-folder JSON contracts; `configs/reason_labels.yaml` skeleton; the synthetic fixture `tests/ml/make_ml_gold.py` | `common/config.py`, `training/{cli,data}.py`, `harmonize/cli.py` | — | window, fold and pin tests (incl. leakage tests 2, 4, 5, 12) pass on the fixture |
| **W1 Training** | 1 | LR pipeline, LightGBM member, log-odds ensemble, OOF + Platt, selection, `PrecursorModel` pyfunc, MLflow logging helper, `ml train`; `ml tune` with the proposal and plots; the PR #49 reproduction check; the PSI check of the two sub-ratings (§1.4); the `d12_turnover_total` decision (§1.4, O9) | `training/{members,ensemble,calibration,select,train,tune,mlflow_log}.py` | W0 | model tests pass; a real eval fit and refit are logged |
| **W2 Evaluation, bias, gates** | 1 | metrics, paired bootstrap, baselines, sanity checks, Fairlearn slices + per-state EB, reweighting (with the `fit_rows` parameter), plots, `model_card.md`, `gates.py`, `ml evaluate`, `ml gate` (registration, aliases, `ml_runs`), `ml promote` | `evaluation/{metrics,bootstrap,baselines,sanity,slices,gates,report}.py`, `bias/mitigation.py` | W0, W1's fit interface | gate tests pass; a real model card exists |
| **W3 Explain and score** | 1 | `serving/explain.py` (§8.1), reason codes and all model-feature labels, global SHAP helpers for W2, `ml score`, roll-up, the scores layout and manifest, `ml publish` (Phase 1: verify + asset) | `serving/{explain,score,rollup,publish}.py`, `configs/reason_labels.yaml` | W0, W1's pyfunc | additivity tests pass; real scores and reasons written |
| **W4 Platform** | 1 | §16.1 service + Dockerfile + `postgres-setup` + `pg_hba` line; MLflow bucket (Terraform PR); extras split + image; `docker_kwargs` (`MLFLOW_TRACKING_URI`, `PI_PIPELINE_IMAGE`); pool `heavy` in `airflow-init` + `build_gold`; CI smoke stage `toy_ml` and image jobs; runbook | `docker/`, `infra/terraform/`, `.github/workflows/ci.yml`, `dags/utils/pipeline_operator.py`, `dvc.yaml`, docs | W0 (names) | a task container on the laptop stack logs a run (`curl` from the tasks network to `mlflow:5000` returns 200); CI green |
| **W6 DAG + monitor** | 1 | `dags/cms_nh_ml.py`, `ml decide`, branches, params, `notify`; `ml monitor` (PSI signals, re-tune request, status file) | `dags/`, `training/cli.py`, `monitoring/{drift,status}.py` | W1–W4; PR #48 | DAG tests pass; the acceptance list (§17.2) is ticked on the laptop stack |
| W5 Quality family (quality ✅ #55, gold rebuilt 2026-11-12) | 2 | port `quality.py` with `configs/qm_measures.yaml` into gold (data spec Checkpoint 4), registry entries, labels, tests; then the §11 verified gate | `gold/features/quality.py`, `configs/` | — | gold carries the quality features; later `verified: true` |
| W7 Production | 2 | VM (§16.5), BigQuery (§11, §16.3), CD + AR rule, realized and early-read performance, Evidently | | Phase 1, VM plan | acceptance repeated on the VM |

---

## 19. Decisions

**Decided by the user on 2026-11-10:**

| # | Topic | Decision | Rule in this spec |
|---|---|---|---|
| ML-1 | Labels | `y_g12` only in v1; label-generic code | §0.2 |
| ML-2 | Retraining | every new matured month: retrain with the current parameters, then the gates; re-tune on demand or on a drift/performance flag | §12.2, §13, §15.2 |
| ML-3 | MLflow | a server in the compose stack; Postgres db `mlflow`; UI through the tunnel on the VM | §14.1, §16.1 |
| ML-4 | Scores | GCS parquet (source of truth) + BigQuery (Phase 2) | §10, §11 |
| ML-5 | CI | a smoke run of the whole chain on synthetic gold; real training in the DAG | §16.2 |
| ML-6 | Image | ML tasks in `precursor-pipeline`; a small `mlflow` image for the server | §16.1, §16.2 |
| ML-7 | Team | 5 people; sequential foundation and integration, parallel middle | §18 |
| ML-8 | Base model | PR #49's best model (LightGBM + LR); PRs #38/#49 are the baseline for features, transforms and parameters | §4, Appendix A |
| ML-9 | Unverified gold | allowed and tagged until the data spec's §11 gate exists | §1.3 |
| ML-10 | Demo | end to end; monitoring may be thin | §18 |
| ML-11 | Chain roll-up | yes, as an aggregation | §10.3 |
| ML-12 | MLflow artifacts | a new bucket `precursor-intel-mlops-mlflow`, proxied | §16.1, §16.3 |
| ML-13 | Bias | report always; block promotion + alert on a material, significant slice regression vs the champion or a calibration limit | §7, §9.1 G7 |
| ML-14 | Explanations | top 5 raising + top 3 lowering reasons per facility; the full SHAP matrix per month; a global summary per model | §8, §10.2 |
| ML-15 | Production fit | evaluate, then refit on the latest matured months | §3.3 |
| ML-16 | Ensemble | average the members in log-odds (measured: same scores, exact SHAP) | §4.2, Appendix A |
| ML-17 | Star sub-ratings | follow the registry (`staffing_rating`, `qm_rating` allowed; overall and health-inspection ratings never) | §1.4 |
| ML-18 | Quality family | in gold since #55; 33 of its 41 columns are model input, the rest excluded in `ml.yaml` (month keys and the measures whose scale changed) | §1.4, W5 |
| ML-19 | Re-tuning | a re-tune produces a challenger and a parameter proposal; parameters change only through a human PR to `ml.yaml` | §12.2, §13 |
| ML-20 | Evaluation gap | the production lag (H + B = 20 months); gold starts in 2022-01 (`spine_start`, data open #10) | §3 |
| ML-21 | 2025-07 / 2025-10 regimes | `era` and `penalty_regime_2025_10` excluded until training covers both values; the two sub-ratings measured and excluded if their training-vs-scoring PSI > 0.25 | §1.4, §12.1 |
| ML-22 | Phases and goal | Phase 1 (course-complete, laptop stack + GCS) by Nov 15; production after; a working system with decent performance, no model optimization push | §18 |
| ML-23 | Phase 1 gold | full backfill with spine 2022-01 into `gs://…-data/ml-dev/` | §16.4 |

**Technical decisions** (from the design and its three reviews; change only with a reason recorded here):

| # | Topic | Decision | Why |
|---|---|---|---|
| ML-24 | Calibration | Platt on OOF predictions of the window's newest 3 months | isotonic always wins on its fitting rows and makes ties; Platt keeps ranks and exact reasons |
| ML-25 | Windows | 11 non-empty months; eval window 20+ months before the test window; refit up to `L` | PR #49's recency result; honest gap |
| ML-26 | Folds | 5 facility-grouped folds (3 for tuning) by a stable `ccn` hash | §3.4 |
| ML-27 | Champion comparison | paired, on the current test frame, against the champion's logged eval fit; "worse" = beyond tolerance and CI excludes 0; BH across slices | cross-window comparisons were mostly noise |
| ML-28 | Selection | among ensemble / LightGBM / LR by OOF PR-AUC after an OOF bias check | course §8.6; LR is a built-in fallback |
| ML-29 | Decide | keyed on `ml_runs` (`v_labels`, `recipe_hash`, gold config) and published scores; a hold counts as done | no retrain loops |
| ML-30 | No lock | gold pinned with hash checks; pool `heavy` for memory-heavy tasks | §1.2, §15.6 |
| ML-31 | One DAG, runs per task | `cms_nh_ml`; MLflow runs grouped by tag, no cross-task parent; registration in `gate` | §14.2, §15 |
| ML-32 | MLflow network | on the `tasks` network only, with its own `pg_hba` line | L11 threat model |
| ML-33 | Drift | PSI month over month (trigger) and training vs scoring (alert); score drift with the same model on two months | §12.1 |
| ML-34 | Reason labels | `configs/reason_labels.yaml` beside the registry | §8.1 |

**Open:**

| # | Topic | Default | Recommendation |
|---|---|---|---|
| O1 | Second label (`y_g6` matures 6 months sooner) | not in v1 | add by config after Phase 1 |
| O2 | Gate and bias thresholds | §9.1 starting values | review at Checkpoint ML-A on a non-development month. G4 (ECE ≤ 0.05) is the likely binding gate: PR #49 measured test ECE rising to ~0.055 after only 7 months out of time, and the eval fit is 20–23 months out of time. W2 checks its first real values before anyone relies on automatic promotion |
| O3 | Rolling-origin folds | facility-grouped | try after spine 2022-01 is live |
| O4 | Reweighting | off | decide at Checkpoint ML-A from the OOF trade-off table |
| O5 | MLflow authentication | none (tasks network + 127.0.0.1 + IAP) | revisit if anything outside the VM needs MLflow |
| O6 | `CLAUDE.md` star-rating wording | changed 2026-11-10 in the maintainer's local copy | teammates update their local `CLAUDE.md` (it is not in git): "overall and health-inspection star ratings are never features" (ML-17) |
| O7 | Data spec edits | §14 #10 recorded (2022-01) on 2026-11-10 | still to do when the team agrees: §9.5 open #16 → resolved by ML-24; §15.1's three ML DAGs → one `cms_nh_ml` |
| O8 | Regime-invariant enforcement features | — | a data-side idea (counts per revision group or per survey) so the 2025-10 fine-count jump doesn't skew scores |
| O9 | `d12_turnover_total` is empty for all of 2022 (§1.4) | in the feature list | W1: exclude it for Phase 1 (`features.exclude`); take it back once the eval window starts in 2023-01 or later (about `V_L` = 2027-05) |

---

## 20. Definition of done and must-nots
**Done** for any package: behaviour-focused tests green (§17.1), ruff and strict mypy clean, configs and this spec updated, its MLflow artifacts produced where relevant, one PR per package.

**Must-nots:**
- Train or evaluate on immature or censored rows; score anything but `t = V`.
- Tune, select a candidate, a feature set, a window, a threshold or a mitigation on the test frame.
- Score the registered champion (a refit) on a test frame; compare with the champion on different rows.
- Read `dim_facility_latest`; put `ablation_only`, `product_only` or label columns into a model.
- Random row splits or K-fold over facility-months.
- Present a batch-relative rank as a probability, or SHAP contributions as probability points.
- Promote a model outside `gates.py` except through the audited `mode=promote`; move the alias by hand in the UI.
- Write scores without the model version and gold run that produced them.
- Hard-code a feature list, threshold, window or path that has a config key; write to the bucket root from a laptop stack.
- Put secrets in MLflow params, tags, artifacts or the command line.

---

## 21. Known limitations
- **Label lag:** outcomes mature 20 months after `t`; the newest training data is always ~20 months old. A discrete-time hazard model trained on partially observed windows could use data up to `V − 8` (future work).
- **Regime changes** (2025-07 ratings, 2025-10 fines): training windows don't contain the new regimes until ~2027; the affected flags are excluded and skew is monitored, but the scale of enforcement counts after 2025-10 is still a training/scoring mismatch (O8).
- **Development month:** 2025-02 was used by PR #49 to choose; honest numbers start with 2025-03.
- **Empty CMS months** (2024-12, 2025-01) shrink the current test window to one month (~14.6k facilities); CIs reflect it.
- **Within-state ranking is weak:** PR #49 found only 3.6 % of AUC pairs within states, and `state` 24.8 % of LightGBM gain; within-state metrics are reported.
- **Per-state miscalibration** (18 of 34 states in PR #49) is reported every month and gated against regression only.
- **Complaint-driven outcomes:** 68 % of positives come from complaint surveys only (PR #49 research) and are hard to predict from public data; an oracle with future complaint counts reached ROC 0.82.
- **Chain expected counts** (Σ p) are exact in expectation; the spread of a chain's outcomes is wider than independence suggests when problems are chain-wide.
- **Realized performance** of published scores is first measurable ~20 months after the first scores (an early decile-lift read after 3–6 months is Phase 2).

---

## Appendix A. Baseline provenance (PR #49)
- **Source:** PR #49 (`feat_ML_model/rev001`, commit `4a0b7e0a`), `notebooks/final_rev001/`: `results/MODEL_CARD.md`, `final_model/final_model_metrics.json` (1,000 bootstrap replicates), `experiments/rev002/exp9_recency.py` (final arm R4), `feature_engineering/boost_spine2022/lgbm_best_params.json`, `experiments/rev002/exp8_lr.py`, `experiments/rev002/feature_registry_rev002.yaml` (133 features).
- **Data:** a local gold build ("all tables", run `20261107T014704Z-c299c0`, spine 2022-01, V = 2026-10), never pushed; training rows 2023-01..2023-11; test 2025-02.
- **Experiments that did not beat it** (test PR-AUC): tuned LightGBM alone 0.474–0.479; CatBoost and gradient-boosting variants 0.479–0.490 on other windows; FT-Transformer, TabNet, MLP, DeepFM, DCN-v2 0.436–0.464; +120 engineered features, two-stage survey × harm, case-mix, +14 tables, PBJ: Δ −0.0004..+0.0017.
- **Log-odds vs probability average** (ML-16), measured 2026-11-10 on `final_model/test_predictions.csv` (14,643 rows, one per facility): PR-AUC 0.4813 / 0.4813; ROC 0.7091 / 0.7092; Brier 0.1689 / 0.1689; ECE 0.0274 / 0.0268; P@5 % 0.682 / 0.682; paired-bootstrap CI of the PR-AUC difference [−0.0003, +0.0003]; Spearman 0.99993; top-10 % overlap 100 %; median |Δp| 0.0004, max 0.077.
- **Issues found in the PRs and how this spec handles them:** no train→calib purge in gold (→ OOF Platt, ML-24); the final model trained on the calib months and shipped uncalibrated (→ §5); window chosen on test (→ fixed config, development-month marking); parameters tuned on another feature set with non-time folds (→ re-tune as a proposal, §13); `state_te` leakage and row-count lags (→ not carried); star sub-ratings (→ ML-17, ML-21); scripts not runnable from the repo (→ everything in `src/` with tests).

## Appendix B. `configs/ml.yaml`
The settings live in `configs/ml.yaml` (W0, #54), one commented file, validated by `MlConfig` in `common/config.py` (an unknown key or a bad value is a config error, exit 3). §0.5 lists the defaults; this spec keeps no second copy, so the file is the only place a value changes (by reviewed PR, ML-19).

## Appendix C. Output schemas
| File / table | Key | Columns |
|---|---|---|
| `facility_scores` | `ccn, score_month` | §10.1 |
| `facility_reasons` | `ccn, score_month, direction, rank` | §10.2 |
| `shap_values` (GCS only) | `ccn, t` | `model_version, base_value`, one float per feature |
| `chain_scores` | `chain_id, score_month` | §10.3 |
| `ml_runs` (Postgres) | `ml_run_id` + one row per promotion action | §15.2 |
| `status/latest.json` | — | §12.3 |

## Appendix D. Course requirements → this spec
| "Model Development Guidelines" item | Where | Phase |
|---|---|---|
| §2.1 Load versioned data from the data pipeline | §1, §2, §14.2 (`dataset_version`, pinned copy) | 1 |
| §2.2 Train and select the best model | §4.2–§4.3 (three candidates, selection), §9 (champion) | 1 |
| §2.3 Validation on a hold-out set | §3 (test window, 20-month gap), §6 | 1 |
| §2.4–2.5 Bias detection with slicing; code; reports; mitigation | §7, §9.1 G7 | 1 |
| §2.6 Push to a model registry | §9.2, §14 | 1 |
| §3 Hyperparameter tuning with a documented search space | §13 | 1 |
| §4 Experiment tracking, visualizations, model-selection explanation | §6.5, §14, `selection.json` | 1 |
| §5 Sensitivity analysis (SHAP; hyperparameters) | §8.2, §8.3 | 1 |
| §6 Bias slicing, metrics across slices, mitigation, documentation | §7 | 1 |
| §7.1 CI/CD triggers training on code changes | §16.2 | 1 (CI), 2 (CD + VM) |
| §7.2 Automated validation against thresholds | §9.1 | 1 |
| §7.3 Automated bias detection that alerts or blocks | §9.1 G7, §12.3 | 1 |
| §7.4 Registry push / deployment | §9.2, §10 (batch "deployment" = scoring with the champion) | 1 |
| §7.5 Notifications | §12.3, §15.5 | 1 |
| §7.6 Rollback | §9.4 | 1 |
| §8.1 Docker | §0.4, §16 | 1 |
| §8.2–8.7 Code for loading, training, validation, bias, selection after bias, registry push | §18.3 W0–W3 | 1 |
