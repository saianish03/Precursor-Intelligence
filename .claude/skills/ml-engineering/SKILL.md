---
name: ml-engineering
description: Model development standards for the Precursor Intelligence risk model. Use when building or changing features, splits, training, calibration, evaluation, promotion gates, SHAP explanations, batch scoring, account roll-ups, monitoring or the automated retraining pipeline.
---

# ML engineering

## Problem framing
Binary classification on facility × month rows (~129 features). Primary label `y_g12` (base rate ~25–27%),
secondary `y_g6` (~15%). Train only on rows with status positive or negative. Insurers need calibrated
probabilities and per-facility explanations.

## Model lineup
- Baselines (always reported): prior-G+ rule, severity heuristic, CMS star ratings (benchmark only),
  logistic regression. The champion must beat logistic regression to be promoted.
- Champion candidate: **LightGBM** with monotone constraints on obvious features (e.g. prior G+ counts
  can only increase risk). Challengers: **CatBoost**, **Explainable Boosting Machine** (InterpretML).
  XGBoost optional; HistGradientBoosting skipped. No SMOTE or oversampling; use class weights.
- Calibration: isotonic on held-out facilities (Platt as fallback). Report ECE and Brier before and after.
- Tuning: Optuna with **time-based, facility-grouped folds**. Never random K-fold on facility-months.

## Splits and leakage
- Walk-forward: train months, then a purge gap of one horizon (12 or 6 months), then test months.
  Splits come from `configs/splits.yaml`; each retrain shifts every window forward one month.
- Never use `overall_rating` or `health_inspection_rating` as features.
- Any feature that alone exceeds PR-AUC 0.6 or 3× the base rate needs a leakage review.

## Evaluation
PR-AUC vs base rate, Precision@5% / @10%, lift, within-state precision, Brier, ECE, reliability curve,
slices (state, ownership, bed size, urban/rural), double lift against star ratings, and facility-level
block-bootstrap confidence intervals on every headline metric.

## Promotion gates (deterministic code in `src/precursorintelligence/evaluation/gates.py`)
Beats baseline; PR-AUC ≥ champion − 0.01 and Brier ≤ champion + 0.005 on the same window; ECE within the
threshold; no slice regression beyond the limit; shuffled-label AUC ≈ 0.5; all upstream data checks passed.
All pass → promote by moving the MLflow `champion` alias and write `promotion_audit`. Any fail → keep the
champion, register as `challenger`, alert. Manual override/rollback only through `promote-model.yml`.

## Retraining pipeline
Asset chain: `build_gold` → `train_evaluate` → `batch_score` → `monitor`. Each new CMS release adds exactly
one fully labelled prediction month. Skip retraining for duplicate or stale releases. Drift (PSI > 0.25 on a
key feature or calibration error above threshold on a newly matured cohort) triggers an extra retrain.

## Outputs
- Every run logs to MLflow: params, metrics JSON, dataset manifest ID, git SHA, feature list, model, SHAP summary.
- Batch scoring writes `facility_score` and `account_score` to BigQuery. Account roll-up: expected events per
  100 beds, P(≥1 event), worst facility, share of beds in the top decile. Score-band → action mapping comes
  from a fixed rule table, never from a model or an agent.
