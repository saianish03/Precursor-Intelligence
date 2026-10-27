# Schema and statistical data validation

`precursor-pipeline validate` checks silver or gold month by month. It reads the tables straight from the object
store and changes nothing in them. It takes no lock and runs no freshness check, so it also works on a silver built
with older configs. Code: `src/precursorintelligence/validation/`. Settings: `configs/validation.yaml`.

## What it checks

| Kind | Check | Severity | Source |
|---|---|---|---|
| Schema (Pandera) | `missing_column`, `unexpected_column`, `wrong_dtype` (the column's other checks are then not reported), `null_values` (in a non-nullable column), `out_of_range` (`min`/`max`), `allowed_values`, `str_length`, `pattern` (a regex, e.g. the CCN format) | error | `configs/contracts/<table>.yaml` |
| Schema | `allowed_values` for a column marked `allowed_severity: warning`: a value CMS may add (a new ownership type) | warning | `configs/contracts/<table>.yaml` |
| Schema | `duplicate_key`: two rows of a month share the `unique` columns | error | `unique` in the contract |
| Schema | `y_null_iff_unlabelled`, `y_matches_status`, `score_iff_immature`: each label's `y`, `status` and `split` agree | error | `rules` in the `labels` contract |
| Schema | `month_mismatch`: `vintage_month` is not the stored month, or `cutoff_date` is not its first day; `before_first_month`: a gold month before `spine_start` | error | `configs/splits.yaml` |
| Schema | `empty_month`: a month with no rows; `table_missing`: a configured table has no months in the store | error | `configs/validation.yaml` |
| Drift vs the previous month | `row_count_change` (more than ±`row_count_change_pct` %) | warning | `drift` in `configs/validation.yaml` |
| | `null_rate_jump` (null % up by more than `null_rate_jump_pts` points) | warning | |
| | `mean_shift` (mean moved by more than `mean_shift_std` previous-month standard deviations) | warning | |

Drift is not checked for the columns in `drift.skip_columns` (`y_*`, `status_*`, `split_*`). Labels are censored for
the last months and splits are rolling windows, so they move by design. Their values are still schema-checked:
`status_*` must be one of positive, negative, censored or immature, and `split_*` one of train, calib, purge, test
or score. The label rules tie them together, as `gold/labels.py` and `gold/splits.py` write them: `y` is null
exactly when the status is censored or immature, `y` is 1 exactly when positive and 0 exactly when negative, and
the split is `score` exactly when the status is immature. A table's first month has no previous month, so it gets
only the schema checks. "Previous month" is the previous month present in the store, so 2025-02 is compared with
2024-11 when 2024-12 and 2025-01 are missing.

Covered now:
- silver: `citations_by_vintage`, `provider_by_vintage`, `penalties_by_vintage` and `surveys_by_vintage` (one file
  per `vintage`); by default the latest month;
- gold: `spine` and `labels` (one file, split by `t`); by default every month (`all_months: true`), because the
  latest `t` holds only immature rows, and the train and test rows are in the earlier months.

## Running it

```
precursor-pipeline validate --layer silver                 # latest month of each table vs the month before
precursor-pipeline validate --layer silver --all-months    # every month vs the month before
precursor-pipeline validate --layer gold                   # every gold month (the layer's default)
precursor-pipeline validate --layer gold --table labels --month 2025-06
precursor-pipeline validate --layer silver --strict        # drift warnings also fail
precursor-pipeline validate --layer silver --write-schema  # (re)infer configs/contracts/<table>.yaml
```

| Exit | Meaning |
|---|---|
| 0 | No schema violation. Drift warnings may be present: they are in `anomalies.json`. |
| 2 | At least one schema violation (error). With `--strict`, any warning too. |
| 3 | Config or usage error: unknown layer or table, a missing or broken schema file, a month the table does not have, `--month` together with `--all-months`, a bad flag. |
| 1 | Unexpected error. |

## Outputs

| Where | What |
|---|---|
| `reports/validation/run=<id>/stats_<table>.json` | Per month looked at (the checked months and the months they were compared with): row count, and per column dtype, null %, distinct count, min/max/mean/std for numbers, min/max for dates, top values for text and booleans |
| `reports/validation/run=<id>/anomalies.json` | Every anomaly, with its table, month, column, check, `severity` (`error` or `warning`), observed and expected values, and a detail line. Warnings are listed even when the run exits 0. |
| `reports/validation/run=<id>/logs/` | The process log |
| `metadata/validation/run_date=<date>/validation_run_<id>_<ts>.json` | Run manifest: exit code, layer, months checked, counts |
| `dq_results` (metadata store) | One row per anomaly (`node = validation`, layer, table, vintage month, check, column, severity, observed, expected) |

## Schema baselines

`--write-schema` infers each table's schema from all its months. It records the dtype, `nullable` (whether any month
has a null), `min`/`max` for numbers, and `allowed` for text columns with at most `low_cardinality_max` distinct
values (never for `vintage_month` or `t`, where every new month is a new value). The files are then tightened by
hand, and every edited line is marked `# hand`:
- the columns are exactly what silver writes: the table's `columns` in `configs/keep_columns.yaml` plus
  `vintage_month` and `cutoff_date`, with the dtypes of `configs/column_map.yaml`. `unique` is the table's `key`
  there. A test fails when the contract and those configs disagree.
- `pattern`s: `ccn` `^[0-9A-Z]{6}$` and `deficiency_tag` `^[0-9]{4}$` (as in `harmonize/normalize.py`), `ftag`
  `^F[0-9]{4}$`, `state` two capital letters, `vintage_month` and `t` `YYYY-MM`;
- `allowed` is taken from the configs, not from history, wherever a config closes the domain: every `*_fn` column
  allows exactly the footnote codes of `footnote_reasons` in `configs/features.yaml` (the codes gold maps), and the
  mapped columns (`sprinkler_status`, `special_focus_status`, `resident_family_council`, `survey_type`) allow
  exactly the values of `configs/value_maps.yaml`. Tests check both. `survey_type` is a warning only, because
  silver writes `OTHER` for an unknown type on purpose.
- free CMS text whose list grows (`ownership_type`, `deficiency_corrected`) keeps its list as a warning only
  (`allowed_severity: warning`). Values that gold logic depends on stay hard errors: `deficiency_category`,
  `penalty_type`, `provider_type` and `scope_severity`.
- `scope_severity` may be any of A–L. The data shows only B–L, but A is a valid CMS code.
- Data-driven maxima are replaced by domain bounds: counts, hours and amounts are `>= 0` with no upper bound,
  ratings stay 1–5, turnover stays 0–100 and the inspection cycle stays 1–3. `payment_denial_length_days` has no
  range, because CMS data has negative lengths.
- The gold `labels` file lists the `status_*` values from `gold/labels.py`, the `split_*` columns and values from
  `gold/splits.py`, and the three label `rules`.

Running `--write-schema` again overwrites the files and drops the hand edits. Commit the files first and re-apply
the edits from the diff.

## Why Pandera

Pandera is already a dependency of the pipeline image and validates Polars frames natively, so no conversion to
pandas and no extra service is needed. The schema lives in a reviewable YAML file per table that turns into code at
run time. Great Expectations and TFDV would each add a heavy dependency and their own store for expectations and
statistics. Here the statistics are plain JSON reports and `dq_results` rows next to the other nodes' outputs.

## DAG wiring (not built yet)

It comes after the `cms_nh_pipeline` DAG (#23). The plan is two tasks:
- `validate_silver`: `precursor-pipeline validate --layer silver --run-id R`, after `build_silver` and before gold,
  so bad silver never reaches gold;
- `validate_gold`: `precursor-pipeline validate --layer gold --run-id R` (every month by default), with the
  `cms_nh_gold` asset moved onto this task, so "published" means "validated".

Exits 2 and 3 are not retried, as for the other nodes. An alerting task can read `anomalies.json` or `dq_results`,
for both errors and warnings.

## Follow-ups

- `facility_month_dataset` and the feature tables: a contract from
  `configs/contracts/facility_month_dataset_spec_columns.csv` (columns, dtypes, `days_since_* >= 0`, counts `>= 0`,
  footnotes, splits), and per-table `partition_key` / `month_column`, since it is stored as `t=YYYY-MM` partitions.
  The next PR.
- Validate gold before it is published: read `gold/_staging/run=<id>/`, and read published gold through
  `read_current_gold` rather than `gold/` directly.
- The same latest month in every table: today each table checks its own latest month, so a table missing its newest
  vintage passes.
- A `--month` that one table lacks: a recorded `table_missing` error (exit 2) instead of a config error (exit 3).
- Date rules, with tolerance: an error when an event date is more than 31 days after `cutoff_date` (the signature of a
  wrong vintage), a warning with a count for any date after it, and `date_first_approved <= cutoff_date`. Not a hard
  `survey_date <= cutoff_date`: valid rows sit up to 9 days after it.
- Drift: thresholds per table (provider and spine ~2 %, citations ~3 %, surveys ~10 %, penalties ~25 %), the known
  breaks listed in config, and the gap length in the message; drift for booleans and category shares (e.g. the G+
  share of `scope_severity`); the positive rate among labelled rows per `t`.
- Inference: warn when a column's dtype differs between months, and round-trip `Datetime`, `Decimal` and `Enum`.
- Retries and reads: replace a run's `dq_results` rows on a retry with the same run id; write NaN as null; read each
  month once and list the partitions once.
- Cover the other silver tables (ledgers, lookups, universe, QM), which are not one file per month.
- The real gold in `data/gold` predates the split columns, and the real silver predates `survey_fn` and
  `high_performing_icon`, so `validate` on them reports those columns as missing until they are rebuilt.
