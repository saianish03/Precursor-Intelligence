# Schema and statistical data validation

`precursor-pipeline validate` checks silver or gold month by month. It reads the tables straight from the object
store and changes nothing in them. It takes no lock and runs no freshness check, so it also works on a silver built
with older configs. Code: `src/precursorintelligence/validation/`. Settings: `configs/validation.yaml`.

## What it checks

| Kind | Check | Severity | Source |
|---|---|---|---|
| Schema (Pandera) | `missing_column`, `unexpected_column`, `wrong_dtype`, `null_values` (in a non-nullable column), `out_of_range` (`min`/`max`), `allowed_values`, `str_length` | error | `configs/contracts/<table>.yaml` |
| Schema | `table_missing`: a configured table has no months in the store | error | `configs/validation.yaml` |
| Drift vs the previous month | `row_count_change` (more than ±`row_count_change_pct` %) | warning | `drift` in `configs/validation.yaml` |
| | `null_rate_jump` (null % up by more than `null_rate_jump_pts` points) | warning | |
| | `mean_shift` (mean moved by more than `mean_shift_std` previous-month standard deviations) | warning | |

Drift is not checked for the columns in `drift.skip_columns` (`y_*`, `status_*`, `split_*`). Labels are censored for
the last months and splits are rolling windows, so they move by design. Their values are still schema-checked:
`status_*` must be one of positive, negative, censored or immature, and `split_*` one of train, calib, purge, test
or score. A table's first month has no previous month, so it gets only the schema checks.

Covered now:
- silver: `citations_by_vintage`, `provider_by_vintage`, `penalties_by_vintage` and `surveys_by_vintage` (one file
  per `vintage`);
- gold: `spine` and `labels` (one file, split by `t`).

## Running it

```
precursor-pipeline validate --layer silver                 # latest month of each table vs the month before
precursor-pipeline validate --layer silver --all-months    # every month vs the month before
precursor-pipeline validate --layer gold --table labels --month 2025-06
precursor-pipeline validate --layer silver --strict        # drift warnings also fail
precursor-pipeline validate --layer silver --write-schema  # (re)infer configs/contracts/<table>.yaml
```

| Exit | Meaning |
|---|---|
| 0 | No schema violation. Drift warnings may be present: they are in `anomalies.json`. |
| 2 | At least one schema violation (error). With `--strict`, any warning too. |
| 3 | Config or usage error: unknown layer or table, a missing or broken schema file, a month the table does not have, a bad flag. |
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
values. The files are then tightened by hand, and every edited line is marked `# hand`:
- `ccn` must be 6 characters, `vintage_month` and `t` 7 (`YYYY-MM`), and `state` 2;
- `scope_severity` may be any of A–L. The data shows only B–L, but A is a valid CMS code.
- Data-driven maxima are replaced by domain bounds: counts, hours and amounts are `>= 0` with no upper bound,
  ratings stay 1–5, turnover stays 0–100 and the inspection cycle stays 1–3. `payment_denial_length_days` has no
  range, because CMS data has negative lengths.
- The gold `labels` file lists the `status_*` values from `gold/labels.py` and the `split_*` columns and values from
  `gold/splits.py`.

Running `--write-schema` again overwrites the files and drops the hand edits. Commit the files first and re-apply
the edits from the diff.

## DAG wiring (not built yet)

It depends on the Airflow DAG branch (`feat/dp_airflow_v1`). The plan is two tasks:
- `validate_silver`: `precursor-pipeline validate --layer silver --run-id R`, in the silver TaskGroup after
  `build_silver`;
- `validate_gold`: `precursor-pipeline validate --layer gold --run-id R`, after `build_gold`.

Exits 2 and 3 are not retried, as for the other nodes. An alerting task can read `anomalies.json` or `dq_results`,
for both errors and warnings.

## Follow-ups

- Date rules: no `min`/`max` for dates yet, because they move every month. Candidates are `survey_date <= cutoff_date`
  and `correction_date >= survey_date`, written as row-level checks in the schema YAML.
- Validate gold in `gold/_staging/` before promotion, instead of after it.
- Cover the other silver tables (ledgers, lookups, universe, QM), which are not one file per month.
- Drift on categories (share of each `allowed` value, e.g. the G+ share of `scope_severity`).
- The real gold in `data/gold` predates the split columns, so `validate --layer gold` on it reports `split_*` as
  missing until gold is rebuilt.
