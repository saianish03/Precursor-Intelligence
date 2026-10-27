# configs/

| Path | Status | Purpose |
|---|---|---|
| `dev.yaml` | ✅ in use | Development backends: local disk for files, SQLite for metadata (local Postgres planned) |
| `prod.yaml` | ✅ in use | Test / production backends: GCS for files (BigQuery metadata planned) |
| `logging.yaml` | ✅ in use | Standard-library logging config (console text/JSON + per-run JSON-lines file) |
| `ingestion/sources.yaml` | ✅ in use | Step 1: CMS endpoints, scope dates, the 7 in-scope tables (file patterns, dataset IDs), HTTP and security limits |
| `labels.yaml` | ✅ in use | Gold labels: G+ severities, horizons, maturity buffers (B_cit, B_pen), strict censoring |
| `splits.yaml` | ✅ in use | Gold spine and training start (`spine_start`), `test_months`, `calib_months`, and `required_labels` (gold stops when one has no training month, or its training rows lack a positive or a negative; `[]` turns the gate off). `precursor-pipeline gold --splits-config` reads another file |
| `validation.yaml` | ✅ in use | `precursor-pipeline validate`: tables per layer, drift thresholds (row count, null rate, mean shift), the columns never checked for drift, and which layers check every month by default (docs/VALIDATION.md) |
| `base.yaml` | planned | Shared project settings: project, region, bucket names, BigQuery datasets |
| `column_map.yaml` | ✅ in use | Silver: raw CMS header → canonical column per table, with vintage ranges and dtype (`to: null` = still valid) |
| `keep_columns.yaml` | ✅ in use | Silver: the columns each silver table keeps, and its within-vintage key |
| `value_maps.yaml` | ✅ in use | Silver: Y/N, survey-type and enum maps, ownership groups, "not corrected" values |
| `calibration.yaml` | ✅ in use | Silver calibration (Checkpoints 3-4): lag coverage, left-censoring floors, fully-observed window, QM bridge pairs and drops. Not part of the silver `config_hash`: it only shapes the reports, so editing it rebuilds no partition |
| `labels.yaml`, `qm_measures.yaml` | planned | Committed by a human from the Checkpoint 3 and 4 proposals (`reports/step4`, `reports/step5`) |
| `contracts/` | ✅ in use | Pandera data contracts: one baseline schema per validated table (`<table>.yaml`: key, dtype, nullable, ranges, allowed values, patterns, label rules), inferred by `validate --write-schema` and tightened by hand (`# hand`); also `facility_month_dataset_spec_columns.csv` |
| `prompts/` | planned | Versioned LLM prompt YAML used by `src/precursorintelligence/agents` |

Select an environment with `--env dev|prod`. No secrets in these files: `${VAR}` placeholders are filled from environment variables or Secret Manager.
