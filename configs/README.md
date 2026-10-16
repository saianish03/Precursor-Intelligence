# configs/

| Path | Status | Purpose |
|---|---|---|
| `dev.yaml` | ✅ in use | Development backends: local disk for files, SQLite for metadata (local Postgres planned) |
| `prod.yaml` | ✅ in use | Test / production backends: GCS for files (BigQuery metadata planned) |
| `logging.yaml` | ✅ in use | Standard-library logging config (console text/JSON + per-run JSON-lines file) |
| `ingestion/sources.yaml` | ✅ in use | Step 1: CMS endpoints, scope dates, the 7 in-scope tables (file patterns, dataset IDs), HTTP and security limits |
| `base.yaml` | planned | Shared project settings: project, region, bucket names, BigQuery datasets |
| `contracts/` | planned | Expected schemas per dataset, versioned by effective month (draft: `docs/data/step2_reference/column_map.draft.yaml`) |
| `prompts/` | planned | Versioned LLM prompt YAML used by `src/precursorintelligence/agents` |

Select an environment with `--env dev|prod`. No secrets in these files: `${VAR}` placeholders are filled from environment variables or Secret Manager.
