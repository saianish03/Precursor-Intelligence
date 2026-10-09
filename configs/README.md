# configs/

| Path | Purpose |
|---|---|
| `base.yaml` (planned) | project, region, bucket names, BigQuery datasets |
| `dev.yaml`, `prod.yaml` (planned) | environment overrides (local disk + Postgres for dev; GCS + BigQuery for prod) |
| `ingestion/datasets.yaml` | dataset registry: which CMS tables, file prefixes, dataset IDs |
| `contracts/` | expected schemas per dataset, versioned by effective month, e.g. `provider_info.v2026-07.yaml` |
| `prompts/` | versioned LLM prompt YAML used by `src/precursorintelligence/agents` |

No secrets here. Use environment variables or Secret Manager.
