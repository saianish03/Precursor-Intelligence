---
paths:
  - "src/precursorintelligence/ingestion/**"
  - "src/precursorintelligence/validation/**"
  - "src/precursorintelligence/features/**"
  - "configs/ingestion/**"
  - "configs/contracts/**"
  - "configs/column_map.yaml"
  - "configs/feature_registry.yaml"
  - "configs/labels.yaml"
  - "dags/**"
---

# Data pipeline rules
- Load the `cms-data-domain` skill before changing ingestion, ledgers, features or labels.
- Read CMS CSVs with `dtype=str` and explicit date parsing; CCN stays 6-character text.
- Every new raw column goes through `configs/column_map.yaml`; every new feature gets a `feature_registry.yaml`
  entry (family, source, as-of rule, earliest vintage, model_use).
- Tasks are idempotent and keyed by vintage; bronze is never overwritten.
- Add or update a Pandera contract (`configs/contracts/`) and, for features or labels, a leakage test.
