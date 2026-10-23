# docs/

| File / folder | Contents |
|---|---|
| `DATA_PIPELINE_SPEC.md` | **The single reference for building the data pipeline**: the 5 DAG nodes (`check_release → ingestion → bronze → silver → gold`), every step's rules, checks and decisions |
| `AIRFLOW_PIPELINE.md` | **How the pipeline runs in Airflow**: the DAG, tasks, run lock, exit codes, the metadata DB schema, logs and artifacts (run it: `runbooks/airflow-local.md`) |
| `STEP1_DATA_INGESTION_PIPELINE.md` | Step 1 (ingestion node) technical design and results |
| `STEP1_TODO.md` | Step 1 backlog |
| `BRONZE_CONTRACT.md` | What the bronze node guarantees and what silver may assume: layout, columns, metadata tables, flags, exit codes, measured numbers |
| `GOLD_HANDOFF.md` | Gold so far (spine, labels, `precursor-pipeline gold`) and what the features and assembly work must build: outputs, runs, toy world, leakage and join rules, open questions |
| `STEP2_HANDOFF.md` | Step 2 profiling facts and reference drafts (`data/step2_reference/`) |
| `adr/` | architecture decision records (`0001-monorepo.md`, …) |
| `data/` | data dictionary notes, footnote codes, schema-change log |
| `reference/` | reference material, e.g. `column_lineage.xlsx` (raw CMS column → feature lineage) |
| `runbooks/` | how to operate, backfill, recover, promote / roll back |
