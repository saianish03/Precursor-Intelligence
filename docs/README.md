# docs/

| File / folder | Contents |
|---|---|
| `DATA_PIPELINE_SPEC.md` | **The single reference for building the data pipeline**: the 5 DAG nodes (`check_release → ingestion → bronze → silver → gold`), every step's rules, checks and decisions |
| `AIRFLOW_PIPELINE.md` | **How the pipeline runs in Airflow**: the DAG, tasks, run lock, exit codes, the metadata DB schema, logs and artifacts (run it: `runbooks/airflow-local.md`) |
| `STEP1_DATA_INGESTION_PIPELINE.md` | Step 1 (ingestion node) technical design and results |
| `STEP1_TODO.md` | Step 1 backlog |
| `BRONZE_CONTRACT.md` | What the bronze node guarantees and what silver may assume: layout, columns, metadata tables, flags, exit codes, measured numbers |
| `ALERTS_HANDOFF.md` | Ticket 2 (pipeline alerts): the signals the pipeline already records, what to alert on, how to force each case, and every problem found while wiring gold into the DAG |
| `GOLD_HANDOFF.md` | Gold so far (spine, labels, `precursor-pipeline gold`) and what the features and assembly work must build: outputs, runs, toy world, leakage and join rules, open questions |
| `VALIDATION.md` | `precursor-pipeline validate`: schema (Pandera) and month-over-month drift checks of silver and gold, outputs, exit codes, baselines, DAG wiring, follow-ups |
| `STEP2_HANDOFF.md` | Step 2 profiling facts and reference drafts (`data/step2_reference/`) |
| `bias/` | Data bias slicing on gold (`precursor-pipeline bias-report`): how to run it, what each output means, the gold schema findings and the assignment mapping |
| `adr/` | architecture decision records (`0001-monorepo.md`, …) |
| `data/` | data dictionary notes, footnote codes, schema-change log |
| `reference/` | reference material, e.g. `column_lineage.xlsx` (raw CMS column → feature lineage) |
| `runbooks/` | how to operate, backfill, recover, promote / roll back; `airflow-local.md` (run the stack), `airflow-e2e-simulation.md` (reproduce the end-to-end test runs, written for Claude Code); `gold-snapshot.md` (snapshot the published gold under DVC) |
