# The data pipeline in Airflow

**What this covers:** how the CMS data pipeline runs as one Airflow DAG, what each task does, how the run lock and the exit codes work, what the metadata database holds (its schema), and where logs and run artifacts go.
**How to run it** on a laptop or on GCP: [`runbooks/airflow-local.md`](runbooks/airflow-local.md).
**Rules behind it:** `DATA_PIPELINE_SPEC.md` §15 (one DAG, five nodes) and §0.6–0.7 (storage and metadata).
**Status (2026-10-21):** ingestion → bronze → silver implemented and run end to end on real CMS data (runbook §8). `check_release` and `gold` are TODO slots in the DAG.

---

## 1. The big picture

```
                      ┌──────────────────────────── docker compose (docker/airflow-local/) ──────────────────────────┐
                      │                                                                                             │
 you / @daily ──────▶ │  airflow-apiserver ── UI + REST (127.0.0.1:8080)                                            │
                      │  airflow-scheduler ── decides what runs; LocalExecutor runs the tasks ──┐                    │
                      │  airflow-dag-processor ── parses dags/                                  │ Docker API          │
                      │  airflow-triggerer                                                      ▼ (internal network) │
                      │  postgres:16  ┌ airflow         (Airflow's own state)            docker-proxy               │
                      │               └ precursor_meta  (pipeline metadata, §5)               │                     │
                      └───────────────────────────────────────────────────────────────────────┼─────────────────────┘
                                                                                              │ starts one container per task
                                                                                              ▼
                                         precursor-pipeline container:  `precursor-ingest …` or `precursor-pipeline bronze|silver …`
                                           ├── reads/writes files:   laptop: a host folder at /app/data   GCP: gs://precursor-intel-mlops-data
                                           └── reads/writes metadata: Postgres `precursor_meta` (PI_DATABASE_URL)
```

**Design rules:**
- **DAGs are thin.** Every task runs exactly **one CLI command** in its own container from the `precursor-pipeline` image; no pipeline logic lives in `dags/`.
  - The same commands run by hand, in tests and in Airflow.
  - Airflow only orders them, retries them, and records their state.
- **One image for all data tasks.** `docker/pipeline.Dockerfile`: the repo package plus `configs/`.
  - The Airflow services run a second image, `docker/airflow.Dockerfile` (Airflow 3.1.8 + the Docker provider 4.6.1).
- **Laptop = GCP.** The compose file, images and DAG are identical; only `docker/airflow-local/.env` differs (§6).
- **Networks are fenced off.** Three Docker networks, each verified:
  - `pi-airflow`: the Airflow services and Postgres;
  - `pi-airflow_docker-api` (internal): only the scheduler and `docker-proxy`;
  - `pi-airflow-tasks`: the task containers, which reach **only** Postgres and internet egress (for CMS). Task containers open untrusted zips, so they cannot reach the Airflow API, the scheduler or the Docker API.
- **Secrets are required.** The stack refuses to start without its passwords, JWT secret and Fernet key (from `.env`).

## 2. The DAG `cms_nh_pipeline`

`dags/cms_nh_pipeline.py`. One DAG for the whole data pipeline; each node is a TaskGroup.

```
[check_release]  (TODO: spec §2.0, gates the run)
      │
      ▼
┌ ingestion ────────────────────────────────────────────────────────────────────────────────────────────┐
│ acquire_lock → discover_download → pending_entries → mapped_commands → extract_inventory[×N] → resolve_report │──▶ asset cms_nh_landing
└───────────────────────────────────────────────────────────────────────────────────────────────────────┘
      │  continue_after_ingestion  (stop_after)
      ▼
┌ bronze ─ load_bronze ┐──▶ asset cms_nh_bronze
      │  continue_after_bronze  (stop_after)
      ▼
┌ silver ─ build_silver ┐──▶ asset cms_nh_silver
      │
[gold]  (TODO: spec §7–§11)
      │
release_lock  (teardown of acquire_lock)
```

| Task | Command (`--env <dev|prod> --run-id {{ run_id }}` appended) | What it does |
|---|---|---|
| `ingestion.acquire_lock` | `precursor-ingest lock acquire` | Takes the `data_pipeline` lock for this run (§3) |
| `ingestion.discover_download` | `precursor-ingest step1 --steps discover,download [--entry-ids …]` | Reads the CMS archive catalog; downloads new or changed zips (HEAD skips unchanged ones); quarantine → validate → landing |
| `ingestion.pending_entries` | `precursor-ingest entries --pending` | Prints the zips still to extract, as a JSON list (the task's XCom) |
| `ingestion.mapped_commands` | (Python, in Airflow) | Turns that list into one command per zip; validates the ids |
| `ingestion.extract_inventory` | `precursor-ingest step1 --steps extract,inventory --entry-ids <id>` | **Mapped, one task per pending zip** (≤ 3 at a time). Extracts the 7 tables and records every member. 0 tasks on a quiet day |
| `ingestion.resolve_report` | `precursor-ingest step1 --steps resolve,report` | Picks one file per (table, month), lists gaps and anomalies, writes the Checkpoint 1 report and the run manifest. `none_failed`, so it also runs when nothing was pending |
| `bronze.load_bronze` | `precursor-pipeline bronze` | Lossless all-text Parquet per selected file; reconciles against the inventory; content hash, `stale_refresh`, `revised_after_publication` |
| `silver.build_silver` | `precursor-pipeline silver --calibrate\|--no-calibrate` | Typed per-vintage tables, ledgers, universe, QM inventory; Checkpoint 2 report (with calibration also Checkpoints 3–4) |
| `continue_after_*` | (Python) | `stop_after` short-circuit: skip the next node |
| `release_lock` | `precursor-ingest lock release` | Releases the lock (§3) |

**Params** (set when you trigger a run):

| Param | Default | Meaning |
|---|---|---|
| `mode` | `monthly` | `backfill` also runs silver's calibration (lag and QM proposals) |
| `calibrate` | `false` | calibration in a monthly run |
| `stop_after` | `null` | `ingestion` or `bronze`: run up to that node (the first backfill, checked node by node) |
| `entry_ids` | `""` | development only: ingest just these catalog entries (comma-separated). Empty = every in-scope zip |

**Schedule:** `PI_DAG_SCHEDULE`. Production polls CMS once a day (CMS sends no notification); a laptop uses manual runs only. `max_active_runs=1`, `catchup=False`.

**Idempotency:** every task can be re-run with the same run id.
- Unchanged zips, files and partitions are skipped.
- A re-run with nothing new is a no-op: all bronze and silver files stay byte-identical (verified).

## 3. Run id and run lock

- **One run id for the whole DAG run:** every task gets `--run-id {{ run_id }}`. It is written on every metadata row, log file, report folder and manifest of that run, so one id finds everything.
- **One run at a time:** the `data_pipeline` lock is one row in `precursor_meta.run_lock`.
  - `acquire_lock` takes it; every other task only *checks* that its run holds it (exit 3 if not).
  - `release_lock` frees it. It is an Airflow **teardown** of `acquire_lock`.
    - It runs whenever the lock was taken: after success, a failure or a `stop_after` skip.
    - It does not run if `acquire_lock` failed (another run holds the lock).
    - A teardown does not decide the run's state, so a failed run shows **failed**.
    - It retries twice: release is owner-checked and idempotent.
  - A lock left by a killed run (for example, the VM stopped mid-run) is taken over after 12 h (`run_lock.stale_after_hours`).
  - Manual CLI runs without `--run-id` take and release the lock themselves.

## 4. Exit codes and retries

Every CLI command uses the same codes. `dags/utils/pipeline_operator.py` (`PipelineDockerOperator`) maps them onto Airflow:

| Exit | Meaning | Airflow |
|---|---|---|
| 0 | ok, or nothing new | success |
| 1 | error (network, storage, bug) | retried, 2 × 5 min |
| 2 | data or validation failure (quarantined zip, unmapped header, reconciliation) | **failed at once, no retry**; the data or config must be fixed first |
| 3 | config or wiring (bad env, a stale task list, the lock not held) | **failed at once, no retry** |

A task that fails makes the downstream tasks `upstream_failed`. The lock is still released, and the run shows `failed`.

## 5. The metadata database `precursor_meta`

**What it is:** Postgres, one database next to Airflow's own `airflow` database on the same server.
- Created by `docker/airflow-local/init-db.sh` with its own login `pi_meta`.
- The tables are created by the pipeline itself (`MetadataStore.ensure_schema`, from `src/precursorintelligence/common/io/schemas.py`).
- Values are portable SQL types; dates and timestamps are ISO-8601 strings.

**How the tables connect:**

```
catalog (entry_id) ─┬─< download_log (entry_id, run_id)          one row per download decision
                    ├─< extracted_files (entry_id, member_path)  one row per extracted file
                    └─< inventory (entry_id, member_path) ── status='selected' gives one file per (table_name, vintage_month)
                                │
                                ├─< bronze_files (table_name, vintage_month, zip_sha256)    one row per bronze load/failure
                                │       └── vintage_flags (vintage_month, flag, table_name)  stale_refresh, revised_after_publication
                                └─< silver_partitions (source_table, vintage_month, zip_sha256)  one row per silver build/failure
anomalies (run_id, entry_id)    run_log (run_id, task)    run_lock (lock_name)    source_datasets (dataset_id)
```

**Tables.** Columns in order (generated from `schemas.py`):

| Table | Key | Written by | Holds | Columns |
|---|---|---|---|---|
| `catalog` | `entry_id` | ingestion: discover | every zip CMS advertises (`theme:<date>:<zip>`, `current:…`) | name, nid, type, theme, zip_date, zip_name, url, size_bytes, source_endpoint, first_seen_at, last_seen_at, active |
| `source_datasets` | `dataset_id` | ingestion: discover | metastore release dates of the 7 datasets | table_name, title, modified, released, next_update, current_download_url, fetched_at |
| `download_log` | `id` (append-only) | ingestion: download | every download decision: `downloaded`, `skipped`, `unchanged`, `republished`, `adopted`, `rejected`, `quarantined`, `failed` | run_id, entry_id, zip_name, zip_date, type, url, catalog_size, expected_size, actual_size, sha256, http_last_modified, http_content_type, object_key, object_uri, status, detail, logged_at |
| `extracted_files` | `entry_id, member_path` | ingestion: extract | each in-scope file taken out of a zip | zip_sha256, table_name, object_key, size_bytes, crc32, run_id, extracted_at |
| `inventory` | `entry_id, member_path` | ingestion: inventory, resolve | every zip member: table, Processing Date, vintage, row count, header hash; `status` = `pending` (not yet resolved), `selected`, `duplicate_vintage`, `out_of_scope`, `other`, `junk`, `unresolved` | zip_name, zip_date, zip_type, zip_sha256, layout, file_name, table_name, extracted_key, processing_date, vintage_month, vintage_source, row_count, n_columns, header_hash, header_json, file_size_bytes, compressed_size_bytes, crc32, in_manifest, status, status_detail, run_id, inventoried_at |
| `anomalies` | `id` | ingestion: resolve | per run: duplicates, gaps, partial zips, size outliers, … | run_id, entry_id, kind, severity, detail, created_at |
| `bronze_files` | `id` (append-only) | bronze | one row per file loaded (`loaded`, `reloaded`) or `failed`; skipped files get none | run_id, table_name, vintage_month, entry_id, member_path, zip_sha256, header_hash, status, row_count, blank_rows, encoding, content_hash, content_unchanged, object_uri, detail, loaded_at |
| `vintage_flags` | `vintage_month, flag, table_name` | bronze | current value of `stale_refresh` (release level, `table_name=''`) and `revised_after_publication` | value, detail, run_id, updated_at |
| `silver_partitions` | `id` (append-only) | silver | one row per silver table and vintage `built` or `failed`; the skip key is content_hash + zip_sha256 + config_hash | run_id, table_name, source_table, vintage_month, status, content_hash, zip_sha256, config_hash, row_count, stats, object_uri, detail, built_at |
| `run_log` | `id` (append-only) | every node (`task_scope`) | one row per step of a run: start, end, duration, status, details (JSON) | run_id, task, entry_id, status, started_at, finished_at, duration_s, detail |
| `run_lock` | `lock_name` | lock commands | the `data_pipeline` lock (0 or 1 row) | run_id, holder, acquired_at |

**Useful queries:**
```sql
-- the file each (table, month) uses
select table_name, vintage_month, zip_name, row_count from inventory where status = 'selected' order by 1, 2;
-- what a run did, per node
select 'download' as node, status, count(*) from download_log where run_id = :run group by 2
union all select 'bronze', status, count(*) from bronze_files where run_id = :run group by 2
union all select 'silver', status, count(*) from silver_partitions where run_id = :run group by 2;
-- release flags
select * from vintage_flags where value order by vintage_month;
-- slowest steps of a run
select task, entry_id, duration_s from run_log where run_id = :run order by duration_s::float desc limit 10;
```

**Known gaps** (spec §0.6, not built yet):
- the BigQuery `meta` mirror (task `mirror_metadata`);
- the `dq_results` table;
- `status/latest.json`;
- append-only writes with "latest" views (D5): `catalog`, `inventory`, `extracted_files` are still upserted.

Also: `run_log` marks a step failed only when it raises. Silver and bronze report partition failures in their own tables (issue #10). Changing a column or type needs a migration: `create_all` never alters an existing table.

**Airflow's own database `airflow`** is managed by Airflow (`airflow db migrate`). It holds DAG runs, task instances (state, tries, timings), XComs and the serialized DAGs. The pipeline never writes to it.

## 6. Logs and run artifacts

| What | Where (under the bucket root) | Written by |
|---|---|---|
| Per-task JSON log (one file per container) | `reports/<node>/run=<run_id>/logs/` (`step1` for ingestion) | every CLI process |
| Checkpoint reports | `reports/step1/` (inventory), `step2/` (harmonisation), `step4/` (posting lag), `step5/` (universe and QM), each as `run=<id>/` and `latest/` | ingestion, silver |
| Per-run manifests (D11) | `metadata/<ingestion\|bronze\|silver>/run_date=<date>/<layer>_run_<run_id>_<ts>.json` | each node |
| Airflow task logs (container output) | `airflow-logs` volume: `dag_id=…/run_id=…/task_id=…/attempt=N.log` | Airflow |
| Data layers | `landing/`, `quarantine/`, `bronze/`, `silver/`, `gold/dim_facility_latest` | the nodes |

**Environment, the only difference between a laptop and GCP** (`docker/airflow-local/.env`):

| Setting | Laptop | GCP VM |
|---|---|---|
| `PI_PIPELINE_ENV` | `dev` (configs/dev.yaml: local disk) | `prod` (configs/prod.yaml: GCS) |
| `PI_HOST_DATA_DIR` | a host folder, bind-mounted at `/app/data` | empty |
| `PI_BUCKET_URI` | — | `gs://precursor-intel-mlops-data` |
| `PI_DAG_SCHEDULE` | empty (manual) | a daily cron inside the VM's up-window |
| `PI_TASK_MEM_LIMIT`, `PI_AUTO_REMOVE` | empty, `success` | `6g`, `force` |
| credentials | none | the VM's service account `sa-pipeline` (no key files) |

## 7. Files

| File | Purpose |
|---|---|
| `dags/cms_nh_pipeline.py` | the DAG |
| `dags/utils/pipeline_operator.py` | `PipelineDockerOperator`, `docker_kwargs()`, the JSON result parser |
| `docker/pipeline.Dockerfile`, `docker/airflow.Dockerfile`, `.dockerignore` | the two images |
| `docker/airflow-local/docker-compose.yaml`, `init-db.sh`, `.env.example` | the stack |
| `tests/dags/` | DAG integrity and exit-code tests (run in the airflow image) |
| `docs/runbooks/airflow-local.md` | how to build, run, inspect and clean up; laptop and GCP; verified results |
