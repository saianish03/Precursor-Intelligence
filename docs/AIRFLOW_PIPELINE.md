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

**Why a teardown, not `trigger_rule="all_done"`** (measured on the local stack):
- **The rule:** Airflow sets a DAG run's state from its **leaf tasks**, the tasks with nothing downstream. The run is `failed` if a leaf failed or is `upstream_failed`, and `success` if every leaf succeeded or was skipped.
- **With `release_lock` as a plain `all_done` task** at the end, it is the only leaf. It succeeds in every case (releasing the lock works even after silver failed), so **every run shows `success`**.
  - Run E: `discover_download` failed with exit 3 and all downstream tasks were `upstream_failed`, yet the run showed `success`.
  - Run B (silver exit 2) did the same.
  - No alert would ever fire.
- **As a teardown** (`release.as_teardown(setups=acquire)`), Airflow leaves it out of the leaves. The leaf becomes `build_silver`, so a failure anywhere upstream makes the run `failed`, while the release still runs (`ALL_DONE_SETUP_SUCCESS`). Runs E2 and E4: `failed`, lock released.
- **Not with `on_failure_fail_dagrun=True`** either. That flag (meant to fail the run when the teardown fails) makes Airflow count the teardown as a leaf again, so its success hid the failure once more (run E3).
- **What covers a release that fails:** 2 retries, and the next run's `acquire_lock` failing loudly (or taking over the lock after 12 h).

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

## 7. What makes it production-grade

**A baseline version** of this DAG would:
- use `PythonOperator`s that import the pipeline code into the scheduler;
- retry every failure the same way;
- release the lock in an `all_done` task;
- read secrets from the DAG file;
- run everything on one Docker network.

It would work on a demo day. The table lists what this implementation does instead, and why it matters once the pipeline runs unattended every day on one shared VM.

| Area | This pipeline | Why it matters | Where |
|---|---|---|---|
| **Isolation and packaging** | Every task is one CLI command in its own `precursor-pipeline` container; the DAG holds no logic | The pipeline's dependencies (polars, gcsfs, …) never mix with Airflow's. A task crash cannot take the scheduler down. The same tested command runs by hand, in CI and in Airflow | `dags/`, `docker/pipeline.Dockerfile` |
| **Network segmentation** | Three networks: Airflow + Postgres; an internal Docker-API network (scheduler and proxy only); a task network (Postgres + internet egress only) | Task containers open untrusted zips from the internet. If one is compromised, it cannot reach the Airflow API, the scheduler or the Docker API, which would mean root on the VM | `docker-compose.yaml` (verified with connection tests) |
| **Docker API through a proxy** | `docker-socket-proxy` exposes only the containers and images API sections | Airflow never holds the raw Docker socket | `docker-compose.yaml` |
| **Least-privilege data access** | Separate login `pi_meta` that owns only `precursor_meta` (no CONNECT on `airflow` or `postgres`); containers run as non-root (uid 10001 / 50000) | A pipeline bug or compromise cannot touch Airflow's own state | `init-db.sh`, Dockerfiles |
| **Secrets handling** | The stack refuses to start without its secrets (`${VAR:?}`); none in images, DAGs or logs; the DB URL is passed as `private_environment` (hidden in the UI); Fernet key set; ports bound to 127.0.0.1 | No default passwords in production; nothing secret lands in Git, image layers or the UI | `.env.example`, `pipeline_operator.py` |
| **Input hardening** | The `entry_ids` param is pattern-checked; ids from `entries --pending` are validated before becoming commands; downloads go through an HTTPS allow-list with zip-safety limits | A trigger parameter cannot inject CLI arguments; a tampered catalog cannot make the pipeline fetch or unpack something dangerous | DAG, `ingestion/http.py`, `zipsafe.py` |
| **Failure semantics** | Exit-code contract: **1** = transient, retried (2 × 5 min); **2** = bad data and **3** = config, failed at once (`AirflowFailException`) | Retrying a quarantined zip or an unmapped header only wastes ten minutes and hides the alert. Transient network or DB errors heal themselves | `PipelineDockerOperator` (tested against the pinned provider) |
| **Fail-closed data checks** | Bronze and silver write nothing for a file or partition that fails a check, record it as `failed`, and retry it next run | Bad data never reaches gold silently. Run B proved it: 5 new CMS columns stopped silver for that month only | bronze and silver nodes |
| **Concurrency control** | One `data_pipeline` lock row for the whole DAG run (re-entrant for the same run id, owner-checked release, stale takeover after 12 h) plus `max_active_runs=1` | A manual run and a scheduled run never write the same metadata at once. A run killed by the VM's nightly stop does not block the pipeline forever | `common/io/run_lock.py`, DAG |
| **Honest run state** | `release_lock` is a **teardown** of `acquire_lock`: it always runs once the lock was taken, but does not decide the run's state | With an `all_done` cleanup leaf, every failed run shows green and no alert fires (verified, §3) | DAG |
| **Safe retries and idempotency** | One run id per DAG run; every task skips work already done (HEAD-checked zips, sha-keyed bronze, content- and config-keyed silver) | Retrying a task, clearing it in the UI or re-running a day is always safe. A rerun with nothing new is a byte-identical no-op (verified) | all nodes |
| **Incremental and parallel** | Per-zip mapped tasks (≤ 3 at a time) only for zips not yet processed; 0 tasks on a quiet day | A normal daily run costs one catalog fetch plus HEAD requests. A new release processes only its own zips (≈ 1 min) | ingestion TaskGroup |
| **Republished data** | A changed zip is stored side by side, reloaded in bronze, and flagged `revised_after_publication`; only the affected partitions are rebuilt | CMS silently corrects old releases. The pipeline picks that up and records that history was rewritten | ingestion, bronze, silver |
| **Resource governance** | Sized for an e2-standard-2: 1 API worker, parallelism 4, optional per-task memory limit, capped container logs, auto-removed containers | On a 2-vCPU/8 GB VM, one heavy task cannot starve Postgres or the scheduler, and logs cannot fill the disk | compose, `pipeline_operator.py` |
| **Observability and audit** | One run id on every metadata row, log file, report folder and manifest; per-process JSON logs shipped to the bucket; per-node assets; Checkpoint reports | "What did run X do, and why did it fail?" is answerable from one id, after the containers are long gone | §5, §6 |
| **Reproducibility** | Pinned base images and tools; `uv.lock --frozen`; Airflow installed under its own constraints; provider pinned, with a test that fails if its error format changes; `PI_GIT_SHA` in manifests | The same commit builds the same images. An Airflow or provider upgrade cannot silently break the exit-code handling | Dockerfiles, `tests/dags` |
| **Environment parity** | The laptop and the VM use identical compose, images and DAG; only `.env` differs. On a laptop, manual runs only, so unpausing cannot start a full-archive run | What was tested locally is what runs in production | `.env.example` |
| **Operability** | Params `mode` (backfill or monthly), `calibrate`, `stop_after` (node-by-node first backfill), `entry_ids` (development subsets); a runbook for build, run, inspect and clean up | Operators control the pipeline from the UI without code changes | DAG, runbook |
| **Tested** | DAG integrity and operator tests in the Airflow image; repo unit tests; real end-to-end runs on CMS data (runbook §8) | Wiring mistakes are caught before deploy, not at 3 a.m. | `tests/dags`, runbook |

## 8. Files

| File | Purpose |
|---|---|
| `dags/cms_nh_pipeline.py` | the DAG |
| `dags/utils/pipeline_operator.py` | `PipelineDockerOperator`, `docker_kwargs()`, the JSON result parser |
| `docker/pipeline.Dockerfile`, `docker/airflow.Dockerfile`, `.dockerignore` | the two images |
| `docker/airflow-local/docker-compose.yaml`, `init-db.sh`, `.env.example` | the stack |
| `tests/dags/` | DAG integrity and exit-code tests (run in the airflow image) |
| `docs/runbooks/airflow-local.md` | how to build, run, inspect and clean up; laptop and GCP; verified results |
