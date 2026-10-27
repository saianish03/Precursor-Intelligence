# The data pipeline in Airflow

- **What this covers:** how the CMS data pipeline runs as one Airflow DAG, what each task does, how the run lock and the exit codes work, what the metadata database holds (its schema), and where logs and run artifacts go.
- **How to run it** on a laptop or on GCP: [`runbooks/airflow-local.md`](runbooks/airflow-local.md).
- **Rules behind it:** `DATA_PIPELINE_SPEC.md` §15 (one DAG, five nodes) and §0.6–0.7 (storage and metadata).
- **Status (2026-10-27):** ingestion → bronze → silver → gold implemented and run end to end on real CMS data, from download to a published gold dataset (runbook §8.2). `check_release` and `mirror_metadata` are TODO slots in the DAG.

---

## 1. The big picture

```
                ┌─ docker compose (docker/airflow-local/) ─────────────────────────────────┐
you / @daily -> │ airflow-apiserver       UI + REST API on 127.0.0.1:8080                  │
                │ airflow-scheduler       decides what runs; LocalExecutor runs the tasks  │
                │ airflow-dag-processor   parses dags/                                     │
                │ airflow-triggerer                                                        │
                │ postgres:16             airflow         Airflow's own state              │
                │                         precursor_meta  pipeline metadata (§5)           │
                │ docker-proxy            the only way to the Docker API (scheduler only)  │
                └─────────────────────────────────────┬────────────────────────────────────┘
                                                      │  the scheduler asks docker-proxy to start one container per task
                                                      v
                ┌─ precursor-pipeline container (one per task) ──────────────────────────────────────┐
                │ runs one command:  precursor-ingest ...  or  precursor-pipeline bronze|silver|gold │
                │ files      laptop: a host folder mounted at /app/data                              │
                │            GCP:    gs://precursor-intel-mlops-data                                 │
                │ metadata   Postgres precursor_meta (PI_DATABASE_URL)                               │
                │ network    pi-airflow-tasks: Postgres + internet egress only                       │
                └────────────────────────────────────────────────────────────────────────────────────┘
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
[check_release]   TODO (spec §2.0): gates the run
       |
       v
┌─ ingestion ──────────────────────────────────────────────────┐
│ acquire_lock                                                 │
│   -> discover_download                                       │
│   -> pending_entries -> mapped_commands                      │
│   -> extract_inventory [x N, one per pending zip]            │
│   -> resolve_report                  => asset cms_nh_landing │
└──────────────────────────────────────────────────────────────┘
       |  continue_after_ingestion   (skips the rest if stop_after=ingestion)
       v
┌─ bronze ─────────────────────────────────────────────────────┐
│ load_bronze                          => asset cms_nh_bronze  │
└──────────────────────────────────────────────────────────────┘
       |  continue_after_bronze      (skips the rest if stop_after=bronze)
       v
┌─ silver ─────────────────────────────────────────────────────┐
│ build_silver                         => asset cms_nh_silver  │
└──────────────────────────────────────────────────────────────┘
       |  continue_after_silver      (skips gold if stop_after=silver)
       v
┌─ gold ───────────────────────────────────────────────────────┐
│ build_gold                           => asset cms_nh_gold    │
│   (emitted only when gold was published)                     │
└──────────────────────────────────────────────────────────────┘
       |
       v
[mirror_metadata] TODO (spec §15.3)
       |
       v
release_lock      teardown of acquire_lock: runs whenever the lock was taken (§3)
```

| Task | Command (every one gets `--env <dev\|prod> --run-id {{ run_id }}` appended) | What it does |
|---|---|---|
| `ingestion.acquire_lock` | `precursor-ingest lock acquire` | Takes the `data_pipeline` lock for this run (§3) |
| `ingestion.discover_download` | `precursor-ingest step1 --steps discover,download [--entry-ids …]` | Reads the CMS archive catalog; downloads new or changed zips (HEAD skips unchanged ones); quarantine → validate → landing |
| `ingestion.pending_entries` | `precursor-ingest entries --pending` | Prints the zips still to extract, as a JSON list (the task's XCom) |
| `ingestion.mapped_commands` | (Python, in Airflow) | Turns that list into one command per zip; validates the ids |
| `ingestion.extract_inventory` | `precursor-ingest step1 --steps extract,inventory --entry-ids <id>` | **Mapped, one task per pending zip** (≤ 3 at a time). Extracts the 7 tables and records every member. 0 tasks on a quiet day |
| `ingestion.resolve_report` | `precursor-ingest step1 --steps resolve,report` | Picks one file per (table, month), lists gaps and anomalies, writes the Checkpoint 1 report and the run manifest. `none_failed`, so it also runs when nothing was pending |
| `bronze.load_bronze` | `precursor-pipeline bronze` | Lossless all-text Parquet per selected file; reconciles against the inventory; content hash, `stale_refresh`, `revised_after_publication` |
| `silver.build_silver` | `precursor-pipeline silver --calibrate\|--no-calibrate` | Typed per-vintage tables, ledgers, universe, QM inventory; Checkpoint 2 report (with calibration also Checkpoints 3–4) |
| `gold.build_gold` | `precursor-pipeline gold` | Spine, labels and splits, 5 feature families and `facility_month_dataset`, built in `gold/_staging/run=<id>/`, checked, then promoted to `gold/` with `_current.json` and `dataset_manifest.json`. A failed check or the training gate (`configs/splits.yaml: required_labels`) exits 2: nothing is published and the run fails without a retry |
| `continue_after_*` | (Python) | `stop_after` short-circuit: skip the next node |
| `release_lock` | `precursor-ingest lock release` | Releases the lock (§3) |

**Params** (set when you trigger a run):

| Param | Default | Meaning |
|---|---|---|
| `mode` | `monthly` | `backfill` also runs silver's calibration (lag and QM proposals) |
| `calibrate` | `false` | calibration in a monthly run |
| `stop_after` | `null` | `ingestion`, `bronze` or `silver`: run up to that node (the first backfill, checked node by node) |
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
- **As a teardown** (`release.as_teardown(setups=acquire)`), Airflow leaves it out of the leaves. The leaf becomes the last node's task (`build_silver` then, `build_gold` now; `tests/dags` checks it is the only leaf), so a failure anywhere upstream makes the run `failed`, while the release still runs (`ALL_DONE_SETUP_SUCCESS`). Runs E2 and E4: `failed`, lock released.
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
| 137 | killed (SIGKILL), usually out of memory (`PI_TASK_MEM_LIMIT`) | **failed at once, no retry**: a retry rarely helps |

A task that fails makes the downstream tasks `upstream_failed`. The lock is still released, and the run shows `failed`. A failed container's exit code is kept as the XCom `pipeline_exit_code`, so the failure alert can name it (`dags/utils/alerts.py`; runbook `airflow-local.md` §3.1).

## 5. The metadata database `precursor_meta`

**What it is:** Postgres, one database next to Airflow's own `airflow` database on the same server.
- Created by `docker/airflow-local/init-db.sh` with its own login `pi_meta`.
- The tables are created by the pipeline itself (`MetadataStore.ensure_schema`, from `src/precursorintelligence/common/io/schemas.py`).
- Values are portable SQL types; dates and timestamps are ISO-8601 strings.

**How the tables connect:**

```
catalog                       one row per CMS catalog entry               (entry_id)
 ├─< download_log             one row per download decision               (entry_id, run_id)
 ├─< extracted_files          one row per extracted file                  (entry_id, member_path)
 └─< inventory                one row per zip member                      (entry_id, member_path)
      │                       status = 'selected': one file per (table_name, vintage_month)
      ├─< bronze_files        one row per bronze load or failure          (table_name, vintage_month, zip_sha256)
      │    └── vintage_flags  stale_refresh, revised_after_publication    (vintage_month, flag, table_name)
      └─< silver_partitions   one row per silver build or failure         (source_table, vintage_month, zip_sha256)

standalone:  anomalies (run_id, entry_id)    run_log (run_id, task)
             run_lock (lock_name)            source_datasets (dataset_id)

─<  one-to-many
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

Also: `run_log` marks a step failed when it raises, or when it reports `n_failures` > 0 in its details (download, bronze, silver, gold; fixed for issue #10 in PR #12). Changing a column or type needs a migration: `create_all` never alters an existing table.

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

## 9. Open follow-ups (infra review of the gold wiring, 2026-10-26)

None of these blocks the DAG. They are outside the DAG wiring (gold code, Terraform, operator policy), so they are recorded here for whoever picks them up.

### 9.1 Gold's peak memory on the full archive
- **What:** gold holds several copies of the dataset at once:
  - the silver inputs;
  - the feature frames and the assembled dataset;
  - per-month copies;
  - the staged Parquet bytes;
  - a re-read of every staged file for its checks (`gold/build.py`, `_run` and `_stage`).
- **Why it matters:** on the full archive (spine 2023-01 → 2026-09, ≈ 590k rows; ≈ 755k with a 2022-01 start, spec §14 #10) the review estimates a 4–5 GB peak. The VM has 8 GB, the Airflow services and Postgres use about 2–2.5 GB, and `PI_TASK_MEM_LIMIT` is 6g. The VM itself could run out of memory before the container limit stops the task.
- **Measured locally (runbook §8.2):** gold peaked at **1.95–2.17 GB** for 207k–237k rows (sampled every 2 s, so short spikes can be missed). Scaled linearly to the full archive's ≈ 590k rows, that is **≈ 5 GB**, above the ~4 GB comfort line, so this fix is needed before the first full-archive run on the VM.
- **Fix, if the full-archive peak is above ~4 GB:**
  1. In `_stage`, check each staged file right after writing it, instead of re-reading all of them at the end.
  2. Drop `silver` after the labels are built, and the feature frames after `assemble`.
  3. Otherwise, lower `PI_TASK_MEM_LIMIT` to about 5g.
- **Owner:** gold.

### 9.2 Every daily run rewrites gold, and nothing cleans up
- **What:**
  - Bronze and silver skip unchanged work; gold rebuilds and re-publishes the whole dataset on every successful run.
  - Until `check_release` exists, that means every day.
  - Each run also leaves `gold/_staging/run=<id>/`.
  - The prod bucket has object versioning, so each overwrite of `gold/*` keeps the old version.
  - The only lifecycle rule is for `quarantine/` (`infra/terraform/modules/gcs/main.tf`).
- **Why it matters:** about 0.3–0.4 GB of extra storage a day, ≈ 10 GB more every month. Cheap at first (cents), but it grows for as long as the pipeline runs.
- **Fix:**
  1. Lifecycle rules: delete `gold/_staging/` after 14 days, and keep at most 3 noncurrent versions (or 30 days) under `gold/`.
  2. In gold, skip the publish when the staged sha256s equal those in `gold/_current.json`. A rerun with nothing new then writes nothing, as in bronze and silver.
  3. `check_release` (spec §2.0) stops most of these runs before they start.
- **Owner:** gold + Terraform.

### 9.3 Commit recorded in the gold manifest
- **Fixed in this branch:**
  - **The commit reaches gold:** the image passes it through `PI_GIT_COMMIT`. Before, every container-built manifest had `git_commit: null`.
  - **No silent default:** `PI_GIT_SHA` has no default, and the build **fails** without it (`RUN test -n "$PI_GIT_SHA"`), so no manifest can record `"unknown"`.
  - **Full SHA everywhere:** every documented build passes the **full** SHA (`--build-arg PI_GIT_SHA=$(git rev-parse HEAD)`), so container and laptop runs record the same 40-character format. MLflow logs the full commit too.
- **Remaining:** an image built from a working tree with uncommitted changes still records `git_dirty: null`.
- **Fix:** optionally pass a `PI_GIT_DIRTY` build arg that `git_state` reads.
- **Owner:** pipeline image + gold.

### 9.4 Operator and scheduling notes
- **Out of memory is retried.** **Fixed (ticket 2):** exit 137 is no-retry, and the failure alert names it.
- **No timeout on `build_gold`.** **Fixed (ticket 2):** `execution_timeout` of 2 h (`PI_GOLD_TIMEOUT_MINUTES`); the failure alert fires when it hits.
  - **Still open:** check the 2 h against a measured full-archive gold duration. A timeout is retried like exit 1 (2 × 5 min), so a hang can hold the lock for about 6 h before the alert.
- **`cms_nh_gold` means published, not verified.**
  - **What:** the asset is emitted on every successful gold run, even when nothing changed. Gold stays `verified: false` until the spec §11 verification gate exists.
  - **Fix:** any DAG triggered by this asset (training) must check `gold/_current.json` (`verified`, `dataset_version`) first.
- **Two variables carry the commit:** `PI_GIT_SHA` (ingestion report) and `PI_GIT_COMMIT` (gold). That's fine as a bridge; unify later.
