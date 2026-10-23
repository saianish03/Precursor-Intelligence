# Precursor Intelligence: Step 1 data ingestion pipeline

**Automated, secure download and inventory of the CMS nursing-home archive**

| | |
|---|---|
| Module | `src/precursorintelligence/` (`common/` + `ingestion/`) in the Precursor-Intelligence repo. CLI: `precursor-ingest` (console command from `pyproject.toml`), or `python -m precursorintelligence.ingestion`. |
| Implements | `dataset/Dataset_build_guide_v1.md` §2 (STEP 1), ending at 🛑 Checkpoint 1 |
| Python env | uv project: `uv sync --all-extras` (Python 3.12 from `.python-version`; 3.11 also supported and tested in CI) |
| Status | Implemented; 149 offline tests (+6 Postgres variants, opt-in) + 2 live contract tests passing; first full archive run completed (§15). **It is the `ingestion` node (TaskGroup) of the one-DAG pipeline `cms_nh_pipeline`** (`DATA_PIPELINE_SPEC.md` §15; rechecked against the spec 2026-10-16): shared run id, run lock (D8), per-zip task list, exit codes, landing backlog for `check_release`, per-run manifest (D11). Other storage and metadata decisions of §0 (BigQuery adapter, append-only writes, warehouse port) are not implemented yet. |
| Implementation reference | **`docs/DATA_PIPELINE_SPEC.md` wins** where this document differs (spec §0 precedence). |

---

## Contents
0. [Architecture decisions (2026-10-08)](#0-architecture-decisions-2026-10-08)
1. [Purpose and scope](#1-purpose-and-scope)
2. [Quick start](#2-quick-start)
3. [Source system: the CMS Provider Data Catalog](#3-source-system-the-cms-provider-data-catalog)
4. [What the archive actually contains](#4-what-the-archive-actually-contains)
5. [Architecture](#5-architecture)
6. [Task-by-task specification](#6-task-by-task-specification)
7. [Storage layout](#7-storage-layout)
8. [Metadata model](#8-metadata-model)
9. [Security design](#9-security-design)
10. [Logging and observability](#10-logging-and-observability)
11. [Configuration reference](#11-configuration-reference)
12. [Extending the pipeline (cloud, databases, DAGs, new sources)](#12-extending-the-pipeline)
13. [Testing](#13-testing)
14. [Operations runbook](#14-operations-runbook)
15. [Results of the first full run](#15-results-of-the-first-full-run)
16. [Known limitations and hand-off to Step 2](#16-known-limitations-and-hand-off-to-step-2)

---

## 0. Architecture decisions (2026-10-08)

These decisions replace the earlier "any cloud / any database" framing in this document. Where a later section still describes the current code, it is marked **(current code)**, and the planned change is listed in §0.3.

### 0.1 Decided

| # | Decision | Detail |
|---|---|---|
| D1 | **Two environments only: local and GCP** | Local disk for development and tests; Google Cloud Storage for test and production. The `aws` env and S3 adapter are dropped (not tested or supported). |
| D2 | **Files stay files** | Landing, bronze, silver and gold are files in the object store (CSV/zip in landing, Parquet from bronze on). There is **no database or table adapter for intermediate layers**, and the DuckDB table sink is dropped. |
| D3 | **Final dataset only in a warehouse** | The final facility × month dataset is loaded into **BigQuery** (GCP) or **Postgres** (local stand-in). Gold Parquet in GCS stays the source of truth; the warehouse table is a reloadable copy. No bronze or silver tables are put in BigQuery. |
| D4 | **Pipeline metadata store** | **Postgres locally** during development (Docker; same instance as the local final table, separate `ops` schema). **Postgres `precursor_meta` on the Airflow VM** for test and production, with a read-only BigQuery mirror (dataset `meta`) refreshed after each run. SQLite in memory remains for automated unit tests only. **Amended 2026-10-16** (it was a BigQuery dataset `precursorintelligence_ops`; see `DATA_PIPELINE_SPEC.md` §0.6 and §14 #13). Passages below that assume a BigQuery metadata adapter are superseded. |
| D5 | **Metadata write pattern: append-only** (recommended; see open decision O2) | Every write appends a new row version with `_written_at`; readers use a "latest version per key" view. No in-place `UPDATE`/`MERGE`. The log tables are already append-only. |
| D6 | **BigQuery writes use batch load jobs** | Load jobs are free and the rows can be changed immediately. Streaming inserts cost money and block updates/deletes for up to ~90 minutes, so they are not used. |
| D7 | **Read once, write once per task** | Each task reads the metadata tables it needs into memory once (they are small) and writes its rows in one batch at the end. BigQuery queries take about 1–2 s and are billed at a 10 MB minimum, so per-zip queries would be slow. |
| D8 | **One pipeline run at a time (run lock)** | BigQuery has no practical cross-statement transactions. A lock object in GCS, created with an "only if absent" precondition, stops a second run; locally, a Postgres advisory lock does the same. Code never relies on transactions. **Implemented 2026-10-13 differently:** the lock is a row in the metadata store's `run_lock` table (primary-key insert = atomic acquire), because a lock must span many DAG task processes (a Postgres advisory lock lives only as long as one connection) and the retention-locked landing bucket cannot delete a lock object. Revisit with the BigQuery adapter (BigQuery does not enforce primary keys). See §12.6. |
| D9 | **Bronze keeps every column, as text** | Bronze is a lossless Parquet copy of each selected CSV. Renaming, pruning and typing happen in silver (Step 2b–2e). Lineage columns that we generate are stored with real types. Details in `STEP2_HANDOFF.md`. |
| D10 | **Republished data rebuilds downstream partitions** | If the selected file for a (table, month) changes fingerprint, Step 2 reloads that bronze partition and marks the matching silver/gold partitions for rebuild. Landing keeps both raw zips, so old bronze versions need not be kept. |
| D11 | **Per-run JSON manifest as the audit record** | Each layer writes `metadata/<layer>/run_date=YYYY-MM-DD/<layer>_run_<run_id>_<timestamp>.json` with run ID, git commit, config hash, inputs and outputs (keys and fingerprints), counts and anomalies. The metadata store remains the source of truth the pipeline acts on. **Ingestion: done 2026-10-16** (`report.write_manifest`, §6.7). |

**Environment summary:**

| | Local development | GCP (test and production) |
|---|---|---|
| Files (landing, bronze, silver, gold) | Local disk `./data` | GCS bucket(s), same key layout |
| Pipeline metadata | Postgres (Docker), schema `ops` | BigQuery dataset `precursorintelligence_ops` |
| Final dataset table | Postgres (same container), schema `gold` | BigQuery dataset `precursorintelligence_gold`, partitioned by month `t` |
| Automated tests | SQLite in memory + local disk / `memory://` store | — |
| Credentials | None needed | Service account / workload identity; no keys in config |

The BigQuery datasets and the GCS bucket must be in the **same region**, so loads from GCS work directly with no cross-region charges.

### 0.2 Open decisions

| # | Question | Recommendation |
|---|---|---|
| O1 | Region for the bucket and both BigQuery datasets | `us-central1` (usual lowest-cost default). Still open (spec §14 #7). |
| O2 | Catalog / inventory updates: append-only plus a "latest" view, or `MERGE` statements | Append-only (D5): faster on BigQuery, within quotas, and gives full status history. Still open (spec §14 #8). **Not implemented:** the code upserts and deletes today, which works on Postgres; it is needed with the BigQuery adapter. Until then, metadata column names and types are the downstream contract: any change needs a migration on `precursor_meta` (`create_all` never alters an existing table). |
| O3 | Per-run JSON manifests (D11) | Decided (spec §14 #13); ingestion writes one. |

### 0.3 Changes to the current code that these decisions require

| Area | Change |
|---|---|
| `common/io/metadata_store.py` | Keep the SQLAlchemy adapter (SQLite for tests, Postgres locally). Add a `BigQueryMetadataStore` (google-cloud-bigquery, load jobs, latest-version views). |
| `common/io/schemas.py` | Add `_written_at` to the upserted tables; store dates and timestamps as `DATE` / `TIMESTAMP` instead of ISO strings; partition BigQuery log tables by `_written_at` date. |
| Tasks (`catalog`, `download`, `extract`, `inventory`, `resolve`) | Replace per-entry reads with one bulk read per task, and upserts / in-place status changes with appended row versions. |
| `pipeline.py` | **Done (2026-10-13):** `run_step1` takes the `data_pipeline` run lock (re-entrant for the same run id) and releases it if it took it. |
| `common/io/table_sink.py`, `common/io/registry.py` | **Done (2026-10-16):** `DuckDBTableSink` and the `s3` object-store registration removed (D1, D2); the `duckdb` package stays in the `pipeline` extra for notebooks. Keep `ParquetTableSink` for bronze/silver/gold. Add a separate **warehouse** port with two adapters, used only for the final dataset: BigQuery (load job from gold Parquet in GCS, replace affected month partitions) and Postgres (`COPY`). |
| `configs/` | `dev.yaml`: Postgres via `PI_DATABASE_URL`, with a SQLite fallback for single-process CLI runs only. `prod.yaml`: Postgres on the VM via `PI_DATABASE_URL` until the BigQuery metadata adapter exists (D4); the warehouse later. |
| `ingestion/report.py` and every later layer | Write the per-run JSON manifest (D11). **Ingestion done (2026-10-16).** |
| Packaging | Done: repo-wide `pyproject.toml` + `uv.lock` (extras per image; ingestion uses `pipeline`), strict mypy, console command `precursor-ingest`. |

The remaining Step 1 backlog, and what the spec changed about each item, is in [`STEP1_TODO.md`](STEP1_TODO.md).

### 0.4 Alignment with the team documents

`PI_Tech_Stack.docx` and `Precursor_Intelligence_Data_Pipeline_Plan.docx` (2026-10-08) agree with D3 and D4: in production, BigQuery holds the final tables and the pipeline state, and Postgres runs on the VM for Airflow and MLflow. The differences below were decided on 2026-10-15 (`DATA_PIPELINE_SPEC.md` §0.8, §14). They are recorded here, and the Word documents have not been changed.

| Topic | Team documents | This implementation | Note |
|---|---|---|---|
| Layer names and raw layout | "Bronze" = raw files as published plus all-string Parquet, stored at `bronze/pdc/vintage=YYYY-MM/` | `landing/cms_nh/zips/zip_date=…/` (raw, by zip date) then `bronze/<table>/vintage=YYYY-MM/` (Parquet) | **Decided (spec §14 #6): `landing/` + `bronze/`.** Raw files are kept by zip date because the vintage is known only after reading `Processing Date`, and one month can arrive in two zips. |
| DAG shape | Separate asset-chained DAGs (`pdc_release_sensor`, `ingest_vintage`, `build_silver`, …) | One DAG `cms_nh_pipeline`; Step 1 is its `ingestion` TaskGroup | **Decided (spec §14 #19).** One run id and one lock end to end (§12.6). |
| Duplicate data month | The team `cms-data-domain` skill: "prefer the complete zip" | The later zip wins; archive beats current on the same date | **Decided (spec §14 #2).** |
| Monthly download method | Per-table CSVs from the metastore `downloadURL` | One monthly zip from the archive/current JSON endpoints | Both work. The zip gives all tables of a release at once and uses the same code for history and new months. |
| Backfill source | Annual zips uploaded to GCS manually (the archive page was thought to need JavaScript) | All monthly zips downloaded automatically through the archive JSON endpoint (§3.2) | Manual upload is no longer needed. |
| Stale refresh check | Hash each table ignoring the date column; flag `stale_refresh` | Not implemented in Step 1 | **Confirmed in the data:** in vintage 2025-09, health citations, penalties, survey dates and claims QMs are identical to 2025-07; only Provider Info and MDS QMs changed. Added to the Step 2 checks (`STEP2_HANDOFF.md` §7.6). |
| Python tooling | Python 3.11+, `uv` | `uv` (`uv.lock`), Python 3.12; CI tests 3.11 and 3.12 | Aligned. |

---

## 1. Purpose and scope

Precursor Intelligence trains a facility × month model. Each prediction month *t* must use only what CMS had published
as of *t* (the guide's feature rule, §0.4). That requires **every historical monthly release ("vintage")** of
the CMS nursing-home files, not just today's download. Step 1 produces that raw historical base:

1. **Discover** every monthly archive zip CMS advertises (2019 → today), plus the current release.
2. **Download** the in-scope zips (data months 2021-01 → V) through a hardened HTTP client. Each zip is validated
   in quarantine, then stored immutably with a sha256.
3. **Extract** the 7 in-scope tables (guide §1) from each zip.
4. **Inventory** every file. The vintage month comes from the `Processing Date` column inside the file, never from
   the zip name. Also recorded: row counts, column counts and header hashes.
5. **Resolve** duplicate vintages, list gaps and flag anomalies.
6. **Report:** write `01_inventory.csv` and `01_inventory_summary.md` for **Checkpoint 1**, then stop.

**Out of scope for Step 1:**
- loading to bronze Parquet, harmonising columns and pruning columns (Step 2);
- the ledgers, features and labels (Steps 3–8).

The `TableSink` port that Step 2 will write through is already defined (§5.2).

**Design goals** (from the project review):
- **Modular and extensible:** storage, database and orchestration can each be swapped without changing task code (§5, §12).
- **Secure:** the download path treats the network and the archives as untrusted (§9).
- **Observable:** standard-library `logging`, structured JSON and audit tables (§10).
- **Reproducible and idempotent:** a re-run downloads nothing that is unchanged, and the outputs are deterministic.

---

## 2. Quick start

Run from the repository root. `uv sync --all-extras` installs the package (editable) with the console command `precursor-ingest`. If the package is installed non-editable (e.g. in a Docker image), set `PI_PROJECT_ROOT` to the directory that holds `configs/`.

```bash
uv sync --all-extras
CLI="uv run precursor-ingest"

$CLI catalog                                       # print what CMS advertises (read-only: stores nothing)
$CLI step1                                         # full Step 1 (~3.1 GB download on first run)
$CLI step1 --zip-dates 2021-01-27,2021-12-27       # subset of zips (smoke test)
$CLI step1 --limit 5                               # oldest 5 in-scope zips
$CLI step1 --steps resolve,report                  # rebuild report from stored metadata
$CLI step1 --log-level DEBUG                       # verbose console
$CLI step1 --env prod                              # GCS backends (configs/prod.yaml)
$CLI step1 --env-file my_env.yaml                  # any custom backend combination
$CLI step1 --run-id R --steps extract,inventory --entry-ids <id>   # one DAG task (§12.6)
$CLI entries --pending                             # JSON list of zips still to extract + inventory
$CLI lock status                                   # who holds the run lock (also: acquire / release)

uv run pytest                                      # 149 offline tests (+6 Postgres variants, skipped)
PI_TEST_POSTGRES_URL=postgresql+psycopg://… uv run pytest   # + run-lock tests on a throwaway Postgres
uv run pytest -m network                           # 2 live contract tests against data.cms.gov
```

**Exit codes** (every command; `pipeline.py` `EXIT_*`):

| Code | Meaning | Examples | Retry? |
|---|---|---|---|
| `0` | Success, or nothing new | | |
| `1` | Error | network failure after retries, storage error, exception in a task, run lock held by another run | Yes, may be transient |
| `2` | Validation / drift: the data is bad or changed | quarantined or rejected zip, unsafe archive, CMS catalog contract changed | No: fix the data or config first |
| `3` | Config or usage error | bad env file, missing `${VAR}`, invalid `--run-id`, unknown step, bad flag | No |

If a run has both errors and validation failures, it exits `1`. Failures are listed on stderr.

**In Airflow** (spec §14 #21, §15.4): exit 1 is retried (2 × 5 min); exits 2 and 3 fail at once without retries, through `PipelineDockerOperator`, a thin `DockerOperator` subclass. A stale `--entry-ids` list and a task whose run does not hold the lock are config errors (3).

---

## 3. Source system: the CMS Provider Data Catalog

### 3.1 Documented API: current release only

**Base URL:** `https://data.cms.gov/provider-data/api/1`, an OpenAPI 3 spec served at `/provider-data/api/1?authentication=false`. No authentication is required.

| Endpoint | Purpose | Used by Step 1 |
|---|---|---|
| `GET /metastore/schemas/dataset/items?show-reference-ids=false` | DCAT catalog of all 237 datasets (18 in the nursing-home theme) | — (discovery reference) |
| `GET /metastore/schemas/dataset/items/{id}` | One dataset: `title`, `modified` (data month), `released`, `nextUpdateDate`, `distribution[].downloadURL` (current CSV) | **Yes**: release metadata in `source_datasets` and the report |
| `GET /datastore/query/{id}/0?limit&offset&count&conditions[...]` | Row-level JSON query over the **current** data, e.g. `count=14690` for Provider Info | No (current only; useful for a future live product) |
| `GET /datastore/query/{id}/0/download?format=csv` | CSV export of a query | No |
| `GET /search?theme=…`, `/search/facets` | Search | No |
| `GET /datastore/imports/{distributionId}` | Column schema / import status of a distribution | No |

**Dataset IDs of the 7 in-scope tables:**

| Table | Dataset ID |
|---|---|
| Provider Information | `4pq5-n9py` |
| Health Deficiencies | `r5ix-sfxw` |
| Inspection Dates | `svdt-c123` |
| Penalties | `g6vv-u9sr` |
| MDS Quality Measures | `djen-97ju` |
| Medicare Claims Quality Measures | `ijh5-nb2v` |
| Citation Code Look-up | `tagd-9999` |

The metastore carries **no checksums**: `distribution[…].checksum` is `null`. Integrity must therefore be established by us (§9).

### 3.2 Archive endpoints: undocumented but stable

The page `https://data.cms.gov/provider-data/archived-data/nursing-homes` is a JavaScript single-page app. Its bundle
(`/provider-data/js/index.js`) calls two JSON endpoints that are not in the OpenAPI spec:

| Endpoint (relative to the API base) | Returns |
|---|---|
| `GET /archive/aggregate/theme/nursing-homes/relative` | `{"data":[{name, nid, identifier, type, theme, url, size, date, access_level}, …]}`: 89 monthly `theme` snapshots (2019-01-17 → 2026-08-26) and 8 `annual_theme` bundles |
| `GET /archive/aggregate/current/theme/all/relative` | The "download all" zip of the current release, one per theme. For nursing homes: `…/dataset-archives/current/theme/theme_nursing-homes_current.zip` |

- **URLs** are relative paths under `/provider-data/sites/default/files/dataset-archives/`.
- **Scraping:** none is needed, because the JSON endpoints give the full list.
- **Risk:** the endpoints are undocumented, so the response shape is checked on every call (`CatalogContractError`), and `pytest -m network` runs contract tests.

### 3.3 HTTP behaviour of the file server (verified)
- **`HEAD`** returns `Content-Length`, `Last-Modified`, `Content-Type: application/zip` and `Accept-Ranges: bytes`.
- **Range requests:**
  - **Resume:** an interrupted download continues from the last byte received.
  - **Remote inspection:** a zip's central directory can be read without downloading the zip. This is how the facts in §4 were gathered.
- **CMS rewrites archive files.** The 2021-01 zip has `Last-Modified: 2026-06-24`. The 2026-07-29 zip was **re-uploaded between exploration and the first run**: it was advertised at 622 MB, while the server served 38.8 MB. The catalog's `size` is therefore advisory; the server's `Content-Length` is authoritative, and every stored zip is pinned by sha256 (§6.3).

---

## 4. What the archive actually contains

All facts below were measured by range-reading every zip's directory and the first row of its Provider Info file.

### 4.1 Three layouts

| Layout | Zip dates | File naming | Notes |
|---|---|---|---|
| **L1** | 2019-01 → 2020-07 | `ProviderInfo_Download.csv`, `HealthDeficiencies_Download.csv`, … | **No `Processing Date` column**; no Inspection Dates or Citation Look-up files. Out of scope (before 2021). |
| **L2** | 2020-08 → 2026-06 | `NH_<Table>_<MonYYYY>.csv`, sometimes under one sub-folder (e.g. `nursing_homes_including_rehab_services_12_2021/`) | The main era. |
| **L3** | 2026-07 → | `<datasetId>_<YYYY-MM-DD>_NH_<Table>_<MonYYYY>.csv`, plus `manifest.json` | Some zips contain `__MACOSX/` and `.DS_Store` junk. |

### 4.2 The zip date is not the data month

CMS builds a zip some days or weeks after the data month it carries. For example:
- `nursing-homes_2021-12-27.zip` holds **Nov 2021** data;
- `nursing-homes_2024-11-07.zip` holds **Oct 2024** data.

That is why the teammate datasets in `dataset/test_1_tanoj` and `test_2_tanoj` mislabelled their vintages. **The vintage is always read from `Processing Date`** (§6.5).

- **Duplicate data months:** each October appears twice (2021, 2022, 2023, 2024). Jul 2026 appears twice (the full 07-29 zip plus a partial 08-06 re-release). Aug 2026 appears twice (the 08-26 archive plus the current zip).
- **Missing data months:** CMS never published 2021-12, 2022-12, 2023-12, 2024-12, 2025-01, 2025-08 or 2026-01.
- **The `annual_theme` bundles** are just the monthly zips nested inside one zip. They fill no gaps, so they are excluded by default.

### 4.3 File-level quirks
- **Citation Code Look-up never has a `Processing Date` column** in any era. Its vintage comes from the file-name token, which is configured as expected (`has_processing_date: false`).
- **Column renames across eras:** the results of the first run (§15) list every distinct header set per table. Examples:
  - `Federal Provider Number` became `CMS Certification Number (CCN)`;
  - `Provider State` became `State`;
  - `Rating Cycle 2`/`3` became `Rating Cycle 2/3`;
  - `Fine ID` was added in 2026.

---

## 5. Architecture

### 5.1 Ports and adapters

No task knows where data physically lives. Tasks receive a `Context` that holds four things:

```
                ┌──────────────────────── Context ─────────────────────────┐
  configs/  ──▶ │ settings │ ObjectStore │ MetadataStore │ TableSink │ http │
  env/*.yaml    └────┬───────────┬──────────────┬─────────────┬─────────┬──┘
                     │           │              │             │         │
            sources.yaml   Local disk / GCS   SQLite (tests) Parquet on  SafeHttpClient
                           / memory (tests)   Postgres (dev) the object  (allow-list, TLS,
                                              BigQuery (prod) store      retries, validation)
```

| Port | Interface (Protocol) | Decided adapters (§0) | Status |
|---|---|---|---|
| `ObjectStore` (`common/io/object_store.py`) | `exists, stat, list, put_stream, put_bytes, open_read, move, remove, make_immutable, local_path, uri_for, prune_empty` | `LocalObjectStore` (development); `FsspecObjectStore` with `gs://` via gcsfs (test and production) and `memory://` (unit tests) | Implemented. `s3://` works through fsspec but is not supported (D1). |
| `MetadataStore` (`common/io/metadata_store.py`) | `ensure_schema, append, upsert, read, delete_where` | `SqlAlchemyMetadataStore`: SQLite (unit tests), Postgres (local development); **`BigQueryMetadataStore`** (test and production) | SQLAlchemy adapter implemented; BigQuery adapter planned (§0.3) |
| `TableSink` (`common/io/table_sink.py`) | `write_partition, list_partitions, read` | `ParquetTableSink` only: Hive-partitioned Parquet for bronze, silver and gold | Implemented. **(current code)** also contains `DuckDBTableSink`, to be removed (D2). |
| `Warehouse` (planned) | `load_dataset(version, partitions)` | BigQuery (load job from gold Parquet in GCS); Postgres (`COPY`, local) | Planned, for the **final dataset only** (D3) |
| `CatalogSource` (`ingestion/sources/base.py`) | `fetch() -> SourceSnapshot` | `CmsPdcArchiveSource` | Implemented. Later: PBJ staffing. |

- **Adapter selection** is a registry (`common/io/registry.py`) keyed by `object_store.kind` and `table_sink.kind`. A new backend means writing an adapter class plus adding one decorator line.
- **Credentials** are never in config. The adapters pick up the standard credential chain: GCP Application Default Credentials or workload identity, and `PI_DATABASE_URL` for the local Postgres.

### 5.2 Module map
```
src/precursorintelligence/
  common/
    config.py            pydantic models for configs/ingestion/sources.yaml + configs/<env>.yaml; ${ENV} expansion; security validation
    io/object_store.py   ObjectStore port, LocalObjectStore, FsspecObjectStore, copy_object, sha256_of
    io/metadata_store.py MetadataStore port, SqlAlchemyMetadataStore
    io/schemas.py        bookkeeping tables (SQLAlchemy Core)
    io/table_sink.py     TableSink port, ParquetTableSink (DuckDBTableSink: to be removed, §0)
    io/registry.py       kind -> adapter factories
    obs/logging.py       ContextFilter, RedactFilter, JsonFormatter, log_context, setup_logging
  ingestion/
    context.py           Context dataclass, run_id, build_context() (ingestion-only: holds the HTTP client)
    cli.py               typer CLI (step1, catalog); ships the run log next to the reports; `python -m precursorintelligence.ingestion`
    pipeline.py          run_step1(): chains tasks; task_scope() = log context + timing + run_log rows
    http.py              SafeHttpClient (allow-list, same-host redirects, retries, resumable validated download)
    sources/             CatalogSource port + CMS PDC adapter
    catalog.py           discover, select
    download.py          download (quarantine -> validate -> promote, republish handling)
    zipsafe.py           archive validation (zip-slip, symlinks, bombs, nesting, CRC)
    extract.py           extract in-scope members (store -> store streaming)
    layout.py            table classifier, layout detection, Processing Date parsing, header hashing
    inventory.py         per-member inventory rows, vintages, row counts
    resolve.py           duplicates, gaps, anomalies
    report.py            01_inventory.csv + 01_inventory_summary.md
configs/ingestion/sources.yaml   configs/dev.yaml   configs/prod.yaml   configs/logging.yaml
tests/common/   tests/ingestion/ (cms_fakes.py = fake CMS server; conftest.py blocks real HTTP in unit tests)
```

### 5.3 Task contract (what makes each step swappable and DAG-ready)
- **A task is a plain function of `Context`** (plus a `CatalogEntry` for the fan-out tasks). It returns a small dataclass of counts and keys.
- **Tasks communicate only through the stores.** For example, `extract` finds the zip to read in the latest successful `download_log` row. Any task can be re-run alone, on another machine, or as a separate DAG node.
- **Idempotency** is checked by each task against its own outputs, so a rerun is a no-op when nothing changed:

  | Task | How it detects work already done |
  |---|---|
  | download | size and `Last-Modified` match the last good download |
  | extract | rows exist for the same zip sha256 and their objects exist |
  | inventory | replaces its own rows each time |
- **Fan-out:**
  - `download` runs in a thread pool (`http.max_parallel_downloads`, default 3). Workers return result rows, and only the main thread writes to the MetadataStore, which keeps every backend safe.
  - `extract` and `inventory` iterate over entries. In a DAG they become mapped tasks: `entries --pending` lists the zips that still need them, and `step1 --entry-ids <id>` runs one (§12.6).
- **One run id per DAG run** (`--run-id`), shared by all of its tasks, and one run lock held from the first task to the last (§12.6).

### 5.4 Data flow
```
CMS archive API ──discover──▶ catalog table + raw JSON (landing/cms_nh/catalog/fetched_at=…/)
                    │
                  select (scope: types, min_zip_date, subset)
                    │
                    ▼  per zip (parallel)
        HEAD ─▶ skip? ─▶ GET (Range-resumable) ─▶ quarantine/cms_nh/<run>/… ─▶ validate ─▶ landing/cms_nh/zips/zip_date=…/
                                                                                │fail
                                                                                └─▶ stays in quarantine, status=quarantined
                    ▼  per stored zip
        extract in-scope members ─▶ landing/cms_nh/extracted/zip_date=…/<zip_stem>/<file>
                    ▼  per stored zip
        inventory: one row per member (+ Processing Date, row_count, header_hash)
                    ▼
        resolve (duplicates/gaps/anomalies) ─▶ report (CSV + summary MD) ─▶ 🛑 Checkpoint 1
```

---

## 6. Task-by-task specification

### 6.1 `discover` (`ingestion/catalog.py`)
| | |
|---|---|
| **Input** | Config only |
| **Calls** | archive endpoint; current endpoint; metastore item for each of the 7 dataset IDs |
| **Writes** | ObjectStore `landing/cms_nh/catalog/fetched_at=<UTC stamp>/{archive,current,metastore}.json`: the exact responses, kept for provenance. MetadataStore `catalog`, upserted, with `first_seen_at` preserved, `last_seen_at` updated and `active=false` for entries CMS stopped advertising. MetadataStore `source_datasets`. |
| **Logs** | Counts of new, removed and size-changed entries. A size change or removal is a WARNING, because it signals a possible re-publication. |
| **Fails when** | The response shape changed (`CatalogContractError`), or the response is not JSON. |

`entry_id = "<type>:<zip_date>:<zip_name>"`. It is unique even for the rolling `theme_nursing-homes_current.zip`, because the current zip gets a new date every month.

### 6.2 `select` (`ingestion/catalog.py`)
- **Filter:** active catalog entries where `type ∈ scope.include_types` (default `theme, current`) and `zip_date ≥ scope.min_zip_date` (default 2021-01-01, the first zip holding a 2021-01 vintage).
- **Optional narrowing:** `--zip-dates` and `--limit`.
- **Order:** oldest first; on the same date, archive before current.

### 6.3 `download` (`ingestion/download.py`), one per zip
1. **Allow-list check** of the catalog URL (§9.2). If it fails: status `rejected`, and no request is made.
2. **`HEAD`.** The server's `Content-Length` becomes the expected size. A mismatch with the catalog size is logged at WARNING and recorded as `catalog_size_mismatch`.
3. **Skip** when the landing object exists **and** the last successful `download_log` row has the same size and `Last-Modified`. Status `skipped`.
4. **Adopt** when the landing object exists but has no log row (for example, copied in by hand). It is fully validated first, then hashed. Status `adopted`.
5. **Stream** `GET` into `quarantine/cms_nh/<run_id>/<zip_date>/<zip_name>`, hashing (sha256) while writing. The transfer resumes with `Range` on connection errors and is capped at `max_download_bytes`.
6. **Validate** the archive (§9.5): the central-directory limits, then a CRC-32 check of every member. If it fails: status `quarantined`, and the bytes stay in quarantine for inspection.
7. **Promote** to `landing/cms_nh/zips/zip_date=<date>/<zip_name>` with an atomic move, then call `make_immutable`.
   - **Same content already there:** quarantine is deleted, status `unchanged`.
   - **Different content already there:** stored side by side as `<stem>__<sha8>.zip`, status `republished`, WARNING. Nothing is ever overwritten.
8. **Log:** one `download_log` row for every outcome. After the fan-out, empty quarantine folders are pruned, so only quarantined files remain.

**Statuses:**

| Group | Statuses |
|---|---|
| Success | `downloaded`, `skipped`, `unchanged`, `republished`, `adopted` |
| Failure | `failed` (network or storage, after retries), `quarantined`, `rejected` |

### 6.4 `extract` (`ingestion/extract.py`), one per stored zip
- **Validation:** the archive is re-validated with `inspect_archive` before any member is read.
- **Members extracted:** in-scope tables only (`extract.only_in_scope: true`). Junk is skipped, as are extensions outside the allow-list.
- **Streaming:** data moves from store to store (`open_read` → `zipfile` → `put_stream`) in 8 MiB chunks, so it works the same way on GCS.
- **Output key:** `landing/cms_nh/extracted/zip_date=<date>/<zip_stem>/<file_name>`. Sub-folders are flattened; the original member path is kept in `extracted_files.member_path`. A name collision gets a `__<crc32>` suffix.
- **Immutability:** extracted objects are made immutable too.

### 6.5 `inventory` (`ingestion/inventory.py`), one per stored zip
One row per member, junk included and flagged `table_name='junk'`, because junk is what explains bloated zips. Fields recorded for every member:
- `table_name`: classified by the configured regexes, otherwise `other` or `junk`;
- `layout` (L1/L2/L3);
- `file_size_bytes`, `compressed_size_bytes`, `crc32`;
- `in_manifest`, for L3 zips.

Extracted CSVs additionally get `header_json`, `n_columns`, `header_hash` (sha256 of the normalised header list: BOM stripped, whitespace collapsed; first 16 hex characters), the full `row_count`, and `processing_date`.

**Vintage rule** (guide §2.5):
```
vintage_month = YYYY-MM(Processing Date)            vintage_source = processing_date
  else  MonYYYY token in file name                  vintage_source = filename_token   (WARNING unless expected)
  else  zip date                                    vintage_source = zip_date         (WARNING)
```

`Processing Date` is read from the **first data row**, in the first header that contains "processing" in any case. Only the first ≤4 MiB of the file is decoded for this.

**Row counts:**
- On local storage: Polars `scan_csv(..., infer_schema=False)` → `pl.len()`.
- On object stores: a streaming `csv.reader`.
- Both count CSV records, so quoted newlines are handled correctly.

### 6.6 `resolve` (`ingestion/resolve.py`)
- **Duplicates:** files are grouped by `(table, vintage_month)`.
  - The **latest `zip_date`** wins, following the guide. On the same date the archive copy beats the rolling current zip, because archive copies are stable.
  - Then the higher `_N` file version (2026-10-08 release on, e.g. `..._Sep2026_3.csv`) wins, compared as a number, so `_10` beats `_9`.
  - Losers get status `duplicate_vintage`, with a detail naming the winning zip and file and saying whether their content is identical (same CRC and size).
- **Out of scope:** files with a vintage before `scope.vintage_start` get status `out_of_scope`.
- **V and gaps:** V is the latest selected Provider Info vintage. Gaps are listed per table for `vintage_start → V`.
- **Anomalies** are written to the `anomalies` table:

| kind | severity | rule |
|---|---|---|
| `no_tables_recognised` | **error** | the zip has no in-scope table and has CSVs that match neither a table pattern nor `out_of_scope_patterns` (e.g. CMS changed its file names); `step1` exits 2. A zip with only known out-of-scope files (FireSafety, Ownership, …) is not an error |
| `partial_zip` | warning | the zip lacks some in-scope tables |
| `size_outlier` | warning | real stored size differs by >1.5× from the median of up to 3 neighbouring monthly zips on each side |
| `member_count` | info | the zip has >25% more or fewer files than the median of its layout |
| `junk_bloat` / `junk_members` | warning / info | macOS metadata present; `junk_bloat` when it is >50% of the zip's bytes |
| `nonstandard_path` | info | a monthly zip served outside `dataset-archives/theme/` |
| `vintage_fallback` | warning | no Processing Date where one was expected |
| `expected_vintage_from_name` | info | a table configured `has_processing_date: false` |
| `duplicate_vintage` | info | which file was kept and which dropped |
| `missing_vintage` | warning (info on subset runs) | gaps |
| `republished`, `catalog_size_mismatch` | warning / info | from `download_log` |
| `download_failed` / `download_quarantined` / `download_rejected` | error | the latest outcome per zip |

### 6.7 `report` (`ingestion/report.py`)
**`01_inventory.csv`** has one row per (zip, file), junk excluded.
- **Columns required by the guide:** `zip_name, zip_sha256, file_name, table, vintage_month, processing_date, row_count, n_columns, header_hash, file_size_bytes`.
- **Extra provenance columns:** `zip_date, zip_type, layout, member_path, vintage_source, status, status_detail, extracted_key`.
- **Formula escaping:** cells starting with `= + - @` are prefixed with `'` so the file can't run a formula when opened in a spreadsheet (§9.6).

**`01_inventory_summary.md`** contains:
- a run header, plus a banner when only a subset of zips was processed;
- the current CMS release table;
- a **coverage matrix** (data month × table);
- gaps per table;
- **zip date vs data month**;
- the duplicates resolved;
- the **earliest survey date** in the first Health Deficiencies vintage;
- the **distinct header sets per table**, with vintage ranges and the columns added or removed between sets;
- the anomalies, grouped and explained.

Both files are written to `reports/step1/run=<run_id>/` and to `reports/step1/latest/`. `latest/` is replaced by every run; `run=<run_id>/` is rewritten only when a task of the same run is retried (Airflow retries keep the run id). Each CLI process also uploads its own JSON log as `reports/step1/run=<run_id>/logs/<command>_<timestamp>-<hex>.jsonl`, so parallel tasks of one run never overwrite each other's log.

**Run manifest (D11).** `report` also writes `metadata/ingestion/run_date=<UTC date>/ingestion_run_<run_id>_<ts>-<hex>.json`: run id, env, `code_version` (`PI_GIT_SHA`, set by the image build), the sha256 of the parsed `sources.yaml`, V, counts (this run's downloads by status, selected files, duplicates, anomalies by kind), gaps, and every selected file with `zip_sha256`, `member_path`, `extracted_key`, `header_hash` and `row_count`. A retry adds a new manifest; none is overwritten.

---

## 7. Storage layout

The logical keys are identical on both backends. Only the root changes:
- local development: `data`;
- GCP test and production: `gs://<bucket>`. Optionally use one bucket per layer (`…-landing`, `…-bronze`, `…-silver`, `…-gold`) for tighter permissions; the key layout stays the same.

**Why landing is organised by zip date, not by month.** The data month (vintage) is known only after reading `Processing Date` inside the files, and one month can arrive in two zips (§4.2). The zip → month mapping lives in the `inventory` table. Bronze is the first layer organised by month (`bronze/<table>/vintage=YYYY-MM/`).

**Re-runs and republished data:**

| Situation | What happens |
|---|---|
| Zip unchanged | Skipped |
| Same zip re-uploaded with identical bytes | `unchanged` |
| Same zip re-uploaded with different bytes | Stored next to the original as `<stem>__<sha8>.zip` (`republished`). Downstream reads the newest copy; the original is kept for audit. |
| New zip for a month already loaded | Duplicate resolution picks the later zip |

In the last two cases the selected file for a (table, month) gets a new fingerprint, so Step 2 must reload that bronze partition and mark downstream partitions for rebuild (§0, D10).

```
<root>/
  landing/cms_nh/catalog/fetched_at=<YYYYMMDDTHHMMSSZ>/archive.json | current.json | metastore.json
  landing/cms_nh/zips/zip_date=<YYYY-MM-DD>/<zip_name>                     immutable, sha256 in download_log
  landing/cms_nh/zips/zip_date=<YYYY-MM-DD>/<stem>__<sha8>.zip            only if CMS re-published
  landing/cms_nh/extracted/zip_date=<YYYY-MM-DD>/<zip_stem>/<file>.csv     7 in-scope tables
  quarantine/cms_nh/<run_id>/<zip_date>/<zip_name>                         only failed validations remain
  reports/step1/run=<run_id>/01_inventory.csv | 01_inventory_summary.md | logs/<command>_<ts>-<hex>.jsonl
  reports/step1/latest/…                                                   copy of the latest run
  ops/metadata.db                                                          SQLite fallback (dev, single process); Postgres via PI_DATABASE_URL otherwise (§0 D4)
  metadata/ingestion/run_date=<YYYY-MM-DD>/ingestion_run_<run_id>_<ts>-<hex>.json   per-run landing manifest (D11)
  bronze/<table>/vintage=<YYYY-MM>/part-0.parquet                          Step 2 (TableSink), not yet written
```

**Key validation:** every key must be relative, contain no `..`, no backslash and no empty segment. The local adapter also rejects keys that resolve outside the root.

**Deletes** are only allowed under `quarantine/` and `tmp/`, so landing data cannot be deleted by pipeline code.

---

## 8. Metadata model

**(current code)** All tables are SQLAlchemy Core (`common/io/schemas.py`), portable between SQLite and Postgres. Dates are stored as ISO strings. `catalog`, `extracted_files` and `inventory` are upserted, meaning rows are replaced by key.

**Decided target (§0, D4–D8; not yet implemented):**

| Aspect | Target |
|---|---|
| Backends | Postgres (local development); BigQuery dataset `precursorintelligence_ops` (test and production); SQLite (unit tests) |
| Write pattern | **Append-only.** Every write adds a row version with `_written_at`. A view per table returns the latest version per key: BigQuery `QUALIFY ROW_NUMBER() OVER (PARTITION BY <key> ORDER BY _written_at DESC) = 1`; Postgres `DISTINCT ON (<key>) … ORDER BY <key>, _written_at DESC`. Status changes (e.g. `selected` → `duplicate_vintage`) become new versions, which gives a full history. |
| BigQuery write method | Batch **load jobs** (free, immediately consistent). No streaming inserts. |
| Access pattern | Each task reads its tables once and writes once, in batch |
| Concurrency | One run at a time. **Implemented** as a `run_lock` row (lock name `data_pipeline`) in the metadata store, so it works across processes on SQLite and Postgres (§12.6). No reliance on transactions. |
| Types | `DATE` / `TIMESTAMP` instead of ISO strings. BigQuery log tables partitioned by `_written_at` date. |

| Table | Key | One row per | Main columns |
|---|---|---|---|
| `catalog` | `entry_id` | advertised archive entry | `name, nid, type, theme, zip_date, zip_name, url, size_bytes, source_endpoint, first_seen_at, last_seen_at, active` |
| `source_datasets` | `dataset_id` | in-scope dataset | `table_name, title, modified, released, next_update, current_download_url, fetched_at` |
| `download_log` | `id` (append-only) | download decision | `run_id, entry_id, zip_name, zip_date, type, url, catalog_size, expected_size, actual_size, sha256, http_last_modified, http_content_type, object_key, object_uri, status, detail, logged_at` |
| `extracted_files` | `entry_id, member_path` | extracted member | `zip_sha256, table_name, object_key, size_bytes, crc32, run_id, extracted_at` |
| `inventory` | `entry_id, member_path` | zip member | see §6.5, plus `status` (`selected` / `duplicate_vintage` / `out_of_scope` / `other` / `junk` / `unresolved`), `status_detail` |
| `anomalies` | `id` | anomaly per run | `run_id, entry_id, kind, severity, detail, created_at` |
| `run_log` | `id` | task invocation | `run_id, task, entry_id, status, started_at, finished_at, duration_s, detail (JSON)` |
| `run_lock` | `lock_name` | held lock (0 or 1 row per name) | `run_id, holder (host:pid), acquired_at`. Inserting the row is the atomic acquire (D8, §12.6). |

Useful queries:
```sql
-- what is stored and where
select zip_date, zip_name, actual_size, sha256, object_uri from download_log
where status in ('downloaded','republished','adopted') order by zip_date;
-- the file Step 2 should load for each (table, vintage)
select table_name, vintage_month, extracted_key, row_count, header_hash from inventory
where status = 'selected' order by table_name, vintage_month;
-- slowest tasks of the last run
select task, entry_id, duration_s from run_log where run_id = :run order by cast(duration_s as real) desc limit 10;
```

---

## 9. Security design

**Threat model:**
- **The data is public**, so confidentiality of the source isn't a concern.
- **Integrity:** corrupted, truncated, tampered or malicious files.
- **Safe handling of untrusted archives:** zip-slip, zip bombs, symlinks.
- **Our infrastructure:** the bucket, the database and our credentials.
- **Supply chain:** the Python dependencies.

### 9.1 Transport
- **HTTPS only:** `source.base_url` must start with `https://`. This is enforced at config load.
- **TLS verification is always on** (certifi CA bundle). `security.verify_tls: false` is a config **validation error**, and `SafeHttpClient` also refuses it, as defence in depth.
- **Timeouts:** every request has connect and read timeouts.
- **`Accept-Encoding: identity`:** the byte counts match the stored file exactly.
- **Redirects:** followed manually (`allow_redirects=False`), at most 3, and only to the same https host. A cross-host redirect raises `UnsafeUrlError`.

### 9.2 URL allow-list (tampered catalog, SSRF)
Catalog `url` values are untrusted. A download URL is built only when the path meets all of these conditions:
- it is **relative**: no scheme or host, and no leading `//`;
- it has no query string or fragment;
- it has no `.`/`..` segments, no backslashes and no `%`-encoding;
- it **fully matches** the anchored regex `^/provider-data/sites/default/files/dataset-archives/[A-Za-z0-9_.\-/]+\.zip$`.

It is then joined to the configured base URL, and the result is re-checked for scheme and host. Rejections are logged at ERROR and recorded with status `rejected`; no request is sent. The live contract test asserts that every one of the 89 advertised monthly URLs passes.

### 9.3 Response validation
Before any byte is trusted:
- **Status:** `200`, or `206` with a matching `Content-Range` when resuming.
- **`Content-Type`:** must be in `security.allowed_content_types` (`application/zip`).
- **`Content-Length`:** must equal the expected size (minus the offset when resuming).
- **Signature:** the first 4 bytes must be the zip signature `PK\x03\x04`.
- **Totals:** the total bytes must equal the expected size exactly, and may never exceed `max_download_bytes` (2 GiB).
- **Truncated or interrupted bodies** resume with `Range`, up to `max_retries` times.
- **Retries:** 429/5xx and connection errors are retried with exponential backoff and jitter (tenacity). `Retry-After` is honoured.

### 9.4 Quarantine, then promote
1. Bytes are written to `quarantine/cms_nh/<run_id>/…`.
2. They are promoted to `landing/` only after the size check, the archive-limit check and the full CRC check all pass.
3. Promotion is an atomic move:
   - local: `os.replace`;
   - GCS: the object only appears when its upload completes.
4. Failed files stay in quarantine and the run exits with code 2 (validation). Downstream tasks only read objects named by a successful `download_log` row, so they never see quarantined bytes.

### 9.5 Safe archive handling (`ingestion/zipsafe.py`)
- **Zip-slip:** absolute paths, Windows drive letters, `..` segments and NUL bytes are rejected. Extraction also flattens to the base name and validates the destination key.
- **Symlink members** are rejected.
- **Zip bombs:** limits on the number of members (500), the total uncompressed size (5 GiB) and the per-member compression ratio (100×, for members over 10 MiB).
- **Nested zips** are rejected (`max_nested_depth: 0`).
- **Extension allow-list:** `.csv .json .pdf .xlsx .txt`. Junk (`__MACOSX/`, `.DS_Store`, `._*`) is skipped.
- **CRC-32** of every member is verified before promotion (`zipfile.testzip`).
- **Memory:** everything streams in fixed-size chunks; nothing is loaded whole.

### 9.6 Content handling
- **Data is never executed:** CSVs are parsed as text only, and `.pdf`/`.xlsx` dictionaries are stored but never opened.
- **Formula injection:** report cells are escaped, so a malicious cell can't run a formula when someone opens the CSV in Excel or Sheets.

### 9.7 Change and tamper detection
- **sha256 at first sight:** CMS publishes no checksums, so each zip's sha256 is recorded the first time it is seen.
- **Re-publications** are detected via `Last-Modified` and size on `HEAD`. They are stored side by side and never overwrite the original (`republished`, WARNING).
- **Catalog diffs** (new, removed or size-changed entries) are logged on every run. The 622 MB → 38.8 MB change of the 2026-07-29 zip was caught exactly this way.

### 9.8 Rate limiting and polite use
- At most `max_parallel_downloads` (3) concurrent downloads.
- Backoff with jitter, and `Retry-After` is honoured.
- A descriptive `User-Agent`. No contact email is sent unless you add one to `http.user_agent`.

### 9.9 Cloud storage and credentials (GCP)

**Bucket (decided 2026-10-16).** The spec keeps every layer under **one object-store root** (`DATA_PIPELINE_SPEC.md` §0.6). That root also holds data that must be rewritten (bronze reloads under D10, gold promotion from staging, `reports/…/latest`) or deleted (quarantine). So it gets **no bucket retention policy**. Instead:
- **Object versioning on**, so an accidental overwrite or delete can be recovered.
- **Landing immutability comes from the code:** writes under `landing/` never overwrite, and republished zips are stored side by side. `prod.yaml` records `immutable: object_versioning`; `FsspecObjectStore.make_immutable` only logs the intent.
- **Lifecycle rule:** delete `quarantine/` objects after 30 days.
- **Later, if needed:** per-object retention on `landing/` objects only (GCS Object Retention Lock), or a separate landing root. Either needs adapter code.

All buckets get uniform bucket-level access, public access prevention, and Google-managed or CMEK encryption.

**Identities (least privilege), one service account per job:**

| Service account | Storage | BigQuery |
|---|---|---|
| Ingestion (Step 1) | `roles/storage.objectCreator` + `objectViewer` on landing; `objectAdmin` on scratch; no delete on landing; no access to silver or gold | `roles/bigquery.dataEditor` on `precursorintelligence_ops`, `roles/bigquery.jobUser` on the project |
| Transform (Steps 2+) | Read on landing; write on bronze, silver and gold | `dataEditor` on `precursorintelligence_ops` and `precursorintelligence_gold` |
| Analysts, API, dashboard | — | `roles/bigquery.dataViewer` on `precursorintelligence_gold` only |

**Credentials and secrets:**
- **No keys anywhere.** Credentials come from workload identity (Cloud Run / VM service account) or `gcloud auth application-default login` for a developer. No service-account key files, no credentials in config, code or logs.
- **Local Postgres:** the URL comes from `PI_DATABASE_URL` (environment or Secret Manager). The logged URL hides the password (`render_as_string(hide_password=True)`), and the log redaction filter masks credentials in URLs.
- **Infrastructure as code:** create the buckets, datasets and IAM bindings with Terraform (team tech-stack document), not by hand.

### 9.10 Supply chain
- Run from the pinned `requirements-lock.txt` (project root) or `environment.yml`.
- `pip-audit` is recommended in CI. YAML is parsed with `yaml.safe_load` only.

---

## 10. Logging and observability

Logging uses only the standard-library **`logging`** module, configured by `logging.config.dictConfig` from `configs/logging.yaml`.

| Element | Detail |
|---|---|
| Loggers | `logging.getLogger(__name__)` in every module, e.g. `precursorintelligence.ingestion.download`. The `precursorintelligence` logger has its own handlers; `propagate: false`. |
| Console handler | INFO, text: `2026-10-06 19:33:35 INFO precursorintelligence.ingestion.download [download nursing-homes_2021-12-27.zip] stored …`. With `console_format: json` (gcp env) it emits JSON to stdout/stderr for Cloud Logging. |
| File handler | DEBUG, JSON lines, `RotatingFileHandler` (50 MB × 5), one file per CLI process: `logs/<command>_<ts>-<hex>.jsonl` (every line still carries `run_id`). `step1` copies it to `reports/step1/run=<run_id>/logs/` at the end. |
| Context | `ContextFilter` adds `run_id, env, task, entry_id, zip_name` to every record from a `contextvars` variable. `log_context(...)` nests, and worker threads run in a copied context, so parallel download logs stay attributable. |
| Redaction | `RedactFilter` masks URL credentials, signed-URL parameters (`X-Amz-Signature`, `X-Goog-Signature`, `token`, …), `Authorization` headers (including `Bearer <token>`), and `*_key`/`*token`/`password`/`secret` values, in both the message and string `extra` fields. |
| JSON format | `{"ts", "level", "logger", "msg", "run_id", "env", "task", "entry_id", "zip_name", …extra, "exc"}` |
| Levels | **INFO:** task start/end, durations, counts. **WARNING:** retries, resumes, size mismatches, re-publications, unexpected vintage fallbacks. **ERROR:** rejections, quarantines, failures (with traceback). **DEBUG:** redirects, expected fallbacks. |
| Audit tables | `run_log` (every task invocation, status, duration, details), `download_log` (every download decision), `anomalies`. |
| Overrides | `--log-level DEBUG` sets both the console level and the `precursorintelligence` logger level. |

`print()` is not used in pipeline code; the CLI uses `typer.echo` only for its final summary.

---

## 11. Configuration reference

### 11.1 `configs/ingestion/sources.yaml`

| Key | Default | Meaning |
|---|---|---|
| `source.base_url` | `https://data.cms.gov` | Must be https |
| `source.api_prefix` | `/provider-data/api/1` | |
| `source.theme_slug` | `nursing-homes` | Archive theme |
| `source.endpoints.archive` / `.current` / `.metastore_item` | see file | Endpoint templates (`{theme}`, `{dataset_id}`) |
| `source.allowed_download_path_regex` | anchored regex | The download allow-list (§9.2) |
| `scope.include_types` | `[theme, current]` | Add `annual_theme` only with `max_nested_depth: 1` |
| `scope.min_zip_date` / `max_zip_date` | `2021-01-01` / `null` | Zip-date window |
| `scope.vintage_start` | `2021-01` | First data month in scope |
| `tables.<name>.patterns` | regexes | File-name patterns covering L1, L2 and L3 |
| `tables.<name>.dataset_id` | | Used for metastore release metadata |
| `tables.<name>.has_processing_date` | `true` | `false` for Citation Look-up |
| `extract.only_in_scope` | `true` | `false` extracts every allowed file |
| `http.*` | | `user_agent`, timeouts, `max_retries`, backoff, `max_parallel_downloads` (1–8), `chunk_bytes`, `max_redirects` |
| `security.verify_tls` | `true` | Cannot be `false` |
| `security.max_download_bytes` | 2 GiB | Per-zip cap |
| `security.allowed_content_types` | `application/zip`, `application/x-zip-compressed` | |
| `security.zip.*` | | `max_members`, `max_total_uncompressed_bytes`, `max_compression_ratio`, `ratio_check_min_bytes`, `allowed_extensions`, `max_nested_depth` |

### 11.2 `configs/<env>.yaml`

| Key | Local (development) | GCP (test and production) |
|---|---|---|
| `env` | `dev` | `prod` |
| `object_store.kind` | `local` (`memory` in unit tests) | `gcs` |
| `object_store.root` | `./data` | `${PI_BUCKET_URI}` (e.g. `gs://precursorintelligence-landing`) |
| `object_store.immutable` | `chmod` | `object_versioning` (informational; no retention policy, §9.9) |
| `object_store.storage_options` | — | passed to gcsfs (e.g. `{project: my-proj}`) |
| `metadata_store` | `url: ${PI_DATABASE_URL:-sqlite:///./data/ops/metadata.db}`: Postgres for Airflow and parallel tasks; the SQLite fallback for single-process CLI runs only | `url: ${PI_DATABASE_URL}` (Postgres on the VM, `precursor_meta`) until the BigQuery adapter (`precursorintelligence_ops`) exists (D4) |
| `run_lock.stale_after_hours` | `12` | `12`: a lock older than this was left by a killed run and is taken over with a WARNING (`null` = never) |
| `table_sink.kind` / `.prefix` | `parquet` / `bronze` | `parquet` / `bronze` |
| `warehouse` (planned, final dataset only) | `kind: postgres`, `url: ${PI_DATABASE_URL}`, `schema: gold` | `kind: bigquery`, `dataset: precursorintelligence_gold`, `location` |
| `logging.config` / `.log_dir` / `.console_format` | `configs/logging.yaml` / `./logs` / `text` | same / `/tmp/…` / `json` (Cloud Logging) |

`${VAR}` and `${VAR:-default}` are expanded from the environment. A missing required variable is a config error (exit 3).

---

## 12. Extending the pipeline

### 12.1 Run on GCP (GCS + BigQuery)
1. **Create the resources with Terraform:** the buckets, BigQuery datasets `precursorintelligence_ops` and `precursorintelligence_gold` in the **same region** (open decision O1, recommended `us-central1`), and the service accounts and IAM bindings of §9.9.
2. **Authenticate:** workload identity (VM or Cloud Run service account), or `gcloud auth application-default login` for a developer. No key files.
3. **Configure:** `export PI_BUCKET_URI=gs://precursorintelligence-landing` and set the BigQuery project and dataset in `configs/prod.yaml`.
4. **Run:** `precursor-ingest step1 --env prod`. The keys and folder structure are exactly those of §7.
5. **Migrate files you already downloaded locally,** instead of re-downloading them: use `copy_object(local, key, gcs, key)` (`common/io/object_store.py`) or `gcloud storage rsync data/landing gs://…/landing`. On the first GCP run, each zip found in the bucket without a metadata row is **adopted after full validation** (§6.3 step 4).

### 12.2 Local development with Postgres
1. Start Postgres in Docker, for example the team's `docker compose` stack (`postgres:16`). Create a database with schemas `ops` (pipeline metadata) and `gold` (final dataset).
2. Run `export PI_DATABASE_URL=postgresql+psycopg://user:pass@localhost:5432/precursorintelligence`. The SQLAlchemy metadata adapter works unchanged.
3. Unit tests keep using in-memory SQLite and need no database. Set `PI_TEST_POSTGRES_URL` to a **throwaway** database to also run the run-lock tests on Postgres (they drop and recreate the metadata tables).
4. **Use Postgres, not SQLite, whenever tasks run in separate containers** (Airflow). Parallel processes on one SQLite file only work by waiting on each other (30 s busy timeout). `ensure_schema` takes a Postgres advisory lock, so parallel tasks can start against an empty database: without it, 7 of 8 simultaneous processes failed with `duplicate key … pg_type_typname_nsp_index` (measured 2026-10-13, Postgres 17).

### 12.3 The final dataset table (warehouse port, planned)
- **What:** only the final facility × month dataset is loaded into a warehouse (D3). Bronze and silver stay as Parquet.
- **BigQuery:** a load job straight from the gold Parquet files in GCS (free). The table is partitioned by month `t`; each run replaces the affected month partitions. A `dataset_version` column (or one table per frozen version) ties it to `dataset_manifest.json`.
- **Postgres (local):** the same data via `COPY`, into the `gold` schema. Partitioning is not needed at about 900k rows.
- **Type discipline:** use only DATE, TIMESTAMP, FLOAT64/INT64, BOOL and STRING, so the schema is identical in both. Keep pipeline logic in Python/Polars, not SQL, so dialect differences never matter.

### 12.4 Add a new backend
The port/adapter structure still allows it (write a class satisfying the Protocol, register it, add an env file, run the parametrised conformance tests in `tests/common/test_stores.py`). Per D1, though, only local disk and GCS are supported and tested.

### 12.5 Add a new source or table
- **New table from the same archive:** add an entry under `tables:` with patterns for each layout.
- **New source** (e.g. the PBJ daily staffing API): implement `CatalogSource.fetch()` returning `CatalogEntry` items, add its download path prefix to a source-specific allow-list, and pass the source to `discover(ctx, source=...)`.

### 12.6 Run in Airflow: the `ingestion` node of `cms_nh_pipeline`

> **Superseded in part (2026-10-15/16).** Step 1 is not a DAG of its own. It is the `ingestion` TaskGroup of the one DAG `cms_nh_pipeline` (`DATA_PIPELINE_SPEC.md` §0.7, §15; decision §14 #19). The commands below are unchanged. What changed: **`release_lock` is the last task of the whole DAG** (after gold), so bronze, silver and gold run under the same lock; `check_release` (spec §2.0) gates the DAG instead of `@daily` alone; and exits 2/3 are not retried (`PipelineDockerOperator`, spec §15.5). The `ingest_vintage` DAG further down is kept only as the parse-checked reference for `docker_kwargs` and `mapped_commands`, which spec §15.5 reuses.

The CLI is the task interface: every task runs the `precursor-pipeline` image with one `precursor-ingest` command through `DockerOperator`. **All tasks of one DAG run pass the same `--run-id {{ run_id }}`**, so they share one run lock, one `run_log` run id and one `reports/step1/run=<id>/` folder.

**Task contract (the ingestion TaskGroup):**

| # | Task | Command | Notes |
|---|---|---|---|
| 1 | `acquire_lock` | `lock acquire --run-id R` | Takes the `data_pipeline` lock for the whole DAG run. Exit 1 (retried) while another run holds it; a lock older than `run_lock.stale_after_hours` (12) is taken over with a WARNING. |
| 2 | `discover_download` | `step1 --run-id R --steps discover,download` | One task: the downloads already run 3 in parallel inside it, and unchanged zips cost one `HEAD` each. |
| 3 | `pending_entries` | `entries --run-id R --pending` | Read-only. **Last stdout line = JSON list of entry ids** whose stored zip still needs extract + inventory. Empty on a quiet day. |
| 4 | `extract_inventory` (mapped, one per id) | `step1 --run-id R --steps extract,inventory --entry-ids <id>` | Entry ids are exact catalog keys (`theme:2026-08-26:nursing-homes_2026-08-26.zip`); an unknown id exits 3. If extract fails, inventory is skipped and the zip stays pending. |
| 5 | `resolve_report` | `step1 --run-id R --steps resolve,report` | Duplicates, gaps, anomalies, Checkpoint 1 report, run manifest. `trigger_rule="none_failed"`, so it also runs when nothing was pending. Emits the `cms_nh_landing` asset; the next node is `bronze` in the same DAG. |
| — | `release_lock` (end of the DAG, after gold) | `lock release --run-id R` | `trigger_rule="all_done"`, `retries=0`: always runs, also after failures and short-circuits. Releasing a free lock is not an error. |

**Rules the code enforces:**
- **Run id:** 1–64 characters from `A-Z a-z 0-9 _ . : + -`; Airflow ids such as `scheduled__2026-10-13T00:00:00+00:00` are valid. Anything else exits 3.
- **Run lock (D8):** a row in the metadata store's `run_lock` table.
  - `step1` **with** `--run-id` (a DAG task) requires that its run already holds the lock and never takes or releases it. If not, it exits 3 (a wiring error). So no Step 1 task can free the lock before gold has run.
  - `step1` **without** `--run-id` (a manual run) takes the lock and releases it at the end; while a DAG run holds it, it exits 1.
  - `catalog` is **read-only**: it prints what CMS advertises and stores nothing, so it never interferes with a running DAG and never hides a change from `check_release`.
  - **A lock left by a killed run** (e.g. the VM's overnight stop) is taken over by the next `lock acquire` or manual run once it is older than `run_lock.stale_after_hours` (default 12; `holder` records the takeover). Sooner: `precursor-ingest lock status`, then `lock release --run-id <any> --force`.
- **Retries are safe** with the same run id: every task is idempotent; `report` rewrites `run=<id>/` and adds a manifest; `resolve` replaces that run's anomaly rows. "Clear" on a whole DAG run therefore replaces that run's report and anomalies rather than keeping both.
- **Exit codes** (§2): 1 = error, retried; 2 = data/validation problem and 3 = config, both failed at once without retries (spec §14 #21).
- **Logs:** each task process writes its own `logs/<command>_<ts>-<hex>.jsonl` and ships it to `reports/step1/run=<id>/logs/`; Airflow also captures the container's stdout/stderr.
- **Metadata store: Postgres** (`PI_DATABASE_URL`, the `precursor_meta` database on the VM), never a SQLite file, when tasks run in separate containers (§12.2).
- **Between tasks 4 and 5, `inventory` is mid-update:** re-inventoried zips have `status='pending'` until `resolve` runs. Bronze refuses to start while any row is `pending` (exit 2); the shared lock keeps everything else out.
- **Landing backlog for `check_release`** (spec §2.0): `inventory.landing_backlog(ctx, select(ctx))` is read-only and lists (a) entries never stored, or re-published since their last good download, (b) stored zips not yet extracted and inventoried, and (c) the number of `pending` inventory rows. It compares with `download_log`, not `catalog`. `discover` stores the new catalog before downloading, so a failed or quarantined download would otherwise disappear from a catalog-only comparison. `check_release` should proceed (CHANGED / RESUME) while the backlog is not empty.

**Superseded reference DAG** (`ingest_vintage`, 2026-10-13). Kept for the parse-checked `docker_kwargs` and `mapped_commands` that spec §15.5 reuses; the real DAG is `cms_nh_pipeline` (spec §15.5). In it, `release_lock` moves to the end of the DAG, and every task, **including the mapped `extract_inventory`**, uses `PipelineDockerOperator`. Parse-checked with Airflow 3.1.8 + `apache-airflow-providers-docker` 4.6.1; never run against real containers.

```python
"""ingest_vintage: Step 1 (CMS archive landing) as Airflow 3 tasks. Reference DAG, not yet deployed.

Every task runs the `precursor-pipeline` image with the `precursor-ingest` CLI; no logic lives here.
All tasks of one DAG run pass the same --run-id, so they share one run lock, one run_log id and one
reports/step1/run=<id>/ folder. Exit codes: 1 = error (retry may help), 2 = data/validation problem,
3 = config. Replace the @daily schedule with the release sensor (STEP1_TODO T1) when it exists.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta

from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sdk import Asset, dag, get_current_context, task

IMAGE = os.environ.get("PI_PIPELINE_IMAGE", "precursor-pipeline:latest")
CLI = ["precursor-ingest"]
ENV_ARGS = [
    "--env",
    "prod",
]  # GCS + PI_DATABASE_URL; use "dev" with a ./data volume for local Airflow
LANDING = Asset("cms_nh_landing")  # Step 2's build_silver DAG is scheduled on this asset

docker_kwargs = dict(
    image=IMAGE,
    docker_url="unix://var/run/docker.sock",
    network_mode=os.environ.get("PI_DOCKER_NETWORK", "bridge"),  # the compose network with Postgres
    environment={"PYTHONUNBUFFERED": "1", "PI_BUCKET_URI": os.environ.get("PI_BUCKET_URI", "")},
    private_environment={"PI_DATABASE_URL": os.environ.get("PI_DATABASE_URL", "")},  # hidden in UI
    mount_tmp_dir=False,
    auto_remove="success",
)


def ingest(task_id: str, *args: str, **kwargs) -> DockerOperator:
    return DockerOperator(
        task_id=task_id,
        command=[*CLI, *args, *ENV_ARGS, "--run-id", "{{ run_id }}"],
        **docker_kwargs,
        **kwargs,
    )


@task
def mapped_commands(raw: str) -> list[list[str]]:
    """`entries --pending` prints a JSON list as its last line; one extract+inventory command per zip."""
    run_id = get_current_context()["run_id"]
    return [
        [*CLI, "step1", "--steps", "extract,inventory", "--entry-ids", entry_id, *ENV_ARGS]
        + ["--run-id", run_id]
        for entry_id in json.loads(raw)
    ]


@dag(
    schedule="@daily",
    start_date=datetime(2026, 10, 1),
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 2, "retry_delay": timedelta(minutes=5)},
    tags=["step1", "ingestion"],
)
def ingest_vintage():
    lock = ingest("acquire_lock", "lock", "acquire")
    fetch = ingest("discover_download", "step1", "--steps", "discover,download")
    pending = ingest("pending_entries", "entries", "--pending", do_xcom_push=True)
    per_zip = DockerOperator.partial(
        task_id="extract_inventory", max_active_tis_per_dagrun=3, **docker_kwargs
    ).expand(command=mapped_commands(pending.output))
    report = ingest(
        "resolve_report",
        "step1",
        "--steps",
        "resolve,report",
        outlets=[LANDING],
        trigger_rule="none_failed",  # also runs when no zip was pending (mapped task skipped)
    )
    unlock = ingest("release_lock", "lock", "release", trigger_rule="all_done", retries=0)
    lock >> fetch >> pending
    per_zip >> report >> unlock


ingest_vintage()
```

**Verified (2026-10-13):**
- `tests/ingestion/test_dag_tasks.py` runs this exact task sequence in-process, with a fresh `Context` per task and a shared Airflow-style run id. The result is identical to a single `step1` run; a quiet day maps 0 tasks; a new zip maps exactly 1; retries are no-ops; a failed extract stays pending.
- `tests/ingestion/test_cli.py` checks the same through the real CLI: exit codes, the JSON line, lock commands, and one shipped log per task.
- **On real Postgres 17:** 5 mapped tasks ran as parallel processes with one run id. All exited 0, nothing was left pending, and the inventory was identical to a single-process SQLite run.
- The `DockerOperator` pushes the **last line of combined stdout/stderr** to XCom (provider 4.6.1 source), which is why `entries` prints its JSON last and the DAG sets `PYTHONUNBUFFERED=1`.

**Still needed before deploying:** the `precursor-pipeline` image, the local Airflow compose (`docker/airflow-local/`), the `check_release` node (spec §2.0), and `tests/dags`.

**What `tests/dags` must cover for exit codes (spec §14 #21).** Provider 4.6.1 raises `DockerContainerFailedException(f"Docker container failed: {result!r}")`, where `result` is docker's `wait()` dict, e.g. `{'Error': None, 'StatusCode': 2}`. `PipelineDockerOperator` matches `'StatusCode': [23]` in that message, so it depends on the message text. The test needs Airflow 3.1.8 + `apache-airflow-providers-docker==4.6.1` in a **separate** uv dependency group or CI job (Airflow is deliberately not in the main lock). It should:
- patch `DockerOperator.cli` (a `MagicMock`: `create_container` → `{'Id': 'x'}`, `attach` → log bytes, `wait` → `{'Error': None, 'StatusCode': n}`) and `hook`, then call `execute`;
- assert `AirflowFailException` for 2 and 3, `DockerContainerFailedException` (retried) for 1, and the last log line as the XCom for 0;
- import the DAG with `DagBag` (no import errors, no cycles), including a check that the mapped task's operator class is `PipelineDockerOperator`.

---

## 13. Testing

| Suite | File | What it proves |
|---|---|---|
| Config safety | `tests/common/test_config.py` | Both env files load (dev, prod; the run-lock default; prod has no retention policy). `verify_tls=false`, an `http://` base URL and an unanchored allow-list are rejected. `${VAR}` / `${VAR:-default}` expansion works, and a missing required variable fails. |
| HTTP security | `tests/ingestion/test_http_security.py` | The allow-list rejects 10 hostile URL forms (other host, `//host`, `..`, query, fragment, `%2e`, non-archive path, `.exe`, empty). Cross-host redirects are refused. A wrong `Content-Type` or `Content-Length`, or a body that isn't a zip, is rejected. The size cap holds. A connection reset mid-download **resumes with `Range`** and yields the identical bytes. A 503 with `Retry-After` is retried. |
| Archive safety | `tests/ingestion/test_zipsafe.py` | Zip-slip (5 variants), symlinks, compression-ratio bombs, member-count and total-size limits, nested zips, bad zips and CRC corruption are all caught. Junk is detected. |
| Layout knowledge | `tests/ingestion/test_layout.py` | 25 **real** member names from all three layouts classify correctly. Layout detection, month tokens, date formats, `Processing Date` with a UTF-8 BOM, and manifest parsing work. |
| Port conformance | `tests/common/test_stores.py` | The same assertions pass on `LocalObjectStore` **and** `FsspecObjectStore(memory://)`: put/read/stat/list, no silent overwrite, key validation, move, deletes restricted to quarantine, no partial object after a failed stream, cross-backend copy, Parquet sink. SQLite in-memory and file metadata stores also pass. |
| Logging | `tests/common/test_logging.py` | Redaction, context fields, the JSON formatter, `setup_logging` writing valid JSON lines with `run_id`. |
| End-to-end (offline) | `tests/ingestion/test_pipeline_offline.py` | Against a fake CMS server with L2, sub-folder L2, L3+junk+manifest, partial-release and current zips: duplicate resolution (later wins; archive beats current), the gap, the partial-zip anomaly, row counts, the earliest survey date, an empty quarantine after success. A **suffixed release** (`..._Sep2026_3.csv`, lookup without a Processing Date) selects all 7 tables; a zip whose **CSVs match no table pattern** exits 2 with a `no_tables_recognised` anomaly; **`_10` beats `_9`** in the same zip. **Idempotent re-run** (0 re-downloads, identical inventory). **Re-publication** stored side by side, and the new copy is the one `selected`. A **corrupt zip is quarantined**. An **unsafe catalog URL is rejected without a request**. **Swapping the backend** (local disk + SQLite file vs `memory://` + in-memory SQLite) gives an identical inventory. |
| Run lock | `tests/common/test_run_lock.py` | Exclusive between runs, re-entrant within a run, only the holder releases, `--force` breaks a lock, 8 concurrent acquirers give exactly one winner, a lock older than the threshold is taken over (and a younger one is not). Also runs on Postgres with `PI_TEST_POSTGRES_URL`. |
| DAG tasks | `tests/ingestion/test_dag_tasks.py` | The §12.6 task sequence with one Context per task and a shared Airflow run id: same inventory as a single run; a quiet day maps 0 zips, a new zip maps 1; retries are no-ops; a failed extract stays pending; a failed or quarantined re-publication stays in the landing backlog until it succeeds; another run is blocked; a task without its run's lock is refused; a stale lock is taken over; the run manifest (and a retry adds one); deterministic tie-breaking; exit codes 0/1/2. |
| CLI | `tests/ingestion/test_cli.py` | Through `main()`: the `entries --pending` JSON last line, mapped-task commands, one shipped log per task, report retry, the `lock` commands and stale takeover, a read-only `catalog` (stores nothing, ignores the lock, contract change → 2), exit 3 for bad run ids, steps, dates, flags, commands, empty or stale `--entry-ids`. |
| Live contract | `tests/ingestion/test_contract_network.py` (`-m network`) | The archive endpoint shape is as expected; ≥89 monthly entries all pass the allow-list; the current zip `HEAD` shows `Accept-Ranges` and `application/zip`. |

Run: `uv run pytest` (offline: 149 passed, 6 Postgres variants skipped; `tests/ingestion/conftest.py` fails any test that tries real HTTP), `PI_TEST_POSTGRES_URL=… uv run pytest` (155) and `uv run pytest -m network` (2).

---

## 14. Operations runbook

| Situation | Action |
|---|---|
| First run | `precursor-ingest step1`. It downloads about 3.1 GB of zips and extracts about 18 GB of in-scope CSVs into `data/`. Review `data/reports/step1/latest/01_inventory_summary.md` (Checkpoint 1). |
| Monthly refresh | In production, the `cms_nh_pipeline` DAG runs it when `check_release` proceeds (spec §15). By hand: `precursor-ingest step1`. Unchanged zips are `skipped`; only new zips (and a new `current` entry) are downloaded. A catalog diff shows what changed. |
| Run lock held (`lock status`) | A DAG run is active, or one was killed. A lock older than 12 h is taken over automatically; otherwise `lock release --run-id x --force` once you are sure no run is active. |
| Exit 2 in Airflow | Not retried (data problem). Read the stderr list and `anomalies`, fix the cause (e.g. a quarantined zip), then let the next scheduled run proceed (the landing backlog keeps it pending) or trigger a run with `run_check_release=false`. |
| `republished` warning | Compare the two zips' inventories (`header_hash`, `row_count`). The newest copy is what downstream uses, and the original is kept for audit. |
| `quarantined` / `failed` | Inspect `quarantine/cms_nh/<run_id>/…` and the `download_log.detail`. Re-run; transient network failures resolve on their own. |
| `rejected` | The catalog advertised a URL outside the allow-list. **Do not widen the regex without review.** |
| Contract test fails | CMS changed the undocumented endpoints. Update `ingestion/sources/cms_pdc.py` and `configs/ingestion/sources.yaml`. |
| Rebuild only the report | `precursor-ingest step1 --steps resolve,report`. |
| Free disk space | The extracted CSVs can be regenerated from the zips: delete `data/landing/cms_nh/extracted/` and the `extracted_files` rows. **Never delete `landing/cms_nh/zips/`**; it is the immutable source of truth. |
| Start from scratch (local) | Delete `data/` and `logs/`. Local zips are `chmod 444`, so use `chmod -R u+w data` first. |

---

## 15. Results of the first full run

**Run** `20261006T233506Z-3a6902` (local env, 2026-10-06). The full report is `data/reports/step1/latest/01_inventory_summary.md`.

### 15.1 Volumes and timing
| Metric | Value |
|---|---|
| Catalog entries discovered | 98 (89 monthly `theme`, 8 `annual_theme`, 1 `current`) |
| Zips in scope / processed | 67 (66 monthly from 2021-01-27, plus the current zip). 64 were downloaded in this run; 3 were skipped from the earlier smoke run. |
| Bytes downloaded | 2.53 GB of zips (`landing/cms_nh/zips`: 2.4 GiB) |
| Extracted in-scope CSVs | 442 files, 18 GiB |
| Inventory rows | 1,492 members: 427 `selected`, 36 `duplicate_vintage`, 1,008 `other` (out-of-scope tables and dictionaries), 21 `junk` |
| Layouts seen in scope | L2 in 64 zips, L3 in 3 zips (2026-07-29, 2026-08-06, 2026-08-26) |
| Task durations | discover 0.6 s, download 43 s (3 parallel), extract 39 s, inventory 16 s, resolve under 0.1 s, report 0.3 s (about 1 min 40 s end to end) |
| Re-run (idempotency) | 67/67 `skipped`, 0 files re-extracted, `01_inventory.csv` **byte-identical**, 19 s |

### 15.2 Vintage coverage
- **Range:** **61 data months, 2021-01 → V = 2026-08**, for all 7 tables.
- **Gaps, identical for every table:** 2021-12, 2022-12, 2023-12, 2024-12, 2025-01, 2025-08, 2026-01. CMS never published these; this matches the exploration in §4.2.
- **Duplicates resolved (36 rows):**
  - 2021-10, 2022-10, 2023-10, 2024-10: the October data was re-issued in the next zip. Content is identical; the later zip is kept.
  - 2026-08: the archive zip and the current zip are identical; the archive is kept.
  - **2026-07 Provider Info:** the 2026-08-06 partial re-release was kept over the 2026-07-29 file, which has **different content** (same row count, 14,693). This follows the guide's "later-published wins" rule. It is a **post-cutoff correction**, though, and is flagged for the Checkpoint 1 decision. To keep strict point-in-time data instead, the rule can be changed to prefer the full zip (`resolve.py`, the `TYPE_RANK` / sort key).
- **Earliest survey date** in the 2021-01 Health Deficiencies file: **2015-01-30**.

### 15.3 Rows per vintage, from the selected files
| Table | Rows per vintage (min – max) | Total rows across 61 vintages |
|---|---|---|
| provider_info | 14,690 – 15,341 | 914,391 |
| health_citations | 367,658 – 419,479 | 23,879,099 |
| survey_dates | 149,705 – 217,687 | 11,158,607 |
| penalties | 7,836 – 40,210 | 1,604,233 |
| mds_qm | 249,730 – 276,121 | 16,207,983 |
| claims_qm | 58,760 – 61,361 | 3,657,447 |
| citation_lookup | 639 – 643 | 39,133 |

The facility count (14.7k–15.3k) is inside the guide's expected 14.5k–15.4k. The 5× swing in penalty rows matches the change in CMS fine-posting policy described in the feasibility review.

### 15.4 Schema eras: input for the Step 2 column map
| Table | Header sets | Main changes |
|---|---|---|
| provider_info | 10 | 2022-01: turnover and administrator columns added. 2022-07: RN staffing rating removed. 2022-08: adjusted weekend staffing. **2023-06: `Federal Provider Number` → `CMS Certification Number (CCN)`, `Provider State` → `State`, … and `Affiliated Entity` columns added.** 2023-08: latitude/longitude. 2024-07: case-mix columns. **2025-07: `Affiliated Entity` → `Chain`, rating cycle 2/3 consolidation.** 2025-10: `Urban`. 2026-02: facility-reported incidents and complaint counts removed. |
| health_citations | 3 | 2023-01: `Citation under IDR` / `IIDR` added. 2023-06: CCN rename. |
| penalties | 3 | 2023-06: CCN rename. **2026-06: `Fine ID` added.** |
| mds_qm, claims_qm, survey_dates | 2 each | 2023-06: CCN and location renames. |
| citation_lookup | 1 | Stable. It has no Processing Date column, by design. |

### 15.5 Anomalies (50 in total)
| kind | count | meaning |
|---|---|---|
| `missing_vintage` | 7 | The 7 gap months, one row per table |
| `duplicate_vintage` | 36 | See §15.2 |
| `partial_zip`, `size_outlier`, `member_count` | 1 each | All the same zip, `nursing-homes_2026-08-06.zip` (3.5 MB, Provider Info only) |
| `catalog_size_mismatch` | 1 | `nursing-homes_2026-07-29.zip`: the catalog says 622 MB, the server says 38.8 MB. CMS replaced the bloated Mac re-pack after our exploration. |
| `junk_members` | 1 | The same zip still holds 21 `__MACOSX` entries, now negligible in size |
| `nonstandard_path` | 1 | `nursing-homes_2026-08-26.zip` is served from `dataset-archives/unknown-type/` |
| `expected_vintage_from_name` | 1 | The 66 Citation Look-up files take their vintage from the file name |

There were **no** failed, quarantined, rejected or re-published downloads, and no unexpected vintage fallbacks.

---

## 16. Known limitations and hand-off to Step 2

**Limitations:**
- **Archive coverage** is whatever CMS kept. Seven data months are permanently missing (§4.2). L1 zips (before 2020-08) have no `Processing Date` and are outside scope.
- **Immutability on GCS** comes from the code's never-overwrite rule plus object versioning, not from a retention policy (§9.9).
- **The archive endpoints are undocumented.** The contract tests detect changes but can't prevent them.
- **Row counts for `other` (out-of-scope) files** aren't computed, because those files aren't extracted. Set `extract.only_in_scope: false` if needed.
- **`annual_theme` bundles** are not processed; they only duplicate the monthly zips.
- **Stale refreshes are not detected in Step 1.** Vintage 2025-09 repeats 2025-07 content for four tables (§0.4). Step 2 adds the check.
- **Not yet in the code** (§0.3): BigQuery metadata adapter, append-only writes (D5), warehouse port. Done: run lock (D8), DuckDB/S3 removal (D1/D2), the ingestion manifest (D11).
- **A killed run blocks for up to 12 h** (`run_lock.stale_after_hours`) before its lock is taken over (§12.6).

**What Step 2 consumes:**
- **Files to load:** `inventory` rows with `status = 'selected'` give exactly one extracted CSV per `(table, vintage_month)`, with `extracted_key`, `processing_date` (the guide's `cutoff_date`), `header_hash` and `row_count`. These row counts are the reconciliation targets for the bronze load.
- **Column map:** the header-set table in the Checkpoint 1 report is the input for `configs/column_map.yaml`. Every `header_hash` must be mapped.
- **Bronze output:** Step 2 writes through `ctx.table_sink.write_partition(table, {"vintage": vintage_month}, df)`. That gives `bronze/<table>/vintage=YYYY-MM/` on local disk or GCS.
- **Full details:** `docs/STEP2_HANDOFF.md`.
