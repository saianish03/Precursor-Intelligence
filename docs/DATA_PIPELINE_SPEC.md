# Precursor Intelligence: Data Pipeline Specification (v2)

> **Purpose.** This is the single reference for building the Precursor Intelligence data pipeline. It goes from a new CMS release to a verified, frozen facility × month table of features and labels, published to the warehouse. It is written for the engineers building and operating it.
>
> **It replaces `../dataset/Dataset_build_guide_v1.md` for implementation.** It keeps the guide's data logic, corrects the rules the archive analysis proved wrong, and organises every step under one of the **five DAG nodes**: `check_release → ingestion → bronze → silver → gold` (§0.7).
>
> **Precedence when sources disagree:**
> 1. **measured facts only** (counts, domains, timelines) in `docs/STEP2_HANDOFF.md`, the Step 1 inventory and `../sample_run_log.md`;
> 2. the rules and decisions in this file (§14), and D1–D11 in `docs/STEP1_DATA_INGESTION_PIPELINE.md` §0;
> 3. the team documents (`PI_Tech_Stack.docx`, `Precursor_Intelligence_Data_Pipeline_Plan.docx`), the team's CMS domain notes (e.g. `cms-data-domain`), and guide v1, as background only.
>    - Known skill conflicts, where this spec wins: dates are ISO only (not M/D/YYYY); severity B–L occurs; the later zip wins for a duplicate month (not "prefer the complete zip"); the dataset IDs live in `configs/ingestion/sources.yaml`.
>
> Where a **recommendation** in STEP2_HANDOFF or STEP1_DATA_INGESTION_PIPELINE differs from this file, this file wins. Examples: the `fine_id` event key, per-table stale flags, a `superseded` bronze status, and separate asset-scheduled DAGs.
>
> **Working rules:**
> - Follow the steps in order. At each 🛑 **CHECKPOINT**, write the report and wait for human approval.
>   - **The first backfill runs node by node** (or with the DAG param `stop_after=<node>`), with approval between nodes: Checkpoint 1 after ingestion; Checkpoints 2–4 after silver; Checkpoint 5 after gold.
>   - **Calibration tasks only write proposals** to `reports/`. `labels.yaml` and `qm_measures.yaml` change only through a human commit after Checkpoints 3 and 4.
>   - Routine monthly runs have no human checkpoint; their automated gates stop them instead.
> - Where it says "default", the value lives in `configs/`. Never hard-code a value that has a config key.
>
> **Location:** this is the canonical copy (`docs/DATA_PIPELINE_SPEC.md`). The evidence it cites lives in the parent project folder, outside the repo: `../sample_run_log.md` (the sample run) and `../sample_pipeline/` (the reference implementation of all 5 nodes, the fake CMS, `run_dag.sh`, `reverify.sh`).
>
> **Status:** v2.0, 2026-10-15. Every step was run and verified on a 39-facility sample of all 61 vintages (`../sample_run_log.md`; reference code in `../sample_pipeline/`). Rules changed by that run are marked *(sample-run finding)* or "measured". Open team decisions are in §14. **Amended 2026-10-29** by the VM deployment decisions §14 #24–#27 (release-check DAG, additive schema drift, VM size, Postgres on the VM).

---

## 0. Context you need before starting

### 0.1 What the product does
Precursor Intelligence scores U.S. skilled nursing facilities (SNFs) for **long-term-care liability insurers**. Every month, each facility gets a calibrated probability that it will have a serious regulatory event within the next 12 months (and, optionally, the next 6). Facility scores are later rolled up to the operator (chain) account. This file covers **only** the data pipeline. Modelling, serving, the agents and the frontend consume its gold output.

### 0.2 Prediction targets
| Label | Definition (short) | Role | Status |
|---|---|---|---|
| `y_g12` | Any health citation with scope/severity **G–L** on a survey dated in the next **12 months** (any health survey type: standard, complaint, infection control) | **Primary model** | Decided |
| `y_g6` | Same as `y_g12` but over **6 months** | Secondary "between-renewal early warning" model | Decided |
| `y_pen12` | Any **fine or payment denial** with penalty date in the next **12 months** | Model B in guide v1. The team skill says "use penalties as features, not as the primary label". | **Open (§14).** It is built, so the comparison stays possible. |

Auxiliary label columns exist for analysis only (§9.4). **No label column may ever be used as a feature.**

### 0.3 Key terms and notation
| Symbol / name | Meaning |
|---|---|
| **vintage** | One monthly CMS data release. Identified by `vintage_month` (`YYYY-MM`), taken from the `Processing Date` inside the files, **never** from the zip file name. |
| `cutoff_date` | The `processing_date` of a vintage, always the 1st of `vintage_month`. This is "today" for that vintage. |
| **t** | A prediction month. It equals a `vintage_month`. A row `(ccn, t)` means "facility `ccn`, scored using only what CMS had published as of vintage t". |
| **V** | The latest vintage ingested. |
| **H** | Label horizon in months (12 or 6). |
| **B** | Maturity buffer in months: time allowed for late-posted records to appear (`B_cit` for citations, `B_pen` for penalties). Measured in **Step 4**. |
| **CCN** | CMS Certification Number. **Always a 6-character string**, about 1.8% alphanumeric (e.g. `01A193`). Never cast it to an integer. |
| **G+** | Scope/severity letters G, H, I (actual harm) and J, K, L (immediate jeopardy, "IJ"). |
| **zip date** | The date in the archive zip name. It is when CMS published the zip; the data month is usually the previous month. |

### 0.4 The two leakage rules (never break these)
1. **Feature rule.** Every feature for row `(ccn, t)` is computed **only from vintage t**, plus the t−6 and t−12 vintages for "change" features. Vintage t holds exactly what CMS had published at that time (about 3 inspection cycles of citations and 3 years of penalties), so late-posted records can't leak in. Feature event windows use `event_date <= cutoff_date`.
2. **Label rule.** Labels come from the **event ledgers** (all vintages unioned, §4). The ledgers are needed because closed facilities and old inspection cycles drop out of later files. A label counts only events dated **strictly after** `cutoff_date` and **within H months**.

### 0.5 Defaults (editable in `configs/`)
| Setting | Default | Config |
|---|---|---|
| Language and tooling | Python 3.11+ (3.12 pinned), `uv` only | `pyproject.toml`, `uv.lock` |
| Vintages ingested | **2021-01 → V** (2019–2020 are archived but never ingested: no Inspection Dates file, different posting regime) | `configs/ingestion/sources.yaml` (`scope`) |
| Prediction rows (spine) | **t from 2023-01 → V**. 2021–2022 are used only for ledgers, posting-lag measurement and the t−6/t−12 history of change features. | `configs/splits.yaml` (`spine_start`) |
| Maturity buffers | **`B_cit = 8`, `B_pen = 12`**, measured on the full archive (Step 4, 2026-10-15). Guide v1 guessed 3 / 9; the team plan 6 / 18. Re-measure in 2027. | `configs/labels.yaml` |
| Standard-survey due interval | **456 days** (15 months) | `configs/labels.yaml` |
| Splits | `calib_months = 3`, `test_months = 3` (open, §14 #11), `required_labels = [g12, g6]` (gate). Windows are **calendar months** ending at `last_labeled_t`. | `configs/splits.yaml` |
| Censoring | strict: observable until `t + H + B` | `configs/labels.yaml` (`censor_requires_buffer`) |
| Check thresholds | Rows per t 14,000–15,600; month-over-month change ≤ 2%; sanity rates (§11.2). Scaled values for samples. | `configs/checks.yaml` |

### 0.6 Storage and directory layout
**Storage rules** (decisions D1–D3, D9–D11):
- **Files stay files.** Every layer (landing, bronze, silver, gold) is files on one object store: local disk in development and tests, Google Cloud Storage in test and production. All I/O goes through the `ObjectStore` / `TableSink` ports in `src/precursorintelligence/common/io/`, so the same code runs on both.
- **No database for intermediate layers, and no DuckDB.** Polars on Parquet.
- **Only the final gold dataset goes to a warehouse:** BigQuery in GCP, Postgres locally. Gold Parquet stays the source of truth.

**Pipeline metadata** (D4–D8):
- **Where:** a metadata store, **Postgres** locally and, in GCP, the `precursor_meta` database on the Airflow VM (next to the `airflow` and `mlflow` databases), dumped to GCS at every VM shutdown (§14 #27); SQLite only in unit tests. Never Cloud SQL. **There is no BigQuery metadata adapter** (amended 2026-10-16 to follow `PI_Metadata_Storage_Plan`, 2026-10-09; this replaces D4's `precursorintelligence_ops`). BigQuery only holds a **read-only mirror** in the dataset `meta`, copied from Postgres by one task at the end of each DAG run (§15.3). The API and dashboard read a status snapshot, `status/latest.json` in GCS, written by the last task of the monitor DAG, so they never reach the VM.
- **How it is written:** append-only rows plus "latest" views (D5); each task reads once and writes once; the BigQuery mirror is a batch load job, never streaming (D6). State that is a current value rather than an event (`vintage_flags`, `run_lock`) is one row per key and is upserted.
- **Concurrency:** one run at a time (the `data_pipeline` run lock).

```
<root>/                                   # local: ./data      prod: gs://<bucket>
  landing/cms_nh/
    catalog/fetched_at=<ts>/              # raw archive/current/metastore JSON as returned
    zips/zip_date=<YYYY-MM-DD>/<zip>      # immutable; republished copies as <stem>__<sha8>.zip
    extracted/zip_date=<YYYY-MM-DD>/<zip_stem>/<file>.csv   # in-scope CSVs only
  quarantine/cms_nh/<run_id>/…            # downloads that failed validation
  bronze/<table>/vintage=<YYYY-MM>/part-0.parquet          # all-text, lossless, lineage columns
  silver/<table>_by_vintage/vintage=<YYYY-MM>/part-0.parquet
  silver/ledger_{citations,penalties,surveys}/part-0.parquet
  silver/citation_lookup/vintage=<YYYY-MM>/  silver/provider_identity_by_vintage/vintage=<YYYY-MM>/
  silver/{facility_presence,facility_universe,qm_inventory,citation_lookup_versions}/part-0.parquet
  gold/spine/part-0.parquet  gold/labels/part-0.parquet   # single files (small); features and dataset by t
  gold/features_<family>/t=<YYYY-MM>/part-0.parquet
  gold/facility_month_dataset/t=<YYYY-MM>/part-0.parquet
  gold/dim_facility_latest/part-0.parquet  gold/dataset_manifest.json   # dim: product display only (as of V)
  gold/_staging/run=<run_id>/…            # gold builds here; promoted to gold/ only after the gate passes
  reports/<step>/run=<run_id>/ and reports/<step>/latest/  # checkpoint reports (md + csv)
  metadata/<layer>/run_date=<YYYY-MM-DD>/<layer>_run_<ts>.json   # per-run manifests (D11)
configs/
  ingestion/sources.yaml   # endpoints, scope, the 7 tables, HTTP and security limits (Step 1)
  dev.yaml  prod.yaml      # backends per environment (object store, metadata store, warehouse)
  column_map.yaml          # raw header → canonical name, with vintage ranges and dtype
  keep_columns.yaml        # canonical columns kept per table
  value_maps.yaml          # enums: survey type, SFF, sprinkler, council, ownership group, Y/N
  calibration.yaml         # Steps 4-5 proposal settings: lag coverage, left-censoring floors, QM bridges and drops
  footnotes.yaml           # CMS footnote code → meaning and missing-reason group
  labels.yaml              # horizons, buffers, severity sets, due interval
  splits.yaml              # spine start; train/calib/purge/test rules per label
  qm_measures.yaml         # selected and bridged quality measures (Step 5)
  feature_registry.yaml    # every feature: family, source, formula, first vintage, dtype, model_use
  checks.yaml              # thresholds for automated checks
```

> **Naming (open, §14):** the team plan calls the raw layer "bronze" (`bronze/pdc/vintage=YYYY-MM/`). This spec keeps raw files in `landing/` by **zip date**, because the vintage is only known after reading `Processing Date` and one month can arrive in two zips. `bronze/` is the first layer keyed by vintage.

### 0.7 The five DAG nodes
Every step in this file belongs to exactly one node. A node is an Airflow `TaskGroup` whose tasks run the `precursor-pipeline` image through `DockerOperator` with one CLI command each. Nodes contain no business logic. Nodes 2–5 form the DAG `cms_nh_pipeline`. Node 1 runs in its own small DAG, `cms_nh_release_check`, which starts the pipeline through an asset (decision §14 #24).

| # | Node | Steps | Reads | Writes | Emits asset |
|---|---|---|---|---|---|
| 1 | `check_release` | Release sensor (§2.0): READY / RESUME / CHANGED / PARTIAL / BUSY / NO_CHANGE | CMS metastore API (7 dataset IDs); the archive catalog; `inventory`, `catalog`, `gold_builds`, `run_lock` | decision row in `vintage_registry` | `cms_nh_release`, only on READY, RESUME or CHANGED; it starts `cms_nh_pipeline` |
| 2 | `ingestion` | Step 1 (§2) | archive/current/metastore endpoints; landing | landing zips + extracted CSVs; `catalog`, `download_log`, `inventory`, `anomalies`; Checkpoint 1 report | `cms_nh_landing` |
| 3 | `bronze` | Step 2a (§3.1) + content hash, `stale_refresh`, `revised_after_publication` | `inventory where status='selected'`; landing CSVs | `bronze/<table>/vintage=…`; `bronze_files`, `vintage_flags` | `cms_nh_bronze` |
| 4 | `silver` | Steps 2b–2e (§3.2–3.5), Step 3 (§4), §6 items 1–2 and the QM inventory. **Calibration tasks**, backfill or on demand only: Step 4 (§5) and the §6 item 3 QM proposal. | bronze; configs | silver tables, ledgers, universe, QM inventory, `gold/dim_facility_latest`; `silver_partitions`; Checkpoint 2–4 reports | `cms_nh_silver` |
| 5 | `gold` | Steps 6–10 (§7–§11) + publish to the warehouse | silver; `inventory`, `vintage_flags` (manifest); configs | `gold/_staging` → spine, features, labels, `facility_month_dataset`, manifest, verification report; warehouse table; `gold_builds` | `cms_nh_gold`, only when published (the ML DAGs start from this) |

**Run modes:**
- **Monthly:** `check_release` proceeds (READY, RESUME or CHANGED) and emits `cms_nh_release`, which starts `cms_nh_pipeline`. Ingestion, bronze and silver process only new or changed inputs. Gold rebuilds the features for every t (unchanged months come out byte-identical) and **recomputes all labels**, because a new release can mature older rows or add positives to them.
- **Backfill:** trigger `cms_nh_pipeline` by hand with `mode=backfill`. Ingestion takes every in-scope zip; bronze, silver and gold process every vintage; the calibration tasks run.
- **Development and test runs** (decisions §14 #20, #24): trigger `cms_nh_pipeline` by hand, and keep `cms_nh_release_check` **paused**, so no release starts a run on its own. A manual run never goes through `check_release`: the 4 nodes run unconditionally. This is for repeated runs on local disk or in the cloud while the project is being built. It is safe because every node is idempotent: unchanged inputs are no-ops, and gold comes out byte-identical.
- **Production:** unpause `cms_nh_release_check`. The pipeline then runs only when `check_release` decides READY, RESUME or CHANGED, plus any manual run.

**Shared between nodes:**
- One `--run-id {{ run_id }}` for every task of a DAG run.
- One run lock (`data_pipeline`): taken by the first **ingestion** task (`acquire_lock`), released by the last task (`release_lock`, an Airflow **teardown** of `acquire_lock`, not `trigger_rule="all_done"`; §15.1).
- Exit codes: 0 ok, 1 error (retried), 2 data/validation failure and 3 config (**both fail without retry**, decision §14 #21; §15.4).

The task list per node, run modes, exit codes and the DAG sketch are in **§15**, verified on the sample run.

### 0.8 Changes from guide v1 (summary)
| Area | Guide v1 | This spec | Evidence |
|---|---|---|---|
| Name, language | SurveyHorizon; Python 3.11 | Precursor Intelligence; Python 3.11+ with uv | team decision |
| Getting history | Archive page "may be rendered by JavaScript"; ask a human to download | Undocumented archive JSON endpoints list every monthly zip; fully automatic | Step 1, built |
| Raw layout | `data/raw/zips/`, `raw/extracted/<vintage>/` | `landing/cms_nh/zips/zip_date=…/`, extracted by zip date | Step 1 |
| Spine start | 2022-01 | **2023-01** (2021–2022 = history only) | team plan + team skill; decision 2026-10-15 |
| Penalty duplicates | Drop exact-duplicate rows | **Keep them**, add `dup_seq`; they are distinct fines | STEP2_HANDOFF §6 |
| Citation survey type | Map `survey_type` to an enum | `Survey Type` is always "Health"; derive from the 3 flag columns | STEP2_HANDOFF §5.3 |
| Severity domain | A–L | Valid A–L, but only B–L ever occurs | profiling |
| Legacy headers | `PROVNUM` etc. | Not present in 2021+; the real renames are the 2023-06 set and the 2025-07 Chain rename | profiling |
| Encoding, blank row | — | UTF-8 then cp1252 fallback; drop the trailing blank row and reconcile counts | profiling |
| Stale refresh | — | Flag `stale_refresh` when a vintage's content equals the previous one (2025-09) | measured |
| Antipsychotic QM | "Expected stable" | 419 → 481 in 2026-02; bridge needed. Also 405→480, 453→479 (2025-02); 471 dropped | QM timeline |
| QM freshness | — | Measure periods differ per measure and lag the cutoff (2–4 quarters); add `qm_age_months` | measured (corrects STEP2_HANDOFF's "frozen" claim) |
| Citation lookup | Latest vintage only | Per-vintage partitions + `citation_lookup_versions` (recommended, §14 #5) | verifier: 7 content changes in 61 vintages |
| Label status order | positive before immature | **immature first** | verifier: 43 immature-zone positives |
| Censoring | observable until window end | until window **+ buffer** | sample: 11 g12 rows |
| Buffers | 3 / 9 (guess) | **8 / 12** (measured) | full-archive lag |
| Storage and metadata | Parquet on local disk | D1–D11: local/GCS files; Postgres metadata with a BigQuery mirror (amended 2026-10-16); only gold in the warehouse; run lock; manifests | decisions 2026-10-08 |
| Step references | "B measured in Step 5", "qm_measures filled at Step 6" | Step 4 and Step 5 | fixed |

**Deviations from the team documents** (2026-10-08; the docx files are not edited):

| Team documents | This spec | Why |
|---|---|---|
| Separate asset-chained DAGs (`pdc_release_sensor`, `ingest_vintage`, `build_silver`, `build_gold`, `backfill_archive`) | One pipeline DAG with 4 TaskGroups (ingestion → gold), started by a small release-check DAG through the asset `cms_nh_release`; backfill and development runs = manual runs of the pipeline DAG with params | One run id and one lock end to end; assets are still emitted per node for downstream DAGs. **Decided 2026-10-15** (§14 #19), release check split out **2026-10-29** (§14 #24); the team plan is being updated. |
| Per-table CSVs via the metastore `downloadURL`; annual zips uploaded by hand | Monthly zips through the archive JSON endpoints (Step 1) | One code path for history and new months |
| Bronze = raw files, never overwritten | Raw files in `landing/` are never overwritten (republished copies sit side by side); a bronze partition is **reloaded** when its source changes (D10) | Bronze is derived from landing |
| Pandera checks | Named checks recorded in `dq_results` (the sample run); Pandera schemas can implement the same rules in the repo | — |
| `vintage_registry` with sha256, row counts, stale flag and status per vintage | `vintage_registry` holds `check_release` decisions. Per-vintage facts sit in `inventory`, `bronze_files` and `vintage_flags`. | — |
| Pipeline state in BigQuery (Tech Stack, 2026-10-08) | Postgres `precursor_meta` on the VM is the primary store and BigQuery `meta` a read-only mirror, as in `PI_Metadata_Storage_Plan` (2026-10-09). No BigQuery metadata adapter, so the `run_lock` row stays in Postgres. | Cost and latency: BigQuery is slow for small reads and writes |
| Month-over-month row limits for citations, penalties and surveys | Rows per t for the spine only (§11.1). **Add the other limits to `checks.yaml`.** | Gap |
| Ledgers updated incrementally | Ledgers rebuilt over all vintages every run | Small; deterministic |
| DuckDB / Polars | Polars only (D2) | — |
| Buffers 6 / 18; prediction months from 2023-01 | 8 / 12 measured; spine 2023-01, which leaves very little training data (§9.5) | Measured |
| Airflow VM e2-standard-2 (8 GB) | **e2-standard-4** (4 vCPU, 16 GB); the machine type is a Terraform variable | Gold holds the whole dataset in memory, and every monthly run rebuilds all of gold, so monthly runs need the backfill's RAM. Full-archive gold is estimated at ≈ 5.8 GB (`AIRFLOW_PIPELINE.md` §9.1), next to ≈ 2–2.5 GB for Airflow and Postgres. **Decided 2026-10-29** (§14 #26) |
| Nightly `pg_dump` to GCS; an instance schedule stops the VM overnight and on weekends | Postgres stays on the VM (as planned). It is dumped to GCS **at every VM shutdown**. The VM is started and stopped by hand, with an idle auto-shutdown | The VM is off most of the time, so a nightly job would rarely run. A fixed stop could cut a long backfill short, but the idle shutdown never stops a running DAG run. **Decided 2026-10-29** (§14 #27) |

---

## 1. Tables used (final selection)

| # | CMS table (file prefix) | Dataset ID | Grain | Used for |
|---|---|---|---|---|
| 1 | **Provider Information** (`NH_ProviderInfo_`) | `4pq5-n9py` | facility × vintage | Facility universe per vintage; facility, staffing, turnover and status features; chain ID (product only) |
| 2 | **Health Deficiencies** (`NH_HealthCitations_`) | `r5ix-sfxw` | one citation | Citation-history features; **G+ labels** |
| 3 | **Inspection Dates** (`NH_SurveyDates_`) | `svdt-c123` | one survey | Survey-timing features (when the next inspection is due) |
| 4 | **Penalties** (`NH_Penalties_`) | `g6vv-u9sr` | one penalty | Enforcement-history features; penalty labels |
| 5 | **MDS Quality Measures** (`NH_QualityMsr_MDS_`) | `djen-97ju` | facility × measure × vintage | Clinical-risk features (falls with major injury, pressure ulcers, antipsychotics, …) |
| 6 | **Medicare Claims Quality Measures** (`NH_QualityMsr_Claims_`) | `ijh5-nb2v` | facility × measure × vintage | Hospitalisation and ED-visit features |
| 7 | **Citation Code Look-up** (`NH_CitationDescriptions_`) | `tagd-9999` | one F-tag | Tag → category and description. The file has **no** `Processing Date`; its vintage comes from the file-name month token. |
| — | Footnote codes (data dictionary Table 15) | — | static | `configs/footnotes.yaml`: footnote → missing-reason category |

**Explicitly out of scope** (do not ingest without asking): Survey Summary, Ownership, Fire Safety Deficiencies, State/US Averages, Health Inspection Cut Points, Data Collection Intervals, SNF QRP, SNF VBP, Swing Beds, PBJ (optional phase 2).

---

## 2. Release check and STEP 1: Download and inventory the archive

### 2.0 Release check (node `check_release`)
CMS publishes no notification, so the DAG `cms_nh_release_check` polls (releases usually land on the 24th–30th, sometimes mid-month). It runs hourly with `catchup=False`, so its first check comes a few minutes after the VM starts, and on a proceed decision it emits the asset `cms_nh_release`, which starts `cms_nh_pipeline` (decision §14 #24).
1. Call `GET https://data.cms.gov/provider-data/api/1/metastore/schemas/dataset/items/{dataset_id}?show-reference-ids=false` for the 7 dataset IDs (§1). Read `modified` (the data month), `released` and `nextUpdateDate`.
2. Fetch the **archive catalog** (the endpoint Step 1 uses) and compare it with the stored `catalog`: new entries, or entries whose size changed.
   - CMS sometimes rewrites a zip without updating the catalog size (the 2026-07-29 zip still lists 622 MB). Such a rewrite is caught only by the next ingestion run's `HEAD` checks.
3. Read the state: the latest **selected** vintage in `inventory`, the latest **published** V in `gold_builds`, and whether a pipeline run holds the `data_pipeline` lock (`run_lock`).
4. Decide, in this precedence order:

   | Condition | Decision | Action |
   |---|---|---|
   | A pipeline run holds the `data_pipeline` lock | `BUSY` | Skip this check; the next one decides again. *(Without this, a check during a run would see the run's own unpublished vintage as RESUME and queue a second run.)* |
   | All 7 `modified` equal one month that is newer than the latest ingested vintage | `READY` | Continue the DAG for that month |
   | Nothing new at CMS, but the latest ingested vintage was never published to gold (a run failed after ingestion, e.g. at the gold gate) | `RESUME` | Continue the DAG: every node is idempotent, so ingestion and bronze are no-ops and silver and gold run again. *(Sample-run finding: without this, a gold failure is never retried automatically, because the next day's check sees nothing new.)* |
   | No new month, but the archive catalog has a new or re-sized zip (CMS re-published an old archive, or posted a corrective partial zip such as 2026-08-06) | `CHANGED` | Continue: Step 1 stores the new bytes side by side; bronze reloads the affected partitions (D10). *(Sample-run finding: a republish does not change the metastore.)* |
   | Some datasets changed in the metastore, but not all 7 to the same new month, and no new zip yet | `PARTIAL` | Skip; alert if still partial after `checks.yaml: partial_alert_days` (default 3) |
   | Nothing is newer than the latest ingested vintage, and gold is published for it | `NO_CHANGE` | Skip |
   | The API call fails or the response shape is wrong | error | Exit 1 (retry); repeated failure alerts |

5. Append the decision (time, the 7 `modified` values, the latest ingested and published months, the catalog changes) to `vintage_registry`. Re-running the same day gives the same decision.
6. Print the decision as JSON on the last stdout line, with `proceed = decision ∈ {READY, RESUME, CHANGED}`.

### 2.1 Ingestion (node `ingestion`): implemented in `src/precursorintelligence/ingestion/`
Details: `docs/STEP1_DATA_INGESTION_PIPELINE.md`. Summary of the rules:

1. **Discover.**
   - The undocumented archive endpoints give the history without HTML scraping:
     - `GET /provider-data/api/1/archive/aggregate/theme/nursing-homes/relative` (every monthly and annual zip);
     - `GET /provider-data/api/1/archive/aggregate/current/theme/all/relative` (the current "download all" zip; exact `theme` match required).
   - Save the raw JSON to `landing/cms_nh/catalog/fetched_at=<ts>/`.
   - Upsert `catalog`; log new, removed and size-changed entries.
   - A changed response shape stops the run (exit 2).
2. **Select:** zip types `theme` + `current` (annual bundles are nested copies of the monthly zips); zip date ≥ 2021-01-01 (the 2021-01-27 zip is the first holding a 2021-01 vintage).
3. **Download** each zip (HTTPS only; anchored URL allow-list; same-host redirects; retries; resumable):
   1. Write into `quarantine/`.
   2. Validate: status, Content-Type, size, `PK` signature, the zip-safety limits, and the CRC of every member.
   3. Promote to `landing/cms_nh/zips/zip_date=…/` and make it immutable.
   4. Record sha256 + Last-Modified in `download_log`.
   - **Never overwrite:** a re-upload with different bytes is stored as `<stem>__<sha8>.zip` (status `republished`).
   - Statuses: `downloaded`, `skipped`, `unchanged`, `republished`, `adopted`, `rejected`, `quarantined`, `failed`.
4. **Extract** in-scope members only (the 7 tables) into `landing/cms_nh/extracted/zip_date=…/<zip_stem>/`.
   - Flatten subfolders.
   - Skip `__MACOSX/`, `._*` and `.DS_Store`.
   - Zip-slip, symlink, nested-zip and bomb limits apply to every member.
   - Archive layouts: **L1** (≤ 2020-07, `*_Download.csv`, no Processing Date; out of scope), **L2** (2020-08 → 2026-06, `NH_<T>_<MonYYYY>.csv`, sometimes in a subfolder), **L3** (2026-07 →, `<datasetId>_<YYYY-MM-DD>_NH_<T>_<MonYYYY>.csv` + `manifest.json`, plus macOS junk).
5. **Inventory:** one row per member, with:
   - the table (one of 7, `other` or `junk`);
   - `processing_date` from the first data row;
   - `vintage_month` and `vintage_source` (`processing_date`, then the file-name month token, then the zip date; every fallback is logged);
   - `row_count`, `n_columns`, `header_hash`, `crc32`;
   - for L3, the `manifest.json` cross-check.
6. **Resolve:**
   - **Duplicates:** for each `(table, vintage_month)`, the file from the later zip is `selected` and the others `duplicate_vintage`; on the same date, archive beats current.
   - **Gaps:** list the gaps.
   - **Anomalies:** `partial_zip`, `missing_vintage`, `duplicate_vintage`, `junk_members`, `junk_bloat`, `member_count`, `size_outlier`, `catalog_size_mismatch`, `vintage_fallback`, `expected_vintage_from_name`, `nonstandard_path`, `no_vintage`, `republished`, `download_<status>`.
7. **Report:** `reports/step1/{run=<id>,latest}/01_inventory.csv` + `01_inventory_summary.md` (CSV cells are escaped against formula injection).

**Known archive facts (2021-01 → 2026-08):**
- **Coverage:** 61 vintages.
- **Never published:** 2021-12, 2022-12, 2023-12, 2024-12, 2025-01, 2025-08, 2026-01.
- **Duplicate data months:**
  - every October 2021–2024 (identical content);
  - 2026-07 Provider Info: the 2026-08-06 partial re-release corrected case-mix and adjusted staffing for about 14.2k facilities, and is `selected` (affects scoring rows only, §14);
  - 2026-08 archive vs current zip (identical).
- **Size anomalies:**
  - the 2026-07-29 zip was a 622 MB repack with stored members; CMS has since rewritten it to 38.8 MB;
  - the 34 MB 2026-08 zip is a normal release without fire-safety, swing-bed and VBP files.

**Task contract in Airflow** (implemented and tested):
1. `lock acquire`
2. `step1 --steps discover,download`
3. `entries --pending` (last stdout line = JSON list)
4. `step1 --steps extract,inventory --entry-ids <id>` (mapped, one per pending zip)
5. `step1 --steps resolve,report`
6. `lock release` (a teardown of `lock acquire`)

All with `--run-id {{ run_id }}`.

**In the 5-node DAG** (§14 #19), steps 1–5 are the ingestion TaskGroup. Step 6 moves to the end of the DAG (`release_lock` after gold), so bronze, silver and gold run under the same lock. The Step 1 CLI already supports this unchanged: a task given `--run-id` requires the lock and never releases it. The separate `ingest_vintage` reference DAG in STEP1 doc §12.6 is superseded.

🛑 **CHECKPOINT 1. Report `01_inventory_summary.md`:**
- vintages found per table, and gaps;
- the earliest survey date in the 2021-01 Health Deficiencies file;
- distinct header sets (by `header_hash`) per table, with the vintage range each covers;
- anomalies.

---

## 3. STEP 2: Bronze, then harmonise schemas and prune columns ("merge across all times")

The goal is **one stacked table per CMS table across all vintages**, with consistent column names and only the columns needed. §3.1 runs in node `bronze`; §3.2–3.5 run in node `silver`.

### 3.1 Bronze (node `bronze`)
Bronze is a **lossless, all-text Parquet copy** of each selected CSV. It reads only from landing; it never downloads or extracts.

**Input:** `inventory where status = 'selected'`, exactly 1 file per `(table, vintage_month)`. Hold the `data_pipeline` lock; refuse to start if any inventory row is `pending` (exit 2).

**Per file:**
1. **Skip** if the partition already exists with the same `_zip_sha256` and `_header_hash`.
2. **Read** the bytes from `landing/…`.
3. **Decode:** strict UTF-8, falling back to cp1252. Record the encoding.
   - Expected: 151 cp1252 files (2021-01 → 2023-04, byte `0x92`), 0 decode failures.
   - Never decode UTF-8 as cp1252; never use lossy decoding.
4. **Parse every column as text**, keeping the original header names and order. Use a real CSV parser: there are quoted commas and newlines.
5. **Drop all-null rows** (the trailing blank line in 2021-01 → 2024-06 files) and count them as `blank_rows`.
6. **Add the lineage columns** (real types):

   | Column | Type | Source |
   |---|---|---|
   | `_vintage_month` | string `YYYY-MM` | `inventory.vintage_month` |
   | `_processing_date` | date | `inventory.processing_date` |
   | `_source_zip` | string | `inventory.zip_name` |
   | `_zip_sha256` | string | `inventory.zip_sha256` |
   | `_source_file` | string | `inventory.member_path` |
   | `_header_hash` | string | `inventory.header_hash` |
   | `_row_number` | int64 | 1-based position in the file, after dropping blank rows |
   | `_ingested_at` | timestamp (UTC) | load time |
7. **Reconcile:** `rows + blank_rows == inventory.row_count`, and the recomputed header hash equals `inventory.header_hash`.
   - **Fail closed:** a file that fails a check is written to **no layer path**. It gets a `bronze_files` row with status `failed`, the node exits 2, and the file is retried on the next run, because only `loaded`/`reloaded` rows ever count as done.
8. **Content hash:** sha256 over the data columns, excluding `Processing Date`, with rows sorted.
   - **Per file:** `content_unchanged = (content_hash == content hash of the same table's previous published vintage)`. This is informational. It is normal for `citation_lookup` (static reference data: 50 of 60 vintages are byte-identical to the previous one) and for the QM tables (values refresh quarterly).
   - **Per release:** `stale_refresh(vintage) = content_unchanged` for **all three event tables**: health citations, survey dates and penalties. Record it in `vintage_flags`.
     - Known case: **2025-09 only**. It equals 2025-07 for citations, penalties, survey dates (and claims QMs); 2025-08 was never published.
     - **Keep stale data:** it is what CMS published at that cutoff.
     - Downstream: point-in-time features for that t stay valid; the ledgers gain no events; the ML DAG skips retraining.
   - *(Sample-run finding: a per-table stale rule flagged 52 lookup and dozens of QM vintages; the release-level rule flags exactly 2025-09.)*
9. **Write** `bronze/<table>/vintage=YYYY-MM/part-0.parquet`; append a `bronze_files` row (rows, blank rows, encoding, fingerprints, content hash, `content_unchanged`, status `loaded` / `reloaded` / `failed`; skipped files get no row). `_processing_date` is null for `citation_lookup`, which has no such column.
10. **Check over every selected file**, not only those processed in this run: each one has a `loaded`/`reloaded` partition with its current `zip_sha256` (`every_selected_file_has_valid_partition`).
11. **Write the run manifest** `metadata/bronze/run_date=…/bronze_run_<run_id>_<ts>.json`. Include the timestamp, so a retry never overwrites it.

**Republished data (D10):** if the selected `zip_sha256` for a (table, vintage) differs from the bronze partition's, bronze **reloads** that partition and records `reloaded`. Silver and gold then rebuild everything downstream of that vintage.
- **When the content differs too,** set `revised_after_publication` for that (table, vintage) in `vintage_flags`. The sample's republish test flagged `2024-03: provider_info`.
- **This rewrites history:** a later CMS correction replaces what was published as of t. The time-travel test cannot detect it, so the flag (and the manifest) must make it visible. Whether historical t should keep the first-published version is decision §14 #18.

**Bronze never** renames, prunes, type-casts, maps values, de-duplicates or stacks months.

### 3.2 Column map (`configs/column_map.yaml`; node `silver`)
For each table, map `canonical_name → [raw aliases]`. Each alias has a valid vintage range (`from`/`to`, inclusive) and a target dtype. Build it from the actual headers found in Step 1; the machine-checked draft is `docs/data/step2_reference/column_map.draft.yaml`.

**Rule (amended 2026-10-29, decision §14 #25):** every raw header in every vintage is mapped, listed under `ignored:`, or **new**. A new header is one the column map doesn't know.
- **Additive change, handled automatically:** the partition has new headers, but every expected column is present. Silver builds the partition without the new columns and logs a warning. It stores them with the partition (`silver_partitions.stats`), lists them in `02_unmapped_headers.csv` and the run manifest, and the DAG emails one warning per run (also after a retry).
  - Bronze keeps the new columns as text, so mapping one later is a config change plus a rebuild.
  - Mapping a header, or adding it to `ignored:`, stops the warning.
  - **A rename CMS publishes under both names for a while:** the old name is used and the new one is dropped with a warning. When the old name disappears, the partition fails as a rename. The fix: open the new alias from the first vintage with both names, close the old alias the month before, and add the old name to `ignored:`. Otherwise the old name is an alias used outside its range.
- **Fail closed (exit 2), as before:**
  - an expected column's header is missing. Usually it's a rename, and the new name is then listed as an unmapped header in the same error;
  - an alias is used outside its vintage range;
  - two aliases of one column are in one file.

  Nothing is written for that partition. Its problems, plus any new headers (a clue, e.g. a new column that splits rows), go to `02_unmapped_headers.csv`.

**Rename and change timeline (observed in the headers, confirmed by dictionary Table 16):**

| Vintage | Table(s) | Change |
|---|---|---|
| 2022-01 | provider_info | Added weekend staffing (total, RN), turnover (total, RN, with footnotes), administrator departures |
| 2022-07 | provider_info | Removed `RN Staffing Rating` |
| 2022-08 | provider_info | Added adjusted weekend total staffing |
| 2023-01 | health_citations | Added `Citation under IDR`, `Citation under IIDR` |
| **2023-06** | all 6 facility-level tables | `Federal Provider Number` → `CMS Certification Number (CCN)`; `Provider City/State/Zip Code` → `City/Town`, `State`, `ZIP Code`. provider_info only: `Provider County Name` → `County/Parish`, `Provider Phone Number` → `Telephone Number`; added `Affiliated Entity ID/Name`. |
| 2023-08 | provider_info | Added `Latitude`, `Longitude`, `Geocoding Footnote` |
| 2024-07 | provider_info | Added case-mix columns |
| **2025-07** | provider_info | `Affiliated Entity ID/Name` → `Chain ID/Name`; chain averages added; `Rating Cycle 2 …` → `Rating Cycle 2/3 …`; `Rating Cycle 3 …` dropped |
| 2025-10 | provider_info | Added `Urban` |
| 2026-02 | provider_info | Removed facility-reported incidents and substantiated complaints |
| 2026-06 | penalties | Added `Fine ID` |

There are no legacy names such as `PROVNUM` in 2021+ files; they only exist in L1 zips, which are out of scope.

### 3.3 Keep-lists (`configs/keep_columns.yaml`; node `silver`)
Keep only these canonical columns. Everything else (street address, phone, `location`, descriptions duplicated by the lookup, rating-cycle detail, provider-level fine counts, etc.) is dropped. Every column also keeps `vintage_month` and `cutoff_date`.

**Provider Information:**
`ccn, state, county_name, ownership_type, certified_beds, avg_residents_per_day, avg_residents_per_day_fn, provider_type, provider_in_hospital, date_first_approved, ccrc, special_focus_status, abuse_icon, most_recent_inspection_over_2y, changed_ownership_12m, resident_family_council, sprinkler_status, overall_rating, overall_rating_fn, health_inspection_rating, health_inspection_rating_fn, qm_rating, qm_rating_fn, staffing_rating, staffing_rating_fn, reported_staffing_fn, pt_staffing_fn, hprd_aide, hprd_lpn, hprd_rn, hprd_licensed, hprd_total, hprd_weekend_total, hprd_weekend_rn, hprd_pt, adj_hprd_aide, adj_hprd_lpn, adj_hprd_rn, adj_hprd_total, turnover_total, turnover_total_fn, turnover_rn, turnover_rn_fn, admin_departures, admin_departures_fn, chain_id, chain_name`

**Product identity only:** `provider_name, provider_address, city, zip, latitude, longitude` (+ `ccn, state, chain_id, chain_name`). They are written to a **separate** table, `silver/provider_identity_by_vintage`, so feature code cannot read them by accident; it feeds `dim_facility_latest`.

Columns that exist only in later vintages are null in earlier ones; their availability is recorded (§3.5):
- turnover and weekend staffing: from 2022-01;
- chain: from 2023-06, under `Affiliated Entity *` until 2025-06 and `Chain *` from 2025-07;
- lat/long: from 2023-08.

**Health Deficiencies:**
`ccn, survey_date, deficiency_prefix, deficiency_category, deficiency_tag, ftag, scope_severity, deficiency_corrected, correction_date, is_standard, is_complaint, is_infection_control, inspection_cycle, citation_under_idr, citation_under_iidr`
(The last three are extras pending decision §14 #3; IDR/IIDR exist from 2023-01.)

**Inspection Dates:** `ccn, survey_date, survey_type, survey_cycle`

**Penalties:** `ccn, penalty_date, penalty_type, fine_amount, payment_denial_start_date, payment_denial_length_days, fine_id, dup_seq, penalty_event_key` (`fine_id` exists only from 2026-06, and only for fines).

**MDS Quality Measures:** `ccn, measure_code, measure_description, resident_type, four_quarter_avg, four_quarter_avg_fn, used_in_5star, measure_period, measure_period_start, measure_period_end`

**Medicare Claims Quality Measures:** `ccn, measure_code, measure_description, resident_type, adjusted_score, observed_score, expected_score, measure_fn, used_in_5star, measure_period, measure_period_start, measure_period_end`

**Citation Code Look-up (all vintages):** `deficiency_prefix, deficiency_tag, deficiency_prefix_and_number, deficiency_description, deficiency_category`

### 3.4 Type and value normalisation (node `silver`)
| Field | Rule |
|---|---|
| `ccn` | Strip, upper-case. Assert length 6 and `^[0-9A-Z]{6}$` (not `^\d{6}$`). Never pad numbers that lost zeros; the source never loses them. |
| Key codes (`code`) | Trim; empty → null. For identifiers that are part of a key (`deficiency_prefix`, `penalty_type`, `measure_code`), so a stray space in one vintage can't fork a key. Names and descriptions (`str`) keep the raw value. |
| Dates | All dates in 2021+ files are ISO `YYYY-MM-DD`. Parse that format only; anything unparseable → null and counted (expected 0). |
| `processing_date` | One value per file, equal to the 1st of `vintage_month`. Assert it. |
| `scope_severity` | Single upper-case letter A–L (only B–L occurs). Anything else → null and counted. |
| `deficiency_tag` | Keep the 4-digit **text** tag (`0880`). `ftag = prefix + tag`, e.g. `F0880`. |
| Citation survey type | The `Survey Type` column is always "Health" and is ignored. Keep the three booleans `is_standard`, `is_complaint`, `is_infection_control`; one citation can carry several. |
| `survey_type` (Inspection Dates) | Enum from `value_maps.yaml`: `Health Standard` → `HEALTH_STANDARD`, `Health Complaint` → `HEALTH_COMPLAINT`, `Infection Control` → `INFECTION_CONTROL`, `Fire Safety Standard` → `FIRE_STANDARD`, `Fire Safety Complaint` → `FIRE_COMPLAINT`, anything else → `OTHER` (counted). |
| Y/N flags | `Y`/`N` → boolean. Anything else → null and counted. |
| Numeric | Float. Empty → null; `nan`/`inf` → null and counted. Keep the footnote column, which holds the missing reason. |
| `sprinkler_status` | `Yes` / `Partial` / `No` → enum; `Data Not Available` → null + flag (the only text sentinel in the data). |
| `ownership_type` | Keep the raw value; add `ownership_group` ∈ {for_profit, non_profit, government} (the prefix before ` - `). |
| `special_focus_status` | `SFF` → `SFF`, `SFF Candidate` → `SFF_CANDIDATE`, blank → `NONE`. |
| `resident_family_council` | `Resident` / `Family` / `Both` → enum; the literal `None` and blank → `NONE`. |
| `measure_period` | MDS `YYYYQn-YYYYQn` and claims `YYYYMMDD-YYYYMMDD` → `measure_period_start` / `_end` dates. |
| Footnotes | Keep the code as text; group via `footnotes.yaml` into missing-reason categories in gold. Code 12 was retired in 2025-01 (replaced by 23–25). |
| `deficiency_corrected` | Keep the raw value. About 97% of citations are `Deficient, Provider has date of correction`, the corrected state. "Not corrected" is a config list (`value_maps.yaml: not_corrected_values`; §14 #4). |

### 3.5 Silver stacked tables (node `silver`)
Write one stacked table per CMS table, partitioned by vintage:
`silver/provider_by_vintage`, `silver/citations_by_vintage`, `silver/surveys_by_vintage`, `silver/penalties_by_vintage`, `silver/mds_qm_by_vintage`, `silver/claims_qm_by_vintage`, `silver/citation_lookup`, plus `silver/provider_identity_by_vintage` (§3.3).

- Each has the keep-list columns plus `vintage_month` and `cutoff_date`.
- Rows are sorted by the table key, so the files are deterministic.
- **Citation lookup versions:** also write `silver/citation_lookup_versions`, one row per distinct `(prefix, tag, number, description, category)` with `valid_from_vintage` / `valid_to_vintage`. Its content changes only 7 times in 61 vintages (2021-04, 2021-06, 2021-07, 2021-11, 2022-06, 2023-01, 2025-04), so 38k rows reduce to about 660.

Also write `reports/step2/02_column_availability.csv`: for each (table, column), the first and last vintage where its non-null rate exceeds 50%, plus `max_non_null_rate`. Footnote and payment-denial columns are mostly null by design and show no 50% range.

**Within-vintage keys** (measured: 0 duplicates on the whole archive):

| Table | Key |
|---|---|
| `provider_by_vintage` | `(ccn, vintage_month)` |
| `citations_by_vintage` | `(ccn, survey_date, deficiency_prefix, deficiency_tag, vintage_month)` |
| `surveys_by_vintage` | `(ccn, survey_date, survey_type, vintage_month)` |
| `mds_qm_by_vintage`, `claims_qm_by_vintage` | `(ccn, measure_code, vintage_month)` |
| `citation_lookup` | `(deficiency_prefix, deficiency_tag, vintage_month)` |

**Penalties (overrides guide v1 §2e):** "exact-duplicate" rows are **distinct fines**. From 2026-06, about 540 fines a month are identical except `Fine ID`, and each has its own Fine ID; before Fine ID existed, such rows appear only in 2025-10 → 2026-05.
1. Keep every row.
2. `dup_seq` = the occurrence number within `(ccn, penalty_date, penalty_type, fine_amount, payment_denial_start_date)` in that vintage, starting at 1, in file order.
3. `penalty_event_key` = the 5 fields + `dup_seq`, **always**. `fine_id` is kept as an attribute.
   - **Don't switch the key to `fine_id`.** From 2026-06 the same fine would then get a new key, and the ledger would count it twice, with a false `first_seen_vintage` (sample run: 38 of 190 events).
   - Identical rows are interchangeable, so the 5 fields + `dup_seq` is stable across vintages.

**Events after the cutoff:** 217 rows in 3 vintages are dated up to 9 days after `processing_date` (survey dates 2025-10 and 2025-12; citations 2025-12). Keep them. The `event_date <= cutoff_date` feature filter and the `event_date > cutoff_date` label filter handle them.

Silver is rebuilt per vintage partition, and is idempotent.
- **Skip key:** a partition is skipped only if its latest `built` row has the same bronze `content_hash`, the same `zip_sha256` **and** the same hash of `column_map.yaml` + `keep_columns.yaml` + `value_maps.yaml`. A config change rebuilds silver, so a monthly run still equals a backfill.
- **Fail closed:** a partition with a missing expected header, an out-of-range alias, a normalisation failure, a wrong `Processing Date` or a duplicate key is written to **no layer path**. New headers alone don't fail it (§3.2). It gets a `silver_partitions` row with status `failed` (and the problems in `detail`); the node exits 2, and the next run retries it. *(Verified before decision §14 #25: with the `Fine ID` alias removed, the 3 penalty partitions 2026-06 → 08 failed and nothing was written; after the fix, RESUME built them.)*
- **Check over every bronze partition:** each has a `built` silver partition with current inputs (`every_bronze_partition_has_current_silver`).
- **`citation_lookup_versions`:** a version is a **contiguous run** of published vintages with identical content for `(prefix, tag)`. A change A → B → A gives three versions.

🛑 **CHECKPOINT 2. Report `02_harmonization.md`:**
- column map summary; header problems (must be zero); new headers dropped with a warning;
- availability table;
- uniqueness results (penalty duplicates reported and retained);
- normalisation null counts;
- encoding counts and blank rows;
- post-cutoff events;
- `stale_refresh` flags.

---

## 4. STEP 3: Event ledgers (node `silver`; all vintages unioned, used for labels and lag analysis)

A ledger records each real-world event **once**, together with when it was first and last visible. Ledgers are rebuilt over all vintages on every run (they are small). A `stale_refresh` vintage adds no new events and doesn't change `first_seen_vintage`.

### 4.1 `silver/ledger_citations`
Group `citations_by_vintage` by the citation key `(ccn, survey_date, deficiency_prefix, deficiency_tag)` and compute:
- `first_seen_vintage`, `last_seen_vintage`, `n_vintages_seen`.
- `final_scope_severity`, `final_category`, `final_is_standard`, `final_is_complaint`, `final_is_infection_control`: values from the **last** vintage the citation appears in. This captures downgrades after disputes.
- `severity_changed`: true if the severity differs across vintages.
- `removed_by_cms`: decides whether a disappearance was a deletion or just normal aging. Let `v_next` be the published vintage right after `last_seen_vintage` (skip the never-published months).
  - It is `true` if `last_seen_vintage < V` **and** the facility is present in `v_next` **and** at least one *other* citation with the same `(ccn, survey_date)` is present in `v_next`.
  - When old surveys age out, all of their tags disappear together. A single tag vanishing while its siblings remain means CMS deleted it (for example after a dispute).

### 4.2 `silver/ledger_penalties`
Group `penalties_by_vintage` by `penalty_event_key` (the 5 fields + `dup_seq`, §3.5) and compute `first_seen_vintage`, `last_seen_vintage`, and the latest non-null `fine_id`.

`revision_group = (ccn, penalty_date, penalty_type)` collects revised amounts of the same penalty.
- For **binary** labels, existence per `revision_group` is enough.
- For **dollar** sums, take the amounts from the latest vintage in which that `revision_group` appears.

### 4.3 `silver/ledger_surveys`
Union `surveys_by_vintage` with the distinct `(ccn, survey_date)` pairs from `ledger_citations`, typed from the citation flags. Key: `(ccn, survey_date, survey_type)`. Keep `first_seen_vintage`.

Inspection Dates lists complaint surveys **only if they produced a citation**, so survey counts for complaints mean *cited* complaint surveys.

### 4.4 `silver/facility_presence`
For each CCN: `first_vintage`, `last_vintage`, `n_vintages`, `gap_vintages` (published vintages missing in between). A facility is treated as **exited** if `last_vintage < V`.

---

## 5. STEP 4: Measure posting lag → set maturity buffers (node `silver`, calibration task; backfill or on demand)

Measure how long after the event date a record first appears. Use only events dated **on or after 2021-06** (citations) and **2021-12** (penalties); earlier events are left-censored by the archive start.
- Until the 2021-11 vintage, CMS posted fines only once paid. Measured: with penalty events from 2021-12, fines B = 10 and denials B = 12, so `B_pen` stays 12.

For each event type (citations by survey type, **G+ citations** (what labels use), penalties by type):
- **The penalty event is the revision group `(ccn, penalty_date, penalty_type)`**, not the row key. CMS revises fine amounts (23k of 37k fines show more than one amount), so a key containing `fine_amount` dates a fine from its last amount and inflates the lag (B = 20 instead of 9–12).

1. `lag_months = months_between(event_date, cutoff_date(first_seen_vintage))` in calendar months. Also report published-vintage steps, because a never-published month adds a month of apparent lag.
2. Report the empirical CDF \(c(a) = P(\text{lag} \le a)\) for \(a = 0..24\), plus the median and the 90th/95th percentiles.
3. Proposed buffer: \(B = \min\{a : c(a) \ge 0.95\}\).

Only events whose survey date is at least 24 months before V can have a fully observed lag. Restrict the CDF to those events, so recent ones don't bias it.

**Cross-check (single latest vintage).** For surveys grouped by age bucket (0–1, 1–2, … 18 months before V), compute:
- `obs_fine_share` = share of surveys with a penalty on the same date;
- `exp_fine_share` = p₁·p_G + p₀·(1 − p_G), using the mature-period conditional rates you measure yourself (about 0.73 and 0.025 in earlier analysis).

`obs/exp` should rise to about 1 at roughly `B_pen`.

**Caveat (measured):** on the 2026-08 vintage, obs/exp reaches 1 only at 17+ months. That is a **cohort effect**: fine rates for recent surveys are falling (p0 dropped from 0.25 to 0.03 between the 2023-08 and 2026-08 vintages), and the same survey cohorts barely change in later vintages. Use the cross-check as a sanity plot, not to set `B_pen`. Mature-period rates: p1 = 0.735, p0 = 0.028.

**Measured on the full archive** (events 2021-06 → 2024-08; calendar months):

| Event | n | median | p95 | B |
|---|---|---|---|---|
| G+ citations | 24,066 | 3 | 7 (c(7) = 0.951) | **8** chosen: complaint-linked G+, 72% of G+, needs 8 |
| All citations | 357,013 | 3 | 6 | 6 |
| Fines (revision group) | 36,870 | 4 | 9 | 9 |
| Payment denials | 4,290 | 5 | 12 | 12 |

→ **`B_cit = 8`, `B_pen = 12`.** Fines dated in 2024H2–2025H1 posted more slowly, so re-measure when they are fully observable (2027).

The calibration task writes the proposed `B_cit` and `B_pen` to the Checkpoint 3 report. A human commits them to `configs/labels.yaml`. Monthly runs read the buffers from config and do not re-measure.

🛑 **CHECKPOINT 3. Report `04_lag_analysis.md`:** lag CDFs, proposed buffers, the cross-check, and the resulting last labelable t for each label (§9.3).

---

## 6. STEP 5: Facility universe, product dimension, quality-measure selection (node `silver`; 5.3 is a calibration task)

1. **`silver/facility_universe`:** `(ccn, vintage_month)` for every CCN present in `provider_by_vintage`. This is the population at each t.
2. **`gold/dim_facility_latest`:** for each CCN, its name, address, city, state, zip, latitude, longitude, chain_id and chain_name from its **last** vintage, plus `is_active = (last_vintage == V)`. Product display only; written by the silver node because it needs no features.
3. **Quality-measure inventory and selection.**
   - Write `silver/qm_inventory` + `05_qm_inventory.csv`: every `(table, measure_code, resident_type, measure_description)` with its first and last vintage.
   - **Known timeline:**

     | Codes | Status |
     |---|---|
     | 401, 404, 406–410, 415, 451, 452, 454 (LS); 430, 434, 472 (SS); claims 521, 522, 551, 552 | Stable over all 61 vintages. 410 = falls with major injury; 434 = new antipsychotic (SS). 451's description changed from "move" to "walk independently" (2025-01, same code). |
     | 405 → 480 (LS incontinence), 453 → 479 (LS pressure ulcers) | Old code to 2024-11, new code from 2025-02. Bridge candidates. |
     | **419 → 481** (LS antipsychotic) | 419 to 2025-12, 481 from 2026-02. **Bridge needed** (guide v1 wrongly expected it stable). |
     | 471 (SS function improvement) | Dropped after 2024-11 |
   - **Freshness (measured on the full archive; corrects STEP2_HANDOFF §0 #10 and §8):** the MDS period was **not** frozen. Most measures advance one quarter per quarter (2024-08: `2023Q2-2024Q1`; 2025-07: `2024Q2-2025Q1`), but the periods differ per measure:
     - the annual flu-vaccine measures 454 and 472 stayed at `2023Q3-2024Q2` from 2024-10 to 2025-09; they are dropped from the features anyway;
     - in 2024, the function measures (401, 405, 451, 453, 471) lagged 2–3 quarters behind the rest.

     Keep `measure_period_start` / `_end` per row, so Step 7 can compute `qm_age_months`, and the change features can tell when a value did not refresh.
   - **Propose `configs/qm_measures.yaml`** (the calibration task writes the proposal to the report; a human commits the config):
     - **stable:** measures present in every vintage from the spine history start (2022-01) → V;
     - **bridged:** old and new codes mapped to one canonical feature (e.g. `ls_pressure_ulcer`), with `qm_def_post_2025_02` and `qm_def_post_2026_02` indicators. **Propose a bridge only where the definitions are close** (guide v1 rule):
       - 453 → 479 (pressure ulcers) and 419 → 481 (antipsychotic): close;
       - **405 → 480 is doubtful:** "low-risk residents who lose control of bowels/bladder" vs "new or worsened incontinence". Decide at Checkpoint 4;
     - **dropped:** vaccination process measures (415, 430, 454, 472), 471 (ended 2024-11), and anything not stable or bridged.

🛑 **CHECKPOINT 4. Report `05_universe_and_qm.md`:**
- facility counts per vintage (expect about 14.5k–15.4k on the full data);
- the count of exited facilities;
- the proposed QM list, with the bridged ones called out.

Wait for approval of `qm_measures.yaml`.

---

## 7. STEP 6: The spine (node `gold`)

`gold/spine`: one row per `(ccn, t)` for every published `t` in **2023-01 → V** and every CCN in `facility_universe` at vintage `t`.

Columns: `ccn, t (= vintage_month), cutoff_date`.

Assert:
- `(ccn, t)` is unique.
- Every published `t` in range is present. The never-published months (2023-12, 2024-12, 2025-01, 2025-08, 2026-01 in range) simply have no rows.

**Every feature table and the label table must have exactly one row per spine row.** All later joins are `spine LEFT JOIN table USING (ccn, t)`. After each join, assert the row count is unchanged and `(ccn, t)` is still unique.

---

## 8. STEP 7: Feature tables (node `gold`; each keyed by `(ccn, t)`)

General rules:
- Each family is its own table under `gold/features_<family>`, partitioned by t.
- **Inputs (whitelist):** **only** the silver `*_by_vintage` tables, and `citation_lookup` (the per-vintage partitions), filtered to `vintage_month == t`. Change features (Δ) may also read the t−6 and t−12 vintages.
  - **Forbidden in feature code:** the ledgers, `facility_presence`, `facility_universe` beyond t, `citation_lookup_versions` (its `valid_to` is in the future), `qm_inventory`, `provider_identity_by_vintage`, and `gold/dim_facility_latest` (`is_active` and the chain are as of V).
  - **Enforced:** the feature reader records every read and refuses anything else. ML training must never join `dim_facility_latest`; add a check in the ML layer.
- **Event windows** are measured backwards from `cutoff_date`: "last 12m" means `cutoff_date − 12 months < event_date <= cutoff_date`.
- **Zero** means "none found in the published data". It does **not** mean the facility is good.
- **Missing values:**
  - left as null, with the missing reason (from footnote codes) in a `_fn` categorical; don't impute, the model handles nulls;
  - if the t−6 or t−12 vintage was never published, or the facility is absent from it, the Δ feature is null.
- **Registry:** every feature is registered in `configs/feature_registry.yaml` with `name, family, source_table, formula (one line), first_available_vintage, dtype, model_use ∈ {model, ablation_only, product_only}`.

### 8.1 Severity weights (Five-Star health-inspection points; confirm against the CMS Technical Users' Guide and record the source in config)
| Letter | A | B | C | D | E | F | G | H | I | J | K | L |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Points | 0 | 0 | 0 | 4 | 8 | 16 | 20 | 35 | 45 | 50 | 100 | 150 |

- Use the base points only; ignore the substandard-quality-of-care uplift.
- Severity bands: `AC` = A–C, `DF` = D–F, `GI` = G–I, `JL` = J–L.
- Ordinal severity: A = 1 … L = 12.

### 8.2 `features_survey_timing` (source: `surveys_by_vintage` at t; health survey types only)
| Feature | Definition |
|---|---|
| `days_since_last_std` | `cutoff_date` minus the latest HEALTH_STANDARD survey date (null if none) |
| `days_to_std_due` | `456 − days_since_last_std` (negative = overdue) |
| `std_overdue` | `days_since_last_std > 456` |
| `n_std_36m` | Number of HEALTH_STANDARD surveys in the last 36 months |
| `n_complaint_surveys_cited_12m`, `_24m` | Number of HEALTH_COMPLAINT survey dates (cited complaint surveys only, §4.3) |
| `n_ic_surveys_12m` | Number of INFECTION_CONTROL surveys |
| `days_since_last_complaint_survey_cited` | As named (null if none) |
| `no_std_on_record` | Boolean |

### 8.3 `features_citations` (source: `citations_by_vintage` at t, `survey_date <= cutoff_date`)
| Feature | Definition |
|---|---|
| `n_cit_12m`, `n_cit_24m` | Citation count |
| `n_cit_{AC,DF,GI,JL}_24m` | Count by band |
| `sev_points_12m`, `sev_points_24m` | Sum of weights (§8.1) |
| `sev_points_per_bed_24m` | `sev_points_24m / certified_beds` (beds from `provider_by_vintage` at t) |
| `max_sev_last_std`, `max_sev_prev_std` | The last and previous standard surveys are the latest two `HEALTH_STANDARD` dates in `surveys_by_vintage` at t (≤ cutoff). Value = max ordinal severity over citations with `is_standard` on that date; 0 if that survey has none; null if there is no such survey. |
| `gplus_last_std`, `gplus_prev_std` | `max_sev_* ≥ 7` (G) |
| `n_gplus_surveys_36m` | Distinct survey dates with any G+ |
| `days_since_last_gplus` | Null if none |
| `n_complaint_cit_12m`, `n_complaint_gplus_24m` | Citations with `is_complaint` |
| `n_ic_cit_24m` | Citations in the infection-control category **or** with `is_infection_control` |
| `n_cat_<k>_24m` | Count for categories: abuse/neglect/exploitation; quality of life & care; resident assessment & care planning; nursing & physician services; resident rights; administration (category names from `citation_lookup` at t, or from `deficiency_category`) |
| `n_repeat_tags_last_std` | F-tags cited (`is_standard`) on both the last and the previous standard survey; null unless both surveys have standard citations |
| `n_uncorrected` | Citations from the last 24 months whose `deficiency_corrected` is "not corrected" in vintage t. The value set is **open (§14 #4)**; default `Deficient, Provider has plan of correction` and `Deficient, Provider has no plan of correction`. |

### 8.4 `features_enforcement` (source: `penalties_by_vintage` at t, `penalty_date <= cutoff_date`)
| Feature | Definition |
|---|---|
| `n_fines_24m`, `fine_dollars_24m` (log1p), `max_fine_24m` | Fines (every row counts, including `dup_seq > 1`) |
| `n_denials_24m`, `denial_days_24m` | Payment denials |
| `any_penalty_36m` | Boolean |
| `days_since_last_penalty` | Null if none |

- Don't use the Provider Info columns "Number of Fines" or "Total Penalties"; they are dropped in Step 2, so there is one definition.
- Fine counts step up from 2025-10 (more CMPs per survey), so the `era` feature (§8.7) must be available to the model.

### 8.5 `features_facility_staffing` (source: `provider_by_vintage` at t, and at t−6 and t−12 for Δ)
| Feature | Definition / note |
|---|---|
| `certified_beds`, `occupancy` | `occupancy` = `avg_residents_per_day / certified_beds`, capped at 1.5 |
| `ownership_group`, `provider_type`, `provider_in_hospital`, `ccrc`, `state` | Categorical |
| `facility_age_years` | From `date_first_approved` to `cutoff_date` |
| `sff_status`, `abuse_icon`, `changed_ownership_12m`, `most_recent_inspection_over_2y` | Status flags |
| `resident_family_council`, `sprinkler_status` | Categorical |
| `hprd_{aide,lpn,rn,total,weekend_total,weekend_rn,pt}`, `adj_hprd_{rn,total}` | Staffing hours per resident day |
| `rn_share` | `hprd_rn / hprd_total` |
| `weekend_gap` | `hprd_total − hprd_weekend_total` |
| `turnover_total`, `turnover_rn`, `admin_departures` | With their `_fn` columns |
| `d6_hprd_total`, `d6_hprd_rn`, `d12_hprd_total`, `d12_turnover_total` | Value at t minus value at t−6 / t−12 (null if that vintage is unpublished or the facility is absent) |
| `staffing_rating`, `qm_rating` | `model` |
| `health_inspection_rating`, `overall_rating` | **`ablation_only`**. Built from citations, so redundant with §8.3; CMS changed the method in 2025-07. |
| `chain_id` | **`product_only`**. Exists only from 2023-06; never a model feature. |

### 8.6 `features_quality` (source: MDS and Claims at t; measures from `qm_measures.yaml`)
For each selected (or bridged) measure `m`:
- `qm_<m>`: the 4-quarter average (MDS) or the adjusted score (claims);
- `qm_<m>_fn`: footnote category.

Also:
- `d6_qm_<m>` = value at t minus value at t−6, for falls with major injury, pressure ulcers and antipsychotics (long stay), and short-stay rehospitalisation. It is **null when the measure code differs between t−6 and t** (a bridge such as 419 → 481), which would compare two definitions, **and null when `measure_period_end` is the same at t and t−6** (the value was not refreshed, as with the annual measures), where 0 would be a false "no change". This rule never fired for the selected measures in the sample; it is defensive;
- `qm_missing_count`;
- `qm_age_months` = months from `measure_period_end` to `cutoff_date` (MDS and claims separately);
- the definition indicators `qm_def_post_2025_02`, `qm_def_post_2026_02`.

### 8.7 `features_context` (computed from the facility × t tables above, within each t)
| Feature | Definition |
|---|---|
| `state_gplus_rate_12m` | Share of facilities in the same state, at t, with any G+ survey in the last 12 months. A proxy for how strict that state's inspectors are. |
| `state_median_hprd_total`, `rel_hprd_total` | Facility ÷ state median |
| `rel_sev_points_24m` | Facility − state median |
| `era` | `pre_2025_07` / `2025_07_plus` (rating-method change), plus `penalty_regime_2025_10` (boolean) |

---

## 9. STEP 8: Labels and splits (node `gold`; `gold/labels`, keyed by `(ccn, t)`)

### 9.1 Definitions
Let `W_H = (cutoff_date, cutoff_date + H months]`.

| Column | = 1 if … | Source |
|---|---|---|
| `y_g12` | A citation exists in `ledger_citations` with matching `ccn`, `survey_date ∈ W_12`, `final_scope_severity ∈ {G..L}` and `removed_by_cms = false` | Citations ledger |
| `y_g6` | Same with `W_6` | Citations ledger |
| `y_pen12` | A `revision_group` exists in `ledger_penalties` with `penalty_date ∈ W_12` (fine or payment denial) | Penalties ledger |

Labels include citations from **all** health survey types. A label equals 0 only when its status is `negative` (§9.2).

### 9.2 Label status (one column per label: `status_g12`, `status_g6`, `status_pen12`)
Evaluate in this order (**changed from guide v1**, which put `positive` first):
1. `immature`: `t + H + B > V` (in months), with `B = B_cit` for G labels and `B_pen` for penalty labels. The window is not yet fully observed; these rows are **scoring-only**, and `y` is null **even if an event is already visible**.
   - In an immature window, positives are visible but negatives are not. Labelling the visible positives would bias every base rate over those months; the sample run showed `y_g12 = 1.00` for all score months until this was fixed (43 rows).
2. `positive`: the label condition is met. This holds even if the facility exits later.
3. `censored`: no positive event was seen, and the facility stopped being observable before the window **and its buffer** had passed: `last_vintage < t + H + B` (`labels.yaml: censor_requires_buffer: true`). Excluded from training; count them.
   - Guide v1 used `last_vintage < t + H`. That calls a facility negative when it closed shortly after the window, although its late-posted citations were never seen.
   - On the sample, the strict rule moves 11 `y_g12` rows (3 facilities) and 28 `y_pen12` rows from negative to censored.
4. `negative`: everything else.

Training uses only the `positive` and `negative` rows of the label being trained. Labels are **recomputed every run**: each new release matures older rows and can add new positives.
- `status_pen6` / `y_pen6` follow the same rules, with H = 6 and `B_pen`.
- If V is a `stale_refresh` vintage, its event tables add nothing new, so maturity is overstated by a month. Count `t + H + B > V_last_non_stale` as immature (known case: 2025-09).

### 9.3 Last labelable month
`last_labeled_t = V − (H + B)` months.

Worked example, V = 2026-08, with the measured buffers `B_cit = 8`, `B_pen = 12`:

| Label | H | B | Last labeled t |
|---|---|---|---|
| `y_g12` | 12 | 8 | 2024-12 (unpublished; the last real month is 2024-11) |
| `y_g6` | 6 | 8 | 2025-06 |
| `y_pen12` | 12 | 12 | 2024-08 |

Each new monthly release moves these forward by one month (by the 2026-12-01 demo, V ≈ 2026-11, the late-November release).

### 9.4 Auxiliary label columns (analysis only; `model_use = label_aux`; never features)
`y_ij12` (J–L only), `n_gplus_surveys_12m_fwd`, `y_g12_std` (from a citation with `is_standard`), `y_g12_complaint`, `pen_dollars_12m_fwd`, `y_pen6`, `exited_within_12m`.

### 9.5 Splits (`configs/splits.yaml`; adds columns `split_g12`, `split_g6`, `split_pen12` to `gold/labels`)
All windows are **calendar months**: they end at `last_labeled_t`, even when that month was never published. The g12 test window 2024-10 → 12 at V = 2026-08 therefore has only 2 real months; the training counts in the table below are published months.
Label windows overlap, so every split leaves a **purge gap of H months** between the end of training and the start of test. Purge rows are not used at all.

**Rules** (recomputed whenever V changes; never keep the example values):
- `test` = the last `test_months` labelled months;
- `purge` = the H months before test;
- `calib` = the last `calib_months` (3) months before purge, held out from fitting;
- `train` = from `spine_start` (2023-01) up to calib;
- every row after `last_labeled_t` has `split = score`.

- **Shared windows (open, §14 #15):** guide v1 had `y_g12` and `y_pen12` share train and calib months, for a fair comparison. With `B_pen = 12` that would shrink `y_g12`'s window to `y_pen12`'s. The sample run computes the windows per label.
- **Calibration months (open, §14 #16):** calib sits directly after train, with no purge gap. On the sample, 29 of 33 calib positives share their qualifying G+ event with a train row of the same facility, so the calibrator sees nearly in-sample rows. Options: an H-month gap before calib (costs training months), or out-of-fold calibration inside train (time-ordered, facility-grouped folds; ML layer).

**Measured consequence (decision needed, §14 #10–12).** With the measured buffers, training months at V = 2026-08 (published months; calib = 3) are:

| Spine / training start | `test_months` | y_g12 train | y_g6 train | y_pen12 train |
|---|---|---|---|---|
| 2023-01 | 6 | **3** (2023-01 → 03) | 14 | **0** |
| 2023-01 | 3 | 6 (2023-01 → 06) | 17 | 2 |
| 2022-01 | 6 | 14 (2022-01 → 2023-03) | 25 | 11 |
| 2022-01 | 3 | 17 | 28 | 13 |

- **How the gate enforces this:** gold refuses to publish (exit 2) when a required label has no training month (`splits.yaml: required_labels`).
- **Sample run settings:** `test_months = 3`, required labels `g12` and `g6`; `y_pen12` reported only.
- **Recommendation:** keep scoring and evaluation months from 2023-01, but let **training rows start in 2022-01**. Their t−6/t−12 history (2021-07, 2021-01) exists in the archive, so every change feature is computable.
- **Fair comparison:** `y_g12` and `y_pen12` sharing train months would restrict `y_g12` to `y_pen12`'s much shorter window. Decide whether the comparison is worth that.

---

## 10. STEP 9: Assemble the gold training dataset (node `gold`)

1. Start from `gold/spine`.
2. Left-join, in order: `features_survey_timing`, `features_citations`, `features_enforcement`, `features_facility_staffing`, `features_quality`, `features_context`, `labels`.
3. After **each** join, assert:
   - the row count equals the spine's;
   - `(ccn, t)` is unique.
4. Write `gold/facility_month_dataset/t=<YYYY-MM>/part-0.parquet`.
5. Write `gold/dataset_manifest.json` with:
   - `dataset_version`, V, buffers, horizons, spine start;
   - the source of every **(table, vintage)**: zip name and `zip_sha256` (2026-07 comes from two zips). Also the `stale_refresh` and `revised_after_publication` flags;
   - the feature registry hash and the config hashes;
   - the git commit;
   - row counts per split per label;
   - the build timestamp (the only field excluded from determinism checks).

---

## 11. STEP 10: Verification, leakage tests and publish (node `gold`; all must pass; write `reports/step10/10_verification.md`)

**Gate:** gold builds into `gold/_staging/run=<run_id>/` and runs every check there.
- **All pass:** promote the staging outputs to `gold/`, then publish to the warehouse.
- **Any fail:** the node exits 2. `gold/` and the warehouse are untouched, and staging is kept for inspection.

Gold therefore only ever holds validated data. *(Verified: a forced gate failure left `gold/`, `silver/` and the warehouse byte-unchanged.)*

### 11.1 Structural
- [ ] `(ccn, t)` is unique. Every spine row has exactly one dataset row.
- [ ] Rows per t are within the `checks.yaml` range (full data: 14,000–15,600). Flag any month that changes by more than 2% from the previous one.
- [ ] Every feature in the registry exists, and no column exists outside the registry, the label list and the keys (`ccn`, `t`, `cutoff_date`).
- [ ] No feature has a non-null rate under 5% inside training months (unless it is listed in the registry with a reason).

### 11.2 Label logic
- [ ] **Nesting:** `y_g6 = 1 ⇒ y_g12 = 1`, and `y_ij12 = 1 ⇒ y_g12 = 1`, on rows where both are labelled. Zero violations.
- [ ] **Immaturity boundary:** the set of `immature` rows is exactly the rows with `t > V − H − B`.
- [ ] **Hand trace:** for random positive and negative rows of each label (20 + 20 on the full data), print the ledger events for that CCN within ±15 months of `cutoff_date` and confirm the label.
- [ ] **Strict future:** no event used by a label has a date ≤ `cutoff_date`. Check every label (`y_g12`, `y_g6`, `y_pen12`): each positive must have a qualifying event in `(cutoff, cutoff + H]`.
- [ ] **Sanity vs prior analysis** (full data only): among mature standard surveys from 2023-08 onward, G+ share ≈ 13%, IJ ≈ 4.7%; share of G+ surveys with a same-day penalty ≈ 73%. A large deviation means a parsing bug.
- [ ] Base rates per label, per split and per t, as a table. Explain any jump larger than 5 percentage points between consecutive months.

### 11.3 Leakage
- [ ] **Time-travel test:** pick 3 values of t (one each in train, test and score). Rebuild every feature table for those t from a store that contains **only vintages ≤ t**. Results must be identical to the stored features. Allowed difference: none. Two forms:
  - **In the gate (every run):** features rebuilt through a reader that raises on any vintage > t.
  - **Strong form (CI, and before freezing a training set):** an empty store; silver rebuilt from bronze ≤ t only; features from that store alone (`../sample_pipeline/verify_time_travel.py`). It also checks that the rebuilt silver is byte-identical.
- [ ] **Single-feature scan:** univariate PR-AUC of each feature against `y_g12` on training rows. Flag any feature above 0.6, or above 3× the base rate, for **human review at Checkpoint 5** (a flag, not a publish blocker).
- [ ] **Shuffled-label check:** a logistic regression on shuffled `y_g12` gives ROC-AUC between 0.48 and 0.52 (full data; small samples use a wider band from `checks.yaml`).
- [ ] No column from §9 appears in the feature registry with `model_use = model`.

### 11.4 Coverage
- [ ] Null rate by t for each feature. Late-arriving columns (e.g. turnover from 2022-01) are non-null from their documented start.
- [ ] The number of censored rows per label, and their share among exits.

### 11.5 Publish
After all checks pass:
- load `facility_month_dataset` to the warehouse table (BigQuery `precursorintelligence_gold.facility_month_dataset` via a load job, partitioned by month; Postgres `gold.facility_month_dataset` locally);
- record the build in the metadata store;
- emit the `cms_nh_gold` asset.

A `stale_refresh` release still publishes features for its t; the ML DAG decides whether to retrain (team plan: no new labels → no retrain).

🛑 **CHECKPOINT 5. Report `10_verification.md`:**
- pass/fail for each check;
- base-rate table;
- rows by split;
- any flagged features.

The dataset counts as **frozen** only after a human approves this checkpoint. Then tag it `training_dataset_vN` (the team's next one is `training_dataset_v2`). Monthly runs append to gold; a training set changes only when a retrain freezes a new version on purpose.

---

## 12. Definition of done
- `gold/facility_month_dataset` and `dataset_manifest.json` exist, pass every check in §11, and are published to the warehouse.
- `configs/` holds the final column map, keep-lists, value maps, footnotes, buffers, splits, QM selection, feature registry and check thresholds.
- `reports/` holds checkpoint reports 1–5.
- **Rebuild:** one command per node rebuilds everything from the landing zips deterministically. Running it twice gives byte-identical feature tables, and a run with no new data changes nothing.
- **Monthly equals backfill:** a monthly incremental run gives the same gold as a backfill that includes that month.

## 13. Things you must not do
- Do not use current (latest-vintage) values for historical rows. Feature code never reads ledgers.
- Do not derive the vintage from the zip file name when `processing_date` exists.
- Do not cast CCN to an integer, or validate it as digits only.
- Do not drop penalty rows that look like exact duplicates.
- Do not fill missing labels with 0. Immature and censored rows are excluded, not negative.
- Do not include `health_inspection_rating`, `overall_rating`, `chain_id` or any label/aux column as a model feature.
- Do not ingest tables listed as out of scope in §1 without asking.
- Do not load intermediate layers (bronze, silver) into BigQuery or Postgres.
- Do not continue past a 🛑 checkpoint without approval.

---

## 14. Decided and open decisions

**Decided by the user on 2026-10-15** (the team documents will be updated by the user):

| # | Topic | Decision | Rule in this spec |
|---|---|---|---|
| 2 | Duplicate data month: which zip wins | **Decided:** the later zip wins, since it carries corrections or more complete data | §2.1 Resolve. The 2026-07 Provider Info comes from the 2026-08-06 re-release. The team `cms-data-domain` skill's "prefer the complete zip" rule is superseded. |
| 6 | Raw-layer naming | **Decided:** `landing/` (raw, by zip date) + `bronze/` (Parquet, by vintage) | §0.6. The team plan's `bronze/pdc/vintage=…` raw layer is superseded. |
| 19 | DAG shape | **Decided:** one DAG, `cms_nh_pipeline`, with the 5 nodes as TaskGroups; it replaces the team plan's separate DAGs. **Amended 2026-10-29 (#24):** `check_release` moves to its own DAG. | §15. The lock is held from `acquire_lock` (ingestion) to `release_lock` (after gold). The Step 1 reference DAG `ingest_vintage` (STEP1 doc §12.6) is superseded; its task commands are reused unchanged. |
| 20 | `check_release` is optional | **Decided:** in production the pipeline runs only when CMS releases new data (`check_release` gates it). The node is **modular and skippable** for development, for repeated test runs locally or on cloud. **Amended 2026-10-29 (#24):** the param `run_check_release` is gone. A manual run of `cms_nh_pipeline` is the development path, and `cms_nh_release_check` stays paused during development. | §0.7, §15. Safe because every node is idempotent. |
| 21 | Exit codes 2 and 3 | **Decided:** fail without retries | §15.4: a thin `DockerOperator` subclass raises `AirflowFailException` on exit 2 or 3; exit 1 keeps its retries |
| 22 | Live CMS API in development and tests | **Decided:** allowed for read-only checks (the `network` tests, catalog and metastore reads). A real ingest of a new release (e.g. 2026-10-08) only when the user asks. | Unit tests stay offline (fake CMS) |
| 10 | Spine start | **Decided 2026-10-15:** 2023-01. **Revisit:** with the measured buffers it leaves 3 training months for `y_g12` and 0 for `y_pen12` (§9.5). | Recommended: training rows from 2022-01; scoring and evaluation from 2023-01 |
| 13 | Storage, metadata, warehouse, run lock, manifests | **Decided 2026-10-08** (D1–D11). **Metadata store amended 2026-10-16:** Postgres `precursor_meta` plus a BigQuery `meta` mirror and `status/latest.json`, no BigQuery adapter (D4, D8 and O2 in `STEP1_DATA_INGESTION_PIPELINE.md` §0 are superseded where they assume one) | §0.6, §15.3 |
| 23 | Lock release in the DAG | **Decided 2026-10-21:** `release_lock` is an Airflow teardown of `acquire_lock`, not `trigger_rule="all_done"` and not `on_failure_fail_dagrun=True` (both made failed runs show `success`, measured); it retries twice | §15.1, §15.3, §15.5 |

**Decided by the team on 2026-10-29** (VM deployment review):

| # | Topic | Decision | Rule in this spec |
|---|---|---|---|
| 24 | Release check in its own DAG | **Decided:** `cms_nh_release_check` (hourly, `catchup=False`, created paused) runs `check_release` and, on READY, RESUME or CHANGED, emits the asset `cms_nh_release`. `cms_nh_pipeline` is scheduled on that asset and can always be triggered by hand. During development and testing, the release check stays paused and the pipeline runs only when triggered manually. | §0.7, §2.0, §15. A check while a run holds the lock decides `BUSY` (skip). |
| 25 | Schema drift | **Decided:** additive drift is handled automatically. New headers, with every expected column present, mean the partition builds without them, with a warning and an email. A missing expected column (often a rename), an alias out of range, or two aliases still fail closed (exit 2). | §3.2; silver code (issue #10 follow-up) |
| 26 | VM size | **Decided:** e2-standard-4 (16 GB); the machine type is a Terraform variable. The team documents say e2-standard-2. | §0.8; `AIRFLOW_PIPELINE.md` §9.1 (gold's peak memory) |
| 27 | Metadata DB and VM uptime | **Decided:** Postgres (databases `airflow`, `precursor_meta`, later `mlflow`) stays on the VM; Cloud SQL was considered and rejected. Both databases are dumped (`pg_dump`) to GCS at every VM shutdown; a restore is tested before the first full run. The VM is started and stopped by hand and shuts itself down after 60 idle minutes (no DAG run, no SSH session); there is no instance schedule. | §0.6, §0.8 |

**Measured or changed by the sample run:**

| # | Topic | Status | Rule in this spec |
|---|---|---|---|
| 12 | Buffers `B_cit`, `B_pen` | **Measured** (full archive) | 8 / 12; re-measure in 2027 |
| 17 | Label status order | **Changed** (verifier finding) | `immature` before `positive` (§9.2) |
| 14 | Censoring rule | Recommended | Strict: observable until `t + H + B` (§9.2) |
| 1 | Penalty exact duplicates and keys | Recommended, needs team sign-off | Keep with `dup_seq`. Row key = the 5 fields + `dup_seq`, never `fine_id` (38 of 190 sample events would split in 2026-06). The event for labels and lag = revision group `(ccn, penalty_date, type)`, because fine amounts get revised. |
| 5 | Citation lookup | Recommended | Per-vintage partitions + `citation_lookup_versions` (contiguous runs); content changed 7 times in 61 vintages |

**Still open (team):**

| # | Topic | Default in this spec | Recommendation |
|---|---|---|---|
| 9 | `y_pen12` role | Built as a label (model B) | Decide whether it stays a model or only feeds features |
| 11 | `test_months` | 3 | 6 only if training starts in 2022-01 (#10) |
| 15 | Shared train window for `y_g12` / `y_pen12` | Not shared | Sharing would shrink `y_g12`'s window to `y_pen12`'s |
| 16 | Purge before calibration | None | Out-of-fold calibration inside train (ML layer) |
| 18 | Republished history | The latest version replaces history; `revised_after_publication` is flagged | Keep it, unless strict "as first published" backtests are needed |
| 3 | Keep-list extras | `inspection_cycle`, IDR/IIDR, `survey_cycle` included | Add MDS quarterly scores and the case-mix index only for ablations |
| 4 | "Not corrected" values for `n_uncorrected` | The two "Deficient, Provider has (no) plan of correction" values | Decide on `Past Non-Compliance` and `No revisit needed` (2025-10+) |
| — | 405 → 480 QM bridge | Bridged | Doubtful definition match; decide at Checkpoint 4 (§6) |
| 7 | Region | — | `us-central1` for buckets and both BigQuery datasets |
| 8 | Metadata write pattern | Append-only + latest views (D5) | Keep |

---

## 15. The DAG: nodes, tasks and run modes (verified on the sample run)

### 15.1 Shape
Two Airflow 3 DAGs (decision §14 #24):
- **`cms_nh_release_check`** (the `check_release` node):
  - schedule hourly, `catchup=False`, `max_active_runs=1`, created paused;
  - one task `check_release`, then `@task.short_circuit` on the JSON's `proceed`, then a task with `outlets=[cms_nh_release]`.
  - It stays paused during development and testing.
- **`cms_nh_pipeline`** (ingestion → gold):
  - schedule: the asset `cms_nh_release`, plus manual triggers; `max_active_runs=1`, `catchup=False`.
  - A manual run is a backfill or development run and never goes through `check_release`.
  - **Params:**

    | Param | Default | Meaning |
    |---|---|---|
    | `mode` | `monthly` | `monthly` or `backfill` |
    | `calibrate` | `false` | Run the silver calibration tasks; implied by `mode=backfill` |
    | `stop_after` | `null` | Node name, for the first backfill run node by node with approvals |
    | `entry_ids` | `""` | Development only: ingest just these catalog entries |

Each node is a `TaskGroup`. Every task is a `DockerOperator` running the `precursor-pipeline` image with **one CLI command**, and all tasks pass `--run-id {{ run_id }}`.

```
cms_nh_release_check (hourly):  check_release ──(short-circuit unless proceed)──▶ emit ──(cms_nh_release)

cms_nh_pipeline (on cms_nh_release, or manual):
                                                  ┌ ingestion ─────────────────────────────────────────────────────┐
                                                  │ acquire_lock → discover_download → pending_entries →           │
                                                  │ extract_inventory[mapped per zip] → resolve_report ──(cms_nh_landing)
                                                  └────────────────────────────────────────────────────────────────┘
                     ──▶ bronze: load_bronze ──(cms_nh_bronze)──▶ silver: build_silver [+ calibrate] ──(cms_nh_silver)──▶ validate_silver ──▶
                     ──▶ gold: build_gold (gate; publish) ──▶ validate_gold ──(cms_nh_gold)──▶ release_lock (teardown of acquire_lock, retries=2)
```

- The `data_pipeline` lock is taken by the first ingestion task and released by the last task, also on failure. Every task in between `require`s it.
- `release_lock` is an Airflow **teardown** of `acquire_lock` (`release.as_teardown(setups=acquire_lock)`, trigger rule `ALL_DONE_SETUP_SUCCESS`): it runs whenever the lock was taken, also after a failure or a `stop_after` skip, and not when `acquire_lock` failed (another run's lock).
  - **Not `trigger_rule="all_done"`:** Airflow sets a run's state from its leaf tasks, so an always-successful `all_done` release at the end would make every failed run show `success` (measured on the local stack: a run with an exit-3 failure showed `success`).
  - **Not `on_failure_fail_dagrun=True`** on the teardown either: that makes Airflow count it as a leaf again, with the same effect.
  - Other end-of-run tasks that must run after a failure (e.g. `mirror_metadata`) are teardowns of `acquire_lock` too, placed before `release_lock`.
- The ML DAGs (`train_evaluate`, `batch_score`, `monitor`) are scheduled on the `cms_nh_gold` asset, which is emitted only when gold publishes.

### 15.2 Every step belongs to exactly one node
| Node | Spec steps | Metadata tables written | Outputs |
|---|---|---|---|
| `check_release` | §2.0 release sensor | `vintage_registry` | decision JSON (last stdout line) |
| `ingestion` | §2.1 Step 1 | `catalog`, `source_datasets`, `download_log`, `extracted_files`, `inventory`, `anomalies`, `run_log`, `run_lock` | `landing/…`, Checkpoint 1 report |
| `bronze` | §3.1 Step 2a, content hash, release-level `stale_refresh` | `bronze_files`, `vintage_flags` | `bronze/<table>/vintage=…`, bronze manifest |
| `silver` | §3.2–3.5 Steps 2b–2e; §4 Step 3 ledgers; §6.1–6.2 universe and `dim_facility_latest`; QM inventory; **calibration** (backfill or on demand): §5 Step 4 lag, §6.3 QM proposal | `silver_partitions` | `silver/…`, `gold/dim_facility_latest`, Checkpoint 2–4 reports, silver manifest |
| `gold` | §7–§10 Steps 6–9; §11 Step 10 gate; §11.5 publish | `gold_builds` | `gold/…`, Checkpoint 5 report, warehouse table, gold manifest |

`dq_results` holds every named check of every node. So far only `precursor-pipeline validate` writes it, one row per anomaly (`docs/VALIDATION.md`).

### 15.3 Tasks
| Node | Task | Command (`precursor-pipeline …` in the repo; `./pi.sh …` in the sample) | Notes |
|---|---|---|---|
| check_release (DAG `cms_nh_release_check`) | `check_release` | `check-release --run-id R` | Then `@task.short_circuit` on the JSON's `proceed`. Decisions: READY, RESUME, CHANGED → proceed; PARTIAL, BUSY, NO_CHANGE → skip. |
| | `emit_release` | (Python, no container) | `outlets=[cms_nh_release]`: starts a `cms_nh_pipeline` run |
| ingestion | `acquire_lock` | `precursor-ingest lock acquire --run-id R` | Exit 1 while another run holds the lock (retried) |
| | `discover_download` | `precursor-ingest step1 --steps discover,download --run-id R` | |
| | `pending_entries` | `precursor-ingest entries --pending --run-id R` | Last stdout line = JSON list → XCom |
| | `extract_inventory` | `precursor-ingest step1 --steps extract,inventory --entry-ids <id> --run-id R` | Mapped, one per pending zip; 0 on a quiet day |
| | `resolve_report` | `precursor-ingest step1 --steps resolve,report --run-id R` | `trigger_rule=none_failed`; emits `cms_nh_landing` |
| bronze | `load_bronze` | `bronze --run-id R` | Refuses to start if any inventory row is `pending` (exit 2). Can later be mapped per (table, vintage). |
| silver | `build_silver` | `silver --run-id R [--calibrate]` | `--calibrate` in backfill (Checkpoints 3–4); monthly runs read buffers and QMs from config |
| silver | `validate_silver` | `validate --layer silver --run-id R [--all-months]` | Schema and drift checks (`docs/VALIDATION.md`); `--all-months` in backfill. Exit 2 on a schema violation: gold does not run. |
| gold | `build_gold` | `gold --run-id R` | Gate: any failed check → exit 2, nothing published. |
| gold | `validate_gold` | `validate --layer gold --run-id R` | Every gold month. Emits `cms_nh_gold` only when gold was published and valid; it runs after the publish, so a failure holds back the asset but cannot unpublish. |
| — | `mirror_metadata` | `precursor-pipeline mirror-metadata --run-id R` | **Not built yet.** Copies the `precursor_meta` tables to BigQuery `meta` with a load job (seconds; a few thousand rows). A teardown of `acquire_lock` placed before `release_lock` (not `trigger_rule=all_done`, §15.1), so a failed run is mirrored too; a failure here is retried but never blocks `release_lock`. |
| — | `release_lock` | `precursor-ingest lock release --run-id R` | teardown of `acquire_lock` (not `trigger_rule=all_done`, §15.1), `retries=2` (30 s apart; the release is owner-checked and idempotent) |

### 15.4 Run modes and what each node does (measured on the sample)
| Situation | check_release | ingestion | bronze | silver | gold |
|---|---|---|---|---|---|
| **Backfill** (manual run, `mode=backfill`) | not run | all in-scope zips | every selected file | every partition + ledgers + calibration | every t |
| **Development / test rerun** (manual run; `cms_nh_release_check` paused) | not run | only new or changed zips (no-op if none) | no-op if unchanged | no-op if unchanged | rebuilt; byte-identical if inputs are unchanged |
| **Bad input** (missing or renamed header, reconciliation) | — | — | failed file not written; exit 2 | failed partition not written; exit 2 | not reached; RESUME after the fix |
| **New CMS columns only** (additive, §3.2) | — | — | loaded as text, like every column | built without them; warning email; exit 0 | runs as usual |
| **Gate failure** | — | — | — | — | staging kept; `gold/` and the warehouse unchanged; exit 2; RESUME after the fix |
| **New complete release** | READY | only the new zips (sample: 2 downloaded, 65 skipped) | only the new vintage (7) | only the new partitions (7) + ledgers | features for every t (unchanged months are byte-identical) + **all labels recomputed** |
| **Previous run failed after ingestion** | RESUME | no-op | no-op | no-op (ledgers rebuilt identically) | builds and publishes |
| **CMS republished an old zip** | CHANGED (catalog size or new entry) | stored side by side as `<stem>__<sha8>.zip`; selected | reloads that vintage's 7 partitions (D10) | rebuilds them (unchanged content gives identical bytes) | only the affected months change (sample: t, t+6, t+12 staffing; t context) |
| **Partial re-post** (one dataset only) | PARTIAL | — | — | — | — |
| **Nothing new** | NO_CHANGE | — | — | — | — |
| **Re-run of the same data** | — | no-op | no-op | no-op | identical dataset hash |

**Exit codes:**

| Code | Meaning | Action |
|---|---|---|
| 0 | ok | — |
| 1 | error | Airflow retries (2 × 5 min); `lock_release` still runs |
| 2 | data or validation failure (missing or renamed header, reconciliation, gate) | **Fail without retry** (decision §14 #21): every task uses `PipelineDockerOperator`, a thin `DockerOperator` subclass that raises `AirflowFailException` on exit 2 or 3. Alert. After the fix, the next release check resumes (RESUME), or trigger `cms_nh_pipeline` by hand. |
| 3 | config | Fail without retry (same subclass); fix the config |

### 15.5 Reference DAG (sketch; same pattern as the parse-checked `ingest_vintage` in `STEP1_DATA_INGESTION_PIPELINE.md` §12.6)
```python
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.exceptions import AirflowFailException
from airflow.providers.docker.exceptions import DockerContainerFailedException
from airflow.sdk import Asset, dag, task, get_current_context
from datetime import datetime, timedelta
import json
import re

RELEASE, LANDING, BRONZE, SILVER, GOLD = (Asset(n) for n in (
    "cms_nh_release", "cms_nh_landing", "cms_nh_bronze", "cms_nh_silver", "cms_nh_gold"))

class PipelineDockerOperator(DockerOperator):
    """Exit 2 (data/validation) and 3 (config) fail at once: retrying cannot help (decision §14 #21)."""
    def execute(self, context):
        try:
            return super().execute(context)
        except DockerContainerFailedException as exc:      # raised by the provider on a non-zero exit
            # The message carries the container result, e.g. {'StatusCode': 2, ...}. VERIFY this against the
            # pinned provider version (apache-airflow-providers-docker 4.6.1) with a DAG test before relying on it.
            if re.search(r"'StatusCode': [23]\b", str(exc)):
                raise AirflowFailException(f"pipeline exit 2/3, not retried: {exc}") from exc
            raise

def run(task_id, *cmd, **kw):  # one CLI command per task; docker_kwargs as in §12.6 (image, network, PI_DATABASE_URL)
    return PipelineDockerOperator(task_id=task_id, command=[*cmd, "--env", "prod", "--run-id", "{{ run_id }}"],
                                  **docker_kwargs, **kw)

# The release check is its own DAG (decision §14 #24); paused while the project is being developed.
@dag(schedule="@hourly", start_date=datetime(2026, 10, 1), max_active_runs=1, catchup=False,
     is_paused_upon_creation=True)
def cms_nh_release_check():
    check = run("check_release", "precursor-pipeline", "check-release", do_xcom_push=True)

    @task.short_circuit                        # NO_CHANGE / PARTIAL / BUSY: nothing to do
    def proceed(raw: str) -> bool:
        return json.loads(raw)["proceed"]

    @task(outlets=[RELEASE])                   # the asset event starts cms_nh_pipeline
    def emit_release() -> None: ...

    proceed(check.output) >> emit_release()

# Runs on a release (the asset) or by hand: a manual run is a backfill or development run.
@dag(schedule=[RELEASE], start_date=datetime(2026, 10, 1), max_active_runs=1, catchup=False,
     params={"mode": "monthly", "calibrate": False, "stop_after": None},
     default_args={"retries": 2, "retry_delay": timedelta(minutes=5)})
def cms_nh_pipeline():
    lock = run("acquire_lock", "precursor-ingest", "lock", "acquire")
    fetch = run("discover_download", "precursor-ingest", "step1", "--steps", "discover,download")
    pending = run("pending_entries", "precursor-ingest", "entries", "--pending", do_xcom_push=True)
    per_zip = DockerOperator.partial(task_id="extract_inventory", max_active_tis_per_dagrun=3, **docker_kwargs).expand(
        command=mapped_commands(pending.output))                      # as in §12.6
    resolve = run("resolve_report", "precursor-ingest", "step1", "--steps", "resolve,report",
                  outlets=[LANDING], trigger_rule="none_failed")
    bronze = run("load_bronze", "precursor-pipeline", "bronze", outlets=[BRONZE])
    silver = run("build_silver", "precursor-pipeline", "silver",
                 "{{ '--calibrate' if params.calibrate or params.mode == 'backfill' else '--no-calibrate' }}",
                 outlets=[SILVER])
    validate_silver = run("validate_silver", "precursor-pipeline", "validate", "--layer", "silver",
                          "{{ '--all-months' if params.mode == 'backfill' else '' }}")
    gold = run("build_gold", "precursor-pipeline", "gold")
    validate_gold = run("validate_gold", "precursor-pipeline", "validate", "--layer", "gold", outlets=[GOLD])
    unlock = run("release_lock", "precursor-ingest", "lock", "release", retries=2,
                 retry_delay=timedelta(seconds=30))   # a teardown, wired below

    lock >> fetch >> pending
    per_zip >> resolve >> bronze >> silver >> validate_silver >> gold >> validate_gold >> unlock.as_teardown(setups=lock)  # not all_done: §15.1

cms_nh_release_check()
cms_nh_pipeline()
```
Before deploying it: the `precursor-pipeline` image, the local Airflow compose, `tests/dags`, and porting `../sample_pipeline/nodes/{check_release,bronze,silver,gold}.py` into `src/precursorintelligence/` behind CLI commands, all as listed in `STEP2_HANDOFF.md` §14.4.

### 15.6 Verified on the sample (details: `../sample_run_log.md`)
| Property | Result |
|---|---|
| Every node green, with independent verifiers recomputing from the raw CSVs | bronze 420/420 partitions exact; labels and features re-verified on the final state (see `../sample_run_log.md` §7) |
| Monthly incremental run == clean backfill | ✅ all 771 silver + gold Parquet files byte-identical |
| Re-run with no new data | ✅ no-op at every node; identical dataset hash |
| Strong time travel (store rebuilt with only vintages ≤ t, for 3 values of t) | ✅ identical features; identical silver |
| Republish (D10) | ✅ exactly the affected partitions and months changed |
| Gate failure → RESUME next day → publish | ✅ |
| Repo untouched | ✅ (checksums of all repo files) |

---

## 16. Known limitations
- **Provider-info status flags** (abuse icon, "inspection over 2 years", ratings) are computed by CMS at publication and can reflect surveys a few days after `cutoff_date`. No effect is visible in the sample; it can't be removed without CMS history.
- **Configs see the whole archive.** `qm_measures.yaml` and the buffers are chosen with data through V: a look-ahead in which columns exist, not in their values. Freeze configs per dataset version (their hashes are in the manifest).
- **The 2026-07 score row** uses the corrected 2026-08-06 Provider Info file, so it can't be reproduced as of 2026-07-29 (§14 #2).
- **`n_uncorrected`** uses the correction status as published in vintage t.
- **Exiting facilities** can be positive but not negative after they exit (they become censored), so closure-linked features lean positive. Report the censored share among exits (§11.4).
- **Gold recomputes the features for every t** each run. That is simple and deterministic (unchanged months come out byte-identical), and cheap: about 575k rows at V = 2026-08 (39 months × 14.7k). Make it incremental only if it becomes slow.
- **The 2024H2–2025H1 penalty cohorts** posted slowly; re-measure the buffers in 2027.
