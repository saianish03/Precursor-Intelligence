# Step 1: to-do backlog

These are improvements to the built Step 1 pipeline, borrowed from an earlier per-table ingestion design (2026-10-08 comparison).

**Status after the recheck against `DATA_PIPELINE_SPEC.md` (2026-10-16).** The spec wins where this file differs.

| Item | Status | What the spec changed |
|---|---|---|
| T1 release sensor | **Moved** to the `check_release` node (spec §2.0), outside Step 1 | Decisions READY / RESUME / CHANGED / PARTIAL / BUSY / NO_CHANGE. It runs in its own DAG, `cms_nh_release_check`, which starts `cms_nh_pipeline` through the asset `cms_nh_release`; manual pipeline runs skip it (spec §14 #24). Step 1 provides `inventory.landing_backlog()` for its CHANGED / RESUME inputs. |
| T2 metastore CSV fallback | Parked | The spec keeps one zip code path for history and new months (§0.8). Revisit only if the archive endpoints break. |
| T3 `--dry-run` | Exit codes **done**; `--dry-run` **dropped** | Development runs are manual runs of `cms_nh_pipeline` (no release check), and every node is idempotent, so a dry run adds little. |
| T4 drift check at ingest | **Narrowed** | Silver handles header drift (spec §3.2, amended 2026-10-29): new headers only warn and email, a missing or renamed expected header fails closed (exit 2). At most, add a report-only header-set check to Checkpoint 1, which already lists the header sets. |
| T5 GCS | Retention part **decided**; race open | No bucket retention policy (spec §0.6 single root; `STEP1_DATA_INGESTION_PIPELINE.md` §9.9). The `if_generation_match=0` precondition is still open, and low-risk under the run lock. |
| T6 | Lock expiry **done**; `resolve` write-back **moot**; run ids with `:`/`+` kept as a nit | The one-DAG lock leaves no concurrent `resolve`. |

They come on top of the architecture changes still open in `STEP1_DATA_INGESTION_PIPELINE.md` §0.3: BigQuery metadata adapter, append-only writes, warehouse port. (Done: run lock, DuckDB/S3 removal, the ingestion manifest.)

**Explicitly not adopted from that design:**
- loading raw tables into BigQuery (conflicts with decision D3: BigQuery holds only the final dataset);
- the 12-table scope (guide §1 limits ingestion to 7 tables, and the raw zips already keep every other table);
- metastore-based backfill (the metastore serves only the current release, so it can't backfill).

---

## T1. Metastore release sensor (the trigger): moved to the `check_release` node (spec §2.0)

> The text below is the original 2026-10-08 idea. The spec's §2.0 replaces it: it adds RESUME and CHANGED, appends to `vintage_registry`, and gates `cms_nh_pipeline` instead of starting `ingest_vintage`.

**Why.** Detect a new monthly release cheaply, through the documented API, before starting a full run. Today the pipeline always runs the archive discovery; this tells it *when* to.

**What.**
- A `precursor-ingest sensor` command, or a `sensor` step, that calls `/metastore/schemas/dataset/items/{id}` for the 7 dataset IDs in `sources.yaml` and reads `modified`, `released` and `nextUpdateDate`.
- **New complete release:** all 7 `modified` values share a data month newer than the latest selected vintage in `inventory`. Exit 0 with "release ready", and Airflow starts `ingest_vintage`.
- **Partial release:** only some datasets moved. Log it and exit "waiting"; no run.
- **Nothing new:** exit "nothing new".
- Store each check in the metadata store (`source_datasets` already holds these fields; append a row per check).

**Done when:**
- runs daily in under 10 s;
- correctly reports ready / partial / nothing-new on recorded fixtures, including the 2026-08-06 Provider-Info-only case;
- the Airflow DAG uses it as the trigger.

---

## T2. Per-table download fallback (parked)

**Why.** The archive endpoints are undocumented. If CMS changes them, monthly runs should continue while the archive path is fixed.

**What.**
- A second `CatalogSource` adapter, `CmsMetastoreCurrentSource`. For each of the 7 dataset IDs it takes the metastore `distribution.downloadURL` and downloads the current CSVs.
- Reuse the same `SafeHttpClient`:
  - add the `…/sites/default/files/resources/…csv` path pattern to the URL allow-list;
  - allow the `text/csv` content type for this source only;
  - skip the zip signature check for CSVs; size and sha256 checks still apply.
- Store as `landing/cms_nh/csv/release=<YYYY-MM>/<file>.csv`, with inventory rows exactly like extracted zip members, so `resolve` and later steps don't care which source was used.
- **Activation:** automatic when the archive contract check fails, or forced with a config flag `source.fallback: metastore_csv`. Log a WARNING whenever the fallback is used.

**Done when:**
- with the archive endpoint mocked as broken, a run produces a complete inventory for the current month from per-table CSVs;
- the same Step 2 inputs result;
- the security tests cover the new allow-list pattern.

---

## T3. CLI polish: `--dry-run` and clearer exit codes (exit codes done; `--dry-run` dropped 2026-10-16)

**Why.** Safe previews, and an orchestrator that can tell "data problem" apart from "bug".

**What.**
- `precursor-ingest step1 --dry-run`: run discover and select only. Print what *would* be downloaded (new, changed and republished candidates, with sizes); write nothing to the object store; write no state rows except one `run_log` entry.
- **Exit codes:**

  | Code | Meaning |
  |---|---|
  | 0 | Success, or nothing new |
  | 1 | Error (exception, network failure after retries, storage error) |
  | **2** | **Validation / drift failure:** quarantined zip, rejected URL, unmapped header (T4), missing in-scope table in a new release |
  | 3 | Config or usage error |

- Document the codes in `--help` and in the Step 1 doc §2. Airflow treats any non-zero code as a failure, but alerts can say which kind.

**Done when:** tests assert each exit code on the matching fixture, and `--dry-run` leaves the object store byte-identical.

**Done (2026-10-13):** the codes above are implemented for every command (`pipeline.py` `EXIT_*`; click's own usage errors are mapped from 2 to 3) and tested in `tests/ingestion/test_cli.py` and `test_dag_tasks.py`. "Unmapped header" waits for T4, and "missing in-scope table" is still only an anomaly. **Open:** `--dry-run`.

---

## T4. Early schema-contract (drift) check after extraction (narrowed: silver owns the failure, spec §3.2)

**Why.** Catch CMS column renames or additions at ingest time, before Step 2 runs, with the same effective-dated contract Step 2 uses.

**What.**
- After `inventory`, compare each selected file's header against `configs/column_map.yaml`, the effective-dated contract. Drafted at `docs/data/step2_reference/column_map.draft.yaml`; it becomes the real config in Step 2.
  - **PASS:** every header is an alias valid for that vintage, or is listed under `ignored`.
  - **WARN:** a mapped column is missing where it is expected (e.g. a column dropped by CMS).
  - **FAIL:** an unknown header.
- Write `reports/step1/run=<id>/01_schema_drift.csv` (table, vintage, header, status) and add a section to the Checkpoint 1 summary.
- A FAIL exits with code 2 (T3), so the release stops until someone adds the mapping in a PR.

**Done when:**
- runs on all 427 current files with **0 FAIL** (the draft map already covers every header);
- a fixture with an invented column name fails with exit code 2;
- the check reads the same YAML Step 2 uses, so there is one contract, not two.

## T5. Follow-ups from the PR #1 review (not blocking)

- **GCS "never overwrite" can race.** `exists()` then upload is two calls; two concurrent runs could both write. The run lock (D8, done 2026-10-13) now stops two runs from overlapping. Inside one run, keys are per-entry, so the race needs a bug to happen. A GCS precondition (`if_generation_match=0`) in the fsspec adapter is still the belt-and-braces fix.
- ~~**Immutability relies on a bucket retention policy**~~ **Decided 2026-10-16:** no retention policy on the single root (spec §0.6). Object versioning on, a 30-day lifecycle rule on `quarantine/`, and the code's never-overwrite rule for `landing/`. `prod.yaml` says `immutable: object_versioning`. Terraform (`infra/terraform/modules/gcs`) must provision versioning and the lifecycle rule.

---

## T6. Follow-ups from the Airflow-readiness review (2026-10-13, not blocking)

- **Lock expiry: done (2026-10-16).** `lock acquire` and standalone runs take over a lock older than `run_lock.stale_after_hours` (default 12) with a WARNING; `holder` records it.
- **Moot under the one-DAG lock (2026-10-16):** `resolve` writes back whole inventory rows it read at its start. If someone runs `resolve` by hand with a DAG's run id while that run's mapped inventory tasks are still running, it can overwrite their fresh rows. The next run flags such a zip as pending again. Fix: update only `status` / `status_detail`.
- **Run ids with `:` and `+`** (Airflow's) appear in object keys (`reports/step1/run=…/`). This works on GCS and POSIX disks. If it ever bothers a tool, map them to `-` when building keys.

## Suggested order (2026-10-16)
1. T1 as the `check_release` node (with the bronze/silver/gold port, spec §15), using `landing_backlog()`.
2. T5 `if_generation_match=0`, when the GCS adapter is exercised for real.
3. T4 report-only, if wanted. T2 stays parked.
