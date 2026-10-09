# Step 1: to-do backlog

These are improvements to the built Step 1 pipeline, borrowed from an earlier per-table ingestion design (2026-10-08 comparison). Status: **not started.**

They come on top of the architecture changes already listed in `STEP1_DATA_INGESTION_PIPELINE.md` §0.3: BigQuery metadata adapter, append-only writes, run lock, warehouse port, DuckDB/AWS removal, per-run manifests.

**Explicitly not adopted from that design:**
- loading raw tables into BigQuery (conflicts with decision D3: BigQuery holds only the final dataset);
- the 12-table scope (guide §1 limits ingestion to 7 tables, and the raw zips already keep every other table);
- metastore-based backfill (the metastore serves only the current release, so it can't backfill).

---

## T1. Metastore release sensor (the trigger)

**Why.** Detect a new monthly release cheaply, through the documented API, before starting a full run. Today the pipeline always runs the archive discovery; this tells it *when* to.

**What.**
- A `shingest sensor` command, or a `sensor` step, that calls `/metastore/schemas/dataset/items/{id}` for the 7 dataset IDs in `sources.yaml` and reads `modified`, `released` and `nextUpdateDate`.
- **New complete release:** all 7 `modified` values share a data month newer than the latest selected vintage in `inventory`. Exit 0 with "release ready", and Airflow starts `ingest_vintage`.
- **Partial release:** only some datasets moved. Log it and exit "waiting"; no run.
- **Nothing new:** exit "nothing new".
- Store each check in the metadata store (`source_datasets` already holds these fields; append a row per check).

**Done when:**
- runs daily in under 10 s;
- correctly reports ready / partial / nothing-new on recorded fixtures, including the 2026-08-06 Provider-Info-only case;
- the Airflow DAG uses it as the trigger.

---

## T2. Per-table download fallback

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

## T3. CLI polish: `--dry-run` and clearer exit codes

**Why.** Safe previews, and an orchestrator that can tell "data problem" apart from "bug".

**What.**
- `shingest step1 --dry-run`: run discover and select only. Print what *would* be downloaded (new, changed and republished candidates, with sizes); write nothing to the object store; write no state rows except one `run_log` entry.
- **Exit codes:**

  | Code | Meaning |
  |---|---|
  | 0 | Success, or nothing new |
  | 1 | Error (exception, network failure after retries, storage error) |
  | **2** | **Validation / drift failure:** quarantined zip, rejected URL, unmapped header (T4), missing in-scope table in a new release |
  | 3 | Config error (today this is 2; it moves to 3) |

- Document the codes in `--help` and in the Step 1 doc §2. Airflow treats any non-zero code as a failure, but alerts can say which kind.

**Done when:** tests assert each exit code on the matching fixture, and `--dry-run` leaves the object store byte-identical.

---

## T4. Early schema-contract (drift) check after extraction

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

---

## Suggested order
**T4 → T3 → T1 → T2.** T4 and T3 are small and protect every run. T1 is needed for the automated monthly trigger (Airflow). T2 is insurance and can follow last.
