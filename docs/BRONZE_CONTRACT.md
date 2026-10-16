# Bronze contract (hand-off to silver)

What the `bronze` node guarantees, and what silver may and may not assume. The rules come from
`DATA_PIPELINE_SPEC.md` §3.1; this file describes what the code in `src/precursorintelligence/harmonize/`
actually does.

## 1. In one paragraph

Bronze is a **lossless, all-text Parquet copy** of each selected CMS CSV, one partition per
`(table, vintage)`, plus 8 lineage columns. It does not rename, prune, cast, map values, de-duplicate or stack
months; that is silver's job. Everything bronze knows about a file is in two metadata tables
(`bronze_files`, `vintage_flags`) and one JSON manifest per run.

## 2. Where it lives

```
bronze/<table>/vintage=<YYYY-MM>/part-0.parquet        # zstd; one file per partition
metadata/bronze/run_date=<YYYY-MM-DD>/bronze_run_<run_id>_<timestamp>.json   # one per run, never overwritten
reports/bronze/run=<run_id>/logs/<file>.json           # the process log of each task
```

`<table>` is one of `provider_info`, `health_citations`, `survey_dates`, `penalties`, `mds_qm`, `claims_qm`,
`citation_lookup`. Read through the `TableSink` port (`ctx.table_sink.read(table, {"vintage": "2026-08"})`);
it adds a `vintage` column taken from the folder name, which equals `_vintage_month`.

## 3. Columns

**Data columns:** every column of the CSV, as `String`, in the original order and with the original header
text (not normalised; silver's `column_map` does that).

- CCNs keep leading zeros and letters (`015009`, `15E123`).
- An **unquoted empty field is `null`; a quoted `""` stays an empty string.** Treat both as missing in silver.
- Rows where every field is empty are dropped (the trailing blank line of 2021-01 to 2024-06 files) and
  counted in `bronze_files.blank_rows`. A row that is only partly empty is data.
- Exact duplicate rows are kept (penalties has them; `_row_number` tells them apart).

**Lineage columns**, appended after the data columns, in this order:

| Column | Type | Meaning |
|---|---|---|
| `_vintage_month` | String `YYYY-MM` | From the inventory (the month of `Processing Date`) |
| `_processing_date` | Date | From the inventory; null for `citation_lookup` (it has no such column) |
| `_source_zip` | String | Zip the file came from (a republished copy is `<stem>__<sha8>.zip`) |
| `_zip_sha256` | String | sha256 of that zip |
| `_source_file` | String | Member path inside the zip |
| `_header_hash` | String | Step 1's normalised-header hash (16 hex characters) |
| `_row_number` | Int64 | 1-based position in the file after blank rows are dropped |
| `_ingested_at` | Datetime UTC | When bronze loaded the file |

## 4. Guarantees

1. **One partition per selected file.** The `inventory` row with `status = 'selected'` for each
   `(table, vintage)` is exactly one partition. Bronze refuses to start (exit 2) if two rows are selected for
   one `(table, vintage)` or if any inventory row is still `pending`.
2. **Reconciled against Step 1.** `rows + blank_rows == inventory.row_count` and the recomputed header hash
   equals `inventory.header_hash`. A file that fails either check writes **no partition** (an existing one is
   left untouched), gets a `failed` row, and the run exits 2. It is retried next run.
3. **Decoded without loss.** Strict UTF-8 first, then cp1252 (2021-01 to 2023-04 files hold byte `0x92`). Bytes
   that neither accepts fail the file; nothing is replaced. The encoding used is recorded per file.
4. **Idempotent.** A file whose zip sha256 and header hash match the published row, and whose partition still
   exists, is skipped without a new row. Re-running changes nothing.
5. **Checked end to end.** After loading, every selected file (not only those touched) must have a published
   row for its current zip and an existing partition (`every_selected_file_has_valid_partition`); otherwise
   exit 2. `precursor-pipeline bronze --verify` runs the same check read-only.

## 5. Metadata tables

**`bronze_files`** is append-only: one row per file loaded or failed (skipped files get none). The **latest
published row** (`status` in `loaded`, `reloaded`) per `(table_name, vintage_month)` is the current one.

| Column | Meaning |
|---|---|
| `status` | `loaded` first load; `reloaded` the partition was rewritten (new zip, or the partition or row was lost); `failed` nothing written |
| `row_count`, `blank_rows` | Data rows kept; blank rows dropped |
| `encoding` | `utf-8` or `cp1252` |
| `zip_sha256`, `header_hash`, `member_path`, `entry_id` | What was loaded |
| `content_hash` | sha256 over the data columns without `Processing Date`, rows sorted |
| `content_unchanged` | Same hash as the table's previous published vintage; null for its first vintage. **Informational** (see §7) |
| `object_uri`, `detail`, `run_id`, `loaded_at` | Where it went, the failure reason, which run |

**`vintage_flags`** holds one current value per `(vintage_month, flag, table_name)`:

- `stale_refresh` (`table_name = ''`): all three event tables (`health_citations`, `survey_dates`, `penalties`)
  have the same content as their previous vintage. Not written until all three have a published file. A
  table's first vintage is never stale. Recomputed from the current hashes on every run, so a republished
  earlier vintage updates it.
- `revised_after_publication` (per table): a reload changed the content of what had been published for that
  vintage. Sticky: nothing clears it.

## 6. What silver should do

- Read the **latest published row** per `(table, vintage)`; ignore `failed` rows.
- On `reloaded`, rebuild everything downstream of that vintage. If `content_hash` changed, the history was
  rewritten (`revised_after_publication`); whether older `t` should keep the first-published version is open
  decision §14 #18.
- On `stale_refresh`, keep the data (it is what CMS published at that cutoff). Ledgers gain no events from it.
- Treat `""` and null the same. Normalise headers with `column_map`, not bronze.
- Do not rely on file order across partitions; within a partition `_row_number` is the file order, which
  `dup_seq` for penalties needs.

Spec decisions that shape silver: penalty events are keyed by the 5 fields plus `dup_seq`, never `fine_id`
(§3.5, §14 #1); the citation lookup stays per vintage with `citation_lookup_versions` (§14 #5); the keep-list
extras and the value set for `n_uncorrected` are open (§14 #4).

## 7. Known limits

- `bronze_files.content_unchanged` is fixed when the file is loaded. If CMS later republishes an earlier
  vintage, the next vintage's stored value is not updated. `stale_refresh` does not have this problem.
- Vintages are loaded in ascending order within a run, so the "previous vintage" of a file already exists. A
  vintage loaded later than its successors would be compared with whatever was published before it.
- Memory: one file is read whole (the largest in-scope file, health citations, is about 165 MB of CSV).
- Bronze has not yet been run on the full 61-vintage archive. See §9.

## 8. Running it

```
precursor-pipeline bronze --run-id R          # DAG task: the run must already hold the data_pipeline lock
precursor-pipeline bronze                     # standalone: takes and releases the lock itself
precursor-pipeline bronze --verify            # read-only report; no lock, nothing written
```

Add `--env prod` or `--env-file <yaml>` to choose the backends.

| Exit | Meaning |
|---|---|
| 0 | All files loaded or skipped |
| 1 | Error: another run holds the lock, or a bug or outage. Retry |
| 2 | Bad data: a failed file, a pending inventory row, two selected files for one `(table, vintage)`, or a missing partition. Fix the data; don't retry |
| 3 | Config or usage error, including a DAG task whose run does not hold the lock |

The manifest lists counts per status, each non-skipped file with its detail, the revised tables, the
`stale_refresh` vintages and the every-file check.

## 9. Measured and expected numbers

**Measured** on the CMS "current" release of Aug 2026 (7 tables, UTF-8, no blank rows), staged outside the
repo: 7 of 7 files loaded in 1.3 s into 13 MB of Parquet; the second run skipped all 7; for provider info,
health citations and penalties the bronze data columns equal a plain read of the CSV.

| Table | Rows |
|---|---|
| `health_citations` | 419,479 |
| `mds_qm` | 249,730 |
| `survey_dates` | 149,985 |
| `claims_qm` | 58,760 |
| `penalties` | 15,696 |
| `provider_info` | 14,690 |
| `citation_lookup` | 643 |

**Expected on the full archive, not yet verified by running bronze** (from `STEP2_HANDOFF.md` §0 and spec §3.1):
427 selected files; 151 cp1252 files (2021-01 to 2023-04); a blank row in each of the 39 vintages from 2021-01
to 2024-06, in every table; `stale_refresh` for 2025-09 only. `precursor-pipeline bronze --verify` prints
these numbers for comparison.
