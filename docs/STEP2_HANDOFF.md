# Step 2 handoff: bronze load, schema harmonisation, column pruning

**Audience:** whoever designs and builds Step 2 of `dataset/Dataset_build_guide_v1.md` (§3: "Load to bronze, harmonize schemas, prune columns").
**Status of inputs:** Step 1 is complete. 427 selected CSVs cover 7 tables × 61 vintages (2021-01 → 2026-08). Checkpoint 1 is awaiting approval.
**How this was produced:**
- every fact below was **measured on the 427 extracted files** (full scans, not samples);
- the CMS data dictionary (July 2026) was cross-checked;
- machine-readable drafts are in [`docs/data/step2_reference/`](data/step2_reference/).

**Storage and metadata decisions (2026-10-08, binding for Step 2).** Full record: `STEP1_DATA_INGESTION_PIPELINE.md` §0.

| Concern | Local development | GCP (test and production) |
|---|---|---|
| Bronze, silver and gold files | Parquet on local disk (`./data`) | Parquet on GCS, same keys |
| Pipeline metadata (incl. Step 2 tables) | Postgres (Docker), schema `ops` | BigQuery dataset `precursorintelligence_ops` |
| Final dataset only (Step 9) | Postgres, schema `gold` | BigQuery dataset `precursorintelligence_gold` |
| Unit tests | SQLite in memory + local / `memory://` store | — |

- **No warehouse for intermediate layers:** bronze and silver are never loaded into BigQuery or any database, and there is no DuckDB table adapter.
- **Metadata writes:** append-only with a "latest version" view; batch load jobs, never streaming inserts.
- **Access pattern:** read once and write once per task.
- **Concurrency:** one run at a time, enforced by a run lock.
- **Audit:** a per-run JSON manifest is written by each layer.

---

## 0. TL;DR: the 13 things that will bite you

| # | Finding | Consequence for Step 2 |
|---|---|---|
| 1 | **151 of 427 files are Windows-1252, not UTF-8.** That is every file from 2021-01 to 2023-04. They contain byte `0x92`, a curly apostrophe. From 2023-05 on, files are UTF-8, and 106 of those contain real multi-byte characters. | Decode per file: strict UTF-8 first, then cp1252 as the fallback. Both are verified to decode all files. **Never decode a UTF-8 file as cp1252, and never use `utf8-lossy` in bronze.** |
| 2 | **39 vintages (2021-01 → 2024-06) end with an extra blank line** (`\r\n\r\n`). Readers turn it into one all-null row per file. Step 1's `row_count` **includes** that row. | Drop all-null rows. Reconcile with `bronze_rows == inventory.row_count − blank_rows` and record `blank_rows`. |
| 3 | **The 2023-06 rename hit every table at once:** `Federal Provider Number` → `CMS Certification Number (CCN)`, and `Provider City/State/Zip Code/County Name/Phone Number` → `City/Town`, `State`, `ZIP Code`, `County/Parish`, `Telephone Number`. | Aliases with month ranges are in `column_map.draft.yaml`. It is already verified: no month has two aliases, and every header is either mapped or ignored. |
| 4 | **About 1.8% of CCNs are alphanumeric**, e.g. `01A193`. Every CCN is exactly 6 characters, and no leading zeros are lost in the source. | Keep CCN as a string. Validate `^[0-9A-Z]{6}$`, **not** `^\d{6}$`. Never cast to an integer. |
| 5 | **Penalty "exact duplicate" rows are real, separate fines.** From 2026-06 on, about 540 fines a month are identical except `Fine ID`, and each has its own Fine ID. Before `Fine ID` existed, exact duplicates appear **only in 2025-10 → 2026-05** (294–340 a month). | **Do not drop exact-duplicate penalty rows** (this contradicts guide §2e). Keep them with an occurrence index `dup_seq`. |
| 6 | **`Survey Type` in Health Deficiencies is always `"Health"`.** | The survey-type enum for citations must come from the flags (`Standard Deficiency`, `Complaint Deficiency`, `Infection Control Inspection Deficiency`), not from that column. |
| 7 | **Inspection Dates has 5 survey types**, including fire safety. It also lists **complaint surveys only if they resulted in a citation** (data dictionary). | Map them to the guide's enum. Note in the feature registry that "complaint surveys" means *cited* complaint surveys. |
| 8 | **Every date is ISO `YYYY-MM-DD`** in all 427 files: Survey, Correction, Penalty, Denial start, Date First Approved, Rating-cycle and Processing Dates. | One date parser. Anything unparseable is a real anomaly and should be counted. |
| 9 | **`Processing Date` is constant within each file and always the 1st of the month.** Events in a file are dated on or before it, except in **3 vintages**: up to 9 days later in survey_dates 2025-10 and 2025-12, and health_citations 2025-12. They are never later than the zip's publish date. | `cutoff_date = processing_date` (guide). Feature filters must use `event_date <= cutoff_date`; those few later events then correctly fall into labels. |
| 10 | **Quality-measure codes change:** 405→480 and 453→479 (2025-02), 471 dropped (2024-11), and **419→481 (antipsychotic, 2026-02)**. That last one is *not* stable, despite what guide §6 expects. MDS values were also **frozen for about a year**: `Measure Period = 2023Q3-2024Q2` in every vintage from 2024-10 to 2025-09. | Store `measure_code` as is in silver; do the bridging in Step 5. Keep `measure_period` so freshness is visible. |
| 11 | **The 2026-07 Provider Info file selected by Step 1 comes from the 2026-08-06 partial re-release.** Against the 2026-07-29 original it changes case-mix and adjusted staffing for about 14,200 of 14,693 facilities. Reported staffing changed for 1 facility. | See §7.4. The 2026-07 vintage only appears in **scoring** rows, so this choice does not affect training. |
| 12 | **No numeric column contains text tokens.** There is no `Data Not Available` in numeric fields and no thousands separators. Missing values are empty strings. The **only** text sentinel is `Automatic Sprinkler Systems…` = `Data Not Available`. | Empty → null. The sentinel → null plus a flag. Keep footnote columns: they hold the missing reason. |
| 13 | **Stale refresh:** vintage **2025-09** carries a new `Processing Date`, but its health citations, penalties, survey dates and claims QMs are **identical to 2025-07**, ignoring the date column. Only Provider Info and MDS QMs changed; 2025-08 was never published. | Hash each table's content (excluding `Processing Date`). Flag `stale_refresh` per (table, vintage) and keep the data, because it is what CMS published at that cutoff (§7.6). |

---

## 1. Inputs from Step 1 (the contract)

### 1.1 Where things are

| What | Where |
|---|---|
| Extracted CSVs | ObjectStore key `landing/cms_nh/extracted/zip_date=<YYYY-MM-DD>/<zip_stem>/<file>.csv`. Locally: `data/…`. |
| Which file to load | `MetadataStore.inventory` rows with **`status = 'selected'`**: exactly 1 per `(table_name, vintage_month)`, 427 rows. |
| Vintage and cutoff | `inventory.vintage_month` (YYYY-MM) and `inventory.processing_date`. Use `processing_date` as the guide's `cutoff_date`. |
| Lineage | `inventory.zip_name`, `zip_sha256`, `zip_date`, `member_path`, `extracted_key`, `header_hash`, `row_count` (includes the blank row, see §0 #2). |
| Zip-level lineage | `download_log` (`sha256`, `object_uri`, `http_last_modified`). Step 9's `dataset_manifest.json` needs the zip sha256 per vintage from here. |
| Release metadata | `source_datasets` (current `modified` / `released` / `nextUpdateDate`). |

### 1.2 The query Step 2 starts from

```sql
select table_name, vintage_month, processing_date, extracted_key, row_count,
       header_hash, header_json, zip_name, zip_sha256, zip_date, member_path
from inventory
where status = 'selected'
order by table_name, vintage_month;
```

In code, reuse Step 1's ports instead of opening paths directly:

```python
from precursorintelligence.common.config import load_settings
from precursorintelligence.common.context import build_context
ctx = build_context(load_settings("dev"))
rows = ctx.metadata.read("inventory", {"status": "selected"})
with ctx.object_store.open_read(rows[0]["extracted_key"]) as f:
    raw = f.read()                                     # bytes; decode per §3
ctx.table_sink.write_partition("provider_info", {"vintage": "2021-01"}, df)   # -> bronze/provider_info/vintage=2021-01/
```

### 1.3 Volumes (selected files only)

| Table | Files | Rows per vintage (min – max) | Total rows | Header sets |
|---|---|---|---|---|
| provider_info | 61 | 14,690 – 15,341 | 914,391 | 10 |
| health_citations | 61 | 367,658 – 419,479 | 23,879,099 | 3 |
| survey_dates | 61 | 149,705 – 217,687 | 11,158,607 | 2 |
| penalties | 61 | 7,836 – 40,210 | 1,604,233 | 3 |
| mds_qm | 61 | 249,730 – 276,121 | 16,207,983 | 2 |
| claims_qm | 61 | 58,760 – 61,361 | 3,657,447 | 2 |
| citation_lookup | 61 | 639 – 643 | 39,133 | 1 |

These totals include the 39 blank rows (one per file in 2021-01 → 2024-06, all tables). Raw extracted size is about 18 GB. As all-string zstd Parquet, expect roughly 1.5–2.5 GB of bronze; that is an estimate, not measured.

The 36 `duplicate_vintage` files are **not** loaded. 35 of them are byte-identical to the selected file; the exception is 2026-07 Provider Info (§7.4).

---

## 2. What Step 2 must produce (guide §3, unchanged)

1. **Bronze (2a):** every selected CSV, all columns as **strings**, written to `bronze/<table>/vintage=<YYYY-MM>/` with lineage columns (§2.1).
2. **Column map (2b):** `configs/column_map.yaml`, canonical name → raw aliases with month ranges and dtype. **Hard rule:** zero unmapped headers. Start from `data/step2_reference/column_map.draft.yaml`.
3. **Keep-lists (2c):** `configs/keep_columns.yaml`.
4. **Normalisation (2d):** CCN, dates, severity, tags, Y/N, numerics, enums (§5).
5. **Silver stacked tables (2e):** `silver/{provider,citations,surveys,penalties,mds_qm,claims_qm}_by_vintage` plus `silver/citation_lookup`, each with `vintage_month` and `cutoff_date`. Also `reports/02_column_availability.csv`, and the within-vintage uniqueness checks (§6).
6. 🛑 **Checkpoint 2:** `reports/02_harmonization.md`. Its contents and the expected values are in §11.

### 2.1 Bronze, precisely (decision D9)

Bronze is a **lossless Parquet copy** of each selected CSV. It reads only from landing; it never downloads or extracts.

**What it does, per selected (table, month) file:**
1. **Skip if loaded.** Skip when the partition already exists with the same `_zip_sha256` and `_header_hash`.
2. **Read** the bytes from `landing/cms_nh/extracted/…`.
3. **Decode:** strict UTF-8, falling back to cp1252. Record the encoding used.
4. **Parse every column as text**, keeping the original names and order.
5. **Drop all-null rows** (the trailing blank line) and count them.
6. **Add the lineage columns** (table below).
7. **Reconcile:** `rows + blank_rows == inventory.row_count`, otherwise fail.
8. **Write** `bronze/<table>/vintage=YYYY-MM/part-0.parquet` through `ctx.table_sink`.
9. **Record** the file in the metadata store (`bronze_files`, append-only).

**What it does not do:** rename, prune, type-cast, clean or map values, de-duplicate, or stack months. All of that is silver.

**Why text.** Type guessing silently damages this data:
- leading zeros drop from CCNs and tags (`015009`, `0880`);
- alphanumeric CCNs fail;
- the same column gets different types in different months, so the months can't be stacked;
- sentinels break numeric parsing.

Text guarantees a faithful copy. Silver applies the declared dtype from `column_map.yaml`, and counts every value that fails to convert.

**Lineage columns.** On every row of all 7 bronze tables. We generate them, so they use real types:

| Column | Type | Source |
|---|---|---|
| `_vintage_month` | string `YYYY-MM` | `inventory.vintage_month` |
| `_processing_date` | DATE | `inventory.processing_date` |
| `_source_zip` | string | `inventory.zip_name` |
| `_zip_sha256` | string | `inventory.zip_sha256` |
| `_source_file` | string | `inventory.member_path` |
| `_header_hash` | string | `inventory.header_hash` |
| `_row_number` | INT64 | 1-based position in the source file, after dropping blank rows |
| `_ingested_at` | TIMESTAMP (UTC) | load time |

Constant-per-file columns compress to almost nothing in Parquet.

**Republished data (decision D10).** If a later Step 1 run selects a different file for a (table, month):
- a CMS re-upload of the same zip;
- or a newer zip for that month.

The selected `zip_sha256` then differs from the bronze partition's. Bronze **reloads that partition** (overwrite) and appends a `superseded` record, and the matching silver partition (and anything downstream) is marked for rebuild. Old bronze versions are not kept, because landing retains both raw zips.

---

## 3. Reading the files correctly

| Property | Observed | Rule |
|---|---|---|
| Encoding | cp1252 for 2021-01 → 2023-04 (151 files); UTF-8 for 2023-05 → 2026-08. No BOM anywhere. | `try raw.decode("utf-8") except UnicodeDecodeError: raw.decode("cp1252")`. Record the encoding used per file. |
| Line endings | CRLF in every era | Standard CSV readers handle this. |
| Trailing blank line | 2021-01 → 2024-06: file ends `\r\n\r\n` | Drop rows where all columns are null. Count them. |
| Quoting | Double quotes; quoted commas and newlines inside text fields (addresses, descriptions) | Use a real CSV parser (Polars or `csv`). Never split on lines. |
| Header normalisation | Step 1's `header_hash` = sha256 of the header list with BOM stripped and whitespace collapsed | Use the same normalisation, so the header sets join back to the inventory. |
| Types in bronze | — | `pl.read_csv(text.encode(), infer_schema=False)` gives all `String`. Polars reads UTF-8 only, so re-encode cp1252 text to UTF-8 first. |
| Memory | Largest file is about 166 MB (health citations) | Process one file at a time, in memory. No chunking needed. |

**Reconciliation, per file:** `bronze_rows + blank_rows == inventory.row_count`. Step 1 counted with `polars.scan_csv`, so use the same reader settings.

---

## 4. Schema eras and the column map

### 4.1 Rename and change timeline (observed in the headers and confirmed by dictionary Table 16)

| Effective vintage | Table(s) | Change |
|---|---|---|
| 2022-01 | provider_info | Added weekend staffing (total, RN), turnover (total, RN, with footnotes), administrator departures (with footnote) |
| 2022-07 | provider_info | Removed `RN Staffing Rating` (+ footnote) |
| 2022-08 | provider_info | Added `Adjusted Weekend Total Nurse Staffing Hours per Resident per Day` |
| 2023-01 | health_citations | Added `Citation under IDR`, `Citation under IIDR` |
| **2023-06** | **all 6 provider-level tables** | `Federal Provider Number`→`CMS Certification Number (CCN)`; `Provider City`→`City/Town`; `Provider State`→`State`; `Provider Zip Code`→`ZIP Code`; provider_info only: `Provider County Name`→`County/Parish`, `Provider Phone Number`→`Telephone Number`. provider_info also added `Affiliated Entity ID/Name`. |
| 2023-08 | provider_info | Added `Latitude`, `Longitude`, `Geocoding Footnote` |
| 2024-07 | provider_info | Added `Nursing Case-Mix Index`, `… Ratio`, `Case-Mix Weekend Total …` |
| **2025-07** | provider_info | `Affiliated Entity ID/Name` → `Chain ID/Name`; added `Number of Facilities in Chain` and 4 `Chain Average …` ratings; `Rating Cycle 2 …` → `Rating Cycle 2/3 …` for 6 columns; dropped all `Rating Cycle 3 …`. **Not renamed:** `Rating Cycle 2 Number of Standard Health Deficiencies` and `Rating Cycle 2 Standard Health Survey Date`. |
| 2025-10 | provider_info | Added `Urban` |
| 2026-02 | provider_info | Removed `Number of Facility Reported Incidents`, `Number of Substantiated Complaints` |
| **2026-06** | penalties | Added `Fine ID` |

`citation_lookup` has one header set for all 61 vintages.

### 4.2 The draft column map (`data/step2_reference/column_map.draft.yaml`)
- **Generated from the real headers.** For each canonical column, aliases carry `from`/`to` month ranges computed from the 427 files. `available_from` and `complete` show late-arriving columns.
- **Validated:**
  - every alias exists;
  - no month has two aliases for the same canonical column;
  - every raw header in every month is either an alias or listed under `ignored:`, so the guide's hard rule already holds for the current archive.
- **Coverage:** it includes the guide's keep-list for each table, the product-dimension columns (`provider_name, provider_address, city, zip, latitude, longitude`), plus these recommended extras:
  - `inspection_cycle`, `citation_under_idr/iidr` (citations; useful for Step 3's `removed_by_cms` analysis);
  - `survey_cycle` (surveys);
  - the MDS quarterly scores `q1..q4` with footnotes (for Δ and trend features).
- **dtype vocabulary:** `ccn, str, int, float, date, bool_yn, enum, footnote, tag, severity, survey_type_enum`. The normaliser maps each to a function (§5).
- **Partial-coverage columns** (null before their start month; record them in `02_column_availability.csv`):

  | Column | Available from |
  |---|---|
  | turnover, weekend staffing, admin departures | 2022-01 |
  | chain_id / chain_name | 2023-06 |
  | latitude / longitude | 2023-08 |
  | IDR flags | 2023-01 |
  | fine_id | 2026-06 |

### 4.3 Ignored columns: worth a second look, but outside the guide's keep-list

| Column | Available from | Notes |
|---|---|---|
| `Nursing Case-Mix Index (Ratio)` | 2024-07 | Acuity control |
| `Urban` | 2025-10 | Only 10 vintages, so `ablation_only` |
| `Adjusted Weekend Total Nurse Staffing …` | 2022-08 | |
| `Long-Stay QM Rating` / `Short-Stay QM Rating` | — | |
| `Number of Facilities in Chain` | 2025-07 | Product-only |
| `Provider SSA County Code` | all eras | Stable, so it is a better county key than the renamed county name |

**Leave ignored, as the guide intends:**
- `Number of Fines`, `Total Amount of Fines in Dollars`, `Number of Payment Denials`, `Total Number of Penalties` (guide §7.4);
- all `Rating Cycle …` scores;
- `Total Weighted Health Survey Score`;
- `Legal Business Name`, `Location`, phone, `Deficiency Description` in citations (the lookup has it).

---

## 5. Per-table field notes and normalisation rules

Domains below are **observed across all 61 vintages**. Counts are rows, and the 39 blank rows are excluded.

### 5.1 Common
- **`ccn`:** strip and upper-case. Assert length 6 and `^[0-9A-Z]{6}$`. Alphanumeric examples: `01A193`, `04E090`, `05A024`.
  - CCNs keep their meaning across the 2023-06 rename: 15,012 of 15,046 (2023-05) / 15,018 (2023-06) facilities appear in both months.
- **`processing_date`:** one value per file, always `YYYY-MM-01`, equal to `vintage_month`. Assert it.
- **Y/N flags:** values are `Y`/`N` only. Map to boolean; anything else becomes null and is counted.

### 5.2 provider_info
- **`ownership_type`:** 13 values, `For profit - {Corporation, Limited Liability company, Individual, Partnership}`, `Non profit - {Corporation, Church related, Other}`, `Government - {County, Hospital district, State, City, City/county, Federal}`. `ownership_group` is the prefix before ` - `.
- **`provider_type`:** `Medicare and Medicaid`, `Medicare`, `Medicaid`.
- **`special_focus_status`:** `SFF`, `SFF Candidate`, or **blank = not SFF** → `NONE`. Blank here is a real category.
- **`resident_family_council`:** `Resident`, `Both`, `Family` (the last seen in 2025-09). Blank → `NONE`.
- **`sprinkler_status`:** `Yes`, `Partial`, `Data Not Available` (379 rows) → null plus flag.
- **Ratings:** `1`–`5`, or blank. Blank rows: Overall 10,523, Staffing 19,573, Health 10,523, QM 15,504. The reason is in `*_fn`, e.g. 18 = SFF not rated, 1 = new facility.
- **Footnote codes seen:** 1, 2, 6, 10, 12, 18, 20, 22, 23, 24, 25, 26, 27.
  - Code **12 was retired in 2025-01**, replaced by 23–25, so missing-reason categories shift across eras.
  - Turnover footnotes (6/26/27) appear on 8–15% of rows.
  - Meanings are in `footnotes.draft.yaml`, from dictionary Table 15.
- **`occupancy`** (computed later): `avg_residents_per_day / certified_beds`. The v0 analysis found 1,533 rows over 1.0, and the guide caps it at 1.5.

### 5.3 health_citations
- **`Survey Type`:** always `Health`, so it is ignored for typing.
- **Type flags:** standard (`Y` 18.28 M), complaint (`Y` 6.69 M), infection control (`Y` 0.81 M). **A citation can carry several flags** (standard + complaint on the same survey), so keep three booleans rather than a single enum.
- **`Deficiency Prefix`:** always `F`.
- **`Deficiency Tag Number`:** always 4 digits as text (e.g. `0880`). Keep the text, an integer `deficiency_tag`, and `ftag = prefix + tag` (e.g. `F0880`).
- **`Scope Severity Code`:** observed `B C D E F G H I J K L`, **never `A`**. Distribution: D 61.7%, E 23.6%, F 6.7%, G 3.0%, B 1.5%, J 1.3%, C 1.3%, K 0.5%, L 0.2%, H 0.14%, I 0.01%.
- **`Deficiency Category`:** 10 values (the guide's §7.3 categories exist verbatim). For example, `Freedom from Abuse, Neglect, and Exploitation Deficiencies`, `Infection Control Deficiencies`, `Pharmacy Service Deficiencies`, and so on.
- **`Deficiency Corrected`:** 5 values, plus **`No revisit needed`, new from 2025-10**. "Not corrected" means:
  - `Deficient, Provider has plan of correction`;
  - `… has no plan of correction`;
  - possibly `Past Non-Compliance`, which is a team decision.
- **`Inspection Cycle`:** `1`, `2`, `3`.
- **IDR / IIDR:** from 2023-01; 27.5 k / 4.4 k rows flagged `Y`.

### 5.4 survey_dates
- **`Type of Survey`** mapping, with counts:

  | Raw value | Enum | Rows |
  |---|---|---|
  | `Health Standard` | HEALTH_STANDARD | 2.73 M |
  | `Health Complaint` | HEALTH_COMPLAINT | 3.02 M |
  | `Infection Control` | INFECTION_CONTROL | 2.66 M |
  | `Fire Safety Standard` | FIRE_STANDARD | 2.72 M |
  | `Fire Safety Complaint` | FIRE_COMPLAINT | 30 k |

- **`Survey Cycle`:** 1/2/3.
- **Complaint surveys** appear only if cited (dictionary Table 1/16).

### 5.5 penalties
- **`Penalty Type`:** `Fine` 1.445 M rows, `Payment Denial` 0.159 M.
- **`Fine ID`:** present for every Fine, always blank for Payment Denials (2026-06 →). It is unique wherever present.
- **`Fine Amount`:** whole dollars, all numeric.
- **`Penalty Date`:** the date of the **inspection that triggered** the penalty, not the date it was imposed (dictionary).
- **Duplicates:** see §6. Keep them.

### 5.6 mds_qm / claims_qm
- **`Measure Code`:** 3-digit strings. MDS has 21 (code, resident-type) pairs over time; claims has 4, stable (`521, 522` short-stay; `551, 552` long-stay). The full timeline is in `qm_measure_timeline.csv` and §8.
- **`Resident type`:** `Long Stay` / `Short Stay`.
- **`Measure Period`:**
  - MDS: `YYYYQn-YYYYQn` (4-quarter window);
  - claims: `YYYYMMDD-YYYYMMDD`.

  Parse both into `measure_period_start` / `_end` dates in silver.
- **`Used in Quality Measure Five Star Rating`:** Y/N. Claims is always `Y`.
- **Footnotes:**
  - MDS: 9 (too small), 10 (missing), 21 (not validated), 28 (annual measure; quarterly blank);
  - claims: 9, 10.
- **Keys:** `(ccn, measure_code)` is unique within every vintage, in both tables.

### 5.7 citation_lookup
- **Columns:** `Deficiency Prefix` ∈ {F, K, E}, `Deficiency Tag Number`, `Deficiency Prefix and Number` (e.g. `F-0880`), description, category (29 values across F/K/E).
- **Key:** `(prefix, tag)` is unique in every vintage.
- **Vintage:** no `Processing Date`; the month comes from the file name (configured in Step 1).
- **Rows:** 639–643 per vintage. The guide uses the latest vintage only, but keeping all vintages in silver is cheap and lets descriptions be joined as of t.

---

## 6. Keys and within-vintage uniqueness (measured; guide §2e checks)

| Table | Key tested | Duplicates (all 61 vintages) | Verdict |
|---|---|---|---|
| provider_info | `(ccn)` | 0 | Unique ✔ |
| health_citations | `(ccn, survey_date, prefix, tag)` | 0 | Unique ✔. The guide's base key suffices; the `is_standard, is_complaint` extension isn't needed. |
| survey_dates | `(ccn, survey_date, survey_type)` | 0 | Unique ✔ |
| mds_qm / claims_qm | `(ccn, measure_code)` | 0 | Unique ✔ |
| citation_lookup | `(prefix, tag)` | 0 | Unique ✔ |
| penalties | exact row | **2,209 rows**, all in 2025-10 → 2026-05 (294–340 per vintage) | **Not real duplicates** (see below) |
| penalties | `fine_id` (2026-06 →) | 0 duplicates; null only for Payment Denials | Unique ✔ |

**Penalties evidence and rule.**
- **Evidence:** in 2026-06/07/08, 541 / 539 / 546 fine rows are identical in every column except `Fine ID`, and each of them has a distinct Fine ID.
- **Interpretation:** the 2025-10 → 2026-05 "duplicates" are the same phenomenon before the column existed: multiple CMPs on one survey, consistent with the FY2025 per-instance and per-day CMP rule.
- **Rule for silver:**
  1. Keep every row.
  2. Add `dup_seq` = occurrence number within `(ccn, penalty_date, penalty_type, fine_amount, payment_denial_start_date)` in that vintage.
  3. Use `penalty_event_key = fine_id` when present, else `(…5 fields…, dup_seq)`.
- **Regime shift:** per-survey fine counts step up from 2025-10. Flag this for the enforcement features (`era`) and for `y_pen12`.

---

## 7. Vintages, cutoff dates and time semantics

### 7.1 Coverage
- **61 vintages, 2021-01 → 2026-08.**
- **Never published, for every table:** 2021-12, 2022-12, 2023-12, 2024-12, 2025-01, 2025-08, 2026-01.
- Step 6's spine therefore has no rows for those months, and Δ features for t−6/t−12 must handle a missing vintage. Use "null if absent", as in guide §7.5.

### 7.2 `cutoff_date`
- `cutoff_date = processing_date`, which is always the 1st of the vintage month. CMS publishes the zip 2–7 weeks later.
- **Measured: the latest event date in each vintage minus `processing_date`:**

  | Table | Range |
  |---|---|
  | health_citations | −81 to +9 days |
  | survey_dates | −74 to +9 days |
  | penalties | −102 to −3 days |
- **Events after the cutoff appear in only 3 vintages:**
  - survey_dates 2025-10: 6 rows, up to 2025-10-09;
  - survey_dates 2025-12: 44 rows, up to 2025-12-10;
  - health_citations 2025-12: 167 rows, up to 2025-12-10.
- **No event is ever later than the zip date.**
- **Rule:** keep these rows in bronze and silver. Steps 7–8 filter features with `event_date <= cutoff_date` and labels with `event_date > cutoff_date`, so they are handled correctly. Mention them in Checkpoint 2.

### 7.3 Data lag inside a vintage
The most recent survey in a vintage is typically 1–3 weeks before `processing_date`. Penalties lag a further 1–3 months. That is consistent with the guide's `B_pen > B_cit`, and Step 4 measures it properly.

### 7.4 The 2026-07 Provider Info decision (carried over from Checkpoint 1)
- **Selected:** the `nursing-homes_2026-08-06.zip` re-release, because the later zip wins.
- **What differs from the 2026-07-29 original:**

  | Column | Facilities changed |
  |---|---|
  | `Case-Mix RN …` | 14,273 |
  | `Case-Mix Nurse Aide …` | 14,271 |
  | `Adjusted Total …` | 14,237 |
  | `Adjusted Weekend Total …` | 14,232 |
  | `Adjusted Nurse Aide …` | 14,154 |
  | `Adjusted LPN …` | 8,378 |
  | `Adjusted RN …` | 6,480 |
  | `Case-Mix Total …` | 4,575 |
  | `Nursing Case-Mix Index Ratio` | 1,985 |
  | reported staffing and ratings | **1** |

  In short, CMS corrected the case-mix adjustment.
- **Impact:** t = 2026-07 is far past the last labelled month (2025-05), so it falls only in the `score` split. **Training is unaffected either way.**
- **Recommendation:** keep the corrected file, which is the guide's rule, and note it in `dataset_manifest.json`. If strict "as first published" data is preferred, swap in the original via `inventory.status`.

### 7.5 Era markers worth carrying as columns
| Marker | Why it matters |
|---|---|
| encoding era (cp1252 until 2023-04) | Text fields only |
| 2023-06 header standardisation | Mapping only |
| 2025-01 QM replacement and footnote-12 retirement | Measure codes and missing-reason codes change |
| 2025-07 rating method (cycle 2/3) and Chain rename | Feature definitions change |
| 2025-10 penalties multiplicity | Fine counts step up |
| 2026-06 `Fine ID` | New identifier |

### 7.6 Stale refreshes: a new `Processing Date` with old content

**Measured:** every table's content was compared with the previous vintage, ignoring the `Processing Date` column (sorted rows, hashed). One case exists in the archive:

| Vintage | Identical to 2025-07 | Changed |
|---|---|---|
| **2025-09** | health_citations, penalties, survey_dates, claims_qm | provider_info, mds_qm |

2025-08 was never published, so CMS appears to have re-stamped the July event files as September.

**Rule for Step 2:**
- **Detect:** compute `content_hash` per (table, vintage) over the data columns only, and set `stale_refresh = (content_hash == previous vintage's content_hash)`. Record it in the metadata store and in Checkpoint 2.
- **Keep the data.** It is what CMS published as of that cutoff, so point-in-time features for t = 2025-09 are still correct.
- **Ledgers (Step 3):** stale vintages add no new events, and `first_seen_vintage` is unaffected.
- **Monitoring and automation:** the team plan treats a stale release as "no new labels, skip retraining". The flag makes that decision possible.

---

## 8. Quality measures (stored in Step 2, selected and bridged in Step 5)

| Code | Resident | Vintages | Status | Note |
|---|---|---|---|---|
| 401 | LS | all 61 | stable | ADL decline |
| 404, 406, 407, 408, 409, 410, 415, 452, 454 | LS | all 61 | stable | 410 = **falls with major injury** |
| 451 | LS | all 61 | stable code | Description changed "move" → "walk independently" (2025-01). Same code; check its definition before bridging. |
| 430, 434, 472 | SS | all 61 | stable | 434 = new antipsychotic (SS) |
| **405 → 480** | LS | 405: 2021-01→2024-11; 480: 2025-02→ | bridge candidate | Low-risk incontinence → new or worsened incontinence. The gap months (2024-12, 2025-01) are unpublished vintages anyway. |
| **453 → 479** | LS | 453: →2024-11; 479: 2025-02→ | bridge candidate | High-risk pressure ulcers → pressure ulcers |
| **419 → 481** | LS | 419: →2025-12; 481: 2026-02→ | **bridge needed** | **Antipsychotic (LS).** The guide lists it as stable; it is not. |
| 471 | SS | →2024-11 | dropped | Function improvement |
| 521, 522, 551, 552 | claims | all 61 | stable | Rehospitalisation, ED visits, hospitalisations |

**Freshness warning.** `Measure Period` shows how stale each value is:
- **MDS stayed at `2023Q3-2024Q2` from 2024-10 to 2025-09**, so 11 vintages share the same underlying quarters.
- Claims periods also advance in jumps of 3–6 months.

Store `measure_period_start/end` in silver. Step 7 can then compute `qm_age_months = cutoff − measure_period_end`, so the model isn't fooled by repeated values.

---

## 9. Corrections and additions to the build guide

| Guide reference | What the guide says | What the data shows → recommendation |
|---|---|---|
| §2e | "penalties … count exact-duplicate rows and drop them" | They are distinct fines (§6). **Keep them**, add `dup_seq`, and key on `fine_id` when available. |
| §2d `survey_type` | Map to an enum "from observed values" | For **citations**, `Survey Type` is always `Health`; use the 3 flag columns. The enum applies to `survey_dates.Type of Survey`. |
| §2d `scope_severity` | "A–L" | Only B–L occurs. A stays valid, but its count is 0. |
| §2d `ccn` | "left-pad to 6" | Source CCNs are already 6 characters and may be alphanumeric. Padding is a no-op; validate `^[0-9A-Z]{6}$`. |
| §2b | Older files use legacy names (`PROVNUM`) | Not in scope: no 2021+ file uses legacy names. They only appear in L1 zips (before 2020-08). |
| §2c | "Chain columns: from 2023-06" | True, but the names change: `Affiliated Entity ID/Name` (2023-06 → 2025-06), then `Chain ID/Name` (2025-07 →). Both are in the draft map. |
| §2c | "Turnover and weekend staffing: from 2022-01" | Confirmed. Adjusted weekend staffing only from 2022-08. |
| §6 stable QMs | Antipsychotic (LS) "expected stable" | 419 ends 2025-12; 481 from 2026-02. Add it to the bridge list. |
| §0.3 | Vintage from `processing_date` | Confirmed and implemented. Citation Look-up has no Processing Date, so its vintage comes from the file name. |
| (new) | — | Encoding, blank-row and reconciliation rules (§3) belong in the guide's Step 2 text. |
| (new) | — | `Inspection Dates` lists complaint surveys only when cited. Document this in the feature registry. |
| (new) | — | Stale-refresh check: a new `Processing Date` can carry old content (2025-09 = 2025-07 for four tables). Flag it per (table, vintage); don't drop it (§7.6). |
| §0.5 storage | "Parquet on local disk … storage helper so it can be pointed at GCS later" | Confirmed and extended by the 2026-10-08 decisions: local disk / GCS for files; Postgres (local) / BigQuery (GCP) for metadata; BigQuery only for the final dataset (`STEP1_DATA_INGESTION_PIPELINE.md` §0). |

---

## 10. Recommended Step 2 design

Reuse Step 1's architecture, so Step 2 inherits backend swapping, logging and security.

```
src/precursorintelligence/
  harmonize/                      # new package for Step 2
    reader.py        # decode (utf-8 → cp1252), parse all-string, drop blank rows, normalise headers
    column_map.py    # load configs/column_map.yaml, resolve aliases for (table, vintage), unmapped-header check
    normalize.py     # dtype functions: ccn, date, bool_yn, severity, tag, footnote, enums, numeric
    bronze.py        # task: load_bronze[file]       -> TableSink bronze/<table>/vintage=YYYY-MM
    silver.py        # task: build_silver[table]     -> silver/<table>_by_vintage (+ vintage_month, cutoff_date)
    checks.py        # reconciliation, uniqueness, domains, null-rate, availability matrix
    report.py        # 02_harmonization.md, 02_unmapped_headers.csv, 02_column_availability.csv
configs/column_map.yaml  configs/keep_columns.yaml  configs/value_maps.yaml  configs/footnotes.yaml
```

**Task chain** (DAG-ready, like Step 1):
```
acquire run lock ─▶ select_files ─▶ load_bronze[file]* ─▶ map_check (fail if unmapped) ─▶ build_silver[table]* ─▶ checks ─▶ report + run manifest (🛑 Checkpoint 2) ─▶ release lock
```

| Topic | Design |
|---|---|
| Inputs | `inventory where status='selected'` (§1.2). Never re-derive vintages. |
| Bronze | As in §2.1. `ctx.table_sink.write_partition(table, {"vintage": vm}, df_all_string)` on local disk or GCS. Skip when the partition's `_zip_sha256`/`_header_hash` match the selected file; reload (overwrite) when they differ (D10). |
| Silver | One Parquet dataset per table on the same object store, partitioned by `vintage_month`, keep-list columns only, typed. **Not loaded into BigQuery or Postgres** (D2/D3). Silver partitions marked stale by a bronze reload are rebuilt. |
| Metadata | New append-only tables in the metadata store (Postgres locally, BigQuery `precursorintelligence_ops` in GCP): `bronze_files` (rows in/out, blank rows, encoding, fingerprints, status `loaded`/`skipped`/`superseded`), `silver_partitions` (table, vintage, source fingerprints, `content_hash`, `stale_refresh`, rebuild flag), `harmonize_checks`. Each task **reads these once and writes once in batch**; BigQuery writes use load jobs. |
| Run lock and manifest | Use Step 1's run lock (GCS lock object / Postgres advisory lock). Write `metadata/silver/run_date=YYYY-MM-DD/silver_run_<ts>.json` with inputs, outputs, fingerprints and counts (D11). |
| Orchestration | Airflow 3 `build_silver` DAG triggered by the Step 1 landing Asset; tasks run the `precursor-pipeline` image via `DockerOperator` (team tech stack). |
| Logging | Reuse `obs.logging` and `pipeline.task_scope` (adds `run_id`, `task`, `table`, `vintage` context). |
| Performance | Each file fits in memory. Use 3–4 parallel workers. Expect minutes, not hours, for the 18 GB. |
| Tests | Fixtures from real headers (`raw_headers_by_vintage.csv`): one per era, a cp1252 file with `0x92`, a blank-row file, a penalties file with duplicate fines, an alphanumeric CCN, a stale-refresh pair, and a republished file forcing a partition reload. The metadata tests run on SQLite and on local Postgres. |

---

## 11. Checkpoint 2: acceptance checks with expected values

| Check | Expected (from this analysis) |
|---|---|
| Unmapped headers | **0**. The draft map already covers every header in all 427 files. |
| Files loaded to bronze | **427** (61 × 7) |
| Row reconciliation | `bronze_rows + blank_rows == row_count` for every file. `blank_rows` = 1 in each file 2021-01 → 2024-06 (39 per table), 0 after. |
| Encoding | 151 files cp1252 (2021-01 → 2023-04), 276 UTF-8; 0 decode failures |
| CCN | 100% length 6; ~1.8% alphanumeric; 0 nulls after dropping blank rows |
| Dates | 0 unparseable among non-empty values (all ISO) |
| `scope_severity` | Domain ⊆ {B…L}; 0 invalid |
| Uniqueness | 0 duplicates for provider, citations, surveys, QM and lookup keys (§6). Penalties: exact duplicates reported (2,209), **retained** with `dup_seq`. |
| Facilities per vintage | 14,690 – 15,341 |
| Column availability | Matches §4.2: turnover and weekend from 2022-01, chain from 2023-06, IDR from 2023-01, lat/long from 2023-08, `fine_id` from 2026-06, `urban` from 2025-10 |
| Post-cutoff events | 217 rows in 3 vintages (§7.2), reported, not dropped |
| 2026-07 provider decision | Recorded (§7.4) |
| Stale refresh | Exactly 4 flags, all in vintage 2025-09 (health_citations, penalties, survey_dates, claims_qm vs 2025-07); none elsewhere (§7.6) |
| Lineage columns | Present on every bronze row; lineage types as in §2.1; `_zip_sha256` matches `inventory.zip_sha256` for every partition |
| Idempotency | A second run loads 0 bronze partitions and rebuilds 0 silver partitions; a forced republish reloads exactly the affected partition |
| Run manifest | `metadata/silver/run_date=…/silver_run_<ts>.json` written, with input and output fingerprints |

---

## 12. Open decisions for the team

1. **Penalties:** confirm the "keep duplicates + `dup_seq`" rule (recommended). This overrides guide §2e.
2. **2026-07 Provider Info:** keep the corrected 08-06 file (recommended; affects scoring rows only) or the original 07-29.
3. **Keep-list extras:** add `inspection_cycle`, IDR/IIDR, `survey_cycle`, MDS quarterly scores, and `nursing_case_mix_index` (2024-07+, ablation). Recommended: yes for the first four.
4. **"Not corrected" definition** for `n_uncorrected` (Step 7): which `Deficiency Corrected` values count. `No revisit needed` is new from 2025-10.
5. **Citation lookup in silver:** keep all vintages (recommended) or latest only, as the guide says.
6. **Region** for the buckets and both BigQuery datasets (recommended `us-central1`; must be the same for all).
7. **Metadata write pattern:** append-only plus a "latest" view (recommended) or `MERGE` statements.
8. **Naming alignment with the team documents:** the team plan calls the raw layer "bronze" (`bronze/pdc/vintage=YYYY-MM/`). This implementation uses `landing/` (raw, by zip date) and `bronze/` (Parquet, by month). Agree on one vocabulary before writing the Airflow DAGs (`STEP1_DATA_INGESTION_PIPELINE.md` §0.4).

**Already decided (2026-10-08):**
- Local disk + GCS only.
- No database or table adapter for intermediate layers.
- BigQuery holds only the final dataset.
- Metadata in local Postgres (development) and BigQuery (test and production).
- Bronze keeps every column as text.
- Republished data triggers a partition reload.

---

## 13. Reference files (`docs/data/step2_reference/`)

| File | Contents |
|---|---|
| `column_map.draft.yaml` | Canonical → aliases with `from`/`to` month ranges, dtype, `available_from`, `complete`; `ignored:` list per table. Machine-verified against all 427 files. |
| `raw_headers_by_vintage.csv` | Every raw header per table, with first and last vintage and number of vintages (243 rows) |
| `qm_measure_timeline.csv` | Every (measure code, resident type) with first and last vintage and latest description |
| `footnotes.draft.yaml` | Footnote code meanings (dictionary Table 15 plus retired codes), proposed missing-reason groups, codes observed per table |
| `value_maps.draft.yaml` | Survey-type enum, SFF / sprinkler / council maps, ownership group rule, Y/N map, citation survey-type rule |

Related documents:
- `docs/STEP1_DATA_INGESTION_PIPELINE.md`: Step 1 design and results.
- `data/reports/step1/latest/01_inventory_summary.md`: the Checkpoint 1 report.
- `dataset/Nursing Homes Current Data/NH_Data_Dictionary.pdf`: CMS data dictionary, July 2026; Tables 2–15 cover columns and footnotes, Table 16 the revision history.
