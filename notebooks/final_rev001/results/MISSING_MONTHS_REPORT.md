# Missing data months — CMS Nursing Home archive (rebuild 2026-11-06)

Source: Step 1 inventory run `20261106T233009Z-fa9e59` (`data/reports/step1/latest/01_inventory_summary.md`)
and bronze run `20261106T233413Z-6dd511`. Scope: data months 2021-01 → 2026-10 (70 months).

## Summary

- **63 of 70 months** have data; **7 months are empty for all 7 tables** (provider_info, health_citations,
  survey_dates, penalties, mds_qm, claims_qm, citation_lookup).
- Nothing was lost in our pipeline: every zip CMS published was downloaded, passed its CRC check and was
  inventoried. The gaps exist because **no CMS file carries those months**.
- Bronze holds exactly 63 monthly partitions per table (2021-01 … 2026-10), matching the inventory.

| Empty month | Type of gap |
|---|---|
| 2021-12 | year-end shift |
| 2022-12 | year-end shift |
| 2023-12 | year-end shift |
| 2024-12 | year-end shift |
| 2025-01 | CMS published nothing |
| 2025-08 | CMS published nothing |
| 2026-01 | year-end shift |

## How a file gets its month

The data month (vintage) is taken from the **Processing Date inside the CSV files**, never from the zip's
date (spec `docs/DATA_PIPELINE_SPEC.md:45`, `ingestion/inventory.py`). The zip date is only when CMS built
the archive. If two zips carry the same data month, the later-published zip is used (`ingestion/resolve.py`);
in every such case this run, the two copies were identical.

## Why each month is empty

### Year-end shift (5 months)

Every autumn CMS re-issues the October data in an early-November zip, publishes November data in a
December zip, and the next zip already carries January (or February). No file is ever stamped December.

| Empty month | Zips around it (zip date → data month inside) |
|---|---|
| **2021-12** | 2021-10-27 → 2021-10 · 2021-11-27 → **2021-10** (repeat) · 2021-12-27 → 2021-11 · 2022-01-27 → 2022-01 |
| **2022-12** | 2022-10-27 → 2022-10 · 2022-11-27 → **2022-10** (repeat) · 2022-12-02 → 2022-11 · 2023-01-02 → 2023-01 |
| **2023-12** | 2023-10-02 → 2023-10 · 2023-11-02 → **2023-10** (repeat) · 2023-12-21 → 2023-11 · 2024-01-31 → 2024-01 |
| **2024-12** | 2024-10-30 → 2024-10 · 2024-11-07 → **2024-10** (repeat) · 2024-12-20 → 2024-11 · 2025-02-26 → 2025-02 |
| **2026-01** | 2025-11-20 → 2025-10 · 2025-12-15 → 2025-11 · 2026-01-14 → 2025-12 · 2026-02-25 → 2026-02 |

In the 2025–26 cycle the shift moved one month later: December 2025 has data, January 2026 is the one skipped.

### CMS published nothing (2 months)

| Empty month | Last zip before | Next zip after |
|---|---|---|
| **2025-01** | 2024-12-20 → 2024-11 | 2025-02-26 → 2025-02 (no zip dated in Jan 2025) |
| **2025-08** | 2025-07-30 → 2025-07 | 2025-09-24 → 2025-09 (no zip dated in Aug 2025) |

(2024-12 and 2025-01 together make a two-month hole: 2024-11 is followed directly by 2025-02.)

## Related data-quality notes

- **Stale release 2025-09:** bronze flagged `stale_refresh` for 2025-09 — its citations, surveys and penalties
  are identical to the previous available month (2025-07). It adds no new events to the ledgers.
- **Partial re-release 2026-08-06:** a small corrective zip containing only Provider Info (data month 2026-07);
  it replaced the 2026-07-29 Provider Info file. All other 2026-07 tables come from the 2026-07-29 zip.

## Impact downstream

1. **No gold rows** for the 7 empty months (one gold folder per data month; those folders will not exist).
2. **Look-back features** use t-6 and t-12. These months exist but one of their look-back months is empty,
   so those lag values may be missing (how gold handles a missing lag has not been checked yet):

   | Month | Missing look-back |
   |---|---|
   | 2022-06 | t-6 = 2021-12 |
   | 2023-06 | t-6 = 2022-12 |
   | 2024-06 | t-6 = 2023-12 |
   | 2025-06 | t-6 = 2024-12 |
   | 2025-07 | t-6 = 2025-01 |
   | 2025-12 | t-12 = 2024-12 |
   | 2026-02 | t-6 = 2025-08 |
   | 2026-07 | t-6 = 2026-01 |
   | 2026-08 | t-12 = 2025-08 |

3. **2021-12** only matters as history: 2021 is lookback-only for the 2022 rows.

## Options (team decision — the pipeline spec would change)

| Option | What it does | Trade-off |
|---|---|---|
| Keep as is (current design) | Empty months stay empty | Simplest and honest to the source; 7 rows-months fewer, some lags missing |
| Forward-fill | Treat the previous month's release as also valid for the empty month | Fills gold and lags; but duplicates a snapshot, so event counts in the window must not be double-counted |
| Nearest-available lag | Keep months empty, but let t-6 / t-12 fall back to the nearest earlier month | Fixes lag features only; no new gold rows |

Recommendation: keep the current design for this rebuild, and raise the forward-fill / nearest-lag choice with
the team as a spec change, since it affects label windows and point-in-time rules.
