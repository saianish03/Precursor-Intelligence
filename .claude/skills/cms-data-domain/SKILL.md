---
name: cms-data-domain
description: CMS Nursing Home Provider Data Catalog knowledge for Precursor Intelligence. Use when reading, parsing, ingesting, harmonizing or building ledgers, features or labels from CMS archive zips, the metastore API, Health Citations, Inspection Dates, Penalties, Provider Information or Quality Measure files, or when reasoning about scope/severity codes, vintages, posting lags or schema changes.
---

# CMS data domain

## Sources and scope
- Provider Data Catalog (PDC) nursing-home datasets. History comes from the monthly archive zips
  (2021–2026 in scope; 2019–2020 are archived but not used). Live releases come from the metastore API:
  `https://data.cms.gov/provider-data/api/1/metastore/schemas/dataset/items/{dataset_id}`.
- Core dataset IDs: Provider Information `4pq5-n9py`, Health Citations `r5ix-sfxw`, Penalties `g6vv-u9sr`,
  Inspection Dates `svdt-c123`, MDS Quality Measures `djen-97ju`, Claims Quality Measures `ijh5-nb2v`,
  Citation Descriptions `tagd-9999`. They live in `configs/ingestion/datasets.yaml` (the dataset registry); read them from there.
- PBJ daily staffing is optional phase 2. Do not add it to the critical path.

## Vintage rules (the most common source of bugs)
1. The data month is the **Processing Date** column inside the files, never the zip name.
2. One snapshot per data month: prefer the complete zip (all core tables), then most provider rows,
   then the latest zip name. Expected: 61 data months 2021-01 → 2026-08, 7 gaps
   (2021-12, 2022-12, 2023-12, 2024-12, 2025-01, 2025-08, 2026-01).
3. Known oddities: duplicate zips sharing a Processing Date (2021-10, 2022-10, 2023-10, 2024-10);
   `2026-08-06` is a Provider-only re-post; **2025-09 citations, penalties, surveys and claims are a stale
   copy of 2025-07** (flag `stale_refresh`, still a valid public vintage).
4. Skip `__MACOSX/` and `._*` entries when classifying zip members.
5. Headers changed: `Federal Provider Number` (older) → `CMS Certification Number (CCN)`; chain columns were
   `Affiliated Entity *` before `Chain *`; turnover/weekend staffing appear from ~2022; Fine ID from 2026-06.
   Map every raw header through `configs/column_map.yaml`; an unmapped header must fail the run.
6. Dates appear as `YYYY-MM-DD` and `M/D/YYYY`; parse both.

## Table semantics
- **CCN is 6-character text.** `zfill(6)`, never int.
- Health Citations: one row per citation; `Scope Severity Code` A–L. G–L = actual harm or immediate jeopardy
  (G+); J–L = immediate jeopardy (IJ). Survey Type "Health"; flags `Standard Deficiency`, `Complaint Deficiency`,
  `Infection Control Inspection Deficiency`. Each file holds about 3 inspection cycles.
- Inspection Dates: one row per survey; `Type of Survey` in {Health Standard, Health Complaint,
  Fire Safety Standard, Fire Safety Complaint, Infection Control}. **Complaint surveys appear only if they
  produced citations**, so name such features `*_cited` (e.g. `n_complaint_surveys_cited_12m`).
- Penalties: exactly a 36-month window per file; Penalty Date is the triggering inspection date, not the date
  imposed. Posting regime changed at the 2021-11 vintage (all fines, not only paid ones); counts fell
  from ~40k (2023) to ~15.7k (2026). Use penalties as features, not as the primary label.
- Quality measures: MDS long-stay antipsychotic code changed 419 → 481 in 2026-02; 401–454 range mostly stable;
  4 claims measures (521, 522, 551, 552) stable. Bridge changed codes explicitly in `configs/qm_measures.yaml`.

## Ledgers and labels
- Ledger key for citations: (ccn, survey_date, deficiency_prefix, deficiency_tag); keep first_seen and
  last_seen vintages and the final severity from the last vintage seen.
- Label `y_g12`: any G+ citation with survey_date in (cutoff, cutoff + 12 months]; `y_g6` same with 6 months.
  Status: positive / negative / immature (window + buffer not elapsed) / censored (facility exited).
- Posting-lag buffers come from `configs/labels.yaml` (re-measured from 2021+ ledgers). Never hard-code them.

## Reference numbers (sanity checks)
- G+ share of standard surveys 2023+: ≈ 12.9%; IJ ≈ 4.6%. 77.6% of G+ citations (2023+) come from complaint
  inspections. Median gap between standard surveys ≈ 448 days. ~25% of facilities overdue (>456 days).
- Same-day fine given a G+ standard survey ≈ 74.5% (vs 4.3% without G+).
- State G+ rates range ~4% (NH) to ~29% (CO): always report within-state metrics too.
- If a rebuild drifts far from these, suspect parsing or vintage selection before modelling.
