# Silver contract and gold hand-off

**For the gold team.** What the `silver` node guarantees, how to read it, what gold may and may not use, and
what gold has to build. The rules come from `DATA_PIPELINE_SPEC.md` §3.2–§6 (silver) and §7–§11 (gold); this file
describes what the code in `src/precursorintelligence/harmonize/` actually does. Bronze's equivalent is
`BRONZE_CONTRACT.md`.

---

## 1. In one paragraph

Silver turns all-text bronze into **typed, cleaned tables, one per CMS table, partitioned by vintage**
(`*_by_vintage`), and builds the **all-vintage tables** gold needs for labels: three event ledgers, facility
presence, the facility universe and a QM inventory. Every row of a `*_by_vintage` table has `vintage_month` and
`cutoff_date`. Silver never computes features or labels. It also writes `gold/dim_facility_latest` (display
only; spec §6 puts it in the silver node).

## 2. State at hand-off (full archive, run locally 2026-10-18)

| Item | Value |
|---|---|
| Vintages | 61, **2021-01 → 2026-08** (V = 2026-08) |
| Never published | 2021-12, 2022-12, 2023-12, 2024-12, 2025-01, 2025-08, 2026-01 (no partitions; not errors) |
| `stale_refresh` | **2025-09** (its event tables equal 2025-07); `revised_after_publication`: none |
| Silver partitions | 488 (427 bronze partitions; provider_info feeds two tables) |
| Checks | 0 header problems, 0 unconvertible values, 0 duplicate keys; rerun is a no-op (byte-identical) |
| Reports | `reports/step2/latest/02_harmonization.md` (Checkpoint 2), `reports/step4/latest/04_lag_analysis.md` (Checkpoint 3), `reports/step5/latest/05_universe_and_qm.md` (Checkpoint 4) |
| Pending human decisions | buffers → `configs/labels.yaml`; QM list → `configs/qm_measures.yaml`; §14 #3, #4 (see §9) |

## 3. Where it lives and how to read it

```
silver/<table>_by_vintage/vintage=<YYYY-MM>/part-0.parquet     # 8 partitioned tables (§4)
silver/citation_lookup/vintage=<YYYY-MM>/part-0.parquet
silver/{ledger_citations,ledger_penalties,ledger_surveys,facility_presence,
        facility_universe,qm_inventory,citation_lookup_versions}/part-0.parquet   # single files (§5)
gold/dim_facility_latest/part-0.parquet                          # display only
metadata/silver/run_date=<YYYY-MM-DD>/silver_run_<run_id>_<ts>.json   # one manifest per run
```

Same keys on local disk (`./data`) and GCS. Read through the ports, never by path:

```python
from precursorintelligence.common.io.table_sink import ParquetTableSink
silver = ParquetTableSink(ctx.object_store, prefix="silver")
cit_t = silver.read("citations_by_vintage", {"vintage": "2026-08"})   # adds a `vintage` column (= vintage_month)

import polars as pl
with ctx.object_store.open_read("silver/ledger_citations/part-0.parquet") as f:
    ledger = pl.read_parquet(f)
```

Within a partition, rows are sorted by the table key, so files are deterministic.

**Freshness check before gold starts:** `silver_pipeline.partitions_without_current_silver(ctx, sink, cfg)` must
return `[]` (every bronze partition has a silver partition built from its current content and configs). The latest
silver manifest records the same check as `every_bronze_partition_has_current_silver`.

## 4. The `*_by_vintage` tables (feature inputs)

All have `vintage_month` (str `YYYY-MM`) and `cutoff_date` (date, the 1st of the vintage month = CMS
`Processing Date`). Types: `str`, `f64`, `i64`, `bool`, `date`. Nulls mean "missing"; bronze's `""` vs null
distinction is gone.

| Table | Grain / key (unique per vintage) | Columns |
|---|---|---|
| `provider_by_vintage` | facility; `ccn` | ccn, state, county_name, ownership_type, **ownership_group** (for_profit / non_profit / government), certified_beds, avg_residents_per_day (+`_fn`), provider_type, provider_in_hospital, date_first_approved, ccrc, special_focus_status (SFF / SFF_CANDIDATE / NONE), abuse_icon, high_performing_icon, most_recent_inspection_over_2y, changed_ownership_12m, resident_family_council (RESIDENT / FAMILY / BOTH / NONE), sprinkler_status (YES / PARTIAL / NO), sprinkler_status_not_available, overall_rating, health_inspection_rating, qm_rating, staffing_rating (each +`_fn`), reported_staffing_fn, pt_staffing_fn, hprd_{aide,lpn,rn,licensed,total,weekend_total,weekend_rn,pt}, adj_hprd_{aide,lpn,rn,total}, turnover_total, turnover_rn, admin_departures (each +`_fn`), chain_id, chain_name |
| `citations_by_vintage` | citation; `ccn, survey_date, deficiency_prefix, deficiency_tag` | ccn, survey_date, deficiency_prefix, deficiency_category, deficiency_tag (4-digit text, `0880`), **ftag** (`F0880`), scope_severity (letter; only B–L occur), deficiency_corrected (raw text), correction_date, **is_standard, is_complaint, is_infection_control** (a citation can carry several), inspection_cycle, citation_under_idr, citation_under_iidr, survey_fn |
| `surveys_by_vintage` | inspection; `ccn, survey_date, survey_type` | ccn, survey_date, survey_type (HEALTH_STANDARD, HEALTH_COMPLAINT, INFECTION_CONTROL, FIRE_STANDARD, FIRE_COMPLAINT), survey_cycle, survey_fn |
| `penalties_by_vintage` | penalty row; `penalty_event_key` | ccn, penalty_date (date of the **triggering inspection**), penalty_type (Fine / Payment Denial), fine_amount, payment_denial_start_date, payment_denial_length_days, fine_id (2026-06+, fines only), **dup_seq**, **penalty_event_key** (`ccn|date|type|amount|denial_start|dup_seq`) |
| `mds_qm_by_vintage` | `ccn, measure_code` | ccn, measure_code (3-digit text), measure_description, resident_type (Long Stay / Short Stay), four_quarter_avg (+`_fn`), used_in_5star, measure_period, **measure_period_start, measure_period_end** |
| `claims_qm_by_vintage` | `ccn, measure_code` | ccn, measure_code, measure_description, resident_type, adjusted_score, observed_score, expected_score, measure_fn, used_in_5star, measure_period, measure_period_start, measure_period_end |
| `citation_lookup` | `deficiency_prefix, deficiency_tag` | deficiency_prefix (F/K/E), deficiency_tag, deficiency_prefix_and_number, deficiency_description, deficiency_category |
| `provider_identity_by_vintage` | `ccn` | ccn, provider_name, provider_address, city, state, zip, latitude, longitude, chain_id, chain_name. **Not a feature input** (§6). |

**Columns that start late** (null before; documented in `reports/step2/latest/02_column_availability.csv`):
turnover, weekend staffing, admin departures 2022-01 · IDR/IIDR 2023-01 · chain_id/name 2023-06 · lat/long 2023-08 ·
fine_id 2026-06 · high_performing_icon, survey_fn (citations, surveys) 2026-09. The Rating Cycle 1/2
Standard Survey Footnote columns (also 2026-09) are ignored, like the Rating Cycle survey dates they annotate. Footnote and payment-denial columns are mostly null by design.

**Events after the cutoff are kept:** 217 rows (citations 2025-12: 167; surveys 2025-10: 6 and 2025-12: 44) are
dated up to 9 days after `cutoff_date`. Feature windows must use `event_date <= cutoff_date`.

## 5. The all-vintage tables (label inputs)

| Table | Rows | Key | Columns and meaning |
|---|---|---|---|
| `ledger_citations` | 971,858 | ccn, survey_date, deficiency_prefix, deficiency_tag | first_seen_vintage, last_seen_vintage, n_vintages_seen, **final_scope_severity**, final_category, final_is_standard / _complaint / _infection_control (values from the **last** vintage it appeared in), severity_changed, **removed_by_cms** (vanished while the facility and other tags of the same survey stayed: deleted, e.g. after a dispute) |
| `ledger_penalties` | 110,511 | penalty_event_key | first/last_seen_vintage, n_vintages_seen, ccn, penalty_date, penalty_type, fine_amount, payment_denial_start_date, payment_denial_length_days, dup_seq, fine_id (latest non-null), **revision_group** (`ccn|penalty_date|penalty_type`; 71,709 groups) |
| `ledger_surveys` | 483,894 | ccn, survey_date, survey_type | first/last_seen_vintage, in_inspection_dates (all true on the current archive) |
| `facility_presence` | 15,636 | ccn | first_vintage, last_vintage, n_vintages, gap_vintages (published months missing in between), **exited** (`last_vintage < V`; 946) |
| `facility_universe` | 914,352 | ccn, vintage_month | every facility in every published vintage (14,690–15,340 per month) |
| `qm_inventory` | 26 | table, measure_code, resident_type, measure_description | first_vintage, last_vintage, n_vintages |
| `citation_lookup_versions` | 664 | deficiency_prefix, deficiency_tag, valid_from_vintage | content runs; **valid_to looks into the future: not for features** |
| `gold/dim_facility_latest` | 15,636 | ccn | name, address, city, state, zip, lat/long, chain from the last vintage; last_vintage; is_active (14,690). **Product display only.** |

Ledgers are rebuilt over all vintages every silver run; the stale vintage 2025-09 adds no events.

## 6. What gold may read (leakage rules, spec §0.4 and §8)

| Gold part | May read | Must never read |
|---|---|---|
| **Features** for row `(ccn, t)` | `*_by_vintage` and `citation_lookup` with `vintage_month == t`; for Δ features also t−6 and t−12 | ledgers, `facility_presence`, `facility_universe` beyond t, `citation_lookup_versions`, `qm_inventory`, `provider_identity_by_vintage`, `dim_facility_latest` |
| **Spine** | `facility_universe` at t | — |
| **Labels** | `ledger_citations`, `ledger_penalties`, `facility_presence` (censoring) | `*_by_vintage` "latest" values |
| **Product / UI** | `dim_facility_latest` | (ML training must never join it) |

Spec §8: the feature reader should **record every read and refuse anything else** (a reader that raises on a
vintage > t is also the in-gate time-travel test, §11.3).

## 7. What the gold team has to build (spec §7–§11)

Suggested home: the empty `src/precursorintelligence/features/` package (plus label/split/assembly modules) and
`tests/features/`, `tests/leakage/`. A `gold` command next to `bronze` and `silver` in `harmonize/cli.py`.

**Target output:** the features and merge rules of `DATA_PIPELINE_SPEC.md` §7–§10, summarised with the column
list and per-table join rules in §13.

| Step | Output | Key rules |
|---|---|---|
| **6 Spine** (§7) | `gold/spine` | one row per `(ccn, t)` for every published t in **2023-01 → V** and every ccn in `facility_universe` at t. Assert unique; all published t present. Every later table: `spine LEFT JOIN … USING (ccn, t)`, row count unchanged after each join. |
| **7 Features** (§8) | `gold/features_<family>/t=YYYY-MM/` for 6 families | survey_timing, citations, enforcement, facility_staffing, quality, context. Windows look back from `cutoff_date` (`cutoff − 12m < date <= cutoff`). Severity points §8.1. Zero = none found; nulls stay null with `_fn` reasons; Δ is null if t−6/t−12 is unpublished or the facility absent. Register every feature in `configs/feature_registry.yaml` (`model` / `ablation_only` / `product_only`). |
| **8 Labels** (§9) | `gold/labels` | see §8 below |
| **8 Splits** (§9.5) | `split_g12`, `split_g6`, `split_pen12` in `gold/labels`; `configs/splits.yaml` | calendar months ending at `last_labeled_t`; test = last `test_months`; purge = H months before test; calib = 3 months before purge; train from spine start; after `last_labeled_t` = `score`. Gate: exit 2 if a `required_labels` label has no training month. |
| **9 Assemble** (§10) | `gold/facility_month_dataset/t=YYYY-MM/`, `gold/dataset_manifest.json` | join order spine → 6 feature tables → labels; manifest lists every (table, vintage) source zip + sha256 (from `bronze_files` / `inventory`), `stale_refresh` / revised flags, config and registry hashes, git commit, rows per split per label. |
| **10 Verify + publish** (§11) | `gold/_staging/run=<id>/` → `gold/`; `reports/step10/10_verification.md` | structural, label-logic, leakage (time travel, single-feature scan, shuffled labels) and coverage checks; any failure → exit 2, `gold/` untouched. Then load to the warehouse (Postgres `gold.facility_month_dataset` locally, BigQuery in GCP), record in `gold_builds`, emit `cms_nh_gold`. **Checkpoint 5**. |

## 8. Labels from the silver ledgers (spec §9)

`W_H = (cutoff_date, cutoff_date + H months]`.

| Label | = 1 if | Buffer B |
|---|---|---|
| `y_g12` / `y_g6` | a `ledger_citations` row for the ccn with `survey_date ∈ W_H`, `final_scope_severity ∈ {G..L}`, **`removed_by_cms = false`** (all health survey types) | `B_cit` |
| `y_pen12` (`y_pen6`) | a **`revision_group`** in `ledger_penalties` with `penalty_date ∈ W_H` (fine or denial). Never key on `fine_id` or `penalty_event_key` for events. | `B_pen` |

Status, in this order: **immature** (`t + H + B > V`; `y` null even if an event is visible) → **positive** →
**censored** (no positive and `facility_presence.last_vintage < t + H + B`) → **negative**. If V is a stale
vintage, use the last non-stale V (2025-09 → 2025-07).

**Proposed buffers (Checkpoint 3, pending approval): `B_cit = 8`, `B_pen = 12`.** Read them from
`configs/labels.yaml` once committed; never hard-code. At V = 2026-08: last labelled t = 2024-12 (`y_g12`; last
real month 2024-11), 2025-06 (`y_g6`), 2024-08 (`y_pen12`). With spine start 2023-01, `y_g12` has very few training
months; spec §9.5 recommends training rows from 2022-01 (decision §14 #10).

## 9. Inputs gold needs that are not final yet

| Config | Status | Owner |
|---|---|---|
| `configs/labels.yaml` (horizons, `B_cit = 8`, `B_pen = 12`, G+ set, due interval 456 days, `censor_requires_buffer: true`) | proposal in `04_lag_analysis.md`, awaiting approval | silver owner commits after Checkpoint 3 |
| `configs/qm_measures.yaml` (14 stable, bridges 453→479 and 419→481, **405→480 doubtful**, 5 dropped) | proposal in `05_qm_measures.proposed.csv`, awaiting approval | silver owner commits after Checkpoint 4 |
| `value_maps.yaml: not_corrected_values` (for `n_uncorrected`) | default = the two "Deficient, Provider has (no) plan of correction" values; §14 #4 open | team |
| `configs/splits.yaml`, `checks.yaml`, `feature_registry.yaml` | not started | gold |

## 10. Pitfalls (each one has bitten the sample run or the analysis)

- **CCN is a 6-character string** (1.7% alphanumeric, e.g. `01A193`). Never cast to int.
- **Penalties:** for **features**, every row counts (including `dup_seq > 1`; they are separate fines). For
  **labels and lag**, the event is the `revision_group`. Fine counts step up from 2025-10: give the model the `era` feature.
- **Citation survey type** comes from the three `is_*` flags, not a column. Complaint surveys in
  `surveys_by_vintage` exist only if they produced a citation: name such features `*_cited`.
- **Star ratings** (`overall_rating`, `health_inspection_rating`) are `ablation_only`; `chain_id` is `product_only`.
- **`high_performing_icon`** (2026-09+) is a status flag CMS assigns, like the star ratings: it is
  `ablation_only` in the contract CSV (guarded by `tests/contracts/test_feature_contract.py`), never `model`.
  It is null before 2026-09.
- **QM bridges:** `measure_code` changes (405→480, 453→479 in 2025-02; 419→481 in 2026-02). A Δ across a code
  change, or across an unchanged `measure_period_end`, must be null (§8.6).
- **Unpublished months** have no partitions: t−6 / t−12 lookups must handle absence (Δ = null).
- **Stale vintage 2025-09:** keep its features; it adds no label events; maturity is measured from 2025-07 when it is V.
- **Do not load bronze or silver into BigQuery/Postgres**; only the final gold dataset goes to the warehouse.

## 11. Run contract to follow (same as bronze and silver)

- `precursor-pipeline gold --run-id R` inside the DAG (the run must hold the `data_pipeline` lock:
  `RunLock(ctx.metadata, LOCK_NAME).require(run_id)`); without `--run-id` it takes and releases the lock itself.
- Exit codes: 0 ok · 1 error (retried) · 2 data/validation or gate failure (no retry) · 3 config (no retry).
- Idempotent and deterministic: unchanged inputs give byte-identical outputs; labels are recomputed every run.
- One manifest per run under `metadata/gold/run_date=…/` with a timestamp in the name.
- Windows note: Airflow-style run ids contain `:`, which local-disk keys on Windows cannot hold; run locally
  without `--run-id` or in Docker.

## 12. Commands

```bash
uv run python --version                        # 3.11 or 3.12
uv run precursor-pipeline bronze --verify      # 427 / 427
uv run precursor-pipeline silver               # stacked tables + ledgers + universe (no-op if nothing changed)
uv run precursor-pipeline silver --calibrate   # also Checkpoints 3-4 (backfill / on demand)
uv run pytest tests/harmonize -q               # silver + bronze tests
```

## 13. Gold columns and how they are merged

`DATA_PIPELINE_SPEC.md` §7–§10 is the authority. The reference Colab dataset `training_dataset_v1`
(`notebooks/DATA_preprocessing_notebooks/`) built the same features the same way, so it is used only as a
cross-check of silver (13.3). The full column list, with dtype, `model_use`, operation, window, fill value and
silver source, is **`configs/contracts/facility_month_dataset_spec_columns.csv`** (159 columns: the reference's 153
under spec names, plus the 5 spec-only columns marked `spec-only addition` and `high_performing_icon`, a
silver addition from 2026-09). A column or two more or less is fine;
the features and the merge rules below are what must hold.

| Family (gold table) | Cols | Silver input (vintage t only, except Δ) |
|---|---|---|
| keys `ccn, t, cutoff_date` (`gold/spine`) | 3 | `facility_universe` at t |
| `features_survey_timing` | 9 | `surveys_by_vintage` |
| `features_citations` | 26 | `citations_by_vintage` (+ `surveys_by_vintage` for last/previous standard survey, `certified_beds`) |
| `features_enforcement` | 7 | `penalties_by_vintage` |
| `features_facility_staffing` | 47 | `provider_by_vintage` at t; `d6_*` / `d12_*` also at t−6 / t−12 |
| `features_quality` | 43 | `mds_qm_by_vintage`, `claims_qm_by_vintage` (17 measures + 17 `_fn`, 4 `d6_qm_*`, `qm_missing_count`, `qm_age_months_{mds,claims}`, `qm_def_post_2025_02`, `qm_def_post_2026_02`) |
| `features_context` | 6 | the feature tables above, grouped by state within t; `era`, `penalty_regime_2025_10` |
| `labels` (+ splits) | 17 | `ledger_citations`, `ledger_penalties`, `facility_presence` |

### 13.1 How each table is merged (spec §7, §8, §10)

| From (silver, at t) | Grain | Aggregate before the join | Join to | Produces |
|---|---|---|---|---|
| `facility_universe` | ccn | none | — (base table) | `gold/spine`: `ccn, t, cutoff_date` |
| `surveys_by_vintage` | inspection | health types only, `survey_date <= cutoff`; COUNT DISTINCT dates / latest date per ccn | spine, LEFT on `(ccn, t)` | survey_timing |
| `citations_by_vintage` ⋈ `surveys_by_vintage` | citation | INNER on `(ccn, survey_date)` to pick the latest two HEALTH_STANDARD dates; then COUNT / SUM points / MAX ordinal / latest date per ccn | spine, LEFT on `(ccn, t)` | citations (`max_sev_*`, `gplus_*`, `n_repeat_tags_last_std`, counts) |
| `penalties_by_vintage` | penalty row (every row, incl. `dup_seq > 1`) | COUNT / SUM (log1p) / MAX / latest date per ccn | spine, LEFT on `(ccn, t)` | enforcement |
| `provider_by_vintage` | ccn | none (1:1) | spine, LEFT on `(ccn, t)` | facility_staffing |
| `provider_by_vintage` at t−6 / t−12 | ccn | value(t) − value(t−k); **null if t−k unpublished or the facility absent** | provider at t, LEFT on ccn | `d6_*`, `d12_*` |
| `mds_qm_by_vintage` / `claims_qm_by_vintage` | ccn × measure | filter to `qm_measures.yaml`, map bridged codes to one name, PIVOT measure → column | spine, LEFT on `(ccn, t)` | `qm_*`, `qm_*_fn` |
| QM at t−6 | ccn × measure | value(t) − value(t−6); null if `measure_code` differs or `measure_period_end` unchanged | QM at t, LEFT on ccn | `d6_qm_*` |
| feature tables at t | facilities in a state | MEDIAN / MEAN by state within t | spine, on `(ccn, t)` | context |
| `ledger_citations` | citation (final severity) | ANY(G–L, `removed_by_cms = false`) with `survey_date ∈ (cutoff, cutoff + H]` | spine, LEFT on `(ccn, t)` | `y_g12`, `y_g6`, aux |
| `ledger_penalties` | `revision_group` | ANY(fine or denial) with `penalty_date ∈ (cutoff, cutoff + H]` | spine, LEFT on `(ccn, t)` | `y_pen12`, `y_pen6` |
| `facility_presence` | ccn | `last_vintage < t + H + B` → censored | labels, LEFT on ccn | `status_*` |

Assembly order (§10): spine → survey_timing → citations → enforcement → facility_staffing → quality → context →
labels. After **every** join: row count = spine, `(ccn, t)` unique. Fill when nothing is found: counts and sums 0,
booleans `False`, dates / ratios / Δ null, `_fn` `'none'`. Features never read ledgers, `facility_presence` or any
vintage after t (§6).

### 13.2 Value conversions gold applies (silver keeps CMS codes)

| Column(s) | Silver | Gold |
|---|---|---|
| every `_fn` (provider and QM) | footnote code as text, null = none | null → `none`; 1 `new_facility` · 2 `insufficient_data` · 6 `staffing_criteria_not_met` · 9 `too_few_residents` · 10 `missing_not_submitted` · 18 `sff_not_rated` · 20 `rating_not_validated` · 23 `no_staffing_data` · 24 `many_days_without_rn` · 25 `staffing_not_validated` · 26 `turnover_invalid_min_points` · 27 `turnover_criteria_not_met`; `survey_fn` (2026-09+) has only 29 so far, meaning not yet confirmed (names as in the reference; keep them in a config) |
| `sff_status` | `special_focus_status` (`SFF / SFF_CANDIDATE / NONE`) | same values, renamed column |
| `n_cat_*_24m` | `deficiency_category` (ends in ` Deficiencies`) | abuse = Freedom from Abuse, Neglect, and Exploitation; qol_care = Quality of Life and Care; assessment = Resident Assessment and Care Planning; nursing_physician = Nursing and Physician Services; resident_rights = Resident Rights; administration = Administration |
| `n_ic_cit_24m` | category `Infection Control Deficiencies` **or** `is_infection_control` | — |
| QM values | MDS `four_quarter_avg`, claims `adjusted_score` | `qm_<name>`; codes: ls_adl_decline 401 · ls_antianxiety 452 · ls_antipsychotic 419 → 481 · ls_catheter 406 · ls_depressive 408 · ls_ed_per_1000 552 · ls_falls_major_injury 410 · ls_hosp_per_1000 551 · ls_incontinence 405 → 480 · ls_mobility_decline 451 · ls_pressure_ulcer 453 → 479 · ls_restraints 409 · ls_uti 407 · ls_weight_loss 404 · ss_ed_visit 522 · ss_new_antipsychotic 434 · ss_rehosp 521 (the Checkpoint 4 proposal) |

### 13.3 Cross-check: silver reproduces the reference inputs (measured 2026-10-19)

| Check | Result |
|---|---|
| Spine: `facility_universe` vs reference, all 28 reference months (2024-01 → 2026-08) | identical ccn sets, 413,729 rows |
| 26 pass-through provider columns, all 28 months | 0 mismatches |
| 17 QM values, months 2024-01, 2025-02, 2026-02, 2026-08 | 0 mismatches |
| `n_cit_24m`, `sev_points_24m`, same 4 months | 0 mismatches |
| `n_fines_24m`, `fine_dollars_24m` | 0 mismatches except 2026-02 (166 facilities, 197 `dup_seq > 1` fines the reference dropped). Spec keeps them (§3.5); expected. |

Where the reference differs from the spec, follow the spec: `*_cited` names, `qm_def_post_2025_02/_2026_02`,
`era = pre_2025_07`, Δ null instead of falling back a month, spine from 2023-01, buffers from `labels.yaml`,
censoring with the buffer, a `calib` split, duplicate fines kept.
