# Gold schema findings for bias slicing (Ticket 5, step 0)

Checked against the gold code (`src/precursorintelligence/gold/`), `configs/feature_registry.yaml`, and a local
gold build (run `20261027T183821Z-f2ea91`, V = 2026-08: 578,975 rows, 15,274 facilities, 39 months from 2023-01 to
2026-08, 53 states and territories).

## Where the dataset lives

- Table: `gold/facility_month_dataset/t=YYYY-MM/part-0.parquet`, one partition per month, with keys `(ccn, t)`.
- Pointer: `gold/_current.json` lists the sha256 of every partition. `gold.build.read_current_gold(store)` checks
  them, and the bias loader calls it before reading (local disk in `dev`, GCS in `prod`, through the same
  `ObjectStore`).
- Run manifest: `gold/dataset_manifest.json`.

## Spec name → actual column

| Spec name | Actual column | dtype | Notes |
|---|---|---|---|
| `ccn` | `ccn` | String | 6-char, never null |
| `t` | `t` | String | `YYYY-MM` |
| `y_g12`, `y_g6`, `y_pen12` | same | Int8 | null where not labeled |
| `status_g12`, `status_g6`, `status_pen12` | same | String | `positive` / `negative` / `immature` / `censored` |
| `split_g12`, `split_g6`, `split_pen12` | same | String | `train` / `calib` / `purge` / `test` / `score` |
| `state` | `state` | String | 53 values (50 states, DC, PR, GU); no nulls |
| `ownership_group` | `ownership_group` | String | `for_profit` / `non_profit` / `government` (snake case, not the spec's hyphens) |
| `certified_beds` | `certified_beds` | Float64 | 57 nulls; range 0–874 |
| `chain_id` | `chain_id` | String | 240,781 nulls (41.6%); from 2023-06 a null → `independent`, before 2023-06 (column not published) → `missing` |
| `provider_in_hospital` | `provider_in_hospital` | Boolean | no nulls |
| `ccrc` | `ccrc` | Boolean | no nulls |
| `sff_status` | `sff_status` | String | `SFF` / `SFF_CANDIDATE` / `NONE`; no nulls |
| `overall_rating` | `overall_rating` | Float64 | 1–5, 5,718 nulls; `ablation_only` in the registry, used here for slicing only |
| `n_std_36m` | `n_std_36m` | Int64 | rater-bias exposure |
| `days_since_last_std` | `days_since_last_std` | Float64 | 1 null |
| `n_complaint_surveys_12m` | **`n_complaint_surveys_cited_12m`** | Int64 | renamed in gold; mapped in `configs/bias_slices.yaml` |
| `state_gplus_rate_12m` | `state_gplus_rate_12m` | Float64 | rater-bias rate column |
| registry features | `configs/feature_registry.yaml` | — | 95 features, read through `load_feature_registry()`; nothing hard-coded |
| `*_fn` footnote columns | 10 columns (`turnover_total_fn`, …, `overall_rating_fn`) | String | gold writes `"none"` for no footnote; scanned as "has reason" rates (the 8 `model` ones) |

No required column is missing, so nothing has to be derived from a substitute.

## Label rows (V = 2026-08)

| Label | positive | negative | immature | censored |
|---|---:|---:|---:|---:|
| `g12` | 87,442 | 236,373 | 250,392 | 4,768 |
| `g6` | 61,147 | 336,781 | 176,549 | 4,498 |
| `pen12` | 76,332 | 203,244 | 294,830 | 4,569 |

The split columns exist, so the skew check compares `train` against `test` and needs no time-thirds fallback on
the real gold.

## Things to know

- `ownership_group` levels are `for_profit` / `non_profit` / `government`, used as they are.
- `cms_region_map` covers every state in gold, including GU and PR; any unknown code becomes `missing` and is
  logged.
- The month gaps in gold (for example 2023-12 and 2025-01) come from vintages CMS never published. They are
  data, not a loader bug.
