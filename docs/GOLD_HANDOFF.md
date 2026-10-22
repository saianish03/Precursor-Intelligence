# Gold hand-off (features and assembly)

For the teammate who builds the **features** (`DATA_PIPELINE_SPEC.md` §8) and the **assembly** (§10). What the
gold node already does, how to run it, how to test against the toy world, and exactly what is left to build.
The rules come from the spec §7–§11 and `docs/SILVER_CONTRACT.md` (silver, merged into `dev` in PR #7); this file describes
what the code in `src/precursorintelligence/gold/` actually does. Issue #8 covers the work described here.

---

## 1. In one paragraph

Gold already builds the **spine** (one row per facility × month) and the **labels** (`y_g12`, `y_g6`, `y_pen12`
with a status each) from silver, and runs as `precursor-pipeline gold`. It takes the run lock, measures labels
from the last fresh vintage, builds in a staging folder, checks there, copies to `gold/` only if every check
passes, and writes one manifest per run. **Features and the final dataset are not built yet**: that is your
part. A small **toy silver world** with hand-written answers lets you test every feature row by row.

**Who does what**

| Who | Done | Next |
|---|---|---|
| Sai (gold lead) | Spine, labels, `precursor-pipeline gold` | Splits (§9.5), the final checks and verification gate (§11), publishing to the warehouse (§11.5) |
| You (features partner) | — | The 6 feature families (§8), the leakage-safe feature reader, `configs/feature_registry.yaml`, the assembly and `dataset_manifest.json` (§10) |

**Start here**

1. **Get the branch, then make your own:** `git fetch origin`, `git checkout feat/gold-foundations`, then create
   your branch from it, e.g. `git checkout -b feat/gold-features`, and run `uv sync --all-extras`. Work only on
   your branch; **never commit to `feat/gold-foundations`**.
2. **Run the tests:** `uv run pytest tests/gold tests/leakage`. Everything passes except the two waiting feature
   tests, which show as `xfail`.
3. **Run gold on the toy world:** `uv run python tests/gold/make_toy_silver.py data/toy`, then
   `uv run precursor-pipeline gold --env-file data/toy/env.yaml`, and open `data/toy/gold/labels/part-0.parquet`.
4. **Read sections 8–10** below: what you build, the leakage rules, the join rules.
5. **Build in this order**, each piece with its own toy-world tests and a leakage test:
   - the feature reader;
   - `survey_timing`, `citations`, `enforcement`, `facility_staffing`, `quality`, `context`;
   - then the assembly.
6. **Turn on the waiting tests:** when `enforcement` and `quality` exist, remove `@not_built` from the two
   waiting tests (section 8).
7. **Commit and push:** one small commit per piece (section 11). Push the branch as a backup after each one,
   and open the PR only when the team agrees.

## 2. What is built

| Piece | File | What it does |
|---|---|---|
| Spine (§7) | `gold/spine.py` → `build_spine` | Every CCN in `facility_universe` at each published t from `spine_start` (2023-01) to V. Fails on a duplicate `(ccn, t)`, a published month without rows, or rows for a never-published month. |
| Labels (§9.1–9.2) | `gold/labels.py` → `build_labels` | Reads only `ledger_citations`, `ledger_penalties`, `facility_presence`. Window `(cutoff_date, cutoff_date + H months]`. Status order **immature → positive → censored → negative**; censored = `last_vintage < t + H + B`. Penalty events counted once per `revision_group`. |
| Run (§11) | `gold/build.py` → `run_gold` | Lock, V, silver freshness check, label vintage, `--v` checks, staging, promotion, manifest (§4). |
| Command | `gold/cli.py`, registered in `harmonize/cli.py` | `precursor-pipeline gold` (§5). |
| Configs | `configs/labels.yaml`, `configs/splits.yaml` | G–L, `B_cit = 8`, `B_pen = 12`, strict censoring, horizons; `spine_start: "2023-01"`. Loaded by `load_labels_config()` / `load_splits_config()` in `common/config.py`. Silver's lag analysis (`harmonize/calibrate.py`) reads its G+ set and horizons from `labels.yaml` too; the file is not part of silver's `config_hash`. |

**Not built:** features (§8), splits (§9.5), aux labels (§9.4), assembly and `dataset_manifest.json` (§10),
the verification report and warehouse publish (§11), `feature_registry.yaml`, `checks.yaml`. Who builds each
is in §1.

## 3. Output files and columns

```
gold/spine/part-0.parquet                  # zstd; sorted by t, then ccn
gold/labels/part-0.parquet                 # same rows, same order as the spine
gold/_staging/run=<run_id>/{spine,labels}/part-0.parquet   # what each run built and checked (kept)
metadata/gold/run_date=<YYYY-MM-DD>/gold_run_<run_id>_<timestamp>.json   # one per run, never overwritten
```

| Table | Columns (type) |
|---|---|
| `gold/spine` | `ccn` (str, 6 characters), `t` (str `YYYY-MM`), `cutoff_date` (date, the 1st of t) |
| `gold/labels` | `ccn`, `t`, then for each of `g12`, `g6`, `pen12`: `y_<label>` (Int8: 1, 0 or null) and `status_<label>` (str: `positive`, `negative`, `censored`, `immature`) |

`y` is 1 only for `positive`, 0 only for `negative`, null for `censored` and `immature`. Training uses only the
positive and negative rows of the label being trained.

The **manifest** records `run_id`, `exit_code`, `v`, `v_labels`, `stale_refresh_skipped`, the published vintages
(count, first, last), both configs, rows per output, the sha256 of each promoted file, `promoted` and `failures`.

## 4. How a run works

1. **Lock.** Without `--run-id` the run takes and releases the `data_pipeline` lock itself; with `--run-id` the
   DAG's run must already hold it.
2. **V.** The published vintages are the `silver/provider_by_vintage` partitions. V is the latest, or `--v`, which
   must be `YYYY-MM`, a published month, not after the latest, and not before `spine_start` (else exit 3).
3. **Silver is current.** Silver's own `partitions_without_current_silver` (`SILVER_CONTRACT.md` §3) must
   return nothing: every bronze partition has a silver partition built from its current content and the current
   silver configs. Otherwise the run stops (exit 2) and lists the partitions; it also stops when the metadata
   database has no bronze records, since nothing could be checked.
4. **Label vintage.** Read `vintage_flags` (`flag = stale_refresh`) from the metadata store and step back from V
   past stale months (2025-09 → 2025-07). The spine still runs to V. If a month it needs has **no flag row, the
   run stops (exit 2)** with a message naming the month; it never guesses.
5. **No peeking.** Ledger events with `first_seen_vintage` after the label vintage are dropped.
6. **Stage, check, promote.** Write to `gold/_staging/run=<id>/`, read back, check that the spine is unique and
   the labels have exactly the spine's rows; then copy the same bytes to `gold/`. Any failure: exit 2, `gold/`
   untouched. Reruns on unchanged silver give byte-identical files.
7. **Manifest.** Written for every run, also a failed one, with the freshness check's result.

## 5. Running it

```bash
uv run precursor-pipeline gold                         # local: dev env (./data), takes and releases the lock
uv run precursor-pipeline gold --run-id R              # DAG task: the run must hold the lock
uv run precursor-pipeline gold --v 2025-07             # an earlier published V
uv run precursor-pipeline gold --env-file <yaml>       # any other store / metadata database
```

| Exit | Meaning |
|---|---|
| 0 | Spine and labels built and promoted |
| 1 | Error: another run holds the lock, or a bug or outage. Retry |
| 2 | Data or gate failure (silver not current, missing silver input, missing `vintage_flags`, a failed check). `gold/` untouched |
| 3 | Config or usage error: bad `--v`, bad config, a DAG task whose run does not hold the lock |

**On the toy world:**

```bash
uv run python tests/gold/make_toy_silver.py data/toy                 # silver, metadata.db, env.yaml
uv run precursor-pipeline gold --env-file data/toy/env.yaml
uv run python tests/gold/make_toy_silver.py data/toy --stale 2025-03 # try a stale V
```

**On real data:**

1. **Get silver's output:** it comes as a zip from the silver owner (or from Sai). Unzip it so you have
   `data/silver/` and `data/gold/dim_facility_latest/`.
2. **Get the metadata database:** copy bronze's real `data/ops/metadata.db` from Sai or the silver owner. It
   holds `vintage_flags` and the bronze and silver records that the freshness check reads, so it must come with
   the same silver build.
3. **Run gold:** make sure `PI_DATABASE_URL` is not set, then run `uv run precursor-pipeline gold`.

> **`vintage_flags`.** The first real-data run used a temporary stand-in `data/ops/metadata.db` (only 2025-09
> marked `stale_refresh`). Gold was then rerun with the **real metadata database** that bronze wrote (61
> `stale_refresh` rows, only 2025-09 true), and the output was **byte-for-byte identical**, so the results in
> section 6 hold. Use the real database; the stand-in is no longer needed.

> **Silver must be built with the silver configs on `dev`.** The silver shared so far was built before PR #7's
> last config change ("Trim key codes", `configs/column_map.yaml`). Its `config_hash` differs from `dev`'s, so
> the freshness check reports all 488 partitions as not current and gold stops (exit 2). Get a silver rebuilt
> from `dev` (`uv run precursor-pipeline silver`, by whoever has bronze) together with its `metadata.db`.

> **Generate gold yourself.** Don't copy `gold/spine` or `gold/labels` from someone else: run
> `uv run precursor-pipeline gold` to build them from silver (it takes seconds). That way they always match your
> silver, your configs and the current code, and the run writes its own manifest.

Windows: Airflow-style run ids contain `:`, which local-disk keys on Windows cannot hold. Run locally without
`--run-id` (the generated id has no `:`) or in Docker.

## 6. Results on the real silver

Silver built by PR #7's code: 61 vintages 2021-01 → 2026-08, V = 2026-08, `stale_refresh` = 2025-09. (This run was
before gold had the freshness check; see the note in section 5.)

| Measure | Value |
|---|---|
| Spine and label rows | **578,975** (39 published months 2023-01 → 2026-08) |
| `y_g12` base rate | **27.0%** |
| `y_g6` base rate | **15.4%** |
| Last labelled t | `y_g12` 2024-12 (last real month 2024-11), `y_g6` 2025-06, `y_pen12` 2024-08, as spec §9.3 expects |

## 7. The toy world (use it for feature tests)

`tests/gold/silver_fakes.py` builds every silver table of `SILVER_CONTRACT.md` §4–§5 with the same names, column
order and types: running silver's own ledger code (`harmonize/ledgers.py`) on the toy `*_by_vintage` tables reproduces the toy ledgers
exactly. Reports 2023-01 → 2025-03, 2023-12 never published, V = 2025-03.

| CCN | Name | Story |
|---|---|---|
| 015001 | Sunny Acres | G citation on a standard survey 2023-06-15 (posted 2023-08); $15,000 fine the same day (posted 2023-09) |
| 015002 | Oak Grove | Only D and E citations; two identical $5,000 fines on 2023-10-05 (posted 2024-01, `dup_seq` 1 and 2) |
| 01A193 | Pine View | Last report 2023-08 (closes); alphanumeric CCN |
| 015004 | Maple House | In every report; G citation on a complaint survey 2024-02-10, **posted late** (first in 2024-06) |
| 015005 | Cedar Hill | First report 2023-04 |

Quality: 3 MDS (410, 419 → 481, 434) and 2 claims (521, 551) measures per facility; the long-stay
antipsychotic code is **419 up to 2024-05 and 481 from 2024-06**. MDS period for 2024-08 = `2023Q2-2024Q1`.

**How to use it in tests:**
- **Fixtures:** the `silver` fixture (`tests/gold/conftest.py`) gives `{table name: DataFrame}`. Each
  `*_by_vintage` frame is all vintages stacked, with `vintage_month` and `cutoff_date`.
- **Spine:** `tests/gold/test_expected_answers.py` has a module-scoped `spine` fixture; move it into `conftest.py`
  if your tests need it.
- **On disk:** `silver_store` (or `make_toy`) writes the same tables at their real paths for end-to-end tests.
- **Answers:** write expected feature values by hand from the story above, as the label tables do.

**Limits of the toy world:**
- **No history before 2023-01**, so every t−6 / t−12 change feature is null for the early months.
- **Constant provider values**, so staffing changes are 0.
- **Only F-tags**; no infection-control or fire surveys.
- **No payment denials.**

## 8. What you build

1. **`src/precursorintelligence/gold/features.py`** (or a package), with
   `build_features(family: str, silver: dict[str, pl.DataFrame], spine: pl.DataFrame) -> pl.DataFrame`,
   one row per spine row, for the 6 families of §8: `survey_timing`, `citations`, `enforcement`,
   `facility_staffing`, `quality`, `context`. The target columns are in §8 and in
   `configs/contracts/facility_month_dataset_spec_columns.csv` (158 columns).
2. **The leakage-safe feature reader** (spec §8): the only way feature code reads silver. It records every read
   and refuses anything outside the allowed inputs or after t (section 9 below); it is also the time-travel test of §11.3.
3. **`configs/feature_registry.yaml`**: every feature with `name, family, source_table, formula,
   first_available_vintage, dtype, model_use` (`model` / `ablation_only` / `product_only`).
4. **Assembly (§10)**: `gold/facility_month_dataset/t=YYYY-MM/` and `gold/dataset_manifest.json`. Join the
   split columns from `gold/labels` once the splits exist (§9.5, built by Sai).
5. **Wire into the run**: add your tables to the dict passed to `_stage` in `gold/build.py`, so they are built,
   checked and promoted with the spine and labels. Features are partitioned by t (`gold/features_<family>/t=…/`);
   `_stage` and `_promote` currently write single files, so extend them for partitions.

**Quality measures:** the `quality` family reads **`configs/qm_measures.yaml`** once it exists. The silver owner
commits it after Checkpoint 4 (`SILVER_CONTRACT.md` §9): 14 stable measures, bridges 453 → 479 and 419 → 481,
405 → 480 doubtful. The canonical names are listed in `SILVER_CONTRACT.md` §13.2. Don't hard-code the list.

**The two waiting tests** (`tests/gold/test_expected_answers.py`, bottom). They are `xfail(strict=True,
raises=ModuleNotFoundError)`, so they turn into real tests as soon as the module exists: then **remove the
`@not_built` marker**.
- The helper `_features(silver, spine, family)` imports
  **`precursorintelligence.gold.features.build_features`** and calls `build_features(family, silver, spine)`. If
  you choose another home (e.g. the empty `features/` package that `SILVER_CONTRACT.md` §7 suggests), change only
  that helper.
- `test_fine_counts_use_only_what_was_posted_at_t`: family **`enforcement`**, column **`n_fines_24m`**.
  Expected: Sunny Acres 2023-08 = 0 (not posted yet), 2023-09 = 1; Oak Grove 2023-11 = 0, 2024-01 = **2** (the
  `dup_seq` 2 twin counts); Maple House 2024-01 = 0.
- `test_d6_antipsychotic_is_null_across_the_code_change`: family **`quality`**, column
  **`d6_qm_ls_antipsychotic`**. For Sunny Acres it must be null exactly in 2023-01 → 2023-06 (no t−6 in the toy
  data), 2024-06 (t−6 = 2023-12, never published) and 2024-07 → 2024-11 (419 at t−6, 481 at t), and non-null in
  every other month.

## 9. Feature leakage rules (spec §0.4, §8; SILVER_CONTRACT §6)

- **Inputs:** only the `*_by_vintage` tables and `citation_lookup`, filtered to `vintage_month == t`; change (Δ)
  features may also read t−6 and t−12.
- **Never read in feature code:**
  - the ledgers, `facility_presence`, `facility_universe` beyond t;
  - `citation_lookup_versions` (its `valid_to` looks into the future), `qm_inventory`;
  - `provider_identity_by_vintage`, `gold/dim_facility_latest`;
  - any label column.
- **Windows** look back from `cutoff_date`: "last 12m" = `cutoff_date − 12 months < event_date <= cutoff_date`.
  Silver keeps 217 rows dated up to 9 days after the cutoff, and this filter removes them.
- **Model use:**
  - star ratings (`overall_rating`, `health_inspection_rating`, `qm_rating`, `staffing_rating` and their `_fn`
    columns): the spec says `ablation_only`, the project rules say never features; the registry must settle it
    (open item 6);
  - `chain_id` is `product_only`;
  - ML training must never join `dim_facility_latest`.
- **Penalties:** for features every row counts, including `dup_seq > 1`; only labels use `revision_group`.
- **Δ is null:**
  - when t−6 / t−12 was never published or the facility is absent;
  - for QMs, also when `measure_code` differs or `measure_period_end` is unchanged.
- **Reader:** the spec asks for a feature reader that records every read and refuses anything after t; it is
  also the time-travel test of §11.3.
- **Tests:** add a leakage test in `tests/leakage/` for each family. `tests/leakage/test_gold_inputs.py`
  already guards the spine and labels: it reads `configs/keep_columns.yaml`, fails if gold's code names a
  star-rating column, and runs gold on the toy world through a store that records every read. Extend the same
  idea to the feature reader and the registry (open item 5).

## 10. Join rules (spec §7, §10; SILVER_CONTRACT §13.1)

- **Base:** every table is `spine LEFT JOIN <table> USING (ccn, t)`. After **every** join, the row count equals
  the spine and `(ccn, t)` is still unique.
- **Order:** spine → survey_timing → citations → enforcement → facility_staffing → quality → context → labels.
- **Fill when nothing is found:**
  - counts and sums 0;
  - booleans `False`;
  - dates, ratios and Δ null;
  - `_fn` columns `'none'`.
- **Zero means none found:** "none found in the published data", not "the facility is good".
- **Aggregate first:** citations, surveys and penalties are aggregated to one row per `(ccn, t)` before joining.
  QMs are pivoted measure → column after mapping bridged codes to one name.

## 11. Commit style (the team's PR rules)

- **One small step per commit:** 7–15 per PR, each one passing its tests.
- **Subject line only**, short and plain, e.g. "Add the enforcement features with every fine row counted".
  No body except `Closes #<issue>` in the last commit, and no co-author or tool trailers.
- **Push** the branch as a backup after each tested piece; open the PR into `dev` only when the team agrees.
- **Checks before pushing:** `uv run ruff check .`, `uv run mypy src`, `uv run pytest`,
  `uv run pytest tests/leakage`. On Windows, some `tests/ingestion` and `tests/harmonize` tests already fail on
  `dev`; compare by test name.

## 12. Open questions and open items

| # | Question or item | Who | Notes |
|---|---|---|---|
| 1 | Where do features live: `gold/features` (what the xfail tests import) or the empty `features/` package (`SILVER_CONTRACT.md` §7)? | Partner | Only the `_features` test helper needs to change |
| 2 | Buffers `B_cit = 8`, `B_pen = 12` | Team | In `configs/labels.yaml`; still a Checkpoint 3 proposal awaiting approval |
| 3 | `configs/qm_measures.yaml`, especially 405 → 480 | Silver owner | Checkpoint 4 decision |
| 4 | Training rows from 2022-01 instead of 2023-01 | Sai | Spec §14 #10: with the buffers, `y_g12` has very few training months |
| 5 | Leakage contract test for features: the feature reader and `feature_registry.yaml` limited to the `*_by_vintage` tables and `citation_lookup` at t (plus t−6, t−12), never the ledgers, `facility_presence`, `facility_universe` beyond t, `citation_lookup_versions` or `dim_facility_latest` | Partner | Review follow-up 1, feature part; the spine/labels part is done (section 9) |
| 6 | Star ratings: `ablation_only` (spec) or never a feature (project rules)? Tag them with `model_use` in the registry | Partner | Review follow-up 2 |
| 7 | Silver edge cases: a provider partition without `ownership_type` exits 1 (retried) instead of 2; `citation_lookup` has no Processing Date, so its vintage comes from the inventory only | Silver owner | Review follow-up 5, not urgent |
| 8 | Rebuild the real silver with the configs on `dev`, and share it with its `metadata.db` | Silver owner | Needed before gold runs on real data again (section 5) |
| 9 | Clean-up of `gold/_staging/run=<id>/` | Sai | Kept, because the object store only deletes under `quarantine/` and `tmp/` |
| 10 | `--v` backtests | Sai | Final severity and `removed_by_cms` still come from the latest silver |
| 11 | Is `y_pen12` a model or only a comparison? | Team | Spec §14 |
