# Data bias slicing (`precursor-pipeline bias-report`)

Before any model is trained, this checks whether the gold facility × month dataset represents or labels some
kinds of nursing homes differently, writes a reproducible report, and proposes one mitigation decision per
finding. It is the **data-level** half of bias work; model-level slice metrics come in the ML phase.

Slices are **facility attributes and state inspection practice**, not protected classes: CMS facility data has
no person-level demographics. The report describes differences between kinds of homes and inspection-intensity
effects. It never claims discrimination.

## Run it

```bash
uv sync --extra pipeline --extra bias        # fairlearn, statsmodels, scipy, matplotlib

uv run precursor-pipeline bias-report                          # gold behind gold/_current.json (dev: ./data, prod: GCS)
uv run precursor-pipeline bias-report --gold data/gold         # a local gold folder (or the table folder / one parquet)
uv run precursor-pipeline bias-report --scope latest           # the latest month only (current population)
uv run precursor-pipeline bias-report --scope train --labels y_g12,y_g6
uv run precursor-pipeline bias-report --out reports/bias --no-charts
```

| Option | Default | Meaning |
|---|---|---|
| `--config` | `configs/bias_slices.yaml` | column mapping, slices, thresholds, mitigation |
| `--gold` | gold/_current.json in the env's object store | local gold folder instead; with a `gold/_current.json` it is read through `read_current_gold` (sha256-checked, listed partitions only), otherwise the files read are hashed; `(ccn, t)` must be unique |
| `--labels` | the config's `labels` | comma-separated `y_<name>` columns |
| `--scope` | `all_labeled` | `all_labeled` (every row; label checks keep labeled rows), `train` (`split_<label> == train`), `latest` (latest t) |
| `--out` | the config's `output_dir` (`reports/bias`) | output folder |
| `--env` / `--env-file` | `dev` | which object store to read gold from |

Exit codes follow the pipeline: **0** done (findings are information, not failures), **2** bad input (a
required column is missing, the dataset or scope is empty, gold/_current.json is missing or out of step),
**3** bad config or usage.

On the real gold (579k rows) a run takes about 4–5 minutes on an 8 GB laptop.

## What it checks

Every column name and threshold comes from `configs/bias_slices.yaml`; the mapping to gold columns was checked
in [`gold_schema_findings.md`](gold_schema_findings.md).

| Check | Module | Method |
|---|---|---|
| Slices | `bias/slices.py` | state, CMS region, ownership, bed band, chain status, in-hospital, CCRC, SFF status, star band (slicing only, never a feature), plus 4 two-way crossings; nulls → `missing`, and so is every month before the source column's first vintage (`chain_id` from 2023-06, feature registry); 0 beds and NaN ratings → `missing`; leakage guard rejects `y_`, `status_`, `split_`, `_fwd` columns |
| Representation | `bias/representation.py` | rows, homes, shares, first/last month; **small** below 30 homes or 500 rows |
| Label-rate disparity | `bias/disparity.py` | Fairlearn `MetricFrame` rates on labeled rows (immature/censored excluded, counted), complement ratio, four-fifths flag; **inference by home**: cluster-robust (by `ccn`) variances, Wilson CI sized on the effective n |
| SliceFinder | `bias/slicefinder.py` | Cohen's h, z-test with the cluster-robust variance of the difference (Fisher's exact on rows below 5 observed per cell), Benjamini–Hochberg per label; problematic = adj p < 0.05, \|h\| ≥ 0.2, not small in that label's labeled rows, not a risk tier (`sff_status`, `star_band`: own report section); boolean slices tested on `true` only |
| Rater bias | `bias/rater_bias.py` | per state, on the label's labeled rows: label rate vs inspection frequency and backlog (`n_std_36m`, `days_since_last_std`, `std_overdue`); ρ vs `state_gplus_rate_12m` reported as persistence only; states under 30 homes not ranked; flag above the 90th percentile on both |
| Missingness | `bias/missingness.py` | null-rate gap per slice and registry **model** feature (`model_use: model`; `*_fn` footnotes as has-reason rates, gold's `"none"` = no reason); flag at 0.10, small slices excepted |
| Skew | `bias/skew.py` | train vs test per slice: Jensen–Shannon distance (numeric, base 2) and L∞ (categorical); flagged only above the 95th percentile of 20 random same-size samples of the whole population (its own drift + sample noise); state and region slices skip state-level features (one value per state and month); time-thirds fallback without splits |
| Mitigation | `bias/mitigation.py` | sample weights (`inverse_frequency`, Kamiran–Calders `balance_label_rate`, `none`), fit on and written for `split_<weight_label> == train` rows only (null elsewhere, so no held-out label is read), clipped and renormalised to mean 1; effective sample size. `weight_slice` may not be `star_band` or `chain_status` |

## Outputs (`reports/bias/`)

Sorted rows and a fixed float format, so two runs on the same gold give byte-identical CSVs (only
`generated_at` in `summary.json` changes).

| File | Contents |
|---|---|
| `slice_summary.csv` | slice, value, n_rows, n_facilities, shares, small_slice, first_t, last_t |
| `slice_by_month.csv` | rows per single-slice value and month (stability) |
| `label_rates.csv` | per label and slice value: n, n_homes, n_eff, positives, rate, Wilson CI, complement, ratio, diff, flag, var_diff, immature/censored counts |
| `censoring_by_slice.csv` | per label and slice value: matured rows, censored share vs the label's overall share, gap, flag (≥ 2 points, not small) |
| `problematic_slices.csv` | top-k per label: Cohen's h, p, BH-adjusted p, test used, problematic, rank |
| `rater_bias_by_state.csv` | per label and state: label rate, exposure means/medians, state G+ rate, flag (ρ in `summary.json`) |
| `missingness_by_slice.csv` | slice, value, feature, kind, null_rate, overall_null_rate, gap, flag |
| `skew_by_slice.csv` | slice, value, feature, type, distance, metric, flag, n_ref, n_cmp |
| `mitigation_decisions.csv` | one proposed decision per finding (check, finding, decision, reason) |
| `sample_weights.parquet` | ccn, t, weight, scheme (keyed like gold, for the ML phase; weight null outside the training split; `summary.json` records the split, gold V and run) |
| `summary.json` | gold pointer and dataset sha256, config sha256, git commit, scope, counts, thresholds, flags per check |
| `charts/*.png` | label rate with CI per slice, a heatmap per crossing, slice sizes, state rate vs exposure |
| `bias_report.md` | the readable report: 10 sections, from summary to future work |

The mitigation decisions are **proposals**: the team confirms or changes them before the ML phase uses them.

## Assignment mapping

| Assignment | Where |
|---|---|
| PDF section 3 "Bias detection and mitigation" / grading criterion 9 | this module (`src/precursorintelligence/bias/`), `configs/bias_slices.yaml`, `reports/bias/bias_report.md` |
| Data slicing (Lecture 4, SliceFinder) | `slices.py`, `slicefinder.py` |
| Group fairness at the data level (Lecture 4) | `disparity.py` (Fairlearn `MetricFrame`) |
| Biased representation (Lecture 4) | `representation.py` |
| Rater bias (Lecture 4) | `rater_bias.py` |
| Skew / TFDV distances (Lecture 5) | `skew.py` |
| Mitigation | `mitigation.py`, report section 8 |
| Tests | `tests/bias/` (offline: synthetic gold with planted rates and the toy world) |

## Differences from the ticket spec

- `n_complaint_surveys_12m` is `n_complaint_surveys_cited_12m` in gold (mapped in the config).
- Three keys were added to the config so nothing is hard-coded: `skew.split_label` (whose split defines
  train/test), `rater_bias.quantile` (0.9), and `mitigation.weight_label` (the label `balance_label_rate` uses).
- An invalid config exits **3**, not 2, following the pipeline's exit codes (2 = data, 3 = config); missing
  columns and empty data exit 2 as specified.
- `--labels` is comma-separated (`y_g12,y_g6`), like the other pipeline CLIs.
- `--scope train` and the label checks use each label's own `split_<label>`; representation, missingness and
  weights use the first label's.
- With `balance_label_rate`, unlabeled rows get no weight (null): the reweighing is defined on labeled rows only.
- The optional Airflow `bias_report` task is not added in this PR.
