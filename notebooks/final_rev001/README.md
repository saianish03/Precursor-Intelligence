# final_rev001: Precursor Intelligence final modelling package

This folder holds the final modelling record for the `y_g12` model, which predicts a serious (G+) CMS health citation within 12 months. It is self-contained: start with `report.md`.

| Path | Contents |
|---|---|
| `report.md` | **Main document.** Problem from first principles, data and splits, timeline of every experiment and score, feature engineering, ceiling diagnosis, external evidence, the stop decision, how to use the model, glossary and file index |
| `final_model/` | Re-run of the recommended model with its complete test metrics: `final_model_metrics.py`, `final_model_metrics.json` (all metrics + bootstrap CIs), `reliability.csv`, `test_predictions.csv` (2025-02 scores) |
| `state_bias/` | State-wise bias audit (`STATE_BIAS_REPORT.md`), produced separately |
| `research/` | Deep research (error analysis, oracle, data-source verification, literature) and the shortlist of ceiling-breaking ideas |
| `results/` | Results tables, all-tables report, model card, missing-months report, rev002 comparison |
| `results/summaries/` | Small JSON/CSV summaries of exp3–exp12 (metrics, CIs, gain rankings) |
| `results/row_filters/` | Row-filter experiment outputs (R0–R11) |
| `reports/` | Historical copies of the stage reports: subset EDA/FE (rev001), Gold 2022 EDA/FE (rev002), row filters, PBJ (exp12) |
| `experiments/` | Scripts behind the numbers: `rev002/` (shared split/purge logic, exp6–exp11, 133-feature registry, feature transformer), `rev003_pbj/` (PBJ download, features, exp12), `row_filters/` |
| `feature_engineering/` | Gold feature-family code (`gold_features/`), the 202-feature registry, the 75- and 66-feature selected sets, selection and boosting scripts/logs, tuned LightGBM parameters, Gold 2022 EDA CSVs |

Notes:

- Every supporting file is a copy. The originals stay in their source folders.
- Copied markdown reports and logs are historical: their file references point to their original locations, and two logs contain machine-specific absolute paths.
- No data, parquet files, model binaries, prediction files or environment configuration are included.
- Scripts expect the repository layout and a local gold build (`data/gold/...`). They are included for traceability, not to be run from this folder.
