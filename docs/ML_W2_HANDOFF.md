# ML W2 hand-off (for W3: explain and score)

For whoever builds **W3** (`ML_PIPELINE_SPEC.md` §8, §10, §11: `serving/{explain,score,rollup,publish}.py`, `ml score`,
`ml publish`). It covers what W2 (evaluation, bias, gates) gives you, the files and tags you read, the two places
where your code and W2's meet, and how to get a champion to score locally. The rules come from the spec; this file
describes what the code in `src/precursorintelligence/evaluation/` actually does.

---

## 1. In one paragraph

After `ml train` (W1), **`ml evaluate`** scores the test frame with the eval-fit packages and writes `evaluation.json`.
**`ml gate`** applies G1–G7, registers the refit in MLflow (`precursor_y_g12`), moves the aliases, writes an
`ml_runs` row and **`champion.json`**. **`ml promote`** moves `champion` by hand (Checkpoint ML-A, or a rollback).
`ml score` (yours) starts from `champion.json`: it names the exact version to load. The first real run is always
held as `challenger`, so **nothing is scored until someone runs `ml promote`** (§9.1, §17.3).

| Who | Done | Next |
|---|---|---|
| W2 | `ml evaluate`, `ml gate`, `ml promote`; metrics, bootstrap, baselines, slices, the reweighting trade-off, gates, the model card | Phase 2: per-state EB, Benjamini–Hochberg in G7(a) |
| W3 (you) | — | `serving/explain.py`, reason codes, `ml score`, the roll-up, the scores layout, `ml publish` |

---

## 2. What you read

### `ml/runs/<ml_run_id>/champion.json` (`training/contracts.py: Champion`)
```json
{"schema_version": 1, "model_name": "precursor_y_g12", "version": 3, "source": "gate"}
```
- `version: null` means **no champion yet: skip scoring** (exit 0, the status says "awaiting Checkpoint ML-A").
- `source` is `gate`, `promote` (and later `decide`, W6). Load **this version**, never whatever `@champion` says at
  load time (§10.1), so a retried task scores with the same model.

### Loading the version
```python
from precursorintelligence.evaluation.registry import default_registry

registry = default_registry(cfg)               # None when MLFLOW_TRACKING_URI is unset
model = registry.load_model(f"models:/{champion.model_name}/{champion.version}")   # a PrecursorModel
scores = model.predict_frame(score_frame)      # p, logit, logit_raw, logit_<member>
```
`load_model` unwraps the pyfunc and checks that it is a `PrecursorModel` (W1's package). The registered version's
source is the **refit** package (trained on the refit window), which is the model that ships.

### Version tags (`evaluation/gate.py: version_tags`)
| Tag | Use in W3 |
|---|---|
| `ml_run_id`, `recipe_hash`, `candidate` | `facility_scores` columns `ml_run_id`/`candidate`; the scores manifest |
| `gold_run_id`, `dataset_version`, `training_dataset`, `v_labels`, `gold_verified` | stamps on the scores |
| `eval_run_id`, `eval_model_uri` | not for scoring (G5/G7 use the eval fit) |
| `gate_outcome`, `failed_gates`, `sanity_passed` | what the gates said |
| `promoted_by`, `promote_reason` | set by `ml promote` |

### `ml_runs` (Postgres, `common/io/schemas.py`) and the integrity check (§9.4)
`score` must refuse (exit 2, alert) unless `@champion` equals `champion_version_after` of the **latest row whose
`action` is `promote`, `manual_promote` or `rollback`**. `gate` writes `promote` (all gates passed) or `hold`; a
`hold` row also carries `champion_version_after` (the unchanged champion), so filter on the action, not on the
column. `promote` writes `manual_promote` or `rollback`. Order by `id` (autoincrement), not by `at`.

W2's rows always have `scores_published = false`. `publish` (yours) records published scores; `decide` (W6) and
`prepare` (`training/data.py: last_scored_v`) read `scores_published` and `v` from `ml_runs`.

### `evaluation.json` and `report/` (optional for you)
- `threshold`: the selected candidate's OOF max-F1 cut-off, if you ever want a flag column (the spec does not ask
  for one).
- `report/shap_mean_abs.csv`: mean |contribution| per feature on the **test** frame (eval fit). §8.2 also wants it
  on the **score** frame for the champion, plus the champion's top 15 as its *key features* (a model-version tag
  for drift monitoring). W2 does not do the score-frame part: it belongs with scoring (you) or monitoring (W6).
  Agree on which before Phase 1 closes.

---

## 3. Where your code and W2 meet

### `serving/explain.py: contributions(model, df)`
`PrecursorModel.explain` (W1) calls `precursorintelligence.serving.explain.contributions` **as soon as the module
exists**, and falls back to the members' own contributions until then. W2 calls `model.explain` in two places:
- **G2 SHAP additivity** (`evaluation/sanity.py: shap_additivity`): on 1,000 test rows,
  `base + sum(contributions)` must equal `predict_frame(...)["logit"]` within **1e-6**. If your `contributions`
  breaks additivity, **G2 fails and no model is promoted**.
- **Top features** (`evaluation/evaluate.py: top_features`): mean |contribution| per column, so the column names
  must be the model's feature names (`model.feature_names`), in calibrated log-odds.

The contract: `contributions(model, df) -> (base: np.ndarray of len(df), contrib: pd.DataFrame)` with one column
per model feature, in calibrated log-odds (Platt `a` applied, `b` in the base; §8.1). Run
`uv run pytest tests/evaluation/test_sanity.py` after you add it: it checks additivity for every candidate.

### `ml_runs`
Write your rows with `meta.append("ml_runs", [...])` (append-only). Reuse
`training/data.py: gold_config_sha(manifest)` for `gold_config_sha`, so the writers and `decide` agree.

---

## 4. Getting a champion locally

The whole chain on the ml-dev gold runs in about 15 minutes on a laptop:

```bash
export MLFLOW_TRACKING_URI="sqlite:///$PWD/mlruns.db"   # a throwaway local store
RID=manual__mytest                                     # no ':' in run ids on Windows
for step in prepare train evaluate gate; do
  uv run precursor-pipeline ml $step --env-file <your env.yaml> --run-id $RID || break
done
PI_ML_VERSION=1 PI_ML_REASON="local test" uv run precursor-pipeline ml promote --env-file <your env.yaml> --run-id $RID
cat <store root>/ml/runs/$RID/champion.json           # version 1, source promote
```
The env file points the object store at a folder that holds `gold/` (as the bucket's `ml-dev/` prefix does).
`ml evaluate --no-mitigation` skips the reweighting refit if you only need a champion fast.

For tests, use the synthetic gold (`tests/ml/make_ml_gold.py`), `tests/evaluation/conftest.py: trained` (a run
folder trained once per module), and `tests/ml/ml_eval_fixtures.py: FakeRegistry` (an in-memory registry, so tests
never need a server).

---

## 5. Numbers from the first real run (ml-dev gold `20261112T020010Z-543cdc`)
LightGBM selected (OOF PR-AUC 0.511). Test month 2025-02, a development month: PR-AUC 0.470 [0.454, 0.486] vs the
prior-G+ rule's 0.311, ECE 0.027, slope 0.835. All gates pass (G5 skipped, no champion). The refit's mean score on
V = 2026-10 is 0.248. Expect `p_g12` around 0.25 on average, and `state` as the largest contribution for many homes
(the spec's known limitation on within-state ranking).
