# ML Phase 2 hand-off (production on the GCP VM)

For **Nishanth (nishanth3112)**, who owns ML Phase 2 from 2026-11-19. Phase 1 built the ML pipeline and proved it on a
laptop stack against GCS. Phase 2 puts it in production on the VM: MLflow there, the champion on real VM gold,
BigQuery, then monitoring, fairness and tuning (`ML_PIPELINE_SPEC.md` §18.1). **The VM itself** (its smoke test, the
bucket clean-up, the full backfill, production start and the VM runbook) **stays with the lead's team**; your ML work
on the VM builds on it (§5.1). The rules stay in the spec; this file gives the starting state, the order of work, the
decisions already made and the traps. State as of `dev` `e04fb12` (2026-11-22).

---

## 1. Where things stand

| Area | State |
|---|---|
| **ML code** | W0–W6 and the follow-ups #77–#80, #85 and #86 are merged. The §17.2 acceptance passed on the laptop stack on 2026-11-18 (`ML_PIPELINE.md` §11, §14). |
| **Phase 1 champion** | v1 (`lgbm`, test 2025-02 PR-AUC 0.470) on the lead's laptop stack `ml_dev`: registry `precursor_y_g12` with `@champion` v1 and `@challenger` v3; scores for 2026-10 in `gs://precursor-intel-mlops-data/ml-dev/scores/`; its packages (v1–v3) backfilled to `gs://precursor-intel-mlops-mlflow/packages/ml-dev/` on 2026-11-19. It does not move to the VM (§3). |
| **VM** | `pi-airflow` (e2-highmem-4, us-central1-a) is built and stops itself when idle: the VM in Terraform, `pi-deploy`, the idle shutdown and the Deploy workflow are done, and **the smoke test (T23) passed on 2026-11-20** (`runbooks/airflow-local.md` §8.4). It runs `8018533` on the bucket root, **with T25's full gold published there** (V 2026-10, 775,400 rows, the official dataset rebuilt), MLflow (item 2) and `cms_nh_ml` unpaused with **champion v1** (item 3, 2026-11-22; `runbooks/vm.md` §14). Its database was reset at T24 and owns the root, with the three-line `pg_hba` (`pi_meta`, `mlflow`, reject). T24 (bucket clean-up, reset) and T25 (full backfill) are done (§5.1); T26 (production start) waits for the team's go. |
| **MLflow on the VM** | On once item 2 is deployed: `docker-compose.vm.yaml` runs `mlflow` (image `precursor/mlflow:<sha>`, artifacts in `gs://precursor-intel-mlops-mlflow/artifacts/vm`, sa-pipeline through the metadata server); `cd.yml` and `deploy.yml` ship three images and the rollback floor is item 2; `pi-pg-backup`/`pi-pg-restore` include `mlflow`; prod ML tasks refuse without tracking (B16); CI writes artifacts to a fake GCS (B15). Operating it: `runbooks/vm.md` §11. |
| **BigQuery** | Built with item 4 (`serving/bigquery.py`, Terraform in `envs/prod/main.tf`, schemas in `configs/bigquery/`): `ml publish` loads each month into `precursorintelligence_scores` before it records the publish. Only the VM sets `PI_BIGQUERY_DATASET`, so laptop stacks load nothing. The first real load was 2026-10, on 2026-11-22 (item 3): 14,686 facility scores, 117,488 reasons and 612 chains, equal to the manifest. Querying it: `runbooks/vm.md` §12. |
| **Gold** | Phase 1 gold is under `ml-dev/` (V = 2026-10, built by the laptop stack). The VM builds its own in the full backfill (T25). |
| **Open issues** | **#73 (yours):** B1, B2, B13, B14 and B19 are fixed (#86), B15 and B16 with item 2, which closes it. #74–#76 are closed. |

---

## 2. Before you start

1. **Access.** T12 gave the team IAP and impersonation of `sa-pipeline`. Check that you can start, enter and stop the VM:
   ```bash
   gcloud compute instances start pi-airflow --zone us-central1-a
   gcloud compute ssh pi-airflow --zone us-central1-a --tunnel-through-iap
   sudo true && echo sudo-ok          # OS Login sudo
   sudo pi-down                       # when done: dumps the databases, then stops the VM
   ```
   If a step is refused, ask the lead for the role (`infra/README.md` "The VM").
2. **The VM is shared** with the lead's team's VM tasks: agree with the lead before you start it, deploy a release
   (the Deploy workflow moves `pi-release` for everyone) or run anything on it.
3. **Read:** the spec §15–§17, §18.1, §19 (decisions), §21 (limitations); `ML_PIPELINE.md` §13–§15 (how to run, the
   Phase 1 runs, known limitations); `infra/vm/README.md` (deploys, up and down, backups, restore);
   `runbooks/airflow-local.md` §2.1 (MLflow) and §5 (the VM); `AIRFLOW_PIPELINE.md`; `ALERTS_HANDOFF.md`.
4. **Cost:** the VM costs about $0.19 an hour (e2-highmem-4, with its NAT; every situation in `infra/README.md`, "Cost of the VM"). It stops itself about 70–85 minutes after the last run and SSH session;
   stop it yourself with `sudo pi-down`.

---

## 3. Decisions already made (change them only with the lead)

- **The VM's registry starts fresh.** Phase 1's champion, run folders and `ml_runs` rows stay with `ml-dev/`. The first
  VM `retrain` registers v1 there as `@challenger`. The lead repeats **Checkpoint ML-A** on 2025-03 or later before `mode=promote` (spec §17.3). 2025-03 comes with the
  next CMS release, which the metastore gives as **2026-12-02**, after the demo: before it, ML-A can read only 2025-02 again.
- **The model recipe stays:** PR #49's parameters (the tuning gain was within noise; parameters change only through a
  reviewed PR, ML-19), reweighting off (O4), `staffing_rating_fn` and `d12_turnover_total` excluded (ML-21, O9).
- **Data side:** the release check stays paused during development; the first full run is manual, node by node; the
  data bucket root is cleared before it, but `ml-dev/` is kept (data spec §14 #24; ML spec §16.4).
- **Deploys:** a release is one git sha; the Deploy workflow sets `pi-release`; a rollback deploys an older sha.
  `terraform apply` runs only in CI, and its guard refuses deletes.
- **Process:** one PR per task, into `dev`. Reviews are posted as comments, and the lead approves. Bugs go to the
  owner of the work package.

---

## 4. Decisions still open (the lead decides; bring the data)

| Question | Today / recommendation | When |
|---|---|---|
| Gate thresholds (O2): G4 ECE ≤ 0.05, G6 Spearman ≥ 0.9 (v1 had 0.92), the 200+ beds slice | review on 2025-03 at ML-A | §5.2 item 5 |
| MLflow login (O5) | none (task network, 127.0.0.1, IAP); keep it unless something outside the VM needs MLflow. `AGENTS_PLAN.md` decision 3 recommends Cloud Trace for production traces and MLflow only for evaluation runs, which keeps it so | with the agents plan's decision 3 |

**Decided by the lead (2026-11-22):** the Dec 1 demo shows the whole product working end to end at production level:
the data pipeline; the whole ML / retraining pipeline, with its monitoring, drift detection and bias slicing (ML-10 is
amended: monitoring is no longer "may be thin"); the backend; the frontend; and Agents 1 and 2, maybe Agent 3 if there
is time (`AGENTS_PLAN.md` decision 1). Working end to end matters more than the model's scores.

---

## 5. The work, in order

**The Dec 1 path:** the lead's team's T23–T25 (§5.1) and your items 1–4 (§5.2; item 1 is done). The code for items 2 and 4 can start
at once, in parallel with T23. **`AGENTS_PLAN.md` (§4) counts on them:** items 2–3 by **Nov 26–28** (MLflow on the VM for
its evaluation traces; prod scores for its `cms_nh_agents` DAG) and item 4 by **Nov 28–30** (BigQuery, for its `llm_calls`
log sink). Its fallbacks: the laptop stack's 2026-10 scores, and GCS without BigQuery.

### 5.1 The VM tasks the lead's team does (your items wait for them)

| VM task | What | Your items that wait |
|---|---|---|
| ✅ **T23 smoke test** (2026-11-20, `runbooks/airflow-local.md` §8.4) | the trial-prefix override; the Deploy workflow's refusals and first real run (`5697619`); a trial backfill (3 min 25 s, gold 44,614 rows), a forced-failure alert, `pi-down`, a restore (110 s) and the shutdown backstop | — |
| ✅ **T24** (2026-11-20) | the data bucket root cleared, all versions, **keeping `ml-dev/`**; the VM's databases reset (a new, empty Postgres); the laptop's `dp_gcp_test` stack removed | — |
| ✅ **T25 full backfill** (2026-11-20/21, `runbooks/airflow-local.md` §8.5) | node by node; every layer matches the official run (`ml-dev/`); gold V 2026-10, 52 months, 775,400 rows, published at the root. The first gold attempt was OOM-killed at 10.46 GB, so the VM became an e2-highmem-4 with a 20g task limit (#90) | — |
| **T26 production start** | decided 2026-11-22: with the VM on `main`, once CMS has published (next: 2026-12-02), unpause `cms_nh_pipeline`, `cms_nh_ml`, then `cms_nh_release_check` (`runbooks/vm.md` §13) | 3 (a champion on the VM; later gold events start `cms_nh_ml` while it is unpaused) |
| ✅ **T27 VM runbook** (2026-11-21) | `docs/runbooks/vm.md`: access, start and stop, deploy and rollback, a manual run and a full backfill, the trial prefix, backups and restore, another database for task containers, alerts, costs, troubleshooting | add the MLflow parts (item 2), and item 2's 7-day registry rule in its section 3 |
| ✅ **`VINTAGE_RANGE`** (L7, 2026-11-21) | it ended at 2026-12, so gold would have failed on the first 2027 vintage. Now every silver vintage must lie between the ingestion scope's `vintage_start` and the run's month (`gold/build.py: resolve_v`) | — |

### 5.2 Yours

| # | Task | Needs | Done when |
|---|---|---|---|
| 1 | ✅ **#73 B1 + B2, done in #86** (with B13, B14, B19): `monitor` waits for `score` and `publish`, never the `scored?` short-circuit; group-level features get equal-width PSI bins. Seen live on the laptop stack (2026-11-19, `b2-check-score-only-1`, `mode=score_only` with v1 loaded from the package store): `drift:state_gplus_rate_12m` is gone and no re-tune is requested, the two real skew flags stay, and every score value is unchanged | — | B1 is seen on the VM's first ML run (no champion yet): `monitor` runs and writes `status/latest.json`. ✅ Seen on 2026-11-22 (`vm-retrain-1`) |
| 2 | ✅ **MLflow on the VM**, done in #109 (2026-11-22), one infra PR:<br>- `docker-compose.vm.yaml`: drop the `phase2` profiles and the empty tracking URI; pin the mlflow image to `${PI_RELEASE_SHA}`; a VM-only `PI_MLFLOW_ARTIFACTS` prefix in the MLflow bucket, never its root (#85's package store lives in `packages/`); credentials from the metadata server, not the `/dev/null` mount;<br>- `cd.yml`: build, test and push `mlflow`, and count it in "already published";<br>- `pi-deploy`: pull and label-check three images; `check_running` checks the mlflow image;<br>- `deploy.yml`: `IMAGES: pipeline airflow mlflow`, and the rollback floor moves to this PR's sha (a rollback below it drops mlflow through `--remove-orphans`);<br>- `pi-pg-backup`/`pi-pg-restore`: add `mlflow` (order airflow → mlflow → precursor_meta);<br>- Artifact Registry: `delete-after-30-days` → 7 days (KEEP 5 stays; `delete-untagged-after-7-days` is unchanged);<br>- #73's B16: with tracking unset in prod, refuse instead of registering nothing;<br>- #73's B15: CI exercises the `gs://` artifact destination (this PR is the first to use one in production);<br>- **no `pg_hba` hand step:** the VM's database already has the `mlflow` line (below);<br>- the MLflow parts of `docs/runbooks/vm.md` | — (T23 ✅) | MLflow healthy on the VM; a task logs a run whose artifact lands in the VM prefix |
| 3 | ✅ **ML on the VM**, done on 2026-11-22 (`runbooks/vm.md` §14). Unpause `cms_nh_ml` (a paused DAG's runs never start), `mode=retrain` → v1 `@challenger`; the lead runs Checkpoint ML-A (model card, `scripts/ml_checkpoint_reasons.py`) and `mode=promote` with a reason; then `mode=auto` (SKIP) and `mode=score_only` (identical scores). Done: `vm-retrain-1` (v1 held, the same metrics as Phase 1's v1), Checkpoint ML-A approved by the lead, `vm-promote-1` (v1 `@champion`, 14,686 homes scored), `vm-auto-1` (SKIP), `vm-score-only-1` (every value identical; only `ml_run_id` and `scored_at` differ); every run's email received | 2, T25 | champion on the VM, scores and `status/latest.json` written, the summary email received |
| 4 | ✅ **BigQuery publish**, done in #112 (2026-11-22; first real load with item 3): Terraform dataset `precursorintelligence_scores` and 3 tables (schemas as JSON files, tested against the parquet writer), `sa-pipeline` roles (spec §16.3); `ml publish` loads each month with `WRITE_TRUNCATE` on its partition and records the publish after the load (spec §11). Every query sets `maximum_bytes_billed`. The code can be built on the toy chain first | 3 for real data | scores queryable in BigQuery |
| 5 | **Threshold review** with the lead on 2025-03 | 3 | `ml.yaml` gates confirmed, or changed by PR |
| 6 | **Monitoring**: the performance trend, realized performance on newly matured cohorts, score drift, an Evidently report (spec §12). **#120:** score drift and the performance trend are built (with `skew_scope`, #73 B20); realized performance waits for scores to mature (2028-06 for 2026-10); the Evidently report is the lead's call | 3 | `monitor` reports them, with tests |
| 7 | **Fairness**: `state` in G7(a) with Benjamini–Hochberg (ML-27); per-state empirical-Bayes shrinkage | 3 | gate and model card updated |
| 8 | **Tuned challenger and automatic re-tune**: `train` applies a tuning proposal as a challenger; `gate` compares it with no tolerance (spec §12.2, §13) | 3 | a `retune` run yields a gated challenger |
| 9 | **Verified gold**: the data spec's §11 gate (W5 part 2), then `allow_unverified: false` | — | prod refuses unverified gold |
| 10 | **Hardening**: anchor the package hashes in `ml_runs` when MLflow gets a login (spec O5) | — | one PR per area |
| 11 | **Downstream, not yours except the frontend:** `AGENTS_PLAN.md` gives the API, the four agents and their GCP rollout to Tanoj, Riya and Aakash, and the frontend to you (its decision 0). Your part: review its infrastructure PRs (its part L), and keep stable what its `build-evidence` step reads (the API reads only that step's output): `scores/v=…/` with its `scores_manifest.json`, `ml_runs.gate_results_json`, the run folders' `evaluation.json`, `monitor`'s status and drift flags, and the `cms_nh_scores` asset its `cms_nh_agents` DAG runs on; plus BigQuery (item 4) for its `llm_calls` sink. A change to any of them breaks the agents' evidence build | 4 | the agents plan's definition of done (its §5) |
| 12 | **Model research**: `y_g6` (O1); rolling-origin folds (O3); regime-invariant enforcement features (O8); bring back excluded features (`d12_turnover_total` once the eval window starts at 2023-01 or later, about `V_L` 2027-05; the four `qm_ls_*` after about 6 months of the new definitions; `staffing_rating_fn` when its PSI is under 0.25); a hazard model (spec §21) | — | each through a reviewed PR and the gates |

**`pg_hba` (item 2): no hand step.** T23's restore (2026-11-20) started a fresh Postgres with `5697619`'s `init-db.sh`,
which writes the `mlflow` line, and T24's reset does the same. The role and database `mlflow` don't exist yet:
`postgres-setup` creates them once the `phase2` profile is gone. Before the deploy, check the order `pi-deploy` accepts
(`pi_meta`, `mlflow`, reject) on the VM:
```bash
pg=$(sudo docker ps -q --filter label=com.docker.compose.project=pi-airflow --filter label=com.docker.compose.service=postgres)
sudo docker exec "$pg" psql -U airflow -d airflow -c 'SELECT line_number, database, user_name, address, auth_method FROM pg_hba_file_rules LIMIT 3'
```
If a restore ever brings back an older data directory (`data.before-restore-*`), the line may be missing again: then add
it as in `runbooks/airflow-local.md` §2.1, directly above the reject line.

---

## 6. Traps we hit

- **Tests and CI use a SQLite MLflow store**, which accepts things a real server refuses: registration without a
  `model_id` passed every test and failed the first DAG run (#72). Check registry or tracking changes against a
  throwaway `mlflow server`.
- **A paused DAG gets no runs and no asset events.** Unpause `cms_nh_ml` before the pipeline run whose gold should start
  it; `pi-busy` counts its runs and queued asset events only while it is unpaused (a running task counts either way).
- **The first version on a fresh registry is always held** as `@challenger`: nothing is scored until `mode=promote`.
- **A code or config change retrains:** the next `auto` run sees a new `recipe_hash` and decides RETRAIN (spec §15.2).
- **`dvc.lock`:** CI fails on a stale lock (#86), also when `dev`'s own lock went stale because two merged PRs changed
  the same inputs: run `uv run dvc repro` and commit `dvc.lock` (CI forces `toy_ml`).
- **The `dev` ruleset:** a push after an approval dismisses it; every thread must be resolved; issues don't close
  themselves for PRs into `dev` (the default branch is `main`).
- **Memory:** the VM has 32 GB (e2-highmem-4 since T25, whose gold needed more than 10 GB); tasks are capped at 20 GB, MLflow at 1 GB. The `heavy` pool (1 slot) serializes
  `build_gold`, `train`, `tune` and `evaluate`. On the laptop the peak was 3.1 GB per ML step.
- **On the VM:** never run `docker compose config` without `-q` (it prints the secrets into the journal); an MLflow
  version bump runs `mlflow db upgrade`, which can't be undone, so dump first; every `gcloud compute ssh` keeps a
  connection open about 10 minutes after you leave (the idle shutdown counts it).
- **Keep `ml-dev/`** until the VM has its own champion (its gold, run folders and scores are Phase 1's evidence).
- **Every stack needs `PI_PACKAGES_URI`** (#85): without it the ML tasks exit 3. The VM's is fixed (`packages/vm`); the laptop stack uses `packages/ml-dev`. Never point two stacks at one prefix.
- **No secrets anywhere:** not in code, logs, docs or screenshots. The alert sender and recipients are kept in Secret
  Manager (`alert-smtp-user`, `alert-email-to`), so redact them from screenshots.

---

## 7. Where to look

| For | Read |
|---|---|
| Design rules, decisions, acceptance | `ML_PIPELINE_SPEC.md` (§15 DAG, §16 deployment, §17 acceptance, §19 decisions) |
| What is built and the Phase 1 evidence | `ML_PIPELINE.md` |
| The VM: deploys, up and down, backups, restore | `infra/README.md` "The VM", `infra/vm/README.md` |
| Running the stack, MLflow, alerts | `runbooks/airflow-local.md`, `AIRFLOW_PIPELINE.md`, `ALERTS_HANDOFF.md` |
| The data pipeline and the verified gate | `DATA_PIPELINE_SPEC.md` (§11, §15) |
| The agents, the API and their rollout | `AGENTS_PLAN.md` |
| Your bugs | none open (#73 closes with item 2) |
