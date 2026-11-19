# ML Phase 2 hand-off (production on the GCP VM)

For **Nishanth (nishanth3112)**, who owns ML Phase 2 from 2026-11-19. Phase 1 built the ML pipeline and proved it on a
laptop stack against GCS. Phase 2 puts it in production on the VM: MLflow there, the champion on real VM gold,
BigQuery, then monitoring, fairness and tuning (`ML_PIPELINE_SPEC.md` §18.1). **The VM itself** (its smoke test, the
bucket clean-up, the full backfill, production start and the VM runbook) **stays with the lead's team**; your ML work
on the VM builds on it (§5.1). The rules stay in the spec; this file gives the starting state, the order of work, the
decisions already made and the traps. State as of `dev` `055f8fe` (2026-11-18).

---

## 1. Where things stand

| Area | State |
|---|---|
| **ML code** | W0–W6 and the follow-ups #77–#80 are merged. The §17.2 acceptance passed on the laptop stack on 2026-11-18 (`ML_PIPELINE.md` §11, §14). |
| **Phase 1 champion** | v1 (`lgbm`, test 2025-02 PR-AUC 0.470) on the lead's laptop stack `ml_dev`: registry `precursor_y_g12` with `@champion` v1 and `@challenger` v3; scores for 2026-10 in `gs://precursor-intel-mlops-data/ml-dev/scores/`. It does not move to the VM (§3). |
| **VM** | `pi-airflow` (e2-standard-4, us-central1-a) is built and **stopped**: the VM in Terraform, `pi-deploy`, the idle shutdown and the Deploy workflow are done. `pi-release` is `d467b7d`, from before W4 (no MLflow, no `cms_nh_ml`). The smoke test, bucket clean-up and full backfill are next, done by the lead's team (§5.1). |
| **MLflow on the VM** | Off. `docker-compose.vm.yaml` keeps `mlflow` and `postgres-setup` in the profile `phase2` and sets `PI_MLFLOW_TRACKING_URI: ""`; `cd.yml` and `deploy.yml` ship two images; `pi-pg-backup`/`pi-pg-restore` skip `mlflow`. Ready: bucket `precursor-intel-mlops-mlflow`, secret `mlflow-db-password` (value set), `pi-deploy` renders it, and its `pg_hba` check knows the `mlflow` line. |
| **BigQuery** | Nothing yet: no Terraform, no code (`serving/publish.py` verifies the GCS scores and emits the asset). |
| **Gold** | Phase 1 gold is under `ml-dev/` (V = 2026-10, built by the laptop stack). The VM builds its own in the full backfill (T25). |
| **Open issues** | **#73 (yours):** B1 and B2 are Phase 1 bugs, so fix them first; B13–B16 and B19 follow. **#75:** only the B17 design (registered packages live only in run folders). **#83:** a docs PR. |

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
4. **Cost:** the VM costs about $0.13 an hour. It stops itself about 70–85 minutes after the last run and SSH session;
   stop it yourself with `sudo pi-down`.

---

## 3. Decisions already made (change them only with the lead)

- **The VM's registry starts fresh.** Phase 1's champion, run folders and `ml_runs` rows stay with `ml-dev/`. The first
  VM `retrain` registers v1 there as `@challenger`. The lead repeats **Checkpoint ML-A** on 2025-03 or later (the
  late-November CMS release brings it in) before `mode=promote` (spec §17.3).
- **The model recipe stays:** PR #49's parameters (the tuning gain was within noise; parameters change only through a
  reviewed PR, ML-19), reweighting off (O4), `staffing_rating_fn` and `d12_turnover_total` excluded (ML-21, O9).
- **Data side:** the release check stays paused during development; the first full run is manual, node by node; the
  data bucket root is cleared before it, but `ml-dev/` is kept (data spec §14 #24; ML spec §16.4).
- **Deploys:** a release is one git sha; the Deploy workflow sets `pi-release`; a rollback deploys an older sha.
  `terraform apply` runs only in CI, and its guard refuses deletes.
- **Process:** one PR per task, into `dev`. Reviews are posted as comments, and the lead approves. Bugs go to the
  owner of the work package (B17 stays with RiyaNTaori in #75).

---

## 4. Decisions still open (the lead decides; bring the data)

| Question | Today / recommendation | When |
|---|---|---|
| Gate thresholds (O2): G4 ECE ≤ 0.05, G6 Spearman ≥ 0.9 (v1 had 0.92), the 200+ beds slice | review on 2025-03 at ML-A | §5.2 item 5 |
| MLflow login (O5) | none (task network, 127.0.0.1, IAP); keep it unless something outside the VM needs MLflow | before the downstream work |
| B17: a stable copy of registered packages (#80's design note) | until then: never delete a registered version's run folder | before the run folders could be cleaned |
| The Dec 1 demo's scope | the spec's MVP (ML-10): end to end on the VM, monitoring may be thin. How much of the API, agents and dashboard? | now |

---

## 5. The work, in order

**The Dec 1 path:** the lead's team's T23–T25 (§5.1) and your items 1–4 (§5.2). The code for items 2 and 4 can start
at once, in parallel with T23.

### 5.1 The VM tasks the lead's team does (your items wait for them)

| VM task | What | Your items that wait |
|---|---|---|
| **T23 smoke test** | a trial-prefix override for `pi-deploy`; the first real Deploy run (the newest `dev` commit with images); a trial `cms_nh_pipeline` run, a forced-failure alert, a backup and restore; then `precursor_meta` is reset | item 2's deploy |
| **T24** | clear the data bucket root, all versions, **keeping `ml-dev/`** | — |
| **T25 full backfill** | `mode=backfill`, node by node, then the full run; `cms_nh_ml` stays paused. If the late-November release is out, it is included and brings 2025-03 into the test window | item 3 |
| **T26 production start** | the team's go: unpause `cms_nh_pipeline`, then `cms_nh_release_check` | — (later gold events start `cms_nh_ml` while it is unpaused) |
| **T27 VM runbook** | `docs/runbooks/vm.md` (access, deploy, rollback, up and down, backup and restore, costs) | add the MLflow parts (item 2) |
| **`VINTAGE_RANGE`** | ends at 2026-12 (`gold/build.py:73`): gold fails on the first 2027 vintage; fixed before late January 2027 | — |

### 5.2 Yours

| # | Task | Needs | Done when |
|---|---|---|---|
| 1 | **#73 B1 + B2** (one PR; B13 and B19 can ride along). B1: `monitor` must run when `score` produces nothing (spec §15.1). B2: PSI bins numeric features with few distinct values by value, not quantile (issue text) | — | merged with tests; on the VM's first ML run, `monitor` runs without scores and `state_gplus_rate_12m` raises no drift |
| 2 | **MLflow on the VM**, one infra PR:<br>- `docker-compose.vm.yaml`: drop the `phase2` profiles and the empty tracking URI; pin the mlflow image to `${PI_RELEASE_SHA}`; a VM-only `PI_MLFLOW_ARTIFACTS` prefix in the MLflow bucket; credentials from the metadata server, not the `/dev/null` mount;<br>- `cd.yml`: build, test and push `mlflow`, and count it in "already published";<br>- `pi-deploy`: pull and label-check three images; `check_running` checks the mlflow image;<br>- `deploy.yml`: `IMAGES: pipeline airflow mlflow`, and the rollback floor moves to this PR's sha (a rollback below it drops mlflow through `--remove-orphans`);<br>- `pi-pg-backup`/`pi-pg-restore`: add `mlflow` (order airflow → mlflow → precursor_meta);<br>- Artifact Registry: the DELETE rule from 30 to 7 days (KEEP 5 stays);<br>- B16: with tracking unset in prod, refuse instead of registering nothing;<br>- **by hand on the VM's data disk, before this deploy:** the `mlflow` line in `pg_hba.conf` (below);<br>- the MLflow parts of `docs/runbooks/vm.md` | T23 for the deploy | MLflow healthy on the VM; a task logs a run whose artifact lands in the VM prefix |
| 3 | **ML on the VM.** Unpause `cms_nh_ml` (a paused DAG's runs never start), `mode=retrain` → v1 `@challenger`; the lead runs Checkpoint ML-A (model card, `scripts/ml_checkpoint_reasons.py`) and `mode=promote` with a reason; then `mode=auto` (SKIP) and `mode=score_only` (identical scores) | 1, 2, T25 | champion on the VM, scores and `status/latest.json` written, the summary email received |
| 4 | **BigQuery publish**: Terraform dataset `precursorintelligence_scores` and 3 tables (schemas as JSON files, tested against the parquet writer), `sa-pipeline` roles (spec §16.3); `ml publish` loads each month with `WRITE_TRUNCATE` on its partition and records the publish after the load (spec §11). Every query sets `maximum_bytes_billed`. The code can be built on the toy chain first | 3 for real data | scores queryable in BigQuery |
| 5 | **Threshold review** with the lead on 2025-03 | 3 | `ml.yaml` gates confirmed, or changed by PR |
| 6 | **Monitoring**: the performance trend, realized performance on newly matured cohorts, score drift, an Evidently report (spec §12) | 3 | `monitor` reports them, with tests |
| 7 | **Fairness**: `state` in G7(a) with Benjamini–Hochberg (ML-27); per-state empirical-Bayes shrinkage | 3 | gate and model card updated |
| 8 | **Tuned challenger and automatic re-tune**: `train` applies a tuning proposal as a challenger; `gate` compares it with no tolerance (spec §12.2, §13) | 3 | a `retune` run yields a gated challenger |
| 9 | **Verified gold**: the data spec's §11 gate (W5 part 2), then `allow_unverified: false` | — | prod refuses unverified gold |
| 10 | **Hardening**: the rest of #73 (B14 CI fails on a stale `dvc.lock`, B15 CI exercises `gs://` artifacts); B17 once decided (with #75) | — | one PR per area |
| 11 | **Downstream**: the FastAPI service, the four agents and the dashboard read `scores/`, `status/latest.json` and BigQuery | 4 | as the demo scope says (§4) |
| 12 | **Model research**: `y_g6` (O1); rolling-origin folds (O3); regime-invariant enforcement features (O8); bring back excluded features (`d12_turnover_total` once the eval window starts at 2023-01 or later, about `V_L` 2027-05; the four `qm_ls_*` after about 6 months of the new definitions; `staffing_rating_fn` when its PSI is under 0.25); a hazard model (spec §21) | — | each through a reviewed PR and the gates |

**The `pg_hba` hand step (item 2).** The VM's data disk was initialized before MLflow existed, and `init-db.sh` never
reruns. Insert the line **directly above the reject line**, never with `1i`: `pi-deploy` accepts only the order
pi_meta → mlflow → reject. It is the command in `runbooks/airflow-local.md` §2.1; on the VM (compose project
`pi-airflow`, as `pi-pg-backup` finds it):
```bash
pg=$(sudo docker ps -q --filter label=com.docker.compose.project=pi-airflow --filter label=com.docker.compose.service=postgres)
sudo docker exec "$pg" sh -c 'sed -i "/^host all all 172\.16\.77\.0\/24 reject\$/i host mlflow mlflow 172.16.77.0/24 scram-sha-256" "$PGDATA/pg_hba.conf"'
sudo docker exec "$pg" psql -U airflow -d airflow -c 'SELECT pg_reload_conf()'
sudo docker exec "$pg" psql -U airflow -d airflow -c 'SELECT line_number, database, user_name, address FROM pg_hba_file_rules LIMIT 3'
```

---

## 6. Traps we hit

- **Tests and CI use a SQLite MLflow store**, which accepts things a real server refuses: registration without a
  `model_id` passed every test and failed the first DAG run (#72). Check registry or tracking changes against a
  throwaway `mlflow server`.
- **A paused DAG gets no runs and no asset events.** Unpause `cms_nh_ml` before the pipeline run whose gold should start
  it; `pi-busy` counts it only while it is unpaused.
- **The first version on a fresh registry is always held** as `@challenger`: nothing is scored until `mode=promote`.
- **A code or config change retrains:** the next `auto` run sees a new `recipe_hash` and decides RETRAIN (spec §15.2).
- **`dvc.lock`:** when two open PRs both touch `src/` or configs, refresh it once after both merge
  (`uv run dvc repro`; CI forces `toy_ml`).
- **The `dev` ruleset:** a push after an approval dismisses it; every thread must be resolved; issues don't close
  themselves for PRs into `dev` (the default branch is `main`).
- **Memory:** the VM has 16 GB; tasks are capped at 10 GB, MLflow at 1 GB. The `heavy` pool (1 slot) serializes
  `build_gold`, `train`, `tune` and `evaluate`. On the laptop the peak was 3.1 GB per ML step.
- **On the VM:** never run `docker compose config` without `-q` (it prints the secrets into the journal); an MLflow
  version bump runs `mlflow db upgrade`, which can't be undone, so dump first; every `gcloud compute ssh` keeps a
  connection open about 10 minutes after you leave (the idle shutdown counts it).
- **Keep `ml-dev/`**, above all `ml-dev/ml/runs/phase1-retrain-2/` (v1's packages), until the VM has its own champion.
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
| Your bugs | #73; B17 in #75 |
