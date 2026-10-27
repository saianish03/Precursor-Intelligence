# Alerts handoff (ticket 2: failure and anomaly alerts)

- **For:** whoever builds the pipeline alerts (`feat/pipeline-alerts`, ticket 2).
- **Depends on:** the Airflow PR from `feat/dp_airflow_v1` (ticket 1). Branch from `dev` after it merges.
- **State on 2026-10-27:**
  - **Nothing sends an alert yet.** The DAG has no `on_failure_callback`, no notifier, and no Slack or email code. (Since built on `feat/pipeline-alerts`: see §6.)
  - **What exists:** every signal an alert needs is already produced and stored (§1), and failed runs really show as failed.
- **Read first:**
  - `AIRFLOW_PIPELINE.md`: the DAG, the lock, exit codes, the metadata DB schema (§5), and the open follow-ups (§9);
  - `runbooks/airflow-local.md`: run the stack;
  - `runbooks/airflow-e2e-simulation.md`: reproduce our test runs, including the forced failures.

## 1. What already exists (the signals to alert on)

| Signal | Where | Fails the run today? | Notes |
|---|---|---|---|
| **DAG run state** | Airflow | — | Honest: `release_lock` is a teardown, so any failed task makes the run `failed` (verified: runs A and G in runbook §8.2). `tests/dags` asserts `gold.build_gold` is the only leaf, so a new node cannot silently break this |
| **Task exit codes** | `dags/utils/pipeline_operator.py` | yes | **1** = error, retried 2 × 5 min; **2** = data or validation failure and **3** = config or usage error, both failed at once with `AirflowFailException`. The task log contains `pipeline exit <n> (<kind>), not retried` |
| **`run_log`** (`precursor_meta`) | one row per step per run | — | `status` `ok` or `failed`; `detail` JSON with `n_failures` (non-zero = failed) and `error`; keyed by `run_id` = the Airflow run id |
| **`anomalies`** (`precursor_meta`) | Step 1 resolve | **only `no_tables_recognised`** | `severity` is `info`, `warning` or `error`. **Error kinds:** `no_tables_recognised` (fails resolve: exit 2), plus `no_vintage` and `download_<status>` (a quarantined, rejected or failed zip), which are `error` but **do not fail resolve**. (A quarantined or rejected zip does fail the run earlier, at `discover_download` with exit 2, and a failed download with exit 1; so in practice `no_vintage` is the error anomaly a green run can carry.) Warning kinds: `partial_zip`, `size_outlier`, `republished` |
| **`vintage_flags`** (`precursor_meta`) | bronze | no (by design) | `stale_refresh` (CMS re-posted old event data; 2025-09) and `revised_after_publication` (CMS corrected an old release; 2026-07 Provider Info). Worth a notice, not a failure |
| **`bronze_files` / `silver_partitions`** | bronze / silver | yes, if any is `failed` | A failed file or partition is written nowhere, recorded as `failed` and retried next run; the task exits 2 |
| **Run manifests** | `metadata/<layer>/run_date=…/<layer>_run_<run_id>_*.json` | — | `exit_code`, `failures`, counts. Gold's also holds `v`, `v_labels`, `split_windows` and `promoted` |
| **`gold/_current.json`** | gold | — | Which run gold holds: `run_id`, `v`, `verified` (always `false` until the spec §11 gate exists) |
| **Asset events** | Airflow (`asset_event` table) | — | `cms_nh_landing`, `cms_nh_bronze`, `cms_nh_silver`, `cms_nh_gold`, emitted only when that task succeeds. `cms_nh_gold` = published |
| **Logs** | Airflow task logs + `reports/<node>/run=<id>/logs/` | — | Per-process JSON logs shipped to the bucket, so they outlive the containers |

## 2. What ticket 2 should alert on (from the ticket and from what we found)
- **Every failed DAG run, once.** Use a DAG-level `on_failure_callback`: Airflow calls it after the last retry, so retries do not each alert. The message should include:
  - DAG id and run id;
  - the failed task ids;
  - the exit code (parse `pipeline exit <n>` from the task log, or read `run_log.detail`);
  - a link to the log.
- **A failed `release_lock`, which the run state does NOT show.** A teardown never decides the run's state, so a run whose lock release failed still shows **success**. The next run's `acquire_lock` then fails (exit 1), or takes the lock over only after 12 h. This needs a **task-level** callback on `release_lock`.
- **Error anomalies on a run that still succeeded.** `no_vintage` and `download_*` (see §1). Query `anomalies` for `run_id = R and severity = 'error'`.
  - **Keep the DAG thin:** do it in a pipeline CLI command (for example `precursor-ingest anomalies --run-id R --severity error`, run as a teardown or after `resolve_report`), not in DAG code.
- **Out-of-memory kills.** Exit 137 is treated like exit 1 today, so it is retried twice (≈ 15 more minutes) and rarely helps. Alert on 137, and consider making it no-retry in `PipelineDockerOperator` (`NO_RETRY_EXIT_CODES`).
- **A run that hangs.** `build_gold` has no `execution_timeout`. A hanging run holds the lock, and with `max_active_runs=1` the next daily runs queue.
  - Airflow 3 removed SLA-miss callbacks.
  - Use `execution_timeout`; the failure callback then fires. Or check Airflow 3.1.8's deadline alerts.
- **A partial CMS release stuck for N days** (from the ticket) needs `check_release` (spec §2.0), which is **not built**. Note it as blocked.
- **Not alerts, notices at most:** `stale_refresh` and `revised_after_publication` flags.

**Destination and secrets:**
- **Where it lives:** a Slack webhook (or email) URL in an Airflow Connection or env var set in `docker/airflow-local/.env`, never committed.
- **When unset:** a no-op, so tests and local runs stay quiet.
- **Slack provider:** `apache-airflow-providers-slack` is **not** in the Airflow image. Adding it means pinning it under the Airflow constraints in `docker/airflow.Dockerfile`.
- **Plain HTTPS:** a POST works too.
  - Callbacks run inside Airflow's own containers: with LocalExecutor, task callbacks run in the scheduler and DAG callbacks in the dag-processor. Both are on the `pi-airflow` network, which has internet egress.
  - Task containers are on `pi-airflow-tasks` (Postgres + internet only).

## 3. How to force each case locally (all reproduced in our runs; see `runbooks/airflow-e2e-simulation.md`)

| Case | How | Seen as |
|---|---|---|
| Config failure (exit 3) | Trigger with a bogus entry id: `{"entry_ids": "theme:2099-01-01:nursing-homes_2099-01-01.zip"}` (run G) | `discover_download` failed on try 1; every later task `upstream_failed`; run failed |
| Data or gate failure (exit 2) | A backfill whose latest zip is the 2025-09 one (run A): V is a `stale_refresh`, so gold's training gate has no g12 month | `gold.build_gold` failed on try 1; run failed; gold manifest `exit_code: 2` |
| `stop_after` skip (must NOT alert) | `{"stop_after": "silver"}` (run F) | Gold skipped; run success |
| Error anomaly | Unit level: `tests/ingestion/cms_fakes.py` builds zips (`make_zip(real_lookup=…)`, `lookup_csv_bytes()`). `tests/ingestion/test_pipeline_offline.py` already produces `no_tables_recognised`. Use a fake sender in tests | `anomalies` rows with `severity = 'error'` for the run id |
| Failed release | Hard to force through the DAG; test the callback in `tests/dags` with a patched operator (as `test_pipeline_operator.py` does) | `release_lock` failed, run success |

## 4. All problems found while wiring gold and testing the DAG (2026-10-26/27)
Details, sizes and fixes are in `AIRFLOW_PIPELINE.md` §9.

| # | Problem | Status | Owner | Ticket 2? |
|---|---|---|---|---|
| 1 | The image did not pass the commit to gold (`PI_GIT_COMMIT`), so every container-built gold manifest had `git_commit: null` | **Fixed** in the Airflow PR; the manifest now records the full SHA | — | no |
| 2 | **Gold's peak memory:** about 2.1 GB on 237k rows (measured); ≈ 5 GB on the full archive's ≈ 590k rows (scaled). Too tight for the 8 GB VM with the 6g task limit | Open: gold-side fix (§9.1) | gold | alert on exit 137 |
| 3 | Every daily run re-publishes gold, even unchanged; `gold/_staging/run=<id>/` and old object versions are never deleted (≈ 10 GB more per month on GCS) | Open: lifecycle rules + skip-unchanged publish (§9.2) | gold + Terraform | no |
| 4 | Gold manifest commit: the build arg defaulted to `"unknown"`; no dirty flag | **Default fixed:** the build fails without `PI_GIT_SHA`. The dirty flag is still open (§9.3) | image + gold | no |
| 5 | Out-of-memory kills (exit 137) are retried twice | Open (§9.4) | operator | **yes** |
| 6 | No `execution_timeout` on `build_gold`: a hung run holds the lock | Open (§9.4) | DAG | **yes** |
| 7 | A failed `release_lock` leaves the run green (teardowns don't set the run state) | By design; needs an alert | alerts | **yes** |
| 8 | `cms_nh_gold` means published, not verified (`verified: false` until spec §11) | Open: downstream training must check `_current.json` | ML phase | no |
| 9 | Error anomalies `no_vintage` / `download_*` don't fail the run | By design; needs an alert | alerts | **yes** |
| 10 | `check_release` not built: the DAG runs daily and rebuilds gold each time, and "partial release stuck" can't be detected | Open (spec §2.0) | pipeline | blocked |
| 11 | Not run on GCP yet: no VM in Terraform; the GCS path (`--env prod`, credentials from the VM's metadata server) is untested end to end | Open | infra | no |
| 12 | Two variables carry the commit (`PI_GIT_SHA` for ingestion, `PI_GIT_COMMIT` for gold) | Open, cosmetic | image | no |

## 5. Definition of done (from the ticket)
- **Forced cases:** a forced failure (run G) and a forced error anomaly each produce **one** alert.
- **Silence:** a `stop_after` run and a clean run produce none, and nothing is sent when the destination is unset.
- **Tests:** offline, with a fake sender, in `tests/dags` and the repo tests.
- **Docs and evidence:** how to set the webhook and what an alert looks like, plus one screenshot of a real alert.

## 6. Status on `feat/pipeline-alerts`
- **Built** (`dags/utils/alerts.py`; setup and examples in `runbooks/airflow-local.md` §3.1):
  - **Failed run:** DAG `on_failure_callback`, one email with the failed tasks, exit codes (XCom `pipeline_exit_code`, pushed by `PipelineDockerOperator`) and log links.
  - **Failed `release_lock`:** task-level `on_failure_callback`.
  - **New error anomalies:** `precursor-ingest anomalies --run-id R --severity error --new-only` (`ingestion/anomalies.py`) runs as `ingestion.check_anomalies` after `resolve_report`; `ingestion.alert_anomalies` posts it. "New" = an (entry_id, kind) the previous run with a successful resolve did not have, because resolve re-records an unfixed problem on every run.
  - **Exit 137:** no-retry, named in the alert. **Hang:** `build_gold` `execution_timeout` 2 h (`PI_GOLD_TIMEOUT_MINUTES`).
  - **Destination:** email over SMTP with the standard library's `smtplib` (no provider; TLS always), settings `PI_ALERT_EMAIL_TO` / `PI_ALERT_SMTP_*` in `.env`, e.g. Gmail with an app password. Empty `PI_ALERT_EMAIL_TO` = nothing sent. Slack was set aside for now; `send()` is the one place to add it.
- **Tests:** `tests/dags/test_alerts.py` (fake SMTP delivery), `tests/dags/test_dag_integrity.py` (wiring), `tests/dags/test_pipeline_operator.py` (137, XCom), `tests/ingestion/test_anomalies.py`.
- **Verified in the local stack** (`runbooks/airflow-local.md` §8.3): run G → 1 failure email; a forced `no_vintage` → 1 anomaly email, and 0 on the repeat; `stop_after` and clean runs → 0.
  - Screenshots of the two real emails: `runbooks/img/alert-email-run-failed.png`, `runbooks/img/alert-email-new-anomaly.png` (shown in `runbooks/airflow-local.md` §3.1).
- **Left to do:**
  - `check_release` / "partial release stuck for N days": blocked (§2).
  - A timeout is retried like exit 1, so a hang alerts after about 6 h (§9.4 of `AIRFLOW_PIPELINE.md`).
