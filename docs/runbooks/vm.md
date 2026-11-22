# Runbook: the GCP VM

How to operate the Airflow VM day to day: get in, start and stop it, deploy and roll back, run the pipeline,
restore its databases, and what to do when something goes wrong. The reference details stay where they are:
`infra/vm/README.md` (the scripts on the VM: deploys, up and down, backups, restore), `infra/README.md` (Terraform,
the VM's build, its cost) and `airflow-local.md` §5 (the stack's settings on the VM). The timings come from the
smoke test and the first full backfill on the VM (T23, T25: `airflow-local.md` §8.4, §8.5).

MLflow runs on the VM since ML Phase 2's item 2 (section 11), and `cms_nh_ml` since item 3 (section 14,
`docs/ML_PHASE2_HANDOFF.md` §5.2).

## 0. At a glance
| | |
|---|---|
| VM | `pi-airflow`, e2-highmem-4 (4 vCPU, 32 GB), zone `us-central1-a`, no external IP (SSH through IAP only) |
| Disks | 30 GB boot; 10 GB Postgres data disk at `/mnt/disks/pgdata`, which outlives the VM |
| What runs | the release (one full git sha, three images: pipeline, airflow, mlflow) that the instance metadata `pi-release` names; `pi-deploy` deploys it at boot and whenever it changes |
| Stack | compose project `pi-airflow` in `/opt/precursor/compose/`; env file `/etc/precursor/.env` (root, mode 600: it holds the secrets, never print it) |
| Data | `gs://precursor-intel-mlops-data` (the root); database dumps in `gs://precursor-intel-mlops-backups/postgres/<db>/` |
| Stops itself | about 70–85 minutes after the last SSH session, when no run is going on |
| Cost | ≈ $0.19 an hour running, ≈ $4 a month stopped (section 9) |

## 1. Access
- **Once per person:** an Owner grants IAP tunnelling and `sa-pipeline` impersonation (`gcp-bootstrap.md`, "People's
  access to the VM"). OS Login gives you `sudo` on the VM.
- **SSH:** `gcloud compute ssh pi-airflow --zone us-central1-a --tunnel-through-iap`.
- **Airflow UI:** no firewall rule opens 8080; tunnel it, then open http://localhost:8080 (user `admin`):
  ```bash
  gcloud compute ssh pi-airflow --zone us-central1-a --tunnel-through-iap -- -L 8080:127.0.0.1:8080
  sudo docker exec pi-airflow-airflow-apiserver-1 cat /opt/airflow/config/simple_auth_manager_passwords.json   # on the VM
  ```
  The alert emails' log links point at `http://localhost:8080`, so they work through this tunnel.
- **MLflow UI:** the same way on port 5001 (no login), then open http://localhost:5001:
  ```bash
  gcloud compute ssh pi-airflow --zone us-central1-a --tunnel-through-iap -- -L 5001:127.0.0.1:5001
  ```
- **Every SSH connection keeps the VM up,** a port forward too: close them when done.
- **The commands below** assume these two lines in your VM shell (absolute paths, as `pi-deploy` uses, so they work
  from any folder):
  ```bash
  dc="docker compose -f /opt/precursor/compose/docker-compose.yaml -f /opt/precursor/compose/docker-compose.vm.yaml --project-directory /opt/precursor/compose --env-file /etc/precursor/.env"
  af() { sudo $dc exec -T airflow-scheduler airflow "$@" </dev/null; }
  ```
  Never run `$dc config` without `-q` or `--services`: it prints the env file's secrets.

## 2. Start and stop
- **Start,** from a laptop: `gcloud compute instances start pi-airflow --zone us-central1-a`. The startup script is
  ready about 45 s later, and the stack about 3.5 min later when the release's images must be pulled (T23). SSH
  before the script is ready fails with IAP error 4003: wait, or watch the serial console:
  `gcloud compute instances get-serial-port-output pi-airflow --zone us-central1-a | grep pi-startup` ("ready").
- **Check:** `sudo pi-deploy status` (what runs, and the last results) and `sudo $dc ps`.
- **Stop:** `sudo pi-down` on the VM, or from a laptop:
  `gcloud compute ssh pi-airflow --zone us-central1-a --tunnel-through-iap -- sudo pi-down`. It dumps both databases
  first (seconds; T23: about 4 s) and stops only if that worked; during a run it refuses unless `--force`. A stop from
  the console or `gcloud compute instances stop` gets only the shutdown backstop's dump (T23: a full pair 33 s after
  the request).
- **The idle shutdown** stops it by itself, the same dump first, once the VM has been up 60 minutes, nobody has used
  SSH for 60 minutes, and no pipeline or ML run is running or queued (`infra/vm/README.md`, "Up and down").
- **To keep it up without a session** (e.g. a restore): `sudo pi-deploy pause` pauses deploys and the idle shutdown
  together; `sudo pi-deploy resume` ends both. A forgotten pause emails a reminder every 6 hours.

## 3. Deploy and roll back
- **Deploy,** from a laptop (the Deploy workflow sets `pi-release`; a stopped VM picks it up at its next boot):
  ```bash
  gh workflow run deploy.yml --ref dev                  # dev's newest commit with images (its last green CD run)
  gh workflow run deploy.yml --ref dev -f sha=<sha>     # a given commit of dev: a rollback is an older sha
  ```
  CD builds images only for merges that change an image input: a docs-only merge has none, and the default picks the
  newest commit that has them. The workflow refuses a sha older than the floor (MLflow on the VM, ML Phase 2 item 2;
  before it, `35b7c76` for L11) or without its three images, and skips the sha `pi-release` already names
  (`sudo pi-deploy now` redeploys it).
- **The VM runs `main` for the demo and in production** (decided 2026-11-22). Until `main` has been moved for the
  first time it has no workflows, so deploy from `dev` as above. Afterwards, deploy from `main`, the same way:
  ```bash
  gh workflow run deploy.yml --ref main                 # main's newest commit with images
  gh workflow run deploy.yml --ref main -f sha=<sha>    # a rollback: one of main's releases (a green CD run on main)
  ```
  Deploy a `dev` commit only to test it. In production (section 13), pause `cms_nh_release_check` first, so the test
  commit starts no production run, and never test a commit that upgrades Airflow or MLflow this way (its migration
  can't be undone, below). Then deploy `main` again and unpause it.
- **Moving `main`** (the lead, the only user `main`'s ruleset lets push): fast-forward it to `dev`'s tip. Nothing else
  moves it: no pull request is ever merged into `main` (the default branch, so check a PR's base is `dev`); a squash
  there would make the next fast-forward impossible.
  ```bash
  git fetch origin
  git push origin origin/dev:main     # git refuses it unless it is a fast-forward
  ```
  - **Not with a pull request's merge button:** `main` requires a linear history, so the button offers only squash or
    rebase, and both write new commits. CD would build every image again, and the next `dev` → `main` would conflict
    (2026-11-22: a squash of an older `dev`, then today's `dev`, conflicted in 58 to 101 files).
  - **Always `dev`'s tip, after its CI passed, and while no Terraform apply runs** (`gh run list --workflow
    terraform.yml --event push --limit 3`). Never an older commit: the push runs Terraform's apply as `main`, and an
    older commit would put back older infrastructure.
  - **What the push starts,** and the first time also tests (`main`'s workflow bindings have never been used,
    `infra/README.md`):
    1. When the push changes an image input, CD logs in as `sa-deployer`, skips the build when `dev` already
       published the commit, and tags the three images `main-<sha>` (rollback limit, below). Check its run is green;
       if it shows cancelled (another run of the same commit replaced it while it waited), run
       `gh workflow run cd.yml --ref main`.
    2. When the push changes Terraform's inputs, the apply logs in as `sa-tf-apply`: expect "No changes".
    3. Then deploy from `main` (above), which logs in as `sa-release`, and check `sudo pi-deploy status` names the
       commit.

    It also starts Dependabot, which reads `.github/dependabot.yml` only from `main`, and gives the Deploy workflow
    its Run button in the Actions tab.
- **A deploy waits for runs:** it doesn't replace the stack while a run of `cms_nh_pipeline`, `cms_nh_ml` or
  `cms_nh_release_check` is running or queued, and emails after 6 hours of waiting.
- **When it fails,** it emails `[pi-airflow VM] deploy of … failed …`, puts back the last good release if it got
  that far, and doesn't retry until `pi-release` changes or someone runs `sudo pi-deploy now`. Look at
  `sudo pi-deploy status` and `journalctl -u pi-deploy -n 50`.
- **Rollback limit: the registry deletes old images.** Artifact Registry keeps each image's 5 newest versions and
  deletes any other version once it is older than 7 days (untagged layers too). A sha older than 7 days with 5 newer
  releases is gone, and the VM itself keeps only the current and the previous release's images. A rollback
  to such a sha works only by hand while its images are still on the VM (`infra/vm/README.md`, "Deploys"). `main`'s
  releases are the exception: CD tags their images `main-<sha>` on a push to `main`, and the cleanup keeps every
  version with that tag. Only the commits `main` was moved to are releases, not every commit on it.
- **A release that upgrades Airflow** migrates its database, and a rollback can't undo that: `sudo pi-pg-backup`
  first, and deploy it by hand.

## 4. Run the pipeline by hand
- **Nothing runs on its own until T26** (production, section 13): the three DAGs are created paused. Keep
  `cms_nh_release_check` paused while you trigger runs by hand (a RESUME it finds would start another run).
  `cms_nh_ml` is unpaused since 2026-11-22 (section 14): every gold a pipeline run publishes starts an ML run.
- **Trigger and watch** (or use the UI):
  ```bash
  af dags unpause cms_nh_pipeline
  af dags trigger cms_nh_pipeline --run-id my-run-1 --conf '{"mode": "monthly"}'
  af dags list-runs cms_nh_pipeline
  af tasks states-for-dag-run cms_nh_pipeline my-run-1
  ```
  Params: `mode` (`monthly` | `backfill`), `stop_after` (`ingestion` | `bronze` | `silver`), `calibrate`, and
  `entry_ids` (development only; empty means every in-scope zip). All are described in the DAG's docstring.
- **A full backfill** (T25, into the bucket root, on an empty `precursor_meta`): node by node, so each layer can be
  checked before the next. A later run finds the earlier nodes' work done and gets past it in under a minute.

  | Run's `--conf` | Took on the VM (T25) |
  |---|---|
  | `{"mode": "backfill", "stop_after": "ingestion"}` | 24.0 min (69 zips downloaded in 138 s, then extracted) |
  | `{"mode": "backfill", "stop_after": "bronze"}` | 10.5 min |
  | `{"mode": "backfill", "stop_after": "silver"}` | 9.4 min |
  | `{"mode": "backfill"}` (gold) | 11.2 min (`build_gold` 356 s) |

  One `{"mode": "backfill"}` run does all of it in about an hour.
- **Between runs the VM is idle:** with no SSH session for 60 minutes it stops (T25: the laptop driving the runs went
  to sleep). Keep a session open and the laptop awake (`caffeinate -i` on macOS), or start the VM again and go on: the
  runs are idempotent.
- **Time limits:** every task stops after 60 min (`build_gold` 2 h) and a pipeline run after 8 h; a timed-out run
  keeps the run lock (section 8).

## 5. A trial prefix (smoke tests only)
To test without touching the bucket root, one line in `/etc/precursor/overrides.env` points the stack at
`gs://precursor-intel-mlops-data/trial-YYYY-MM-DD` (`infra/vm/README.md`, "Deploys", with the clean-up steps). What
T23 taught:
- start only on an empty `precursor_meta`, and keep `cms_nh_ml` paused;
- every deploy and boot applies the override and emails `… runs on a trial prefix` until it is removed;
- afterwards: remove the override and deploy, reset the databases (section 6), and a person deletes the prefix.

## 6. Backups, restore and reset
- **Dumps:** `pi-pg-backup` dumps `airflow`, `mlflow` and `precursor_meta` before every stop (and on demand:
  `sudo pi-pg-backup`) to `gs://precursor-intel-mlops-backups/postgres/<db>/<UTC time>-<random>.dump`, the same name
  for all three. Until a release with MLflow has run, `mlflow` doesn't exist and is skipped (it says so). The VM can
  add dumps but not read, list or delete them; they are kept 180 days.
- **Restore** into a new, empty Postgres, by a person with their own login: `infra/vm/README.md`, "Restore" (T23:
  110 s). It needs a release at least as new as the dump's; task logs are not restored. For `mlflow`, start
  `postgres-setup` too (it creates the empty database), never `mlflow` (its `db upgrade` fills the database first).
- **Reset** (empty databases, e.g. after a trial): the restore's step 3 without the two `pi-pg-restore` lines (T24).
  It empties `airflow` too (run history, queued asset events). The old database stays on the data disk as
  `/mnt/disks/pgdata/data.before-…`: check the disk with `sudo du -sh /mnt/disks/pgdata/*` and remove an old copy
  once it is no longer needed.

## 7. Let task containers use another database
Task containers open untrusted files and, on the VM, get `sa-pipeline`'s identity, which can read every secret. So
Postgres refuses every login from the task network's subnet `172.16.77.0/24` except `pi_meta` on `precursor_meta`
and `mlflow` on `mlflow` (`docker/airflow-local/init-db.sh`). These are the first three lines of `pg_hba.conf`, and
`pi-deploy` checks that they are exactly those three. To let tasks reach another database:
1. **Code, one PR:** create the role and database (as `postgres-setup` does for `mlflow`); add its line to
   `init-db.sh` above the reject line (new disks); and make `pi-deploy`'s `check_running` accept the new first lines.
   Without that last change, every deploy fails its final check, and so does the rollback, because the rule is on the
   disk.
2. **Get the new `pi-deploy` onto the VM:** the Terraform apply puts it in the metadata; the VM installs it at its
   next boot (or `sudo google_metadata_script_runner startup`, then `sudo systemctl restart pi-deploy`, with nothing
   running).
3. **On the existing disk,** which `init-db.sh` never touches again: add the line above the reject line, reload, check.
   ```bash
   sudo $dc exec -T postgres sh -c 'sed -i "/^host all all 172\.16\.77\.0\/24 reject\$/i host <db> <role> 172.16.77.0/24 scram-sha-256" "$PGDATA/pg_hba.conf"'
   sudo $dc exec -T postgres psql -U airflow -d airflow -c 'SELECT pg_reload_conf()'
   sudo $dc exec -T postgres psql -U airflow -d airflow -c 'SELECT line_number, database, user_name, address, auth_method FROM pg_hba_file_rules LIMIT 4'
   ```
4. **Deploy** the release with the new `init-db.sh`.

**Never change the task subnet** (`pi-airflow-tasks` in `docker-compose.yaml`) without the same edit on the disk: the
rules name the subnet, so with another one the reject line stops matching and a task can log in as any role whose
password it has, the superuser included. `pi-deploy` then fails every deploy ("don't match the network's subnet").

## 8. Alerts and what to do
Emails go to the `alert-email-to` secret's recipients, from the Gmail account in `alert-smtp-user`.

| Subject | From | What to do |
|---|---|---|
| `[cms_nh_pipeline] run failed: <tasks>` | the DAG's failure callback, once per run | open the log links; exit codes in `AIRFLOW_PIPELINE.md` §4 (137 = out of memory, section 10) |
| `[cms_nh_pipeline] run failed: see the run`, body `Reason: timed_out` | the run hit its 8 h limit | the run lock stays held: free it (below) or let the next run take it over after 12 h |
| `[cms_nh_pipeline] release_lock failed (the run may still show success)` | `release_lock`'s callback | free the lock (below) |
| `[cms_nh_pipeline] N new error anomalies` / `new CMS columns in N silver partitions` | the run, which goes on | read the listed lines (`ALERTS_HANDOFF.md` §6) |
| `[cms_nh_ml] <DECISION> <V>`, e.g. `SKIP 2026-10`, `PROMOTE 2026-10: 2 flag(s)` | `notify`, after every ML run | read it; a `held (…)`, a flag or `re-tune requested` in the subject needs a look (section 14) |
| `[cms_nh_ml] run failed: <tasks>` | the DAG's failure callback | open the log links; a REFUSE is exit 2 and isn't retried (section 14) |
| `[cms_nh_release_check] …` | the release check | a CMS release partial for days, or proceeds that start no run (`airflow-local.md` §5) |
| `[pi-airflow VM] deploy of … failed …` / `… runs on a trial prefix` | `pi-deploy` | section 3 / section 5 |
| `[pi-airflow VM] the VM stays up: …` / `… up 6+ hours and a run keeps it up` | the idle shutdown | a failed dump, deploys paused, or a long run: look, then stop it or fail the run |

**The run lock** (`data_pipeline`, a row in `precursor_meta`): check who holds it, and free it only when no pipeline
run is running or queued. The URL with the password stays in root's environment, never on a command line:
```bash
sudo bash -c 'sch=$(docker ps -q --filter label=com.docker.compose.project=pi-airflow --filter label=com.docker.compose.service=airflow-scheduler)
export PI_DATABASE_URL=$(docker exec "$sch" printenv PI_DATABASE_URL) PI_BUCKET_URI=$(docker exec "$sch" printenv PI_BUCKET_URI)
docker run --rm --network pi-airflow-tasks -e PI_DATABASE_URL -e PI_BUCKET_URI "$(docker exec "$sch" printenv PI_PIPELINE_IMAGE)" \
  precursor-ingest lock status --env prod'
```
To free it, the same with `precursor-ingest lock release --run-id <the holder's run id> --force --env prod`. (Tested
with `lock status` on the laptop stack, which adds its credentials file; the VM's tasks need none.)

## 9. Costs
Running ≈ $0.187 an hour (the machine $0.1808, Cloud NAT and its IP), stopped ≈ $4 a month (the disks); a normal
month ≈ $4.40–4.60, development weeks ≈ $7.75–11.50, left running all month ≈ $141. Every situation and the edge
cases: `infra/README.md`, "Cost of the VM". The budget alerts never stop spending: stop the VM when done.

## 10. Troubleshooting
| Symptom | Cause | Do |
|---|---|---|
| SSH fails with IAP error 4003 right after a start | the startup script isn't done | wait for "pi-startup: ready" on the serial console (section 2) |
| A task fails with exit 137 | the 20g task memory limit (`PI_TASK_MEM_LIMIT`): full-archive gold needs more than 10.46 GB (killed there under the old 10g limit, T25) | keep the limit and the 32 GB VM; a smaller VM needs the gold memory fix (`AIRFLOW_PIPELINE.md` §9.1) |
| A run seems stuck | a task over its limit fails by itself, but a run whose tasks stay queued stops only at its 8 h limit | look in the UI; fail the run by hand, then check the lock (section 8) |
| A deploy doesn't happen | a run is going on, deploys are paused, or the sha failed before | `sudo pi-deploy status`; `journalctl -u pi-deploy -n 50` |
| A script piped over SSH (`bash -s`) stops after `pi-deploy` or `docker compose` | they read stdin and swallow the rest of the script | give each `</dev/null` |
| The VM stopped between your runs | the idle shutdown (no SSH for 60 minutes) | start it again (section 4) |
| `gcloud storage cp` of a large object fails its CRC check | the sliced download | `CLOUDSDK_STORAGE_SLICED_OBJECT_DOWNLOAD_THRESHOLD=0 gcloud storage cp …` |
| The boot disk fills | images of two releases, Airflow task logs (not pruned yet) | `df -h /`, `sudo docker system df` |

## 11. MLflow
- **What runs:** the `mlflow` service of the release (image `…/precursor/mlflow:<sha>`, 1 GB memory limit), its
  Postgres database `mlflow`, and its artifacts in `gs://precursor-intel-mlops-mlflow/artifacts/vm/`, beside this
  VM's model packages in `packages/vm/` (#85). It writes as `sa-pipeline`, the VM's identity, through the metadata
  server; no key file. Task containers reach it as `http://mlflow:5000`. In prod an ML task without that address
  exits 3 instead of running and registering nothing (#73 B16).
- **Is it healthy?** `sudo $dc ps mlflow` shows `healthy`; `sudo $dc logs --tail 50 mlflow` shows why not.
- **Does a run reach the bucket?** On the VM (the pipeline image of the running release, on the task network):
  ```bash
  img=$(sudo docker inspect --format '{{.Config.Image}}' pi-airflow-mlflow-1 | sed 's|/mlflow:|/pipeline:|')
  sudo docker run --rm --network pi-airflow-tasks -e MLFLOW_TRACKING_URI=http://mlflow:5000 "$img" python -c "
  import mlflow
  mlflow.set_experiment('vm-smoke')
  with mlflow.start_run() as run:
      mlflow.log_text('hello', 'hello.txt')
  print(run.info.experiment_id, run.info.run_id)" </dev/null
  gcloud storage ls gs://precursor-intel-mlops-mlflow/artifacts/vm/<experiment id>/<run id>/artifacts/
  ```
  `hello.txt` is listed. Delete the `vm-smoke` experiment in the UI afterwards.
- **The first deploy with MLflow** (once). The merge's Terraform apply puts the new VM scripts in the instance
  metadata, but the VM installs them only at boot, and CD pushes the three images:
  1. Before it, check that Postgres's first `pg_hba.conf` lines are `pi_meta`, `mlflow`, reject
     (`docs/ML_PHASE2_HANDOFF.md`, "`pg_hba` (item 2)"); `pi-deploy` fails the deploy otherwise.
  2. **With the VM stopped,** run the Deploy workflow (section 3), then start the VM: the startup script installs the
     new scripts before `pi-deploy` deploys. If the VM is running, first install them by hand while no deploy runs
     (`sudo pi-deploy status`), so the old `pi-deploy` doesn't deploy it and the old `pi-pg-backup` doesn't leave
     `mlflow` out of the next dumps: `sudo google_metadata_script_runner startup && sudo systemctl restart pi-deploy`.
     `postgres-setup` creates the `mlflow` role and database, then `mlflow` starts.
  3. **Required:** the "Does a run reach the bucket?" check above. `/health` and CI's fake GCS don't test the VM's
     credentials, so this is the first real write as `sa-pipeline`.
  4. If the VM boots with the new scripts while `pi-release` still names the old release, `pi-deploy` refuses it (an
     email at every boot); the stack keeps running it and `pi-pg-backup` skips the missing `mlflow` database. Run
     the Deploy workflow then.
  5. If this first deploy fails after `up`, it rolls back to the release before MLflow, which runs no `mlflow`.
- **An MLflow version bump** runs `mlflow db upgrade` at start, which can't be undone: `sudo pi-pg-backup` before
  deploying it, and a rollback below it needs the `mlflow` dump restored.

## 12. BigQuery (the scores for SQL)
- **What:** `ml publish` loads each scored month into `precursor-intel-mlops.precursorintelligence_scores`: tables
  `facility_scores`, `facility_reasons` and `chain_scores`, one monthly partition per `score_month`. A rerun of
  `publish` for the month replaces exactly that partition (`WRITE_TRUNCATE`). Only this stack loads (it sets
  `PI_BIGQUERY_DATASET`); a laptop stack loads nothing. The SHAP matrix stays in GCS.
- **Query it** from a laptop, always with a `score_month` filter (the tables refuse a query without one) and a byte
  cap:
  ```bash
  bq query --project_id=precursor-intel-mlops --use_legacy_sql=false --maximum_bytes_billed=100000000 \
    'SELECT risk_decile, COUNT(*) AS n, ROUND(AVG(p), 3) AS mean_p FROM precursorintelligence_scores.facility_scores WHERE score_month = DATE "2026-10-01" GROUP BY 1 ORDER BY 1'
  ```
- **The latest month:** `SELECT MAX(score_month)` is refused (no month filter); read the partitions' metadata instead:
  ```bash
  bq query --project_id=precursor-intel-mlops --use_legacy_sql=false --maximum_bytes_billed=100000000 \
    'SELECT MAX(partition_id) AS latest FROM precursorintelligence_scores.INFORMATION_SCHEMA.PARTITIONS WHERE table_name = "facility_scores" AND partition_id NOT IN ("__NULL__", "__UNPARTITIONED__")'
  ```
  It prints `YYYYMM`. Don't take the month from `status/latest.json`: its `v` is the run's gold month even when
  nothing was scored (`n_facilities` 0, e.g. before the first champion).
- **Check a load:** the `publish` task's summary line has `bigquery_rows`, which must equal the manifest's `rows` for
  the three tables (`publish` refuses otherwise).
- **A trial prefix loads nothing:** `pi-deploy` blanks `PI_BIGQUERY_DATASET` whenever `overrides.env` sets one.
- **Changing a schema** (`configs/bigquery/`): only add `NULLABLE` columns at the end, or relax a mode. Removing,
  renaming or retyping a column, or making one `REQUIRED`, replaces the table, which `deletion_protection` and CI's
  apply guard refuse. Let the Terraform apply finish before deploying the image that writes the new column.

## 13. Production: the monthly CMS release (T26)
Decided 2026-11-22: production runs on `main` (section 3) and starts when the release check is unpaused, which then
starts the pipeline for each new CMS release. Nothing starts the VM: a person does, once CMS has published. CMS's
next release is due 2026-12-02.
1. **Before the first time:** the VM runs `main`'s release, and `cms_nh_ml` has a champion (ML Phase 2 item 3 and
   Checkpoint ML-A; done 2026-11-22, v1, section 14); without one, `score` skips and nothing is published.
2. **Has CMS published?** From a laptop, without the VM. The check starts the pipeline on READY: all 7 datasets
   report one `modified` month, newer than the latest ingested vintage. This is the only case this laptop check
   sees; the check also starts it on RESUME (the last pipeline work didn't finish) and CHANGED (unfinished landing
   work, e.g. a zip CMS re-posted):
   ```bash
   for id in 4pq5-n9py r5ix-sfxw svdt-c123 g6vv-u9sr djen-97ju ijh5-nb2v tagd-9999; do
     curl -s "https://data.cms.gov/provider-data/api/1/metastore/schemas/dataset/items/$id" |
       python3 -c 'import json, sys; d = json.load(sys.stdin); print(d["identifier"], d["modified"], d["released"])'
   done
   ```
   On 2026-11-22 all 7 showed `2026-10-01`, released 2026-10-28, with the next update due 2026-12-02; that release
   should show `2026-11-01` on all 7.
3. **Then start the VM** (section 2), not before: the idle shutdown doesn't count release-check runs, so a VM started
   before CMS publishes finds nothing (unless RESUME or CHANGED applies) and stops itself about 70–85 minutes after
   the last SSH session (section 0).
4. **The first time only, unpause the three DAGs in this order.** Airflow queues no asset event for a paused DAG, so
   the pipeline and the ML DAG go first:
   ```bash
   af dags unpause cms_nh_pipeline
   af dags unpause cms_nh_ml
   af dags unpause cms_nh_release_check
   af dags trigger cms_nh_release_check      # check now instead of at the next hour
   ```
   `cms_nh_ml` is already unpaused since 2026-11-22 (section 14); its line changes nothing then. They stay unpaused
   from then on. For each later release: steps 2 and 3, then the `trigger` line. For a manual run
   or a backfill, pause `cms_nh_release_check` first (section 4) and unpause it afterwards.
5. **What follows:** READY → a `cms_nh_pipeline` run → gold → a `cms_nh_ml` run → its `publish` to GCS and BigQuery
   (section 12). Failures email (section 8). The idle shutdown waits for both runs, then stops the VM.

## 14. ML runs (`cms_nh_ml`)
- **What starts it:** every published gold (the `cms_nh_gold` asset) starts a `mode=auto` run; or start one by hand.
  Unpaused on the VM since 2026-11-22 (ML Phase 2 item 3). One run at a time, no run lock; `train`, `tune` and
  `evaluate` share the `heavy` pool's one slot with gold's `build_gold`. A deploy and the idle shutdown wait for its
  runs. What each mode and task does: the DAG's docstring (`dags/cms_nh_ml.py`) and `ML_PIPELINE.md`.
- **Trigger and watch** (or use the UI):
  ```bash
  af dags trigger cms_nh_ml --run-id my-ml-1 --conf '{"mode": "auto"}'
  af tasks states-for-dag-run cms_nh_ml my-ml-1
  ```
  `mode`: `auto` (decides by itself), `retrain`, `score_only`, `retune`, or `promote`, which also needs `version`
  and `reason` (any earlier version = a rollback):
  ```bash
  af dags trigger cms_nh_ml --run-id my-promote-1 --conf '{"mode": "promote", "version": 1, "reason": "<who approved it, when, after reading what>"}'
  ```
- **Took on the VM** (item 3, 2026-11-22, gold V 2026-10):

  | Run | Took | Result |
  |---|---|---|
  | `vm-retrain-1`, `retrain` | about 22 min (`train` 12 min, `evaluate` 8 min) | v1 registered and held as `@challenger` (a first version always is); `score` skipped, no champion yet |
  | `vm-promote-1`, `promote` v1 | 2.1 min (`score` 28 s, `publish` 20 s) | v1 `@champion`; 14,686 homes scored and loaded to BigQuery |
  | `vm-auto-1`, `auto` | 0.4 min | SKIP: nothing new for this gold and recipe; nothing rewritten |
  | `vm-score-only-1`, `score_only` | 1.8 min | every score, reason, SHAP value and chain score identical to the promote's; only `ml_run_id` and `scored_at` differ |

- **Where the results are:** the run folder `gs://precursor-intel-mlops-data/ml/runs/<run id>/` (`decision.json`;
  after a retrain `report/model_card.md`, `gates.json`, `slices.csv`), the scores in `scores/v=<V>/` with
  `scores_manifest.json`, `status/latest.json`, BigQuery (section 12), the registry (section 11) and the email
  (section 8).
- **After a run that scored,** check: `publish`'s `bigquery_rows` equals the manifest's `rows` (2026-10: 14,686
  facility scores, 117,488 reasons, 612 chains); `status/latest.json` names the champion and its `n_facilities`; the
  latest-partition query (section 12) returns the month.
- **The first model on a new registry** (Checkpoint ML-A): `retrain`, then the lead reads the model card, the gates,
  the slices and 20 random homes' reasons, then `promote` with the lead's approval as the reason. The reasons script
  (`scripts/ml_checkpoint_reasons.py`) isn't in the pipeline image and needs `ml_runs`, so it runs on the VM. Copy it
  there from a laptop:
  ```bash
  gcloud compute scp scripts/ml_checkpoint_reasons.py pi-airflow:/tmp/ --zone us-central1-a --tunnel-through-iap
  ```
  On the VM, one line at a time. The database URL goes through a root-only file, never a command line (`sudo` logs
  those):
  ```bash
  img=$(sudo docker inspect --format '{{.Config.Image}}' pi-airflow-mlflow-1 | sed 's|/mlflow:|/pipeline:|')
  sudo ${dc} exec -T airflow-scheduler printenv PI_DATABASE_URL | sudo sh -c 'umask 077; sed "s/^/PI_DATABASE_URL=/" > /root/reasons.env'
  sudo docker run --rm --network pi-airflow-tasks --env-file /root/reasons.env -e PI_BUCKET_URI=gs://precursor-intel-mlops-data -e PI_PACKAGES_URI=gs://precursor-intel-mlops-mlflow/packages/vm -v /tmp/ml_checkpoint_reasons.py:/app/scripts/ml_checkpoint_reasons.py:ro "${img}" python scripts/ml_checkpoint_reasons.py --run-id <the retrain's run id> --version <its version> </dev/null > /tmp/reasons.txt
  sudo rm /root/reasons.env
  ```
  Copy `/tmp/reasons.txt` back with `gcloud compute scp` and send it to the lead with the model card.
- **The flags:** v1's runs flag skew on `state_gplus_rate_12m` and `staffing_rating_fn`, the same two as on the
  laptop stack's `ml-dev`. They are information: the run still scores and publishes.
