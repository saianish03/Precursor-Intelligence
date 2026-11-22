# Runbook: the GCP VM

How to operate the Airflow VM day to day: get in, start and stop it, deploy and roll back, run the pipeline,
restore its databases, and what to do when something goes wrong. The reference details stay where they are:
`infra/vm/README.md` (the scripts on the VM: deploys, up and down, backups, restore), `infra/README.md` (Terraform,
the VM's build, its cost) and `airflow-local.md` §5 (the stack's settings on the VM). The timings come from the
smoke test and the first full backfill on the VM (T23, T25: `airflow-local.md` §8.4, §8.5).

MLflow runs on the VM since ML Phase 2's item 2 (section 11). `cms_nh_ml` on the VM is item 3
(`docs/ML_PHASE2_HANDOFF.md` §5.2); its owner adds that part here.

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
- **A deploy waits for runs:** it doesn't replace the stack while a run of `cms_nh_pipeline`, `cms_nh_ml` or
  `cms_nh_release_check` is running or queued, and emails after 6 hours of waiting.
- **When it fails,** it emails `[pi-airflow VM] deploy of … failed …`, puts back the last good release if it got
  that far, and doesn't retry until `pi-release` changes or someone runs `sudo pi-deploy now`. Look at
  `sudo pi-deploy status` and `journalctl -u pi-deploy -n 50`.
- **Rollback limit: the registry deletes old images.** Artifact Registry keeps each image's 5 newest versions and
  deletes any other version once it is older than 7 days (untagged layers too). A sha older than 7 days with 5 newer
  releases is gone, and the VM itself keeps only the current and the previous release's images. A rollback
  to such a sha works only by hand while its images are still on the VM (`infra/vm/README.md`, "Deploys"). Fine for
  the demo; protecting every deployed release would need a `release-<sha>` tag and a KEEP rule for it.
- **A release that upgrades Airflow** migrates its database, and a rollback can't undo that: `sudo pi-pg-backup`
  first, and deploy it by hand.

## 4. Run the pipeline by hand
- **Nothing runs on its own until T26:** the three DAGs are created paused. Keep `cms_nh_release_check` paused while
  you trigger runs by hand (a RESUME it finds would start another run), and `cms_nh_ml` paused until its Phase 2
  deploy.
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
