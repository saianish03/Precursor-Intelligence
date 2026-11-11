# infra/vm/

Scripts and systemd units that run on the Airflow VM itself, outside the containers. Terraform puts each file in a
metadata value of the VM (`infra/terraform/envs/prod/main.tf`), and the startup script installs them at every boot:
scripts in `/usr/local/bin`, units in `/etc/systemd/system`. A merged change therefore reaches the VM at its next start,
or at once with `sudo google_metadata_script_runner startup`. No file here may hold a secret: every process on the VM
can read its metadata.

| File | What |
|---|---|
| `startup.sh` | The VM's startup script (metadata `startup-script`), idempotent, run at every boot: a persistent journal; formats the Postgres data disk only when it holds nothing, mounts it at `/mnt/disks/pgdata` and creates `data/` only then; makes Docker wait for that mount; installs Docker CE and Compose at pinned versions (held); installs the files below and enables the two units. Log: `journalctl -u google-startup-scripts.service` |
| `pi-deploy` | Runs the release that the instance metadata `pi-release` names (below, "Deploys"): `watch` (the service), `now [SHA]`, `status`, `pause`, `resume`. |
| `pi-deploy.service` | Runs `pi-deploy watch`: deploys at boot, after the startup script, and whenever `pi-release` changes. Log: `journalctl -u pi-deploy` |
| `pi-busy` | Exit 0 while a run of `PI_BUSY_DAGS` (default `cms_nh_pipeline cms_nh_ml`) is running or queued, an asset event is queued for one, or one of their tasks runs; runs of a paused DAG count only by a running task. 1 when idle (also when no scheduler runs); 2 when it can't tell. Used by `pi-deploy` (which adds the release check) and later by the idle shutdown (T21). |
| `pi-alert` | `pi-alert SUBJECT < BODY`: emails the alert recipients from the VM itself (subject prefix `[pi-airflow VM]`), with the settings in `/etc/precursor/.env`, or straight from Secret Manager when that file is missing or lacks them (e.g. before the first deploy). For failures outside Airflow. |
| `pi-pg-backup` | Dumps `airflow` and `precursor_meta` (`pg_dump -Fc`, over the Postgres container's local socket) to `gs://precursor-intel-mlops-backups/postgres/<db>/<UTC time>-<random>.dump`, the same name for both. Prints each dump's MD5. Exits 0 only when both uploaded. |
| `pi-pg-restore` | `pi-pg-restore airflow\|precursor_meta <file.dump>`: restores a dump of that database into an **empty** database; refuses anything else (exit 2). |
| `pi-pg-backup.service` | Runs `pi-pg-backup` whenever the OS shuts down or Docker stops, while both are still up. A backstop: best effort within GCE's ~90 s. |

## Deploys (T20)
- **A release is one full git sha** whose two images CD pushed (`precursor/airflow:<sha>`, `precursor/pipeline:<sha>`).
  The instance metadata key `pi-release` names the release the VM runs. The Deploy workflow (T22) will set it; until
  then a person sets it (a stopped VM picks it up at its next boot):
  ```bash
  gcloud compute instances add-metadata pi-airflow --zone us-central1-a --metadata pi-release=<full sha>
  ```
- **What a deploy does:**
  1. pulls both images (unless they are already on the VM) and checks that each one's revision label is the sha;
  2. takes the compose bundle out of the Airflow image and writes the env file from Secret Manager, then checks them
     with `docker compose config -q`;
  3. waits while a run of `cms_nh_pipeline`, `cms_nh_ml` or `cms_nh_release_check` is running or queued (`pi-busy`),
     with one email after 6 hours, unless nothing changes for the running containers (e.g. at a boot). To deploy
     during a run anyway, stop the scheduler first;
  4. puts the bundle in `/opt/precursor/compose/` and the env file in `/etc/precursor/.env` (root, mode 600), then
     runs `docker compose up -d --wait`;
  5. checks that the Airflow containers run that image and that Postgres's task-network rule matches the network.
- **When it fails:**
  - A failure in steps 1–3 changes nothing and is tried twice more, a minute apart.
  - A failure from step 4 on puts back the files saved at the last good deploy (its bundle and env file, so its
    secrets too) at once. That also covers a redeploy of the same sha broken by a new secret.
  - A release that upgrades Airflow migrates its database, which a rollback can't undo: dump first
    (`sudo pi-pg-backup`) and deploy it by hand.
  - Every failure emails `[pi-airflow VM] deploy of … failed …`.
  - The watcher doesn't retry a failed sha until `pi-release` names another one, or a person runs `pi-deploy now`.
    Setting `pi-release` to the value it already has wakes nothing; use `now`.
- **Refused releases:** a sha whose bundle has no `docker-compose.vm.yaml` (before #43), or no task-network rule in
  `init-db.sh` (before L11, #46). The oldest sha the VM accepts is L11's merge, `35b7c76`.
- **On the VM:**
  ```bash
  sudo pi-deploy status              # pointer, deployed, previous and failed sha, paused, the last results
  sudo pi-deploy now [SHA]           # deploy by hand (default: the pointer's sha), even one that failed before
  sudo pi-deploy pause               # no deploy at all, by hand or by the watcher; `sudo pi-deploy resume` afterwards
  journalctl -u pi-deploy -n 50
  ```
  - `pause` returns once no deploy is running. A deploy that is still changing the stack must finish first: run
    `pause` again until it says so.
  - A sha deployed by hand with `now` stays until the next boot or the next change of `pi-release`; then the pointer
    wins again.
  - After re-running the startup script by hand, `sudo systemctl restart pi-deploy` (when no deploy runs): the
    running watcher keeps its old code.
  - Without SSH, `gcloud compute instances get-guest-attributes pi-airflow --zone us-central1-a --query-path=precursor/`
    shows the last result. It is informational only: anything on the VM may write it.
- **Secrets:** no value may hold a quote, a backslash or a line break; database passwords only letters, digits,
  `_` and `-` (they go into URLs). To rotate one, add a new version in Secret Manager, then `sudo pi-deploy now`. A
  database password needs `ALTER ROLE` in Postgres first, which keeps its own copy.
- **Images:** a deploy keeps the current and the previous release's images and deletes older ones.
- **State:** `/var/lib/precursor/` (`deployed-sha`, `previous-sha`, `failed-sha`, `history`, `deploy.paused`, and
  `good/`, the last good release's files).

## Backups (spec §14 decision 27)
- **Who runs it:** `pi-down` and the idle shutdown (T21) run `pi-pg-backup` first and stop the VM only if it succeeds. The unit dumps again at any OS shutdown, e.g. a stop from the console. A dump waiting more than 30 s for a lock, or an upload taking more than 2 minutes, fails instead of keeping the VM up.
- **Write-only:** the VM's account (`sa-pipeline`) may create objects in the backups bucket but not read, list, overwrite or delete them. A broken or compromised VM cannot destroy its backups, and a task container, which gets the VM's identity, cannot read them. `gcloud storage cp` needs to read the destination, so each dump goes up in one Storage JSON API request instead: it fails on an existing name (`ifGenerationMatch=0`), and the MD5 the bucket reports back must equal the local file's.
- **But it can add files:** a compromised task could upload a fake dump, and a restore runs the dump's SQL as the Postgres superuser. So a person picks dumps by creation time and checks them (step 1 below), and the random part of each name keeps anyone from blocking the next upload by taking its name first.
- **Readable by people:** project Owners, Editors and Viewers can download dumps. They hold no secrets: Airflow stores no connections or variables, and passwords are never in the databases' tables.
- **Kept 180 days** (the bucket's lifecycle rule). Each dump is a few MB.

## Restore (a person, with their own login)
A restore always goes into a new Postgres: an empty data folder, so that `init-db.sh` first creates the `pi_meta` role,
the access rules and the `REVOKE`s, and the dumps then bring back the data. It needs a release at least as new as
the one that made the dump (`airflow db migrate` can't go backwards). Task logs are not restored, only run history.

1. **On your laptop, pick the dumps** by creation time, not by name, and check them:
   ```bash
   B=gs://precursor-intel-mlops-backups/postgres
   gcloud storage ls -l $B/airflow/ | grep -v '^TOTAL' | sort -k2 | tail -3          # size, creation time, name
   gcloud storage ls -l $B/precursor_meta/ | grep -v '^TOTAL' | sort -k2 | tail -3
   ```
   - The same name must exist for both databases. A name listed for only one is a failed run.
   - The time in the name must match the creation time to within a minute.
   - If the VM's journal is still there, its `pi-pg-backup` lines show each dump's MD5; compare it with
     `gcloud storage ls -L <object>` ("Hash (MD5)").
2. **Download them and copy them to the VM:**
   ```bash
   N=<name>   # e.g. 20261107T183000Z-1a2b3c4d
   gcloud storage cp $B/airflow/$N.dump ./airflow-$N.dump
   gcloud storage cp $B/precursor_meta/$N.dump ./precursor_meta-$N.dump
   gcloud compute scp --zone us-central1-a --tunnel-through-iap ./*-$N.dump pi-airflow:/tmp/
   ```
3. **On the VM** (`gcloud compute ssh pi-airflow --zone us-central1-a --tunnel-through-iap`). Pause the deploy
   watcher and the idle shutdown (T21) first, so neither acts in the middle.
   ```bash
   sudo pi-deploy pause                      # go on only once it says no deploy is running
   cd /opt/precursor/compose
   dc="docker compose -f docker-compose.yaml -f docker-compose.vm.yaml --env-file /etc/precursor/.env"
   sudo $dc down
   # keep the current database aside instead of deleting it
   sudo mv /mnt/disks/pgdata/data /mnt/disks/pgdata/data.before-restore-$(date -u +%Y%m%dT%H%M%SZ)
   sudo mkdir /mnt/disks/pgdata/data
   sudo $dc up -d --wait postgres            # a new, empty Postgres: init-db.sh runs
   until sudo $dc exec -T postgres pg_isready -q -h 127.0.0.1; do sleep 1; done   # the real server, not the init one
   sudo pi-pg-restore airflow /tmp/airflow-$N.dump
   sudo pi-pg-restore precursor_meta /tmp/precursor_meta-$N.dump
   sudo $dc up -d --wait                     # the rest
   sudo $dc exec airflow-scheduler airflow dags list-runs cms_nh_pipeline   # the run history is back
   sudo rm /tmp/*-$N.dump
   sudo pi-deploy resume
   ```
4. **Check what came back:** a run that was running or queued when the dump was taken shows as such, and a queued
   one starts at once. A run lock taken at that moment is taken over only after 12 h:
   `docker run --rm --network pi-airflow-tasks -e PI_DATABASE_URL=… <pipeline image> precursor-ingest lock status`.

On a laptop: the same steps with a new Postgres volume (another compose project, or `down -v` when the old one may
go), and `PI_COMPOSE_PROJECT=<project>` for `pi-pg-restore`.
