# infra/vm/

Scripts and systemd units that run on the Airflow VM itself, outside the containers. The VM's startup script (task
T19) installs the scripts in `/usr/local/bin` and the units in `/etc/systemd/system`.

| File | What |
|---|---|
| `pi-pg-backup` | Dumps `airflow` and `precursor_meta` (`pg_dump -Fc`, over the Postgres container's local socket) to `gs://precursor-intel-mlops-backups/postgres/<db>/<UTC time>-<random>.dump`, the same name for both. Prints each dump's MD5. Exits 0 only when both uploaded. |
| `pi-pg-restore` | `pi-pg-restore airflow\|precursor_meta <file.dump>`: restores a dump of that database into an **empty** database; refuses anything else (exit 2). |
| `pi-pg-backup.service` | Runs `pi-pg-backup` whenever the OS shuts down or Docker stops, while both are still up. A backstop: best effort within GCE's ~90 s. |

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
   gcloud storage ls -l $B/airflow/ | sort -k2 | tail -3          # size, creation time, name
   gcloud storage ls -l $B/precursor_meta/ | sort -k2 | tail -3
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
3. **On the VM** (`gcloud compute ssh pi-airflow --zone us-central1-a --tunnel-through-iap`), in the compose bundle
   folder. Pause the deploy watcher (T20) and the idle shutdown (T21) first, so neither acts in the middle.
   ```bash
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
   ```
4. **Check what came back:** a run that was running or queued when the dump was taken shows as such, and a queued
   one starts at once. A run lock taken at that moment is taken over only after 12 h:
   `docker run --rm --network pi-airflow-tasks -e PI_DATABASE_URL=… <pipeline image> precursor-ingest lock status`.

On a laptop: the same steps with a new Postgres volume (another compose project, or `down -v` when the old one may
go), and `PI_COMPOSE_PROJECT=<project>` for `pi-pg-restore`.
