# infra/terraform/

One project, `precursor-intel-mlops`, one environment: `envs/prod`. The reusable pieces are in `modules/`:

| Module | Contents |
|---|---|
| `gcs` | A bucket: versioning, uniform access, public access blocked, optional `quarantine/` or whole-bucket expiry, `prevent_destroy` |
| `wif` | The GitHub workload identity pool, and the `sa-tf-plan` / `sa-tf-apply` accounts with their bindings |
| `artifact_registry` | The Docker repository: immutable tags, keep the 5 newest versions, delete the rest after 30 days and untagged after 7, `prevent_destroy` |

`bigquery`, `cloud_run_job`, `cloud_run_service` and `composer` are still empty placeholders.

`infra/vm/` holds the scripts and systemd units that run on the VM itself (Postgres backup and restore).

`envs/prod` creates:
- the data, DVC, state and backups buckets;
- `sa-pipeline` (the VM's identity) and its access: the data bucket, upload only on the backups bucket (never read, list,
  overwrite or delete), pull from the image repository, read on each VM secret, and writing logs and metrics;
- the image repository `precursor` and `sa-deployer`, which only runs on `dev` or `main` can act as (CD pushes);
- eight empty Secret Manager secrets for the VM (values are added by hand, never by Terraform);
- the VM network: `pi-vpc`, `pi-subnet` with Private Google Access, Cloud NAT, and SSH from IAP as the only
  inbound rule;
- the VM `pi-airflow` (e2-standard-4, Debian 12, Shielded VM, OS Login, no external IP, runs as `sa-pipeline`) and its
  Postgres data disk `pi-airflow-pgdata` (section "The VM");
- the `wif` module.

The data and state buckets, `sa-pipeline`, the CI accounts and one `sa-tf-apply` role were created by hand
before Terraform could manage them (`docs/runbooks/gcp-bootstrap.md`); `imports.tf` adopts them so nothing is
recreated.

## The VM
- **What Terraform builds:** the machine only. A 30 GB boot disk, plus a 10 GB data disk for Postgres that outlives the
  VM (`prevent_destroy`). The startup script (`infra/vm/startup.sh`) installs Docker and the backup scripts at every boot.
  What runs on it comes from the release pointer `pi-release`, which the Deploy workflow sets (T20, T22). Terraform
  ignores that pointer and new Debian images, so neither ever replaces the VM. CI refuses any replacement anyway.
- **Created running:** the apply that creates it starts it. Stop it when you're done: until the idle shutdown exists
  (T21) nothing stops it for you, and it costs about $0.13 an hour ($3.22 a day). A stopped VM costs only its disks
  (about $4 a month).
  ```bash
  gcloud compute instances stop pi-airflow --zone us-central1-a      # start: ... start ...
  gcloud compute ssh pi-airflow --zone us-central1-a --tunnel-through-iap
  ```
- **An apply never stops it:** a change to the machine type, service account, scopes or Shielded VM settings fails while
  the VM runs. Stop the VM and re-run the apply job. Metadata changes (the files in `infra/vm/`) apply while it runs and
  take effect at the next boot.
- **Re-running the startup script** (`sudo google_metadata_script_runner startup`) after a change to the Docker pins
  restarts Docker and everything in it: do it only when nothing is running.
- **Until the deploy exists (T20),** every stop logs a failed `pi-pg-backup` ("no running postgres container"). That is
  expected: there is no database yet.
- **Check it after a change** (on the VM):
  ```bash
  sudo journalctl -u google-startup-scripts.service -b | grep pi-startup   # contains "pi-startup: ready: ..."
  findmnt /mnt/disks/pgdata && sudo ls -ld /mnt/disks/pgdata/data
  systemctl show docker -p RequiresMountsFor
  sudo docker info --format '{{.ServerVersion}} {{.SecurityOptions}}' && docker compose version   # 28.5.2, apparmor
  systemctl is-enabled pi-pg-backup && systemctl is-active pi-pg-backup
  sudo journalctl --list-boots | tail -3                             # the journal survives reboots
  apt-config dump APT::Periodic::Unattended-Upgrade                  # "1": Debian security updates in place
  systemctl is-enabled apt-daily-upgrade.timer
  sudo true && echo sudo-ok                                          # OS Login gives you sudo (roles/compute.osAdminLogin)
  ```
  The next CI plan after the apply must show no changes for the instance and the disk.
- **When SSH fails:** the startup script also logs to the serial console:
  `gcloud compute instances get-serial-port-output pi-airflow --zone us-central1-a | grep pi-startup`.

## How changes reach GCP

- Open a pull request that touches `infra/`. CI runs `fmt`, `validate` and `plan` as the read-only account; read
  the plan in the job summary.
- A teammate reviews the plan. Merging into `dev` or `main` runs `apply` as `sa-tf-apply`.
- Do not change these resources in the console, and never run `apply` from a laptop.

## Running `plan` locally (read-only)

```bash
gcloud auth application-default login
cd infra/terraform/envs/prod
terraform init
terraform plan -lock=false
```
State lives in `gs://precursor-intel-mlops-tfstate` (prefix `prod`). CI's apply locks it during a run; a local
plan passes `-lock=false` so it never writes to the state bucket.
State and `*.tfvars` are git-ignored; commit `.terraform.lock.hcl`.
