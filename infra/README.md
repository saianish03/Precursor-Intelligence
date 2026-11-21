# infra/terraform/

One project, `precursor-intel-mlops`, one environment: `envs/prod`. The reusable pieces are in `modules/`:

| Module | Contents |
|---|---|
| `gcs` | A bucket: versioning, uniform access, public access blocked, optional `quarantine/` or whole-bucket expiry, `prevent_destroy` |
| `wif` | The GitHub workload identity pool, and the `sa-tf-plan` / `sa-tf-apply` accounts with their bindings |
| `artifact_registry` | The Docker repository: immutable tags, keep the 5 newest versions, delete the rest after 30 days and untagged after 7, `prevent_destroy` |

`bigquery`, `cloud_run_job`, `cloud_run_service` and `composer` are still empty placeholders.

`infra/vm/` holds the scripts and systemd units that run on the VM itself (deploys, Postgres backup and restore, the
idle shutdown).

`envs/prod` creates:
- the data, DVC, state, backups and MLflow buckets (the last holds the MLflow server's artifacts, no expiry);
- `sa-pipeline` (the VM's identity) and its access: the data bucket, upload only on the backups bucket (never read, list,
  overwrite or delete), objects in the MLflow bucket, pull from the image repository, read on each VM secret, and writing logs and metrics;
- the image repository `precursor` and `sa-deployer`, which only runs on `dev` or `main` can act as (CD pushes);
- `sa-release`, the same way (the Deploy workflow): sets `pi-release` on the VM (`compute.instanceAdmin.v1` on that
  instance, actAs `sa-pipeline`), reads the image repository and compute settings;
- nine empty Secret Manager secrets for the VM (values are added by hand, never by Terraform);
- the VM network: `pi-vpc`, `pi-subnet` with Private Google Access, Cloud NAT, and SSH from IAP as the only
  inbound rule;
- the VM `pi-airflow` (e2-highmem-4, 4 vCPU and 32 GB, Debian 12, Shielded VM, OS Login, no external IP, runs as `sa-pipeline`) and its
  Postgres data disk `pi-airflow-pgdata` (section "The VM");
- the `wif` module.

The data and state buckets, `sa-pipeline`, the CI accounts and one `sa-tf-apply` role were created by hand
before Terraform could manage them (`docs/runbooks/gcp-bootstrap.md`); `imports.tf` adopts them so nothing is
recreated.

## The VM
- **What Terraform builds:** the machine only. A 30 GB boot disk, plus a 10 GB data disk for Postgres that outlives the
  VM (`prevent_destroy`). The startup script (`infra/vm/startup.sh`) installs Docker and the files of `infra/vm/` at
  every boot. What runs on it is the release that the instance metadata `pi-release` names: `pi-deploy` deploys it at
  boot and whenever it changes (`infra/vm/README.md` "Deploys"); the Deploy workflow (T22) sets it. Terraform
  ignores that pointer and new Debian images, so neither ever replaces the VM. CI refuses any replacement anyway.
- **Created running:** the apply that creates it starts it. It costs about $0.19 an hour ($4.50 a day); a stopped VM
  costs only its disks (about $4 a month). Details: "Cost of the VM" below. The idle shutdown (T21) stops it after an hour with no run and no SSH
  connection; stop it yourself with `sudo pi-down` when you're done (`infra/vm/README.md`, "Up and down").
  ```bash
  gcloud compute instances start pi-airflow --zone us-central1-a
  gcloud compute ssh pi-airflow --zone us-central1-a --tunnel-through-iap
  ```
- **An apply never stops it:** a change to the machine type, service account, scopes or Shielded VM settings fails while
  the VM runs. Stop the VM and re-run the apply job. Metadata changes (the files in `infra/vm/`) apply while it runs and
  take effect at the next boot.
- **Re-running the startup script** (`sudo google_metadata_script_runner startup`) after a change to the Docker pins
  restarts Docker and everything in it: do it only when nothing is running.
- **Until the first deploy,** every stop logs a failed `pi-pg-backup` ("no running postgres container") and the idle
  shutdown emails that it stops without a backup. That is expected: there is no database yet.
- **Check it after a change** (on the VM):
  ```bash
  sudo journalctl -u google-startup-scripts.service -b | grep pi-startup   # contains "pi-startup: ready: ..."
  findmnt /mnt/disks/pgdata && sudo ls -ld /mnt/disks/pgdata/data
  systemctl show docker -p RequiresMountsFor
  sudo docker info --format '{{.ServerVersion}} {{.SecurityOptions}}' && docker compose version   # 28.5.2, apparmor
  systemctl is-enabled pi-pg-backup && systemctl is-active pi-pg-backup
  systemctl is-active pi-deploy && sudo pi-deploy status             # the watcher, and what it deployed
  systemctl list-timers pi-idle-shutdown.timer                       # the idle shutdown's next check
  sudo journalctl --list-boots | tail -3                             # the journal survives reboots
  apt-config dump APT::Periodic::Unattended-Upgrade                  # "1": Debian security updates in place
  systemctl is-enabled apt-daily-upgrade.timer
  sudo true && echo sudo-ok                                          # OS Login gives you sudo (roles/compute.osAdminLogin)
  ```
  The next CI plan after the apply must show no changes for the instance and the disk.
- **When SSH fails:** the startup script also logs to the serial console:
  `gcloud compute instances get-serial-port-output pi-airflow --zone us-central1-a | grep pi-startup`.

## Cost of the VM (e2-highmem-4, us-central1, list prices checked 2026-11-21)
On-demand list prices, before the $300 trial credit; E2 machines get no sustained-use discount. Check the
[pricing calculator](https://cloud.google.com/products/calculator) or the billing report for exact figures.

| Part | Price | Charged |
|---|---|---|
| Machine (4 vCPU, 32 GB) | **$0.1808/h** ([general-purpose pricing](https://cloud.google.com/products/compute/pricing/general-purpose)); e2-standard-4 was $0.1340/h | only while the VM runs |
| Cloud NAT, per VM | $0.0014/h ([Cloud NAT pricing](https://cloud.google.com/nat/pricing)) | while the VM uses it |
| Cloud NAT data | $0.045 per GiB | CMS downloads only: ≈ 38 MB a month, 2.6 GB for a full backfill (images come from Artifact Registry over Private Google Access, not NAT) |
| Disks: 30 GB boot + 10 GB Postgres, pd-balanced | ≈ $0.10 per GB-month, ≈ **$4 a month** ([disk pricing](https://cloud.google.com/compute/disks-image-pricing)) | always, also while stopped |
| NAT external IP (`AUTO_ONLY`) | ≈ $0.005/h | while the VM runs; $0–3.65 a month if Google keeps the IP while it is stopped (the billing report shows which) |

**Running: ≈ $0.187 an hour** (the machine $0.1808, the NAT $0.0014 and its IP $0.005; the e2-standard-4 was ≈ $0.140). **Stopped: ≈ $4 a month.**

| Situation | Running hours a month | Cost a month (e2-standard-4) |
|---|---|---|
| Stopped all month | 0 | **≈ $4** (up to ≈ $7.65 with a held NAT IP) |
| Production: one CMS release a month. **A person starts the VM on the release day** (e.g. 2026-12-02), since nothing starts it automatically: the hourly release check runs only while the VM is up. Then a ~1–1.5 h run (the new zip, silver, a full gold rebuild, the ML retrain and scores) and the idle shutdown's ~1–1.5 h. Keeping the VM up to wait for a release is the last row | 2–3 | **≈ $4.40–4.60** ($4.30–4.40) |
| Development or demo weeks (VM sessions of 2–4 h on about 10 days) | 20–40 | **≈ $7.75–11.50** ($6.80–9.60) |
| A heavy month (debugging, several backfills) | 100 | **≈ $22.70** ($18) |
| One extra full backfill | ≈ 1 | **+ ≈ $0.30** (with 2.6 GB through NAT) |
| **Edge: left running a day / a week** (an open SSH session, `pi-deploy pause` left on, a hung run) | 24 / 168 | **+ ≈ $4.50 / + ≈ $31.50** on top of the month's other rows |
| **Edge: left running all month** | 730 | **≈ $141** ($106) |

- **The upgrade (T25, #90) adds $0.047 per running hour:** about $1–2 in a normal month, $34 only if the VM ran all
  month.
- **What stops the edge cases:** the idle shutdown (about 70–85 minutes after the last session), the 6-hour reminder for
  a forgotten pause, the 6-hour email for a run that keeps the VM up, and the project budget alerts. An open SSH session
  keeps the VM up: close it when done.
- **Around the VM, the same in every situation:** GCS ≈ 52 GB (root + `ml-dev/`) ≈ $1; Artifact Registry (≤ 5 releases
  of two images) ≈ $1–2; Secret Manager (9 secrets) ≈ $0.55; backups cents; together ≈ $3–5 a month.
- **Not used:** Spot VMs (about half price) can be stopped mid-run; committed-use discounts need a 1- or 3-year term.

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
