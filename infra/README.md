# infra/terraform/

One project, `precursor-intel-mlops`, one environment: `envs/prod`. The reusable pieces are in `modules/`:

| Module | Contents |
|---|---|
| `gcs` | A bucket: versioning, uniform access, public access blocked, optional `quarantine/` or whole-bucket expiry, `prevent_destroy` |
| `wif` | The GitHub workload identity pool, and the `sa-tf-plan` / `sa-tf-apply` accounts with their bindings ("Which workflow can act as which account") |
| `artifact_registry` | The Docker repository: immutable tags, keep the 5 newest versions, delete the rest after 30 days and untagged after 7, `prevent_destroy` |

`bigquery`, `cloud_run_job`, `cloud_run_service` and `composer` are still empty placeholders.

`infra/vm/` holds the scripts and systemd units that run on the VM itself (deploys, Postgres backup and restore, the
idle shutdown).

`envs/prod` creates:
- the data, DVC, state, backups and MLflow buckets (the last holds the MLflow server's artifacts, no expiry);
- `sa-pipeline` (the VM's identity) and its access: the data bucket, upload only on the backups bucket (never read, list,
  overwrite or delete), objects in the MLflow bucket, pull from the image repository, read on each VM secret, and writing logs and metrics;
- the image repository `precursor` and `sa-deployer`, which only CD's runs on `dev` or `main` can act as (it pushes);
- `sa-release`, only the Deploy workflow's runs on those branches ("Which workflow can act as which account"): sets `pi-release` on the VM (`compute.instanceAdmin.v1` on that
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
| **Edge: left running a day / a week** (an open SSH session, `pi-deploy pause` left on; a hung run now ends within 8 h, L5) | 24 / 168 | **+ ≈ $4.50 / + ≈ $31.50** on top of the month's other rows |
| **Edge: left running all month** | 730 | **≈ $141** ($106) |

- **The upgrade (T25, #90) adds $0.047 per running hour:** about $1–2 in a normal month, $34 only if the VM ran all
  month.
- **What stops the edge cases:** the idle shutdown (about 70–85 minutes after the last session), the time limits on
  every task and run (L5: a pipeline run stops after at most 8 h), the 6-hour reminder for a forgotten pause, the
  6-hour email for a run that keeps the VM up, and the project budget alerts. An open SSH session keeps the VM up:
  close it when done.
- **Around the VM, the same in every situation:** GCS ≈ 52 GB (root + `ml-dev/`) ≈ $1; Artifact Registry (≤ 5 releases
  of two images) ≈ $1–2; Secret Manager (9 secrets) ≈ $0.55; backups cents; together ≈ $3–5 a month.
- **Not used:** Spot VMs (about half price) can be stopped mid-run; committed-use discounts need a 1- or 3-year term.

## How changes reach GCP

- Open a pull request that touches `infra/`. CI runs `fmt`, `validate` and `plan` as the read-only account; read
  the plan in the job summary.
- A teammate reviews the plan. Merging into `dev` or `main` runs `apply` as `sa-tf-apply`.
- Do not change these resources in the console, and never run `apply` from a laptop.

## Which workflow can act as which account
The provider maps `attribute.workflow` = the token's subject (with GitHub's permanent owner and repository ids) + `:` +
`job_workflow_ref` (the workflow file and branch that asked for the token), so an account can be bound to one workflow
file. A compromised action in one workflow then can't act as another workflow's account.

| Account | Bound to (on `dev` and `main`) | Old binding: any workflow on the branch |
|---|---|---|
| `sa-tf-apply` | `terraform.yml` (`apply_from_workflow`, `modules/wif`) | `apply_from_branch`: removed last |
| `sa-deployer` | `cd.yml` (`deployer_from_workflow`) | `deployer_from_branch` |
| `sa-release` | `deploy.yml` (`release_from_workflow`) | `release_from_branch`: removed first |
| `sa-tf-plan` | unchanged: any pull request (read-only; a PR controls its own workflow files anyway) | — |

**Removing the old bindings (an Owner, once, by hand).** Terraform no longer manages them (`removed` blocks with
`destroy = false`: CI refuses deletes), but they stay in GCP, and an IAM binding grants if any member matches, so each
account is restricted only once its old binding is gone. One account per step, each step only after the previous check
passed. Putting a binding back by hand (`add-iam-policy-binding`, same arguments) never conflicts with Terraform.
```bash
P=precursor-intel-mlops
S=principal://iam.googleapis.com/projects/852874974077/locations/global/workloadIdentityPools/github/subject/repo:saianish03@89864897/Precursor-Intelligence@1402453226
drop() { for b in dev main; do gcloud iam service-accounts remove-iam-policy-binding "$1@$P.iam.gserviceaccount.com" \
  --role=roles/iam.workloadIdentityUser --member="$S:ref:refs/heads/$b" --project=$P --condition=None >/dev/null || return 1; done; }
left() { gcloud iam service-accounts get-iam-policy "$1@$P.iam.gserviceaccount.com" --project=$P --format='value(bindings.members)'; }
```
0. **Before removing anything,** run step 1's check once. The provider computes `attribute.workflow` on every login,
   so if Google refused the new value (about 155 characters; a length limit is documented only for `google.subject`),
   every login would fail, through the old bindings too: then use the mapping rollback below. A login that works here
   proves the value is accepted, and step 1 then tests only that the new binding matches it.
1. `drop sa-release`, then `left sa-release` shows only the two `principalSet://…/deploy.yml@…` members. **Wait 10
   minutes** (IAM takes minutes to apply a removal; an earlier check could still pass through the old binding). Check:
   ```bash
   PTR=$(gcloud compute instances describe pi-airflow --zone us-central1-a --project $P --format='value(metadata.items.pi-release)')
   [ -n "$PTR" ] && gh workflow run deploy.yml --ref dev -f sha="$PTR"
   ```
   It passes when the run's summary says `pi-release` already names that sha (it logs in, then changes nothing). A failed
   "Both images in the registry?" step is a refused login only if its log shows a permission error from the token
   exchange (the auth step itself only prepares the credentials).
2. `drop sa-deployer`, `left sa-deployer`, wait 10 minutes. Check: rerun dev's last green CD run (it logs in; its sha's
   images exist, so nothing is pushed). GitHub reruns only runs from the last 30 days; otherwise `gh workflow run cd.yml
   --ref dev`, which may build and push the tip's images (harmless).
   ```bash
   gh run rerun "$(gh run list --workflow cd.yml --branch dev --status success -L 1 --json databaseId -q '.[0].databaseId')"
   ```
3. Only after both checks: `drop sa-tf-apply`, `left sa-tf-apply`, wait 10 minutes. Check: rerun the apply of the
   merge that added these bindings, or of any later commit, **never an older one** (it would apply the old config and
   put the old bindings back); its plan has no changes. If it can't log in, put the binding back by hand and rerun.

- **`main` is untested:** it has never run these workflows, so its bindings are first exercised by the first run there
  (do one before relying on `main`, e.g. for the demo).
- **New CI accounts start workflow-bound:** bind them with `module.wif.workflow_principal_set` (as
  `deployer_from_workflow` does), never `subject_prefix`, so they never need this migration. E.g. the agents plan's
  `sa-api-deployer` (`deploy-api.yml`, L9) and `sa-llm-eval` (`llm-eval.yml`, J3).
- **Renaming `terraform.yml`, `cd.yml` or `deploy.yml`,** or moving its login into a reusable workflow (whose file
  `job_workflow_ref` then names), breaks that account's login once the old bindings are gone; for `terraform.yml` CI
  can't repair it itself. Add the new binding in one PR, rename in the next.
- **If every CI login fails right after the provider change** (the mapping can't be evaluated), an Owner puts the old
  mapping back, and a PR reverts it in `modules/wif` (else the next apply sets it again). This is safe only while the
  old subject bindings still exist; after step 3 it would lock out all three accounts:
  `gcloud iam workload-identity-pools providers update-oidc github-repo --workload-identity-pool=github --location=global --project=precursor-intel-mlops --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository"`

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
