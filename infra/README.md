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
- the `wif` module.

The data and state buckets, `sa-pipeline`, the CI accounts and one `sa-tf-apply` role were created by hand
before Terraform could manage them (`docs/runbooks/gcp-bootstrap.md`); `imports.tf` adopts them so nothing is
recreated.

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
