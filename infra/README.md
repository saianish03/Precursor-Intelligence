# infra/terraform/

One project, `precursor-intel-mlops`, one environment: `envs/prod`. The reusable pieces are in `modules/`:

| Module | Contents |
|---|---|
| `gcs` | A bucket: versioning, uniform access, public access blocked, optional `quarantine/` expiry, `prevent_destroy` |
| `wif` | The GitHub workload identity pool, and the `sa-tf-plan` / `sa-tf-apply` accounts with their bindings |

`bigquery`, `artifact_registry`, `cloud_run_job`, `cloud_run_service` and `composer` are still empty placeholders.

`envs/prod` creates the two buckets and `sa-pipeline`, and calls `wif`. The first three things in it, and the CI
accounts, were created by hand before Terraform could run (`docs/runbooks/gcp-bootstrap.md`); `imports.tf`
adopts them so nothing is recreated.

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
terraform plan
```
State lives in `gs://precursor-intel-mlops-tfstate` (prefix `prod`) and is locked during a run.
State and `*.tfvars` are git-ignored; commit `.terraform.lock.hcl`.
