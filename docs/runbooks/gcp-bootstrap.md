# GCP bootstrap (one time, by hand)

What had to exist before Terraform and CI could manage the project, and exactly what was created on
2026-10-16. Everything after this is Terraform, applied by CI (`infra/terraform/`). Do not change these
resources in the console: change the Terraform and open a pull request.

Project `precursor-intel-mlops` (number `852874974077`), region `us-central1`, owned by the billing account
that holds the $300 trial credit. The project has no organisation, so Owner can only be added by invitation.

## What exists

| Resource | Name | Purpose |
|---|---|---|
| Data bucket | `precursor-intel-mlops-data` | The one object-store root: landing, bronze, silver, gold, reports, metadata manifests. Versioned, no retention policy, public access blocked, `quarantine/` objects deleted after 30 days |
| State bucket | `precursor-intel-mlops-tfstate` | Terraform state. Versioned, public access blocked |
| `sa-pipeline` | service account | What the pipeline VM runs as. `storage.objectAdmin` and `storage.legacyBucketReader` on the data bucket only. The reader role is needed because gcsfs checks the bucket itself (`storage.buckets.get`); without it every write fails with "bucket does not exist" |
| `sa-tf-plan` | service account | Terraform `plan` on pull requests. Project `viewer` and `iam.securityReviewer` (plan must read IAM policies), and only `objectViewer` on the state bucket. Usable only by the `pull_request` workflow identity of the repo |
| `sa-tf-apply` | service account | Terraform `apply`. Project `editor` and `resourcemanager.projectIamAdmin`, `objectAdmin` on the state bucket. Usable only by workflow runs on `dev` or `main` |
| Workload identity pool `github`, provider `github-repo` | keyless CI login | Accepts only tokens from the repository `saianish03/Precursor-Intelligence` |
| Budget `pi-trial-credit` | $300 | Alerts at about $25, $50, $100 and $150. It only sends emails and does not stop spending |

Team access: the four teammates are `roles/editor`; a second Owner was invited (an invitee must accept by
email). Emails are kept out of the repository.

## Commands, in the order they were run

Run in a terminal on a laptop with `gcloud` installed and signed in as the project owner. Variables:
```bash
export PROJECT=precursor-intel-mlops PROJECT_NUMBER=852874974077 REGION=us-central1
export BILLING=<billing account id>
```
Do not put a `#` comment after a command in zsh (it is not a comment on the command line, and `$25` expands to
nothing), and always write `${P}/${R}:ref:...` with braces: in zsh `$R:ref` is a filename modifier and silently
drops the colon.

1. **APIs and budget**
   ```bash
   gcloud services enable compute.googleapis.com storage.googleapis.com iam.googleapis.com iamcredentials.googleapis.com sts.googleapis.com cloudresourcemanager.googleapis.com iap.googleapis.com bigquery.googleapis.com artifactregistry.googleapis.com secretmanager.googleapis.com billingbudgets.googleapis.com
   gcloud billing budgets create --billing-account=$BILLING --display-name=pi-trial-credit --budget-amount=300USD --threshold-rule=percent=0.0833 --threshold-rule=percent=0.1667 --threshold-rule=percent=0.3333 --threshold-rule=percent=0.5
   ```
2. **Buckets**
   ```bash
   gcloud storage buckets create gs://$PROJECT-tfstate --location=$REGION --uniform-bucket-level-access --public-access-prevention
   gcloud storage buckets update gs://$PROJECT-tfstate --versioning
   gcloud storage buckets create gs://$PROJECT-data --location=$REGION --uniform-bucket-level-access --public-access-prevention
   gcloud storage buckets update gs://$PROJECT-data --versioning
   echo '{"rule":[{"action":{"type":"Delete"},"condition":{"age":30,"matchesPrefix":["quarantine/"]}}]}' > /tmp/lifecycle.json
   gcloud storage buckets update gs://$PROJECT-data --lifecycle-file=/tmp/lifecycle.json
   ```
3. **Pipeline service account**
   ```bash
   gcloud iam service-accounts create sa-pipeline --display-name="Pipeline runtime (VM)"
   gcloud storage buckets add-iam-policy-binding gs://$PROJECT-data --member=serviceAccount:sa-pipeline@$PROJECT.iam.gserviceaccount.com --role=roles/storage.objectAdmin
   gcloud storage buckets add-iam-policy-binding gs://$PROJECT-data --member=serviceAccount:sa-pipeline@$PROJECT.iam.gserviceaccount.com --role=roles/storage.legacyBucketReader
   ```
   A binding made seconds after creating the account can fail with "does not exist"; wait a minute and repeat.
4. **Keyless CI**
   ```bash
   gcloud iam workload-identity-pools create github --location=global --display-name="GitHub Actions"
   gcloud iam workload-identity-pools providers create-oidc github-repo --location=global --workload-identity-pool=github --issuer-uri=https://token.actions.githubusercontent.com --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository" --attribute-condition="assertion.repository=='saianish03/Precursor-Intelligence'"
   gcloud iam service-accounts create sa-tf-plan --display-name="Terraform plan (PRs, read-only)"
   gcloud iam service-accounts create sa-tf-apply --display-name="Terraform apply (merged branches only)"
   P=principal://iam.googleapis.com/projects/$PROJECT_NUMBER/locations/global/workloadIdentityPools/github/subject
   R=repo:saianish03/Precursor-Intelligence
   gcloud iam service-accounts add-iam-policy-binding sa-tf-plan@$PROJECT.iam.gserviceaccount.com --role=roles/iam.workloadIdentityUser --member="${P}/${R}:pull_request"
   gcloud iam service-accounts add-iam-policy-binding sa-tf-apply@$PROJECT.iam.gserviceaccount.com --role=roles/iam.workloadIdentityUser --member="${P}/${R}:ref:refs/heads/dev"
   gcloud iam service-accounts add-iam-policy-binding sa-tf-apply@$PROJECT.iam.gserviceaccount.com --role=roles/iam.workloadIdentityUser --member="${P}/${R}:ref:refs/heads/main"
   gcloud projects add-iam-policy-binding $PROJECT --member=serviceAccount:sa-tf-plan@$PROJECT.iam.gserviceaccount.com --role=roles/viewer
   gcloud projects add-iam-policy-binding $PROJECT --member=serviceAccount:sa-tf-plan@$PROJECT.iam.gserviceaccount.com --role=roles/iam.securityReviewer
   gcloud projects add-iam-policy-binding $PROJECT --member=serviceAccount:sa-tf-apply@$PROJECT.iam.gserviceaccount.com --role=roles/editor
   gcloud projects add-iam-policy-binding $PROJECT --member=serviceAccount:sa-tf-apply@$PROJECT.iam.gserviceaccount.com --role=roles/resourcemanager.projectIamAdmin
   gcloud storage buckets add-iam-policy-binding gs://$PROJECT-tfstate --member=serviceAccount:sa-tf-plan@$PROJECT.iam.gserviceaccount.com --role=roles/storage.objectViewer
   gcloud storage buckets add-iam-policy-binding gs://$PROJECT-tfstate --member=serviceAccount:sa-tf-apply@$PROJECT.iam.gserviceaccount.com --role=roles/storage.objectAdmin
   ```
5. **Team access** (Editor for teammates; Owner needs an invitation: console, IAM, Grant access)
   ```bash
   gcloud projects add-iam-policy-binding $PROJECT --member=user:<email> --role=roles/editor
   ```

### Two corrections to step 4, after the first review

On 2026-10-16 the first version of step 4 gave `sa-tf-plan` `objectAdmin` on the state bucket and no
`iam.securityReviewer`. The commands above are the corrected ones. If you ran the first version, fix the live
project once:
```bash
gcloud projects add-iam-policy-binding $PROJECT --member=serviceAccount:sa-tf-plan@$PROJECT.iam.gserviceaccount.com --role=roles/iam.securityReviewer
gcloud storage buckets add-iam-policy-binding gs://$PROJECT-tfstate --member=serviceAccount:sa-tf-plan@$PROJECT.iam.gserviceaccount.com --role=roles/storage.objectViewer
gcloud storage buckets remove-iam-policy-binding gs://$PROJECT-tfstate --member=serviceAccount:sa-tf-plan@$PROJECT.iam.gserviceaccount.com --role=roles/storage.objectAdmin
```
Why: a pull request runs its own code as `sa-tf-plan`. With write access to the state it could delete or replace
the state that the next apply trusts, and plain `viewer` cannot read IAM policies, so `plan` would fail on the
first refresh. The plan job therefore reads the state only and runs with `-lock=false`.

## Checks

```bash
gcloud storage buckets describe gs://$PROJECT-data --format="yaml(lifecycle_config,versioning_enabled,public_access_prevention)"
gcloud iam service-accounts get-iam-policy sa-tf-apply@$PROJECT.iam.gserviceaccount.com --format="yaml(bindings)"
gcloud projects get-iam-policy $PROJECT --flatten="bindings[].members" --filter="bindings.members:user:" --format="table(bindings.members,bindings.role)"
```
The `sa-tf-apply` policy must list exactly two members, ending in `:ref:refs/heads/dev` and `:ref:refs/heads/main`.

## Why two Terraform accounts

Anyone with write access to the repository can edit a workflow file. A pull request therefore runs as
`sa-tf-plan`, which can only read. Only a merge into `dev` or `main` can act as `sa-tf-apply`, so branch
protection on those two branches is what guards infrastructure changes.

## First plan: what to expect

`terraform plan` in `envs/prod` should show the imports and nothing else: **18 to import, 0 to add, 0 to
destroy**. Two harmless in-place updates are normal and do not mean an import failed: `force_destroy` on each
bucket (an import does not set it; the code sets `false`), and possibly nothing else. Anything that says
"destroy" or "replace" is a real problem: stop and read it.

## Rules that keep infrastructure safe

- **`prevent_destroy`** is set on the buckets and on every resource CI itself depends on (the pool, provider,
  both Terraform accounts and their bindings). Removing one makes the plan fail instead of cutting the branch
  Terraform stands on. The apply job also refuses any plan that deletes or replaces something.
- **Protect `dev` and `main` on GitHub** (repository settings, by an admin): require a pull request with one
  approval, require the CI checks, and block direct pushes. A merge into either branch runs `apply` as
  `editor` plus `projectIamAdmin`, so this protection is the only gate between a pull request and the cloud.
  Do not mark the Terraform `plan` job itself as a required check: it only runs when `infra/` changes, so a
  required check would stay pending forever on other pull requests.
- **Do not add a GitHub `environment:` to the apply job.** It changes the token subject to
  `...:environment:<name>`, which no binding matches, and CI could no longer log in.
- **Known risk, accepted for now:** the teammates are project Editors. Editor can run resources as any service
  account in the project, so a teammate could attach `sa-tf-apply` to something they create and use its
  permissions without going through a pull request. Narrowing teammates to specific roles removes this; revisit
  it after the demo.
- Actions are pinned to major version tags, as in `ci.yml`. Pinning to commit SHAs is stricter and is a
  possible later hardening.

## Sizing note

The full-archive bronze run (427 files, 61 vintages) peaked at about 2.2 GiB of memory, so the planned
e2-standard-2 VM (8 GB) is enough for the backfill; the resize to e2-standard-4 in the tech-stack document is
not needed.

## Cost and cleanup

A stopped VM costs only its disk. Delete throwaway resources when done (`gcloud compute instances delete`).
The budget sends alerts but never stops spending.
