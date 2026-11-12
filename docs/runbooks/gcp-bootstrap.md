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
| `sa-tf-apply` | service account | Terraform `apply`. Project `editor`, `resourcemanager.projectIamAdmin` and `iam.securityAdmin` (step 6), `objectAdmin` on the state bucket. Usable only by workflow runs on `dev` or `main` |
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
   R=repo:saianish03@89864897/Precursor-Intelligence@1402453226
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
6. **Resource-level IAM for `sa-tf-apply`** (2026-11-04, before the PR that adopts it through `imports.tf`)
   ```bash
   gcloud projects add-iam-policy-binding $PROJECT --member=serviceAccount:sa-tf-apply@$PROJECT.iam.gserviceaccount.com --role=roles/iam.securityAdmin --condition=None
   ```
   Granted by hand because the same apply then uses it, and a role created inside that apply would not have
   propagated yet. Run it before the first `plan` of that PR: the import fails while the binding is missing.

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

### The token subject contains numeric ids

GitHub puts the owner and repository ids in the token subject:
`repo:saianish03@89864897/Precursor-Intelligence@1402453226:pull_request`. A binding written with the plain name
(`repo:saianish03/Precursor-Intelligence`) never matches and the login fails with
`iam.serviceAccounts.getAccessToken denied`. The ids are permanent, so a renamed or re-created repository cannot
take over the login. The workflow prints the claims it presents (step "Show the login claims"); read the real
`sub` there if this ever changes, and update `github_subject` in `envs/prod/variables.tf`. The provider
condition still uses the plain `repository` claim.

If the first version of step 4 (plain name) was run, replace those three bindings:
```bash
R_OLD=repo:saianish03/Precursor-Intelligence
R=repo:saianish03@89864897/Precursor-Intelligence@1402453226
P=principal://iam.googleapis.com/projects/$PROJECT_NUMBER/locations/global/workloadIdentityPools/github/subject
gcloud iam service-accounts add-iam-policy-binding sa-tf-plan@$PROJECT.iam.gserviceaccount.com --role=roles/iam.workloadIdentityUser --member="${P}/${R}:pull_request"
gcloud iam service-accounts add-iam-policy-binding sa-tf-apply@$PROJECT.iam.gserviceaccount.com --role=roles/iam.workloadIdentityUser --member="${P}/${R}:ref:refs/heads/dev"
gcloud iam service-accounts add-iam-policy-binding sa-tf-apply@$PROJECT.iam.gserviceaccount.com --role=roles/iam.workloadIdentityUser --member="${P}/${R}:ref:refs/heads/main"
gcloud iam service-accounts remove-iam-policy-binding sa-tf-plan@$PROJECT.iam.gserviceaccount.com --role=roles/iam.workloadIdentityUser --member="${P}/${R_OLD}:pull_request"
gcloud iam service-accounts remove-iam-policy-binding sa-tf-apply@$PROJECT.iam.gserviceaccount.com --role=roles/iam.workloadIdentityUser --member="${P}/${R_OLD}:ref:refs/heads/dev"
gcloud iam service-accounts remove-iam-policy-binding sa-tf-apply@$PROJECT.iam.gserviceaccount.com --role=roles/iam.workloadIdentityUser --member="${P}/${R_OLD}:ref:refs/heads/main"
```

## Checks

```bash
gcloud storage buckets describe gs://$PROJECT-data --format="yaml(lifecycle_config,versioning_enabled,public_access_prevention)"
gcloud iam service-accounts get-iam-policy sa-tf-apply@$PROJECT.iam.gserviceaccount.com --format="yaml(bindings)"
gcloud projects get-iam-policy $PROJECT --flatten="bindings[].members" --filter="bindings.members:user:" --format="table(bindings.members,bindings.role)"
```
The `sa-tf-apply` policy must list exactly two members, ending in `:ref:refs/heads/dev` and `:ref:refs/heads/main`, each
with the numeric ids in the repository part. Its project roles must be exactly three:
```bash
gcloud projects get-iam-policy $PROJECT --flatten="bindings[].members" --filter="bindings.members:sa-tf-apply@" --format="value(bindings.role)"
```
`roles/editor`, `roles/iam.securityAdmin` and `roles/resourcemanager.projectIamAdmin`.

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
  `editor`, `projectIamAdmin` and `securityAdmin`, so this protection is the only gate between a pull request and
  the cloud.
- **Why `sa-tf-apply` holds `iam.securityAdmin`.** `editor` has no `setIamPolicy` permission at all, and
  `projectIamAdmin` only the project's own, so apply could create a secret, the image repository or a bucket but
  not grant anyone access to it (`403 … setIamPolicy`, and `plan` cannot show this in advance). `securityAdmin`
  adds the `setIamPolicy` of single resources without any data access: it cannot read a secret value or an object.
  It adds no real power either, because `projectIamAdmin` could already grant the account any role. The four
  service admin roles (`secretmanager.admin` and others) were rejected: `secretmanager.admin` can read every secret
  value, which today only the Owners can (Editors and the CI accounts cannot).
  Do not mark the Terraform `plan` job itself as a required check: it only runs when `infra/` changes, so a
  required check would stay pending forever on other pull requests.
- **Do not add a GitHub `environment:` to the apply job.** It changes the token subject to
  `...:environment:<name>`, which no binding matches, and CI could no longer log in.
- **Known risk, accepted for now:** the teammates are project Editors. Editor can run resources as any service
  account in the project, so a teammate could attach `sa-tf-apply` to something they create and use its
  permissions without going through a pull request. Narrowing teammates to specific roles removes this; revisit
  it after the demo.
- Third-party actions in `terraform.yml` and `ci.yml` are pinned to full commit SHAs, with the exact version kept
  as a comment, because `terraform.yml` can act as `sa-tf-apply`. `id-token: write` is granted per job, not for
  the whole workflow.
- Dependabot (`.github/dependabot.yml`) proposes the SHA bumps against `dev`, never `main`. Review each one by
  hand and never auto-merge it: the first grouped PR will include major upgrades, and a bump of `terraform.yml`
  runs `apply` as `sa-tf-apply` once it reaches `dev` or `main`.

## People's access to the VM (by hand, once per teammate)

Emails stay out of the repository, so these two grants are made by an Owner, like step 5. Editors already have
OS Login and can start and stop VMs; they lack the IAP tunnel and the right to act as `sa-pipeline`.
```bash
gcloud projects add-iam-policy-binding $PROJECT --member=user:<email> --role=roles/iap.tunnelResourceAccessor
gcloud iam service-accounts add-iam-policy-binding sa-pipeline@$PROJECT.iam.gserviceaccount.com --member=user:<email> --role=roles/iam.serviceAccountTokenCreator
```
The first is SSH through IAP (the VM has no external IP). The second lets a laptop test with the VM's exact
permissions instead of a personal login. Check it without printing the token:
```bash
gcloud auth print-access-token --impersonate-service-account=sa-pipeline@$PROJECT.iam.gserviceaccount.com >/dev/null && echo ok
```

## Secret values (by hand, after Terraform created the secrets)

Terraform creates the VM secrets empty, so no value ever passes through Terraform, its state or CI. A teammate
adds each value once. Editors can add a version but not read one back, which is intended.
- Generate fresh values; never reuse a laptop `.env`.
- Hex for the three database passwords: they go into connection URLs, where `/`, `+` and `=` break.
- No trailing newline: pipe with `printf %s`.
```bash
add() { printf %s "$2" | gcloud secrets versions add "$1" --data-file=- --project=$PROJECT; }
add airflow-db-password "$(openssl rand -hex 24)"
add pi-meta-db-password "$(openssl rand -hex 24)"
add mlflow-db-password "$(openssl rand -hex 32)"
add airflow-fernet-key "$(python3 -c 'import base64,os;print(base64.urlsafe_b64encode(os.urandom(32)).decode())')"
add airflow-jwt-secret "$(openssl rand -hex 32)"
add airflow-api-secret-key "$(openssl rand -hex 32)"
add smtp-password "<the alert sender's Gmail app password>"
add alert-smtp-user "<the alert sender's Gmail address>"
add alert-email-to "<recipient emails, comma-separated, no spaces>"
```
The last three come from the alert Gmail account: `smtp-password` (its app password, pasted without the spaces
Google shows) is added by the person who owns it, `alert-smtp-user` is its address, and `alert-email-to` holds
the team's recipient emails. The two addresses aren't secret, but email addresses stay out of this
public repository, so they live here too. Check that each secret has one enabled version, without reading it:
`gcloud secrets versions list <name> --format="table(name,state)"`.

**Never change `airflow-fernet-key` once the VM has run:** it decrypts the connections Airflow has stored. Rotate
the others by adding a new version; the VM picks it up at its next deploy.

## Sizing note

The full-archive bronze run (427 files, 61 vintages) peaked at about 2.2 GiB of memory, but gold is larger: it
holds the whole dataset in memory, estimated at about 5.8 GB on the full archive (`AIRFLOW_PIPELINE.md` §9.1),
and every monthly run rebuilds all of gold. The Airflow VM is therefore an e2-standard-4 (16 GB), decided
2026-10-29 (spec §14 #26); the machine type is a Terraform variable.

## Cost and cleanup

A stopped VM costs only its disk. Delete throwaway resources when done (`gcloud compute instances delete`).
The budget sends alerts but never stops spending.
