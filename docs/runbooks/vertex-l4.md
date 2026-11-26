# Runbook: Vertex AI access and the L4 proof

How to get gpt-oss on Vertex AI working for the agents, and how to rerun the laptop proof that a LangChain run with
a tool call and a structured answer works on it (`AGENTS_PLAN.md` L4). The proof itself is
`scripts/agents_vertex_proof.py` (#104), and its results (2026-11-22/23) are recorded under L4 in `AGENTS_PLAN.md`.
This runbook covers the steps around it: who enables what, and how a teammate signs in from a laptop.

Project `precursor-intel-mlops`, region `us-central1`. Nothing here is Terraform: APIs are enabled by hand
(`gcp-bootstrap.md` step 1), and enabling a model in Model Garden has no Terraform resource.

## 0. At a glance
| | |
|---|---|
| Model | `gpt-oss-120b` as a managed API (MaaS): no endpoint to deploy, pay per token. The project gets a 404 on `gpt-oss-20b` (us-central1 and global, 2026-11-22), so there is no fallback model; set `PI_AGENTS_FALLBACK_MODEL` once access exists |
| Model id in requests | `openai/gpt-oss-120b-maas` (the model card shows it without the `openai/` prefix; requests use the prefix). Vertex reports the model name exactly as configured |
| Region | `us-central1`, where #104's runs passed. Set in `configs/agents_runtime.yaml` (`PI_AGENTS_REGION`) |
| Endpoint (OpenAI-compatible) | `https://us-central1-aiplatform.googleapis.com/v1/projects/$PROJECT/locations/us-central1/endpoints/openapi`, built from `configs/agents_runtime.yaml` |
| Structured output | `json_schema` is the default (`model.structured_output_method`); `function_calling` also works but often spends the run's retry. Tool choice is `auto`, which the runtime already sends: gpt-oss on Vertex refuses a forced tool call |
| Auth | a Google OAuth token as the OpenAI client's API key; it lasts one hour, so `agents/model.py` refreshes it per request (A2). No API keys |
| Roles | to call: `roles/aiplatform.user` (Editors already have its permissions). To enable a model: `roles/consumerprocurement.entitlementManager` (Owners have it) |
| Cost | about $0.0003 per tools-agent run in L4's runs; prices and budgets are in `AGENTS_PLAN.md` §7 |

## 1. Checklist

| # | Step | Who |
|---|---|---|
| 1 | Enable the APIs | project owner |
| 2 | Enable gpt-oss-120b in Model Garden (never Deploy) | project owner |
| 3 | Check your own roles | teammate running the proof |
| 4 | Install gcloud on Windows | teammate |
| 5 | Sign in for Application Default Credentials (ADC) and set the quota project | teammate |
| 6 | Run the proof | teammate |
| 7 | Later: the same proof from Cloud Run (waits on L1–L3 and L7) | Aakash |

### 1. Enable the APIs (project owner)
```bash
gcloud services enable aiplatform.googleapis.com run.googleapis.com --project=precursor-intel-mlops
```
`aiplatform` is already on; `run` is needed for the API service (L3). This is the same line as
`gcp-bootstrap.md` step 1. Only if Model Garden asks for it in step 2, also enable
`cloudcommerceconsumerprocurement.googleapis.com`; #104's runs worked without it.

### 2. Enable the model in Model Garden (project owner)
In the console: Vertex AI, Model Garden, search "gpt-oss", open **gpt-oss 120B**, click **Enable**, and accept the
terms if a form appears. Enabling costs nothing; you pay only per token.

**Never click Deploy.** That self-deploys the model on a GPU endpoint, which bills by the hour.

The Owner does this because enabling needs `roles/consumerprocurement.entitlementManager`, which Owners have and
Editors don't. If the card shows the managed API as ready with no **Enable** button, there is nothing to do.

### 3. Check your roles
The teammates are project Editors, which includes `aiplatform.endpoints.predict`, so a laptop call works without a
new grant. Later, `sa-api` gets `roles/aiplatform.user` through Terraform (L1), and `sa-llm-eval` the same (J3). If
someone without Editor needs to call Vertex, the Owner grants:
```bash
gcloud projects add-iam-policy-binding precursor-intel-mlops --member=user:<email> --role=roles/aiplatform.user
```

### 4. Install gcloud on Windows
In PowerShell:
```powershell
winget install --id Google.CloudSDK -e
```
Open a new terminal afterwards so `gcloud` is on the `PATH`, then `gcloud --version`. If winget isn't available,
use Google's installer: https://dl.google.com/dl/cloudsdk/channels/rapid/GoogleCloudSDKInstaller.exe. Then:
```powershell
gcloud init
```
Sign in with your university Google account and pick `precursor-intel-mlops`.

### 5. ADC sign-in and quota project
`gcloud init` signs in the CLI; the Python libraries use ADC, which is a separate sign-in:
```powershell
gcloud auth application-default login
gcloud auth application-default set-quota-project precursor-intel-mlops
```
The second line bills the calls and their quota to our project. It needs `serviceusage.services.use` on the
project, which Editors have. Without it, a user login can fail with a 403 that names a quota project.

### 6. Run the proof
From the repo root. It makes real, billed calls (about a cent per method), so run it by hand, never in CI:
```powershell
uv run --frozen --extra agents python scripts/agents_vertex_proof.py                                   # both methods, both agent shapes
uv run --frozen --extra agents python scripts/agents_vertex_proof.py --method json_schema --agent gather
```
- `--method {function_calling,json_schema}` tries one `structured_output_method` instead of both.
- `--agent {tools,gather}` runs one agent shape: `tools` is shaped like Agent 1 (a tool call, then an answer citing
  its evidence ID); `gather` like Agents 2-4 (the evidence in the prompt, one structured call).
- `--only {primary,fallback}` calls one model alone; `fallback` needs `PI_AGENTS_FALLBACK_MODEL` set.

The project, region and model ids come from `configs/agents_runtime.yaml` and its environment variables
(`PI_GCP_PROJECT`, `PI_AGENTS_REGION`, `PI_AGENTS_MODEL`, `PI_AGENTS_FALLBACK_MODEL`).

**Success looks like:** for each method it prints the model name Vertex reports, the checks (tool call: yes; cited
the evidence ID: yes), and a final `works:` table; it exits 0 when at least one method works for every agent shape.
If a rerun changes which method works, update `model.structured_output_method` and the L4 entry in `AGENTS_PLAN.md`.

### 7. Then from Cloud Run
L4 is done only when the same proof passes from Cloud Run as `sa-api`. That waits on L1 (`sa-api` with
`roles/aiplatform.user`), L2, L3 (the service) and L7 (the image). Run it as part of L9's first smoke test.

## 2. Troubleshooting

| Error | Likely cause | Fix |
|---|---|---|
| 403 `SERVICE_DISABLED` or "API has not been used in project" | `aiplatform` not enabled | step 1; wait a minute after enabling |
| 403 `PERMISSION_DENIED` on `aiplatform.endpoints.predict` | the caller lacks the role | step 3; on Cloud Run, check that `sa-api` has `roles/aiplatform.user` |
| 403 naming a quota project, or "user credentials not supported" | ADC has no quota project | step 5, second line |
| 403 or 400 saying the model isn't enabled, or terms not accepted | Model Garden step not done | step 2; if Model Garden asks for Consumer Procurement, enable that API (step 1) |
| 404 model not found | wrong region or id, or no access to that model (20b gets this today) | check `PI_AGENTS_REGION`, the `openai/` prefix and the `-maas` suffix |
| 429 `RESOURCE_EXHAUSTED` | per-minute quota hit | wait and retry; lower L11's concurrency; ask for more in IAM & Admin, Quotas |
| 400 on `tool_choice` | something sent `required` or a named tool; gpt-oss on Vertex accepts only `auto` | the runtime sends `auto`; check any new code path that binds tools |
| 401 after about an hour | the token expired | the factory refreshes per request (A2); a script that caches `credentials.token` breaks |

## Sources
- [OpenAI models on Vertex AI](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/maas/openai) and [the gpt-oss 120B card](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/maas/openai/gpt-oss-120b)
- [Call MaaS APIs for open models](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/maas/call-open-model-apis): **Enable** on the model card, the roles, the endpoint
- [Grant user access to open models](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/maas/grant-access-open-models): `consumerprocurement.entitlementManager` and `aiplatform.user`
- [Function calling](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/maas/capabilities/function-calling) (gpt-oss: no `required`, no named tool)
- [Using OpenAI libraries with Vertex AI](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/start/openai): the base URL and token auth
- [Install the gcloud CLI](https://docs.cloud.google.com/sdk/docs/install-sdk) and [set-quota-project](https://docs.cloud.google.com/sdk/gcloud/reference/auth/application-default/set-quota-project)
