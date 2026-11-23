# Runbook: Vertex AI access and the day-one proof (L4)

How to get gpt-oss on Vertex AI working for the agents, and how to prove from a laptop that a Pydantic AI run
with a tool call and structured output works on it (`AGENTS_PLAN.md` L4). This is the riskiest unknown in the
plan, so it runs first. The facts below were checked on 2026-11-22 against Google's docs (sources at the end);
confirm the ones marked "check" in the console, because the docs disagree with each other in places.

Project `precursor-intel-mlops`, region `us-central1`. Nothing here is Terraform: APIs are enabled by hand
(`gcp-bootstrap.md` step 1), and enabling a model in Model Garden has no Terraform resource.

## 0. At a glance
| | |
|---|---|
| Models | `gpt-oss-120b` (the proof, golden sets, demo) and `gpt-oss-20b` (development runs), as managed APIs (MaaS): no endpoint to deploy, pay per token |
| Model id in requests | `openai/gpt-oss-120b-maas`, `openai/gpt-oss-20b-maas` (the model card shows the id without the `openai/` prefix; requests use the prefix) |
| Region | `global` (the script's default; Google's current model card lists only `global`); if that fails, try `us-central1` |
| Endpoint (OpenAI-compatible) | `https://aiplatform.googleapis.com/v1/projects/$PROJECT/locations/global/endpoints/openapi`, or for a region: `https://us-central1-aiplatform.googleapis.com/v1/projects/$PROJECT/locations/us-central1/endpoints/openapi` |
| Auth | a Google OAuth token as the OpenAI client's API key; it lasts one hour, so the code refreshes it (A2). No API keys |
| Roles | to call: `roles/aiplatform.user` (Editors already have its permissions). To enable a model: `roles/consumerprocurement.entitlementManager` (Owners have it) |
| Price | 120b: $0.09 in / $0.36 out per million tokens (third-party listings, Oct 2026) or $0.15 / $0.60 (Google, at launch). 20b at launch: $0.075 / $0.30. Check the pricing page |
| Default quota | at launch, in us-central1: 120b 650 requests a minute, 790k input and 120k output tokens a minute; 20b 1,200 requests a minute. Check in the console |

## 1. Checklist

| # | Step | Who |
|---|---|---|
| 1 | Enable the three APIs | project owner |
| 2 | Enable gpt-oss-120b and gpt-oss-20b in Model Garden | project owner |
| 3 | Check your own roles | Aakash |
| 4 | Install gcloud on Windows | Aakash |
| 5 | Sign in for Application Default Credentials (ADC) and set the quota project | Aakash |
| 6 | Confirm the region and the model id | Aakash |
| 7 | Run the smoke test, `--dry-run` first | Aakash |
| 8 | Record the numbers in `AGENTS_PLAN.md` §6 and §7 | Aakash |
| 9 | Later: the same proof from Cloud Run (waits on L1–L3 and L7) | Aakash |

### 1. Enable the APIs (project owner)
```bash
gcloud services enable aiplatform.googleapis.com run.googleapis.com cloudcommerceconsumerprocurement.googleapis.com --project=precursor-intel-mlops
```
`cloudcommerceconsumerprocurement` is what Model Garden uses to enable an open model; without it step 2 fails.
This is the same line as `gcp-bootstrap.md` step 1.

### 2. Enable the models in Model Garden (project owner)
In the console: Vertex AI, Model Garden, search "gpt-oss", open **gpt-oss 120B**, click **Enable**, and
accept the terms if a form appears. Repeat for **gpt-oss 20B**. Enabling costs nothing; you pay only per token.
Do not click **Deploy**: that self-deploys the model on a GPU endpoint, which bills by the hour.

The Owner does this because enabling needs `roles/consumerprocurement.entitlementManager`, which Owners have
and Editors don't. If the console shows the managed API as ready with no **Enable** button, there is nothing to
do: note that in the PR.

### 3. Check your roles (Aakash)
The teammates are project Editors, which includes `aiplatform.endpoints.predict`, so a laptop call works without
a new grant. Later, `sa-api` gets `roles/aiplatform.user` through Terraform (L1), and `sa-llm-eval` the same (J3).
If someone without Editor needs to call Vertex, the Owner grants:
```bash
gcloud projects add-iam-policy-binding precursor-intel-mlops --member=user:<email> --role=roles/aiplatform.user
```

### 4. Install gcloud on Windows (Aakash)
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

### 5. ADC sign-in and quota project (Aakash)
`gcloud init` signs in the CLI; the Python libraries use ADC, which is a separate sign-in:
```powershell
gcloud auth application-default login
gcloud auth application-default set-quota-project precursor-intel-mlops
```
The second line bills the calls and their quota to our project. It needs `serviceusage.services.use` on the
project, which Editors have. Without it, a user login can fail with a 403 that names a quota project.

### 6. Confirm the region and the model id (Aakash)
- On the model card in Model Garden, read the model id (`gpt-oss-120b-maas`) and the regions listed.
- Use `global` (the default). Google's current card lists `global` (with processing in
  the `us` multi-region); the launch announcement listed `us-central1`, which may still work. The smoke test settles it: a 404 means
  the model isn't served in that region.
- In the console, IAM & Admin, Quotas: filter by "gpt-oss" to read the real per-minute limits.

### 7. Run the smoke test (Aakash)
In PowerShell, from the repo root:
```powershell
$PROJECT = "precursor-intel-mlops"
uv run --extra agents python scripts/vertex_smoke.py --project $PROJECT --model openai/gpt-oss-120b-maas --dry-run
uv run --extra agents python scripts/vertex_smoke.py --project $PROJECT --model openai/gpt-oss-120b-maas
```
(In Git Bash, `export PROJECT=precursor-intel-mlops` instead of the first line.)

`--dry-run` builds the agent and prints the endpoint and model without calling Vertex, so it costs nothing. Fix
anything it reports before the real run. The real run makes a few calls and costs well under a cent.

Use 120b for this proof: a pass on 20b says nothing about 120b (§7). Run 20b once afterwards, because it is
the fallback (A2) and the development model:
```powershell
uv run --extra agents python scripts/vertex_smoke.py --project $PROJECT --model openai/gpt-oss-20b-maas
```

**Success looks like:** the run calls the tool at least once, returns an output that validates against the
Pydantic model, and exits 0, printing the model, the region, the time taken and the token usage.

### 8. Record the results (Aakash)
Put these in the PR and in `AGENTS_PLAN.md`:
- §6 decision 2: the region and the exact model ids that worked, and who enabled the models and when.
- §7: the measured latency, input and output tokens per call, and the cost per run at the price on Google's
  pricing page (replace the "third-party listings" line with the real price).
- The quota numbers from the console, for L11 (5 briefs at a time must stay under them).
- After the first day of calls: the billing report shows the gpt-oss charges and whether the credits cover
  them (§7, "Credits").

### 9. Then from Cloud Run
L4 is done only when the same proof passes from Cloud Run as `sa-api`. That waits on L1 (`sa-api` with
`roles/aiplatform.user`), L2, L3 (the service) and L7 (the image). Run it as part of L9's first smoke test.

## 2. Troubleshooting

| Error | Likely cause | Fix |
|---|---|---|
| 403 `SERVICE_DISABLED` or "API has not been used in project" | `aiplatform` not enabled | step 1; wait a minute after enabling |
| 403 `PERMISSION_DENIED` on `aiplatform.endpoints.predict` | the caller lacks the role | step 3; on Cloud Run, check that `sa-api` has `roles/aiplatform.user` |
| 403 naming a quota project, or "user credentials not supported" | ADC has no quota project | step 5, second line |
| 403 or 400 saying the model isn't enabled, or terms not accepted | Model Garden step not done | step 2 |
| 404 model not found | wrong region or wrong id | try `--region us-central1`; check the `openai/` prefix and the `-maas` suffix |
| 429 `RESOURCE_EXHAUSTED` | per-minute quota hit | wait and retry with backoff; lower L11's concurrency; ask for more in IAM & Admin, Quotas (edit the quota, give a reason) |
| 400 on `tool_choice` | gpt-oss on Vertex supports only `tool_choice="auto"`: not `required`, not a named tool. Pydantic AI's default tool output sends `required` | in the model profile set `openai_supports_tool_choice_required=False`, use `OpenAIChatModel` (not the Responses model), or use `NativeOutput`/`PromptedOutput` |
| 400 on `response_format` or `json_schema`, or `strict` in a tool | Vertex documents JSON mode (`json_object`) for open models; full JSON-schema enforcement isn't confirmed | use tool output with `auto`, or `PromptedOutput`, and let the validator retry; set `openai_supports_strict_tool_definition=False` if `strict` is rejected |
| 401 after about an hour | the token expired | the factory must refresh per request (A2); a script that caches `credentials.token` breaks |

## Sources
- [OpenAI models on Vertex AI](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/maas/openai) and [the gpt-oss 120B card](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/maas/openai/gpt-oss-120b): model id, `global` region, function calling and structured output
- [Call MaaS APIs for open models](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/maas/call-open-model-apis): the APIs, **Enable** on the model card, the roles, the endpoint, `openai/gpt-oss-120b-maas`
- [Grant user access to open models](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/maas/grant-access-open-models): `consumerprocurement.entitlementManager` and `aiplatform.user`
- [Function calling](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/maas/capabilities/function-calling) (gpt-oss: no `required`, no named tool) and [structured output](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/maas/capabilities/structured-output)
- [Using OpenAI libraries with Vertex AI](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/start/openai): the base URL and token auth
- [Launch announcement, Aug 2025](https://discuss.google.dev/t/now-ga-openais-gpt-oss-qwen3-models-on-vertex-ai-as-open-model-apis/253945): launch regions, quotas and prices
- [Vertex AI pricing](https://cloud.google.com/vertex-ai/generative-ai/pricing) and [Vertex AI quotas](https://docs.cloud.google.com/vertex-ai/docs/quotas)
- [Install the gcloud CLI](https://docs.cloud.google.com/sdk/docs/install-sdk) and [set-quota-project](https://docs.cloud.google.com/sdk/gcloud/reference/auth/application-default/set-quota-project)
- [Pydantic AI model profiles](https://ai.pydantic.dev/api/profiles) and [issue #3270](https://github.com/pydantic/pydantic-ai/issues/3270) (`tool_choice=required` on gpt-oss)
