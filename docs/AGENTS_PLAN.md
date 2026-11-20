# Agents: detailed implementation plan and GCP production rollout

**Owners:** Tanoj, Riya, Aakash, plus a frontend owner (decision 0). Nishanth reviews the infrastructure. **Target:** the Dec 1 demo.

**What this covers:** the tasks in every part, how the agents and the UI run in production on our current GCP setup, and the order.

**Where the rules come from:** the agent rules (read-only tools with evidence IDs, the five validators, retry then fallback, versioned prompts, golden sets, one log line per call) are in each person's local setup (`.claude/`, which is gitignored), so they aren't in the repo. Sections B, C, D and J restate every rule a reviewer needs, so this file is the reference. The original project proposal isn't in the repo either; where this plan differs from it, raise it in the group.

**The UI:** we're not using Base44. The UI is our own React app (Vite, TypeScript, shadcn/ui), which also replaces the Streamlit placeholder in `frontend/README.md`. Its hosting and sign-in choice is the first ADR (M1).

**Ownership change from the first plan:** the production rollout (Terraform, images, deploys) is a full workstream, so Aakash takes it, and Portfolio Briefing (Agent 4) moves to Riya. Agent 4 reuses her tools and her chain roll-ups, so the move costs nothing.

| Person | Owns |
|---|---|
| **Tanoj** | the runtime, guardrails, observability, caching and cost control, the evaluation harness, Agent 3, the API's shared middleware |
| **Riya** | the evidence data layer and its build job, Agent 1, Agent 4, the API endpoints for both |
| **Aakash** | the retrieval corpus, Agent 2 and its endpoint, the GCP production rollout |
| **Frontend owner** (decision 0) | the React app: its views, states, sign-in, build and CI |

---

## 1. Where it runs in production

### What we have today
- **One VM** (`pi-airflow`, e2-highmem-4, us-central1-a) runs Airflow, Postgres and, from Phase 2, MLflow, under Docker Compose.
  - It **stops itself when idle**, about 70–85 minutes after the last activity.
  - MLflow has **no login** and is reachable only from the VM's task network.
- **GCS:** `precursor-intel-mlops-data` is meant to hold `scores/v=YYYY-MM/` and `status/latest.json` at its root, and `-mlflow` holds the package store.
  - **Today there are no prod scores.** The only real scores are the laptop stack's, under `ml-dev/scores/v=2026-10/` (14,686 facilities, v1) and `ml-dev/status/latest.json`.
  - Prod scores arrive with hand-off items 2-3 (MLflow on the VM, then ML on the VM). Until then, E, F, H and I develop against `ml-dev/` and the toy chain.
- **Artifact Registry** gets images that CD pushes on each merge to `dev` or `main` (`pipeline`, `airflow`, and `mlflow` from Phase 2). The Deploy workflow points the VM at a release sha.
- **GitHub Actions** authenticates through Workload Identity Federation (`sa-deployer`, `sa-release`); there are no JSON keys.
- **Secret Manager** holds the VM's secrets. A person adds each value by hand; values never pass through Terraform.
- **APIs are enabled by hand** (`docs/runbooks/gcp-bootstrap.md` step 1); Terraform has no `google_project_service`. `cloudtrace`, `bigquery` and `secretmanager` are on; **`aiplatform` and `run` are not yet.**
- **The budget exists:** `pi-trial-credit`, $300, alerts at about $25, $50, $100 and $150, created by hand.
- **Not built yet:** Cloud Run, the BigQuery dataset (Nishanth's Phase 2 item 4), and Vertex AI access.

### What follows from that
1. **The agents run on Cloud Run** as `precursor-api` (FastAPI plus the agents): min 0 and max 2 instances, in us-central1. A VM that stops itself can't serve users.
2. **The API never touches the VM.** It reads GCS, and BigQuery once that exists. It never reaches the VM's Postgres or MLflow; the API rules require this.
3. **Production traces go to Cloud Trace, not MLflow.** Cloud Run can't reach MLflow on the VM, and the VM is often off. The golden-set evaluation runs still log their traces to MLflow, so prompt versions can be compared there.
4. **Heavy data work happens in a batch step after `publish`,** not on each request. A new step builds small per-month evidence files that the API reads quickly. It runs on the VM, so it can also read Postgres (`ml_runs`) and the run folders, and it writes the ops evidence for Agent 3 too.
5. **The API finds the current month only through a pointer.** `build-evidence` writes `agents/evidence/_current.json` **last**: the month `v`, the scores manifest sha and the evidence manifest sha. The API never uses `status/latest.json` (written by `monitor`; its `v` can be a month with nothing scored) or "the newest `v=` folder" (`score` writes it before `publish` checks it), and it never lists a prefix.
6. **Everything shared lives in GCS:** the cache, the daily call counter and the evidence files. Cloud Run instances share no memory, and they scale to zero.
7. **One Cloud Run service serves the UI and the API.** The Docker build has a Node stage that builds the React app to static files, and FastAPI serves `dist/`. Same origin (no CORS), one image, one deploy, one rollback. Node is build tooling only; FastAPI is the only backend.
8. **Users sign in with Google.** The React app gets a Google ID token, and FastAPI verifies it against an email allowlist. A browser app can't keep an API key secret, so there is no API key.

```
 VM (Airflow, monthly)                       GCS                                    Cloud Run
 cms_nh_ml: ... score -> publish --asset-->  scores/v=YYYY-MM/ (read by build-evidence only)
 cms_nh_agents (new DAG):
   build-evidence (scores, silver, ml_runs,
     run folders, monitor) ----------------> agents/evidence/v=YYYY-MM/  ---->  precursor-api (FastAPI)
                                   last ---> agents/evidence/_current.json        |  React app (dist/)
   build-corpus ---------------------------> agents/corpus/<version>/             |  agents 1-4
                                                                                  |  guardrails, cache, cap
                                             agents-cache bucket  <--------------+  cache + daily counter
                                                                                  |
                         Vertex AI (gpt-oss-120b, OpenAI-compatible API)  <-------+  LLM calls
                         Cloud Trace  <-------------------------------------------+  OTel spans
                         Cloud Logging --sink--> BigQuery llm_calls  <------------+  one JSON line per call
 Browser --(Google sign-in, ID token)--> precursor-api
```

---

## 2. Tasks by part

Each task is one PR (or part of one) into `dev`. Agent and prompt PRs get the `agent-reviewer` subagent; infrastructure PRs get the `infra-reviewer` subagent and Nishanth.

### A. Runtime and settings (Tanoj), the foundation everything else imports

| ID | Task | Done when |
|---|---|---|
| A1 | `agents/settings.py`: model id, fallback model id, `max_tokens`, temperature, tool timeout, daily call cap, cache bucket, region, all from `configs/agents.yaml` plus environment variables; nothing hard-coded | settings load in tests; a typo is a config error |
| A2 | `agents/model.py`: a model factory. Production uses Vertex AI's OpenAI-compatible endpoint with Cloud Run's service account (no keys). The bearer token expires after about an hour, so it is **refreshed per request**: `google.auth` credentials with `refresh()` behind a small `httpx.Auth`, passed to the OpenAI client. Tests use Pydantic AI's `TestModel` and `FunctionModel`. If the primary model fails, it falls back to `gpt-oss-20b` | a unit test builds both; a test shows the factory never caches a token string; the Vertex path is proven in L4 |
| A3 | `agents/runner.py`: one `run_agent()` wrapper that does input guardrails, then the daily cap, cache lookup, agent run, validators, one retry, fallback, the log line, a trace span and the cache store | the whole path is tested with `TestModel`, including retry and fallback |
| A4 | `agents/prompts.py`: loads `configs/prompts/<agent>.yaml` (`version`, `system`, `instructions`) and refuses a prompt with no version | a missing version fails |
| A5 | Shared output models: `Evidence`, `Driver`, `Section`, and the `data_as_of` plus caveat fields, which every agent's output type extends | every agent output has `data_as_of`, the caveat and the evidence list |

### B. Guardrails, validators and fallbacks (Tanoj)

| ID | Task | Done when |
|---|---|---|
| B1 | **Input checks:** CCN format (6 characters), question length limit, allowed characters, and portfolio size limit | bad inputs return a clear 400 error before any LLM call |
| B2 | **Prompt-injection defense:** tool output reaches the model only as labelled data blocks; the system prompt says data never carries instructions; a test plants "ignore your instructions" in a record | the planted text is never followed, and the validators still pass |
| B3 | **Validator 1, evidence IDs:** every ID in the output was returned by a tool in this run (the runner keeps a per-run evidence ledger) | a made-up ID fails |
| B4 | **Validator 2, numbers:** every number in the output matches its source record within the documented rounding | a changed number fails |
| B5 | **Validator 3, driver direction:** "raises" and "lowers" match the SHAP sign | a flipped driver fails |
| B6 | **Validator 4, quote fidelity:** a quote must be an exact substring of the stored passage (for Agent 2) | an edited quote fails |
| B7 | **Validator 5, structure:** the required sections are present, within length limits, with `data_as_of` and the caveat | a missing section fails |
| B8 | **Language check:** banned phrases ("will be cited", "decline", "approve", "guarantee"); the band-to-action wording comes from a rule table | a banned phrase fails |
| B9 | **Fallback templates** per agent: a fixed, deterministic write-up from the tool data alone, marked `fallback=true` | when the model fails twice, the user still gets a correct, plain answer |

### C. Observability (Tanoj)

| ID | Task | Done when |
|---|---|---|
| C1 | OpenTelemetry setup: Pydantic AI's built-in instrumentation plus one span per tool call. The exporter goes to **Cloud Trace** in production and to **MLflow** in evaluation runs. Spans carry agent, prompt version, model and request id, never secrets | one request shows as one trace with the tool calls under it |
| C2 | **One JSON log line per LLM call:** agent, prompt version, model, input and output tokens, latency, cost estimate, cache hit, fallback, validator failures and request id | the line matches a schema test |
| C3 | Cost estimate per call from a price table in `configs/agents.yaml` (per million tokens, input and output) | the cost matches a hand calculation in a test |
| C4 | **Dashboards:** a Cloud Monitoring log-based metric for calls, errors, fallbacks and cost per day, and an alert when the fallback rate is over 5% or spend is over the daily budget | the alert fires on a test log line |

### D. Caching and cost control (Tanoj)

| ID | Task | Done when |
|---|---|---|
| D1 | **Answer cache** in a new GCS bucket `precursor-intel-mlops-agents-cache`. The key is sha256 of the agent, prompt version, model, normalized input, the **evidence manifest sha** from `_current.json`, and the **corpus version** (for Agent 2, and for Agent 1 because it calls Agent 2). A rebuilt month or a new corpus misses, even for the same scores. Only answers that passed validation are stored; a 45-day lifecycle rule deletes old entries | a repeated request makes no LLM call; a new month, evidence rebuild, corpus or prompt version misses |
| D2 | **Tool-result cache** in memory, per instance, for the parquet reads (they don't change within a month) | a second read in the same instance takes no GCS call |
| D3 | **Daily call cap** shared across instances: one counter object per day in the cache bucket, updated with a generation-match write (no race between the 2 instances). **Nested runs count:** Agent 2 called by Agent 1, and Agent 1 called by Agent 4, each count as calls | over the cap, the API returns 429 with a clear message; fallbacks don't count |
| D4 | `max_tokens` per agent, plus Pydantic AI usage limits per run: a request limit, a tool-call limit, and **at most k Agent 2 calls per Agent 1 run** (k from settings) | a runaway loop stops; a test shows the Agent 2 limit holds |

### E. Evidence data layer (Riya)

| ID | Task | Done when |
|---|---|---|
| E1 | `agents/tools/source.py`: one `EvidenceSource` interface. It finds the month only from `agents/evidence/_current.json`. The GCS parquet version comes now; a BigQuery version can come later with the same methods, and every BigQuery query sets `maximum_bytes_billed` | the agents never know where the data lives |
| E2 | **Build job** `precursor-pipeline agents build-evidence --v YYYY-MM`, which writes `agents/evidence/v=YYYY-MM/`: per-facility citations (last 36 months), penalties, the latest scores and reasons, the chain roll-ups, the **ops evidence** for Agent 3 (status, the champion's gate results from `ml_runs.gate_results_json`, the evaluation summary, drift flags), and a manifest with sha256s. It writes `_current.json` last. During development it reads `ml-dev/` | it runs on the toy chain in CI; a new DVC stage refreshes `dvc.lock` in the same PR (the stale-lock check fails CI otherwise); the output is small (MBs, not GBs) |
| E3 | **Tools** for Agents 1 and 4: `get_facility_score`, `get_reasons`, `get_citations`, `get_penalties`, `get_chain`, `portfolio_stats` (counting done in polars, never by the model). Each returns records with evidence IDs (`score:<ccn>:<v>`, `citation:<ccn>:<date>:<tag>`, `penalty:<ccn>:<date>:<type>`, `reason:<ccn>:<v>:<rank>`) | each tool has typed output, a timeout and a unit test |
| E4 | Data-as-of on every record: the score month and the gold run | every tool record carries `v` and `gold_run_id` |

### F. Agent 1, Facility Investigation (Riya)

| ID | Task | Done when |
|---|---|---|
| F1 | `InvestigationBrief`: risk (probability, band, percentile), the top raising and lowering drivers, recent citations and penalties as evidence, "what to look at", data-as-of and the caveat | the model and the fallback both fill it |
| F2 | `configs/prompts/facility_investigation.yaml` v1 | the prompt is reviewed by `agent-reviewer` |
| F3 | The agent with E3's tools, plus Agent 2 as a tool for regulation quotes | unit tests on `TestModel` pass, including the fallback |
| F4 | A golden set of about 40 facilities (high, mid, low risk; with and without recent citations; one closed facility) with recorded tool outputs in `evals/facility_investigation/` | the evaluation targets are met |

### G. Corpus and Agent 2, Regulatory Evidence Retriever (Aakash)

| ID | Task | Done when |
|---|---|---|
| G1 | **Corpus build** `precursor-pipeline agents build-corpus`: CMS deficiency-tag descriptions (`citation_lookup`), and Appendix PP regulation text if the team agrees, cut into passages with stable IDs (`passage:<source>:<tag>:<n>`), written to `agents/corpus/<version>/` with a manifest | passages are deterministic: the same input gives the same IDs |
| G2 | **BM25 search** (`rank-bm25`) over the passages, loaded once per instance; a score threshold under which the agent abstains | the search returns the right tag for the known queries in a test |
| G3 | `EvidenceSet`: quotes (exact text plus passage ID), each claim's support, or an explicit `abstained: true` with the reason | the quote-fidelity validator (B6) passes |
| G4 | `configs/prompts/evidence_retriever.yaml` v1 and the agent | unit tests pass, including abstention |
| G5 | A golden set of about 50 questions, about 10 of them unanswerable, in `evals/evidence_retriever/` | 100% quote fidelity; correct abstention |

### H. Agent 4, Portfolio Briefing (Riya)

| ID | Task | Done when |
|---|---|---|
| H1 | `ExecutiveBrief`: book size, the band mix, the top movers, chain concentration, and the top N facilities summarised through Agent 1; all counts come from `portfolio_stats` | every count matches the tool's output |
| H2 | Input: a list of CCNs (the insurer's book), capped at a size limit; the top N go to Agent 1 (N from settings). With D4's limit of k Agent 2 calls per Agent 1 run, one brief is at most **1 + N x (1 + k)** agent runs, each with at most one retry | a book of 500 stays inside that bound in a test; every nested run counts toward the daily cap (D3) |
| H3 | The prompt v1 and the agent | unit tests pass |
| H4 | A golden set of about 10 portfolios | 100% numeric consistency |

### I. Agent 3, Model Ops Analyst (Tanoj)

| ID | Task | Done when |
|---|---|---|
| I1 | **Tools:** `get_status`, `get_gate_results`, `get_model_card` and `get_drift`, all reading the ops evidence that `build-evidence` wrote into `agents/evidence/v=.../`. The API can't reach Postgres (`ml_runs`) or the run folders (`ml/runs/`), so it never reads them directly | each tool has a test on toy ops evidence |
| I2 | `OpsReport`: champion version and age, the last decision, gates (pass, fail, why), drift flags, and "what a person should check"; it never recommends promoting | the language check rejects "promote" or "roll back" as advice |
| I3 | The prompt v1 and the agent | unit tests pass |
| I4 | A golden set of about 15 scenarios: a hold, drift, a stale champion, no champion yet, a skip | 100% numeric consistency |

### J. Evaluation (Tanoj builds the harness; each owner writes their golden set)

| ID | Task | Done when |
|---|---|---|
| J1 | Harness `python -m precursorintelligence.agents.eval <agent>`: replays recorded tool outputs, runs the agent, scores each target, writes a report, and logs traces and metrics to MLflow | it runs offline on `TestModel` in CI |
| J2 | Metric code: citation validity, numeric consistency, driver direction, quote fidelity, abstention, fallback rate | each metric is unit-tested |
| J3 | `.github/workflows/llm-eval.yml`: manual run (plus required before a prompt version bump), WIF as a new `sa-llm-eval` (Vertex AI user only), a hard cap on calls per run | one run per agent costs a known, small amount |
| J4 | Version rule: a prompt's `version` goes up only when its golden set passes; the passing report is linked in the PR | the PR template asks for the link |

### K. API, `backend/` (Tanoj does the shared parts; each agent owner adds their endpoint)

| ID | Task | Done when |
|---|---|---|
| K1 | **(Tanoj)** The FastAPI app, `/health`, the error shape (no stack traces), request ids, and **Google sign-in**: every endpoint except `/health` verifies the Google ID token (`google.oauth2.id_token`; `google-auth` is already in the `agents` extra) against the OAuth client ID and an email allowlist in Secret Manager. FastAPI also serves the React `dist/` | the OpenAPI document is complete; no token or a bad token gets 401, an email off the list gets 403 |
| K2 | **(Tanoj)** The daily-cap middleware (D3) and the log line (C2) on every LLM endpoint | the cap is tested |
| K3 | **(Riya)** `POST /facilities/{ccn}/brief`, `POST /portfolio/brief` | contract tests pass |
| K4 | **(Aakash)** `POST /evidence/search` | contract tests pass |
| K5 | **(Tanoj)** `GET /ops/report` | contract tests pass |
| K6 | **(Riya)** Read-only data endpoints the dashboard needs without an LLM: `GET /facilities/{ccn}` and `GET /status` | contract tests pass |

### L. GCP production rollout (Aakash; Nishanth reviews)

| ID | Task | Done when |
|---|---|---|
| L1 | **Terraform, the service account** `sa-api`: object read on `agents/` in the data bucket only (an IAM condition on the prefix), plus `legacyBucketReader` on the bucket, because gcsfs calls `storage.buckets.get` first (the same fix `sa-pipeline` has in `infra/terraform/envs/prod/main.tf`). The API reads objects by exact name from `_current.json`, so it needs no `objects.list`, which prefix conditions don't cover. Read and write on the cache bucket; `roles/aiplatform.user`; `logging.logWriter`; `cloudtrace.agent`; read on the allowlist secret; nothing else | `terraform plan` shows only these grants; a smoke test reads one evidence file as `sa-api` |
| L2 | **Terraform, the cache bucket** `precursor-intel-mlops-agents-cache` (us-central1, uniform access, 45-day lifecycle) | it exists, and `sa-api` is its only writer |
| L3 | **Terraform, Cloud Run** `precursor-api`: min 0, max 2, 1 vCPU and 1 GiB to start, concurrency 10, request timeout 120 s, the `sa-api` service account, the image from Artifact Registry, environment variables from settings, the OAuth client ID as a variable, and the allowlist from Secret Manager | the service exists, serving traffic to the latest revision |
| L4 | **Vertex AI, by hand:** add `aiplatform.googleapis.com` and `run.googleapis.com` to the `gcloud services enable` command in `gcp-bootstrap.md` step 1 (the project owner runs it); enable the gpt-oss models in Model Garden; confirm the region and the exact model id. **Day one, from a laptop:** prove that a Pydantic AI run with a tool call and structured output works on the gpt-oss endpoint; it's the riskiest unknown | a tool call with structured output succeeds, first from a laptop, then from Cloud Run |
| L5 | **Terraform, observability:** a log sink from Cloud Run's `llm_call` lines to BigQuery `llm_calls` (once Nishanth's BigQuery dataset exists), and the C4 metrics and alerts | a call shows up in BigQuery within minutes |
| L6 | **Budgets, by hand:** the main budget (`pi-trial-credit`) already exists. Add a Vertex-only budget in the console and record it in `gcp-bootstrap.md`. Not Terraform: `sa-tf-apply` has no billing-account role | alert emails go to the team |
| L7 | **Image:** `docker/backend.Dockerfile` (the name `backend/README.md` uses). A pinned `node` stage runs `npm ci` and `vite build`; the Python stage uses uv with `--extra agents --extra api`, non-root, pinned base, and copies in `dist/`. Add what the API needs to the `agents` or `api` extras: `polars`, `pyarrow`, `gcsfs`, `opentelemetry-sdk` and the Cloud Trace exporter. Not the whole `pipeline` extra: duckdb, pandas, pandera and psycopg stay out | `docker run ... /health` returns 200 in CI, and `/` serves the app |
| L8 | **CD:** `cd.yml` builds, tests and pushes `api:<sha>`, and counts it in "already published" | the image is in Artifact Registry after a merge |
| L9 | **Deploy:** `deploy-api.yml` deploys a sha to Cloud Run with no traffic, runs a smoke test on the new revision's URL (health plus one cached brief), then moves traffic to 100%. A rollback deploys an older sha. WIF uses a new `sa-api-deployer` (`run.developer` on this one service, `actAs sa-api`) | a bad revision never gets traffic; a rollback works |
| L10 | **The batch step on the VM:** a new DAG `cms_nh_agents`, scheduled on the asset `cms_nh_scores` (which `publish` emits), runs `build-evidence` and then `build-corpus` (when its version changes) in the `pipeline` image. So the idle shutdown waits for it, add it to the `PI_BUSY_DAGS` default in `infra/vm/pi-busy` and to `RUN_DAGS` in `infra/vm/pi-down`. Both ship to the VM as metadata through Terraform, so this is an infra change, picked up on the VM's next boot | after a monthly publish, `agents/evidence/v=<new month>/` and the new `_current.json` appear |
| L11 | **Pre-warming (optional):** after `build-evidence`, generate and cache briefs for the top 50 riskiest facilities, so the demo and the busiest requests are instant | the cost stays inside the daily cap |
| L12 | **Runbook** `docs/runbooks/agents.md`: deploy, rollback, adding a user to the allowlist, raising the cap, reading traces, cost checks | a teammate can deploy and roll back from it |

### M. Frontend, the React app (owner: decision 0)

| ID | Task | Done when |
|---|---|---|
| M1 | **ADR** `docs/adr/0001-ui-hosting-and-auth.md`: one Cloud Run service with FastAPI serving the built app (recommended) versus a separate `precursor-ui` nginx service with CORS; Google sign-in versus IAP (check that IAP works on a project with no organization; ours has none) | the team agrees and it's merged |
| M2 | **Scaffold** `frontend/`: Vite, TypeScript, shadcn/ui, and a typed API client generated from the OpenAPI document. Replace the Streamlit plan in `frontend/README.md` and drop the `ui` extra from `pyproject.toml` | `npm run build` produces `dist/` |
| M3 | **Sign-in:** Google sign-in, with the ID token sent on every API call; signed-out and not-allowed screens. One OAuth client ID, created by hand in the console and recorded in `gcp-bootstrap.md` | a signed-out user sees only the sign-in screen |
| M4 | **Views:** facility (brief, score, drivers, evidence), portfolio (upload a CCN list, get the brief), evidence search, and ops | each view renders a brief from recorded API responses |
| M5 | **Every state** in every view: loading, empty, error, a fallback answer (marked as such), and 429 over the daily cap. Always show `data_as_of` and the caveat | a component test covers each state |
| M6 | **CI** in `ci.yml`: `npm ci`, lint, typecheck, `vite build` and component tests; L7's image includes the build | CI fails on a type error |

---

## 3. A request in production, step by step
1. The React app, signed in with Google, calls `POST /facilities/{ccn}/brief` with the user's ID token.
2. Cloud Run (`precursor-api`) verifies the token and the allowlist, then checks the CCN and the size limits. Bad input returns 400 with no LLM call.
3. The API reads `agents/evidence/_current.json`, and the cache key is built from its evidence manifest sha (plus the corpus version). On a hit, the cached answer comes back in milliseconds.
4. The daily counter is checked and incremented. Over the cap, the API returns 429.
5. The agent runs on Vertex AI's `gpt-oss-120b`; its tools read `agents/evidence/v=YYYY-MM/` from GCS, for the month in the pointer.
6. The validators check the answer, with one retry if they fail; if the retry fails too, the fallback template answers.
7. A validated answer is stored in the cache. One log line goes to Cloud Logging, then BigQuery; the trace goes to Cloud Trace.
8. The answer returns with `data_as_of`, the caveat and its evidence IDs.

**Monthly:** `cms_nh_ml` publishes, the asset starts `cms_nh_agents`, the new evidence is written, `_current.json` moves last, the next request's cache key changes, and the answers refresh by themselves.

---

## 4. Order and dependencies

| When | Tanoj | Riya | Aakash | Frontend | Needs |
|---|---|---|---|---|---|
| **Nov 19-21** | A1-A5 (merge first); B1, B3, B7 | E1, E3 on `ml-dev/` and toy data | G1, G2; L4's laptop proof; L7 | M1, M2 | the project owner enables `aiplatform` and `run`; no prod scores yet, so development uses `ml-dev/` |
| **Nov 21-24** | B2, B4-B9; C1-C3; D1-D4 | E2, E4; F1-F3 | G3, G4; L1-L3 (plan reviewed) | M3, M4 on recorded responses | A merged; the OAuth client ID |
| **Nov 24-26** | I1-I3; J1, J2; K1, K2, K5 | H1-H3; K3, K6 | K4; L4, L8, L9 | M5, M6 | Vertex access (L4) |
| **Nov 26-28** | J3; run evaluations | F4, H4 | G5; L10, L6 | wire the views to the live API | golden sets; MLflow on the VM for evaluation traces; prod scores (hand-off items 2-3) for L10 |
| **Nov 28-30** | C4; fixes | fixes | L5 (if BigQuery is ready), L11, L12 | fixes | Nishanth's BigQuery |
| **Dec 1** | demo | demo | demo | demo | |

**The critical path:** A, then the agents, then Vertex access (L4), then the evaluations, then deploy (L9). Vertex access is the riskiest dependency: L4's laptop proof of tool calling with structured output happens on day one.

**If prod scores aren't ready by Dec 1:** `build-evidence` runs once from `ml-dev/scores/v=2026-10/`, and the demo uses that month.

**If BigQuery isn't ready by Dec 1:** nothing breaks. The tools read GCS (E1), and the `llm_calls` lines stay in Cloud Logging until the sink exists.

---

## 5. Definition of done for the demo
- [ ] `precursor-api` on Cloud Run, deployed by a sha through `deploy-api.yml`, with a tested rollback.
- [ ] The React app served by `precursor-api`, behind Google sign-in, with every state handled.
- [ ] Agents 1 and 2 answer through the API with validated output, evidence IDs, `data_as_of` and the caveat.
- [ ] Golden-set targets met for Agents 1 and 2; Agents 3 and 4 if time allows.
- [ ] Traces visible in Cloud Trace; one log line per call; the fallback rate under 5%.
- [ ] The cache and daily cap work, nested runs included; the Vertex budget and alerts are set.
- [ ] `cms_nh_agents` builds the evidence and moves `_current.json` after a publish.
- [ ] ADR 0001 (UI hosting and auth) is merged.
- [ ] `docs/runbooks/agents.md` is written.

---

## 6. Decisions needed now
0. **The frontend owner.** The React app is a new workstream (M), and it needs an owner.
1. **Demo scope.** Recommendation: Agents 1 and 2 are the must-haves; Agents 3 and 4 are stretch goals.
2. **Vertex AI.** Who enables the gpt-oss models in Model Garden; the region (us-central1 if available); the quota; the daily call cap and the monthly LLM budget.
3. **Tracing in production.** Cloud Trace for production and MLflow for evaluation runs (recommended), or opening MLflow to Cloud Run, which needs a login (spec O5) and an always-on VM.
4. **UI hosting and auth (ADR 0001).** Recommended: one Cloud Run service with FastAPI serving the built app, and Google sign-in verified in FastAPI against an email allowlist. The alternatives are a separate nginx `precursor-ui` service with CORS, and IAP in front of Cloud Run (no auth code, but it has to work on a project with no organization). Either way, `/health` stays open and every other endpoint needs a signed-in user.
5. **The Retriever's corpus.** Deficiency-tag descriptions only, or Appendix PP too.
6. **The cache.** A GCS bucket (recommended) or Firestore.
