# Agents: detailed implementation plan and GCP production rollout

**Owners:** Tanoj, Riya, Aakash. Nishanth reviews the infrastructure. **Target:** the Dec 1 demo.

**What this adds to the first plan:** the tasks in every part, how the agents run in production on our current GCP setup, and the order. It follows the repo's agent standards (the `agent-development` skill and `.claude/rules/agents.md`) and the deployment standards (`mlops-deployment`). If the plan document says anything different, the plan document wins.

**Ownership change from the first plan:** the production rollout (Terraform, images, deploys) is a full workstream, so Aakash takes it, and Portfolio Briefing (Agent 4) moves to Riya. Agent 4 reuses her tools and her chain roll-ups, so the move costs nothing.

| Person | Owns |
|---|---|
| **Tanoj** | the runtime, guardrails, observability, caching and cost control, the evaluation harness, Agent 3, the API's shared middleware |
| **Riya** | the evidence data layer and its build job, Agent 1, Agent 4, the API endpoints for both |
| **Aakash** | the retrieval corpus, Agent 2 and its endpoint, the GCP production rollout |

---

## 1. Where it runs in production

### What we have today
- **One VM** (`pi-airflow`, e2-standard-4, us-central1-a) runs Airflow, Postgres and, from Phase 2, MLflow, under Docker Compose.
  - It **stops itself when idle**, about 70–85 minutes after the last activity.
  - MLflow has **no login** and is reachable only from the VM's task network.
- **GCS:** `precursor-intel-mlops-data` holds `scores/v=YYYY-MM/` and `status/latest.json`, and `-mlflow` holds the package store.
- **Artifact Registry** gets images that CD pushes on each merge to `dev` or `main` (`pipeline`, `airflow`, and `mlflow` from Phase 2). The Deploy workflow points the VM at a release sha.
- **GitHub Actions** authenticates through Workload Identity Federation (`sa-deployer`, `sa-release`); there are no JSON keys.
- **Secret Manager** holds the VM's secrets. A person adds each value by hand; values never pass through Terraform.
- **Not built yet:** Cloud Run, BigQuery (Nishanth's Phase 2 item 4), Vertex AI access, and billing budgets.

### What follows from that
1. **The agents run on Cloud Run** as `precursor-api` (FastAPI plus the agents): min 0 and max 2 instances, in us-central1. A VM that stops itself can't serve users.
2. **The API never touches the VM.** It reads GCS, and BigQuery once that exists. It never reaches the VM's Postgres or MLflow; the API rules require this.
3. **Production traces go to Cloud Trace, not MLflow.** Cloud Run can't reach MLflow on the VM, and the VM is often off. The golden-set evaluation runs still log their traces to MLflow, so prompt versions can be compared there.
4. **Heavy data work happens in a batch step after `publish`,** not on each request. A new step builds small per-month evidence files that the API reads quickly.
5. **Everything shared lives in GCS:** the cache, the daily call counter and the evidence files. Cloud Run instances share no memory, and they scale to zero.

```
 VM (Airflow, monthly)                                GCS                              Cloud Run
 cms_nh_ml: ... score -> publish --asset-->  scores/v=YYYY-MM/
                                             status/latest.json            ---->  precursor-api (FastAPI)
 cms_nh_agents (new DAG): build-evidence --> agents/evidence/v=YYYY-MM/            |  agents 1-4
                          build-corpus   --> agents/corpus/<version>/              |  guardrails, cache, cap
                                                                                   |
                                             agents-cache bucket  <---------------+  cache + daily counter
                                                                                   |
                         Vertex AI (gpt-oss-120b, OpenAI-compatible API)  <--------+  LLM calls
                         Cloud Trace  <--------------------------------------------+  OTel spans
                         Cloud Logging --sink--> BigQuery llm_calls  <-------------+  one JSON line per call
 Base44 UI --(API key)--> precursor-api
```

---

## 2. Tasks by part

Each task is one PR (or part of one) into `dev`. Agent and prompt PRs get the `agent-reviewer` subagent; infrastructure PRs get the `infra-reviewer` subagent and Nishanth.

### A. Runtime and settings (Tanoj), the foundation everything else imports

| ID | Task | Done when |
|---|---|---|
| A1 | `agents/settings.py`: model id, fallback model id, `max_tokens`, temperature, tool timeout, daily call cap, cache bucket, region, all from `configs/agents.yaml` plus environment variables; nothing hard-coded | settings load in tests; a typo is a config error |
| A2 | `agents/model.py`: a model factory. Production uses Vertex AI's OpenAI-compatible endpoint with a `google-auth` token (Cloud Run's service account, no keys). Tests use Pydantic AI's `TestModel` and `FunctionModel`. If the primary model fails, it falls back to `gpt-oss-20b` | a unit test builds both; the Vertex path is checked by hand once Vertex access exists (L4) |
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
| D1 | **Answer cache** in a new GCS bucket `precursor-intel-mlops-agents-cache`. The key is sha256 of the agent, prompt version, model, normalized input and the month's `scores_manifest` sha. Only answers that passed validation are stored; a 45-day lifecycle rule deletes old entries | a repeated request makes no LLM call; a new month or prompt version misses |
| D2 | **Tool-result cache** in memory, per instance, for the parquet reads (they don't change within a month) | a second read in the same instance takes no GCS call |
| D3 | **Daily call cap** shared across instances: one counter object per day in the cache bucket, updated with a generation-match write (no race between the 2 instances) | over the cap, the API returns 429 with a clear message; fallbacks don't count |
| D4 | `max_tokens` per agent, plus a per-request tool-call limit (Pydantic AI usage limits) | a runaway loop stops |

### E. Evidence data layer (Riya)

| ID | Task | Done when |
|---|---|---|
| E1 | `agents/tools/source.py`: one `EvidenceSource` interface. The GCS parquet version comes now; a BigQuery version can come later with the same methods, and every BigQuery query sets `maximum_bytes_billed` | the agents never know where the data lives |
| E2 | **Build job** `precursor-pipeline agents build-evidence --v YYYY-MM`, which writes `agents/evidence/v=YYYY-MM/`: per-facility citations (last 36 months), penalties, the latest scores and reasons, the chain roll-ups, and a manifest with sha256s | it runs on the toy chain in CI; the output is small (MBs, not GBs) |
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
| H2 | Input: a list of CCNs (the insurer's book), capped at a size limit; the top N go to Agent 1 (N from settings, to keep cost bounded) | a book of 500 makes at most N + 1 LLM runs |
| H3 | The prompt v1 and the agent | unit tests pass |
| H4 | A golden set of about 10 portfolios | 100% numeric consistency |

### I. Agent 3, Model Ops Analyst (Tanoj)

| ID | Task | Done when |
|---|---|---|
| I1 | **Tools:** `get_status` (`status/latest.json`), `get_gate_results` and `get_model_card` (the champion's run folder in GCS), `get_drift` (monitor's flags and PSI). The API can't read `ml_runs` in Postgres, so these read GCS, and BigQuery `meta` later | each tool has a test on a toy run folder |
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
| K1 | **(Tanoj)** The FastAPI app, `/health`, the error shape (no stack traces), request ids, API-key auth (key in Secret Manager, checked in the app) on every endpoint except `/health` | the OpenAPI document is complete; calls without a key get 401 |
| K2 | **(Tanoj)** The daily-cap middleware (D3) and the log line (C2) on every LLM endpoint | the cap is tested |
| K3 | **(Riya)** `POST /facilities/{ccn}/brief`, `POST /portfolio/brief` | contract tests pass |
| K4 | **(Aakash)** `POST /evidence/search` | contract tests pass |
| K5 | **(Tanoj)** `GET /ops/report` | contract tests pass |
| K6 | **(Riya)** Read-only data endpoints the dashboard needs without an LLM: `GET /facilities/{ccn}` and `GET /status` | contract tests pass |

### L. GCP production rollout (Aakash; Nishanth reviews)

| ID | Task | Done when |
|---|---|---|
| L1 | **Terraform, the service account** `sa-api`: read on `scores/`, `status/` and `agents/` in the data bucket (IAM conditions on the prefixes); read and write on the cache bucket; `roles/aiplatform.user`; `logging.logWriter`; `cloudtrace.agent`; read on the API-key secret; nothing else | `terraform plan` shows only these grants |
| L2 | **Terraform, the cache bucket** `precursor-intel-mlops-agents-cache` (us-central1, uniform access, 45-day lifecycle) | it exists, and `sa-api` is its only writer |
| L3 | **Terraform, Cloud Run** `precursor-api`: min 0, max 2, 1 vCPU and 1 GiB to start, concurrency 10, request timeout 120 s, the `sa-api` service account, the image from Artifact Registry, environment variables from settings, and the API key from Secret Manager | the service exists, serving traffic to the latest revision |
| L4 | **Terraform, Vertex AI:** enable `aiplatform.googleapis.com`; enable the gpt-oss models in Model Garden (once, by hand, recorded in the runbook); confirm the region and the exact model id | a test call from Cloud Run succeeds |
| L5 | **Terraform, observability:** a log sink from Cloud Run's `llm_call` lines to BigQuery `llm_calls` (once Nishanth's BigQuery dataset exists), and the C4 metrics and alerts | a call shows up in BigQuery within minutes |
| L6 | **Terraform, budgets:** a billing budget with alerts at $25, $50, $100 and $150, plus a Vertex-only budget | alert emails go to the team |
| L7 | **Image:** `docker/api.Dockerfile` (uv, `--extra agents --extra api`, non-root, pinned base); CI builds and smoke-tests it | `docker run ... /health` returns 200 in CI |
| L8 | **CD:** `cd.yml` builds, tests and pushes `api:<sha>`, and counts it in "already published" | the image is in Artifact Registry after a merge |
| L9 | **Deploy:** `deploy-api.yml` deploys a sha to Cloud Run with no traffic, runs a smoke test on the new revision's URL (health plus one cached brief), then moves traffic to 100%. A rollback deploys an older sha. WIF uses a new `sa-api-deployer` (`run.developer` on this one service, `actAs sa-api`) | a bad revision never gets traffic; a rollback works |
| L10 | **The batch step on the VM:** a new DAG `cms_nh_agents`, scheduled on the asset `cms_nh_scores` (which `publish` emits), runs `build-evidence` and then `build-corpus` (when its version changes) in the `pipeline` image. Add it to `PI_BUSY_DAGS` so the idle shutdown waits for it | after a monthly publish, `agents/evidence/v=<new month>/` appears |
| L11 | **Pre-warming (optional):** after `build-evidence`, generate and cache briefs for the top 50 riskiest facilities, so the demo and the busiest requests are instant | the cost stays inside the daily cap |
| L12 | **Runbook** `docs/runbooks/agents.md`: deploy, rollback, rotating the API key, raising the cap, reading traces, cost checks | a teammate can deploy and roll back from it |

---

## 3. A request in production, step by step
1. The Base44 UI calls `POST /facilities/{ccn}/brief` with the API key.
2. Cloud Run (`precursor-api`) checks the key, the CCN and the size limits. Bad input returns 400 with no LLM call.
3. The cache key is built from the current month's `scores_manifest` sha. On a hit, the cached answer comes back in milliseconds.
4. The daily counter is checked and incremented. Over the cap, the API returns 429.
5. The agent runs on Vertex AI's `gpt-oss-120b`; its tools read `agents/evidence/v=YYYY-MM/` from GCS.
6. The validators check the answer, with one retry if they fail; if the retry fails too, the fallback template answers.
7. A validated answer is stored in the cache. One log line goes to Cloud Logging, then BigQuery; the trace goes to Cloud Trace.
8. The answer returns with `data_as_of`, the caveat and its evidence IDs.

**Monthly:** `cms_nh_ml` publishes, the asset starts `cms_nh_agents`, the new evidence is written, the next request's cache key changes, and the answers refresh by themselves.

---

## 4. Order and dependencies

| When | Tanoj | Riya | Aakash | Needs |
|---|---|---|---|---|
| **Nov 19-21** | A1-A5 (merge first); B1, B3, B7 | E1, E3 on toy data | G1, G2; L7 | none |
| **Nov 21-24** | B2, B4-B9; C1-C3; D1-D4 | E2, E4; F1-F3 | G3, G4; L1-L3 (plan reviewed) | A merged |
| **Nov 24-26** | I1-I3; J1, J2; K1, K2, K5 | H1-H3; K3, K6 | K4; L4, L8, L9 | Vertex access (L4) |
| **Nov 26-28** | J3; run evaluations | F4, H4 | G5; L10, L6 | golden sets; MLflow on the VM for evaluation traces |
| **Nov 28-30** | C4; fixes | fixes | L5 (if BigQuery is ready), L11, L12 | Nishanth's BigQuery |
| **Dec 1** | demo | demo | demo | |

**The critical path:** A, then the agents, then Vertex access (L4), then the evaluations, then deploy (L9). Vertex access is the riskiest dependency: start L4 on day one.

**If BigQuery isn't ready by Dec 1:** nothing breaks. The tools read GCS (E1), and the `llm_calls` lines stay in Cloud Logging until the sink exists.

---

## 5. Definition of done for the demo
- [ ] `precursor-api` on Cloud Run, deployed by a sha through `deploy-api.yml`, with a tested rollback.
- [ ] Agents 1 and 2 answer through the API with validated output, evidence IDs, `data_as_of` and the caveat.
- [ ] Golden-set targets met for Agents 1 and 2; Agents 3 and 4 if time allows.
- [ ] Traces visible in Cloud Trace; one log line per call; the fallback rate under 5%.
- [ ] The cache and daily cap work; budgets and alerts are set.
- [ ] `cms_nh_agents` builds the evidence after a publish.
- [ ] `docs/runbooks/agents.md` is written.

---

## 6. Decisions needed now
1. **Demo scope.** Recommendation: Agents 1 and 2 are the must-haves; Agents 3 and 4 are stretch goals.
2. **Vertex AI.** Who enables the gpt-oss models in Model Garden; the region (us-central1 if available); the quota; the daily call cap and the monthly LLM budget.
3. **Tracing in production.** Cloud Trace for production and MLflow for evaluation runs (recommended), or opening MLflow to Cloud Run, which needs a login (spec O5) and an always-on VM.
4. **API auth for Base44.** An API key in Secret Manager checked by the app (recommended, since Base44 can't mint Google ID tokens), or Cloud Run IAM.
5. **The Retriever's corpus.** Deficiency-tag descriptions only, or Appendix PP too.
6. **The cache.** A GCS bucket (recommended) or Firestore.
