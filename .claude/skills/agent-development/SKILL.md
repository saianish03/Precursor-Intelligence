---
name: agent-development
description: Standards for the four Pydantic AI agents in Precursor Intelligence (Facility Investigation Agent, Regulatory Evidence Retriever, Model Ops Analyst, Portfolio Briefing Agent), their tools, prompts, validators, retrieval corpus, golden-set evaluations, tracing and cost controls. Use when creating or editing anything under src/precursorintelligence/agents, configs/prompts/ or evals/.
---

# Agent development

## The four agents (normal single agents; no orchestration graph)
| Agent | Level | Output | Notes |
|---|---|---|---|
| Facility Investigation Agent | one facility | `InvestigationBrief` | calls the Retriever as a tool (agent delegation) |
| Regulatory Evidence Retriever | one claim/question | `EvidenceSet` | verbatim quotes only; may abstain |
| Model Ops Analyst | the ML system | `OpsReport` | explains gates and drift; never decides |
| Portfolio Briefing Agent | the whole book | `ExecutiveBrief` | SQL tools do the counting; delegates top facilities to Agent 1 |

## Framework and model
- For Pydantic AI APIs (agents, tools, structured output, streaming, testing, delegation), also load the
  `building-pydantic-ai-agents` skill from the `pydantic-ai@claude-plugins-official` plugin; it tracks the
  current framework. This skill holds the project-specific rules.
- Pydantic AI with typed `output_type` models and output validators. Model ID comes from config
  (default `gpt-oss-120b` on Vertex AI managed open models; `gpt-oss-20b` fallback). Never hard-code it.
- Use Pydantic AI test models in unit tests; CI never calls a paid LLM except the `llm-eval` workflow.

## Tool rules
- Tools are **read-only**, typed, with timeouts. No tool writes, promotes, retrains, deploys or sends email.
- Tools return records with stable evidence IDs (e.g. `citation:<ccn>:<date>:<tag>`, `penalty:<ccn>:<date>:<type>`).
- BigQuery tools always set `maximum_bytes_billed`.

## Validators (all required)
1. Every evidence ID in the output exists in this run's tool outputs.
2. Every number matches the source record (tolerance only for documented rounding).
3. Every driver direction matches the SHAP sign.
4. Retriever quotes are exact substrings of stored passages (check in code, not by the model).
5. Required sections present; length limits respected.
On failure: one retry with the validation error, then the deterministic template fallback. Log `fallback=true`.

## Language rules
- Plain, specific, sentence case. Never "will be cited"; say "estimated probability".
- Always include data-as-of and the caveat: regulatory risk, not a claims prediction.
- No coverage decisions ("decline", "approve") from the model; band-to-action mapping is a rule table.

## Prompts, versions, evaluation
- Prompts live in `configs/prompts/<agent>.yaml` with a `version`. Agent versions start at v1 and increase only after
  the golden-set evaluation passes.
- Golden sets: ~40 facilities (Agent 1), ~50 labelled queries incl. unanswerable (Agent 2), ~15 ops scenarios
  (Agent 3), ~10 portfolios (Agent 4), all with recorded tool outputs in `evals/`.
- Targets: citation validity 100%, numeric consistency ≥ 98% (100% for Agents 3–4), driver direction ≥ 95%,
  quote fidelity 100%, correct abstention, fallback rate < 5%.

## Observability and cost
OpenTelemetry spans per run and per tool call → MLflow traces. One structured log line per LLM call
(agent, prompt version, model, tokens, latency, cost, fallback) → Cloud Logging → BigQuery `llm_calls`.
Respect `max_tokens` and the daily call cap from settings.
