---
name: testing-evaluation
description: Testing strategy for Precursor Intelligence. Use when writing or reviewing tests, data contracts, leakage tests, DAG tests, model tests, API contract tests, agent evaluations or the end-to-end nightly run, or when deciding what to test for a change.
---

# Testing and evaluation

## What each suite proves
| Suite | Path | Proves |
|---|---|---|
| Unit | `tests/<stage>/` (mirrors `src/precursorintelligence/<stage>/`) | parsing, column mapping, date formats, CCN handling, feature functions |
| Data contracts | `tests/contracts` | Pandera schemas, value domains, uniqueness, row-count tolerances on fixtures |
| Leakage | `tests/leakage` (to be created) | late-posting fixture never reaches features; strict-future labels; time-travel rebuild identical |
| DAGs | `tests/dags` | DAGs import, no cycles, tasks use the pipeline image, Asset wiring |
| Model | `tests/training`, `tests/evaluation` | gates logic, calibration reduces ECE on fixtures, shuffled-label AUC ≈ 0.5 |
| API | `tests/backend` | OpenAPI contract, auth required, error shapes, daily cap behaviour |
| Agents | `tests/agents` | validators reject fabricated IDs/numbers; fallback path; uses Pydantic AI test models |
| End to end | `tests/e2e` (to be created) | full chain on sample data in Docker Compose (nightly) |

## Rules
- A bug fix starts with a failing regression test.
- Fixtures are small, synthetic or trimmed real CMS rows with fake CCNs; never real secrets; never full archives.
- Any change to `ingestion/ validation/ features/` or `configs/labels.yaml` must keep `tests/leakage` green.
- Tests must not call paid services. Agent quality is measured by `llm-eval` golden sets, not unit tests.
- Don't mock the code under test; mock only external boundaries (CMS API, GCS, BigQuery, Vertex AI).

## Commands
`uv run pytest tests/<stage> -q` (fast) · `uv run pytest` (all except e2e) · `uv run pytest tests/leakage -v`
· `uv run python -m precursorintelligence.agents.eval --suite <name>` (costs tokens; respects the cap).
