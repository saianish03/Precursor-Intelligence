---
paths:
  - "src/precursorintelligence/agents/**"
  - "configs/prompts/**"
  - "evals/**"
  - "tests/agents/**"
---

# Agent rules
- Load the `agent-development` skill before changing an agent, tool, prompt or golden set.
- Tools are read-only. Outputs are typed Pydantic models with validators and a deterministic fallback.
- Bump the prompt `version` on any prompt change and run the matching golden-set evaluation.
- Unit tests use Pydantic AI test models; never call a paid model in unit tests.
