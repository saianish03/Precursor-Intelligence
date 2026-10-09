---
name: pr-ready
description: Pre-PR checklist for this repository. Runs formatting, lint, types and tests, checks branch naming and the linked issue, and summarizes what the PR description needs.
disable-model-invocation: true
argument-hint: "[issue-number]"
---

## Current branch and changes
!`git branch --show-current`
!`git status --short`
!`git diff --stat origin/dev...HEAD`

Prepare this branch for a pull request for issue $ARGUMENTS:

1. Check the branch name matches `<feat|fix|docs|test|refactor|ci|chore>/<short-name>` (lowercase; branched from the latest `dev`).
2. Run `uv run ruff format .`, `uv run ruff check .`, `uv run mypy src` and `uv run pytest`. Fix failures.
3. If the diff touches `src/precursorintelligence/{ingestion,validation,features}` or `configs/labels.yaml`, run
   `uv run pytest tests/leakage -v` and ask the `leakage-reviewer` subagent to review the diff.
4. If it touches `src/precursorintelligence/agents`, `configs/prompts/` or `evals/`, ask the `agent-reviewer`.
5. If it touches `infra/`, `docker/` or `.github/workflows/`, ask the `infra-reviewer`.
6. Confirm configs, feature registry and docs are updated where relevant, and no secrets are in the diff.
7. Draft the PR description for a PR into `dev` following the README PR rules (what and why, how it was tested, config or schema changes, breaking changes), including `Closes #$ARGUMENTS`.
Do not push or open the PR; show me the summary and the draft description.
