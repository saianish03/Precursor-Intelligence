# backend/: API service (FastAPI + Pydantic AI agents) → Cloud Run `precursor-api`

Thin web layer. Business logic lives in `src/precursorintelligence/` (`serving/`, `agents/`); the backend imports it.

```
backend/
  app/
    main.py        # FastAPI app factory, health check   (to be added)
    routers/       # endpoints: facility scores, account roll-ups, risk briefs, ops report
    schemas/       # Pydantic request/response models
    services/      # BigQuery reads, calls into precursorintelligence.serving / precursorintelligence.agents
```
Tests: `tests/backend/`. Image: `docker/backend.Dockerfile` (planned).
