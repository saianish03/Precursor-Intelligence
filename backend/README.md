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
Tests: `tests/backend/`. Image: `docker/backend.Dockerfile` (`docker/README.md`): it runs
`uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT` (8080 by default, as on Cloud Run) from `/app`, with the
built React app at `frontend/dist/` under `PI_PROJECT_ROOT` (`/app`), so `main.py` must expose `app` and serve that
folder. Only the `agents` and `api` extras are installed: anything the API imports belongs in one of them.
