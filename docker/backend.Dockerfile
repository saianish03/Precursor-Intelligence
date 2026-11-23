# precursor-api: the FastAPI service (backend/) with the agents, serving the React app (frontend/) from the same
# origin. One Cloud Run service, `precursor-api` (docs/adr/0001-ui-hosting-and-auth.md; AGENTS_PLAN L3, L7).
#
#   docker build -f docker/backend.Dockerfile --build-arg PI_GIT_SHA=$(git rev-parse HEAD) \
#     --build-arg VITE_GOOGLE_CLIENT_ID=<client id> -t precursor-api:<tag> .
#   docker run --rm -p 8080:8080 precursor-api:<tag>        # then http://localhost:8080/health
# Base images by tag and digest (L10): rebuilds start from the same bytes; Dependabot proposes digest updates.

# ---- Stage 1: the React app's production build (frontend/dist). Node 24 LTS, as frontend/package.json's engines
# field and CI's frontend job.
FROM node:24.22.3-bookworm-slim@sha256:dc26ae92c9fb15cf6b1d63e20f0944b2efe0f115b20e00be48fb23fceb061c54 AS frontend

WORKDIR /frontend

# Dependencies first, from the lockfile only, so a source change does not reinstall them.
COPY frontend/package.json frontend/package-lock.json ./
RUN --mount=type=cache,target=/root/.npm \
    npm ci --no-audit --no-fund

# .dockerignore keeps a laptop's node_modules/ and dist/ out of the context, so this cannot overwrite the install.
COPY frontend/ ./
# The OAuth client ID is not a secret (ADR 0001): Vite inlines it into the bundle at build time. Without it the app
# still builds and serves; only Google sign-in cannot start. `npm run build` = typecheck, then `vite build`.
ARG VITE_GOOGLE_CLIENT_ID=""
RUN npm run build

# ---- Stage 2: the API. Only the `agents` and `api` extras: nothing from `pipeline`, `ml` or `dvc` (no duckdb,
# pandas, pandera or psycopg).
FROM python:3.12.12-slim-bookworm@sha256:593bd06efe90efa80dc4eee3948be7c0fde4134606dd40d8dd8dbcade98e669c

COPY --from=ghcr.io/astral-sh/uv:0.13.0@sha256:cdc6093146eb3ff6a40107b38f008b789e050e77ad87865e381d9917da55a168 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1 \
    PI_PROJECT_ROOT=/app \
    PATH=/app/.venv/bin:$PATH \
    PORT=8080

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra agents --extra api --no-install-project

# The package, plus configs/ (agents.yaml, prompts): read at runtime from PI_PROJECT_ROOT.
COPY src ./src
COPY configs ./configs
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra agents --extra api --no-editable

# The web layer (not part of the wheel: it is imported as `backend.app` from /app) and the built app, at the same
# paths as in the repository (frontend/dist under PI_PROJECT_ROOT).
COPY backend ./backend
COPY --from=frontend /frontend/dist ./frontend/dist

# Non-root, and the code stays root-owned (read-only to the service). The service writes nothing locally: the
# cache and the counter live in GCS.
RUN useradd --create-home --uid 10001 api
USER api

ARG PI_GIT_SHA
RUN test -n "$PI_GIT_SHA" || { echo 'build with --build-arg PI_GIT_SHA=$(git rev-parse HEAD)' >&2; exit 1; }
ENV PI_GIT_SHA=${PI_GIT_SHA}
LABEL org.opencontainers.image.source=https://github.com/saianish03/Precursor-Intelligence \
      org.opencontainers.image.revision=${PI_GIT_SHA}

# Cloud Run sets PORT (8080 unless configured otherwise): `sh -c` reads it at start-up, and `exec` makes uvicorn
# PID 1, so it gets Cloud Run's SIGTERM and shuts down cleanly. The app module is K1's (backend/app/main.py).
EXPOSE 8080
CMD ["sh", "-c", "exec uvicorn backend.app.main:app --host 0.0.0.0 --port \"${PORT:-8080}\""]
