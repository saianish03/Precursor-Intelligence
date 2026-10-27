# precursor-pipeline: the image every data-pipeline DAG task runs, one CLI command per container
# (`precursor-ingest …`, `precursor-pipeline bronze|silver|gold …`). Same image locally and on the VM.
#
#   docker build -f docker/pipeline.Dockerfile --build-arg PI_GIT_SHA=$(git rev-parse HEAD) \
#     -t precursor-pipeline:<tag> .
FROM python:3.12.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.13.0 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1 \
    PI_PROJECT_ROOT=/app \
    PATH=/app/.venv/bin:$PATH

WORKDIR /app

# Dependencies first, so a code change does not reinstall them. Only the `pipeline` extra (ingestion → gold).
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra pipeline --no-install-project

# The package, plus configs/ (sources, env yamls, column maps): read at runtime from PI_PROJECT_ROOT.
COPY src ./src
COPY configs ./configs
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra pipeline --no-editable

# Non-root. /app/data is the local object store root (`--env dev`, bind-mounted from the host) and
# /app/logs the per-process JSON logs; on GCP (`--env prod`) data goes to GCS and neither needs a mount.
RUN useradd --create-home --uid 10001 pipeline \
    && mkdir -p /app/data /app/logs \
    && chown pipeline /app/data /app/logs
USER pipeline

ARG PI_GIT_SHA
# The build must name its commit: gold's dataset manifest records PI_GIT_COMMIT (the image carries no .git).
RUN test -n "$PI_GIT_SHA" || { echo 'build with --build-arg PI_GIT_SHA=$(git rev-parse HEAD)' >&2; exit 1; }
ENV PI_GIT_SHA=${PI_GIT_SHA} \
    PI_GIT_COMMIT=${PI_GIT_SHA}

CMD ["precursor-ingest", "--help"]
