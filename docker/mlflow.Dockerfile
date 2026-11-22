# precursor-mlflow: the MLflow tracking server and model registry of the compose stack (ML spec §16.1). Runs on the
# tasks network only; the command and settings are in docker/airflow-local/docker-compose.yaml.
#
#   docker build -f docker/mlflow.Dockerfile --build-arg PI_GIT_SHA=$(git rev-parse HEAD) -t precursor-mlflow:<tag> .
# Base images by tag and digest (L10): rebuilds start from the same bytes; Dependabot proposes digest updates.
FROM python:3.12.12-slim-bookworm@sha256:593bd06efe90efa80dc4eee3948be7c0fde4134606dd40d8dd8dbcade98e669c

COPY --from=ghcr.io/astral-sh/uv:0.13.0@sha256:cdc6093146eb3ff6a40107b38f008b789e050e77ad87865e381d9917da55a168 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1 \
    PATH=/app/.venv/bin:$PATH

WORKDIR /app

# Only the `mlflow-server` extra (mlflow, the GCS client, the Postgres driver), never the project's code.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project --extra mlflow-server

# Non-root. /mlflow/artifacts is the artifact root when PI_MLFLOW_ARTIFACTS is not a bucket: a named volume mounted
# there starts with this owner.
RUN useradd --create-home --uid 10001 mlflow \
    && mkdir -p /mlflow/artifacts \
    && chown mlflow /mlflow/artifacts
USER mlflow

ARG PI_GIT_SHA
RUN test -n "$PI_GIT_SHA" || { echo 'build with --build-arg PI_GIT_SHA=$(git rev-parse HEAD)' >&2; exit 1; }
ENV PI_GIT_SHA=${PI_GIT_SHA}
LABEL org.opencontainers.image.source=https://github.com/saianish03/Precursor-Intelligence \
      org.opencontainers.image.revision=${PI_GIT_SHA}

EXPOSE 5000
CMD ["mlflow", "server", "--help"]
