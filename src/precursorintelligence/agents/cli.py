"""``precursor-pipeline agents ...``: the agents' batch steps (docs/AGENTS_PLAN.md §1, L10).

  precursor-pipeline agents build-corpus             the Retriever's corpus from silver (G1), agents/corpus/<version>/
  precursor-pipeline agents build-evidence --v YYYY-MM  the month's evidence (E2), agents/evidence/v=YYYY-MM/

Exit codes follow the pipeline contract: 0 ok, 1 error (retried), 2 data failure (an input is missing or malformed),
3 config or usage error.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer

from ..common.config import ConfigError, load_agents_config, load_ml_config
from ..common.io.registry import build_object_store
from ..ingestion.cli import ENV_FILE_OPT, ENV_OPT, RUN_ID_OPT, _init
from ..ingestion.pipeline import EXIT_CONFIG, EXIT_VALIDATION
from ..training.cli import LOG_LEVEL_OPT, ML_CONFIG_OPT, _run_task
from .corpus import DEFAULT_PREFIX, CorpusError, build_corpus

agents_app = typer.Typer(
    add_completion=False,
    help="The agents' batch steps: build the Retriever's corpus and the month's evidence (docs/AGENTS_PLAN.md)",
    no_args_is_help=True,
)


@agents_app.command("build-corpus")
def build_corpus_command(
    prefix: str = typer.Option(
        DEFAULT_PREFIX,
        help="where the corpus goes in the object store; _current.json is written last",
    ),
    env: str = ENV_OPT,
    env_file: Path | None = ENV_FILE_OPT,
    log_level: str | None = typer.Option(None, help="override the configured log level"),
) -> None:
    """Build the Regulatory Evidence Retriever's corpus from silver citation_lookup_versions.

    The same input gives the same version and the same passage IDs; re-running it writes only the pointer. It opens
    only the object store, never the metadata store.
    """
    settings, _, _ = _init(env, env_file, log_level, None, label="build-corpus")
    try:
        res = build_corpus(build_object_store(settings), prefix=prefix)
    except ConfigError as exc:
        typer.echo(f"config error: {exc}", err=True)
        raise typer.Exit(EXIT_CONFIG) from None
    except CorpusError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(EXIT_VALIDATION) from None
    state = "built" if res.written else "unchanged (already published)"
    typer.echo(
        f"corpus {res.version}: {res.passages} passages, {state}; {res.prefix}_current.json points at it"
    )


@agents_app.command("build-evidence")
def build_evidence(
    v: str = typer.Option(..., "--v", help="the score month, YYYY-MM"),
    run_id: str | None = RUN_ID_OPT,
    env: str = ENV_OPT,
    env_file: Path | None = ENV_FILE_OPT,
    log_level: str | None = LOG_LEVEL_OPT,
    ml_config: Path | None = ML_CONFIG_OPT,
    agents_config: Path | None = typer.Option(
        None, "--agents-config", help="agents settings file (default: configs/agents.yaml)"
    ),
) -> None:
    """Write the month's evidence under agents/evidence/v=YYYY-MM/, then move _current.json to it (E2). Last stdout
    line: a JSON summary."""

    def body(ctx: Any, rid: str) -> dict[str, Any]:
        from ..evidence.build import run_build_evidence

        out = run_build_evidence(
            ctx.object_store,
            ctx.metadata,
            v=v,
            ml_cfg=load_ml_config(ml_config),
            cfg=load_agents_config(agents_config).evidence,
        )
        typer.echo(f"build-evidence {out.v}: {out.rows}")
        return out.summary()

    _run_task("agents_build_evidence", env, env_file, log_level, run_id, body, needs_tracking=False)
