"""``precursor-pipeline agents ...``: the agents' batch steps (docs/AGENTS_PLAN.md §1, L10).

  precursor-pipeline agents build-corpus    the Retriever's corpus from silver (G1), agents/corpus/<version>/

Exit codes follow the pipeline contract: 0 ok, 2 data failure (the corpus input is missing or malformed),
3 config or usage error.
"""

from __future__ import annotations

from pathlib import Path

import typer

from ..common.config import ConfigError
from ..common.io.registry import build_object_store
from ..ingestion.cli import ENV_FILE_OPT, ENV_OPT, _init
from ..ingestion.pipeline import EXIT_CONFIG, EXIT_VALIDATION
from .corpus import DEFAULT_PREFIX, CorpusError, build_corpus

agents_app = typer.Typer(
    add_completion=False,
    help="The agents' batch steps: build the Retriever's corpus (and, later, the evidence)",
    no_args_is_help=True,
)


@agents_app.callback()
def agents_root() -> None:
    """Keeps `build-corpus` a named sub-command until `build-evidence` (E2) joins it."""


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
