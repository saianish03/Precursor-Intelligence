"""Command line entry point: ``precursor-pipeline`` (or ``python -m precursorintelligence.harmonize``).

One command per DAG task (docs/DATA_PIPELINE_SPEC.md §15.3), all sharing the DAG run's --run-id:
  precursor-pipeline bronze --run-id R

Exit codes follow the pipeline contract: 0 ok, 1 error (retried), 2 data or validation failure,
3 config or usage error. The DAG fails 2 and 3 without retry.
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer

from ..ingestion.cli import ENV_FILE_OPT, ENV_OPT, RUN_ID_OPT, _init
from ..ingestion.pipeline import EXIT_CONFIG, EXIT_ERROR

app = typer.Typer(
    add_completion=False,
    help="Precursor Intelligence pipeline nodes after ingestion (bronze, then silver and gold)",
)


@app.callback()
def root() -> None:
    """Keeps `bronze` a named sub-command, as silver and gold will be (a lone command would collapse)."""


@app.command()
def bronze(
    run_id: str | None = RUN_ID_OPT,
    env: str = ENV_OPT,
    env_file: Path | None = ENV_FILE_OPT,
    log_level: str | None = typer.Option(None, help="override the configured log level"),
) -> None:
    """Load every selected CMS file into lossless all-text bronze Parquet (spec §3.1)."""
    _init(env, env_file, log_level, run_id, label="bronze")
    typer.echo("bronze is not implemented yet", err=True)
    raise typer.Exit(EXIT_ERROR)


def main() -> None:
    """Console entry point. Click reports bad flags with exit code 2, which here means a data problem,
    so usage errors are mapped to the config code (3) instead."""
    try:
        code = app(standalone_mode=False)
    except typer.TyperException as exc:
        show = getattr(exc, "show", None)
        if callable(show):
            show()
        code = getattr(exc, "exit_code", EXIT_ERROR)
        code = EXIT_CONFIG if code == 2 else code
    except typer.Abort:
        code = EXIT_ERROR
    sys.exit(code if isinstance(code, int) else 0)


if __name__ == "__main__":  # pragma: no cover
    main()
