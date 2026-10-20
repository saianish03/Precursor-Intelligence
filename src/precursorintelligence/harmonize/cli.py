"""Command line entry point: ``precursor-pipeline`` (or ``python -m precursorintelligence.harmonize``).

One command per DAG task (docs/DATA_PIPELINE_SPEC.md §15.3), all sharing the DAG run's --run-id:
  precursor-pipeline bronze --run-id R

Exit codes follow the pipeline contract: 0 ok, 1 error (retried), 2 data or validation failure,
3 config or usage error. The DAG fails 2 and 3 without retry.
"""

from __future__ import annotations

import logging
import sys
from collections import Counter
from pathlib import Path

import typer

from ..common.io.run_lock import RunLockHeldError, RunLockNotHeldError
from ..gold.cli import gold
from ..ingestion.cli import ENV_FILE_OPT, ENV_OPT, RUN_ID_OPT, _init
from ..ingestion.context import Context, build_context
from ..ingestion.pipeline import EXIT_CONFIG, EXIT_ERROR, EXIT_OK, EXIT_VALIDATION
from .pipeline import run_bronze
from .verify import verify_bronze

app = typer.Typer(
    add_completion=False,
    help="Precursor Intelligence pipeline nodes after ingestion (bronze, then silver and gold)",
)
app.command()(gold)  # the gold node; its code lives in src/precursorintelligence/gold/


@app.callback()
def root() -> None:
    """Keeps `bronze` a named sub-command, as silver and gold will be (a lone command would collapse)."""


def _ship_log(ctx: Context, log_file: Path) -> None:
    """Copy this process's JSON log next to the run's other bronze outputs in the object store."""
    for h in logging.getLogger("precursorintelligence").handlers:
        h.flush()
    if log_file.exists():
        with log_file.open("rb") as f:
            ctx.object_store.put_stream(
                f"reports/bronze/run={ctx.run_id}/logs/{log_file.name}",
                iter(lambda: f.read(1 << 20), b""),
                overwrite=True,
            )


@app.command()
def bronze(
    run_id: str | None = RUN_ID_OPT,
    env: str = ENV_OPT,
    env_file: Path | None = ENV_FILE_OPT,
    log_level: str | None = typer.Option(None, help="override the configured log level"),
    verify: bool = typer.Option(
        False,
        help="read-only: report what bronze holds against the inventory (no lock, nothing written)",
    ),
) -> None:
    """Load every selected CMS file into lossless all-text bronze Parquet (spec §3.1)."""
    orchestrated = run_id is not None
    settings, run_id, log_file = _init(env, env_file, log_level, run_id, label="bronze")
    ctx = build_context(settings, run_id=run_id)
    if verify:
        report = verify_bronze(ctx)
        typer.echo(report.render())
        raise typer.Exit(EXIT_OK if report.ok else EXIT_VALIDATION)
    try:
        res = run_bronze(ctx, own_lock=not orchestrated)
    except RunLockHeldError as exc:  # another run is busy: retry later
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(EXIT_ERROR) from None
    except RunLockNotHeldError as exc:  # DAG wiring: the lock task never ran
        typer.echo(f"config error: {exc}", err=True)
        raise typer.Exit(EXIT_CONFIG) from None
    finally:
        _ship_log(ctx, log_file)
    counts = Counter(f.status for f in res.files)
    summary = " ".join(
        f"{status}={counts[status]}" for status in ("loaded", "reloaded", "failed", "skipped")
    )
    typer.echo(f"run {res.run_id}: {summary} stale_refresh={res.stale_vintages}")
    typer.echo(f"  manifest: {res.manifest_key}")
    if res.failures:
        typer.echo(f"{len(res.failures)} failures:", err=True)
        for f in res.failures[:20]:
            typer.echo(f"  - {f}", err=True)
        raise typer.Exit(res.exit_code)


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
