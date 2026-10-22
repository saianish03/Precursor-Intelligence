"""Command line entry point: ``precursor-ingest`` (or ``python -m precursorintelligence.ingestion``).

Examples:
  precursor-ingest step1                                   # full Step 1 on the dev env (local disk)
  precursor-ingest step1 --zip-dates 2021-01-27,2021-12-27 # a subset of archive zips
  precursor-ingest step1 --steps resolve,report            # rebuild the report from stored metadata
  precursor-ingest step1 --env prod                        # same pipeline, GCS + Postgres backends
  precursor-ingest catalog                                 # print what CMS advertises (read-only)

As the `ingestion` TaskGroup of the one-DAG pipeline (docs/DATA_PIPELINE_SPEC.md §15; one shared
--run-id per DAG run):
  precursor-ingest lock acquire --run-id R                 # first ingestion task
  precursor-ingest step1 --run-id R --steps discover,download
  precursor-ingest entries --run-id R --pending            # last stdout line: JSON list of entry ids
  precursor-ingest step1 --run-id R --steps extract,inventory --entry-ids <id>   # one per entry
  precursor-ingest step1 --run-id R --steps resolve,report
  ... bronze, silver, gold ...
  precursor-ingest lock release --run-id R                 # last task of the whole DAG (all_done)

Exit codes: 0 ok (or nothing new); 1 error, retrying may help; 2 validation / drift (quarantined or
rejected zip, unsafe archive, changed CMS catalog contract); 3 config or usage error (incl. a stale
--entry-ids list). The DAG retries 1 and fails 2 and 3 at once.
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
from datetime import date
from pathlib import Path

import typer

from ..common.config import ConfigError, Settings, load_settings
from ..common.io.run_lock import RunLock, RunLockHeldError, RunLockNotHeldError
from ..common.obs.logging import setup_logging
from .catalog import UnknownEntryError, default_source, select
from .context import Context, build_context, new_run_id, validate_run_id
from .inventory import pending_entries
from .pipeline import (
    EXIT_CONFIG,
    EXIT_ERROR,
    EXIT_VALIDATION,
    LOCK_NAME,
    STEPS,
    VALIDATION_ERRORS,
    run_step1,
)

app = typer.Typer(
    add_completion=False,
    help="Precursor Intelligence data ingestion (Step 1: CMS archive download + inventory)",
)
lock_app = typer.Typer(
    help="The run lock (one pipeline run at a time). Used by the DAG's first and last task."
)
app.add_typer(lock_app, name="lock")
log = logging.getLogger("precursorintelligence.cli")

ENV_OPT = typer.Option("dev", help="environment config: configs/<env>.yaml (dev | prod)")
ENV_FILE_OPT = typer.Option(None, help="explicit env yaml (overrides --env)")
RUN_ID_OPT = typer.Option(
    None,
    help="run id shared by every task of one DAG run (Airflow: {{ run_id }}). With it, step1 requires "
    "that the run holds the lock (`lock acquire`); without it, step1 takes and releases the lock itself",
)


def _csv(value: str | None) -> list[str] | None:
    return [v.strip() for v in value.split(",") if v.strip()] if value else None


def _init(
    env: str,
    env_file: Path | None,
    log_level: str | None,
    run_id: str | None = None,
    label: str = "run",
) -> tuple[Settings, str, Path]:
    try:
        settings = load_settings(env, env_path=env_file)
        run_id = validate_run_id(run_id) if run_id else new_run_id()
    except (ConfigError, ValueError) as exc:
        typer.echo(f"config error: {exc}", err=True)
        raise typer.Exit(EXIT_CONFIG) from None
    lc = settings.env.logging
    log_file = setup_logging(
        settings.resolve_path(lc.config),
        log_dir=settings.resolve_path(lc.log_dir),
        run_id=run_id,
        console_format=lc.console_format,
        level=log_level,
        file_stem=f"{label}_{new_run_id()}",  # one file per process, even when the run id is shared
    )
    return settings, run_id, log_file


def _ship_log(ctx: Context, log_file: Path, node: str = "step1") -> None:
    """Copy this process's JSON log next to the run's reports of this node in the object store (local or GCS):
    reports/<node>/run=<id>/logs/. Used by every node's CLI (step1, bronze, silver, gold)."""
    for h in logging.getLogger("precursorintelligence").handlers:
        h.flush()
    if log_file.exists():
        with log_file.open("rb") as f:
            ctx.object_store.put_stream(
                f"reports/{node}/run={ctx.run_id}/logs/{log_file.name}",
                iter(lambda: f.read(1 << 20), b""),
                overwrite=True,
            )


@app.command()
def step1(
    env: str = ENV_OPT,
    env_file: Path | None = ENV_FILE_OPT,
    steps: str = typer.Option(",".join(STEPS), help=f"comma-separated subset of {','.join(STEPS)}"),
    zip_dates: str | None = typer.Option(
        None, help="only these zip dates, e.g. 2021-01-27,2026-07-29"
    ),
    entry_ids: str | None = typer.Option(
        None, help="only these catalog entry ids (comma-separated), e.g. from `entries --pending`"
    ),
    limit: int | None = typer.Option(None, help="process at most N selected zips (oldest first)"),
    run_id: str | None = RUN_ID_OPT,
    log_level: str | None = typer.Option(
        None, help="console/module log level override, e.g. DEBUG"
    ),
) -> None:
    """Run Step 1: discover -> download -> extract -> inventory -> resolve -> report."""
    step_list = tuple(_csv(steps) or ())
    if unknown := set(step_list) - set(STEPS):
        typer.echo(f"config error: unknown steps {sorted(unknown)}; valid: {STEPS}", err=True)
        raise typer.Exit(EXIT_CONFIG)
    try:
        dates = [date.fromisoformat(d) for d in _csv(zip_dates) or []] or None
    except ValueError as exc:
        typer.echo(f"config error: --zip-dates: {exc}", err=True)
        raise typer.Exit(EXIT_CONFIG) from None
    if entry_ids is not None and not _csv(entry_ids):
        typer.echo("config error: --entry-ids is empty (omit it to process every entry)", err=True)
        raise typer.Exit(EXIT_CONFIG)
    orchestrated = run_id is not None
    settings, run_id, log_file = _init(
        env, env_file, log_level, run_id, label="step1-" + "+".join(step_list)
    )
    ctx = build_context(settings, run_id=run_id)
    try:
        res = run_step1(
            ctx,
            steps=step_list,
            zip_dates=dates,
            limit=limit,
            entry_ids=_csv(entry_ids),
            own_lock=not orchestrated,
        )
    except RunLockHeldError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(EXIT_ERROR) from None
    except (RunLockNotHeldError, UnknownEntryError) as exc:  # DAG wiring: lock task, task list
        typer.echo(f"config error: {exc}", err=True)
        raise typer.Exit(EXIT_CONFIG) from None
    except VALIDATION_ERRORS as exc:  # e.g. CMS changed the catalog contract
        typer.echo(f"validation error: {exc}", err=True)
        raise typer.Exit(EXIT_VALIDATION) from None
    finally:
        _ship_log(ctx, log_file)
    typer.echo(
        f"run {res.run_id}: selected={res.selected} downloads={res.downloads} "
        f"extracted_files={res.extracted} zips_inventoried={res.inventoried}"
    )
    for v in res.reports.values():
        typer.echo(f"  report: {v}")
    if res.failures:
        typer.echo(f"{len(res.failures)} failures:", err=True)
        for f in res.failures[:20]:
            typer.echo(f"  - {f}", err=True)
        raise typer.Exit(res.exit_code)


@app.command()
def entries(
    env: str = ENV_OPT,
    env_file: Path | None = ENV_FILE_OPT,
    pending: bool = typer.Option(
        False, help="only entries whose stored zip still needs extract + inventory"
    ),
    zip_dates: str | None = typer.Option(None, help="only these zip dates"),
    limit: int | None = typer.Option(None, help="at most N entries (oldest first)"),
    run_id: str | None = RUN_ID_OPT,
) -> None:
    """Print the selected catalog entry ids as a JSON list on the last stdout line (for DAG mapping).

    Reads the metadata store only: run `step1 --steps discover,download` first."""
    try:
        dates = [date.fromisoformat(d) for d in _csv(zip_dates) or []] or None
    except ValueError as exc:
        typer.echo(f"config error: --zip-dates: {exc}", err=True)
        raise typer.Exit(EXIT_CONFIG) from None
    settings, run_id, _ = _init(env, env_file, None, run_id, label="entries")
    ctx = build_context(settings, run_id=run_id)
    chosen = select(ctx, zip_dates=dates, limit=limit)
    if pending:
        chosen = pending_entries(ctx, chosen)
    log.info("entries: %d %s", len(chosen), "pending" if pending else "selected")
    typer.echo(json.dumps([e.entry_id for e in chosen]))


@lock_app.command("acquire")
def lock_acquire(
    run_id: str = typer.Option(..., help="the DAG run's id"),
    env: str = ENV_OPT,
    env_file: Path | None = ENV_FILE_OPT,
) -> None:
    """Take the run lock for RUN_ID (succeeds again if RUN_ID already holds it). A lock older than
    `run_lock.stale_after_hours` (env config, default 12) was left by a killed run and is taken over."""
    settings, run_id, _ = _init(env, env_file, None, run_id, label="lock")
    lock = RunLock(build_context(settings, run_id=run_id).metadata, LOCK_NAME)
    try:
        lock.acquire(run_id, stale_after=settings.env.run_lock.stale_after)
    except RunLockHeldError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(EXIT_ERROR) from None
    typer.echo(f"lock {LOCK_NAME!r} held by run {run_id}")


@lock_app.command("release")
def lock_release(
    run_id: str = typer.Option(..., help="the DAG run's id"),
    force: bool = typer.Option(False, help="release whoever holds it (lock left by a killed run)"),
    env: str = ENV_OPT,
    env_file: Path | None = ENV_FILE_OPT,
) -> None:
    """Release the run lock held by RUN_ID. Releasing a free lock is not an error."""
    settings, run_id, _ = _init(env, env_file, None, run_id, label="lock")
    lock = RunLock(build_context(settings, run_id=run_id).metadata, LOCK_NAME)
    try:
        released = lock.release(run_id, force=force)
    except RunLockHeldError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(EXIT_ERROR) from None
    typer.echo(f"lock {LOCK_NAME!r} {'released' if released else 'was not held'}")


@lock_app.command("status")
def lock_status(env: str = ENV_OPT, env_file: Path | None = ENV_FILE_OPT) -> None:
    """Show who holds the run lock."""
    settings, run_id, _ = _init(env, env_file, None, label="lock")
    held = RunLock(build_context(settings, run_id=run_id).metadata, LOCK_NAME).holder()
    typer.echo(
        f"lock {LOCK_NAME!r} held by run {held['run_id']} ({held['holder']}, since {held['acquired_at']})"
        if held
        else f"lock {LOCK_NAME!r} is free"
    )


@app.command()
def catalog(env: str = ENV_OPT, env_file: Path | None = ENV_FILE_OPT) -> None:
    """Fetch the CMS archive catalog and print what is advertised. Read-only: it stores nothing, so it
    never hides a change from `check_release` and needs no lock (only `step1 --steps discover` stores)."""
    settings, run_id, _ = _init(env, env_file, None, label="catalog")
    ctx = build_context(settings, run_id=run_id)
    try:
        entries = default_source(ctx).fetch().entries
    except VALIDATION_ERRORS as exc:
        typer.echo(f"validation error: {exc}", err=True)
        raise typer.Exit(EXIT_VALIDATION) from None
    width = shutil.get_terminal_size((120, 20)).columns
    for e in sorted(entries, key=lambda e: (e.zip_date, e.type, e.zip_name)):
        line = f"{e.zip_date}  {e.type:<13} {e.size_bytes / 1e6:8.1f} MB  {e.zip_name}"
        typer.echo(line[:width])
    typer.echo(f"{len(entries)} entries advertised")


def main() -> None:
    """Console entry point. Click reports bad flags with exit code 2, which here means a data problem,
    so usage errors are mapped to the config code (3) instead."""
    try:
        code = app(standalone_mode=False)
    # typer bundles its own click; its UsageError (exit_code 2) and friends derive from TyperException
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
