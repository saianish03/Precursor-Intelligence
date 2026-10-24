"""The ``gold`` sub-command of ``precursor-pipeline`` (registered in harmonize/cli.py; spec §15.3).

  precursor-pipeline gold --run-id R        # DAG task: the run must hold the data_pipeline lock
  precursor-pipeline gold                   # local run: takes and releases the lock itself
  precursor-pipeline gold --v 2025-07       # an earlier published V (events first seen after it are ignored)
  precursor-pipeline gold --splits-config data/toy/splits.yaml   # split settings from another file

Exit codes: 0 ok, 1 error (retried, e.g. another run holds the lock), 2 data or gate failure (gold/ not promoted),
3 config or usage error (e.g. a bad --v).
"""

from __future__ import annotations

from pathlib import Path

import typer

from ..common.config import ConfigError, load_labels_config, load_splits_config
from ..common.io.run_lock import RunLockHeldError, RunLockNotHeldError
from ..ingestion.cli import ENV_FILE_OPT, ENV_OPT, RUN_ID_OPT, _init, _ship_log
from ..ingestion.context import build_context
from ..ingestion.pipeline import EXIT_CONFIG, EXIT_ERROR
from .build import run_gold

SPLITS_OPT = typer.Option(
    None,
    "--splits-config",
    help="split settings file (test_months, calib_months, required_labels, spine_start); "
    "default: configs/splits.yaml",
)
V_OPT = typer.Option(
    None,
    "--v",
    help="V, the latest vintage to build up to (YYYY-MM, a published month); default: the latest in silver. "
    "Events first seen after V are ignored, but severity, CMS removals and facility exits stay as of the latest "
    "silver",
)


def gold(
    run_id: str | None = RUN_ID_OPT,
    env: str = ENV_OPT,
    env_file: Path | None = ENV_FILE_OPT,
    log_level: str | None = typer.Option(None, help="override the configured log level"),
    v: str | None = V_OPT,
    splits_config: Path | None = SPLITS_OPT,
) -> None:
    """Build gold/spine and gold/labels from silver, checked in gold/_staging first (spec §7, §9, §11)."""
    orchestrated = run_id is not None
    settings, run_id, log_file = _init(env, env_file, log_level, run_id, label="gold")
    ctx = build_context(settings, run_id=run_id)
    try:
        res = run_gold(
            ctx,
            labels_cfg=load_labels_config(),
            splits_cfg=load_splits_config(splits_config),
            v=v,
            own_lock=not orchestrated,
            splits_file=splits_config,
        )
    except ConfigError as exc:
        typer.echo(f"config error: {exc}", err=True)
        raise typer.Exit(EXIT_CONFIG) from None
    except RunLockHeldError as exc:  # another run is busy: retry later
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(EXIT_ERROR) from None
    except RunLockNotHeldError as exc:  # DAG wiring: the lock task never ran
        typer.echo(f"config error: {exc}", err=True)
        raise typer.Exit(EXIT_CONFIG) from None
    finally:
        _ship_log(ctx, log_file, node="gold")
    typer.echo(f"run {res.run_id}: V={res.v or '-'} labels measured from {res.v_labels or '-'}")
    for name, rows in res.rows.items():
        typer.echo(f"  gold/{name}: {rows} rows")
    typer.echo(f"  manifest: {res.manifest_key}")
    if res.failures:
        typer.echo("gold check failed, gold/ left unchanged:", err=True)
        for f in res.failures:
            typer.echo(f"  - {f}", err=True)
        raise typer.Exit(res.exit_code)
