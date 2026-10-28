"""The ``bias-report`` sub-command of ``precursor-pipeline`` (registered in harmonize/cli.py).

  precursor-pipeline bias-report                                  # gold behind gold/_current.json (dev: ./data)
  precursor-pipeline bias-report --gold data/gold --scope latest  # a local gold folder, the latest month only
  precursor-pipeline bias-report --labels y_g12,y_g6 --out reports/bias

Exit codes: 0 done (findings are information, not failures), 2 bad input (a required column is missing, the
dataset or scope is empty, gold/_current.json is missing or out of step), 3 config or usage error.
"""

from __future__ import annotations

import logging
from pathlib import Path

import typer

from ..common.config import ConfigError, load_feature_registry, project_root
from ..gold import GoldCheckError
from ..ingestion.cli import ENV_FILE_OPT, ENV_OPT, _csv, _init
from ..ingestion.context import build_context
from ..ingestion.pipeline import EXIT_CONFIG, EXIT_VALIDATION
from . import BiasInputError
from .config import SCOPES, default_config_path, load_bias_config
from .io import load_gold_path, load_gold_store

log = logging.getLogger(__name__)


def bias_report(
    config: Path | None = typer.Option(
        None, "--config", help="bias slice config; default: configs/bias_slices.yaml"
    ),
    gold: Path | None = typer.Option(
        None,
        "--gold",
        help="local gold folder (holding facility_month_dataset/), the table folder or one parquet file; "
        "default: the dataset gold/_current.json points at in the env's object store",
    ),
    labels: str | None = typer.Option(
        None, "--labels", help="comma-separated labels, e.g. y_g12,y_g6; default: the config's"
    ),
    scope: str = typer.Option("all_labeled", "--scope", help=f"one of {', '.join(SCOPES)}"),
    out: Path | None = typer.Option(
        None, "--out", help="output folder; default: the config's output_dir"
    ),
    charts: bool = typer.Option(True, "--charts/--no-charts", help="write charts/*.png"),
    env: str = ENV_OPT,
    env_file: Path | None = ENV_FILE_OPT,
    log_level: str | None = typer.Option(None, help="override the configured log level"),
) -> None:
    """Data-level bias slicing on gold/facility_month_dataset: representation, label-rate disparity,
    SliceFinder tests with FDR control, rater bias by state, missingness, skew and sample weights."""
    settings, run_id, _ = _init(env, env_file, log_level, None, label="bias")
    try:  # the analysis stack is the optional "bias" extra, so the pipeline image can import this CLI
        from .report import run_bias_report
    except ImportError as exc:
        typer.echo(
            f"config error: {exc}; install the extra: uv sync --extra pipeline --extra bias",
            err=True,
        )
        raise typer.Exit(EXIT_CONFIG) from None
    config_path = config or default_config_path()
    try:
        cfg = load_bias_config(config_path)
        registry = load_feature_registry()
    except ConfigError as exc:
        typer.echo(f"config error: {exc}", err=True)
        raise typer.Exit(EXIT_CONFIG) from None
    if scope not in SCOPES:
        typer.echo(f"config error: --scope must be one of {list(SCOPES)}, not {scope}", err=True)
        raise typer.Exit(EXIT_CONFIG)
    chosen = _csv(labels) or list(cfg.labels)
    if bad := [lb for lb in chosen if not lb.startswith("y_")]:
        typer.echo(f"config error: labels must be y_<name> columns: {bad}", err=True)
        raise typer.Exit(EXIT_CONFIG)
    out_dir = out or Path(cfg.output_dir)
    if not out_dir.is_absolute() and out is None:
        out_dir = project_root() / out_dir
    try:
        if gold is not None:
            data = load_gold_path(gold, cfg)
        else:
            data = load_gold_store(build_context(settings, run_id=run_id).object_store, cfg)
        log.info("bias: loaded %d rows from %s", data.df.height, data.source)
        res = run_bias_report(
            data,
            cfg,
            labels=chosen,
            scope=scope,
            out_dir=out_dir,
            registry=registry,
            config_path=Path(config_path),
            charts=charts,
        )
    except (BiasInputError, GoldCheckError) as exc:
        typer.echo(f"input error: {exc}", err=True)
        raise typer.Exit(EXIT_VALIDATION) from None
    s = res.summary
    typer.echo(
        f"bias report ({scope}): {s['rows_in_scope']} rows, {s['n_slices']} slice values, "
        f"{s['n_tests']} tests"
    )
    if s["no_labeled_rows"]:
        typer.echo("  no labeled rows in this scope: representation and missingness only")
    for name, n in s["flagged"].items():
        typer.echo(f"  {name}: {n} flagged")
    typer.echo(f"  report: {(res.out_dir / 'bias_report.md').as_posix()}")
