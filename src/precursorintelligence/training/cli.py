"""``precursor-pipeline ml <command>``: one command per cms_nh_ml task (docs/ML_PIPELINE_SPEC.md §15.3).

All share the DAG run's ``--run-id`` (the run folder ``ml/runs/<run id>/``) and ``--env``. ``prepare`` is built (W0);
every other command is a stub that exits 3 until its work package lands, so the DAG and the packages can be wired
against the final names now. Exit codes follow the pipeline contract: 0 ok, 1 error (retried), 2 data refusal,
3 config or usage error. No log shipping: the Airflow task log holds the output, and a store rooted at the bucket
must not get files outside ``outputs.root_prefix``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import typer

from ..common.config import (
    ConfigError,
    load_feature_registry,
    load_labels_config,
    load_ml_config,
    load_splits_config,
)
from ..common.io.object_store import ObjectStoreError
from ..ingestion.cli import ENV_FILE_OPT, ENV_OPT, RUN_ID_OPT, _init
from ..ingestion.context import build_context
from ..ingestion.pipeline import EXIT_CONFIG, EXIT_ERROR, EXIT_VALIDATION
from . import GoldPinError, MlInputError

ml_app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="The ML pipeline's tasks (docs/ML_PIPELINE_SPEC.md §15.3)",
)

ML_CONFIG_OPT = typer.Option(
    None, "--ml-config", help="ML settings file (default: configs/ml.yaml)"
)
LOG_LEVEL_OPT = typer.Option(None, help="override the configured log level")


@ml_app.command()
def prepare(
    run_id: str | None = RUN_ID_OPT,
    env: str = ENV_OPT,
    env_file: Path | None = ENV_FILE_OPT,
    log_level: str | None = LOG_LEVEL_OPT,
    ml_config: Path | None = ML_CONFIG_OPT,
    gold_run_id: str | None = typer.Option(
        None, help="the gold run decide evaluated (refuses any other); default: the current one"
    ),
    allow_unverified: bool | None = typer.Option(
        None,
        "--allow-unverified/--refuse-unverified",
        help="override ml.yaml allow_unverified for this run",
    ),
) -> None:
    """Pin gold into the run folder, check it, and write the frames, windows.json and input/manifest.json
    (§1.2, §1.5, §3). Last stdout line: a JSON summary."""
    from .data import run_prepare  # numpy, pandas: imported only by the commands that need them

    settings, run_id, _ = _init(env, env_file, log_level, run_id, label="ml_prepare")
    try:
        cfg = load_ml_config(ml_config)
        labels_cfg, splits_cfg, registry = (
            load_labels_config(),
            load_splits_config(),
            load_feature_registry(),
        )
        ctx = build_context(settings, run_id=run_id)
        res = run_prepare(
            ctx.object_store,
            ctx.metadata,
            ml_run_id=run_id,
            cfg=cfg,
            labels_cfg=labels_cfg,
            splits_cfg=splits_cfg,
            registry=registry,
            gold_run_id=gold_run_id,
            allow_unverified=allow_unverified,
        )
    except ConfigError as exc:
        typer.echo(f"config error: {exc}", err=True)
        raise typer.Exit(EXIT_CONFIG) from None
    except MlInputError as exc:
        typer.echo(f"refused: {exc}", err=True)
        raise typer.Exit(EXIT_VALIDATION) from None
    except (GoldPinError, ObjectStoreError, OSError) as exc:
        typer.echo(f"error: {exc.__class__.__name__}: {exc}", err=True)
        raise typer.Exit(EXIT_ERROR) from None
    w = res.windows
    typer.echo(f"run {run_id}: gold {res.manifest.gold_run_id}, V {w.v}, V_L {w.v_labels}")
    typer.echo(
        f"  eval {w.eval.first}..{w.eval.last} ({len(w.eval.months)} months, gap {w.eval_gap})"
    )
    typer.echo(f"  test {w.test.first}..{w.test.last} (rows in {w.test.months})")
    typer.echo(f"  refit {w.refit.first}..{w.refit.last} ({len(w.refit.months)} months)")
    typer.echo(f"  rows {res.manifest.rows}")
    for flag in res.manifest.leakage_flags:
        state = "cleared" if flag.cleared else "holds promotion"
        typer.echo(
            f"  leakage flag ({state}): {flag.feature} on {flag.frame}: "
            f"PR-AUC {flag.pr_auc} > {flag.threshold}"
        )
    for regime in res.manifest.regime_flags:
        typer.echo(f"  regime flag: {regime.feature}: {regime.detail}")
    typer.echo(f"  folder: {res.paths.run_dir}")
    typer.echo(json.dumps(res.summary(), sort_keys=True))


# Commands of the later work packages (§18.3). Same options as prepare's core, so the DAG can call them today.
_STUBS = {
    "decide": (
        "W6",
        "Decide what this run does: SKIP, RETRAIN, RETUNE, SCORE, PROMOTE or REFUSE (§15.2).",
    ),
    "tune": ("W1", "Re-tune the LightGBM member with Optuna and write a parameter proposal (§13)."),
    "train": (
        "W1",
        "Fit the candidates on the eval window, calibrate, select, then refit (§4, §5).",
    ),
    "evaluate": (
        "W2",
        "Evaluate the eval fit on the test frame, with baselines and bias slices (§6-§8).",
    ),
    "gate": ("W2", "Apply the gates, register the refit and move the aliases (§9)."),
    "promote": (
        "W2",
        "Manual promotion or rollback; the version and reason come from the environment (§9.4).",
    ),
    "score": ("W3", "Score V with the champion, with reasons and the chain roll-up (§10)."),
    "publish": ("W3", "Publish the scores (§11)."),
    "monitor": ("W6", "Drift and performance signals, the status file and the run summary (§12)."),
}


def _stub(name: str, package: str, doc: str) -> Callable[..., None]:
    def command(
        run_id: str | None = RUN_ID_OPT,
        env: str = ENV_OPT,
        env_file: Path | None = ENV_FILE_OPT,
        ml_config: Path | None = ML_CONFIG_OPT,
    ) -> None:
        typer.echo(
            f"not built yet: `ml {name}` is work package {package} (docs/ML_PIPELINE_SPEC.md §18.3)",
            err=True,
        )
        raise typer.Exit(EXIT_CONFIG)

    command.__doc__ = f"{doc} Not built yet ({package})."
    return command


for _name, (_package, _doc) in _STUBS.items():
    ml_app.command(_name)(_stub(_name, _package, _doc))
