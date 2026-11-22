"""``precursor-pipeline agents <command>``: the agents' batch jobs (docs/AGENTS_PLAN.md), run by the DAG
``cms_nh_agents`` (L10). Exit codes follow the pipeline contract: 0 ok, 1 error (retried), 2 data refusal, 3 config
or usage error.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer

from ..common.config import load_agents_config, load_ml_config
from ..ingestion.cli import ENV_FILE_OPT, ENV_OPT, RUN_ID_OPT
from ..training.cli import LOG_LEVEL_OPT, ML_CONFIG_OPT, _run_task

agents_app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="The agents' batch jobs (docs/AGENTS_PLAN.md)",
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
        from .build import run_build_evidence

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
