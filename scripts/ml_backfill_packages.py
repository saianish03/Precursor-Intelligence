"""Copy the packages of versions registered before #75 to the package store (docs/ML_PIPELINE_SPEC.md §9.4).

    uv run python scripts/ml_backfill_packages.py [--env prod] [--env-file <env yaml>] [--ml-config <ml yaml>]

Since #75 `gate` copies a run's train.json and its selected refit and eval packages to the package store before it
registers the version, and `score`, `evaluate` and `promote` load only from there. Versions registered earlier have
no copy, so they refuse until this has run once. For every run whose gate registered a version (ml_runs), it copies
the same files the same way: each checked against the sha256 its train.json recorded, never overwriting. Safe to
rerun: a run already copied writes nothing. Run it against the same env as the stack (PI_BUCKET_URI,
PI_PACKAGES_URI, PI_DATABASE_URL), before the stack's next score or promote.
"""

import argparse
from pathlib import Path

from precursorintelligence.common.config import load_ml_config, load_settings
from precursorintelligence.common.io.registry import build_package_store
from precursorintelligence.evaluation.packages import backfill_packages, stable_prefix
from precursorintelligence.ingestion.context import build_context
from precursorintelligence.training import MlInputError


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--env", default="prod", help="configs/<env>.yaml (default prod)")
    p.add_argument("--env-file", type=Path, help="an env yaml instead of configs/<env>.yaml")
    p.add_argument("--ml-config", type=Path, help="ML settings file (default configs/ml.yaml)")
    a = p.parse_args()

    cfg = load_ml_config(a.ml_config)
    settings = load_settings(a.env, env_path=a.env_file)
    ctx = build_context(settings, run_id="backfill-packages")
    packages = build_package_store(settings)
    try:
        copied = backfill_packages(ctx.object_store, packages, ctx.metadata, cfg)
    except MlInputError as exc:
        raise SystemExit(f"backfill refused: {exc}") from None
    if not copied:
        print("no registered versions in ml_runs: nothing to copy")
    for run, keys in copied.items():
        where = f"{packages.root_uri.rstrip('/')}/{stable_prefix(cfg, run)}"
        print(f"{run}: {len(keys)} file(s) copied to {where}" if keys else f"{run}: already copied")


if __name__ == "__main__":
    main()
