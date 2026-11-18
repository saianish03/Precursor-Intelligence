"""Checkpoint ML-A's reasons sample (docs/ML_PIPELINE_SPEC.md §17.3): the plain-language reasons of N random facilities
for a registered model version, e.g. the challenger awaiting promotion.

`ml score` explains only the champion, so before the first promotion this is how a person reads a challenger's
reasons ("20 random facilities' reasons read sensibly").

    uv run python scripts/ml_checkpoint_reasons.py --run-id <run that ran ml prepare> --version <v> \\
        [--env prod] [--env-file <env yaml>] [--ml-config <ml yaml>] [--n 20] [--seed 0] [--out reasons.csv]

It reads the way `ml score` does, with the same checks:
- the version's refit package, verified against its registering run's train.json (`load_verified_package`),
  refused unless its training months are the refit window;
- the run's pinned, hash-checked score frame, which must hold every model feature;
- contributions that add up to the calibrated log-odds (`SHAP_TOLERANCE`);
- the same reason labels and top-k settings.

Nothing is written to the bucket. The only metadata-database write is `build_context`'s `ensure_schema()`, which
creates missing tables. The table goes to stdout, and to --out as CSV.
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from precursorintelligence.common.config import load_ml_config, load_settings, project_root
from precursorintelligence.evaluation.registry import load_verified_package
from precursorintelligence.evaluation.sanity import SHAP_TOLERANCE
from precursorintelligence.ingestion.context import build_context
from precursorintelligence.serving import explain
from precursorintelligence.training.contracts import RunPaths
from precursorintelligence.training.data import read_frame, read_input_manifest


def main() -> None:
    p = argparse.ArgumentParser(
        description="Checkpoint ML-A reasons sample for a registered version"
    )
    p.add_argument(
        "--run-id", required=True, help="a run that ran `ml prepare` (its pinned score frame)"
    )
    p.add_argument(
        "--version", type=int, required=True, help="the registered model version to explain"
    )
    p.add_argument("--env", default="prod", help="configs/<env>.yaml (default prod)")
    p.add_argument("--env-file", type=Path, help="an env yaml instead of configs/<env>.yaml")
    p.add_argument("--ml-config", type=Path, help="ML settings file (default configs/ml.yaml)")
    p.add_argument("--n", type=int, default=20, help="facilities to sample (default 20)")
    p.add_argument("--seed", type=int, default=0, help="sampling seed (default 0)")
    p.add_argument("--out", type=Path, help="also write the table here as CSV")
    a = p.parse_args()

    cfg = load_ml_config(a.ml_config)
    ctx = build_context(load_settings(a.env, env_path=a.env_file), run_id=f"reasons-{a.run_id}")
    paths = RunPaths.of(cfg, a.run_id)
    manifest = read_input_manifest(ctx.object_store, paths)
    frame = read_frame(ctx.object_store, paths, "score")

    rows = np.sort(
        np.random.default_rng(a.seed).choice(
            frame.height, size=min(a.n, frame.height), replace=False
        )
    )
    sample = frame[rows.tolist()]
    model = load_verified_package(ctx.object_store, ctx.metadata, cfg, a.version, "refit")
    if model.meta is None or set(model.meta.training_months) != {"refit"}:
        raise SystemExit(f"version {a.version} is not a refit package")
    missing = [f for f in model.feature_names if f not in frame.columns]
    if missing:
        raise SystemExit(f"the score frame lacks model features: {missing}")
    pred = model.predict_frame(sample)
    base, contrib = model.explain(sample)
    err = float(np.abs(base + contrib.to_numpy().sum(axis=1) - pred["logit"].to_numpy()).max())
    if err > SHAP_TOLERANCE:
        raise SystemExit(
            f"contributions do not add up to the calibrated log-odds (max error {err:.2e})"
        )
    table = explain.reasons(
        contrib,
        sample,
        score_month=date(int(manifest.v[:4]), int(manifest.v[5:7]), 1),
        model_version=a.version,
        gold_run_id=manifest.gold_run_id,
        labels=explain.load_reason_labels(project_root() / cfg.explain.labels),
        top_raise=cfg.explain.top_raise,
        top_lower=cfg.explain.top_lower,
    )
    with pl.Config(tbl_rows=-1, tbl_cols=-1, fmt_str_lengths=80, tbl_width_chars=200):
        print(table)
    if a.out:
        table.write_csv(a.out)
        print(f"wrote {table.height} reason rows for {sample.height} facilities to {a.out}")


if __name__ == "__main__":
    main()
