"""PR #49 reproduction check for W1 (docs/ML_PIPELINE_SPEC.md §17.2, §18.0 item 6).

Scores every eval-fit package of a trained run on the run's test frame and prints the headline numbers per test
month, with a facility-block bootstrap CI on PR-AUC. A report only: nothing here feeds a choice (selection already
happened on OOF predictions), and it never scores the refit (it trained on the test months, §3.3).

    uv run python scripts/ml_w1_reproduce.py --env-file <env.yaml> --run-id <run id> [--ml-config <ml.yaml>]

The expected result with ``windows.eval_gap: purge`` (eval window 2023-01..2023-11) on test 2025-02: PR-AUC within
0.465-0.499 (PR #49: 0.481), ROC ~0.71, Brier ~0.169, ECE ~0.03, P@5 % ~0.68 (§18.0).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

from precursorintelligence.common.config import MlEvaluationCfg, load_ml_config, load_settings
from precursorintelligence.ingestion.context import build_context
from precursorintelligence.training.calibration import brier, ece
from precursorintelligence.training.contracts import RunPaths, TrainResult
from precursorintelligence.training.data import average_precision, read_frame, read_windows
from precursorintelligence.training.ensemble import read_package


def precision_at(y: np.ndarray, p: np.ndarray, frac: float) -> float:
    n = max(1, int(len(y) * frac))
    return float(y[np.argsort(-p, kind="mergesort")[:n]].mean())


def block_bootstrap_ap(
    ccn: np.ndarray, y: np.ndarray, p: np.ndarray, reps: int, seed: int
) -> tuple[float, float]:
    """95 % CI of PR-AUC, resampling facilities (all of a facility's rows together)."""
    rng = np.random.default_rng(seed)
    codes, idx = np.unique(ccn, return_inverse=True)
    rows = [np.flatnonzero(idx == i) for i in range(len(codes))]
    stats = []
    for _ in range(reps):
        pick = np.concatenate([rows[i] for i in rng.integers(0, len(codes), len(codes))])
        stats.append(average_precision(y[pick], p[pick]))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def metrics(
    ccn: np.ndarray, y: np.ndarray, p: np.ndarray, ev: MlEvaluationCfg, reps: int
) -> dict[str, float]:
    """The test metrics, with ``evaluation.*`` of ml.yaml (bins, precision@k, bootstrap seed); ``reps`` is
    ``evaluation.bootstrap.replicates`` unless ``--replicates`` overrides it."""
    lo, hi = block_bootstrap_ap(ccn, y, p, reps, ev.bootstrap.seed)
    out = {
        "rows": float(len(y)),
        "base_rate": float(y.mean()),
        "pr_auc": average_precision(y, p),
        "pr_auc_ci_low": lo,
        "pr_auc_ci_high": hi,
        "roc_auc": float(roc_auc_score(y, p)),
        "brier": brier(y, p),
        "ece": ece(y, p, ev.ece_bins),
    }
    for frac in ev.precision_at:
        out[f"p_at_{frac * 100:g}"] = precision_at(y, p, frac)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--env-file", type=Path, default=None)
    ap.add_argument(
        "--env", default="prod", help="configs/<env>.yaml (with PI_BUCKET_URI for the ml-dev store)"
    )
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--ml-config", type=Path, default=None)
    ap.add_argument(
        "--replicates", type=int, default=None, help="default: evaluation.bootstrap.replicates"
    )
    a = ap.parse_args()
    settings = load_settings(a.env, env_path=a.env_file)
    store = build_context(settings, run_id=a.run_id).object_store
    cfg = load_ml_config(a.ml_config)
    reps = a.replicates if a.replicates is not None else cfg.evaluation.bootstrap.replicates
    paths = RunPaths.of(cfg, a.run_id)
    with store.open_read(paths.train_result) as f:
        result = TrainResult.model_validate_json(f.read())
    windows = read_windows(store, paths)
    test = read_frame(store, paths, "test")
    report: dict[str, object] = {
        "run_id": a.run_id,
        "eval_gap": windows.eval_gap,
        "eval_window": [windows.eval.first, windows.eval.last],
        "test_months": windows.test.months,
        "development_months": windows.development_months,
        "n_features": len(result.features),
        "selected": result.selected,
        "note": "report only: the candidate was selected on OOF predictions; these test numbers never change it",
        "candidates": {},
    }
    for cand, prefix in result.eval_packages.items():
        model = read_package(store, prefix)
        p = model.predict_frame(test)["p"].to_numpy()
        per_month = {}
        for month in windows.test.months:
            m = (test["t"] == month).to_numpy()
            per_month[month] = metrics(
                test["ccn"].to_numpy()[m],
                test[f"y_{cfg.label}"].cast(pl.Float64).to_numpy()[m],
                p[m],
                cfg.evaluation,
                reps,
            )
        key = f"{cand} (selected)" if cand == result.selected else cand
        report["candidates"][key] = per_month  # type: ignore[index]
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
