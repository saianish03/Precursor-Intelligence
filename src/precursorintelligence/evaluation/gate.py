"""The ``gate`` task (docs/ML_PIPELINE_SPEC.md §9): apply G1-G7 to ``evaluation.json``, register the refit,
move the aliases, write the ``ml_runs`` row, ``champion.json`` and the model card.

- All gates pass, and a champion exists: the version becomes ``champion``. Any gate fails: it is ``challenger``
  and the champion stays (a **hold**, exit 0).
- First run (no champion): G5 and G7(a) skip; the version is ``challenger`` and a human promotes it after
  Checkpoint ML-A (``gates.first_run: challenger``). ``first_run: promote`` is for the toy smoke run only.
- Registration is idempotent (§9.2): a retried task finds the version tagged with its ``ml_run_id``, and an
  ``ml_runs`` row already written for the run is not written again.
- Without MLflow tracking nothing is registered: the gates still run and are recorded, with no version.
"""

from __future__ import annotations

import io
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
import polars as pl
from scipy.stats import spearmanr

from ..common.config import MlConfig
from ..common.io.metadata_store import MetadataStore
from ..common.io.object_store import ObjectStore
from ..training import MlInputError
from ..training.contracts import Champion, InputManifest, RunPaths, TrainResult, Windows
from ..training.data import (
    average_precision,
    gold_config_sha,
    read_frame,
    read_input_manifest,
    read_windows,
)
from ..training.ensemble import read_package
from ..training.train import read_decision
from . import gates
from .evaluate import CHAMPION, RULE, report_prefix
from .gates import GateReport, GateResult
from .model_card import render_model_card
from .registry import RegisteredVersion, Registry
from .report import EvaluationReport

log = logging.getLogger(__name__)

RECENT_MONTHS = (
    3  # G6: the refit window's newest months whose observed rate the score's mean is held to
)
GATE_ACTIONS = ("promote", "hold")


def _read(store: ObjectStore, key: str, what: str) -> bytes:
    if not store.exists(key):
        raise MlInputError(f"{key} is missing: run `ml {what}` first")
    with store.open_read(key) as f:
        return f.read()


def _oof_lift(store: ObjectStore, key: str, candidate: str) -> float:
    oof = pl.read_parquet(io.BytesIO(_read(store, key, "train")))
    y = oof["y"].cast(pl.Float64).to_numpy()
    return average_precision(y, oof[f"p_{candidate}"].to_numpy()) / float(y.mean())


@dataclass(frozen=True)
class RefitCheck:
    refit_oof_lift: float
    eval_oof_lift: float
    score_mean_p: float
    recent_observed_rate: float
    spearman: float


def refit_check(
    store: ObjectStore, paths: RunPaths, train: TrainResult, windows: Windows, cfg: MlConfig
) -> RefitCheck:
    """G6's numbers: OOF lift of the refit vs the eval fit; on the score frame, the refit's mean score, the refit
    window's newest observed rate, and the Spearman correlation of the refit's and the eval fit's scores."""
    chosen = train.selected
    score = read_frame(store, paths, "score")
    refit_p = read_package(store, train.refit_package).predict_frame(score)["p"].to_numpy()
    eval_p = read_package(store, train.eval_packages[chosen]).predict_frame(score)["p"].to_numpy()
    refit_frame = read_frame(store, paths, "refit_train")
    recent = sorted(set(refit_frame["t"].to_list()))[-RECENT_MONTHS:]
    rate = refit_frame.filter(pl.col("t").is_in(recent))[f"y_{cfg.label}"].cast(pl.Float64).mean()
    rho = spearmanr(refit_p, eval_p).statistic
    return RefitCheck(
        refit_oof_lift=_oof_lift(store, train.oof["refit"], chosen),
        eval_oof_lift=_oof_lift(store, train.oof["eval"], chosen),
        score_mean_p=float(np.mean(refit_p)),
        recent_observed_rate=float(rate),  # type: ignore[arg-type]
        spearman=float(rho),
    )


def apply_gates(report: EvaluationReport, refit: RefitCheck, cfg: MlConfig) -> GateReport:
    """G1-G7 on the evaluation's numbers (§9.1)."""
    chosen = report.selected
    prob = report.candidates[chosen].probability or {}
    rule = report.comparison("pr_auc", chosen, RULE)
    if rule is None:
        raise MlInputError("evaluation.json has no PR-AUC comparison with the prior-G+ rule")
    has_champion = report.champion is not None
    results: list[GateResult] = [
        gates.g1_data(gold_verified=report.gold_verified, allow_unverified=report.allow_unverified),
        gates.g2_sanity(
            uncleared_leakage_flags=report.sanity.uncleared_leakage_flags,
            shap_additive=report.sanity.shap_additive,
            shuffled_label_roc=report.sanity.shuffled_label_roc,
            roc_range=cfg.evaluation.shuffled_label.roc_range,
        ),
        gates.g3_beats_rule(rule),
        gates.g4_calibration(
            ece=float(prob["ece"]), slope=float(prob["calibration_slope"]), cfg=cfg
        ),
        gates.g5_vs_champion(
            pr_auc_diff=report.comparison("pr_auc", chosen, CHAMPION) if has_champion else None,
            brier_diff=report.comparison("brier", chosen, CHAMPION) if has_champion else None,
            cfg=cfg,
        ),
        gates.g6_refit(
            refit_oof_lift=refit.refit_oof_lift,
            eval_oof_lift=refit.eval_oof_lift,
            score_mean_p=refit.score_mean_p,
            recent_observed_rate=refit.recent_observed_rate,
            spearman=refit.spearman,
            cfg=cfg,
        ),
        gates.g7_bias(
            calibration_failures=report.calibration_failures,
            vs_champion=report.slices_vs_champion if has_champion else None,
            cfg=cfg,
        ),
    ]
    return GateReport(results=results)


def version_tags(
    report: EvaluationReport, train: TrainResult, manifest: InputManifest, outcome: GateReport
) -> dict[str, str]:
    """The registry version's tags (§9.2)."""
    return {
        "ml_run_id": report.ml_run_id,
        "recipe_hash": report.recipe_hash,
        "candidate": report.selected,
        "eval_run_id": train.mlflow_run_id or "",
        "eval_model_uri": train.eval_model_uris.get(report.selected, ""),
        "gold_run_id": manifest.gold_run_id,
        "dataset_version": manifest.dataset_version,
        "training_dataset": manifest.dataset_version,
        "v_labels": manifest.v_labels,
        "gold_verified": "true" if manifest.verified else "false",
        "gate_outcome": "pass" if outcome.passed else "hold",
        "failed_gates": ",".join(outcome.failed),
        "sanity_passed": "true" if _sanity_passed(outcome) else "false",
    }


def _sanity_passed(outcome: GateReport) -> bool:
    """G1 and G2 passed: the only versions ``promote`` may move to by hand (§9.4)."""
    return all(r.status != "fail" for r in outcome.results if r.gate in ("G1", "G2"))


@dataclass(frozen=True)
class GateOutput:
    gates: GateReport
    action: str  # promote | hold
    version: int | None
    alias: str | None
    champion_before: int | None
    champion_after: int | None
    model_card: str


def run_gate(
    store: ObjectStore,
    meta: MetadataStore,
    *,
    ml_run_id: str,
    cfg: MlConfig,
    registry: Registry | None = None,
    mode: str = "auto",
    actor: str = "cms_nh_ml",
) -> GateOutput:
    """The ``gate`` task on an evaluated run folder."""
    from ..training import mlflow_log
    from .plots import log_files

    paths = RunPaths.of(cfg, ml_run_id)
    report = EvaluationReport.model_validate_json(_read(store, paths.evaluation, "evaluate"))
    train = TrainResult.model_validate_json(_read(store, paths.train_result, "train"))
    manifest = read_input_manifest(store, paths)
    windows = read_windows(store, paths)
    if report.ml_run_id != ml_run_id or report.recipe_hash != train.recipe_hash:
        raise MlInputError("evaluation.json is from another run or recipe: run `ml evaluate` again")

    outcome = apply_gates(report, refit_check(store, paths, train, windows, cfg), cfg)
    before_v = registry.by_alias(cfg.registry.champion_alias) if registry else None
    champion_before = before_v.version if before_v else None

    version: RegisteredVersion | None = None
    alias: str | None = None
    promote = outcome.passed and (champion_before is not None or cfg.gates.first_run == "promote")
    if registry is not None:
        if not train.refit_model_uri:
            raise MlInputError("train logged no refit model to MLflow: nothing to register")
        tags = version_tags(report, train, manifest, outcome)
        version = registry.find_by_ml_run_id(ml_run_id)
        if version is None:
            version = registry.register(train.refit_model_uri, tags)
        else:
            registry.set_tags(version.version, tags)
        alias = cfg.registry.champion_alias if promote else cfg.registry.challenger_alias
        registry.set_alias(alias, version.version)
    champion_after = version.version if (promote and version) else champion_before
    action = "promote" if promote else "hold"

    _write_ml_runs(
        meta, report, manifest, outcome, action, mode, actor, store, paths,
        champion_before, champion_after, version.version if version else None, train,
    )  # fmt: skip
    champion = Champion(model_name=cfg.registry.model_name, version=champion_after, source="gate")
    store.put_bytes(paths.champion, champion.model_dump_json(indent=2).encode(), overwrite=True)
    card = render_model_card(
        report,
        outcome,
        train,
        windows,
        manifest,
        version=version.version if version else None,
        action=action,
    )
    store.put_bytes(f"{report_prefix(paths)}model_card.md", card.encode("utf-8"), overwrite=True)
    store.put_bytes(
        f"{report_prefix(paths)}gates.json",
        outcome.model_dump_json(indent=2).encode(),
        overwrite=True,
    )

    tags = mlflow_log.base_tags(
        manifest,
        ml_run_id=ml_run_id,
        recipe_hash=report.recipe_hash,
        decision=read_decision(store, paths),
    )
    with mlflow_log.task_run(cfg, "gate", tags) as run_id:
        if run_id is not None:
            mlflow_log.log_metrics(
                {f"gate_{r.gate}_passed": float(r.status != "fail") for r in outcome.results}
                | {"gates_passed": float(outcome.passed)}
            )
            mlflow_log.log_json_artifact(outcome.model_dump(mode="json"), "gates.json")
            log_files({"model_card.md": card.encode("utf-8")})
    log.info(
        "gate %s: %s (failed: %s); version %s -> %s",
        ml_run_id, action, outcome.failed or "none", version.version if version else None, alias,
    )  # fmt: skip
    return GateOutput(
        outcome,
        action,
        version.version if version else None,
        alias,
        champion_before,
        champion_after,
        card,
    )


def gate_reason(outcome: GateReport, action: str) -> str:
    if outcome.failed:
        return f"held: failed {', '.join(outcome.failed)}"
    if action == "promote":
        return "all gates passed"
    return "all gates passed; first run: challenger until Checkpoint ML-A (§17.3)"


def _write_ml_runs(
    meta: MetadataStore,
    report: EvaluationReport,
    manifest: InputManifest,
    outcome: GateReport,
    action: str,
    mode: str,
    actor: str,
    store: ObjectStore,
    paths: RunPaths,
    champion_before: int | None,
    champion_after: int | None,
    version: int | None,
    train: TrainResult,
) -> None:
    """One append-only ``ml_runs`` row (§9.3), unless a retried gate already wrote it for this run."""
    done = meta.read("ml_runs", {"ml_run_id": report.ml_run_id, "action": list(GATE_ACTIONS)})
    if done:
        log.info("ml_runs already holds the gate row of %s: not written again", report.ml_run_id)
        return
    decision = read_decision(store, paths)
    row: dict[str, Any] = {
        "ml_run_id": report.ml_run_id,
        "at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "mode": mode,
        "decision": decision,
        "action": action,
        "reason": gate_reason(outcome, action),
        "actor": actor,
        "gold_run_id": manifest.gold_run_id,
        "v": manifest.v,
        "v_labels": manifest.v_labels,
        "dataset_version": manifest.dataset_version,
        "gold_config_sha": gold_config_sha(manifest),
        "gold_verified": manifest.verified,
        "recipe_hash": report.recipe_hash,
        "champion_version_before": champion_before,
        "champion_version_after": champion_after,
        "candidate_version": version,
        "gate_outcome": "pass" if outcome.passed else "hold",
        "gate_results_json": outcome.model_dump_json(),
        "scores_published": False,
        "flags_json": json.dumps(train.flags),
        "retune_requested": False,
    }
    meta.append("ml_runs", [row])
