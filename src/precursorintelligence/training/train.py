"""Task ``train``: fit, calibrate, select, refit (docs/ML_PIPELINE_SPEC.md §3.3, §4, §5).

Reads only the run folder ``prepare`` wrote (never gold, never the test frame):

1. **Eval fit** on ``eval_train``: each member (LightGBM, LR) once on the whole window and once per facility fold
   (OOF, §3.4); the candidates' raw log-odds (ensemble = weighted mean of the members', ML-16); Platt per candidate
   on the OOF rows of the window's newest months (§5); the OOF bias check and the selection rule (§4.3).
2. **Refit** of the selected candidate on ``refit_train``, again with OOF fits (its Platt and G6).
3. Writes one eval-fit package per candidate and the refit package (``RunPaths.package``), the OOF parquets
   (``RunPaths.oof``), ``selection.json`` and ``train.json``; logs one MLflow run ``train`` when tracking is on.

It also measures the training-vs-scoring PSI of the regime-sensitive features (§1.4, ML-21) and the wall time of
each stage (§4.4).
"""

from __future__ import annotations

import io
import json
import logging
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from pydantic import ValidationError

from ..common.config import FeatureRegistry, FeatureSpec, MlConfig
from ..common.io.object_store import ObjectStore
from . import MlInputError
from .calibration import apply_platt, diagnostics, fit_platt, recent_months, sigmoid
from .contracts import (
    CalibrationDiagnostics,
    Candidate,
    Decision,
    DecisionName,
    FitName,
    InputManifest,
    Member,
    Mode,
    PlattCalibration,
    Psi,
    RunPaths,
    Selection,
    TrainResult,
    Windows,
    oof_columns,
)
from .data import (
    MISSING_LEVEL,
    feature_list,
    read_frame,
    read_input_manifest,
    read_windows,
    to_model_frame,
)
from .ensemble import PrecursorModel, build_meta, candidate_weights, combine, write_package
from .members import LgbmMember, LrMember, categorical_levels
from .select import BiasCheck, BiasOutcome, default_bias_check, select_candidate

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- features


def resolve_features(
    registry: FeatureRegistry, cfg: MlConfig, manifest: InputManifest
) -> list[FeatureSpec]:
    """The feature list prepare recorded, rebuilt from the registry: a registry or ml.yaml changed since prepare
    is a data refusal (rerun prepare), never a silently different model."""
    features = feature_list(registry, cfg)
    names = [f.name for f in features]
    if names != manifest.features:
        raise MlInputError(
            "the feature list differs from the one prepare recorded in input/manifest.json: rerun prepare "
            f"(now {len(names)} features, prepare {len(manifest.features)})"
        )
    return features


def members_of(candidates: Iterable[Candidate], cfg: MlConfig) -> list[Member]:
    need: set[str] = set()
    for c in candidates:
        need |= set(candidate_weights(c, cfg.ensemble.weights))
    order: tuple[Member, ...] = ("lgbm", "lr")
    return [m for m in order if m in need]


# --------------------------------------------------------------------------- fitting one window


@dataclass
class WindowFit:
    """The members fitted on one window, their OOF log-odds and the window's rows."""

    fit: FitName
    months: list[str]
    keys: pl.DataFrame  # ccn, t, fold, y
    y: np.ndarray
    full: dict[str, LgbmMember | LrMember]
    oof: dict[str, np.ndarray]
    levels: dict[str, list[str]]
    seconds: dict[str, float] = field(default_factory=dict)


def _fit_member(
    member: Member,
    x: pd.DataFrame,
    y: np.ndarray,
    features: Sequence[FeatureSpec],
    cfg: MlConfig,
    levels: dict[str, list[str]] | None = None,
) -> LgbmMember | LrMember:
    if member == "lgbm":
        return LgbmMember.fit(x, y, features, cfg.members.lgbm, levels=levels)
    return LrMember.fit(x, y, features, cfg.members.lr)


def fit_window(
    fit: FitName,
    frame: pl.DataFrame,
    features: Sequence[FeatureSpec],
    cfg: MlConfig,
    members: Sequence[Member],
    months: Sequence[str],
) -> WindowFit:
    """Each member on the whole window, and once per fold on the other folds to predict that fold (OOF). Every
    facility is in one fold (``fold`` from prepare), so an OOF prediction never comes from a fit that saw it."""
    label = cfg.label
    if frame.is_empty():
        raise MlInputError(f"the {fit} frame is empty")
    if frame[f"status_{label}"].is_in(["censored", "immature"]).any():
        raise MlInputError(f"the {fit} frame holds immature or censored rows")
    x = to_model_frame(frame, features)
    y = frame[f"y_{label}"].cast(pl.Float64).to_numpy()
    folds = frame["fold"].to_numpy()
    levels = categorical_levels(x, features)
    full: dict[str, LgbmMember | LrMember] = {}
    oof: dict[str, np.ndarray] = {}
    seconds: dict[str, float] = {}
    for m in members:
        start = time.perf_counter()
        full[m] = _fit_member(m, x, y, features, cfg, levels)
        seconds[f"{fit}.{m}.full"] = time.perf_counter() - start
        start = time.perf_counter()
        pred = np.full(len(y), np.nan)
        for k in np.unique(folds):
            held = folds == k
            if held.all() or len(np.unique(y[~held])) < 2:
                raise MlInputError(f"{fit} fold {k} leaves no usable training rows")
            member = _fit_member(m, x[~held].reset_index(drop=True), y[~held], features, cfg)
            pred[held] = member.logit(x[held].reset_index(drop=True))
        oof[m] = pred
        seconds[f"{fit}.{m}.oof"] = time.perf_counter() - start
        log.info(
            "%s fit: %s full + %d folds in %.1fs",
            fit,
            m,
            len(np.unique(folds)),
            seconds[f"{fit}.{m}.full"] + seconds[f"{fit}.{m}.oof"],
        )
    keys = frame.select(
        "ccn", "t", pl.col("fold").cast(pl.Int8), pl.col(f"y_{label}").cast(pl.Int8).alias("y")
    )
    return WindowFit(fit, list(months), keys, y, full, oof, levels, seconds)


@dataclass(frozen=True)
class CandidateFit:
    candidate: Candidate
    weights: dict[str, float]
    logit_raw: np.ndarray  # OOF
    platt: PlattCalibration
    diagnostics: list[CalibrationDiagnostics]

    @property
    def logit(self) -> np.ndarray:
        return apply_platt(self.platt, self.logit_raw)

    @property
    def p(self) -> np.ndarray:
        return sigmoid(self.logit)


def calibrate_candidate(candidate: Candidate, wf: WindowFit, cfg: MlConfig) -> CandidateFit:
    """The candidate's OOF log-odds and its Platt fit on the newest ``calibration.fit_recent_months`` months (§5)."""
    weights = candidate_weights(candidate, cfg.ensemble.weights)
    raw = combine({m: wf.oof[m] for m in weights}, weights)
    months = recent_months(wf.months, cfg.calibration.fit_recent_months)
    mask = wf.keys["t"].is_in(months).to_numpy()
    platt = fit_platt(raw[mask], wf.y[mask], months)
    bins = cfg.evaluation.ece_bins
    diags = [
        diagnostics(candidate, "fit_months", raw[mask], wf.y[mask], platt, bins=bins),
        diagnostics(
            candidate, "all_oof", raw, wf.y, platt, isotonic_fit=(raw[mask], wf.y[mask]), bins=bins
        ),
    ]
    return CandidateFit(candidate, weights, raw, platt, diags)


def package_for(
    cf: CandidateFit,
    wf: WindowFit,
    features: Sequence[FeatureSpec],
    recipe_hash: str,
) -> PrecursorModel:
    meta = build_meta(
        cf.candidate,
        features,
        wf.levels,
        cf.weights,
        {wf.fit: wf.months},
        recipe_hash,
    )
    lgbm = wf.full.get("lgbm") if "lgbm" in cf.weights else None
    lr = wf.full.get("lr") if "lr" in cf.weights else None
    return PrecursorModel(
        meta,
        cf.platt,
        lgbm=lgbm if isinstance(lgbm, LgbmMember) else None,
        lr=lr if isinstance(lr, LrMember) else None,
    )


def oof_frame(wf: WindowFit, fits: Sequence[CandidateFit]) -> pl.DataFrame:
    cols: dict[str, Any] = {}
    for cf in fits:
        raw_c, logit_c, p_c = oof_columns(cf.candidate)
        cols[raw_c], cols[logit_c], cols[p_c] = cf.logit_raw, cf.logit, cf.p
    return wf.keys.with_columns([pl.Series(k, v, dtype=pl.Float64) for k, v in cols.items()])


# --------------------------------------------------------------------------- PSI (§1.4)


def psi(
    reference: pl.Series, current: pl.Series, bins: int = 10, categorical: bool = False
) -> float:
    """Population stability index of ``current`` against ``reference``: 10 quantile bins of the reference (nulls as
    their own bin), or one bin per level for a categorical; shares floored at 1e-4."""
    eps = 1e-4
    if categorical:
        ref = reference.cast(pl.String).fill_null(MISSING_LEVEL).to_list()
        cur = current.cast(pl.String).fill_null(MISSING_LEVEL).to_list()
        levels = sorted(set(ref) | set(cur))
        r = np.array([ref.count(v) for v in levels], float)
        c = np.array([cur.count(v) for v in levels], float)
    else:
        ref_np = reference.cast(pl.Float64).to_numpy()
        cur_np = current.cast(pl.Float64).to_numpy()
        finite = ref_np[~np.isnan(ref_np)]
        if finite.size == 0:
            return float("nan")
        edges = np.unique(np.quantile(finite, np.linspace(0, 1, bins + 1)[1:-1]))

        def counts(v: np.ndarray) -> np.ndarray:
            nan = np.isnan(v)
            idx = np.searchsorted(edges, v[~nan], side="right")
            out = np.bincount(idx, minlength=len(edges) + 1).astype(float)
            return np.asarray(np.r_[out, float(nan.sum())], dtype=float)

        r, c = counts(ref_np), counts(cur_np)
    rs = np.maximum(r / max(r.sum(), 1.0), eps)
    cs = np.maximum(c / max(c.sum(), 1.0), eps)
    return float(np.sum((cs - rs) * np.log(cs / rs)))


def regime_psi(
    refit: pl.DataFrame, score: pl.DataFrame, registry: FeatureRegistry, cfg: MlConfig
) -> list[Psi]:
    """PSI of each ``features.regime_sensitive`` feature, refit window vs the score month (excluded or not: it is
    the evidence for keeping them excluded)."""
    dtypes = {f.name: f.dtype for f in registry.features}
    threshold = cfg.monitor.feature_psi_max
    out = []
    for name in cfg.features.regime_sensitive:
        if name not in refit.columns or name not in score.columns or score.is_empty():
            continue
        value = psi(refit[name], score[name], cfg.monitor.psi_bins, dtypes.get(name) == "str")
        out.append(
            Psi(
                feature=name,
                psi=round(value, 6),
                threshold=threshold,
                above=bool(value > threshold),
            )
        )
    return out


# --------------------------------------------------------------------------- the task


def _put(store: ObjectStore, key: str, data: bytes) -> None:
    store.put_bytes(key, data, overwrite=True)


def _parquet(df: pl.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.write_parquet(buf)
    return buf.getvalue()


@dataclass(frozen=True)
class TrainOutput:
    result: TrainResult
    selection: Selection
    eval_models: dict[str, PrecursorModel]
    refit_model: PrecursorModel


def run_train(
    store: ObjectStore,
    *,
    ml_run_id: str,
    cfg: MlConfig,
    registry: FeatureRegistry,
    bias_check: BiasCheck | None = None,
    recipe_hash: str | None = None,
    ml_config_path: Path | None = None,
    mode: Mode = "auto",
) -> TrainOutput:
    """The ``train`` task on a prepared run folder. ``ml_config_path`` is the ml.yaml that ``cfg`` came from (it
    goes into the ``recipe_hash``, §14.3)."""
    from . import mlflow_log

    started = time.perf_counter()
    paths = RunPaths.of(cfg, ml_run_id)
    manifest = read_input_manifest(store, paths)
    windows: Windows = read_windows(store, paths)
    if manifest.label != cfg.label:
        raise MlInputError(f"prepare ran for label {manifest.label}, ml.yaml says {cfg.label}")
    features = resolve_features(registry, cfg, manifest)
    rhash = recipe_hash or mlflow_log.recipe_hash(features, ml_config_path=ml_config_path)
    decision = read_decision(store, paths)  # before any fit: a corrupt decision.json refuses early
    check = bias_check or default_bias_check()
    seconds: dict[str, float] = {}

    # ---- eval fit: all candidates
    start = time.perf_counter()
    eval_frame = read_frame(store, paths, "eval_train")
    candidates = list(cfg.selection.candidates)
    wf_eval = fit_window(
        "eval", eval_frame, features, cfg, members_of(candidates, cfg), windows.eval.months
    )
    seconds.update(wf_eval.seconds)
    eval_fits = {c: calibrate_candidate(c, wf_eval, cfg) for c in candidates}
    bias: dict[Candidate, BiasOutcome] = {
        c: check(eval_frame, eval_fits[c].p, cfg) for c in candidates
    }
    selection = select_candidate(
        wf_eval.y,
        {c: f.p for c, f in eval_fits.items()},
        bias,
        cfg.selection.tie_margin,
        candidates,
    )
    eval_models: dict[str, PrecursorModel] = {
        c: package_for(eval_fits[c], wf_eval, features, rhash) for c in candidates
    }
    seconds["eval.total"] = time.perf_counter() - start
    del eval_frame

    # ---- refit: the selected candidate
    start = time.perf_counter()
    chosen = selection.selected
    refit_frame = read_frame(store, paths, "refit_train")
    wf_refit = fit_window(
        "refit", refit_frame, features, cfg, members_of([chosen], cfg), windows.refit.months
    )
    seconds.update(wf_refit.seconds)
    refit_fit = calibrate_candidate(chosen, wf_refit, cfg)
    refit_model = package_for(refit_fit, wf_refit, features, rhash)
    score_frame = read_frame(store, paths, "score")
    psis = regime_psi(refit_frame, score_frame, registry, cfg)
    seconds["refit.total"] = time.perf_counter() - start
    del refit_frame

    # ---- write the run folder
    eval_packages: dict[str, str] = {c: paths.package("eval", c) for c in candidates}
    for c, model in eval_models.items():
        write_package(store, eval_packages[c], model)
    refit_package = paths.package("refit", chosen)
    write_package(store, refit_package, refit_model)
    _put(store, paths.oof("eval"), _parquet(oof_frame(wf_eval, list(eval_fits.values()))))
    _put(store, paths.oof("refit"), _parquet(oof_frame(wf_refit, [refit_fit])))
    _put(store, paths.selection, selection.model_dump_json(indent=2).encode("utf-8"))

    calibration = {f"eval/{c}": f.platt for c, f in eval_fits.items()}
    calibration[f"refit/{chosen}"] = refit_fit.platt
    flags = [f"platt_identity:{k}" for k, v in calibration.items() if v.identity]
    flags += [f"psi:{p.feature}" for p in psis if p.above]
    flags += [
        f"bias_check_failed:{r.candidate}" for r in selection.candidates if not r.bias_check_passed
    ]
    fit_diags = [("eval", d) for f in eval_fits.values() for d in f.diagnostics]
    fit_diags += [("refit", d) for d in refit_fit.diagnostics]
    diags = [d for _, d in fit_diags]

    # ---- MLflow (one run for the task, §14.2); without a tracking URI this block only writes train.json
    eval_uris: dict[str, str] = {c: "" for c in candidates}
    refit_uri = ""
    tags = mlflow_log.base_tags(
        manifest,
        ml_run_id=ml_run_id,
        recipe_hash=rhash,
        decision=decision,
        mode=mode,
    )
    with mlflow_log.task_run(cfg, "train", tags) as run_id:
        if run_id is not None:
            mlflow_log.log_gold_input(
                manifest, store.uri_for(paths.input_file("facility_month_dataset.parquet"))
            )
            mlflow_log.log_params(train_params(cfg, windows, manifest, calibration, chosen))
            sample = to_model_frame(score_frame.head(50), features)
            for c, model in eval_models.items():
                eval_uris[c] = mlflow_log.log_package(model, f"eval_{c}", sample)
            refit_uri = mlflow_log.log_package(refit_model, f"refit_{chosen}", sample)
            mlflow_log.log_json_artifact(selection, "selection.json")
            mlflow_log.log_json_artifact(windows, "windows.json")
            mlflow_log.log_json_artifact(manifest, "input_manifest.json")
            mlflow_log.log_json_artifact(
                {"features": [f.name for f in features]}, "feature_list.json"
            )
        seconds["total"] = time.perf_counter() - started
        result = TrainResult(
            ml_run_id=ml_run_id,
            gold_run_id=manifest.gold_run_id,
            dataset_version=manifest.dataset_version,
            recipe_hash=rhash,
            mlflow_run_id=run_id,
            selected=chosen,
            eval_packages=eval_packages,
            eval_model_uris=eval_uris,
            refit_package=refit_package,
            refit_model_uri=refit_uri,
            oof={"eval": paths.oof("eval"), "refit": paths.oof("refit")},
            calibration=calibration,
            calibration_diagnostics=diags,
            flags=flags,
            psi=psis,
            features=[f.name for f in features],
            excluded_features=list(cfg.features.exclude),
            seconds={k: round(v, 3) for k, v in seconds.items()},
        )
        _put(store, paths.train_result, result.model_dump_json(indent=2).encode("utf-8"))
        if run_id is not None:
            mlflow_log.log_metrics(train_metrics(selection, fit_diags, psis, seconds))
            mlflow_log.log_json_artifact(result, "train.json")
    log.info("train %s: selected %s; %s", ml_run_id, chosen, selection.reason)
    return TrainOutput(result, selection, eval_models, refit_model)


def read_decision(store: ObjectStore, paths: RunPaths) -> DecisionName:
    """The run's decision for the §14.2 tag: ``decision.json`` when ``decide`` (W6) wrote one, else RETRAIN (a
    manual ``ml train`` is a retrain). A file that is there but unreadable is a data refusal, not a silent RETRAIN."""
    if not store.exists(paths.decision):
        return "RETRAIN"
    with store.open_read(paths.decision) as f:
        data = f.read()
    try:
        return Decision.model_validate_json(data).decision
    except ValidationError as exc:
        raise MlInputError(
            f"{paths.decision} is not a valid decision file: {exc.error_count()} error(s)"
        ) from None


def train_params(
    cfg: MlConfig,
    w: Windows,
    manifest: InputManifest,
    calibration: dict[str, PlattCalibration],
    chosen: Candidate,
) -> dict[str, Any]:
    return {
        "members.lgbm": cfg.members.lgbm.as_dict(),
        "members.lr": cfg.members.lr.model_dump(),
        "ensemble.weights": dict(cfg.ensemble.weights),
        "selected": chosen,
        "platt": {k: {"a": v.a, "b": v.b, "identity": v.identity} for k, v in calibration.items()},
        "train_months": cfg.windows.train_months,
        "eval_gap": w.eval_gap,
        "eval_window": f"{w.eval.first}..{w.eval.last}",
        "refit_window": f"{w.refit.first}..{w.refit.last}",
        "test_window": f"{w.test.first}..{w.test.last}",
        "n_features": len(manifest.features),
        "features_sha256": manifest.features_sha256,
        "excluded_features": json.dumps(list(cfg.features.exclude)),
        "folds": cfg.folds.k,
    }


def train_metrics(
    selection: Selection,
    diags: Sequence[tuple[str, CalibrationDiagnostics]],
    psis: Sequence[Psi],
    seconds: dict[str, float],
) -> dict[str, float]:
    out: dict[str, float] = {}
    for r in selection.candidates:
        out[f"oof_pr_auc_{r.candidate}"] = r.oof_pr_auc
        out[f"oof_brier_{r.candidate}"] = r.oof_brier
    for fit, d in diags:
        prefix = f"cal_{fit}_{d.candidate}_{d.rows}"
        for k in (
            "brier_raw",
            "brier_platt",
            "brier_isotonic",
            "ece_raw",
            "ece_platt",
            "ece_isotonic",
        ):
            out[f"{prefix}_{k}"] = float(getattr(d, k))
    for p in psis:
        out[f"psi_{p.feature}"] = p.psi
    for k, v in seconds.items():
        out[f"seconds_{k.replace('.', '_')}"] = v
    return out
