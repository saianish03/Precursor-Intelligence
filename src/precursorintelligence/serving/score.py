"""The ``score`` task (docs/ML_PIPELINE_SPEC.md §10, §9.4, §8.2): every facility at ``t = V`` with the champion.

1. Read the run's ``champion.json``: the **explicit** version to load (never whatever ``@champion`` says later).
   No champion (no file, or ``version: null``) -> skip, nothing written.
2. Integrity check (§9.4): ``@champion`` must equal the version of the latest promote / manual_promote / rollback
   row in ``ml_runs`` (``evaluation/registry.check_champion_integrity``, exit 2 otherwise). The version loaded must
   be that one too, and a refit package. It is read from the GCS run folder of the run that registered it and
   checked against the sha256 in that run's ``train.json`` (``load_champion_package``); MLflow gives only the
   alias.
3. Score the pinned ``score`` frame (refused unless every row is ``t = V``); exact contributions (§8.1).
4. Write ``<prefix>scores/v=YYYY-MM/``: ``facility_scores``, ``facility_reasons``, ``shap_values``,
   ``chain_scores`` and, last, ``scores_manifest.json`` (§10.4). A rescore overwrites the month.
5. The champion's global importance on the score frame (§8.2) into the run folder's ``score/``; the first score
   of a version sets its top ``explain.key_features`` as the version's ``key_features`` tag (drift monitoring,
   §12.1), kept for the version's life. ``score/key_features.json`` is that fixed list, so ``monitor`` keeps
   watching a feature that drifts out of the month's top features.

Every score row carries the model version and the gold run that produced it (§20).
"""

from __future__ import annotations

import io
import json
import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any, Literal

import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict, ValidationError

from ..common.config import (
    ConfigError,
    FeatureRegistry,
    MlConfig,
    load_feature_registry,
    project_root,
)
from ..common.io.metadata_store import MetadataStore
from ..common.io.object_store import ObjectStore
from ..evaluation.registry import Registry, check_champion_integrity, load_verified_package
from ..evaluation.sanity import SHAP_TOLERANCE
from ..training import MlInputError
from ..training.contracts import Champion, InputManifest, Mode, RunPaths
from ..training.data import (
    is_forbidden,
    read_frame,
    read_input_manifest,
    read_windows,
)
from ..training.ensemble import PrecursorModel
from . import explain
from .rollup import chain_scores

log = logging.getLogger(__name__)

SCORE_FILES = ("facility_scores", "facility_reasons", "shap_values", "chain_scores")
SCORES_MANIFEST = "scores_manifest.json"


class ScoresManifest(BaseModel):
    """``scores_manifest.json`` (§10.4): what produced the month's files, and each file's rows and sha256."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    v: str
    v_labels: str
    label: str
    model_name: str
    model_version: int
    candidate: str
    gold_run_id: str
    dataset_version: str
    gold_verified: bool
    stale_refresh: bool
    ml_run_id: str
    scored_at: str
    files: dict[str, str]  # name -> object key
    rows: dict[str, int]
    sha256: dict[str, str]


def scores_dir(cfg: MlConfig, v: str) -> str:
    """``<root_prefix><scores_prefix>/v=YYYY-MM/`` (§10.4)."""
    return f"{cfg.outputs.root_prefix}{cfg.outputs.scores_prefix}/v={v}/"


def score_report_dir(paths: RunPaths) -> str:
    return f"{paths.run_dir}score/"


def read_champion(store: ObjectStore, paths: RunPaths) -> Champion | None:
    """The run's ``champion.json``, or None when no task wrote one."""
    if not store.exists(paths.champion):
        return None
    with store.open_read(paths.champion) as f:
        data = f.read()
    try:
        return Champion.model_validate_json(data)
    except ValidationError as exc:
        raise MlInputError(
            f"{paths.champion} is not a valid champion file: {exc.error_count()} error(s)"
        ) from None


def load_champion_package(
    store: ObjectStore, meta: MetadataStore, cfg: MlConfig, version: int
) -> PrecursorModel:
    """The champion's refit package, verified against its registering run's ``train.json`` (§9.4); see
    ``evaluation.registry.load_verified_package``. Any mismatch refuses (exit 2)."""
    return load_verified_package(store, meta, cfg, version, "refit")


# --------------------------------------------------------------------------- the score table


def _month_start(v: str) -> date:
    return date(int(v[:4]), int(v[5:7]), 1)


def rank_columns(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``risk_percentile`` (0-100, higher = riskier; ties share the average) and ``risk_decile`` (1 = the riskiest
    tenth; ties share the better decile), both within the month. Ranks, never probabilities (§20)."""
    n = len(p)
    if n == 0:
        return np.zeros(0), np.zeros(0, dtype=np.int64)
    s = pl.Series(p)
    percentile = 100.0 * s.rank("average").to_numpy() / n
    from_top = (-s).rank("min").to_numpy()  # 1 = the highest p
    decile = np.minimum(10, np.floor((from_top - 1) * 10 / n).astype(np.int64) + 1)
    return percentile, decile


def facility_scores(
    frame: pl.DataFrame,
    pred: Any,
    *,
    manifest: InputManifest,
    champion: Champion,
    version: int,
    candidate: str,
    ml_run_id: str,
    scored_at: str,
) -> pl.DataFrame:
    """``facility_scores`` (§10.1) from the score frame and ``predict_frame``'s output (same rows, same order)."""
    p = pred["p"].to_numpy(dtype=float)
    percentile, decile = rank_columns(p)
    n = frame.height

    def const(value: Any, dtype: Any) -> pl.Series:
        return pl.Series([value] * n, dtype=dtype)

    return pl.DataFrame(
        {
            "ccn": frame["ccn"],
            "t": frame["t"],
            "score_month": const(_month_start(manifest.v), pl.Date),
            "state": frame["state"] if "state" in frame.columns else const(None, pl.String),
            "chain_id": frame["chain_id"]
            if "chain_id" in frame.columns
            else const(None, pl.String),
            f"p_{manifest.label}": pl.Series(p),
            "logit": pl.Series(pred["logit"].to_numpy(dtype=float)),
            "risk_percentile": pl.Series(percentile),
            "risk_decile": pl.Series(decile, dtype=pl.Int64),
            "model_name": const(champion.model_name, pl.String),
            "model_version": const(version, pl.Int64),
            "candidate": const(candidate, pl.String),
            "gold_run_id": const(manifest.gold_run_id, pl.String),
            "dataset_version": const(manifest.dataset_version, pl.String),
            "gold_verified": const(manifest.verified, pl.Boolean),
            "stale_refresh": const(manifest.v != manifest.v_labels, pl.Boolean),
            "ml_run_id": const(ml_run_id, pl.String),
            "scored_at": const(scored_at, pl.String),
        }
    )


# --------------------------------------------------------------------------- the task


@dataclass(frozen=True)
class ScoreOutput:
    skipped: bool
    reason: str
    v: str | None = None
    model_version: int | None = None
    n_facilities: int = 0
    mean_p: float | None = None
    stale_refresh: bool = False
    scores_dir: str | None = None
    key_features: tuple[str, ...] = ()

    def summary(self, ml_run_id: str) -> dict[str, Any]:
        return {
            "ml_run_id": ml_run_id,
            "skipped": self.skipped,
            "reason": self.reason,
            "v": self.v,
            "model_version": self.model_version,
            "n_facilities": self.n_facilities,
            "mean_p": self.mean_p,
            "stale_refresh": self.stale_refresh,
            "scores_dir": self.scores_dir,
        }


def _parquet(df: pl.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.write_parquet(buf)
    return buf.getvalue()


def _csv(df: pl.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.write_csv(buf)
    return buf.getvalue()


def run_score(
    store: ObjectStore,
    meta: MetadataStore,
    *,
    ml_run_id: str,
    cfg: MlConfig,
    registry: Registry | None,
    feature_registry: FeatureRegistry | None = None,
    mode: Mode = "auto",
    now: datetime | None = None,
) -> ScoreOutput:
    """The ``score`` task. Raises ``MlInputError`` (exit 2) on an integrity or data refusal, ``ConfigError``
    (exit 3) when a champion is named but no registry is configured."""
    from ..training import mlflow_log
    from ..training.train import read_decision

    paths = RunPaths.of(cfg, ml_run_id)
    champion = read_champion(store, paths)
    if champion is None or champion.version is None:
        reason = (
            "no champion.json in the run"
            if champion is None
            else "no champion yet: awaiting Checkpoint ML-A"
        )
        log.info("score %s: skipped (%s)", ml_run_id, reason)
        return ScoreOutput(skipped=True, reason=reason)
    if registry is None:  # wiring, not data: exit 3, as in promote (§15.5)
        raise ConfigError(
            "champion.json names a version but no model registry is configured (MLFLOW_TRACKING_URI)"
        )

    if champion.model_name != cfg.registry.model_name:
        raise MlInputError(
            f"champion.json names model {champion.model_name!r}, the registry is {cfg.registry.model_name!r}"
        )
    alias = registry.by_alias(cfg.registry.champion_alias)
    check_champion_integrity(meta, alias.version if alias else None)
    if alias is None or alias.version != champion.version:
        raise MlInputError(
            f"champion.json names version {champion.version} but @champion is "
            f"{alias.version if alias else None}: the champion moved after this run recorded it"
        )
    version = champion.version
    model = load_champion_package(store, meta, cfg, version)
    if model.meta is None:
        raise MlInputError(f"version {version} has no package meta")
    if set(model.meta.training_months) != {"refit"}:
        raise MlInputError(
            f"version {version} is not a refit package (training_months {sorted(model.meta.training_months)})"
        )

    manifest = read_input_manifest(store, paths)
    windows = read_windows(store, paths)
    frame = read_frame(store, paths, "score")
    if frame.is_empty():
        raise MlInputError("the score frame is empty")
    months = set(frame["t"].cast(pl.String).str.slice(0, 7).to_list())
    if months != {windows.score_month} or windows.score_month != manifest.v:
        raise MlInputError(
            f"the score frame holds months {sorted(months)}, not only V = {manifest.v}"
        )
    # Defence in depth: training refuses these too (§1.4, §20). By name, and by the registry's model_use.
    forbidden = [f for f in model.feature_names if is_forbidden(f)]
    if forbidden:
        raise MlInputError(f"version {version} uses forbidden model inputs: {forbidden}")
    registry_features = {f.name: f for f in (feature_registry or load_feature_registry()).features}
    not_model = [
        f
        for f in model.feature_names
        if f not in registry_features or registry_features[f].model_use != "model"
    ]
    if not_model:
        raise MlInputError(
            f"version {version} uses inputs the feature registry doesn't mark model_use: model: {not_model}"
        )
    missing = [f for f in model.feature_names if f not in frame.columns]
    if missing:
        raise MlInputError(f"the score frame lacks model features: {missing}")

    scored_at = (now or datetime.now(UTC)).strftime("%Y-%m-%dT%H:%M:%SZ")
    pred = model.predict_frame(frame)
    base, contrib = model.explain(frame)
    err = float(np.abs(base + contrib.to_numpy().sum(axis=1) - pred["logit"].to_numpy()).max())
    if err > SHAP_TOLERANCE:
        raise MlInputError(
            f"contributions do not add up to the calibrated log-odds (max error {err:.2e})"
        )

    scores = facility_scores(
        frame,
        pred,
        manifest=manifest,
        champion=champion,
        version=version,
        candidate=model.meta.candidate,
        ml_run_id=ml_run_id,
        scored_at=scored_at,
    )
    reasons = explain.reasons(
        contrib,
        frame,
        score_month=_month_start(manifest.v),
        model_version=version,
        gold_run_id=manifest.gold_run_id,
        labels=explain.load_reason_labels(project_root() / cfg.explain.labels),
        top_raise=cfg.explain.top_raise,
        top_lower=cfg.explain.top_lower,
    )
    shap = explain.shap_table(
        base, contrib, frame, model_version=version, gold_run_id=manifest.gold_run_id
    )
    beds = (
        frame["certified_beds"]
        if "certified_beds" in frame.columns
        else pl.Series([None] * frame.height)
    )
    chains = chain_scores(scores.with_columns(certified_beds=beds), f"p_{manifest.label}")
    tables = {
        "facility_scores": scores,
        "facility_reasons": reasons,
        "shap_values": shap,
        "chain_scores": chains,
    }

    # Global importance of the champion on the score frame (§8.2) and its key features (§12.1). Built, plots
    # included, before anything is written: a failure here leaves the month as it was.
    mean_abs = explain.mean_abs_table(contrib)
    families = explain.family_table(mean_abs, [registry_features[f] for f in model.feature_names])
    top = explain.key_features(mean_abs, cfg.explain.key_features)
    # the drift watch list is the version's, fixed at its first score: a feature that drifts can leave this
    # month's top list, and monitor must still watch it
    watch = json.loads(alias.tags["key_features"]) if "key_features" in alias.tags else top
    report_files = {
        "shap_mean_abs.csv": _csv(mean_abs),
        "shap_families.csv": _csv(families),
        "key_features.json": json.dumps(watch, indent=2).encode(),
    }
    try:
        report_files["shap_bar.png"] = explain.bar_chart(mean_abs, title="Champion, score month")
        report_files["shap_beeswarm.png"] = explain.beeswarm(contrib, model.model_frame(frame))
    except ImportError as exc:  # matplotlib/shap missing: the plots are skipped, never the scores
        log.warning("score %s: SHAP plots skipped (%s)", ml_run_id, exc)

    # The old manifest goes first: a rescore that dies between the tables reads as "not scored", never as a
    # valid month mixing two writes. It moves to the prefix's tmp/ (the store deletes only under scratch
    # prefixes), so it stays recoverable. The new manifest is written last.
    out_dir = scores_dir(cfg, manifest.v)
    if store.exists(f"{out_dir}{SCORES_MANIFEST}"):
        store.move(
            f"{out_dir}{SCORES_MANIFEST}",
            f"{cfg.outputs.root_prefix}tmp/{cfg.outputs.scores_prefix}/v={manifest.v}/"
            f"{SCORES_MANIFEST}.{uuid.uuid4().hex}",
        )
    keys, sha, rows = {}, {}, {}
    for name, table in tables.items():
        key = f"{out_dir}{name}.parquet"
        sha[name] = store.put_bytes(key, _parquet(table), overwrite=True).sha256
        keys[name], rows[name] = key, table.height
    stale = manifest.v != manifest.v_labels
    scores_manifest = ScoresManifest(
        v=manifest.v,
        v_labels=manifest.v_labels,
        label=manifest.label,
        model_name=champion.model_name,
        model_version=version,
        candidate=model.meta.candidate,
        gold_run_id=manifest.gold_run_id,
        dataset_version=manifest.dataset_version,
        gold_verified=manifest.verified,
        stale_refresh=stale,
        ml_run_id=ml_run_id,
        scored_at=scored_at,
        files=keys,
        rows=rows,
        sha256=sha,
    )
    store.put_bytes(
        f"{out_dir}{SCORES_MANIFEST}",
        scores_manifest.model_dump_json(indent=2).encode(),
        overwrite=True,
    )
    for name, data in report_files.items():
        store.put_bytes(f"{score_report_dir(paths)}{name}", data, overwrite=True)
    if (
        "key_features" not in alias.tags
    ):  # set once per version (§8.2): the drift monitor's fixed watch list
        registry.set_tags(version, {"key_features": json.dumps(top)})

    mean_p = float(scores[f"p_{manifest.label}"].mean())  # type: ignore[arg-type]
    tags = mlflow_log.base_tags(
        manifest,
        ml_run_id=ml_run_id,
        recipe_hash=model.meta.recipe_hash,
        decision=read_decision(store, paths),
        mode=mode,
    )
    with mlflow_log.task_run(cfg, "score", {**tags, "model_version": str(version)}) as run_id:
        if run_id is not None:
            mlflow_log.log_metrics({"n_facilities": float(scores.height), "mean_p": mean_p})
            mlflow_log.log_json_artifact(scores_manifest.model_dump(mode="json"), SCORES_MANIFEST)
            from ..evaluation.plots import log_files

            log_files(report_files, artifact_path="score")
    log.info(
        "score %s: %d facilities at %s with version %d, mean p %.3f",
        ml_run_id,
        scores.height,
        manifest.v,
        version,
        mean_p,
    )
    return ScoreOutput(
        skipped=False,
        reason="scored",
        v=manifest.v,
        model_version=version,
        n_facilities=scores.height,
        mean_p=mean_p,
        stale_refresh=stale,
        scores_dir=out_dir,
        key_features=tuple(top),
    )
