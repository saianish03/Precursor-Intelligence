"""The ``publish`` task, Phase 1 (docs/ML_PIPELINE_SPEC.md §11): verify the month's scores and record them.

``scores_manifest.json`` (written last by ``score``) must match the files beside it: every listed file present,
with the recorded sha256 and row count, stamped with the same model version. Then one append-only ``ml_runs`` row
(``action: score``, ``scores_published: true``) records the published month, which ``decide`` and ``prepare``
(``training/data.last_scored_v``) read. The DAG task emits the ``cms_nh_scores`` asset on success (W6); BigQuery
is Phase 2.
"""

from __future__ import annotations

import io
import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import polars as pl
from pydantic import ValidationError

from ..common.config import MlConfig
from ..common.io.metadata_store import MetadataStore
from ..common.io.object_store import ObjectStore
from ..training import MlInputError
from ..training.contracts import Mode, RunPaths
from ..training.data import gold_config_sha, read_input_manifest, sha256_hex
from .score import SCORE_FILES, SCORES_MANIFEST, ScoresManifest, read_champion, scores_dir

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PublishOutput:
    skipped: bool
    reason: str
    v: str | None = None
    model_version: int | None = None
    rows: dict[str, int] | None = None

    def summary(self, ml_run_id: str) -> dict[str, Any]:
        return {
            "ml_run_id": ml_run_id,
            "skipped": self.skipped,
            "reason": self.reason,
            "v": self.v,
            "model_version": self.model_version,
            "rows": self.rows,
        }


def verify_scores(store: ObjectStore, directory: str) -> ScoresManifest:
    """Read ``scores_manifest.json`` in ``directory`` and check every file against it (``MlInputError`` if not)."""
    key = f"{directory}{SCORES_MANIFEST}"
    if not store.exists(key):
        raise MlInputError(f"{key} is missing: nothing was scored for this month")
    with store.open_read(key) as f:
        raw = f.read()
    try:
        manifest = ScoresManifest.model_validate_json(raw)
    except ValidationError as exc:
        raise MlInputError(
            f"{key} is not a valid scores manifest: {exc.error_count()} error(s)"
        ) from None
    if set(manifest.files) != set(SCORE_FILES):
        raise MlInputError(f"{key} lists {sorted(manifest.files)}, expected {sorted(SCORE_FILES)}")
    for name, file_key in manifest.files.items():
        if file_key != f"{directory}{name}.parquet":  # never another month's file
            raise MlInputError(f"{key} points {name} at {file_key}, outside {directory}")
        if not store.exists(file_key):
            raise MlInputError(f"{file_key} is missing")
        with store.open_read(file_key) as f:
            data = f.read()
        if sha256_hex(data) != manifest.sha256.get(name):
            raise MlInputError(f"{file_key} does not match its sha256 in {SCORES_MANIFEST}")
        table = pl.read_parquet(io.BytesIO(data))
        if table.height != manifest.rows.get(name):
            raise MlInputError(
                f"{file_key} has {table.height} rows, the manifest says {manifest.rows.get(name)}"
            )
        if "model_version" in table.columns and set(table["model_version"].unique().to_list()) - {
            manifest.model_version
        }:
            raise MlInputError(f"{file_key} holds rows of another model version")
    return manifest


def run_publish(
    store: ObjectStore,
    meta: MetadataStore,
    *,
    ml_run_id: str,
    cfg: MlConfig,
    mode: Mode = "auto",
    actor: str | None = None,
) -> PublishOutput:
    """The ``publish`` task. Skips when this run has no champion (``score`` skipped too); refuses (exit 2) when the
    scores don't match their manifest or were written by another run."""
    from ..training.train import read_decision

    paths = RunPaths.of(cfg, ml_run_id)
    champion = read_champion(store, paths)
    if champion is None or champion.version is None:
        return PublishOutput(skipped=True, reason="no champion: nothing was scored")
    manifest = read_input_manifest(store, paths)
    scores = verify_scores(store, scores_dir(cfg, manifest.v))
    if scores.ml_run_id != ml_run_id or scores.model_version != champion.version:
        raise MlInputError(
            f"the scores for {manifest.v} come from run {scores.ml_run_id} (version {scores.model_version}), "
            f"not this run's version {champion.version}"
        )
    mixed = {
        field: (getattr(scores, field), getattr(manifest, field))
        for field in ("v", "gold_run_id", "dataset_version")
        if getattr(scores, field) != getattr(manifest, field)
    }
    if mixed:  # e.g. the run was prepared again after score: one row must not mix two golds
        raise MlInputError(f"the scores and this run's input manifest disagree: {mixed}")

    done = meta.read("ml_runs", {"ml_run_id": ml_run_id, "action": ["score"]})
    if any(r.get("scores_published") for r in done):
        log.info("ml_runs already records the published scores of %s", ml_run_id)
    else:
        meta.append(
            "ml_runs",
            [
                {
                    "ml_run_id": ml_run_id,
                    "at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "mode": mode,
                    "decision": read_decision(store, paths),
                    "action": "score",
                    "reason": f"scores for {scores.v} published with version {scores.model_version}",
                    "actor": actor or os.environ.get("PI_ML_ACTOR", "cms_nh_ml"),
                    "gold_run_id": scores.gold_run_id,
                    "v": scores.v,
                    "v_labels": scores.v_labels,
                    "dataset_version": scores.dataset_version,
                    "gold_config_sha": gold_config_sha(manifest),
                    "gold_verified": scores.gold_verified,
                    "champion_version_before": scores.model_version,
                    "champion_version_after": scores.model_version,
                    "scores_published": True,
                    "flags_json": '["stale_refresh"]' if scores.stale_refresh else "[]",
                    "retune_requested": False,
                }
            ],
        )
    log.info("publish %s: %s verified (%s)", ml_run_id, scores.v, scores.rows)
    return PublishOutput(
        skipped=False,
        reason="published",
        v=scores.v,
        model_version=scores.model_version,
        rows=scores.rows,
    )
