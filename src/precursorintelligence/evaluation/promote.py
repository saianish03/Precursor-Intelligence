"""The ``promote`` task (docs/ML_PIPELINE_SPEC.md §9.4): move ``champion`` to a chosen version by hand.

This is both the manual promotion after Checkpoint ML-A and the rollback (any earlier version). The version and
the reason come from the DAG params through the environment (``PI_ML_VERSION``, ``PI_ML_REASON``), never the
command line. Only a version that passed G1 and G2 can be promoted: a hand-picked model still needs clean data and
a sane evaluation. Every promotion is an ``ml_runs`` row (``manual_promote`` or ``rollback``), which ``score``'s
integrity check matches against ``@champion``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from ..common.config import MlConfig
from ..common.io.metadata_store import MetadataStore
from ..common.io.object_store import ObjectStore
from ..training import MlInputError
from ..training.contracts import Champion, RunPaths
from ..training.data import gold_config_sha, read_input_manifest
from .registry import Registry

log = logging.getLogger(__name__)

VERSION_ENV = "PI_ML_VERSION"
REASON_ENV = "PI_ML_REASON"
PROMOTE_ACTIONS = ("manual_promote", "rollback")


class PromoteUsageError(ValueError):
    """The promotion request itself is wrong (no version or reason, no registry): a usage error, exit 3."""


def request_from_env() -> tuple[int, str]:
    """``PI_ML_VERSION`` (an integer >= 1) and ``PI_ML_REASON`` (required, non-empty)."""
    raw, reason = os.environ.get(VERSION_ENV, "").strip(), os.environ.get(REASON_ENV, "").strip()
    try:
        version = int(raw)
    except ValueError:
        raise PromoteUsageError(f"{VERSION_ENV} must be a version number, got {raw!r}") from None
    if version < 1:
        raise PromoteUsageError(f"{VERSION_ENV} must be >= 1, got {version}")
    if not reason:
        raise PromoteUsageError(
            f"{REASON_ENV} is required: say why (e.g. the Checkpoint ML-A approval)"
        )
    return version, reason


@dataclass(frozen=True)
class PromoteOutput:
    version: int
    action: str  # manual_promote | rollback
    champion_before: int | None
    reason: str


def run_promote(
    store: ObjectStore,
    meta: MetadataStore,
    *,
    ml_run_id: str,
    cfg: MlConfig,
    registry: Registry | None,
    version: int,
    reason: str,
    actor: str = "cms_nh_ml",
) -> PromoteOutput:
    if registry is None:
        raise PromoteUsageError(
            "promote needs the MLflow registry (MLFLOW_TRACKING_URI is not set)"
        )
    target = registry.get(version)
    if target is None:
        raise MlInputError(f"{cfg.registry.model_name} has no version {version}")
    if target.tags.get("sanity_passed") != "true":
        raise MlInputError(
            f"version {version} did not pass G1 and G2 (failed: {target.tags.get('failed_gates') or 'unknown'}): "
            "it cannot be promoted by hand"
        )
    current = registry.by_alias(cfg.registry.champion_alias)
    before = current.version if current else None
    action = "rollback" if before is not None and version < before else "manual_promote"
    registry.set_alias(cfg.registry.champion_alias, version)
    challenger = registry.by_alias(cfg.registry.challenger_alias)
    if challenger is not None and challenger.version == version:  # it is no longer waiting
        registry.delete_alias(cfg.registry.challenger_alias)
    registry.set_tags(version, {"promoted_by": actor, "promote_reason": reason[:500]})

    paths = RunPaths.of(cfg, ml_run_id)
    done = meta.read("ml_runs", {"ml_run_id": ml_run_id, "action": list(PROMOTE_ACTIONS)})
    if done:
        log.info("ml_runs already holds the promote row of %s: not written again", ml_run_id)
    else:
        meta.append(
            "ml_runs",
            [_row(store, paths, ml_run_id, target.tags, action, reason, actor, before, version)],
        )
    champion = Champion(model_name=cfg.registry.model_name, version=version, source="promote")
    store.put_bytes(paths.champion, champion.model_dump_json(indent=2).encode(), overwrite=True)
    log.info("promote %s: champion %s -> %s (%s): %s", ml_run_id, before, version, action, reason)
    return PromoteOutput(version, action, before, reason)


def _row(
    store: ObjectStore,
    paths: RunPaths,
    ml_run_id: str,
    tags: dict[str, str],
    action: str,
    reason: str,
    actor: str,
    before: int | None,
    version: int,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "ml_run_id": ml_run_id,
        "at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "mode": "promote",
        "decision": "PROMOTE",
        "action": action,
        "reason": reason,
        "actor": actor,
        "recipe_hash": tags.get("recipe_hash"),
        "champion_version_before": before,
        "champion_version_after": version,
        "candidate_version": version,
        "scores_published": False,
        "retune_requested": False,
    }
    if store.exists(paths.input_manifest):  # the PROMOTE branch runs prepare first (§15.2)
        m = read_input_manifest(store, paths)
        row |= {
            "gold_run_id": m.gold_run_id,
            "v": m.v,
            "v_labels": m.v_labels,
            "dataset_version": m.dataset_version,
            "gold_config_sha": gold_config_sha(m),
            "gold_verified": m.verified,
        }
    return row
