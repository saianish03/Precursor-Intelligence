"""The ``decide`` task (docs/ML_PIPELINE_SPEC.md §2, §15.2): what this cms_nh_ml run does.

It reads gold's ``_current.json`` and ``dataset_manifest.json`` (cheap, read-only), the ``ml_runs`` rows and the
recorded champion, and applies the decision table; ``decision_for`` is the table on its own (tested row by row).

- The run folder gets ``decision.json`` (``prepare`` pins the gold run named in it; every task tags its decision).
- SCORE also writes ``champion.json`` with the recorded champion (on the other paths ``gate`` or ``promote`` does).
- SKIP, REFUSE and RETUNE are the run's ``ml_runs`` row (``gate`` and ``publish`` write the others); REFUSE then
  exits 2.

An explicit mode wins over the automatic rules: ``score_only`` never retrains because the recipe changed. RETUNE
comes only from ``mode=retune`` and runs ``tune`` alone, since ``train`` doesn't apply a proposal; a drift flag
records a re-tune request for a human (Phase 1, §12.2). Gold not current is exit 1 (retried), as in ``prepare``.

The module lives outside ``training``/``evaluation``/``serving`` on purpose: those are the ``recipe_hash`` packages
(§14.3), and a change to how runs are scheduled must not count as a new model recipe.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..common.config import FeatureRegistry, MlConfig
from ..common.io.metadata_store import MetadataStore
from ..common.io.object_store import ObjectStore
from ..evaluation.gate import GATE_ACTIONS
from ..evaluation.registry import recorded_champion
from ..gold.dataset import MANIFEST_NAME
from ..training import GoldPinError
from ..training.contracts import Champion, Decision, DecisionName, Mode, RunPaths
from ..training.data import (
    CURRENT_NAME,
    config_sha,
    feature_list,
    last_scored_v,
    read_json_object,
)

log = logging.getLogger(__name__)

MONITOR_ACTION = "monitor"  # the ml_runs row `monitor` writes; it carries retune_requested (§12.2)
PROCEED = {"RETRAIN", "RETUNE", "SCORE", "PROMOTE"}
OWN_ROW = (
    "SKIP",
    "REFUSE",
    "RETUNE",
)  # runs no gate or publish writes a row for, so decide writes it


def _retrained(
    rows: Sequence[Mapping[str, Any]], v_labels: str, recipe_hash: str, gold_sha: str
) -> bool:
    """A RETRAIN went through the gate for these labels, this recipe and this gold build. A hold counts too, so a
    no-op gold publish after a hold doesn't retrain again."""
    return any(
        r.get("action") in GATE_ACTIONS
        and r.get("decision") == "RETRAIN"
        and r.get("v_labels") == v_labels
        and r.get("recipe_hash") == recipe_hash
        and r.get("gold_config_sha") == gold_sha
        for r in rows
    )


def _published(
    rows: Sequence[Mapping[str, Any]], v: str, dataset_version: str, version: int
) -> bool:
    return any(
        r.get("scores_published")
        and r.get("v") == v
        and r.get("dataset_version") == dataset_version
        and r.get("champion_version_after") == version
        for r in rows
    )


def decision_for(
    *,
    mode: Mode,
    v: str,
    v_labels: str,
    dataset_version: str,
    verified: bool,
    allow_unverified: bool,
    recipe_hash: str,
    gold_sha: str,
    champion: int | None,
    last_scored: str | None,
    rows: Sequence[Mapping[str, Any]],
) -> tuple[DecisionName, str]:
    """The §15.2 table: the first matching row wins."""
    if mode == "promote":
        return "PROMOTE", "manual promotion requested"
    if last_scored is not None and v < last_scored:
        return "REFUSE", f"gold V {v} is older than the last scored V {last_scored}"
    if not verified and not allow_unverified:
        return "REFUSE", "gold is not verified and allow_unverified is false"
    if mode == "retune":  # Phase 1: by hand only, and tune alone (its proposal and plots); §12.2
        return "RETUNE", "re-tune requested"
    if mode == "retrain":
        return "RETRAIN", "retrain requested"
    if mode == "auto" and not _retrained(rows, v_labels, recipe_hash, gold_sha):
        return "RETRAIN", f"no completed retrain for V_L {v_labels} with this recipe and gold build"
    if mode == "score_only":
        return "SCORE", "score requested"
    if champion is not None and not _published(rows, v, dataset_version, champion):
        return "SCORE", f"no published scores for {v} ({dataset_version}) with version {champion}"
    return "SKIP", "nothing new: trained for this gold and recipe, and its scores are published"


def run_decide(
    store: ObjectStore,
    meta: MetadataStore,
    *,
    ml_run_id: str,
    cfg: MlConfig,
    registry: FeatureRegistry,
    mode: Mode = "auto",
    allow_unverified: bool | None = None,
    ml_config_path: Path | None = None,
    actor: str = "cms_nh_ml",
) -> Decision:
    """Decide, write ``decision.json`` (and ``champion.json`` on SCORE, the ``ml_runs`` row on SKIP, REFUSE and
    RETUNE)."""
    from ..training.mlflow_log import recipe_hash  # imports MLflow: only when decide runs

    paths = RunPaths.of(cfg, ml_run_id)
    current, _ = read_json_object(store, f"{paths.gold_prefix}{CURRENT_NAME}")
    if current.get("state") != "current":
        raise GoldPinError(
            f"gold's {CURRENT_NAME} says state={current.get('state')!r}, not 'current'"
        )
    manifest, _ = read_json_object(store, f"{paths.gold_prefix}{MANIFEST_NAME}")
    v, v_labels = str(current.get("v")), str(current.get("v_labels"))
    dataset_version = str(manifest.get("dataset_version"))
    gold_sha = config_sha(manifest.get("config_sha256") or {}, manifest.get("git_commit"))
    recipe = recipe_hash(feature_list(registry, cfg), ml_config_path=ml_config_path)
    champion = recorded_champion(meta)
    rows = meta.read("ml_runs")
    allowed = cfg.allow_unverified if allow_unverified is None else allow_unverified
    name, reason = decision_for(
        mode=mode,
        v=v,
        v_labels=v_labels,
        dataset_version=dataset_version,
        verified=current.get("verified") is True and manifest.get("verified") is True,
        allow_unverified=allowed,
        recipe_hash=recipe,
        gold_sha=gold_sha,
        champion=champion,
        last_scored=last_scored_v(meta),
        rows=rows,
    )
    decision = Decision(
        decision=name,
        proceed=name in PROCEED,
        reason=reason,
        mode=mode,
        gold_run_id=current.get("run_id"),
        v=v,
        v_labels=v_labels,
        dataset_version=dataset_version,
        champion_version=champion,
    )
    store.put_bytes(paths.decision, decision.model_dump_json(indent=2).encode(), overwrite=True)
    if name == "SCORE":
        file = Champion(model_name=cfg.registry.model_name, version=champion, source="decide")
        store.put_bytes(paths.champion, file.model_dump_json(indent=2).encode(), overwrite=True)
    if name in OWN_ROW and not meta.read(
        "ml_runs", {"ml_run_id": ml_run_id, "action": [n.lower() for n in OWN_ROW]}
    ):  # a retried decide writes its row once
        meta.append(
            "ml_runs",
            [
                {
                    "ml_run_id": ml_run_id,
                    "at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "mode": mode,
                    "decision": name,
                    "action": name.lower(),
                    "reason": reason,
                    "actor": actor,
                    "gold_run_id": decision.gold_run_id,
                    "v": v,
                    "v_labels": v_labels,
                    "dataset_version": dataset_version,
                    "gold_config_sha": gold_sha,
                    "gold_verified": current.get("verified") is True,
                    "recipe_hash": recipe,
                    "champion_version_before": champion,
                    "champion_version_after": champion,
                    "scores_published": False,
                    "flags_json": json.dumps([]),
                    "retune_requested": False,
                }
            ],
        )
    log.info("decide %s: %s (%s)", ml_run_id, name, reason)
    return decision
