"""The ``monitor`` task (docs/ML_PIPELINE_SPEC.md §12): this run's signals, ``status/latest.json`` and its summary.

Signals (§12.1), on the run's pinned frames:
- **feature drift:** PSI of each key feature, score month vs the previous month (``drift:<feature>``);
- **training vs scoring skew:** PSI of the key, enforcement and regime-sensitive features, refit window vs score month
  (``skew:<feature>``), plus ``prepare``'s regime-coverage flags (``regime:<feature>``);
- **stale champion:** the champion's ``v_labels`` more than ``champion_lag_months_max`` behind this gold's;
- **score drift:** ``score``'s ``score_drift.json``, the champion's ``p`` on the previous month vs V: PSI over
  ``score_psi_max`` (``score_drift:psi``), mean moving more than ``mean_shift_max`` (``score_drift:mean``);
- **performance trend:** on a run that evaluated, each test month's ``pr_auc_lift`` vs the median lift of the earlier
  test months that earlier evaluated runs recorded (the latest run counts per month); more than ``lift_drop_max``
  under it is ``performance:lift_drop:<month>``.

Key features are the champion version's fixed list that ``score`` recorded (``score/key_features.json``, from the
version's ``key_features`` tag, §8.2), so a feature that drifts out of a month's top list stays watched; without scores this
run (no champion yet) there is no drift signal, and the skew check covers only the enforcement and regime-sensitive
features (``skew_scope`` says so, #73 B20). Realized performance waits for published scores to mature (20 months).

A feature-drift, score-drift or performance flag sets ``retune_requested`` on this run's ``ml_runs`` row (action ``monitor``) and in the email: a
request for a human, who re-tunes with ``mode=retune``. Phase 1 never re-tunes on its own (§12.2), so a request
can't loop. Skew and stale-champion flags only alert. The status file is written last; its fields
and the headline metrics (test PR-AUC and Brier with their CIs, when this run evaluated a model) are the summary
``notify`` emails.
"""

from __future__ import annotations

import json
import logging
import math
import statistics
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import polars as pl

from ..common.config import FeatureRegistry, MlConfig
from ..common.io.metadata_store import MetadataStore
from ..common.io.object_store import ObjectStore
from ..evaluation.gate import GATE_ACTIONS
from ..evaluation.registry import recorded_champion
from ..evaluation.report import EvaluationReport
from ..serving.score import SCORES_MANIFEST, ScoresManifest, score_report_dir, scores_dir
from ..training.contracts import RunPaths, TrainResult
from ..training.data import read_frame, read_input_manifest
from ..training.train import read_decision
from .decide import MONITOR_ACTION
from .drift import psi

log = logging.getLogger(__name__)

HEADLINE = ("pr_auc", "brier")
# the flags that ask a human for a re-tune (§12.2); skew and a stale champion only alert
RETUNE_FLAGS = ("drift:", "score_drift:", "performance:")


@dataclass
class MonitorOutput:
    status: dict[str, Any]
    psi: dict[str, dict[str, float]] = field(
        default_factory=dict
    )  # "drift" / "skew" -> feature -> PSI
    headline: dict[str, dict[str, float | None]] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {**self.status, "psi": self.psi, "headline": self.headline}


def _months(ym: str) -> int:
    return int(ym[:4]) * 12 + int(ym[5:7])


def _read(store: ObjectStore, key: str) -> bytes | None:
    if not store.exists(key):
        return None
    with store.open_read(key) as f:
        return f.read()


def _lift(entry: dict[str, Any]) -> float | None:
    lift = entry.get("pr_auc_lift")
    return None if lift is None or math.isnan(lift) else float(lift)


def _performance_trend(
    store: ObjectStore,
    cfg: MlConfig,
    rows: list[dict[str, Any]],
    ml_run_id: str,
    per_month: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Each of this run's test months: its lift vs the median lift of the earlier test months recorded by earlier
    evaluated runs (their gate rows lead to their evaluation.json; the latest run counts per month)."""
    first_row: dict[str, int] = {}
    for r in rows:
        if r.get("action") in GATE_ACTIONS and r["ml_run_id"] != ml_run_id:
            first_row.setdefault(r["ml_run_id"], r["id"])
    recorded: dict[str, float] = {}
    for run in sorted(first_row, key=first_row.__getitem__):
        raw = _read(store, RunPaths.of(cfg, run).evaluation)
        for entry in json.loads(raw).get("per_month", []) if raw else []:
            lift = _lift(entry)
            if lift is not None:
                recorded[entry["month"]] = lift
    trend: dict[str, dict[str, Any]] = {}
    for entry in per_month:
        lift = _lift(entry)
        if lift is None:
            continue
        earlier = [v for month, v in recorded.items() if month < entry["month"]]
        median = statistics.median(earlier) if earlier else None
        trend[entry["month"]] = {
            "lift": lift,
            "n_earlier": len(earlier),
            "median_earlier": median,
            "flagged": median is not None and lift < (1 - cfg.monitor.lift_drop_max) * median,
        }
    return trend


def _psi_table(
    reference: pl.DataFrame, current: pl.DataFrame, features: list[str], bins: int
) -> dict[str, float]:
    if reference.is_empty() or current.is_empty():
        return {}
    return {
        f: round(psi(reference[f], current[f], bins), 4)
        for f in features
        if f in reference.columns and f in current.columns
    }


def run_monitor(
    store: ObjectStore,
    meta: MetadataStore,
    *,
    ml_run_id: str,
    cfg: MlConfig,
    registry: FeatureRegistry,
    actor: str = "cms_nh_ml",
) -> MonitorOutput:
    paths = RunPaths.of(cfg, ml_run_id)
    m = cfg.monitor
    manifest = read_input_manifest(store, paths)
    decision = read_decision(store, paths)
    rows = meta.read("ml_runs")
    mine = [r for r in rows if r.get("ml_run_id") == ml_run_id]
    gate = max(
        (r for r in mine if r.get("action") in GATE_ACTIONS), key=lambda r: r["id"], default=None
    )
    published = any(r.get("action") == "score" and r.get("scores_published") for r in mine)

    scores: ScoresManifest | None = None
    raw = _read(store, f"{scores_dir(cfg, manifest.v)}{SCORES_MANIFEST}") if published else None
    if raw is not None:
        scores = ScoresManifest.model_validate_json(raw)
    key_raw = _read(store, f"{score_report_dir(paths)}key_features.json")
    key: list[str] = json.loads(key_raw) if key_raw else []

    out = MonitorOutput(status={})
    flags: list[str] = []
    score = read_frame(store, paths, "score")
    if key:
        out.psi["drift"] = _psi_table(
            read_frame(store, paths, "prev_score"), score, key, m.psi_bins
        )
        flags += [f"drift:{f}" for f, v in out.psi["drift"].items() if v > m.feature_psi_max]
    enforcement = [
        f.name for f in registry.family("enforcement")
    ]  # §12.1; names not in the frames are skipped
    skew_features = list(dict.fromkeys([*key, *enforcement, *cfg.features.regime_sensitive]))
    out.psi["skew"] = _psi_table(
        read_frame(store, paths, "refit_train"), score, skew_features, m.psi_bins
    )
    flags += [f"skew:{f}" for f, v in out.psi["skew"].items() if v > m.feature_psi_max]
    flags += [f"regime:{r.feature}" for r in manifest.regime_flags]

    champion = scores.model_version if scores else recorded_champion(meta)
    registered = [
        r
        for r in rows
        if r.get("action") in GATE_ACTIONS and r.get("candidate_version") == champion
    ]
    if champion is not None and registered:
        trained_on = min(registered, key=lambda r: r["id"])["v_labels"]
        lag = _months(manifest.v_labels) - _months(trained_on)
        if lag > m.champion_lag_months_max:
            flags.append(f"stale_champion:{lag}_months")
    drift_raw = _read(store, f"{score_report_dir(paths)}score_drift.json")
    score_drift = json.loads(drift_raw) if drift_raw else None
    if score_drift and score_drift["psi"] is not None and score_drift["psi"] > m.score_psi_max:
        flags.append("score_drift:psi")
    if (
        score_drift
        and score_drift["mean_shift"] is not None
        and abs(score_drift["mean_shift"]) > m.mean_shift_max
    ):
        flags.append("score_drift:mean")

    candidate = None
    raw_train = _read(store, paths.train_result)
    if raw_train is not None:
        candidate = TrainResult.model_validate_json(raw_train).selected
    raw_eval = _read(store, paths.evaluation)
    trend: dict[str, dict[str, Any]] | None = None
    if raw_eval is not None:
        report = EvaluationReport.model_validate_json(raw_eval)
        trend = _performance_trend(store, cfg, rows, ml_run_id, report.per_month)
        flags += [f"performance:lift_drop:{month}" for month, t in trend.items() if t["flagged"]]
        intervals = report.candidates[report.selected].intervals
        out.headline = {
            k: {"estimate": iv.estimate, "ci_low": iv.ci_low, "ci_high": iv.ci_high}
            for k, iv in intervals.items()
            if k in HEADLINE
        }
    retune = any(f.startswith(RETUNE_FLAGS) for f in flags)
    gate_results = (
        json.loads(gate["gate_results_json"]) if gate and gate.get("gate_results_json") else {}
    )
    out.status = {
        "v": manifest.v,
        "v_labels": manifest.v_labels,
        "scored_at": scores.scored_at if scores else None,
        "model_name": cfg.registry.model_name,
        "model_version": champion,
        "candidate": scores.candidate if scores else candidate,
        "gold_run_id": manifest.gold_run_id,
        "gold_verified": manifest.verified,
        "stale_refresh": scores.stale_refresh if scores else manifest.v != manifest.v_labels,
        "decision": decision,
        "gate_outcome": gate.get("gate_outcome") if gate else None,
        "failed_gates": [
            r["gate"] for r in gate_results.get("results", []) if r.get("status") == "fail"
        ],
        "n_facilities": scores.rows.get("facility_scores") if scores else 0,
        "flags": flags,
        "retune_requested": retune,
        "score_drift": score_drift,
        "performance_trend": trend,
        "skew_scope": "full" if key else "partial: no key features (this run didn't score)",
        "ml_run_id": ml_run_id,
        "note": None if champion is not None else "no champion yet: awaiting Checkpoint ML-A",
    }

    if not any(
        r.get("action") == MONITOR_ACTION for r in mine
    ):  # a retried monitor writes its row once
        meta.append(
            "ml_runs",
            [
                {
                    "ml_run_id": ml_run_id,
                    "at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "mode": json.loads(_read(store, paths.decision) or b"{}").get("mode"),
                    "decision": decision,
                    "action": MONITOR_ACTION,
                    "reason": ", ".join(flags) or "no flags",
                    "actor": actor,
                    "gold_run_id": manifest.gold_run_id,
                    "v": manifest.v,
                    "v_labels": manifest.v_labels,
                    "dataset_version": manifest.dataset_version,
                    "gold_verified": manifest.verified,
                    "champion_version_before": champion,
                    "champion_version_after": champion,
                    "scores_published": False,
                    "flags_json": json.dumps(flags),
                    "retune_requested": retune,
                }
            ],
        )
    store.put_bytes(
        f"{cfg.outputs.root_prefix}{cfg.outputs.status_key}",
        json.dumps(out.status, indent=2, sort_keys=True).encode(),
        overwrite=True,
    )
    log.info("monitor %s: %s", ml_run_id, flags or "no flags")
    return out
