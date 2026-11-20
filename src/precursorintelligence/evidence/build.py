"""``precursor-pipeline agents build-evidence --v YYYY-MM`` (docs/AGENTS_PLAN.md E2): one month's evidence for the
agents, in the layout of ``evidence.contract``.

Runs after ``publish`` (the DAG ``cms_nh_agents``, L10). It refuses (``MlInputError``, exit 2) unless the month's
scores match their manifest and ``ml_runs`` records them as published, so the agents never serve an unpublished
month. From the month's files it writes:

- ``facility_scores``, ``facility_reasons``, ``chain_scores``: from ``scores/v=YYYY-MM/``, ``p_<label>`` renamed ``p``;
- ``citations``, ``penalties``: from the silver partition of vintage ``v`` (what CMS had published by then), for the
  scored facilities, dated in (cutoff - N months, cutoff] where the cutoff is the first day of ``v``: the same
  point-in-time rule as the features;
- ``ops_gates.json`` (the gate run that registered the scoring model: its ``ml_runs`` row with the gate results),
  ``ops_evaluation.json`` (that run's ``evaluation.json``) and ``ops_status.json`` (``status/latest.json`` when it
  exists; ``monitor`` writes it, and its ``v`` can be another month).

Each build goes to a folder of its own (``v=YYYY-MM/build=<id>/``), then ``evidence_manifest.json``, and
``_current.json`` last, so the month being served is never changed under the agents. The pointer never moves to an
older month (checked again right before it is written), and a build from development outputs (``root_prefix`` such
as ``ml-dev/``) never replaces a production build; a production build replaces a development one (the demo fallback,
AGENTS_PLAN.md §4).
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any

import polars as pl
from pydantic import ValidationError

from ..common.config import EvidenceCfg, MlConfig
from ..common.io.metadata_store import MetadataStore
from ..common.io.object_store import ObjectStore
from ..evaluation.registry import registering_gate_row
from ..harmonize.ledgers import SilverReader
from ..serving.publish import verify_scores
from ..serving.score import SCORES_MANIFEST, ScoresManifest, scores_dir
from ..training import MlInputError
from ..training.contracts import RunPaths
from .contract import (
    CURRENT,
    EVIDENCE_MANIFEST,
    TABLE_COLUMNS,
    V_PATTERN,
    CurrentPointer,
    EvidenceManifest,
    build_dir,
)

log = logging.getLogger(__name__)

# The silver table and event date of each windowed table.
SILVER = {
    "citations": ("citations_by_vintage", "survey_date", ["survey_date", "ftag"]),
    "penalties": (
        "penalties_by_vintage",
        "penalty_date",
        ["penalty_date", "penalty_type", "dup_seq"],
    ),
}


@dataclass(frozen=True)
class BuildOutput:
    v: str
    model_version: int
    gold_run_id: str
    source: str
    gold_verified: bool
    rows: dict[str, int]
    files: list[str]

    def summary(self) -> dict[str, Any]:
        return {
            "v": self.v,
            "model_version": self.model_version,
            "gold_run_id": self.gold_run_id,
            "source": self.source,
            "gold_verified": self.gold_verified,
            "rows": self.rows,
            "files": self.files,
        }


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read(store: ObjectStore, key: str) -> bytes:
    with store.open_read(key) as f:
        return f.read()


def _parquet(df: pl.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.write_parquet(buf, compression="zstd")
    return buf.getvalue()


def _json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _months_before(d: date, months: int) -> date:
    """The first of the month ``months`` before ``d`` (a first of the month)."""
    index = d.year * 12 + d.month - 1 - months
    return date(index // 12, index % 12 + 1, 1)


def _require_published(meta: MetadataStore, scores: ScoresManifest) -> None:
    rows = meta.read("ml_runs", {"action": ["score"], "v": scores.v})
    if not any(r.get("scores_published") and r.get("ml_run_id") == scores.ml_run_id for r in rows):
        raise MlInputError(
            f"ml_runs records no published scores of run {scores.ml_run_id} for {scores.v}: "
            "run `ml publish` first"
        )


def _score_tables(store: ObjectStore, scores: ScoresManifest) -> dict[str, pl.DataFrame]:
    def read(name: str) -> pl.DataFrame:
        return pl.read_parquet(io.BytesIO(_read(store, scores.files[name])))

    p_col = f"p_{scores.label}"
    facility = read("facility_scores")
    if p_col not in facility.columns:
        raise MlInputError(f"facility_scores has no {p_col} for label {scores.label}")
    return {
        "facility_scores": facility.rename({p_col: "p"})
        .select(TABLE_COLUMNS["facility_scores"])
        .sort("ccn"),
        "facility_reasons": read("facility_reasons")
        .select(TABLE_COLUMNS["facility_reasons"])
        .sort(["ccn", "direction", "rank"], descending=[False, True, False]),
        "chain_scores": read("chain_scores").select(TABLE_COLUMNS["chain_scores"]).sort("chain_id"),
    }


def _windowed(
    silver: SilverReader, name: str, v: str, months: int, scored: pl.Series
) -> pl.DataFrame:
    table, date_col, order = SILVER[name]
    columns = list(TABLE_COLUMNS[name])  # type: ignore[index]
    key = f"silver/{table}/vintage={v}/part-0.parquet"  # SilverReader's key
    if not silver.store.exists(key):
        raise MlInputError(f"{key} is missing: silver has no partition for vintage {v}")
    df = silver.read(table, v, [*columns, "cutoff_date"])
    cutoff = date(int(v[:4]), int(v[5:7]), 1)
    if df.height and set(df["cutoff_date"].unique().to_list()) != {cutoff}:
        raise MlInputError(f"silver/{table} vintage {v} has a cutoff_date other than {cutoff}")
    since = _months_before(cutoff, months)
    return (
        df.filter(
            pl.col("ccn").is_in(scored.implode())
            & (pl.col(date_col) > since)
            & (pl.col(date_col) <= cutoff)
        )
        .select(columns)
        .sort(["ccn", *order])
    )


def _ops(
    store: ObjectStore, meta: MetadataStore, ml_cfg: MlConfig, scores: ScoresManifest
) -> dict[str, bytes]:
    gate = registering_gate_row(meta, scores.model_version)
    results = gate.get("gate_results_json")
    out = {
        "ops_gates": _json(
            {
                "schema_version": 1,
                "model_version": scores.model_version,
                "ml_run_id": gate["ml_run_id"],
                "v": gate.get("v"),
                "action": gate.get("action"),
                "gate_outcome": gate.get("gate_outcome"),
                "gate_results": json.loads(results) if results else None,
            }
        )
    }
    evaluation = RunPaths.of(ml_cfg, gate["ml_run_id"]).evaluation
    if not store.exists(evaluation):
        raise MlInputError(
            f"{evaluation} is missing: the scoring model's gate run has no evaluation"
        )
    out["ops_evaluation"] = _read(store, evaluation)
    status = f"{ml_cfg.outputs.root_prefix}{ml_cfg.outputs.status_key}"
    if store.exists(status):
        out["ops_status"] = _read(store, status)
    return out


def _check_pointer(store: ObjectStore, prefix: str, v: str, source: str) -> None:
    """Refuse to move ``_current.json`` to an older month, or from a production build to a development one."""
    pointer_key = f"{prefix}{CURRENT}"
    if not store.exists(pointer_key):
        return
    try:
        current = CurrentPointer.model_validate_json(_read(store, pointer_key))
        manifest_key = f"{build_dir(prefix, current.v, current.build)}{EVIDENCE_MANIFEST}"
        served = EvidenceManifest.model_validate_json(_read(store, manifest_key))
    except (ValidationError, FileNotFoundError, OSError) as exc:
        raise MlInputError(
            f"{pointer_key} or the manifest it names is unreadable ({exc.__class__.__name__}): "
            "fix or remove it by hand"
        ) from None
    if current.v > v:
        raise MlInputError(f"{pointer_key} already serves {current.v}, later than {v}")
    if source and not served.source:
        raise MlInputError(
            f"{pointer_key} serves a production build of {current.v}: a build from {source} "
            "never replaces it"
        )


def run_build_evidence(
    store: ObjectStore,
    meta: MetadataStore,
    *,
    v: str,
    ml_cfg: MlConfig,
    cfg: EvidenceCfg,
    built_at: str | None = None,
) -> BuildOutput:
    """Build the evidence of month ``v`` and point ``_current.json`` at it."""
    if not V_PATTERN.match(v):
        raise MlInputError(f"--v must be YYYY-MM in 2021-2026, got {v!r}")
    source = ml_cfg.outputs.root_prefix
    _check_pointer(store, cfg.prefix, v, source)

    directory = scores_dir(ml_cfg, v)
    scores = verify_scores(store, directory)
    if scores.v != v:
        raise MlInputError(f"{directory}{SCORES_MANIFEST} is for {scores.v}, not {v}")
    _require_published(meta, scores)

    tables = _score_tables(store, scores)
    scored = tables["facility_scores"]["ccn"]
    silver = SilverReader(store, [v])
    tables["citations"] = _windowed(silver, "citations", v, cfg.citation_months, scored)
    tables["penalties"] = _windowed(silver, "penalties", v, cfg.penalty_months, scored)

    blobs = {f"{name}.parquet": _parquet(df) for name, df in tables.items()}
    blobs |= {f"{name}.json": data for name, data in _ops(store, meta, ml_cfg, scores).items()}
    built_at = built_at or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    sha = {filename.rsplit(".", 1)[0]: _sha256(data) for filename, data in blobs.items()}
    build = _sha256("\n".join([built_at, *(f"{n}:{h}" for n, h in sorted(sha.items()))]).encode())[
        :12
    ]
    out_dir = build_dir(cfg.prefix, v, build)
    files = {}
    for filename, data in sorted(blobs.items()):
        key = f"{out_dir}{filename}"
        store.put_bytes(key, data, overwrite=True)
        files[filename.rsplit(".", 1)[0]] = key

    manifest = EvidenceManifest(
        v=v,
        gold_run_id=scores.gold_run_id,
        model_version=scores.model_version,
        source=source,
        gold_verified=scores.gold_verified,
        built_at=built_at,
        files=files,
        rows={name: df.height for name, df in tables.items()},
        sha256=sha,
    )
    raw_manifest = _json(manifest.model_dump(mode="json"))
    store.put_bytes(f"{out_dir}{EVIDENCE_MANIFEST}", raw_manifest, overwrite=True)

    pointer = CurrentPointer(
        v=v,
        build=build,
        scores_manifest_sha256=_sha256(_read(store, f"{directory}{SCORES_MANIFEST}")),
        evidence_manifest_sha256=_sha256(raw_manifest),
    )
    # Again right before the write: another build may have moved the pointer meanwhile. (Not atomic: on GCS a
    # generation-match write would close the gap; cms_nh_agents runs one build at a time.)
    _check_pointer(store, cfg.prefix, v, source)
    pointer_key = f"{cfg.prefix}{CURRENT}"
    store.put_bytes(pointer_key, _json(pointer.model_dump(mode="json")), overwrite=True)  # last
    log.info("build-evidence %s build %s: %s", v, build, manifest.rows)
    return BuildOutput(
        v=v,
        model_version=scores.model_version,
        gold_run_id=scores.gold_run_id,
        source=source,
        gold_verified=scores.gold_verified,
        rows=manifest.rows,
        files=sorted(files),
    )
