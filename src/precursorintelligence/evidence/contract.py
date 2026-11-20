"""What ``build-evidence`` writes and the agents read (docs/AGENTS_PLAN.md E1, E2): the file layout, the pointer and
the manifest. The producer (``evidence.build``) and the reader (``agents.tools.source``) both import it, so the
agents runtime stays out of the ``precursor-pipeline`` entrypoint.

Layout under the evidence prefix (``configs/agents.yaml``, e.g. ``agents/evidence/``)::

    _current.json                               written last: {"schema_version", "v", "build",
                                                "scores_manifest_sha256", "evidence_manifest_sha256"}
    v=YYYY-MM/build=<id>/evidence_manifest.json what the build's files are: v, gold_run_id, model_version, source,
                                                gold_verified, built_at, and per file its key and sha256 (and rows
                                                for a table)
    v=YYYY-MM/build=<id>/<table>.parquet        one file per name in TABLE_COLUMNS
    v=YYYY-MM/build=<id>/ops_*.json             the ops evidence of Agent 3 (I1): OPS_FILES

Each build writes a folder of its own and never changes another, so rebuilding the month being served (or a build
that fails halfway) never breaks it: the agents keep reading the old build until the pointer moves. The month and
the build come only from ``_current.json``: never from ``status/latest.json`` (its ``v`` can be a month with nothing
scored) and never from listing the prefix.
"""

from __future__ import annotations

import re
from typing import Literal

import polars as pl
from pydantic import BaseModel, ConfigDict

CURRENT = "_current.json"
EVIDENCE_MANIFEST = "evidence_manifest.json"
BUILD_PATTERN = re.compile(r"^[0-9a-f]{12}$")
# The ops evidence of Agent 3 (I1): ops_gates (the gate run that registered the scoring model, with its gate
# results), ops_evaluation (that run's evaluation.json) and ops_status (status/latest.json, whose own ``v`` can be
# another month; absent when monitor has not written it).
OPS_FILES = ("ops_gates", "ops_evaluation", "ops_status")
V_PATTERN = re.compile(r"^20(2[1-6])-(0[1-9]|1[0-2])$")  # vintages 2021-2026 only

Table = Literal["facility_scores", "facility_reasons", "citations", "penalties", "chain_scores"]

# The columns each table must have; a file may carry more. ``p`` is the calibrated probability of the champion's
# label (``p_<label>`` in scores/); build-evidence renames it so the tools need not know the label.
TABLE_COLUMNS: dict[Table, tuple[str, ...]] = {
    "facility_scores": (
        "ccn",
        "state",
        "chain_id",
        "p",
        "risk_percentile",
        "risk_decile",
        "model_version",
        "gold_run_id",
    ),
    "facility_reasons": (
        "ccn",
        "direction",
        "rank",
        "feature",
        "feature_value",
        "contribution_logodds",
        "label",
        "model_version",
        "gold_run_id",
    ),
    "citations": (
        "ccn",
        "survey_date",
        "ftag",
        "deficiency_category",
        "scope_severity",
        "is_standard",
        "is_complaint",
        "correction_date",
    ),
    "penalties": (
        "ccn",
        "penalty_date",
        "penalty_type",
        "fine_amount",
        "payment_denial_start_date",
        "payment_denial_length_days",
        "dup_seq",
    ),
    "chain_scores": (
        "chain_id",
        "n_facilities",
        "beds",
        "expected_facilities_with_event",
        "p_any_event",
        "bed_weighted_mean_p",
        "max_p",
        "worst_ccn",
        "share_top_decile",
        "model_version",
        "gold_run_id",
    ),
}

# Columns whose type the tools rely on: a CCN read as a number loses its leading zeros and matches nothing.
COLUMN_TYPES: dict[str, type[pl.DataType]] = {
    "ccn": pl.String,
    "worst_ccn": pl.String,
    "chain_id": pl.String,
    "survey_date": pl.Date,
    "correction_date": pl.Date,
    "penalty_date": pl.Date,
    "payment_denial_start_date": pl.Date,
}


def build_dir(prefix: str, v: str, build: str) -> str:
    return f"{prefix}v={v}/build={build}/"


class CurrentPointer(BaseModel):
    """``_current.json``: the month the agents serve."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    v: str
    build: str  # the folder v=YYYY-MM/build=<build>/
    scores_manifest_sha256: str
    evidence_manifest_sha256: str


class EvidenceManifest(BaseModel):
    """``v=YYYY-MM/evidence_manifest.json``."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    v: str
    gold_run_id: str
    model_version: int
    source: str  # the ML outputs' root prefix the scores came from: "" in production, "ml-dev/" from a laptop
    gold_verified: bool  # from the scores manifest; false for a month built from an unverified gold
    built_at: str
    files: dict[str, str]  # file name -> object key
    rows: dict[str, int]  # tables only
    sha256: dict[str, str]
