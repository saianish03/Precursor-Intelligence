"""Toy evidence files, laid out as ``build-evidence`` writes them (agents/tools/source.py). A module of its own
(not conftest.py), because the repo has several conftest.py files and `from conftest import ...` resolves to whichever
was loaded first."""

import hashlib
import io
import json
from datetime import date
from typing import Any

import polars as pl

from precursorintelligence.common.io.object_store import LocalObjectStore

PREFIX = "agents/evidence/"
V = "2026-10"
GOLD = "gold-20261015T000000Z"
VERSION = 3


def toy_tables() -> dict[str, pl.DataFrame]:
    prov = {"model_version": VERSION, "gold_run_id": GOLD}
    scores = pl.DataFrame(
        {
            "ccn": ["015009", "01A193", "450123", "450124"],
            "state": ["AL", "AL", "TX", None],
            "chain_id": ["C1", "C1", None, "C2"],
            "p": [0.42, 0.10, 0.25, 0.05],
            "risk_percentile": [100.0, 50.0, 75.0, 25.0],
            "risk_decile": [1, 5, 3, 8],
        }
    ).with_columns(**{k: pl.lit(v) for k, v in prov.items()})
    reasons = pl.DataFrame(
        {
            "ccn": ["015009"] * 3,
            "direction": ["lowers", "raises", "raises"],
            "rank": [1, 2, 1],
            "feature": ["rn_hours", "cit_g_36m", "complaints_12m"],
            "feature_value": ["4.1", "3", "7"],
            "contribution_logodds": [-0.20, 0.31, 0.55],
            "label": ["RN hours per resident day", "Serious citations, 36 months", "Complaints"],
        }
    ).with_columns(**{k: pl.lit(v) for k, v in prov.items()})
    citations = pl.DataFrame(
        {
            "ccn": ["015009", "015009", "450123"],
            "survey_date": [date(2025, 3, 4), date(2026, 1, 20), date(2025, 6, 1)],
            "ftag": ["F0689", "F0880", "F0689"],
            "deficiency_category": ["Quality of Life and Care Deficiencies"] * 3,
            "scope_severity": ["G", "D", "E"],
            "is_standard": [True, False, True],
            "is_complaint": [False, True, False],
            "correction_date": [date(2025, 4, 1), None, None],
        }
    )
    penalties = pl.DataFrame(
        {
            "ccn": ["015009"] * 3,
            "penalty_date": [date(2025, 5, 1)] * 2 + [date(2025, 9, 1)],
            "penalty_type": ["Fine", "Fine", "Payment Denial"],
            "fine_amount": [6500.0, 6500.0, None],
            "payment_denial_start_date": [None, None, date(2025, 9, 1)],
            "payment_denial_length_days": [None, None, 30.0],
            "dup_seq": [2, 1, 1],
        }
    )
    chains = pl.DataFrame(
        {
            "chain_id": ["C1", "C2"],
            "n_facilities": [2, 1],
            "beds": [220.0, 90.0],
            "expected_facilities_with_event": [0.52, 0.05],
            "p_any_event": [0.478, 0.05],
            "bed_weighted_mean_p": [0.30, None],
            "max_p": [0.42, 0.05],
            "worst_ccn": ["015009", "450124"],
            "share_top_decile": [0.5, 0.0],
        }
    ).with_columns(**{k: pl.lit(v) for k, v in prov.items()})
    return {
        "facility_scores": scores,
        "facility_reasons": reasons,
        "citations": citations,
        "penalties": penalties,
        "chain_scores": chains,
    }


def parquet(df: pl.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.write_parquet(buf)
    return buf.getvalue()


def write_evidence(
    store: LocalObjectStore,
    tables: dict[str, pl.DataFrame] | None = None,
    *,
    v: str = V,
    manifest_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write one month's evidence, then its manifest, then ``_current.json`` last. Returns the manifest."""
    tables = toy_tables() if tables is None else tables
    directory = f"{PREFIX}v={v}/"
    files, rows, sha = {}, {}, {}
    for name, df in tables.items():
        data = parquet(df)
        key = f"{directory}{name}.parquet"
        store.put_bytes(key, data, overwrite=True)
        files[name], rows[name], sha[name] = key, df.height, hashlib.sha256(data).hexdigest()
    manifest = {
        "schema_version": 1,
        "v": v,
        "gold_run_id": GOLD,
        "model_version": VERSION,
        "built_at": "2026-10-16T00:00:00Z",
        "files": files,
        "rows": rows,
        "sha256": sha,
        **(manifest_overrides or {}),
    }
    raw = json.dumps(manifest).encode()
    store.put_bytes(f"{directory}evidence_manifest.json", raw, overwrite=True)
    pointer = {
        "schema_version": 1,
        "v": v,
        "scores_manifest_sha256": "0" * 64,
        "evidence_manifest_sha256": hashlib.sha256(raw).hexdigest(),
    }
    store.put_bytes(f"{PREFIX}_current.json", json.dumps(pointer).encode(), overwrite=True)
    return manifest
