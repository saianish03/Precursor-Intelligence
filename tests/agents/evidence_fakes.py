"""Toy evidence files, laid out as ``build-evidence`` writes them (evidence/contract.py), and toy silver for
``build-evidence`` to read (``write_toy_silver``; also used by scripts/dvc_toy_evidence.py). A module of its own
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
BUILD = "0123456789ab"
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
    build: str = BUILD,
    manifest_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write one build of a month, then its manifest, then ``_current.json`` last. Returns the manifest."""
    tables = toy_tables() if tables is None else tables
    directory = f"{PREFIX}v={v}/build={build}/"
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
        "source": "",
        "gold_verified": True,
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
        "build": build,
        "scores_manifest_sha256": "0" * 64,
        "evidence_manifest_sha256": hashlib.sha256(raw).hexdigest(),
    }
    store.put_bytes(f"{PREFIX}_current.json", json.dumps(pointer).encode(), overwrite=True)
    return manifest


def write_toy_silver(store: LocalObjectStore, v: str, ccns: list[str]) -> dict[str, int]:
    """Silver ``citations_by_vintage`` and ``penalties_by_vintage`` partitions of vintage ``v`` for the first three
    ``ccns`` (and one unscored CCN). Each facility gets rows inside the window (one on the cutoff itself) and rows
    ``build-evidence`` must drop: 36 months before the cutoff or older, after the cutoff, or of a facility it did not
    score. Returns the rows it must keep."""
    cutoff = date(int(v[:4]), int(v[5:7]), 1)

    def months_before(n: int, day: int = 15) -> date:
        i = cutoff.year * 12 + cutoff.month - 1 - n
        return date(i // 12, i % 12 + 1, day)

    after = date(cutoff.year, cutoff.month, 5)  # in the vintage file, but after the cutoff
    kept = {"citations": 0, "penalties": 0}
    cit, pen = [], []
    for i, ccn in enumerate([*ccns[:3], "ZZ9999"]):
        keep = ccn != "ZZ9999"
        for when, inside in (
            (cutoff, True),  # the window is (cutoff - 36 months, cutoff]
            (months_before(2), True),
            (months_before(35), True),
            (months_before(36, day=1), False),
            (months_before(37), False),
            (after, False),
        ):
            cit.append((ccn, when, f"F0{680 + i}", "G" if inside else "D"))
            pen.append((ccn, when, "Fine", 1000.0 * (i + 1)))
            if keep and inside:
                kept["citations"] += 1
                kept["penalties"] += 1
    common = {"vintage_month": pl.lit(v), "cutoff_date": pl.lit(cutoff)}
    citations = pl.DataFrame(
        cit,
        schema={
            "ccn": pl.String,
            "survey_date": pl.Date,
            "ftag": pl.String,
            "scope_severity": pl.String,
        },
        orient="row",
    ).with_columns(
        deficiency_prefix=pl.lit("F"),
        deficiency_tag=pl.col("ftag").str.slice(1),
        deficiency_category=pl.lit("Quality of Life and Care Deficiencies"),
        is_standard=pl.lit(True),
        is_complaint=pl.lit(False),
        correction_date=pl.lit(None, dtype=pl.Date),
        **common,
    )
    penalties = pl.DataFrame(
        pen,
        schema={
            "ccn": pl.String,
            "penalty_date": pl.Date,
            "penalty_type": pl.String,
            "fine_amount": pl.Float64,
        },
        orient="row",
    ).with_columns(
        payment_denial_start_date=pl.lit(None, dtype=pl.Date),
        payment_denial_length_days=pl.lit(None, dtype=pl.Float64),
        dup_seq=pl.lit(1, dtype=pl.Int64),
        **common,
    )
    for table, df in (("citations_by_vintage", citations), ("penalties_by_vintage", penalties)):
        store.put_bytes(f"silver/{table}/vintage={v}/part-0.parquet", parquet(df), overwrite=True)
    return kept
