"""A small published scores month for build-evidence (a module of its own, not conftest.py: `from conftest import`
resolves to whichever conftest.py was loaded first): the four score files with their manifest, the ml_runs rows of
the gate that registered the model and of publish, that gate run's evaluation.json, status/latest.json, and toy
silver for the month."""

import hashlib
import io
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parents[1] / "agents"))  # evidence_fakes

from evidence_fakes import write_toy_silver  # noqa: E402

from precursorintelligence.common.config import load_agents_config, load_ml_config  # noqa: E402
from precursorintelligence.common.io.metadata_store import SqlAlchemyMetadataStore  # noqa: E402
from precursorintelligence.common.io.object_store import LocalObjectStore  # noqa: E402

V = "2026-10"
GOLD = "gold-20261015T000000Z"
GATE_RUN = "gate-run-1"
SCORE_RUN = "score-run-1"
VERSION = 2
CCNS = ["015009", "01A193", "450123"]


@dataclass
class Month:
    store: LocalObjectStore
    meta: SqlAlchemyMetadataStore
    ml_cfg: object
    cfg: object
    kept: dict


def _parquet(df: pl.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.write_parquet(buf)
    return buf.getvalue()


def write_scores(store: LocalObjectStore, v: str = V) -> None:
    prov = {"model_version": pl.lit(VERSION), "gold_run_id": pl.lit(GOLD)}
    tables = {
        "facility_scores": pl.DataFrame(
            {
                "ccn": CCNS,
                "state": ["AL", "AL", "TX"],
                "chain_id": ["C1", "C1", None],
                "p_g12": [0.4, 0.1, 0.2],
                "risk_percentile": [100.0, 33.3, 66.7],
                "risk_decile": [1, 7, 4],
                "logit": [0.0, 0.0, 0.0],
            }
        ).with_columns(**prov),
        "facility_reasons": pl.DataFrame(
            {
                "ccn": ["015009", "015009"],
                "direction": ["raises", "lowers"],
                "rank": [1, 1],
                "feature": ["a", "b"],
                "feature_value": ["1", "2"],
                "contribution_logodds": [0.3, -0.1],
                "label": ["A", "B"],
            }
        ).with_columns(**prov),
        "shap_values": pl.DataFrame({"ccn": CCNS, "base_value": [0.0] * 3}).with_columns(**prov),
        "chain_scores": pl.DataFrame(
            {
                "chain_id": ["C1"],
                "n_facilities": [2],
                "beds": [200.0],
                "expected_facilities_with_event": [0.5],
                "p_any_event": [0.46],
                "bed_weighted_mean_p": [0.25],
                "max_p": [0.4],
                "worst_ccn": ["015009"],
                "share_top_decile": [0.5],
            }
        ).with_columns(**prov),
    }
    directory = f"scores/v={v}/"
    files, rows, sha = {}, {}, {}
    for name, df in tables.items():
        data = _parquet(df)
        key = f"{directory}{name}.parquet"
        store.put_bytes(key, data, overwrite=True)
        files[name], rows[name], sha[name] = key, df.height, hashlib.sha256(data).hexdigest()
    manifest = {
        "v": v,
        "v_labels": v,
        "label": "g12",
        "model_name": "precursor_y_g12",
        "model_version": VERSION,
        "candidate": "ensemble",
        "gold_run_id": GOLD,
        "dataset_version": "d1",
        "gold_verified": True,
        "stale_refresh": False,
        "ml_run_id": SCORE_RUN,
        "scored_at": "2026-10-16T00:00:00Z",
        "files": files,
        "rows": rows,
        "sha256": sha,
    }
    store.put_bytes(
        f"{directory}scores_manifest.json", json.dumps(manifest).encode(), overwrite=True
    )


def make_month(tmp_path: Path) -> Month:
    store = LocalObjectStore(tmp_path / "bucket", immutable=None)
    meta = SqlAlchemyMetadataStore(f"sqlite:///{(tmp_path / 'meta.db').as_posix()}")
    meta.ensure_schema()
    write_scores(store)
    gates = {
        "schema_version": 1,
        "results": [{"gate": "G1", "status": "pass", "reason": "ok", "values": {}}],
    }
    meta.append(
        "ml_runs",
        [
            {
                "ml_run_id": GATE_RUN,
                "action": "promote",
                "v": "2026-09",
                "candidate_version": VERSION,
                "champion_version_after": VERSION,
                "gate_outcome": "pass",
                "gate_results_json": json.dumps(gates),
            },
            {
                "ml_run_id": SCORE_RUN,
                "action": "score",
                "v": V,
                "scores_published": True,
                "champion_version_after": VERSION,
            },
        ],
    )
    store.put_bytes(f"ml/runs/{GATE_RUN}/evaluation.json", b'{"selected": "ensemble"}')
    store.put_bytes("status/latest.json", json.dumps({"v": V, "flags": ["drift:x"]}).encode())
    kept = write_toy_silver(store, V, CCNS)
    return Month(store, meta, load_ml_config(), load_agents_config().evidence, kept)
