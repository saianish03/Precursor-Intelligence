"""E2: build-evidence writes a month the agents' reader accepts, from published scores and the month's silver only,
and moves the pointer last."""

import json
from datetime import date

import polars as pl
import pytest
from evidence_fakes import parquet
from month_fakes import CCNS, GATE_RUN, GOLD, VERSION, V

from precursorintelligence.agents.tools.source import ParquetEvidenceSource
from precursorintelligence.common.config import ConfigError, load_agents_config
from precursorintelligence.common.io.object_store import LocalObjectStore
from precursorintelligence.evidence.build import run_build_evidence
from precursorintelligence.training import MlInputError

PREFIX = "agents/evidence/"
POINTER = f"{PREFIX}_current.json"


def build(month, v=V, built_at="t", ml_cfg=None):
    return run_build_evidence(
        month.store,
        month.meta,
        v=v,
        ml_cfg=ml_cfg or month.ml_cfg,
        cfg=month.cfg,
        built_at=built_at,
    )


def served(store):
    """The folder of the build ``_current.json`` names."""
    pointer = json.loads(store.open_read(POINTER).read())
    return f"{PREFIX}v={pointer['v']}/build={pointer['build']}/"


def serve_as(store, v=None, **manifest):
    """Rewrite the served build as month ``v`` and/or with other manifest fields (a stand-in for another build)."""
    pointer = json.loads(store.open_read(POINTER).read())
    raw = json.loads(store.open_read(f"{served(store)}evidence_manifest.json").read()) | manifest
    pointer["v"] = v or pointer["v"]
    target = f"{PREFIX}v={pointer['v']}/build={pointer['build']}/evidence_manifest.json"
    store.put_bytes(target, json.dumps(raw).encode(), overwrite=True)
    store.put_bytes(POINTER, json.dumps(pointer).encode(), overwrite=True)


def test_builds_a_month_the_reader_accepts(month):
    out = build(month)
    source = ParquetEvidenceSource.open(month.store, PREFIX)
    m = source.manifest
    assert (m.v, m.gold_run_id, m.model_version, m.source, m.gold_verified) == (
        V,
        GOLD,
        VERSION,
        "",
        True,
    )
    scores = source.rows("facility_scores", "ccn", CCNS)
    assert sorted(scores["p"].to_list()) == [0.1, 0.2, 0.4]  # p_g12 renamed p
    assert out.files == sorted(
        [
            "facility_scores",
            "facility_reasons",
            "chain_scores",
            "citations",
            "penalties",
            "ops_gates",
            "ops_evaluation",
            "ops_status",
        ]
    )


def test_citations_and_penalties_keep_the_window_and_the_scored_facilities(month):
    out = build(month)
    assert out.rows["citations"] == month.kept["citations"]
    assert out.rows["penalties"] == month.kept["penalties"]
    source = ParquetEvidenceSource.open(month.store, PREFIX)
    cit = source.rows("citations", "ccn", [*CCNS, "ZZ9999"])
    assert "ZZ9999" not in cit["ccn"].to_list()  # not scored
    assert cit["survey_date"].max() == date(2026, 10, 1)  # the cutoff itself is in the window
    assert cit["survey_date"].min() > date(2023, 10, 1)  # 36 months before the cutoff is not


def test_ops_evidence_comes_from_the_gate_that_registered_the_model(month):
    build(month)
    d = served(month.store)
    gates = json.loads(month.store.open_read(f"{d}ops_gates.json").read())
    assert (gates["ml_run_id"], gates["model_version"], gates["action"]) == (
        GATE_RUN,
        VERSION,
        "promote",
    )
    assert gates["gate_results"]["results"][0]["gate"] == "G1"
    assert json.loads(month.store.open_read(f"{d}ops_evaluation.json").read()) == {
        "selected": "ensemble"
    }
    assert json.loads(month.store.open_read(f"{d}ops_status.json").read())["flags"] == ["drift:x"]


def test_the_pointer_is_written_last(month):
    keys = []

    class Recording(LocalObjectStore):
        def put_bytes(self, key, data, *, overwrite=False):
            keys.append(key)
            return super().put_bytes(key, data, overwrite=overwrite)

    month.store = Recording(month.store.root, immutable=None)
    build(month)
    assert keys[-1] == f"{PREFIX}_current.json"
    assert keys[-2] == f"{served(month.store)}evidence_manifest.json"


def test_unpublished_scores_are_refused_and_nothing_is_written(month):
    month.meta.engine.dispose()
    with month.meta.engine.begin() as conn:
        conn.exec_driver_sql("DELETE FROM ml_runs WHERE action = 'score'")
    with pytest.raises(MlInputError, match="no published scores"):
        build(month)
    assert month.store.list(PREFIX) == []


def test_changed_scores_are_refused(month):
    month.store.put_bytes(f"scores/v={V}/chain_scores.parquet", b"changed", overwrite=True)
    with pytest.raises(MlInputError, match="sha256"):
        build(month)


def test_a_missing_silver_partition_is_refused(month):
    month.store.local_path(f"silver/penalties_by_vintage/vintage={V}/part-0.parquet").unlink()
    with pytest.raises(MlInputError, match="no partition for vintage"):
        build(month)


def test_the_pointer_never_moves_to_an_older_month(month):
    build(month)
    serve_as(month.store, v="2026-11")
    with pytest.raises(MlInputError, match="already serves 2026-11"):
        build(month)


def test_the_pointer_is_checked_again_right_before_it_moves(month):
    build(month)

    class Racing(LocalObjectStore):
        raced = False

        def put_bytes(self, key, data, *, overwrite=False):
            out = super().put_bytes(key, data, overwrite=overwrite)
            if key.endswith("evidence_manifest.json") and not Racing.raced:
                Racing.raced = True
                serve_as(self, v="2026-11")  # another build moved the pointer meanwhile
            return out

    month.store = Racing(month.store.root, immutable=None)
    with pytest.raises(MlInputError, match="already serves 2026-11"):
        build(month, built_at="later")


def test_a_rebuild_goes_to_a_new_folder_and_moves_the_pointer(month):
    build(month)
    first = served(month.store)
    build(month, built_at="later")
    assert served(month.store) != first
    assert month.store.exists(f"{first}citations.parquet")  # the old build is never changed
    assert ParquetEvidenceSource.open(month.store, PREFIX).manifest.built_at == "later"


def test_a_rebuild_that_fails_halfway_leaves_the_served_month_readable(month):
    build(month)

    class Failing(LocalObjectStore):
        writes = 0

        def put_bytes(self, key, data, *, overwrite=False):
            Failing.writes += 1
            if Failing.writes == 3:
                raise OSError("network dropped")
            return super().put_bytes(key, data, overwrite=overwrite)

    month.store = Failing(month.store.root, immutable=None)
    with pytest.raises(OSError):
        build(month, built_at="later")
    source = ParquetEvidenceSource.open(month.store, PREFIX)
    assert source.manifest.built_at == "t"
    assert source.rows("citations", "ccn", CCNS).height == month.kept["citations"]


def test_a_development_build_never_replaces_a_production_one(month):
    build(month)
    dev = month.ml_cfg.model_copy(
        update={"outputs": month.ml_cfg.outputs.model_copy(update={"root_prefix": "ml-dev/"})}
    )
    with pytest.raises(MlInputError, match="production build"):
        build(month, ml_cfg=dev)


def test_a_production_build_replaces_a_development_one(month):
    build(month)
    serve_as(month.store, source="ml-dev/")  # the demo fallback, built from ml-dev/
    build(month, built_at="later")
    assert ParquetEvidenceSource.open(month.store, PREFIX).manifest.source == ""


def test_an_unreadable_pointer_is_a_refusal(month):
    month.store.put_bytes(POINTER, b"{not json", overwrite=True)
    with pytest.raises(MlInputError, match="unreadable"):
        build(month)


def test_a_cutoff_date_other_than_the_month_is_refused(month):
    key = f"silver/citations_by_vintage/vintage={V}/part-0.parquet"
    df = pl.read_parquet(month.store.local_path(key))
    month.store.put_bytes(
        key, parquet(df.with_columns(cutoff_date=pl.lit(date(2026, 9, 1)))), overwrite=True
    )
    with pytest.raises(MlInputError, match="cutoff_date"):
        build(month)


def test_a_missing_evaluation_is_refused_and_nothing_is_written(month):
    month.store.local_path(f"ml/runs/{GATE_RUN}/evaluation.json").unlink()
    with pytest.raises(MlInputError, match="evaluation"):
        build(month)
    assert month.store.list(PREFIX) == []


def test_without_a_status_the_build_has_no_ops_status(month):
    month.store.local_path("status/latest.json").unlink()
    assert "ops_status" not in build(month).files


def test_ops_gates_follow_the_scoring_model_not_a_later_gate(month):
    month.meta.append(
        "ml_runs",
        [
            {"ml_run_id": "later-hold", "action": "hold", "v": V, "candidate_version": VERSION + 1},
            {
                "ml_run_id": "later-promote",
                "action": "promote",
                "v": V,
                "candidate_version": VERSION + 2,
            },
        ],
    )
    build(month)
    gates = json.loads(month.store.open_read(f"{served(month.store)}ops_gates.json").read())
    assert (gates["ml_run_id"], gates["model_version"]) == (GATE_RUN, VERSION)


def test_agents_config_refuses_unknown_keys_and_bad_prefixes(tmp_path):
    good = "version: 1\nevidence: {prefix: agents/evidence/, citation_months: 36, penalty_months: 36}\n"
    for bad in (good.replace("36}", "36, extra: 1}"), good.replace("agents/evidence/", "/abs")):
        path = tmp_path / "agents.yaml"
        path.write_text(bad)
        with pytest.raises(ConfigError):
            load_agents_config(path)
