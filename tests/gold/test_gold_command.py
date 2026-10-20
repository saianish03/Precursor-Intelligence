"""`precursor-pipeline gold` on the toy silver world (SILVER_CONTRACT §11; spec §9.2, §11, §15.3)."""

import hashlib
import json
import sys

import polars as pl
import pytest
from make_toy_silver import make_toy

from precursorintelligence.common.io.metadata_store import SqlAlchemyMetadataStore
from precursorintelligence.common.io.object_store import LocalObjectStore
from precursorintelligence.common.io.run_lock import RunLock
from precursorintelligence.harmonize.cli import main
from precursorintelligence.ingestion.context import new_run_id
from precursorintelligence.ingestion.pipeline import (
    EXIT_CONFIG,
    EXIT_ERROR,
    EXIT_OK,
    EXIT_VALIDATION,
    LOCK_NAME,
)

RUN = "manual__2026-10-19T120000"  # an Airflow-style id without ':' (local-disk keys on Windows)


def run(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["precursor-pipeline", *args])
    with pytest.raises(SystemExit) as exc:
        main()
    return exc.value.code


class Toy:
    def __init__(self, root, stale=()):
        self.root = root
        self.env_file = str(make_toy(root, stale))
        self.store = LocalObjectStore(root, immutable=None)
        self.meta = SqlAlchemyMetadataStore(
            f"sqlite:///{(root / 'ops' / 'metadata.db').as_posix()}"
        )

    def gold(self, monkeypatch, *extra):
        return run(monkeypatch, "gold", "--env-file", self.env_file, *extra)

    def read(self, name):
        return pl.read_parquet(self.root / "gold" / name / "part-0.parquet")

    def sha(self, name):
        return hashlib.sha256(
            (self.root / "gold" / name / "part-0.parquet").read_bytes()
        ).hexdigest()

    def manifests(self):
        return [json.loads((self.root / k).read_text()) for k in self.store.list("metadata/gold")]


@pytest.fixture
def toy(tmp_path):
    return Toy(tmp_path / "toy")


def test_help_lists_the_gold_command(monkeypatch, capsys):
    assert run(monkeypatch, "--help") == EXIT_OK
    assert "gold" in capsys.readouterr().out


def test_gold_builds_spine_and_labels_and_records_the_run(toy, monkeypatch, capsys):
    assert toy.gold(monkeypatch) == EXIT_OK
    out = capsys.readouterr().out
    assert "V=2025-03 labels measured from 2025-03" in out
    assert "gold/spine: 109 rows" in out and "gold/labels: 109 rows" in out
    labels = toy.read("labels")
    maple = labels.filter(pl.col("ccn") == "015004", pl.col("t").is_in(["2023-02", "2023-03"]))
    assert maple.select("t", "y_g12").rows() == [("2023-02", 0), ("2023-03", 1)]
    (m,) = toy.manifests()
    assert (m["exit_code"], m["v"], m["v_labels"], m["promoted"]) == (0, "2025-03", "2025-03", True)
    assert m["outputs_sha256"] == {"spine": toy.sha("spine"), "labels": toy.sha("labels")}


def test_a_rerun_with_the_same_silver_gives_byte_identical_gold(toy, monkeypatch):
    assert toy.gold(monkeypatch) == EXIT_OK
    first = (toy.sha("spine"), toy.sha("labels"))
    assert toy.gold(monkeypatch) == EXIT_OK
    assert (toy.sha("spine"), toy.sha("labels")) == first
    assert len(toy.manifests()) == 2  # one manifest per run


def test_a_stale_v_measures_labels_from_the_last_fresh_vintage(tmp_path, monkeypatch, capsys):
    toy = Toy(tmp_path / "toy", stale=("2025-03",))
    assert toy.gold(monkeypatch) == EXIT_OK
    assert "V=2025-03 labels measured from 2025-02" in capsys.readouterr().out
    labels = toy.read("labels")
    assert labels.height == 109  # the stale month keeps its spine rows
    # t + 12 + 8 = 2025-03 is mature only if V is fresh: with V stale it is immature, not negative.
    sunny = labels.filter(pl.col("ccn") == "015001", pl.col("t") == "2023-07")
    assert sunny.select("status_g12", "y_g12").rows() == [("immature", None)]
    assert toy.manifests()[0]["stale_refresh_skipped"] == ["2025-03"]


def test_missing_vintage_flags_stops_instead_of_guessing(toy, monkeypatch, capsys):
    toy.meta.delete_where("vintage_flags", {"flag": "stale_refresh"})
    assert toy.gold(monkeypatch) == EXIT_VALIDATION
    assert "vintage_flags has no stale_refresh row for 2025-03" in capsys.readouterr().err
    assert not (toy.root / "gold" / "labels").exists()


@pytest.mark.parametrize(
    ("v", "message"),
    [
        ("2025-7", "is not a YYYY-MM month"),
        (
            "2025-04",
            "is after the latest published vintage 2025-03",
        ),  # in the future for this silver
        ("2023-12", "was never published"),
    ],
)
def test_a_bad_v_is_a_config_error(toy, monkeypatch, capsys, v, message):
    assert toy.gold(monkeypatch, "--v", v) == EXIT_CONFIG
    assert message in capsys.readouterr().err
    assert not (toy.root / "gold" / "labels").exists()


def test_an_earlier_v_builds_the_spine_only_up_to_it(toy, monkeypatch, capsys):
    assert toy.gold(monkeypatch, "--v", "2024-06") == EXIT_OK
    assert "V=2024-06 labels measured from 2024-06" in capsys.readouterr().out
    assert toy.read("spine")["t"].max() == "2024-06"


def test_a_failed_check_leaves_gold_untouched(toy, monkeypatch, capsys):
    assert toy.gold(monkeypatch) == EXIT_OK
    before = (toy.sha("spine"), toy.sha("labels"))
    presence = toy.root / "silver" / "facility_presence" / "part-0.parquet"
    pl.read_parquet(presence).filter(pl.col("ccn") != "015005").write_parquet(presence)
    assert toy.gold(monkeypatch) == EXIT_VALIDATION
    assert "spine CCNs missing from facility_presence" in capsys.readouterr().err
    assert (toy.sha("spine"), toy.sha("labels")) == before
    assert [m["promoted"] for m in toy.manifests()].count(False) == 1


def test_a_dag_task_needs_its_run_to_hold_the_lock(toy, monkeypatch):
    assert toy.gold(monkeypatch, "--run-id", RUN) == EXIT_CONFIG  # the lock task never ran
    RunLock(toy.meta, LOCK_NAME).acquire(RUN)
    assert toy.gold(monkeypatch, "--run-id", RUN) == EXIT_OK
    assert RunLock(toy.meta, LOCK_NAME).holder()["run_id"] == RUN  # the DAG's last task releases it


def test_a_local_run_while_another_run_holds_the_lock_is_a_retryable_error(toy, monkeypatch):
    RunLock(toy.meta, LOCK_NAME).acquire("someone-else")
    assert toy.gold(monkeypatch) == EXIT_ERROR
    assert not (toy.root / "gold" / "labels").exists()


def test_a_local_run_generates_a_windows_safe_run_id(toy, monkeypatch):
    assert ":" not in new_run_id()
    assert toy.gold(monkeypatch) == EXIT_OK
    (m,) = toy.manifests()
    assert ":" not in m["run_id"]
    assert RunLock(toy.meta, LOCK_NAME).holder() is None  # taken and released by the run itself
