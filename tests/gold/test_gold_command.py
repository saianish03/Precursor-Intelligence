"""`precursor-pipeline gold` on the toy silver world (SILVER_CONTRACT §11; spec §9.2, §11, §15.3)."""

import hashlib
import json
import sys

import polars as pl
import pytest
from make_toy_silver import make_toy

from precursorintelligence.common.config import (
    load_labels_config,
    load_settings,
    load_splits_config,
)
from precursorintelligence.common.io.metadata_store import SqlAlchemyMetadataStore
from precursorintelligence.common.io.object_store import LocalObjectStore
from precursorintelligence.common.io.run_lock import RunLock
from precursorintelligence.gold import GoldCheckError
from precursorintelligence.gold.build import read_current_gold, run_gold
from precursorintelligence.harmonize.cli import main
from precursorintelligence.ingestion.context import build_context, new_run_id
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
    assert m["outputs_sha256"] == {name: toy.sha(name) for name in m["outputs_sha256"]}
    assert {"spine", "labels", "facility_month_dataset/t=2025-03"} <= set(m["outputs_sha256"])


def test_gold_builds_the_features_and_the_dataset(toy, monkeypatch, capsys):
    assert toy.gold(monkeypatch) == EXIT_OK
    out = capsys.readouterr().out
    assert (
        "gold/features_citations: 109 rows" in out
        and "gold/facility_month_dataset: 109 rows" in out
    )
    months = sorted(set(toy.read("spine")["t"]))
    for table in ("features_enforcement", "facility_month_dataset"):
        parts = sorted(p.name.removeprefix("t=") for p in (toy.root / "gold" / table).iterdir())
        assert parts == months
    ds = toy.read("facility_month_dataset/t=2024-01")
    assert ds.columns[:3] == ["ccn", "t", "cutoff_date"]
    assert ds.columns[-4:] == ["status_pen12", "split_g12", "split_g6", "split_pen12"]
    oak = ds.filter(pl.col("ccn") == "015002").row(0, named=True)
    assert (oak["n_fines_24m"], oak["y_g12"]) == (2, None)  # immature at t = 2024-01
    manifest = json.loads((toy.root / "gold" / "dataset_manifest.json").read_text())
    assert manifest["feature_families_pending"] == ["quality"]
    assert manifest["flags"] == {} and manifest["rows"] == {"total": 109, "months": 26}
    assert set(manifest["partitions_sha256"]) == {f"facility_month_dataset/t={t}" for t in months}


def test_gold_marks_its_outputs_unverified_until_the_section_11_gate_exists(toy, monkeypatch):
    assert toy.gold(monkeypatch) == EXIT_OK
    manifest = json.loads((toy.root / "gold" / "dataset_manifest.json").read_text())
    assert manifest["verified"] is False
    assert read_current_gold(toy.store)["verified"] is False
    (m,) = toy.manifests()
    assert m["verified"] is False


def test_a_rerun_with_the_same_silver_gives_byte_identical_gold(toy, monkeypatch):
    assert toy.gold(monkeypatch) == EXIT_OK
    first = (toy.sha("spine"), toy.sha("labels"))
    assert toy.gold(monkeypatch) == EXIT_OK
    assert (toy.sha("spine"), toy.sha("labels")) == first
    assert len(toy.manifests()) == 2  # one manifest per run
    a, b = toy.manifests()
    assert a["outputs_sha256"] == b["outputs_sha256"]  # every feature and dataset partition too


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


def test_gold_checks_that_silver_is_current_first(toy, monkeypatch):
    assert toy.gold(monkeypatch) == EXIT_OK
    check = toy.manifests()[0]["every_bronze_partition_has_current_silver"]
    assert check == {"passed": True, "missing": []}


def test_stale_silver_stops_gold_and_leaves_gold_untouched(toy, monkeypatch, capsys):
    # The latest silver build of one partition is gone, as after a failed silver rebuild.
    toy.meta.delete_where(
        "silver_partitions", {"table_name": "citations_by_vintage", "vintage_month": "2024-03"}
    )
    assert toy.gold(monkeypatch) == EXIT_VALIDATION
    err = capsys.readouterr().err
    assert "silver is not current for 1 partitions" in err and "citations_by_vintage 2024-03" in err
    assert not (toy.root / "gold" / "labels").exists()
    check = toy.manifests()[0]["every_bronze_partition_has_current_silver"]
    assert check == {"passed": False, "missing": ["citations_by_vintage 2024-03"]}


def test_a_metadata_database_without_bronze_records_stops_gold(toy, monkeypatch, capsys):
    toy.meta.delete_where("bronze_files", {"run_id": "toy"})
    assert toy.gold(monkeypatch) == EXIT_VALIDATION
    assert "has no bronze_files rows" in capsys.readouterr().err
    assert not (toy.root / "gold" / "labels").exists()


def _gold_run_log(toy, exit_code):
    """run_log status of the gold run that ended with ``exit_code`` (manifest keys don't sort by time
    when two runs start in the same second)."""
    (m,) = [m for m in toy.manifests() if m["exit_code"] == exit_code]
    (row,) = [r for r in toy.meta.read("run_log", {"run_id": m["run_id"]}) if r["task"] == "gold"]
    return row["status"]


def test_run_log_records_a_failed_gold_run_as_failed(toy, monkeypatch):
    assert toy.gold(monkeypatch) == EXIT_OK
    toy.meta.delete_where(
        "silver_partitions", {"table_name": "citations_by_vintage", "vintage_month": "2024-03"}
    )
    assert toy.gold(monkeypatch) == EXIT_VALIDATION
    assert _gold_run_log(toy, EXIT_OK) == "ok"
    assert _gold_run_log(toy, EXIT_VALIDATION) == "failed"


def test_a_failure_during_promotion_is_never_read_as_a_valid_gold(toy, monkeypatch):
    assert toy.gold(monkeypatch) == EXIT_OK
    assert read_current_gold(toy.store)["state"] == "current"

    # Second run: the labels write fails after the spine has been replaced.
    ctx = build_context(load_settings(env_path=toy.env_file))
    real_put = ctx.object_store.put_bytes

    def put(key, data, **kw):
        if key == "gold/labels/part-0.parquet":
            raise OSError("disk full (injected)")
        return real_put(key, data, **kw)

    monkeypatch.setattr(ctx.object_store, "put_bytes", put)
    with pytest.raises(OSError, match="injected"):
        run_gold(ctx, labels_cfg=load_labels_config(), splits_cfg=load_splits_config(), v="2024-06")

    assert toy.read("spine")["t"].max() == "2024-06"  # new spine, old labels: torn
    assert toy.read("labels")["t"].max() == "2025-03"
    with pytest.raises(GoldCheckError, match="did not finish promoting"):
        read_current_gold(toy.store)
    (failed,) = [m for m in toy.manifests() if m["run_id"] == ctx.run_id]
    assert failed["promoted"] is False and "disk full (injected)" in failed["error"]

    assert toy.gold(monkeypatch) == EXIT_OK  # a rerun heals it
    sha = read_current_gold(toy.store)["sha256"]
    assert (sha["spine"], sha["labels"]) == (toy.sha("spine"), toy.sha("labels"))


def test_read_current_gold_refuses_files_that_changed_after_promotion(toy, monkeypatch):
    assert toy.gold(monkeypatch) == EXIT_OK
    labels = toy.root / "gold" / "labels" / "part-0.parquet"
    pl.read_parquet(labels).head(5).write_parquet(labels)
    with pytest.raises(GoldCheckError, match="does not match"):
        read_current_gold(toy.store)


def test_the_process_log_is_shipped_next_to_the_gold_run(toy, monkeypatch):
    assert toy.gold(monkeypatch) == EXIT_OK
    (m,) = toy.manifests()
    assert len(toy.store.list(f"reports/gold/run={m['run_id']}/logs")) == 1


def test_gold_labels_carry_a_split_column_per_label(toy, monkeypatch):
    assert toy.gold(monkeypatch) == EXIT_OK
    labels = toy.read("labels")
    assert [c for c in labels.columns if c.startswith("split_")] == [
        "split_g12",
        "split_g6",
        "split_pen12",
    ]
    for name in ("g12", "g6", "pen12"):  # score rows are exactly the immature ones
        assert labels.filter(pl.col(f"split_{name}") == "score").equals(
            labels.filter(pl.col(f"status_{name}") == "immature")
        )
    dataset = pl.read_parquet(
        toy.root / "gold" / "facility_month_dataset" / "t=2023-01" / "part-0.parquet"
    )
    assert {"split_g12", "split_g6", "split_pen12"} <= set(dataset.columns)
