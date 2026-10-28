"""`precursor-pipeline gold` on the toy silver world (SILVER_CONTRACT §11; spec §9.2, §11, §15.3)."""

import hashlib
import json
import sys

import polars as pl
import pytest
from make_toy_silver import TOY_SPLITS, make_toy, no_gate_splits

from precursorintelligence.common.config import (
    ConfigError,
    load_labels_config,
    load_settings,
    load_splits_config,
    project_root,
)
from precursorintelligence.common.io.metadata_store import SqlAlchemyMetadataStore
from precursorintelligence.common.io.object_store import LocalObjectStore
from precursorintelligence.common.io.run_lock import RunLock
from precursorintelligence.gold import GoldCheckError, build
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
        self.splits_file = str(root / TOY_SPLITS)
        self.store = LocalObjectStore(root, immutable=None)
        self.meta = SqlAlchemyMetadataStore(
            f"sqlite:///{(root / 'ops' / 'metadata.db').as_posix()}"
        )

    def gold(self, monkeypatch, *extra):
        return run(
            monkeypatch,
            "gold",
            "--env-file",
            self.env_file,
            "--splits-config",
            self.splits_file,
            *extra,
        )

    def read(self, name):
        return pl.read_parquet(self.root / "gold" / name / "part-0.parquet")

    def sha(self, name):
        path = self.root / "gold" / name
        if not name.endswith(".parquet"):
            path = path / "part-0.parquet"
        return hashlib.sha256(path.read_bytes()).hexdigest()

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
    toy.splits_file = str(no_gate_splits(toy.root))  # labels from 2025-02: no toy training month
    assert toy.gold(monkeypatch) == EXIT_OK
    assert "V=2025-03 labels measured from 2025-02" in capsys.readouterr().out
    labels = toy.read("labels")
    assert labels.height == 109  # the stale month keeps its spine rows
    # t + 12 + 8 = 2025-03 is mature only if V is fresh: with V stale it is immature, not negative.
    sunny = labels.filter(pl.col("ccn") == "015001", pl.col("t") == "2023-07")
    assert sunny.select("status_g12", "y_g12").rows() == [("immature", None)]
    (m,) = toy.manifests()
    assert m["stale_refresh_skipped"] == ["2025-03"]
    # Splits are measured from the same fresh vintage: score rows are exactly the immature ones.
    for name in ("g12", "g6", "pen12"):
        assert labels.filter(pl.col(f"split_{name}") == "score").equals(
            labels.filter(pl.col(f"status_{name}") == "immature")
        )
    assert (
        m["split_windows"]["g12"]["last_labeled_t"] == "2023-06"
    )  # 2025-02 - 20, not 2025-03 - 20


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
    toy.splits_file = str(no_gate_splits(toy.root))  # no toy training month at V = 2024-06
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
        run_gold(
            ctx,
            labels_cfg=load_labels_config(),
            splits_cfg=load_splits_config(
                no_gate_splits(toy.root)
            ),  # no toy training month at 2024-06
            v="2024-06",
        )

    assert toy.read("spine")["t"].max() == "2024-06"  # new spine, old labels: torn
    assert toy.read("labels")["t"].max() == "2025-03"
    with pytest.raises(GoldCheckError, match="did not finish promoting"):
        read_current_gold(toy.store)
    (failed,) = [m for m in toy.manifests() if m["run_id"] == ctx.run_id]
    assert failed["promoted"] is False and "disk full (injected)" in failed["error"]
    assert failed["exit_code"] == EXIT_ERROR  # what the CLI exits with when the run raises

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


def test_splits_config_is_read_from_the_given_file_and_hashed_in_the_dataset_manifest(
    toy, monkeypatch
):
    assert toy.gold(monkeypatch) == EXIT_OK
    manifest = json.loads((toy.root / "gold" / "dataset_manifest.json").read_text())
    used = load_splits_config(toy.root / TOY_SPLITS)
    assert manifest["config_sha256"]["splits.yaml"] == _settings_sha(used)
    assert manifest["config_sha256"]["splits.yaml"] != _settings_sha(load_splits_config())
    assert toy.manifests()[0]["configs"]["splits"]["required_labels"] == ["g6"]


def _settings_sha(cfg) -> str:
    return hashlib.sha256(
        json.dumps(cfg.model_dump(mode="json"), sort_keys=True).encode()
    ).hexdigest()


def test_the_dataset_manifest_hashes_the_splits_settings_passed_from_code(toy):
    # No --splits-config here: run_gold gets the toy settings directly, as the leakage tests call it.
    used = load_splits_config(toy.root / TOY_SPLITS)
    ctx = build_context(load_settings(env_path=toy.env_file))
    assert run_gold(ctx, labels_cfg=load_labels_config(), splits_cfg=used).exit_code == EXIT_OK
    manifest = json.loads((toy.root / "gold" / "dataset_manifest.json").read_text())
    assert manifest["config_sha256"]["splits.yaml"] == _settings_sha(used)


def test_a_missing_splits_config_is_a_config_error(toy, monkeypatch, capsys):
    toy.splits_file = str(toy.root / "no-such-splits.yaml")
    assert toy.gold(monkeypatch) == EXIT_CONFIG
    assert "config error" in capsys.readouterr().err


def test_gold_stops_when_a_required_label_has_no_training_month(toy, monkeypatch, capsys):
    toy.splits_file = str(
        project_root() / "configs" / "splits.yaml"
    )  # g12 required: no train month at toy V
    assert toy.gold(monkeypatch) == EXIT_VALIDATION
    err = capsys.readouterr().err
    assert "no training month for the required labels g12 (train window None)" in err
    assert not (toy.root / "gold" / "labels").exists()  # nothing promoted
    (m,) = toy.manifests()
    assert m["promoted"] is False and m["split_windows"]["g12"]["train"] is None
    assert m["split_windows"]["g6"]["train"] == ["2023-01", "2023-01"]


def test_gold_stops_when_a_required_labels_training_rows_are_all_censored(toy, monkeypatch, capsys):
    real = build.build_labels

    def censored_g6(*args, **kwargs):  # g6 trains on 2023-01 only: censor all its rows
        labels = real(*args, **kwargs)
        train = pl.col("t") == "2023-01"
        return labels.with_columns(
            pl.when(train).then(pl.lit(None, pl.Int8)).otherwise(pl.col("y_g6")).alias("y_g6"),
            pl.when(train)
            .then(pl.lit("censored"))
            .otherwise(pl.col("status_g6"))
            .alias("status_g6"),
        )

    monkeypatch.setattr(build, "build_labels", censored_g6)
    assert toy.gold(monkeypatch) == EXIT_VALIDATION
    err = capsys.readouterr().err
    assert (
        "required label g6: no labelled training rows in 2023-01..2023-01 "
        "(4 train rows, none positive or negative)" in err
    )
    assert "no training month" not in err  # the window is not empty
    assert not (toy.root / "gold" / "labels").exists()


def test_a_null_split_stops_gold_before_promotion(toy, monkeypatch, capsys):
    real = build.add_splits

    def one_null(*args, **kwargs):
        labels = real(*args, **kwargs)
        first = pl.int_range(pl.len()) == 0
        return labels.with_columns(
            pl.when(first).then(None).otherwise(pl.col("split_pen12")).alias("split_pen12")
        )

    monkeypatch.setattr(build, "add_splits", one_null)
    assert toy.gold(monkeypatch) == EXIT_VALIDATION
    assert "staged labels: null values in ['split_pen12']" in capsys.readouterr().err
    assert not (toy.root / "gold" / "labels").exists()


def test_a_required_label_missing_from_labels_yaml_is_a_config_error(toy, monkeypatch, capsys):
    path = toy.root / "splits_bad.yaml"
    path.write_text((toy.root / TOY_SPLITS).read_text().replace("- g6", "- g24"), encoding="utf-8")
    toy.splits_file = str(path)
    assert toy.gold(monkeypatch) == EXIT_CONFIG
    assert "required_labels ['g24'] are not labels in labels.yaml" in capsys.readouterr().err


@pytest.mark.parametrize("bad", ["required_label", "v"])
def test_the_run_manifest_records_the_clis_exit_code_for_a_config_error(toy, monkeypatch, bad):
    extra: tuple[str, ...] = ("--v", "2023-12")  # never published
    if bad == "required_label":
        path = toy.root / "splits_bad.yaml"
        path.write_text(
            (toy.root / TOY_SPLITS).read_text().replace("- g6", "- g24"), encoding="utf-8"
        )
        toy.splits_file, extra = str(path), ()
    code = toy.gold(monkeypatch, *extra)
    assert code == EXIT_CONFIG
    (m,) = toy.manifests()
    assert m["exit_code"] == code and m["promoted"] is False and "ConfigError" in m["error"]
    assert RunLock(toy.meta, LOCK_NAME).holder() is None  # checked before the lock was taken


def _config_error(*_a, **_k):
    raise ConfigError("broken config")


@pytest.mark.parametrize("loader", ["load_features_config", "load_silver_config"])
def test_a_broken_feature_or_silver_config_stops_gold_before_the_lock(toy, monkeypatch, loader):
    monkeypatch.setattr(build, loader, _config_error)
    code = toy.gold(monkeypatch)
    assert code == EXIT_CONFIG
    (m,) = toy.manifests()
    assert m["exit_code"] == code and m["promoted"] is False and "broken config" in m["error"]
    assert RunLock(toy.meta, LOCK_NAME).holder() is None
    assert not [
        r for r in toy.meta.read("run_log") if r["task"] == "gold"
    ]  # the step never started


def test_an_unexpected_error_before_the_lock_is_still_recorded(toy, monkeypatch):
    # e.g. a silver YAML syntax error, which load_silver_config does not wrap in ConfigError
    def broken(*_a, **_k):
        raise KeyError("raw")

    monkeypatch.setattr(build, "load_silver_config", broken)
    ctx = build_context(load_settings(env_path=toy.env_file))
    with pytest.raises(KeyError):
        run_gold(
            ctx, labels_cfg=load_labels_config(), splits_cfg=load_splits_config(toy.splits_file)
        )
    (m,) = toy.manifests()
    assert m["exit_code"] == EXIT_ERROR and "KeyError" in m["error"]
    assert RunLock(toy.meta, LOCK_NAME).holder() is None


def test_a_config_error_inside_the_run_is_recorded_as_exit_3(toy, monkeypatch):
    # e.g. silver published a month between the pre-lock --v check and the run
    monkeypatch.setattr(build, "label_vintage", _config_error)
    assert toy.gold(monkeypatch) == EXIT_CONFIG
    (m,) = toy.manifests()
    assert m["exit_code"] == EXIT_CONFIG and "broken config" in m["error"]


def test_a_g_plus_set_features_cannot_use_is_a_config_error_before_the_lock(toy):
    labels_cfg = load_labels_config().model_copy(update={"gplus_severities": ["G", "J", "K", "L"]})
    ctx = build_context(load_settings(env_path=toy.env_file))
    with pytest.raises(ConfigError, match="every letter above it"):
        run_gold(ctx, labels_cfg=labels_cfg, splits_cfg=load_splits_config(toy.root / TOY_SPLITS))
    (m,) = toy.manifests()
    assert m["exit_code"] == EXIT_CONFIG
    assert RunLock(toy.meta, LOCK_NAME).holder() is None
    assert not [
        r for r in toy.meta.read("run_log") if r["task"] == "gold"
    ]  # the step never started


def test_features_and_labels_use_the_g_plus_set_of_the_run(toy):
    # Only L is G+ here: features must agree with the labels, not with configs/labels.yaml (G-L).
    labels_cfg = load_labels_config().model_copy(update={"gplus_severities": ["L"]})
    ctx = build_context(load_settings(env_path=toy.env_file))
    splits_cfg = load_splits_config(no_gate_splits(toy.root))  # few L citations: no gate
    assert run_gold(ctx, labels_cfg=labels_cfg, splits_cfg=splits_cfg).exit_code == EXIT_OK
    cit = pl.concat(
        pl.read_parquet(p)
        for p in sorted((toy.root / "gold").glob("features_citations/t=*/*.parquet"))
    )
    sev = cit.drop_nulls("max_sev_last_std")
    assert sev.filter(pl.col("max_sev_last_std").is_between(7, 11)).height  # G-K: G+ by labels.yaml
    assert sev["gplus_last_std"].to_list() == (sev["max_sev_last_std"] >= 12).to_list()
    manifest = json.loads((toy.root / "gold" / "dataset_manifest.json").read_text())
    assert manifest["config_sha256"]["labels.yaml"] == _settings_sha(labels_cfg)
    assert manifest["config_sha256"]["labels.yaml"] != _settings_sha(load_labels_config())


def test_the_run_manifest_records_every_labels_split_windows(toy, monkeypatch):
    assert toy.gold(monkeypatch) == EXIT_OK
    w = toy.manifests()[0]["split_windows"]
    assert w["g6"] == {
        "last_labeled_t": "2024-01", "train": ["2023-01", "2023-01"], "calib": ["2023-02", "2023-04"],
        "purge": ["2023-05", "2023-10"], "test": ["2023-11", "2024-01"], "score": ["2024-02", None],
    }  # fmt: skip
    assert w["g12"]["last_labeled_t"] == "2023-07" and w["pen12"]["last_labeled_t"] == "2023-03"


def _shipped_log(toy) -> list[dict]:
    (m,) = toy.manifests()
    (key,) = toy.store.list(f"reports/gold/run={m['run_id']}/logs")
    lines = (toy.root / key).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def test_a_disabled_split_gate_logs_a_warning_and_is_recorded(toy, monkeypatch):
    toy.splits_file = str(no_gate_splits(toy.root))
    assert toy.gold(monkeypatch) == EXIT_OK
    # The run's own log (shipped to reports/gold/run=<id>/logs/) carries the warning.
    warnings = [r for r in _shipped_log(toy) if r.get("level") == "WARNING"]
    assert any("split gate disabled: no required labels" in r.get("msg", "") for r in warnings)
    assert toy.manifests()[0]["split_gate"] == {"required_labels": [], "enabled": False}


def test_an_enabled_split_gate_is_recorded_without_a_warning(toy, monkeypatch):
    assert toy.gold(monkeypatch) == EXIT_OK
    assert toy.manifests()[0]["split_gate"] == {"required_labels": ["g6"], "enabled": True}
    assert not any("split gate disabled" in r.get("msg", "") for r in _shipped_log(toy))


def _merged(toy):
    return pl.read_parquet(toy.root / "gold" / "facility_month_dataset.parquet")


def _merged_sha(toy):
    return hashlib.sha256(
        (toy.root / "gold" / "facility_month_dataset.parquet").read_bytes()
    ).hexdigest()


def test_the_merged_dataset_is_the_monthly_partitions_in_one_file(toy, monkeypatch):
    assert toy.gold(monkeypatch) == EXIT_OK
    months = sorted(set(toy.read("spine")["t"]))
    parts = pl.concat([toy.read(f"facility_month_dataset/t={t}") for t in months])
    merged = _merged(toy)
    assert merged.schema == parts.schema
    assert merged.equals(parts.sort("t", "ccn"))
    assert merged.select("t", "ccn").equals(merged.select("t", "ccn").sort("t", "ccn"))
    current = read_current_gold(toy.store)  # sha256-checks every listed file, the merged one too
    assert current["sha256"]["facility_month_dataset.parquet"] == _merged_sha(toy)
    manifest = json.loads((toy.root / "gold" / "dataset_manifest.json").read_text())
    assert manifest["merged_file"] == {
        "path": "gold/facility_month_dataset.parquet",
        "sha256": _merged_sha(toy),
    }
    assert "facility_month_dataset.parquet" not in manifest["partitions_sha256"]


def test_a_later_run_with_a_new_month_replaces_the_merged_dataset(toy, monkeypatch):
    toy.splits_file = str(no_gate_splits(toy.root))
    assert toy.gold(monkeypatch, "--v", "2024-06") == EXIT_OK
    assert _merged(toy)["t"].max() == "2024-06"
    assert toy.gold(monkeypatch) == EXIT_OK
    merged = _merged(toy)
    assert merged["t"].max() == "2025-03"
    assert merged.height == toy.read("spine").height  # rebuilt, not appended to
    assert not merged.select("ccn", "t").is_duplicated().any()


def test_a_failed_gate_leaves_the_published_merged_dataset_untouched(toy, monkeypatch, capsys):
    assert toy.gold(monkeypatch) == EXIT_OK
    before = _merged_sha(toy)
    real = build.build_labels

    def no_g6(*args, **kwargs):  # every g6 training row censored: the gate refuses to publish
        labels = real(*args, **kwargs)
        return labels.with_columns(
            pl.when(pl.col("t") == "2023-01").then(None).otherwise(pl.col("y_g6")).alias("y_g6")
        )

    monkeypatch.setattr(build, "build_labels", no_g6)
    assert toy.gold(monkeypatch) == EXIT_VALIDATION
    assert "required label g6" in capsys.readouterr().err
    assert _merged_sha(toy) == before
    assert read_current_gold(toy.store)["sha256"]["facility_month_dataset.parquet"] == before


def test_read_current_gold_refuses_a_merged_dataset_changed_after_promotion(toy, monkeypatch):
    assert toy.gold(monkeypatch) == EXIT_OK
    merged = toy.root / "gold" / "facility_month_dataset.parquet"
    _merged(toy).head(1).write_parquet(merged)
    with pytest.raises(GoldCheckError, match="facility_month_dataset.parquet does not match"):
        read_current_gold(toy.store)
