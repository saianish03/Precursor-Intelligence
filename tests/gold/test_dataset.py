"""Assembly (spec §10): the join keeps exactly the spine's rows; the manifest is deterministic."""

from datetime import UTC, datetime

import polars as pl
import pytest
from silver_fakes import VINTAGES, V

from precursorintelligence.common.config import load_labels_config, load_splits_config
from precursorintelligence.common.io.metadata_store import SqlAlchemyMetadataStore
from precursorintelligence.gold import GoldCheckError
from precursorintelligence.gold.build import build_spine
from precursorintelligence.gold.dataset import assemble, by_t, dataset_manifest
from precursorintelligence.gold.features import FAMILIES, build_features
from precursorintelligence.gold.labels import build_labels


@pytest.fixture(scope="module")
def parts(silver):
    spine = build_spine(
        silver["facility_universe"], published_vintages=VINTAGES, spine_start="2023-01", v=V
    )
    labels = build_labels(
        spine,
        ledger_citations=silver["ledger_citations"],
        ledger_penalties=silver["ledger_penalties"],
        facility_presence=silver["facility_presence"],
        cfg=load_labels_config(),
        v=V,
    )
    features = {f: build_features(f, silver, spine) for f in FAMILIES}
    return spine, features, labels


def test_assembly_joins_in_the_spec_order(parts):
    spine, features, labels = parts
    ds = assemble(spine, features, labels)
    assert ds.select("ccn", "t").equals(spine.select("ccn", "t"))
    order = ["survey_timing", "citations", "enforcement", "facility_staffing", "context"]
    expected = list(spine.columns)
    for f in order:
        expected += features[f].columns[2:]
    expected += labels.columns[2:]
    assert ds.columns == expected


def test_assembly_refuses_a_family_that_adds_rows(parts):
    spine, features, labels = parts
    bad = features | {
        "enforcement": pl.concat([features["enforcement"], features["enforcement"][:1]])
    }
    with pytest.raises(GoldCheckError, match="features_enforcement"):
        assemble(spine, bad, labels)


def test_assembly_refuses_a_column_two_tables_both_carry(parts):
    spine, features, labels = parts
    bad = features | {"context": features["context"].with_columns(n_fines_24m=pl.lit(0))}
    with pytest.raises(GoldCheckError, match="repeats columns"):
        assemble(spine, bad, labels)


def test_by_t_splits_into_month_partitions(parts):
    spine, _, _ = parts
    months = by_t(spine)
    assert list(months) == sorted(set(spine["t"]))
    assert sum(p.height for p in months.values()) == spine.height


def test_the_manifest_is_the_same_for_the_same_inputs_but_the_timestamp(parts, tmp_path):
    spine, features, labels = parts
    meta = SqlAlchemyMetadataStore(f"sqlite:///{(tmp_path / 'm.db').as_posix()}")
    meta.ensure_schema()
    kwargs = dict(
        meta=meta, v=V, v_labels=V, partition_sha256={"facility_month_dataset/t=2025-03": "ab"},
        families=list(features), labels=labels, labels_cfg=load_labels_config(),
        splits_cfg=load_splits_config(), verified=False,
    )  # fmt: skip
    a = dataset_manifest(**kwargs, built_at=datetime(2026, 1, 1, tzinfo=UTC))
    b = dataset_manifest(**kwargs, built_at=datetime(2026, 2, 1, tzinfo=UTC))
    assert {k: v for k, v in a.items() if k != "built_at"} == {
        k: v for k, v in b.items() if k != "built_at"
    }
    assert a["rows_per_label_status"]["g12"]["positive"] == labels["y_g12"].sum()
    assert a["feature_families_pending"] == ["quality"]
    assert a["verified"] is False
    assert set(a) >= {"git_commit", "git_dirty"}


def _repo(tmp_path):
    import subprocess

    def git(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")
    (tmp_path / ".gitignore").write_text("data/\n")
    (tmp_path / "code.py").write_text("x = 1\n")
    git("add", ".")
    git("commit", "-q", "-m", "init")
    return tmp_path


def test_git_state_marks_a_dirty_working_tree(tmp_path, monkeypatch):
    from precursorintelligence.gold.dataset import GIT_COMMIT_ENV, git_state

    monkeypatch.delenv(GIT_COMMIT_ENV, raising=False)
    repo = _repo(tmp_path)
    commit, dirty = git_state(repo)
    assert commit and len(commit) == 40 and dirty is False
    (repo / "data").mkdir()
    (repo / "data" / "big.parquet").write_text("ignored")  # git-ignored files are not a change
    assert git_state(repo) == (commit, False)
    (repo / "code.py").write_text("x = 2\n")  # a tracked change
    assert git_state(repo) == (commit, True)
    (repo / "code.py").write_text("x = 1\n")
    (repo / "new.py").write_text("y = 1\n")  # new code git does not ignore
    assert git_state(repo) == (commit, True)


def test_git_state_without_a_checkout(tmp_path, monkeypatch):
    from precursorintelligence.gold.dataset import GIT_COMMIT_ENV, git_state

    monkeypatch.delenv(GIT_COMMIT_ENV, raising=False)
    assert git_state(tmp_path) == (None, None)
    monkeypatch.setenv(GIT_COMMIT_ENV, "abc123")  # images: the commit is known, the tree is not
    assert git_state(tmp_path) == ("abc123", None)
