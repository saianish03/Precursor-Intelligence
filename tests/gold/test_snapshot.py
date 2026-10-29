"""`snapshot-gold`: the published gold copied out for DVC (docs/runbooks/gold-snapshot.md).

The toy world publishes a real gold; its manifest is then edited to stand in for a verified gold (the §11 gate that
sets `verified` is not built, so every real manifest says false today).
"""

import json

import pytest
from test_gold_command import EXIT_OK, Toy, run

from precursorintelligence.common.config import ConfigError
from precursorintelligence.gold import GoldCheckError
from precursorintelligence.gold.dataset import MERGED
from precursorintelligence.gold.snapshot import FILES, snapshot_gold
from precursorintelligence.ingestion.pipeline import EXIT_CONFIG, EXIT_VALIDATION


@pytest.fixture
def toy(tmp_path, monkeypatch):
    t = Toy(tmp_path / "toy")
    assert t.gold(monkeypatch) == EXIT_OK
    return t


def set_manifest(toy, **fields):
    """Edit gold/dataset_manifest.json (it is not one of the files `_current.json` hashes)."""
    path = toy.root / "gold" / "dataset_manifest.json"
    manifest = json.loads(path.read_text())
    path.write_text(json.dumps(manifest | fields))


def published(toy, **fields):
    set_manifest(toy, **({"verified": True, "git_dirty": False} | fields))
    return toy


def test_the_three_published_files_are_copied_byte_identical(toy, tmp_path):
    published(toy)
    out = tmp_path / "gold_real"
    snap = snapshot_gold(toy.store, out)
    assert sorted(p.name for p in out.iterdir()) == sorted(FILES)
    for name in FILES:
        assert (out / name).read_bytes() == (toy.root / "gold" / name).read_bytes()
    assert snap.files[MERGED] == toy.sha(MERGED)
    assert snap.rows == 109 and snap.v == "2025-03" and snap.verified is True


def test_the_tag_names_the_month_and_the_content_hash(toy, tmp_path):
    published(toy)
    manifest = json.loads((toy.root / "gold" / "dataset_manifest.json").read_text())
    snap = snapshot_gold(toy.store, tmp_path / "out")
    assert snap.tag == f"gold-V2025-03-{manifest['dataset_version'][:8]}"
    assert snap.dataset_version == manifest["dataset_version"]


def test_an_unverified_gold_is_refused_unless_allowed_and_the_tag_says_so(toy, tmp_path):
    set_manifest(toy, verified=False, git_dirty=False)
    with pytest.raises(GoldCheckError, match="verified"):
        snapshot_gold(toy.store, tmp_path / "out")
    assert not (tmp_path / "out").exists()  # nothing is written for a refused snapshot
    snap = snapshot_gold(toy.store, tmp_path / "out", allow_unverified=True)
    assert snap.tag.endswith("-unverified") and snap.verified is False


def test_a_gold_built_from_a_dirty_tree_is_refused_unless_allowed(toy, tmp_path):
    published(toy, git_dirty=True)
    with pytest.raises(GoldCheckError, match="dirty"):
        snapshot_gold(toy.store, tmp_path / "out")
    assert snapshot_gold(toy.store, tmp_path / "out", allow_unverified=True).tag.endswith(
        "-unverified"
    )


def test_an_unknown_dirty_flag_is_reported_not_refused(toy, tmp_path):
    published(toy, git_dirty=None, git_commit="a" * 40)  # gold built in the image: no .git
    snap = snapshot_gold(toy.store, tmp_path / "out")
    assert snap.git_dirty is None and snap.git_commit == "a" * 40
    assert not snap.tag.endswith("-unverified")


def test_a_merged_file_changed_after_publishing_is_refused(toy, tmp_path):
    published(toy)
    merged = toy.root / "gold" / MERGED
    merged.write_bytes(merged.read_bytes()[:-8])
    with pytest.raises(GoldCheckError, match="does not match"):
        snapshot_gold(toy.store, tmp_path / "out")


def test_a_gold_that_never_finished_promoting_is_refused(toy, tmp_path):
    published(toy)
    (toy.root / "gold" / "_current.json").write_text(
        json.dumps({"state": "promoting", "run_id": "r"})
    )
    with pytest.raises(GoldCheckError, match="not current"):
        snapshot_gold(toy.store, tmp_path / "out")


def test_a_manifest_that_disagrees_about_the_merged_file_is_refused(toy, tmp_path):
    published(toy, merged_file={"path": f"gold/{MERGED}", "sha256": "0" * 64})
    with pytest.raises(GoldCheckError, match="merged"):
        snapshot_gold(toy.store, tmp_path / "out")


def test_a_second_snapshot_replaces_the_first_but_never_foreign_files(toy, tmp_path):
    published(toy)
    out = tmp_path / "gold_real"
    snapshot_gold(toy.store, out)
    (out / FILES[0]).write_bytes(b"stale")
    snapshot_gold(toy.store, out)
    assert (out / FILES[0]).read_bytes() == (toy.root / "gold" / FILES[0]).read_bytes()
    (out / "notes.txt").write_text("mine")
    with pytest.raises(ConfigError, match="notes.txt"):
        snapshot_gold(toy.store, out)
    assert (out / "notes.txt").read_text() == "mine"


def test_the_command_prints_the_tag_and_the_next_steps(toy, tmp_path, monkeypatch, capsys):
    published(toy)
    out = tmp_path / "gold_real"
    args = ["snapshot-gold", "--env-file", toy.env_file, "--out", str(out)]
    assert run(monkeypatch, *args) == EXIT_OK
    text = capsys.readouterr().out
    assert "gold-V2025-03-" in text and "dvc add" in text and "dvc push" in text
    assert (out / MERGED).is_file()


def test_the_command_exits_2_for_a_refused_gold_and_3_for_a_foreign_out_dir(
    toy, tmp_path, monkeypatch, capsys
):
    base = ["snapshot-gold", "--env-file", toy.env_file]
    assert (
        run(monkeypatch, *base, "--out", str(tmp_path / "a")) == EXIT_VALIDATION
    )  # verified=false
    assert "verified" in capsys.readouterr().err
    published(toy)
    foreign = tmp_path / "b"
    foreign.mkdir()
    (foreign / "notes.txt").write_text("mine")
    assert run(monkeypatch, *base, "--out", str(foreign)) == EXIT_CONFIG
