"""`snapshot-gold`: the published gold copied out for DVC (docs/runbooks/gold-snapshot.md).

The toy world publishes a real gold; its manifest is then edited to stand in for a verified gold (the §11 gate that
sets `verified` is not built, so every real manifest says false today).
"""

import io
import json

import pytest
from test_gold_command import EXIT_OK, Toy, run

from precursorintelligence.common.config import ConfigError
from precursorintelligence.gold import GoldCheckError, cli
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
    # --allow-unverified is needed for every snapshot until the §11 gate exists, so it must not also cover this
    with pytest.raises(GoldCheckError, match="dirty"):
        snapshot_gold(toy.store, tmp_path / "out", allow_unverified=True)
    assert snapshot_gold(toy.store, tmp_path / "out", allow_dirty=True).tag.endswith("-dirty")


def test_an_unverified_gold_from_a_dirty_tree_needs_both_flags_and_says_both(toy, tmp_path):
    set_manifest(toy, verified=False, git_dirty=True)
    snap = snapshot_gold(toy.store, tmp_path / "out", allow_unverified=True, allow_dirty=True)
    assert snap.tag.endswith("-unverified-dirty")


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


def test_a_gold_whose_current_json_lists_no_merged_file_is_refused(toy, tmp_path):
    published(toy)
    path = toy.root / "gold" / "_current.json"
    current = json.loads(path.read_text())
    current["sha256"].pop(MERGED)
    path.write_text(json.dumps(current))
    with pytest.raises(GoldCheckError, match="lists no"):
        snapshot_gold(toy.store, tmp_path / "out")


class ChangingStore:
    """The toy store, except that the second read of `key` returns `data`: gold changing mid-snapshot."""

    def __init__(self, store, key, data):
        self.store, self.key, self.data, self.reads = store, key, data, 0

    def __getattr__(self, name):
        return getattr(self.store, name)

    def open_read(self, key):
        if key == self.key:
            self.reads += 1
            if self.reads == 2:
                return io.BytesIO(self.data)
        return self.store.open_read(key)


def first_snapshot(toy, tmp_path):
    published(toy)
    out = tmp_path / "gold_real"
    snapshot_gold(toy.store, out)
    return out, {p.name: p.read_bytes() for p in out.iterdir()}


@pytest.mark.parametrize(
    "key, data",
    [
        (f"gold/{MERGED}", b"another run's merged file"),
        ("gold/_current.json", json.dumps({"state": "current", "run_id": "later"}).encode()),
    ],
    ids=["merged-file", "current-json"],
)
def test_gold_that_changes_while_snapshotting_is_refused_and_the_old_snapshot_kept(
    toy, tmp_path, key, data
):
    out, before = first_snapshot(toy, tmp_path)
    with pytest.raises(GoldCheckError, match="rerun"):
        snapshot_gold(ChangingStore(toy.store, key, data), out)
    assert {p.name: p.read_bytes() for p in out.iterdir()} == before


def drop_field(manifest, field):
    out = dict(manifest)
    if field == "rows.total":
        out["rows"] = {k: v for k, v in manifest["rows"].items() if k != "total"}
    else:
        out.pop(field)
    return json.dumps(out)


@pytest.mark.parametrize("damage", ["dataset_version", "v", "rows.total", "not json", "missing"])
def test_a_damaged_manifest_is_refused_before_the_old_snapshot_is_touched(toy, tmp_path, damage):
    out, before = first_snapshot(toy, tmp_path)
    path = toy.root / "gold" / "dataset_manifest.json"
    if damage == "missing":
        path.unlink()
    elif damage == "not json":
        path.write_text("{")
    else:
        path.write_text(drop_field(json.loads(path.read_text()), damage))
    with pytest.raises(GoldCheckError, match="dataset_manifest.json"):
        snapshot_gold(toy.store, out)
    assert {p.name: p.read_bytes() for p in out.iterdir()} == before


def test_a_macos_ds_store_in_the_out_folder_is_replaced_not_refused(toy, tmp_path):
    out, _ = first_snapshot(toy, tmp_path)
    (out / ".DS_Store").write_bytes(b"finder")
    snapshot_gold(toy.store, out)
    assert sorted(p.name for p in out.iterdir()) == sorted(FILES)


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


def test_the_default_out_is_under_the_repo_root_not_the_current_folder(
    toy, tmp_path, monkeypatch, capsys
):
    published(toy)
    repo, elsewhere = tmp_path / "repo", tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.setattr(cli, "project_root", lambda: repo)
    monkeypatch.chdir(elsewhere)
    assert run(monkeypatch, "snapshot-gold", "--env-file", toy.env_file) == EXIT_OK
    assert (repo / "snapshots" / "gold_real" / MERGED).is_file()
    assert not (elsewhere / "snapshots").exists()
    assert (
        "dvc add snapshots/gold_real" in capsys.readouterr().out
    )  # the next steps run from the repo root


def test_the_command_never_opens_the_metadata_store(toy, tmp_path, monkeypatch):
    published(toy)

    def no_metadata(*_args, **_kwargs):
        raise AssertionError("snapshot-gold must not open the metadata store")

    monkeypatch.setattr(cli, "build_context", no_metadata)
    args = ["snapshot-gold", "--env-file", toy.env_file, "--out", str(tmp_path / "out")]
    assert run(monkeypatch, *args) == EXIT_OK


def test_the_command_exits_2_for_a_damaged_manifest(toy, tmp_path, monkeypatch, capsys):
    published(toy)
    (toy.root / "gold" / "dataset_manifest.json").write_text("{")
    args = ["snapshot-gold", "--env-file", toy.env_file, "--out", str(tmp_path / "out")]
    assert run(monkeypatch, *args) == EXIT_VALIDATION
    assert "dataset_manifest.json" in capsys.readouterr().err
