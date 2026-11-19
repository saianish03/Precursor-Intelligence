"""evaluation/packages.py: write-once copies of registered versions' packages, outside the run folders
(docs/ML_PIPELINE_SPEC.md §9.4; #75 B17)."""

import shutil

import pytest

from precursorintelligence.evaluation.packages import (
    backfill_packages,
    copy_packages,
    load_verified_package,
    stable_prefix,
)
from precursorintelligence.training import MlInputError
from precursorintelligence.training.contracts import RunPaths, TrainResult
from precursorintelligence.training.ensemble import PACKAGE_META, PrecursorModel


def copy(run, ml_run_id=None):
    return copy_packages(run.store(), run.packages(), run.cfg, ml_run_id or run.run_id)


def register(run, *versions, ml_run_id=None):
    """Gate rows of ``ml_run_id`` (default: the run) that registered ``versions``."""
    run.meta().append(
        "ml_runs",
        [
            {"ml_run_id": ml_run_id or run.run_id, "action": "hold", "candidate_version": v}
            for v in versions
        ],
    )


def load(run, version=1, fit="refit"):
    return load_verified_package(run.packages(), run.meta(), run.cfg, version, fit)


def train_json(run):
    p = RunPaths.of(run.cfg, run.run_id)
    with run.store().open_read(p.train_result) as f:
        return p.train_result, TrainResult.model_validate_json(f.read())


def read(store, key):
    with store.open_read(key) as f:
        return f.read()


def test_the_copy_holds_train_json_and_the_selected_candidates_packages(trained):
    train = trained.train.result
    prefix = stable_prefix(trained.cfg, trained.run_id)
    assert prefix == f"{trained.cfg.registry.model_name}/run={trained.run_id}/"
    written = copy(trained)
    expected = {
        f"{prefix}train.json",
        *(f"{prefix}refit/{train.selected}/{n}" for n in train.refit_package_sha256),
        *(f"{prefix}eval/{train.selected}/{n}" for n in train.eval_package_sha256[train.selected]),
    }
    assert set(written) == expected == set(trained.packages().list(prefix))
    # last, so a copy cut short has no train.json and nothing loads from it
    assert written[-1] == f"{prefix}train.json"


def test_a_version_loads_after_its_run_folder_packages_are_deleted(trained):
    copy(trained)
    register(trained, 1)
    p = RunPaths.of(trained.cfg, trained.run_id)
    shutil.rmtree(trained.gold.root / f"{p.run_dir}models")
    (trained.gold.root / p.train_result).unlink()
    for fit in ("refit", "eval"):
        assert set(load(trained, fit=fit).meta.training_months) == {fit}


def test_a_retried_copy_writes_nothing_and_never_overwrites(trained):
    first = copy(trained)
    assert copy(trained) == []  # a retried gate finds its copy
    key = first[0]
    trained.packages().put_bytes(key, read(trained.packages(), key) + b" ", overwrite=True)
    with pytest.raises(MlInputError, match="already holds a different file"):
        copy(trained)


@pytest.mark.parametrize("how", ["altered", "deleted", "no hashes", "outside its run folder"])
def test_a_run_folder_that_fails_verification_is_not_copied(trained, how):
    key, train = train_json(trained)
    store = trained.store()
    package = f"{train.refit_package}{PACKAGE_META}"
    if how == "altered":
        store.put_bytes(package, read(store, package) + b" ", overwrite=True)
        match = "does not match the sha256"
    elif how == "deleted":
        (trained.gold.root / package).unlink()
        match = "is missing"
    else:
        update = (
            {"refit_package_sha256": {}}
            if how == "no hashes"
            else {
                "refit_package": RunPaths.of(trained.cfg, "other").package("refit", train.selected)
            }
        )
        store.put_bytes(
            key, train.model_copy(update=update).model_dump_json().encode(), overwrite=True
        )
        match = (
            "records no refit package hashes" if how == "no hashes" else "outside its run folder"
        )
    with pytest.raises(MlInputError, match=match):
        copy(trained)
    assert trained.packages().list("") == []


def test_a_tampered_copy_is_refused_before_loading(trained, monkeypatch):
    copy(trained)
    register(trained, 1)
    selected = trained.train.result.selected
    key = f"{stable_prefix(trained.cfg, trained.run_id)}refit/{selected}/{PACKAGE_META}"
    trained.packages().put_bytes(key, read(trained.packages(), key) + b" ", overwrite=True)
    real = PrecursorModel.from_files

    def guarded(files):
        if files.get(PACKAGE_META, b"").endswith(b" "):
            raise AssertionError("the tampered package was unpickled before its sha256 was checked")
        return real(files)

    monkeypatch.setattr(PrecursorModel, "from_files", staticmethod(guarded))
    with pytest.raises(MlInputError, match="does not match the sha256"):
        load(trained)


def test_a_version_without_a_copy_is_refused_and_names_the_backfill(trained):
    # no fallback to the run folder, although its packages are all there
    register(trained, 1)
    with pytest.raises(MlInputError, match="no package copy.*ml_backfill_packages"):
        load(trained)


def test_a_copy_without_hashes_is_refused(trained):
    copy(trained)
    register(trained, 1)
    key = f"{stable_prefix(trained.cfg, trained.run_id)}train.json"
    train = TrainResult.model_validate_json(read(trained.packages(), key))
    trained.packages().put_bytes(
        key,
        train.model_copy(update={"eval_package_sha256": {}}).model_dump_json().encode(),
        overwrite=True,
    )
    with pytest.raises(MlInputError, match="records no eval package hashes"):
        load(trained, fit="eval")


def test_the_backfill_copies_every_registering_run_once(trained):
    register(trained, 1, 2)  # e.g. a gate retried after its version moved: one run, two rows
    trained.meta().append("ml_runs", [{"ml_run_id": "scored", "action": "score"}])
    out = backfill_packages(trained.store(), trained.packages(), trained.meta(), trained.cfg)
    assert list(out) == [trained.run_id] and out[trained.run_id]
    assert load(trained, 2).meta is not None
    again = backfill_packages(trained.store(), trained.packages(), trained.meta(), trained.cfg)
    assert again == {trained.run_id: []}
