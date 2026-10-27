"""Gold and validation through an fsspec object store (`memory://`, the same adapter as gcsfs): every key a node
writes lies in that node's own folders, so on GCS each layer stays in its own prefix of the bucket."""

import uuid

import yaml
from make_toy_silver import TOY_SPLITS, make_toy
from test_gold_command import run

from precursorintelligence.common.io.object_store import (
    FsspecObjectStore,
    LocalObjectStore,
    copy_object,
)
from precursorintelligence.ingestion.pipeline import EXIT_OK, EXIT_VALIDATION


def test_gold_and_validation_write_only_their_own_folders(tmp_path, monkeypatch):
    toy = tmp_path / "toy"
    local_env = make_toy(toy)
    local = LocalObjectStore(toy, immutable=None)
    store = FsspecObjectStore(f"memory://layers-{uuid.uuid4().hex}")
    for key in local.list("silver") + local.list(
        "gold"
    ):  # the toy silver world, now in the "bucket"
        copy_object(local, key, store, key)
    env = yaml.safe_load(local_env.read_text())
    env["object_store"] = {"kind": "memory", "root": store.root_uri}
    env_file = tmp_path / "memory_env.yaml"
    env_file.write_text(yaml.safe_dump(env))
    before = set(store.list(""))

    splits = str(toy / TOY_SPLITS)
    assert (
        run(monkeypatch, "gold", "--env-file", str(env_file), "--splits-config", splits) == EXIT_OK
    )
    gold = set(store.list("")) - before
    assert {"gold/_current.json", "gold/spine/part-0.parquet"} <= gold
    assert [k for k in gold if not k.startswith(("gold/", "reports/gold/", "metadata/gold/"))] == []

    code = run(monkeypatch, "validate", "--layer", "gold", "--env-file", str(env_file))
    assert code in (
        EXIT_OK,
        EXIT_VALIDATION,
    )  # the finding doesn't matter here, only where it is written
    validation = set(store.list("")) - before - gold
    assert validation
    assert [
        k for k in validation if not k.startswith(("reports/validation/", "metadata/validation/"))
    ] == []
