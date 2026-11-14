"""training/data.py: gold_config_sha, the gold identity ``gate`` writes and ``decide`` matches (§15.2)."""

import hashlib
import json

from precursorintelligence.training.data import gold_config_sha


def test_it_hashes_gold_manifests_config_and_commit(store, meta, prepare):
    manifest = prepare(store, meta).manifest
    with store.open_read("gold/dataset_manifest.json") as f:
        gold = json.load(f)
    canonical = json.dumps(
        {"config_sha256": gold["config_sha256"], "git_commit": gold.get("git_commit")},
        sort_keys=True,
        separators=(",", ":"),
    )
    assert gold_config_sha(manifest) == hashlib.sha256(canonical.encode()).hexdigest()


def test_other_code_or_configs_are_another_gold(store, meta, prepare):
    manifest = prepare(store, meta).manifest
    other_commit = manifest.model_copy(update={"gold_git_commit": "deadbeef"})
    other_config = manifest.model_copy(
        update={"gold_config_sha256": {**manifest.gold_config_sha256, "x.yaml": "0" * 64}}
    )
    shas = {gold_config_sha(m) for m in (manifest, other_commit, other_config)}
    assert len(shas) == 3
