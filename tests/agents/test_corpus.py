"""G1: the Retriever's corpus: stable passage IDs, determinism, publishing behind _current.json, refusals."""

from __future__ import annotations

import json

import polars as pl
import pytest
from corpus_fakes import NEWEST, ROWS, lookup_frame, write_lookup

from precursorintelligence.agents.corpus import (
    CURRENT,
    MANIFEST,
    PASSAGES,
    CorpusError,
    build_corpus,
    corpus_version,
    is_legacy_tag,
    load_corpus,
    passages_from_lookup,
    read_current_version,
    serialize_passages,
)
from precursorintelligence.common.io.object_store import LocalObjectStore

PREFIX = "agents/corpus/"


@pytest.fixture
def store(tmp_path) -> LocalObjectStore:
    s = LocalObjectStore(tmp_path, immutable=None)
    write_lookup(s)
    return s


def by_id(passages):
    return {p.passage_id: p for p in passages}


def test_one_passage_per_tag_wording_with_stable_ids():
    ps = passages_from_lookup(lookup_frame())
    assert len(ps) == len(ROWS)
    ids = by_id(ps)
    p = ids["passage:cms_tag:F0689:0"]
    assert (p.tag, p.n, p.source) == ("F0689", 0, "cms_tag")
    assert p.text.startswith("Ensure that a nursing home area is free from accident hazards")
    assert p.category == "Quality of Life and Care Deficiencies"


def test_reworded_tag_numbers_its_wordings_oldest_first():
    ids = by_id(passages_from_lookup(lookup_frame()))
    old, new = ids["passage:cms_tag:F0605:0"], ids["passage:cms_tag:F0605:1"]
    assert (old.valid_from_vintage, old.valid_to_vintage, old.is_current) == (
        "2021-01",
        "2025-03",
        False,
    )
    assert (new.valid_from_vintage, new.valid_to_vintage, new.is_current) == (
        "2025-04",
        NEWEST,
        True,
    )
    assert "psychotropic" in new.text and "psychotropic" not in old.text


def test_a_new_wording_keeps_the_old_ids():
    """CMS rewords F0880 in 2026-11: every other wording stays current, F0880 gets n=1, and no old ID changes."""
    before = by_id(passages_from_lookup(lookup_frame()))
    nov = "2026-11"
    rows = [(*r[:5], nov) if r[5] == NEWEST and r[:2] != ("F", "0880") else r for r in ROWS]
    rows.append(("F", "0880", "Infection Control Deficiencies", "A new wording.", nov, nov))
    after = by_id(passages_from_lookup(lookup_frame(rows)))
    for pid, p in before.items():
        assert after[pid].text == p.text
    assert after["passage:cms_tag:F0880:1"].text == "A new wording."
    assert not after["passage:cms_tag:F0880:0"].is_current


def test_text_is_whitespace_normalized():
    p = by_id(passages_from_lookup(lookup_frame()))["passage:cms_tag:F0690:0"]
    assert "  " not in p.text
    assert "appropriate catheter care, and appropriate care to prevent" in p.text


def test_legacy_f_tags():
    assert is_legacy_tag("F0323") and is_legacy_tag("F0150")
    assert not is_legacy_tag("F0540") and not is_legacy_tag("F0689") and not is_legacy_tag("K0311")
    ids = by_id(passages_from_lookup(lookup_frame()))
    assert ids["passage:cms_tag:F0323:0"].legacy
    assert not ids["passage:cms_tag:F0689:0"].legacy


def test_same_input_in_any_order_gives_the_same_bytes_and_version():
    a = serialize_passages(passages_from_lookup(lookup_frame()))
    shuffled = lookup_frame().sample(fraction=1.0, shuffle=True, seed=7)
    b = serialize_passages(passages_from_lookup(shuffled))
    assert a == b
    assert corpus_version(a) == corpus_version(b)
    assert corpus_version(a).startswith("c1-") and len(corpus_version(a)) == 15


def test_a_changed_wording_changes_the_version():
    a = serialize_passages(passages_from_lookup(lookup_frame()))
    rows = [(*r[:3], r[3] + " Edited.", *r[4:]) if r[:2] == ("F", "0880") else r for r in ROWS]
    b = serialize_passages(passages_from_lookup(lookup_frame(rows)))
    assert corpus_version(a) != corpus_version(b)


@pytest.mark.parametrize(
    ("frame", "match"),
    [
        (lookup_frame().drop("deficiency_category"), "lacks columns"),
        (lookup_frame().head(0), "no rows"),
        (lookup_frame().with_columns(pl.lit("68").alias("deficiency_tag")), "malformed"),
        (
            lookup_frame(ROWS + [("F", "0880", "x", "Another wording.", "2021-01", NEWEST)]),
            "same vintage",
        ),
    ],
)
def test_bad_input_is_refused(frame, match):
    with pytest.raises(CorpusError, match=match):
        passages_from_lookup(frame)


def test_build_publishes_and_load_reads_it_back(store):
    res = build_corpus(store)
    assert res.written and res.passages == len(ROWS)
    corpus = load_corpus(store)
    assert corpus.version == res.version == read_current_version(store).version
    assert corpus.get("passage:cms_tag:F0689:0").tag == "F0689"
    assert corpus.get("passage:cms_tag:F9999:0") is None
    manifest = json.loads(store.open_read(f"{PREFIX}{res.version}/{MANIFEST}").read())
    assert manifest["sources"][0]["name"] == "cms_tag"
    assert manifest["sources"][0]["input_key"] == "silver/citation_lookup_versions/part-0.parquet"
    assert manifest["passages"] == len(ROWS)


def test_rerun_writes_only_the_pointer(store):
    first = build_corpus(store)
    second = build_corpus(store)
    assert second.version == first.version and not second.written


def test_pointer_is_written_last(store):
    order: list[str] = []
    put = store.put_bytes

    def recording(key, data, *, overwrite=False):
        order.append(key)
        return put(key, data, overwrite=overwrite)

    store.put_bytes = recording  # type: ignore[method-assign]
    build_corpus(store)
    assert order[-1] == f"{PREFIX}{CURRENT}"
    assert {k.rsplit("/", 1)[1] for k in order[:-1]} == {PASSAGES, MANIFEST}


def test_missing_input_is_refused(tmp_path):
    with pytest.raises(CorpusError, match="run silver first"):
        build_corpus(LocalObjectStore(tmp_path, immutable=None))


def test_missing_pointer_is_refused(store):
    with pytest.raises(CorpusError, match="missing"):
        load_corpus(store)


def test_a_changed_passages_file_is_refused(store):
    res = build_corpus(store)
    key = f"{PREFIX}{res.version}/{PASSAGES}"
    body = store.open_read(key).read().replace(b"accident hazards", b"accident risks")
    store.put_bytes(key, body, overwrite=True)
    with pytest.raises(CorpusError, match="sha256"):
        load_corpus(store)
    with pytest.raises(CorpusError, match="sha256"):
        build_corpus(store)  # a rebuild refuses a folder that doesn't hold what its name says


def test_a_changed_manifest_is_refused(store):
    res = build_corpus(store)
    key = f"{PREFIX}{res.version}/{MANIFEST}"
    store.put_bytes(key, store.open_read(key).read() + b"\n", overwrite=True)
    with pytest.raises(CorpusError, match="does not match"):
        load_corpus(store)


def test_a_custom_prefix(store):
    res = build_corpus(store, prefix="ml-dev/agents/corpus")
    assert res.prefix == "ml-dev/agents/corpus/"
    assert load_corpus(store, prefix="ml-dev/agents/corpus").version == res.version
