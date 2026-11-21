"""G2: BM25 search over the corpus: the right tag for known queries, abstention, vintages, legacy tags."""

from __future__ import annotations

import pytest
from corpus_fakes import lookup_frame

from precursorintelligence.agents import retrieval
from precursorintelligence.agents.corpus import (
    Corpus,
    corpus_version,
    passages_from_lookup,
    serialize_passages,
)
from precursorintelligence.agents.retrieval import (
    MAX_K,
    Bm25Index,
    get_index,
    search,
    tags_in,
    tokenize,
)


def make_corpus(frame=None) -> Corpus:
    ps = passages_from_lookup(lookup_frame() if frame is None else frame)
    return Corpus(version=corpus_version(serialize_passages(ps)), passages=tuple(ps))


@pytest.fixture(scope="module")
def corpus() -> Corpus:
    return make_corpus()


KNOWN = [
    ("falls and accident supervision", "F0689"),
    ("infection control program", "F0880"),
    ("pressure ulcer prevention", "F0686"),
    ("not enough nursing staff", "F0725"),
    ("failure to report abuse to authorities", "F0609"),
    ("psychotropic medication gradual dose reduction", "F0758"),
    ("significant medication errors", "F0760"),
    ("drugs stored in locked compartments", "F0761"),
    ("automatic sprinkler systems", "K0353"),
    ("simulated fire drills", "K0712"),
    ("generator power source", "K0918"),
    ("catheter care urinary tract infection", "F0690"),
    ("dignity and self-determination", "F0550"),
    ("dementia training for nurse aides", "F0947"),
    ("pain management", "F0697"),
    ("quality assessment and assurance group", "F0867"),
    ("serve food in accordance with professional standards", "F0812"),
]


@pytest.mark.parametrize(("query", "tag"), KNOWN)
def test_known_queries_find_their_tag_first(corpus, query, tag):
    res = search(corpus, query)
    assert not res.abstained
    assert res.hits[0].passage.tag == tag


@pytest.mark.parametrize(
    "query", ["F689", "F-0689", "f 689", "what does F0689 require?", "tag F689"]
)
def test_a_tag_by_number(corpus, query):
    res = search(corpus, query)
    assert not res.abstained
    assert res.hits[0].passage.tag == "F0689"


@pytest.mark.parametrize(
    "query",
    [
        "what is the stock price of Apple",
        "weather in Boston tomorrow",
        "recipe for chocolate cake",
        "tell me a joke",
        "python list comprehension",
        "F9999",
    ],
)
def test_off_topic_queries_abstain_with_a_reason(corpus, query):
    res = search(corpus, query)
    assert res.abstained and res.hits == () and res.reason


def test_a_low_score_abstains(corpus):
    res = search(corpus, "infection control program", min_score=1e6)
    assert res.abstained and "scores at least" in res.reason


def test_stopwords_only_abstains(corpus):
    res = search(corpus, "what is the")
    assert res.abstained and "no searchable words" in res.reason


@pytest.mark.parametrize("query", ["", "   ", "x" * 501])
def test_bad_query_is_a_usage_error(corpus, query):
    with pytest.raises(ValueError):
        search(corpus, query)


@pytest.mark.parametrize("k", [0, MAX_K + 1])
def test_bad_k_is_a_usage_error(corpus, k):
    with pytest.raises(ValueError):
        search(corpus, "abuse", k=k)


def test_k_bounds_the_hits_and_hits_are_ranked(corpus):
    res = search(corpus, "abuse neglect resident", k=2)
    assert len(res.hits) <= 2
    scores = [h.score for h in res.hits]
    assert scores == sorted(scores, reverse=True)


def test_hit_text_is_the_stored_passage_text(corpus):
    """Validator B6 checks quotes against the stored text, so the hit must carry exactly that text."""
    for h in search(corpus, "abuse neglect resident", k=5).hits:
        assert corpus.get(h.passage.passage_id).text == h.passage.text


def test_result_carries_the_corpus_version(corpus):
    assert search(corpus, "abuse").corpus_version == corpus.version


def test_as_of_serves_the_wording_in_force(corpus):
    old = search(corpus, "medications that restrain", as_of="2024-06")
    new = search(corpus, "medications that restrain")
    assert old.hits[0].passage.passage_id == "passage:cms_tag:F0605:2021-01"
    assert new.hits[0].passage.passage_id == "passage:cms_tag:F0605:2025-04"
    assert all(h.passage.passage_id != "passage:cms_tag:F0605:2021-01" for h in new.hits)


def test_as_of_after_the_newest_vintage_uses_the_current_wordings(corpus):
    ids = {p.passage_id for p in Bm25Index(corpus, as_of="2027-03").passages}
    assert "passage:cms_tag:F0605:2025-04" in ids and "passage:cms_tag:F0605:2021-01" not in ids


def test_bad_as_of_is_a_usage_error(corpus):
    with pytest.raises(ValueError):
        Bm25Index(corpus, as_of="2024-6")


def test_legacy_tags_are_left_out_unless_asked(corpus):
    tags = {h.passage.tag for h in search(corpus, "accident hazards supervision", k=5).hits}
    assert "F0323" not in tags and "F0689" in tags
    tags = {
        h.passage.tag
        for h in search(corpus, "accident hazards supervision", k=5, include_legacy=True).hits
    }
    assert {"F0323", "F0689"} <= tags


def test_the_index_is_built_once_per_version(corpus):
    assert get_index(corpus) is get_index(corpus)
    other = make_corpus(lookup_frame().head(10))
    get_index(other)
    assert all(k[0] == other.version for k in retrieval._indexes)
    assert all(k[0] == other.version for k in retrieval._by_passages)


def test_tokenize_folds_plurals_drops_stopwords_and_reads_tags():
    assert tokenize("The ulcers of F-689") == ["tag_f0689", "ulcer", "f", "689"]
    assert "the" not in tokenize("the abuse")
    assert tokenize("access")[0] == "access"  # a double s is not a plural


def test_tags_in():
    assert tags_in("F689 and K353, then F0689 again, and F 880") == ["F0689", "K0353", "F0880"]


@pytest.mark.parametrize("text", ["vitamin E 400", "part K 12", "F 12 residents", "k 353"])
def test_ordinary_text_is_not_a_tag(text):
    assert tags_in(text) == []


def test_a_named_tag_plus_off_topic_words_returns_that_tag_alone(corpus):
    res = search(corpus, "F689 python list comprehension")
    assert not res.abstained
    assert {h.passage.tag for h in res.hits} == {"F0689"}


def test_a_named_tag_plus_on_topic_words_returns_others_too(corpus):
    res = search(corpus, "F689 and abuse neglect")
    tags = [h.passage.tag for h in res.hits]
    assert tags[0] == "F0689" and "F0600" in tags


def test_months_with_the_same_wordings_share_one_index(corpus):
    assert get_index(corpus, as_of="2025-06") is get_index(corpus, as_of="2026-02")
    assert get_index(corpus, as_of="2024-06") is not get_index(corpus, as_of="2025-06")
