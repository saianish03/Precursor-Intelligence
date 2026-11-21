"""BM25 search over the Retriever's corpus (docs/AGENTS_PLAN.md G2), loaded once per instance.

The index covers each passage's tag, category and text. A tag written any common way in a query ("F689", "F-0689",
"f 689", "tag 689" is not one) becomes one token, ``tag_f0689``, which only that tag's passages hold, so asking for
a tag by number finds it first.

The search abstains (no hits, and the reason) rather than hand Agent 2 a passage that doesn't answer the question, so
Agent 2 can report ``abstained: true``. It abstains when either check fails:
- **coverage:** at least ``min_coverage`` of the query's words must occur somewhere in the corpus. A query that names
  a tag the corpus holds passes this check by itself.
- **score:** the best passage must score at least ``min_score``. A query that names a tag the corpus holds skips this
  check (in a small corpus a tag token's idf is modest), gets that tag's passages first, then every other passage that scores above zero.
Both checks are needed. BM25 scores are not normalized, and on the real CMS tag corpus (643 current wordings,
2026-10) a raw score alone can't separate the two kinds of question: the weakest on-topic query ("F689") scored 5.0
and an off-topic one ("python list comprehension", matching "list") scored 6.1. Coverage separates them (1.0 for the
on-topic probes, at most 0.5 for the off-topic ones). The defaults move into ``configs/agents.yaml`` with the
settings (A1).

A search is restricted to the wordings in force in one vintage (``as_of``), by default the current ones, so a quote
from an old citation uses the text the citation was written against. The F-tags retired in 2017 (``legacy``) are left
out unless ``include_legacy``.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Sequence
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from .corpus import Corpus, Passage

DEFAULT_MIN_SCORE = 3.0
DEFAULT_MIN_COVERAGE = 0.6
DEFAULT_K = 5
MAX_K = 20
MAX_QUERY_CHARS = 500

_TAG_IN_TEXT = re.compile(r"\b([efk])\s?-?\s?0*(\d{1,4})\b", re.IGNORECASE)
_WORD = re.compile(r"[a-z0-9]+")
_VINTAGE = re.compile(r"^20\d\d-(0[1-9]|1[0-2])$")

# Words that carry no meaning for matching a regulation. Kept short on purpose: BM25's idf already discounts the
# common words of this corpus ("resident", "provide"), and removing domain words here would hide them from queries.
_STOPWORDS_TEXT = """
a an and are as at be been but by can could did do does for from had has have how i if in into is
it its me my no not of on or our so such than that the their them then there these they this those
to was we were what when where which who whom why will with would you your
"""
STOPWORDS = frozenset(_STOPWORDS_TEXT.split())


def tag_token(prefix: str, number: str) -> str:
    return f"tag_{prefix.lower()}{int(number):04d}"


def _stem(word: str) -> str:
    """A light plural fold so "ulcers" matches "ulcer" and "falls" matches "fall"; no other suffixes."""
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def tokenize(text: str) -> list[str]:
    """Lower-case words without stopwords, plural-folded, plus one ``tag_x0000`` token per tag mentioned."""
    tags = [tag_token(m.group(1), m.group(2)) for m in _TAG_IN_TEXT.finditer(text)]
    words = [_stem(w) for w in _WORD.findall(text.lower()) if w not in STOPWORDS]
    return tags + words


def passage_tokens(p: Passage) -> list[str]:
    return [tag_token(p.tag[0], p.tag[1:])] + tokenize(f"{p.category} {p.text}")


@dataclass(frozen=True)
class Hit:
    passage: Passage
    score: float


@dataclass(frozen=True)
class SearchResult:
    query: str
    corpus_version: str
    as_of: str | None
    hits: tuple[Hit, ...]
    abstained: bool
    reason: str | None  # why it abstained; None when it returned hits


class Bm25Index:
    """A BM25 index over the passages in force in one vintage. Build it once; ``search`` is read-only."""

    def __init__(
        self, corpus: Corpus, as_of: str | None = None, include_legacy: bool = False
    ) -> None:
        if as_of is not None and not _VINTAGE.match(as_of):
            raise ValueError(f"as_of must be YYYY-MM, got {as_of!r}")
        self.corpus_version = corpus.version
        self.as_of = as_of
        self.include_legacy = include_legacy
        self.passages: tuple[Passage, ...] = tuple(
            p for p in corpus.passages if _in_force(p, as_of) and (include_legacy or not p.legacy)
        )
        if not self.passages:
            raise ValueError(f"no passages in force for as_of={as_of}")
        docs = [passage_tokens(p) for p in self.passages]
        self._vocab = frozenset(t for d in docs for t in d)
        self._tag_tokens = [d[0] for d in docs]
        self._bm25 = BM25Okapi(docs)

    def coverage(self, tokens: Sequence[str]) -> float:
        """The share of the query's word tokens that occur in the corpus; 1.0 when it names a tag the corpus holds."""
        if any(t.startswith("tag_") and t in self._vocab for t in tokens):
            return 1.0
        words = [t for t in tokens if not t.startswith("tag_")]
        if not words:
            return 0.0
        return sum(t in self._vocab for t in words) / len(words)

    def search(
        self,
        query: str,
        k: int = DEFAULT_K,
        min_score: float = DEFAULT_MIN_SCORE,
        min_coverage: float = DEFAULT_MIN_COVERAGE,
    ) -> SearchResult:
        if not 1 <= k <= MAX_K:
            raise ValueError(f"k must be between 1 and {MAX_K}")
        query = query.strip()
        if not query or len(query) > MAX_QUERY_CHARS:
            raise ValueError(f"the query must be 1-{MAX_QUERY_CHARS} characters")
        tokens = tokenize(query)
        if not tokens:
            return self._abstain(query, "the query has no searchable words")
        cov = self.coverage(tokens)
        if cov < min_coverage:
            return self._abstain(
                query,
                f"only {cov:.0%} of the query's words occur in the corpus (needs {min_coverage:.0%})",
            )
        scores = self._bm25.get_scores(tokens)
        named = {t for t in tokens if t.startswith("tag_")}
        names_a_tag = bool(named & self._vocab)
        # the passages of a tag the query names come first, then by score
        order = sorted(
            range(len(scores)),
            key=lambda i: (
                self._tag_tokens[i] not in named,
                -scores[i],
                self.passages[i].passage_id,
            ),
        )
        hits = tuple(
            Hit(self.passages[i], float(scores[i]))
            for i in order[:k]
            if (scores[i] > 0 if names_a_tag else scores[i] >= min_score)
        )
        if not hits:
            best = float(scores[order[0]]) if order else 0.0
            return self._abstain(
                query, f"no passage scores at least {min_score:g} (best {best:.2f})"
            )
        return SearchResult(query, self.corpus_version, self.as_of, hits, False, None)

    def _abstain(self, query: str, reason: str) -> SearchResult:
        return SearchResult(query, self.corpus_version, self.as_of, (), True, reason)


def _in_force(p: Passage, as_of: str | None) -> bool:
    if as_of is None:
        return p.is_current
    if p.valid_from_vintage > as_of:
        return False
    # a vintage after the corpus's newest one is served by the current wordings
    return as_of <= p.valid_to_vintage or p.is_current


_lock = threading.Lock()
_indexes: dict[tuple[str, str | None, bool], Bm25Index] = {}


def get_index(corpus: Corpus, as_of: str | None = None, include_legacy: bool = False) -> Bm25Index:
    """The index for this corpus version and vintage, built on first use and kept for the instance's life.

    A new corpus version builds a new index; old ones are dropped, since an instance serves one corpus at a time.
    """
    key = (corpus.version, as_of, include_legacy)
    with _lock:
        idx = _indexes.get(key)
        if idx is None:
            for old in [k for k in _indexes if k[0] != corpus.version]:
                del _indexes[old]
            idx = _indexes[key] = Bm25Index(corpus, as_of, include_legacy)
        return idx


def search(
    corpus: Corpus,
    query: str,
    k: int = DEFAULT_K,
    as_of: str | None = None,
    min_score: float = DEFAULT_MIN_SCORE,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    include_legacy: bool = False,
) -> SearchResult:
    return get_index(corpus, as_of, include_legacy).search(
        query, k=k, min_score=min_score, min_coverage=min_coverage
    )


def tags_in(text: str) -> Sequence[str]:
    """The tags a text mentions, in corpus spelling (``F0689``), first mention first."""
    seen: dict[str, None] = {}
    for m in _TAG_IN_TEXT.finditer(text):
        seen.setdefault(f"{m.group(1).upper()}{int(m.group(2)):04d}", None)
    return list(seen)
