"""The Regulatory Evidence Retriever's corpus (docs/AGENTS_PLAN.md G1): passages with stable IDs that Agent 2 quotes.

Source today: the CMS deficiency-tag descriptions (silver ``citation_lookup_versions``, one row per tag and wording).
Appendix PP regulation text can be added as a second source once the team decides (plan §6, decision 5); a source is
just a function that returns passages, so adding one changes no reader.

Passage IDs are ``passage:<source>:<tag>:<from>``, e.g. ``passage:cms_tag:F0689:2021-01``. The tag uses the same
spelling as the ``ftag`` of a citation (prefix plus four digits), so Agent 1 can join a citation to its text.
``<from>`` is the first vintage the wording was published in: CMS reworded a few tags (F0605 in 2025-04, K0311 in
2021-04), and a citation from before the change must be quoted with the wording it was written against. Two wordings
of one tag can't start in the same vintage (refused), so the ID is unique, and it never moves: a rebuild of silver
with older vintages added, or a wording dropping out, leaves every other wording's ID as it was (a position number
would shift). Each passage also carries the last vintage its wording was published in.

CMS's lookup still lists the F-tags retired by the F-tag renumbering of 2017-11-28 (F0150-F0522; the current tags
are F0540-F0949), with the same wording as their successors (F0323 and F0689). They are marked ``legacy``: in the
local silver, none of them is cited on a survey after 2017-11, so no citation in the last 36 months uses one, and the
search leaves them out unless asked.

Layout under ``prefix`` (default ``agents/corpus/``)::

    _current.json                       written last: {"schema_version", "version", "manifest_sha256"}
    <version>/corpus_manifest.json      version, sources (input key and sha256), passage count, files with sha256
    <version>/passages.jsonl            one passage per line, canonical JSON, sorted by passage ID

``<version>`` is ``c<schema>-<first 12 hex of the sha256 of passages.jsonl>``: the same input always gives the same
version and the same bytes, and the answer cache (D1) keys on it. Readers find the version only through
``_current.json`` and refuse a file whose sha256 does not match (``CorpusError``), as the evidence reader does.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime

import polars as pl
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..common.io.object_store import ObjectStore

SCHEMA_VERSION = 1
DEFAULT_PREFIX = "agents/corpus/"
CURRENT = "_current.json"
MANIFEST = "corpus_manifest.json"
PASSAGES = "passages.jsonl"

CMS_TAG_SOURCE = "cms_tag"
LOOKUP_VERSIONS_KEY = "silver/citation_lookup_versions/part-0.parquet"
LOOKUP_COLUMNS = (
    "deficiency_prefix",
    "deficiency_tag",
    "deficiency_description",
    "deficiency_category",
    "valid_from_vintage",
    "valid_to_vintage",
)

TAG_PATTERN = re.compile(r"^[A-Z]\d{4}$")
VINTAGE_PATTERN = re.compile(r"^20\d\d-(0[1-9]|1[0-2])$")
PASSAGE_ID_PATTERN = re.compile(r"^passage:[a-z_]+:[A-Z]\d{4}:20\d\d-(0[1-9]|1[0-2])$")
VERSION_PATTERN = re.compile(r"^c\d+-[0-9a-f]{12}$")
_SPACES = re.compile(r"\s+")
FIRST_CURRENT_F_TAG = 540  # F-tag renumbering of 2017-11-28: F0540-F0949 replaced F0150-F0522


class CorpusError(RuntimeError):
    """The corpus input or the published corpus is missing, inconsistent or does not match its manifest."""


class Passage(BaseModel):
    """One quotable passage. ``text`` is what a quote must be an exact substring of (validator B6)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    passage_id: str = Field(pattern=PASSAGE_ID_PATTERN.pattern)
    source: str
    tag: str = Field(pattern=TAG_PATTERN.pattern)
    category: str
    text: str = Field(min_length=1)
    valid_from_vintage: str = Field(pattern=VINTAGE_PATTERN.pattern)
    valid_to_vintage: str = Field(pattern=VINTAGE_PATTERN.pattern)
    is_current: bool  # the wording was in the newest vintage of citation_lookup
    legacy: bool  # a tag retired in 2017 (see the module docstring)


class CorpusPointer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int
    version: str = Field(pattern=VERSION_PATTERN.pattern)
    manifest_sha256: str


class FileEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    sha256: str
    rows: int


class SourceEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    input_key: str
    input_sha256: str
    passages: int


class CorpusManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int
    version: str
    built_at: str
    sources: list[SourceEntry]
    passages: int
    files: list[FileEntry]


@dataclass(frozen=True)
class Corpus:
    """A loaded corpus: its version (for the cache key) and its passages, sorted by ID."""

    version: str
    passages: tuple[Passage, ...]
    by_id: dict[str, Passage] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "by_id", {p.passage_id: p for p in self.passages})

    def get(self, passage_id: str) -> Passage | None:
        return self.by_id.get(passage_id)


@dataclass(frozen=True)
class BuildResult:
    version: str
    passages: int
    written: bool  # False when this exact version was already published; only the pointer was (re)written
    prefix: str


# ---------------------------------------------------------------- building


def normalize_text(text: str) -> str:
    """Collapse runs of whitespace and trim: CMS's text has double spaces ("catheter care,  and"). The stored passage
    is the normalized text, so quotes are checked against exactly what the model was shown."""
    return _SPACES.sub(" ", text).strip()


def is_legacy_tag(tag: str) -> bool:
    return tag[0] == "F" and int(tag[1:]) < FIRST_CURRENT_F_TAG


def passages_from_lookup(lookup: pl.DataFrame, source: str = CMS_TAG_SOURCE) -> list[Passage]:
    """One passage per tag wording, from silver ``citation_lookup_versions``.

    A wording's ID ends in its ``valid_from_vintage``. A wording is current when it was in the newest vintage of the
    table. Raises ``CorpusError`` on missing columns, an empty table, a malformed tag or vintage, or two wordings of
    one tag starting in the same vintage.
    """
    missing = [c for c in LOOKUP_COLUMNS if c not in lookup.columns]
    if missing:
        raise CorpusError(f"citation_lookup_versions lacks columns {missing}")
    df = (
        lookup.select(LOOKUP_COLUMNS)
        .with_columns(
            (
                pl.col("deficiency_prefix").str.strip_chars()
                + pl.col("deficiency_tag").str.strip_chars()
            ).alias("tag")
        )
        .filter(pl.col("deficiency_description").is_not_null())
    )
    if df.height == 0:
        raise CorpusError("citation_lookup_versions has no rows with a description")
    bad = (
        df.filter(~pl.col("tag").str.contains(TAG_PATTERN.pattern))["tag"].unique().sort().to_list()
    )
    if bad:
        raise CorpusError(f"malformed deficiency tags: {bad[:10]}")
    for col in ("valid_from_vintage", "valid_to_vintage"):
        bad = df.filter(~pl.col(col).str.contains(VINTAGE_PATTERN.pattern))[col].unique().to_list()
        if bad:
            raise CorpusError(f"malformed {col}: {sorted(map(str, bad))[:10]}")
    dup = df.group_by("tag", "valid_from_vintage").len().filter(pl.col("len") > 1)
    if dup.height:
        raise CorpusError(
            f"two wordings of one tag start in the same vintage: {dup['tag'].to_list()[:10]}"
        )
    newest = df["valid_to_vintage"].max()
    df = df.sort("tag", "valid_from_vintage")
    out = []
    for r in df.iter_rows(named=True):
        out.append(
            Passage(
                passage_id=f"passage:{source}:{r['tag']}:{r['valid_from_vintage']}",
                source=source,
                tag=r["tag"],
                category=normalize_text(r["deficiency_category"] or ""),
                text=normalize_text(r["deficiency_description"]),
                valid_from_vintage=r["valid_from_vintage"],
                valid_to_vintage=r["valid_to_vintage"],
                is_current=r["valid_to_vintage"] == newest,
                legacy=is_legacy_tag(r["tag"]),
            )
        )
    return sorted(out, key=lambda p: (p.source, p.tag, p.valid_from_vintage))


def serialize_passages(passages: Sequence[Passage]) -> bytes:
    """Canonical JSONL: sorted keys, no extra whitespace, UTF-8, one passage per line, sorted by ID."""
    ordered = sorted(passages, key=lambda p: (p.source, p.tag, p.valid_from_vintage))
    ids = [p.passage_id for p in ordered]
    if len(set(ids)) != len(ids):
        raise CorpusError("duplicate passage IDs")
    lines = [
        json.dumps(p.model_dump(), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        for p in ordered
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


def corpus_version(passages_bytes: bytes) -> str:
    return f"c{SCHEMA_VERSION}-{hashlib.sha256(passages_bytes).hexdigest()[:12]}"


def _read_parquet(store: ObjectStore, key: str) -> tuple[pl.DataFrame, str]:
    if not store.exists(key):
        raise CorpusError(f"corpus input {key} is missing; run silver first")
    with store.open_read(key) as f:
        data = f.read()
    return pl.read_parquet(io.BytesIO(data)), hashlib.sha256(data).hexdigest()


SourceFn = Callable[[ObjectStore], tuple[list[Passage], SourceEntry]]


def cms_tag_source(store: ObjectStore) -> tuple[list[Passage], SourceEntry]:
    lookup, sha = _read_parquet(store, LOOKUP_VERSIONS_KEY)
    passages = passages_from_lookup(lookup)
    return passages, SourceEntry(
        name=CMS_TAG_SOURCE, input_key=LOOKUP_VERSIONS_KEY, input_sha256=sha, passages=len(passages)
    )


DEFAULT_SOURCES: tuple[SourceFn, ...] = (cms_tag_source,)


def _norm_prefix(prefix: str) -> str:
    return prefix if prefix.endswith("/") else prefix + "/"


def build_corpus(
    store: ObjectStore,
    prefix: str = DEFAULT_PREFIX,
    sources: Sequence[SourceFn] = DEFAULT_SOURCES,
    now: datetime | None = None,
) -> BuildResult:
    """Build the corpus from ``sources`` and publish it under ``prefix``; ``_current.json`` is written last.

    Re-running on the same input publishes nothing new: the version folder already holds the same bytes, so only the
    pointer is rewritten (pointing at the same version).
    """
    prefix = _norm_prefix(prefix)
    passages: list[Passage] = []
    entries: list[SourceEntry] = []
    for fn in sources:
        ps, entry = fn(store)
        passages.extend(ps)
        entries.append(entry)
    body = serialize_passages(passages)
    version = corpus_version(body)
    folder = f"{prefix}{version}/"
    manifest_key = f"{folder}{MANIFEST}"
    written = False
    if store.exists(manifest_key):
        manifest_bytes = _read_bytes(store, manifest_key)
        _check_published(
            store, prefix, version, manifest_bytes
        )  # the folder must hold what its name says
    else:
        store.put_bytes(f"{folder}{PASSAGES}", body, overwrite=True)
        manifest = CorpusManifest(
            schema_version=SCHEMA_VERSION,
            version=version,
            built_at=(now or datetime.now(UTC)).isoformat(timespec="seconds"),
            sources=entries,
            passages=len(passages),
            files=[
                FileEntry(
                    key=f"{folder}{PASSAGES}",
                    sha256=hashlib.sha256(body).hexdigest(),
                    rows=len(passages),
                )
            ],
        )
        manifest_bytes = json.dumps(manifest.model_dump(), indent=2, sort_keys=True).encode("utf-8")
        store.put_bytes(manifest_key, manifest_bytes, overwrite=True)
        written = True
    pointer = CorpusPointer(
        schema_version=SCHEMA_VERSION,
        version=version,
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
    )
    store.put_bytes(
        f"{prefix}{CURRENT}",
        json.dumps(pointer.model_dump(), indent=2, sort_keys=True).encode("utf-8"),
        overwrite=True,
    )
    return BuildResult(version=version, passages=len(passages), written=written, prefix=prefix)


# ---------------------------------------------------------------- reading


def _read_bytes(store: ObjectStore, key: str) -> bytes:
    if not store.exists(key):
        raise CorpusError(f"{key} is missing")
    with store.open_read(key) as f:
        return f.read()


def _check_published(
    store: ObjectStore, prefix: str, version: str, manifest_bytes: bytes
) -> tuple[CorpusManifest, bytes]:
    try:
        manifest = CorpusManifest.model_validate_json(manifest_bytes)
    except ValidationError as exc:
        raise CorpusError(f"corpus manifest of {version} is malformed: {exc}") from None
    if manifest.version != version or manifest.schema_version != SCHEMA_VERSION:
        raise CorpusError(
            f"corpus manifest says version {manifest.version} schema {manifest.schema_version}, "
            f"expected {version} schema {SCHEMA_VERSION}"
        )
    entry = next((f for f in manifest.files if f.key == f"{prefix}{version}/{PASSAGES}"), None)
    if entry is None:
        raise CorpusError(f"corpus manifest of {version} does not list {PASSAGES}")
    body = _read_bytes(store, entry.key)
    if hashlib.sha256(body).hexdigest() != entry.sha256:
        raise CorpusError(f"{entry.key} does not match its manifest sha256")
    if corpus_version(body) != version:
        raise CorpusError(f"{entry.key} does not match its version {version}")
    return manifest, body


def read_current_version(store: ObjectStore, prefix: str = DEFAULT_PREFIX) -> CorpusPointer:
    """The pointer alone, for the cache key (D1), without reading the passages."""
    prefix = _norm_prefix(prefix)
    raw = _read_bytes(store, f"{prefix}{CURRENT}")
    try:
        pointer = CorpusPointer.model_validate_json(raw)
    except ValidationError as exc:
        raise CorpusError(f"{prefix}{CURRENT} is malformed: {exc}") from None
    if pointer.schema_version != SCHEMA_VERSION:
        raise CorpusError(f"corpus schema {pointer.schema_version} is not {SCHEMA_VERSION}")
    return pointer


def load_corpus(store: ObjectStore, prefix: str = DEFAULT_PREFIX) -> Corpus:
    """The corpus ``_current.json`` points at, every sha256 checked; reads objects by exact name, never lists."""
    prefix = _norm_prefix(prefix)
    pointer = read_current_version(store, prefix)
    manifest_bytes = _read_bytes(store, f"{prefix}{pointer.version}/{MANIFEST}")
    if hashlib.sha256(manifest_bytes).hexdigest() != pointer.manifest_sha256:
        raise CorpusError(f"the manifest of {pointer.version} does not match {CURRENT}")
    manifest, body = _check_published(store, prefix, pointer.version, manifest_bytes)
    try:
        passages = tuple(
            Passage.model_validate_json(line) for line in body.decode("utf-8").splitlines() if line
        )
    except ValidationError as exc:
        raise CorpusError(f"a passage of {pointer.version} is malformed: {exc}") from None
    if len(passages) != manifest.passages:
        raise CorpusError(
            f"{pointer.version} has {len(passages)} passages, its manifest says {manifest.passages}"
        )
    return Corpus(version=pointer.version, passages=passages)
