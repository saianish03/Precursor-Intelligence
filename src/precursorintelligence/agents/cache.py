"""The answer cache (docs/AGENTS_PLAN.md D1): the key, the interface and an in-memory version.

The GCS version (D1) implements the same ``AnswerCache`` interface. Only answers that passed validation are stored;
fallback answers never are.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class CacheKey:
    """Everything an answer depends on, by name. A change to any part misses the cache.

    ``input`` is the agent's validated input as JSON (so it is already normalized: a CCN upper-cased, a
    ``question_type`` from its fixed list). ``corpus_version`` is set for the agents that read the corpus (Agent 2,
    and Agent 1 because it calls Agent 2), else None.
    """

    agent: str
    prompt_version: int
    model: str
    question_type: str | None
    input: Mapping[str, Any]
    evidence_manifest_sha: str
    corpus_version: str | None

    def parts(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "prompt_version": self.prompt_version,
            "model": self.model,
            "question_type": self.question_type,
            "input": dict(self.input),
            "evidence_manifest_sha": self.evidence_manifest_sha,
            "corpus_version": self.corpus_version,
        }

    def digest(self) -> str:
        """sha256 of the parts as canonical JSON: the object name in the cache bucket."""
        canonical = json.dumps(self.parts(), sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class AnswerCache(Protocol):
    async def get(self, key: str) -> dict[str, Any] | None:
        """The stored answer (its JSON form) under ``key``, or None."""
        ...

    async def put(self, key: str, answer: dict[str, Any]) -> None:
        """Store a validated answer under ``key``."""
        ...


class InMemoryAnswerCache:
    """``AnswerCache`` in a dict, for tests and local runs. Not shared across instances."""

    def __init__(self) -> None:
        self._entries: dict[str, str] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> dict[str, Any] | None:
        async with self._lock:
            raw = self._entries.get(key)
        # a fresh copy each time, so a caller can't change the stored entry
        return None if raw is None else json.loads(raw)

    async def put(self, key: str, answer: dict[str, Any]) -> None:
        async with self._lock:
            self._entries[key] = json.dumps(answer)

    def __len__(self) -> int:
        return len(self._entries)
