"""Versioned prompts, from ``configs/prompts/<agent>.yaml`` (docs/AGENTS_PLAN.md A4).

Each file holds exactly ``version``, ``system`` and ``instructions``. A prompt without a version is refused: the
version is part of the cache key (D1) and of every log line and trace (C1, C2), and it only goes up when the agent's
golden set passes (J4).
"""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import Field

from precursorintelligence.common.config import ConfigError, _load_model, _Strict, project_root

AGENT_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


class Prompt(_Strict):
    version: int = Field(ge=1)
    system: str = Field(min_length=1)
    instructions: str = Field(min_length=1)


def load_prompt(agent: str, prompts_dir: str | Path | None = None) -> Prompt:
    """The prompt of ``agent``; ``ConfigError`` when the file is missing, has no version, or has an unknown key."""
    if not AGENT_NAME.match(agent):
        raise ConfigError(f"not an agent name: {agent!r}")
    base = Path(prompts_dir) if prompts_dir is not None else project_root() / "configs" / "prompts"
    return _load_model(Prompt, base / f"{agent}.yaml")
