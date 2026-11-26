"""The prompt loader (docs/AGENTS_PLAN.md A4)."""

from __future__ import annotations

import pytest

from precursorintelligence.agents.prompts import load_prompt
from precursorintelligence.common.config import ConfigError

GOOD = "version: 3\nsystem: You are careful.\ninstructions: Cite evidence IDs.\n"


def test_a_good_prompt_loads(tmp_path):
    (tmp_path / "demo.yaml").write_text(GOOD, encoding="utf-8")
    p = load_prompt("demo", tmp_path)
    assert (p.version, p.system, p.instructions) == (3, "You are careful.", "Cite evidence IDs.")


def test_a_missing_version_fails(tmp_path):
    (tmp_path / "demo.yaml").write_text(
        "system: You are careful.\ninstructions: Cite.\n", encoding="utf-8"
    )
    with pytest.raises(ConfigError, match="version"):
        load_prompt("demo", tmp_path)


@pytest.mark.parametrize("version", ["0", "null", "'v1'"])
def test_a_bad_version_fails(tmp_path, version):
    (tmp_path / "demo.yaml").write_text(GOOD.replace("3", version), encoding="utf-8")
    with pytest.raises(ConfigError, match="version"):
        load_prompt("demo", tmp_path)


def test_an_unknown_key_fails(tmp_path):
    (tmp_path / "demo.yaml").write_text(GOOD + "temprature: 0.2\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="temprature"):
        load_prompt("demo", tmp_path)


def test_a_missing_file_fails(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_prompt("demo", tmp_path)


def test_a_path_like_agent_name_is_refused(tmp_path):
    with pytest.raises(ConfigError, match="not an agent name"):
        load_prompt("../secrets", tmp_path)
