"""configs/agents_runtime.yaml and agents/settings.py (docs/AGENTS_PLAN.md A1)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from precursorintelligence.agents.settings import load_agents_settings
from precursorintelligence.common.config import ConfigError, project_root

REPO_FILE = project_root() / "configs" / "agents_runtime.yaml"


def _write(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "agents_runtime.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def _repo_data() -> dict:
    return yaml.safe_load(REPO_FILE.read_text(encoding="utf-8"))


def test_repo_file_loads_with_the_plan_limits():
    s = load_agents_settings()
    assert s.limits.request_limit == 6
    assert s.limits.input_tokens_limit == 200_000
    assert s.limits.output_tokens_limit == 12_000
    assert s.limits.max_subagent_calls >= 1
    assert s.cost_control.daily_run_cap == 100
    assert s.cost_control.daily_spend_stop_usd == 2.0
    assert s.model.base_url.startswith("https://")
    for name in ("facility_investigation", "evidence_retriever", "model_ops", "portfolio_briefing"):
        assert s.agent(name).max_tokens > 0


def test_a_misspelled_key_is_a_config_error(tmp_path):
    data = _repo_data()
    data["limits"]["request_limt"] = data["limits"].pop("request_limit")
    with pytest.raises(ConfigError, match="request_limt"):
        load_agents_settings(_write(tmp_path, data))


def test_an_unknown_top_level_key_is_a_config_error(tmp_path):
    data = _repo_data()
    data["extra"] = 1
    with pytest.raises(ConfigError):
        load_agents_settings(_write(tmp_path, data))


def test_environment_overrides_the_model_and_region(monkeypatch):
    monkeypatch.setenv("PI_AGENTS_REGION", "europe-west4")
    monkeypatch.setenv("PI_AGENTS_MODEL", "openai/gpt-oss-20b-maas")
    s = load_agents_settings()
    assert s.gcp.region == "europe-west4"
    assert "europe-west4-aiplatform.googleapis.com" in s.model.base_url
    assert "/locations/europe-west4/" in s.model.base_url
    assert s.model.primary == "openai/gpt-oss-20b-maas"


def test_json_schema_is_the_default_structured_output_method(monkeypatch):
    assert load_agents_settings().model.structured_output_method == "json_schema"  # L4's choice
    monkeypatch.setenv("PI_AGENTS_STRUCTURED_OUTPUT", "function_calling")
    assert load_agents_settings().model.structured_output_method == "function_calling"


def test_the_fallback_model_is_empty_by_default_and_set_from_the_environment(monkeypatch):
    assert load_agents_settings().model.fallback is None
    monkeypatch.setenv("PI_AGENTS_FALLBACK_MODEL", "openai/gpt-oss-20b-maas")
    assert load_agents_settings().model.fallback == "openai/gpt-oss-20b-maas"


def test_a_fallback_model_without_a_price_is_a_config_error(monkeypatch):
    monkeypatch.setenv("PI_AGENTS_FALLBACK_MODEL", "openai/some-new-model")
    with pytest.raises(ConfigError, match="prices"):
        load_agents_settings()


def test_a_model_without_a_price_is_a_config_error(monkeypatch):
    monkeypatch.setenv("PI_AGENTS_MODEL", "openai/some-new-model")
    with pytest.raises(ConfigError, match="prices"):
        load_agents_settings()


def test_an_unknown_agent_is_a_config_error():
    with pytest.raises(ConfigError, match="no settings for agent"):
        load_agents_settings().agent("facility_investigaton")


def test_http_base_url_is_refused(tmp_path):
    data = _repo_data()
    data["model"]["base_url"] = "http://example.com"
    with pytest.raises(ConfigError, match="https"):
        load_agents_settings(_write(tmp_path, data))


def test_cost_matches_a_hand_calculation():
    s = load_agents_settings()
    # 50k input at $0.15/M = $0.0075, 5k output at $0.60/M = $0.003
    assert s.cost_usd("openai/gpt-oss-120b-maas", 50_000, 5_000) == pytest.approx(0.0105)


def test_an_unlisted_model_is_priced_at_the_highest_price():
    s = load_agents_settings()
    assert s.cost_usd("test", 1_000_000, 1_000_000) == pytest.approx(0.15 + 0.60)
