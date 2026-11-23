"""The agents' runtime settings, from ``configs/agents_runtime.yaml`` (docs/AGENTS_PLAN.md A1).

``configs/agents.yaml`` holds the evidence build's settings (E2), loaded by ``common/config.py``.

Loaded like ``configs/ml.yaml``: ``${VAR}`` placeholders are expanded from the environment, and an unknown or
misspelled key is a ``ConfigError``. Every value is in the file; the models below have no defaults to fall back on.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator

from precursorintelligence.common.config import ConfigError, _load_model, _Strict, project_root


class GcpCfg(_Strict):
    project: str = Field(min_length=1)
    region: str = Field(min_length=1)


class ModelCfg(_Strict):
    base_url: str
    primary: str = Field(min_length=1)
    # None (empty in the file): no fallback model, a failed call fails the run.
    fallback: str | None
    request_timeout_s: float = Field(gt=0)
    # How an agent's final answer is asked for (the two methods of LangChain's with_structured_output):
    # "function_calling" (a final_answer tool) or "json_schema" (a response_format). L4 checks which one gpt-oss on
    # Vertex supports.
    structured_output_method: Literal["function_calling", "json_schema"]

    @field_validator("base_url")
    @classmethod
    def _https_only(cls, v: str) -> str:
        if not v.startswith("https://"):
            raise ValueError("model.base_url must use https://")
        return v.rstrip("/")

    @field_validator("fallback", mode="before")
    @classmethod
    def _empty_is_none(cls, v: object) -> object:
        return None if v is None or (isinstance(v, str) and not v.strip()) else v


class AgentCfg(_Strict):
    max_tokens: int = Field(gt=0)
    temperature: float = Field(ge=0, le=2)


class LimitsCfg(_Strict):
    request_limit: int = Field(ge=1)
    input_tokens_limit: int = Field(ge=1)
    output_tokens_limit: int = Field(ge=1)
    tool_calls_limit: int = Field(ge=1)
    max_subagent_calls: int = Field(ge=0)  # k: Agent 2 calls per Agent 1 run


class ToolsCfg(_Strict):
    timeout_s: float = Field(gt=0)


class CostControlCfg(_Strict):
    daily_run_cap: int = Field(ge=1)
    daily_spend_stop_usd: float = Field(gt=0)


class CacheCfg(_Strict):
    bucket: str = Field(min_length=1)


class PriceCfg(_Strict):
    """USD per million tokens."""

    input: float = Field(ge=0)
    output: float = Field(ge=0)


class AgentsSettings(_Strict):
    """``configs/agents_runtime.yaml``."""

    version: Literal[1]
    gcp: GcpCfg
    model: ModelCfg
    agents: dict[str, AgentCfg]
    limits: LimitsCfg
    tools: ToolsCfg
    cost_control: CostControlCfg
    cache: CacheCfg
    prices: dict[str, PriceCfg]

    @model_validator(mode="after")
    def _every_model_has_a_price(self) -> AgentsSettings:
        models = [self.model.primary] + ([self.model.fallback] if self.model.fallback else [])
        missing = [m for m in models if m not in self.prices]
        if missing:
            raise ValueError(f"prices has no entry for the models {missing}")
        return self

    def agent(self, name: str) -> AgentCfg:
        """The settings of agent ``name``; ``ConfigError`` for an agent the file doesn't list."""
        try:
            return self.agents[name]
        except KeyError:
            raise ConfigError(
                f"configs/agents_runtime.yaml has no settings for agent {name!r}; known: {sorted(self.agents)}"
            ) from None

    def price(self, model_name: str) -> PriceCfg:
        """The price of ``model_name``, or the highest listed price when it isn't listed, so a cost estimate never
        comes out low."""
        if model_name in self.prices:
            return self.prices[model_name]
        return PriceCfg(
            input=max(p.input for p in self.prices.values()),
            output=max(p.output for p in self.prices.values()),
        )

    def cost_usd(self, model_name: str, input_tokens: int, output_tokens: int) -> float:
        """The estimated cost of one run's tokens on ``model_name`` (C3)."""
        p = self.price(model_name)
        return (input_tokens * p.input + output_tokens * p.output) / 1_000_000


def load_agents_settings(path: str | Path | None = None) -> AgentsSettings:
    """Load and validate ``configs/agents_runtime.yaml``."""
    return _load_model(AgentsSettings, path or project_root() / "configs" / "agents_runtime.yaml")
