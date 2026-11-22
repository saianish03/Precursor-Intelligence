"""``run_agent()``: the one path every agent call takes (docs/AGENTS_PLAN.md A3).

In order:

1. **Input checks**: the raw input is validated against the agent's input model (``AgentInputError``, a 400).
2. **Cache lookup** (D1): a hit returns at once. It is free, so the daily cap never blocks or counts it.
3. **Daily cap** (D3): one run is reserved, or ``DailyCapReached`` (a 429). Nested runs reserve their own.
4. **The agent run**, with a fresh evidence ledger (every tool result is recorded in it) and the D4 usage limits.
5. **Validators**: validator 1 (the evidence IDs are in the ledger), then the agent's own. A failure raises
   ``ModelRetry``, so Pydantic AI sends the error back to the model **inside the same run** (one output retry): the
   ledger and the request limit carry over.
6. **Fallback**: when the run still fails (a second invalid answer, a model error, a usage limit, a tool timeout),
   the agent's template answers from the tool data alone, marked ``fallback=True``. It is never cached, and its run
   is given back to the daily cap (its cost stays).
7. **Cache store** of a validated answer, the **log line** and the **trace span** (hooks; C1 and C2 make them real).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, ValidationError
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.exceptions import AgentRunError, FallbackExceptionGroup
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.usage import RunUsage, UsageLimits

from .cache import AnswerCache, CacheKey
from .counter import DailyCapReached, DailyCounter
from .ledger import EvidenceLedger, check_evidence_ids, with_ledger
from .model import configured_model_name
from .prompts import Prompt, load_prompt
from .settings import AgentsSettings

InT = TypeVar("InT", bound=BaseModel)
OutT = TypeVar("OutT", bound=BaseModel)

# What ends a run in the fallback answer rather than an error: the model or the tools failed, not our code.
FALLBACK_ON: tuple[type[BaseException], ...] = (AgentRunError, FallbackExceptionGroup, TimeoutError)

Validator = Callable[[OutT, "RunDeps"], None]
"""Checks an output; raises ``ModelRetry`` with a message the model can act on when it fails."""


class AgentInputError(ValueError):
    """The input failed its checks; the API answers 400 before any model call."""


@dataclass(frozen=True)
class DataVersion:
    """Which data a run answers from, read from ``agents/evidence/_current.json`` (and the corpus manifest)."""

    v: str
    evidence_manifest_sha: str
    corpus_version: str | None = None


@dataclass(frozen=True)
class AgentDefinition(Generic[InT, OutT]):
    """One agent: its types, its tools, its validators and its fallback template. The prompt is loaded by name."""

    name: str
    input_type: type[InT]
    output_type: type[OutT]
    fallback: Callable[[InT, RunDeps], Awaitable[OutT]]
    toolsets: Sequence[AbstractToolset[Any]] = ()
    validators: Sequence[Validator[OutT]] = ()
    uses_corpus: bool = (
        False  # Agent 2, and Agent 1 (it calls Agent 2): the corpus version joins the cache key
    )
    user_prompt: Callable[[InT], str] = lambda inp: inp.model_dump_json()


@dataclass(frozen=True)
class RunLog:
    """One line per agent run (C2 turns this into the JSON log line)."""

    agent: str
    prompt_version: int | None
    model: str
    request_id: str
    parent: str | None
    outcome: Literal["answered", "cache_hit", "fallback", "capped", "error"]
    cache_hit: bool
    fallback: bool
    requests: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    validation_failures: tuple[str, ...]
    error: str | None


class RunHooks:
    """The log and trace hooks; both do nothing here (C1 and C2 subclass this)."""

    def log(self, line: RunLog) -> None:
        return None

    @contextmanager
    def span(self, name: str, attributes: dict[str, Any]) -> Iterator[None]:
        yield


def _utc_today() -> date:
    return datetime.now(UTC).date()


@dataclass
class Runtime:
    """What every run shares: settings, the model, the cache, the daily counter and the hooks."""

    settings: AgentsSettings
    model: Model
    cache: AnswerCache
    counter: DailyCounter
    hooks: RunHooks = field(default_factory=RunHooks)
    prompts_dir: Path | None = None
    today: Callable[[], date] = _utc_today
    clock: Callable[[], float] = time.perf_counter
    _prompts: dict[str, Prompt] = field(default_factory=dict, repr=False)

    def prompt(self, agent: str) -> Prompt:
        if agent not in self._prompts:
            self._prompts[agent] = load_prompt(agent, self.prompts_dir)
        return self._prompts[agent]


@dataclass
class RunDeps:
    """The ``deps`` of one run: the tools reach their data through ``tools``, and the runner's state lives here."""

    runtime: Runtime
    agent: str
    data: DataVersion
    request_id: str
    tools: Any = None
    ledger: EvidenceLedger = field(default_factory=EvidenceLedger)
    validation_failures: list[str] = field(default_factory=list)
    subagent_calls: int = 0

    async def run_subagent(
        self, definition: AgentDefinition[Any, OutT], raw_input: Any, *, tools: Any = None
    ) -> OutT:
        """Run another agent from a tool (Agent 2 from Agent 1). It counts as a run of its own (the daily cap) and
        at most ``limits.max_subagent_calls`` are allowed per run (D4). Its answer's evidence IDs join this run's
        ledger when the tool returns it."""
        k = self.runtime.settings.limits.max_subagent_calls
        if self.subagent_calls >= k:
            raise ModelRetry(
                f"{definition.name} can be called at most {k} times in one run; answer with the evidence you have."
            )
        self.subagent_calls += 1
        try:
            outcome = await run_agent(
                self.runtime,
                definition,
                raw_input,
                data=self.data,
                request_id=self.request_id,
                tools=tools,
                parent=self.agent,
            )
        except AgentInputError as exc:
            # the model wrote the nested input, not the user: tell the model, don't fail this run with a 400
            raise ModelRetry(f"{definition.name} refused that input: {exc}") from exc
        except DailyCapReached as exc:
            raise ModelRetry(
                f"{definition.name} is unavailable ({exc}); answer with the evidence you have."
            ) from exc
        return outcome.output


@dataclass(frozen=True)
class RunOutcome(Generic[OutT]):
    output: OutT
    cache_key: str
    cache_hit: bool
    fallback: bool
    cost_usd: float
    input_tokens: int
    output_tokens: int
    validation_failures: tuple[str, ...]


def build_cache_key(
    runtime: Runtime, definition: AgentDefinition[Any, Any], inp: BaseModel, data: DataVersion
) -> CacheKey:
    if definition.uses_corpus and not data.corpus_version:
        raise ValueError(f"{definition.name} reads the corpus, so its data needs a corpus_version")
    dumped = inp.model_dump(mode="json")
    question_type = dumped.get("question_type")
    return CacheKey(
        agent=definition.name,
        prompt_version=runtime.prompt(definition.name).version,
        model=runtime.settings.model.primary,
        question_type=str(question_type) if question_type is not None else None,
        input=dumped,
        evidence_manifest_sha=data.evidence_manifest_sha,
        corpus_version=data.corpus_version if definition.uses_corpus else None,
    )


def _build_agent(definition: AgentDefinition[Any, OutT], prompt: Prompt) -> Agent[RunDeps, OutT]:
    agent: Agent[RunDeps, OutT] = Agent(
        output_type=definition.output_type,
        system_prompt=prompt.system,
        instructions=prompt.instructions,
        deps_type=RunDeps,
        toolsets=[with_ledger(ts) for ts in definition.toolsets],
        name=definition.name,
        defer_model_check=True,
    )

    @agent.output_validator
    def _validate(ctx: RunContext[RunDeps], output: OutT) -> OutT:
        checks: list[Callable[[OutT], None]] = [
            lambda o: check_evidence_ids(o, ctx.deps.ledger),
            *(lambda o, v=v: v(o, ctx.deps) for v in definition.validators),
        ]
        for check in checks:
            try:
                check(output)
            except ModelRetry as exc:
                ctx.deps.validation_failures.append(str(exc))
                raise
        return output

    return agent


def _parse_cached(output_type: type[OutT], cached: dict[str, Any] | None) -> OutT | None:
    """A cache entry as the output type, or None (a miss) when there is none or it no longer fits the type (the
    output model changed without a prompt version bump); the fresh answer then overwrites it."""
    if cached is None:
        return None
    try:
        return output_type.model_validate(cached)
    except ValidationError:
        return None


def _answered_by(messages: Sequence[ModelMessage], model_name: str) -> bool:
    """True when the configured model ``model_name`` wrote every response in the run. Decided from which model
    object answered (``ConfiguredModel``'s tag), not the name the endpoint reports, which may be spelled
    differently."""
    responses = [m for m in messages if isinstance(m, ModelResponse)]
    return bool(responses) and all(configured_model_name(r) == model_name for r in responses)


def _set_fallback_flag(output: OutT) -> OutT:
    if "fallback" in type(output).model_fields:
        return output.model_copy(update={"fallback": True})
    return output


async def run_agent(
    runtime: Runtime,
    definition: AgentDefinition[InT, OutT],
    raw_input: Any,
    *,
    data: DataVersion,
    request_id: str,
    tools: Any = None,
    parent: str | None = None,
) -> RunOutcome[OutT]:
    """Answer ``raw_input`` with ``definition``'s agent; see the module docstring for the steps."""
    s = runtime.settings
    started = runtime.clock()
    prompt_version: int | None = None
    usage = RunUsage()
    failures: list[str] = []

    def emit(
        outcome: Literal["answered", "cache_hit", "fallback", "capped", "error"],
        *,
        cost: float = 0.0,
        model: str = s.model.primary,
        error: str | None = None,
    ) -> None:
        runtime.hooks.log(
            RunLog(
                agent=definition.name,
                prompt_version=prompt_version,
                model=model,
                request_id=request_id,
                parent=parent,
                outcome=outcome,
                cache_hit=outcome == "cache_hit",
                fallback=outcome == "fallback",
                requests=usage.requests,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cost_usd=cost,
                latency_ms=(runtime.clock() - started) * 1000,
                validation_failures=tuple(failures),
                error=error,
            )
        )

    span_attrs = {"agent": definition.name, "request_id": request_id, "parent": parent or ""}
    with runtime.hooks.span(f"agent.{definition.name}", span_attrs):
        # 1. input checks
        try:
            inp = definition.input_type.model_validate(
                raw_input.model_dump() if isinstance(raw_input, BaseModel) else raw_input
            )
        except ValidationError as exc:
            raise AgentInputError(f"invalid input for {definition.name}: {exc}") from exc
        prompt = runtime.prompt(definition.name)
        prompt_version = prompt.version
        agent_cfg = s.agent(definition.name)

        # 2. cache lookup: free, so it comes before the cap
        key = build_cache_key(runtime, definition, inp, data)
        digest = key.digest()
        cached = await runtime.cache.get(digest)
        hit = _parse_cached(definition.output_type, cached)
        if hit is not None:
            emit("cache_hit")
            return RunOutcome(
                output=hit,
                cache_key=digest,
                cache_hit=True,
                fallback=False,
                cost_usd=0.0,
                input_tokens=0,
                output_tokens=0,
                validation_failures=(),
            )

        # 3. daily cap
        day = runtime.today()
        try:
            await runtime.counter.reserve_run(day)
        except DailyCapReached as exc:
            emit("capped", error=str(exc))
            raise

        # 4-5. the run, with its validators and one output retry inside it
        deps = RunDeps(
            runtime=runtime,
            agent=definition.name,
            data=data,
            request_id=request_id,
            tools=tools,
            validation_failures=failures,
        )
        model_name = s.model.primary
        error: str | None = None
        outcome: Literal["answered", "fallback", "error"] = "error"
        fallback = False
        by_primary = False
        cost = 0.0
        try:
            try:
                result = await _build_agent(definition, prompt).run(
                    definition.user_prompt(inp),
                    deps=deps,
                    model=runtime.model,
                    model_settings=ModelSettings(
                        max_tokens=agent_cfg.max_tokens, temperature=agent_cfg.temperature
                    ),
                    usage_limits=UsageLimits(
                        request_limit=s.limits.request_limit,
                        input_tokens_limit=s.limits.input_tokens_limit,
                        output_tokens_limit=s.limits.output_tokens_limit,
                        tool_calls_limit=s.limits.tool_calls_limit,
                    ),
                    usage=usage,
                    retries={"output": 1},
                )
                output = result.output
                model_name = configured_model_name(result.response) or model_name
                by_primary = _answered_by(result.all_messages(), s.model.primary)
            except FALLBACK_ON as exc:
                # 6. fallback: the template answers from the tool data alone
                error = f"{type(exc).__name__}: {exc}"
                fallback = True
                output = _set_fallback_flag(await definition.fallback(inp, deps))
            if fallback:
                await runtime.counter.release_run(day)
            elif by_primary:
                # 7. only a validated answer is stored, and only one the primary model wrote: the cache key names
                # the primary, so an answer from the fallback model is served this once and never cached
                await runtime.cache.put(digest, output.model_dump(mode="json"))
            outcome = "fallback" if fallback else "answered"
        except BaseException as exc:
            # our bug, a data error, a failing fallback template or a cancelled request: no answer, but the cost and
            # the log line below still count (BaseException, so a cancellation is counted too)
            what = f"{type(exc).__name__}: {exc}"
            error = f"{error}; then {what}" if error else what
            raise
        finally:
            cost = s.cost_usd(model_name, usage.input_tokens, usage.output_tokens)
            if cost:
                # shielded, so a cancellation can't stop the spend from being counted
                await asyncio.shield(runtime.counter.add_cost(day, cost))
            emit(outcome, cost=cost, model=model_name, error=error)
        return RunOutcome(
            output=output,
            cache_key=digest,
            cache_hit=False,
            fallback=fallback,
            cost_usd=cost,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            validation_failures=tuple(failures),
        )
