"""A scripted LangChain chat model for the agents' unit tests: no network, no cost (docs/AGENTS_PLAN.md A2).

``ScriptedChatModel`` plays prepared replies in order. Each step is a function of the messages so far and the call's
settings (the tools offered, the answer format), so one step works under either ``structured_output_method``:
``answer(...)`` calls ``final_answer`` under ``function_calling`` and replies with JSON under ``json_schema``.

When several agents share one model (Agent 2 inside Agent 1), ``steps`` is keyed by a tool name each agent offers
("*" matches any), so every agent plays its own queue.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field

Step = Callable[[list[BaseMessage], dict[str, Any]], AIMessage]
"""One scripted reply: given the messages and the call's settings, the AIMessage the model sends."""

ANSWER_TOOL = "final_answer"
USAGE = {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}


def _offered_tools(settings: dict[str, Any]) -> list[str]:
    return [t["function"]["name"] for t in settings.get("tools", [])]


def call_tool(name: str, call_id: str | None = None, **args: Any) -> Step:
    """A reply that calls the tool ``name`` with ``args``."""

    def step(messages: list[BaseMessage], settings: dict[str, Any]) -> AIMessage:
        cid = call_id or f"call_{len(messages)}"
        return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": cid}])

    return step


def answer(**fields: Any) -> Step:
    """The final answer, in whichever form the call asked for: a ``final_answer`` call, or a JSON reply."""

    def step(messages: list[BaseMessage], settings: dict[str, Any]) -> AIMessage:
        if ANSWER_TOOL in _offered_tools(settings):
            call = {"name": ANSWER_TOOL, "args": fields, "id": f"answer_{len(messages)}"}
            return AIMessage(content="", tool_calls=[call])
        return AIMessage(content=json.dumps(fields))

    return step


def say(text: str) -> Step:
    """A plain text reply with no tool call."""
    return lambda messages, settings: AIMessage(content=text)


def raise_(exc: BaseException) -> Step:
    """A model call that fails with ``exc``."""

    def step(messages: list[BaseMessage], settings: dict[str, Any]) -> AIMessage:
        raise exc

    return step


class ScriptedChatModel(BaseChatModel):
    """Plays ``steps`` in order, per agent; records every call's messages and settings in ``calls``."""

    steps: dict[str, list[Step]]
    reported_name: str = "scripted"
    calls: list[tuple[list[BaseMessage], dict[str, Any]]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(
        self, tools: Sequence[Any], *, tool_choice: Any = None, **kwargs: Any
    ) -> Runnable[LanguageModelInput, AIMessage]:
        specs = [t if isinstance(t, dict) else convert_to_openai_tool(t) for t in tools]
        return self.bind(tools=specs, tool_choice=tool_choice, **kwargs)

    def _queue(self, settings: dict[str, Any]) -> tuple[str, list[Step]]:
        offered = set(_offered_tools(settings))
        for key, queue in self.steps.items():
            if key == "*" or key in offered:
                return key, queue
        raise AssertionError(f"no script for a call offering {sorted(offered)}")

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        key, queue = self._queue(kwargs)
        self.calls.append((list(messages), dict(kwargs)))
        if not queue:
            raise AssertionError(f"the script for {key} ran out")
        reply = queue.pop(0)(list(messages), kwargs)
        if reply.usage_metadata is None:
            reply.usage_metadata = dict(USAGE)  # type: ignore[assignment]
        reply.response_metadata.setdefault("model_name", self.reported_name)
        return ChatResult(generations=[ChatGeneration(message=reply)])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self._generate(messages, stop, None, **kwargs)
