"""The model factory (docs/AGENTS_PLAN.md A2).

Production talks to Vertex AI's OpenAI-compatible endpoint as Cloud Run's service account: no API keys. A Google
access token lasts about an hour, so ``GoogleTokenAuth`` refreshes the credentials before **every** request and
writes the fresh token into that request's header. It keeps the credentials object, never a token string, so a
long-lived instance can't send an expired token.

The OpenAI client is built on ``httpx2`` (the ``openai`` package's HTTP library), so the auth is an ``httpx2.Auth``;
the API is the same as ``httpx.Auth``.

The two models are LangChain ``ChatOpenAI`` models in a ``ModelPair``: each call tries the primary and moves to the
fallback when it fails, and every reply comes back with the configured id of the model that wrote it. The name the
endpoint reports (``response_metadata["model_name"]``) is kept, but Vertex may spell it differently from our id (e.g.
without ``openai/``), so the runner decides "the primary answered" (and the price) from the configured id, never from
the reported string.

The unit tests use ``agents/testing.py``'s scripted chat model, or this factory over a mocked HTTP transport; none
reaches the network. ``scripts/agents_vertex_proof.py`` is L4's laptop proof against the real endpoint.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncGenerator, Callable, Generator, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import httpx2
import openai
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from openai import AsyncOpenAI
from pydantic import SecretStr

from .errors import ModelCallFailed
from .settings import AgentsSettings

CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"
# The OpenAI client insists on a key; the real Authorization header is set by GoogleTokenAuth on each request.
_PLACEHOLDER_KEY = "unused-google-token-set-per-request"

log = logging.getLogger(__name__)

# Every environment variable LangSmith (it comes with langchain-core) reads to turn tracing on.
LANGSMITH_TRACING_VARS = (
    "LANGSMITH_TRACING",
    "LANGSMITH_TRACING_V2",
    "LANGCHAIN_TRACING",
    "LANGCHAIN_TRACING_V2",
)


def disable_langsmith_tracing() -> None:
    """Turn LangSmith tracing off, whatever the environment says, so prompts and tool data never go to a third
    party by accident. Called when this module is imported and again by ``build_models``. Tracing is C1's job, in
    Cloud Trace."""
    on = [
        name
        for name in LANGSMITH_TRACING_VARS
        if os.environ.get(name, "").strip().lower() == "true"
    ]
    if on:
        log.warning("LangSmith tracing is forced off (it was turned on by %s)", ", ".join(on))
    for name in LANGSMITH_TRACING_VARS:
        os.environ[name] = "false"
    import langsmith

    langsmith.configure(enabled=False)


disable_langsmith_tracing()


class RefreshableCredentials(Protocol):
    """The part of ``google.auth.credentials.Credentials`` the auth uses."""

    token: str | None

    def refresh(self, request: Any) -> None: ...


def _google_transport_request() -> Any:
    from google.auth.transport.requests import Request

    return Request()


class GoogleTokenAuth(httpx2.Auth):
    """Sets ``Authorization: Bearer <token>`` from credentials refreshed for each request."""

    def __init__(self, credentials: RefreshableCredentials, transport_request: Any = None) -> None:
        self._credentials = credentials
        self._transport_request = (
            transport_request if transport_request is not None else _google_transport_request()
        )

    def _fresh_token(self) -> str:
        self._credentials.refresh(self._transport_request)
        token = self._credentials.token
        if not token:
            raise RuntimeError("the Google credentials returned no access token")
        return token

    def sync_auth_flow(self, request: httpx2.Request) -> Generator[httpx2.Request, httpx2.Response]:
        request.headers["Authorization"] = f"Bearer {self._fresh_token()}"
        yield request

    async def async_auth_flow(
        self, request: httpx2.Request
    ) -> AsyncGenerator[httpx2.Request, httpx2.Response]:
        # refresh() does blocking network I/O (the metadata server on Cloud Run), so keep it off the event loop
        token = await asyncio.to_thread(self._fresh_token)
        request.headers["Authorization"] = f"Bearer {token}"
        yield request


def default_credentials() -> RefreshableCredentials:
    """Application Default Credentials: Cloud Run's service account, or ``gcloud auth application-default login``
    on a laptop (L4)."""
    import google.auth

    credentials, _ = google.auth.default(scopes=[CLOUD_PLATFORM_SCOPE])
    return credentials


def build_openai_client(
    settings: AgentsSettings,
    credentials: RefreshableCredentials,
    *,
    transport: httpx2.AsyncBaseTransport | None = None,
    transport_request: Any = None,
) -> AsyncOpenAI:
    """An OpenAI client for the Vertex endpoint whose every request carries a freshly refreshed token."""
    http_client = httpx2.AsyncClient(
        auth=GoogleTokenAuth(credentials, transport_request),
        timeout=settings.model.request_timeout_s,
        transport=transport,
    )
    return AsyncOpenAI(
        base_url=settings.model.base_url,
        api_key=_PLACEHOLDER_KEY,
        http_client=http_client,
        timeout=settings.model.request_timeout_s,
        # no retries inside the client: the runner has its own fallback, and a hidden retry would double a slow
        # call's time and go uncounted in the run's usage
        max_retries=0,
    )


MODEL_ERRORS: tuple[type[BaseException], ...] = (openai.APIError, httpx2.HTTPError, TimeoutError)
"""What makes one model call move from the primary to the fallback model (an HTTP error, a refused request, a
timeout). langchain-openai re-raises the API errors as subclasses of ``openai.APIError``."""

Bind = Callable[[BaseChatModel], Runnable[Any, BaseMessage]]


@dataclass(frozen=True)
class ModelAnswer:
    """One model reply and the configured id of the model that wrote it (from settings, not the reported name)."""

    message: AIMessage
    configured_id: str

    @property
    def reported_name(self) -> str | None:
        """The model name the endpoint reported, which may be spelled differently from the configured id."""
        name = self.message.response_metadata.get("model_name")
        return name if isinstance(name, str) else None


@dataclass(frozen=True)
class ModelPair:
    """The primary chat model and the fallback behind it, each with its configured id. Every call tries the
    primary first and moves to the fallback when it fails, so the runner always knows which model answered."""

    primary: BaseChatModel
    primary_id: str
    fallback: BaseChatModel | None = None
    fallback_id: str | None = None

    async def ainvoke(self, messages: Sequence[BaseMessage], bind: Bind) -> ModelAnswer:
        """``bind`` adds the tools, the answer format and the sampling settings to a model."""
        try:
            return ModelAnswer(
                _as_ai(await bind(self.primary).ainvoke(list(messages))), self.primary_id
            )
        except MODEL_ERRORS as first:
            if self.fallback is None or self.fallback_id is None:
                raise ModelCallFailed(f"{self.primary_id}: {_describe(first)}") from first
            try:
                reply = await bind(self.fallback).ainvoke(list(messages))
            except MODEL_ERRORS as second:
                raise ModelCallFailed(
                    f"{self.primary_id}: {_describe(first)}; then {self.fallback_id}: {_describe(second)}"
                ) from second
            return ModelAnswer(_as_ai(reply), self.fallback_id)


def _describe(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _as_ai(message: BaseMessage) -> AIMessage:
    if not isinstance(message, AIMessage):
        raise TypeError(f"the chat model returned a {type(message).__name__}, not an AIMessage")
    return message


def build_chat_model(model_id: str, client: AsyncOpenAI) -> ChatOpenAI:
    """A LangChain chat model for one Vertex model id, sending through ``client`` (and its token auth)."""
    return ChatOpenAI(
        model=model_id,
        base_url=str(client.base_url),
        api_key=SecretStr(_PLACEHOLDER_KEY),
        root_async_client=client,
        async_client=client.chat.completions,
    )


def build_models(
    settings: AgentsSettings,
    *,
    credentials: RefreshableCredentials | None = None,
    transport: httpx2.AsyncBaseTransport | None = None,
    transport_request: Any = None,
) -> ModelPair:
    """The primary model with the fallback model behind it, if one is configured (ids from settings), sharing one
    client."""
    disable_langsmith_tracing()
    client = build_openai_client(
        settings,
        credentials if credentials is not None else default_credentials(),
        transport=transport,
        transport_request=transport_request,
    )
    primary, fallback = settings.model.primary, settings.model.fallback
    return ModelPair(
        primary=build_chat_model(primary, client),
        primary_id=primary,
        fallback=build_chat_model(fallback, client) if fallback else None,
        fallback_id=fallback or None,
    )
