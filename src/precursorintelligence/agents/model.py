"""The model factory (docs/AGENTS_PLAN.md A2).

Production talks to Vertex AI's OpenAI-compatible endpoint as Cloud Run's service account: no API keys. A Google
access token lasts about an hour, so ``GoogleTokenAuth`` refreshes the credentials before **every** request and
writes the fresh token into that request's header. It keeps the credentials object, never a token string, so a
long-lived instance can't send an expired token.

The OpenAI client is built on ``httpx2`` (the ``openai`` package's HTTP library), so the auth is an ``httpx2.Auth``;
the API is the same as ``httpx.Auth``.

Each of the two models is wrapped in ``ConfiguredModel``, which records on every response which configured model
answered (``provider_details["configured_model"]``, the id from settings). The name the endpoint reports stays in
``ModelResponse.model_name`` untouched, but Vertex may spell it differently from our id (e.g. without ``openai/``),
so the runner decides "the primary answered" (and the price) from the tag, never from the reported string.

The unit tests use Pydantic AI's ``TestModel`` and ``FunctionModel``, or this factory over a mocked HTTP transport;
none reaches the network. ``scripts/agents_vertex_proof.py`` is L4's laptop proof against the real endpoint.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Generator
from dataclasses import replace
from typing import Any, Protocol

import httpx2
from openai import AsyncOpenAI
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

from .settings import AgentsSettings

CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"
# The OpenAI client insists on a key; the real Authorization header is set by GoogleTokenAuth on each request.
_PLACEHOLDER_KEY = "unused-google-token-set-per-request"


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
    )


CONFIGURED_MODEL_KEY = "configured_model"


class ConfiguredModel(WrapperModel):
    """Tags each response with the configured id of the model object that wrote it (``configured_model`` in
    ``provider_details``). Only ``request`` is tagged: the runner doesn't stream."""

    def __init__(self, wrapped: Model, configured_id: str) -> None:
        super().__init__(wrapped)
        self.configured_id = configured_id

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        response = await super().request(messages, model_settings, model_request_parameters)
        details = {**(response.provider_details or {}), CONFIGURED_MODEL_KEY: self.configured_id}
        return replace(response, provider_details=details)


def configured_model_name(response: ModelResponse) -> str | None:
    """The configured id of the model that wrote ``response``: the ``ConfiguredModel`` tag, else (a test model,
    which isn't wrapped) the reported name."""
    tagged = (response.provider_details or {}).get(CONFIGURED_MODEL_KEY)
    return tagged if isinstance(tagged, str) else response.model_name


def build_model(
    settings: AgentsSettings,
    *,
    credentials: RefreshableCredentials | None = None,
    transport: httpx2.AsyncBaseTransport | None = None,
    transport_request: Any = None,
) -> Model:
    """The primary model, falling back to the fallback model when a call to it fails (both ids from settings)."""
    client = build_openai_client(
        settings,
        credentials if credentials is not None else default_credentials(),
        transport=transport,
        transport_request=transport_request,
    )
    provider = OpenAIProvider(openai_client=client)
    primary, fallback = settings.model.primary, settings.model.fallback
    return FallbackModel(
        ConfiguredModel(OpenAIChatModel(primary, provider=provider), primary),
        ConfiguredModel(OpenAIChatModel(fallback, provider=provider), fallback),
    )
