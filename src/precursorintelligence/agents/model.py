"""The model factory (docs/AGENTS_PLAN.md A2).

Production talks to Vertex AI's OpenAI-compatible endpoint as Cloud Run's service account: no API keys. A Google
access token lasts about an hour, so ``GoogleTokenAuth`` refreshes the credentials before **every** request and
writes the fresh token into that request's header. It keeps the credentials object, never a token string, so a
long-lived instance can't send an expired token.

The OpenAI client is built on ``httpx2`` (the ``openai`` package's HTTP library), so the auth is an ``httpx2.Auth``;
the API is the same as ``httpx.Auth``.

Tests never build this; they pass Pydantic AI's ``TestModel`` or ``FunctionModel`` to the runner instead.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Generator
from typing import Any, Protocol

import httpx2
from openai import AsyncOpenAI
from pydantic_ai.models import Model
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

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
    primary = OpenAIChatModel(settings.model.primary, provider=provider)
    fallback = OpenAIChatModel(settings.model.fallback, provider=provider)
    return FallbackModel(primary, fallback)
