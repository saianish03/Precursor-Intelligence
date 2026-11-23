"""A minimal Vertex AI model factory for L4's laptop proof (docs/AGENTS_PLAN.md L4).

A stand-in for A2's factory (``agents/model.py``, PR #104): same endpoint, auth and model ids, so the proof
(``scripts/vertex_smoke.py``) exercises what production will run. When #104 lands, the smoke script can take its
model from ``build_model`` instead; ``gpt_oss_profile`` below is what that factory needs too.

gpt-oss on Vertex AI is a Model-as-a-Service open model behind the OpenAI-compatible Chat Completions endpoint:

- endpoint: ``https://aiplatform.googleapis.com/v1/projects/{project}/locations/global/endpoints/openapi`` for the
  global location (the one Google's gpt-oss pages list), or ``https://{region}-aiplatform.googleapis.com/...
  /locations/{region}/...`` for a single region such as ``us-central1``;
- model ids: ``openai/gpt-oss-120b-maas`` and ``openai/gpt-oss-20b-maas`` (with the publisher prefix);
- auth: a Google OAuth access token (scope cloud-platform) from Application Default Credentials, as a bearer token.

Two quirks shape the model profile. gpt-oss (OpenAI's Harmony format) rejects a forced ``tool_choice``
(``required`` or a named tool), which Pydantic AI sends by default for an output tool, so the profile disables
forced tool choice (Pydantic AI then sends ``auto`` and keeps the output tool). And Vertex documents only JSON mode
(``json_object``) for open models, not ``json_schema``, so native structured output is switched off: the
structured output comes back through the output tool (``ToolOutput``, Pydantic AI's default mode).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Generator
from typing import Any, Protocol

import httpx2
from openai import AsyncOpenAI
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.profiles import ModelProfile, merge_profile
from pydantic_ai.profiles.harmony import harmony_model_profile
from pydantic_ai.providers.openai import OpenAIProvider

CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"
GLOBAL_REGION = "global"
DEFAULT_REGION = GLOBAL_REGION
DEFAULT_MODEL = "openai/gpt-oss-120b-maas"
FALLBACK_MODEL = "openai/gpt-oss-20b-maas"
# The OpenAI client insists on a key; the real Authorization header is set by GoogleTokenAuth on each request.
_PLACEHOLDER_KEY = "unused-google-token-set-per-request"


class RefreshableCredentials(Protocol):
    """The part of ``google.auth.credentials.Credentials`` the auth uses."""

    token: str | None

    @property
    def valid(self) -> bool: ...

    def refresh(self, request: Any) -> None: ...


def vertex_base_url(project: str, region: str = DEFAULT_REGION) -> str:
    """Vertex AI's OpenAI-compatible base URL; the client appends ``/chat/completions``."""
    if not project or not region:
        raise ValueError("project and region are required")
    host = (
        "aiplatform.googleapis.com"
        if region == GLOBAL_REGION
        else f"{region}-aiplatform.googleapis.com"
    )
    return f"https://{host}/v1/projects/{project}/locations/{region}/endpoints/openapi"


def gpt_oss_profile(model_id: str) -> ModelProfile:
    """The Pydantic AI profile for gpt-oss on Vertex: no forced tool choice, no ``json_schema`` output."""
    base = harmony_model_profile(model_id.split("/", 1)[-1])
    return merge_profile(
        base,
        ModelProfile(
            supports_forced_tool_choice=False,
            supports_json_schema_output=False,
            supports_json_object_output=True,
            default_structured_output_mode="tool",
        ),
    )


def _google_transport_request() -> Any:
    from google.auth.transport.requests import Request

    return Request()


class GoogleTokenAuth(httpx2.Auth):
    """Sets ``Authorization: Bearer <token>`` on each request, refreshing the credentials when they are not
    valid (google-auth counts a token as invalid shortly before it expires). It keeps the credentials object,
    never a token string, so a long-lived client can't send an expired token."""

    def __init__(self, credentials: RefreshableCredentials, transport_request: Any = None) -> None:
        self._credentials = credentials
        self._transport_request = transport_request

    def _fresh_token(self) -> str:
        if not self._credentials.valid:
            if self._transport_request is None:
                self._transport_request = _google_transport_request()
            self._credentials.refresh(self._transport_request)
        token = self._credentials.token
        if not token:
            raise RuntimeError("the Google credentials returned no access token")
        return token

    def sync_auth_flow(
        self, request: httpx2.Request
    ) -> Generator[httpx2.Request, httpx2.Response, None]:
        request.headers["Authorization"] = f"Bearer {self._fresh_token()}"
        yield request

    async def async_auth_flow(
        self, request: httpx2.Request
    ) -> AsyncGenerator[httpx2.Request, httpx2.Response]:
        # refresh() does blocking network I/O, so keep it off the event loop
        token = await asyncio.to_thread(self._fresh_token)
        request.headers["Authorization"] = f"Bearer {token}"
        yield request


def default_credentials() -> RefreshableCredentials:
    """Application Default Credentials: ``gcloud auth application-default login`` on a laptop, the service
    account on Cloud Run."""
    import google.auth

    credentials, _ = google.auth.default(scopes=[CLOUD_PLATFORM_SCOPE])
    result: RefreshableCredentials = credentials
    return result


def vertex_model(
    project: str,
    region: str = DEFAULT_REGION,
    model_id: str = DEFAULT_MODEL,
    *,
    credentials: RefreshableCredentials | None = None,
    timeout_s: float = 60.0,
    transport: httpx2.AsyncBaseTransport | None = None,
    transport_request: Any = None,
) -> OpenAIChatModel:
    """gpt-oss on Vertex AI as a Pydantic AI Chat Completions model. Builds no token and makes no call: the
    first token is fetched by the first request."""
    http_client = httpx2.AsyncClient(
        auth=GoogleTokenAuth(
            credentials if credentials is not None else default_credentials(), transport_request
        ),
        timeout=timeout_s,
        transport=transport,
    )
    client = AsyncOpenAI(
        base_url=vertex_base_url(project, region),
        api_key=_PLACEHOLDER_KEY,
        http_client=http_client,
        timeout=timeout_s,
    )
    return OpenAIChatModel(
        model_id,
        provider=OpenAIProvider(openai_client=client),
        profile=gpt_oss_profile(model_id),
    )
