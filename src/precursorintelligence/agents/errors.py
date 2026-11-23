"""The agents runtime's exceptions (docs/AGENTS_PLAN.md A3).

``ModelRetry`` is the one a validator or a tool raises: its message goes back to the model inside the same run. The
``AgentRunError`` family ends a run in the fallback template (B9): the model or its budget failed, not our code.
"""

from __future__ import annotations


class ModelRetry(Exception):
    """Raised by a validator or a tool: the message is sent back to the model, which may try again in the same run.
    From a validator it uses up the run's one output retry; from a tool it is that tool call's result."""


class AgentRunError(RuntimeError):
    """The run failed for a reason the fallback template covers: the model, its answers or its budget."""


class ModelCallFailed(AgentRunError):
    """A model call failed on the primary and then on the fallback model."""


class UsageLimitExceeded(AgentRunError):
    """A D4 limit was reached: model requests, input or output tokens, or tool calls."""


class OutputRejected(AgentRunError):
    """The answer still failed its checks after the one output retry."""
