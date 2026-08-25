"""Lazy, bounded Google ADK adapter for advisory Gemini reasoning."""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterable
from typing import Any

from agentpulse.findings import Finding
from agentpulse.reasoning import (
    MODEL_NAME,
    PROMPT_VERSION,
    REASONING_INSTRUCTION,
    FailureCode,
    ReasoningFailure,
    ReasoningOutcome,
    ReasoningResult,
    canonical_payload_json,
    make_failure,
    payload_sha256,
    reasoning_wire_schema,
    validate_reasoning_output,
)

# One interactive request may run for at most 60 seconds per HTTP attempt.
REQUEST_TIMEOUT_MS = 60_000
APP_NAME = "agentpulse"
USER_ID = "local_operator"
SESSION_ID = "sanitized_reasoning"


def _outcome(
    payload_hash: str,
    value: ReasoningResult | ReasoningFailure,
) -> ReasoningOutcome:
    common = {
        "model": MODEL_NAME,
        "prompt_version": PROMPT_VERSION,
        "payload_sha256": payload_hash,
    }
    if isinstance(value, ReasoningResult):
        return ReasoningOutcome(**common, result=value)
    return ReasoningOutcome(**common, failure=value)


def _credential_is_configured() -> bool:
    # This mirrors google-genai precedence without exposing a credential value.
    return bool(os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"))


async def _final_text(events: AsyncIterable[Any]) -> str | None:
    final_text: str | None = None
    async for event in events:
        if not event.is_final_response():
            continue
        parts = event.content.parts if event.content and event.content.parts else []
        final_text = "".join(
            part.text
            for part in parts
            if part.text and not getattr(part, "thought", False)
        )
    return final_text


def _run_adk(payload_json: str) -> str | None:
    """Run one synchronous, tool-free ADK request and extract final JSON text."""
    from google.adk.agents import LlmAgent
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types

    transport = types.HttpOptions(
        timeout=REQUEST_TIMEOUT_MS,
        retry_options=types.HttpRetryOptions(
            attempts=2,
            http_status_codes=[503],
        ),
    )
    agent = LlmAgent(
        name="agentpulse_reasoning",
        description="Advisory classification of one sanitized secret-hygiene finding.",
        model=MODEL_NAME,
        instruction=REASONING_INSTRUCTION,
        output_schema=reasoning_wire_schema(),
        tools=[],
        generate_content_config=types.GenerateContentConfig(
            http_options=transport,
            thinking_config=types.ThinkingConfig(thinking_level="low"),
        ),
    )
    runner = Runner(
        app_name=APP_NAME,
        agent=agent,
        session_service=InMemorySessionService(),
        auto_create_session=True,
    )
    message = types.Content(
        role="user",
        parts=[
            types.Part.from_text(
                text=f"Sanitized finding JSON:\n{payload_json}",
            )
        ],
    )
    async def consume_events() -> str | None:
        return await _final_text(
            runner.run_async(
                user_id=USER_ID,
                session_id=SESSION_ID,
                new_message=message,
            )
        )

    return asyncio.run(consume_events())


def _classify_exception(exception: Exception) -> FailureCode:
    """Map installed google-genai/httpx exception types to bounded failures."""
    try:
        import httpx
        from google.genai import errors
    except Exception:
        return FailureCode.ADK_ERROR

    if isinstance(exception, httpx.TimeoutException):
        return FailureCode.TIMEOUT
    if isinstance(exception, httpx.NetworkError):
        return FailureCode.NETWORK_ERROR
    if isinstance(exception, errors.ClientError) and exception.code in (401, 403):
        return FailureCode.AUTH_FAILED
    if isinstance(exception, errors.ServerError):
        if exception.code == 504 or exception.status == "DEADLINE_EXCEEDED":
            return FailureCode.TIMEOUT
        return FailureCode.UNAVAILABLE
    return FailureCode.ADK_ERROR


def reason_finding(finding: Finding) -> ReasoningOutcome:
    """Reason over one finding without allowing provider failures to escape."""
    payload_json = canonical_payload_json(finding)
    payload_hash = payload_sha256(payload_json)
    if not _credential_is_configured():
        return _outcome(payload_hash, make_failure(FailureCode.NOT_CONFIGURED))

    try:
        raw_output = _run_adk(payload_json)
    except Exception as exception:
        code = _classify_exception(exception)
        return _outcome(payload_hash, make_failure(code, exception))
    return _outcome(payload_hash, validate_reasoning_output(raw_output))
