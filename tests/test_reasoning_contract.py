"""Strict local validation tests for Gemini structured output."""

from __future__ import annotations

import json

import pytest

from agentpulse.reasoning import (
    REASONING_INSTRUCTION,
    FailureCode,
    Priority,
    ReasoningFailure,
    ReasoningResult,
    RecommendationClass,
    Severity,
    make_failure,
    validate_reasoning_output,
)
from tests.conftest import DUMMY_SECRET


def _valid_payload() -> dict[str, object]:
    return {
        "severity": "HIGH",
        "priority": "P1",
        "rationale": "The confirmed finding requires prompt human attention.",
        "recommendation_class": "IGNORE_FILE",
    }


def _assert_invalid(raw_output: str) -> None:
    result = validate_reasoning_output(raw_output)
    assert isinstance(result, ReasoningFailure)
    assert result.code is FailureCode.INVALID_OUTPUT


def test_valid_output_returns_frozen_reasoning_result() -> None:
    result = validate_reasoning_output(json.dumps(_valid_payload()))
    assert isinstance(result, ReasoningResult)
    assert result.severity is Severity.HIGH
    assert result.priority is Priority.P1
    assert result.recommendation_class is RecommendationClass.IGNORE_FILE


@pytest.mark.parametrize(
    "mutation",
    [
        {"severity": "URGENT"},
        {"priority": "NOW"},
        {"recommendation_class": "DELETE_FILE"},
        {"rationale": 7},
        {"rationale": "x" * 281},
        {"rationale": ""},
    ],
)
def test_invalid_field_values_are_rejected(mutation: dict[str, object]) -> None:
    payload = {**_valid_payload(), **mutation}
    _assert_invalid(json.dumps(payload))


def test_missing_required_field_is_rejected() -> None:
    payload = _valid_payload()
    del payload["priority"]
    _assert_invalid(json.dumps(payload))


def test_extra_field_is_rejected() -> None:
    _assert_invalid(json.dumps({**_valid_payload(), "confidence": 0.99}))


@pytest.mark.parametrize(
    "raw_output",
    [
        "not json",
        '```json\n{"severity":"HIGH"}\n```',
    ],
)
def test_non_json_and_fenced_json_are_invalid(raw_output: str) -> None:
    _assert_invalid(raw_output)


@pytest.mark.parametrize("raw_output", [None, "", " \r\n\t"])
def test_empty_output_is_classified_separately(raw_output: str | None) -> None:
    result = validate_reasoning_output(raw_output)
    assert isinstance(result, ReasoningFailure)
    assert result.code is FailureCode.EMPTY_OUTPUT


def test_model_contract_has_only_required_fields_and_forbids_extra() -> None:
    assert set(ReasoningResult.model_fields) == {
        "severity",
        "priority",
        "rationale",
        "recommendation_class",
    }
    assert ReasoningResult.model_config["extra"] == "forbid"
    assert ReasoningResult.model_config["frozen"] is True
    assert ReasoningResult.model_json_schema()["additionalProperties"] is False


def test_failure_detail_never_contains_raw_exception_text() -> None:
    failure = make_failure(FailureCode.ADK_ERROR, RuntimeError(DUMMY_SECRET))
    assert DUMMY_SECRET not in failure.detail
    assert "RuntimeError" in failure.detail


def test_instruction_has_no_adk_template_braces() -> None:
    assert "{" not in REASONING_INSTRUCTION
    assert "}" not in REASONING_INSTRUCTION
