"""Offline bridge, CLI integration, and authority-containment tests."""

from __future__ import annotations

import io
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from google.genai import _transformers, errors

import agentpulse.cli as cli_module
import agentpulse.gemini_bridge as bridge
from agentpulse.cli import main
from agentpulse.env_hygiene import check_env_hygiene
from agentpulse.findings import ReasonCode, Status
from agentpulse.reasoning import (
    MODEL_NAME,
    PROMPT_VERSION,
    FailureCode,
    Priority,
    ReasoningOutcome,
    ReasoningResult,
    RecommendationClass,
    Severity,
    make_failure,
    reasoning_wire_schema,
)
from agentpulse.remediation import Approval, Decision
from tests.conftest import DUMMY_SECRET


class TtyInput(io.StringIO):
    def isatty(self) -> bool:
        return True


def _result(
    severity: Severity = Severity.HIGH,
    priority: Priority = Priority.P1,
    recommendation: RecommendationClass = RecommendationClass.IGNORE_FILE,
) -> ReasoningOutcome:
    return ReasoningOutcome(
        model=MODEL_NAME,
        prompt_version=PROMPT_VERSION,
        payload_sha256="a" * 64,
        result=ReasoningResult(
            severity=severity,
            priority=priority,
            rationale="Advisory classification from sanitized evidence only.",
            recommendation_class=recommendation,
        ),
    )


def _failure(code: FailureCode) -> ReasoningOutcome:
    return ReasoningOutcome(
        model=MODEL_NAME,
        prompt_version=PROMPT_VERSION,
        payload_sha256="b" * 64,
        failure=make_failure(code),
    )


def test_valid_check_reason_prints_advice_without_changing_truth_or_bytes(
    repo_factory: Callable[..., Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = repo_factory()
    env_before = (root / ".env").read_bytes()
    ignore_before = (root / ".gitignore").read_bytes()

    assert main(
        ["check", "--target", str(root), "--reason"],
        reasoner=lambda finding: _result(),
    ) == 1

    captured = capsys.readouterr()
    assert "Reasoning:" in captured.out
    assert '"severity": "HIGH"' in captured.out
    assert DUMMY_SECRET not in captured.out
    assert DUMMY_SECRET not in captured.err
    assert (root / ".env").read_bytes() == env_before
    assert (root / ".gitignore").read_bytes() == ignore_before
    assert check_env_hygiene(root).status is Status.FAIL


def test_check_reason_skips_reasoner_for_pass(
    repo_factory: Callable[..., Path],
) -> None:
    root = repo_factory(gitignore=b"/.env\n")

    def forbidden_reasoner(finding):
        raise AssertionError("reasoning must not run for PASS")

    assert main(
        ["check", "--target", str(root), "--reason"],
        reasoner=forbidden_reasoner,
    ) == 0


@pytest.mark.parametrize("failure_code", list(FailureCode))
def test_fix_reasoning_failure_blocks_authorization_and_mutation(
    failure_code: FailureCode,
    repo_factory: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = repo_factory()
    before = (root / ".gitignore").read_bytes()
    fake_stdin = TtyInput("yes\n")
    monkeypatch.setattr("sys.stdin", fake_stdin)

    def forbidden_input(prompt: str) -> str:
        raise AssertionError("input must not be reached after reasoning failure")

    monkeypatch.setattr("builtins.input", forbidden_input)
    assert main(
        ["fix", "--target", str(root), "--reason"],
        reasoner=lambda finding: _failure(failure_code),
    ) == 1

    captured = capsys.readouterr()
    assert f'"code": "{failure_code.value}"' in captured.out
    assert "Authorization was not requested because reasoning failed." in captured.out
    assert "Apply this exact remediation?" not in captured.out
    assert "Applied: false" in captured.out
    assert "Outcome: APPROVAL_REQUIRED" in captured.out
    assert "Before/after: FAIL->FAIL" in captured.out
    assert "Verified: false" in captured.out
    assert DUMMY_SECRET not in captured.out
    assert DUMMY_SECRET not in captured.err
    assert fake_stdin.tell() == 0
    assert (root / ".gitignore").read_bytes() == before
    assert check_env_hygiene(root).status is Status.FAIL


def test_not_configured_does_not_construct_adk(
    repo_factory: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = repo_factory()
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(
        bridge,
        "_run_adk",
        lambda payload: (_ for _ in ()).throw(AssertionError("ADK must not run")),
    )
    outcome = bridge.reason_finding(check_env_hygiene(root))
    assert outcome.failure is not None
    assert outcome.failure.code is FailureCode.NOT_CONFIGURED


def test_unexpected_adapter_error_is_sanitized_and_does_not_escape(
    repo_factory: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = repo_factory()
    monkeypatch.setenv("GOOGLE_API_KEY", "configured-for-offline-test")
    monkeypatch.setattr(
        bridge,
        "_run_adk",
        lambda payload: (_ for _ in ()).throw(RuntimeError(DUMMY_SECRET)),
    )
    outcome = bridge.reason_finding(check_env_hygiene(root))
    assert outcome.failure is not None
    assert outcome.failure.code is FailureCode.ADK_ERROR
    assert DUMMY_SECRET not in outcome.failure.detail
    captured = capsys.readouterr()
    assert DUMMY_SECRET not in captured.out
    assert DUMMY_SECRET not in captured.err


def test_provider_400_from_runner_iteration_maps_to_adk_error(
    repo_factory: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = repo_factory()
    monkeypatch.setenv("GOOGLE_API_KEY", "configured-for-offline-test")

    class FailingRunner:
        def __init__(self, **kwargs) -> None:
            pass

        async def run_async(self, **kwargs):
            raise errors.ClientError(400, {"message": DUMMY_SECRET}, None)
            yield

    monkeypatch.setattr("google.adk.runners.Runner", FailingRunner)
    outcome = bridge.reason_finding(check_env_hygiene(root))
    assert outcome.failure is not None
    assert outcome.failure.code is FailureCode.ADK_ERROR
    assert outcome.failure.code is not FailureCode.EMPTY_OUTPUT
    assert DUMMY_SECRET not in outcome.failure.detail


def test_successful_runner_with_no_final_content_is_empty_output(
    repo_factory: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = repo_factory()
    monkeypatch.setenv("GOOGLE_API_KEY", "configured-for-offline-test")

    class EmptyRunner:
        def __init__(self, **kwargs) -> None:
            pass

        async def run_async(self, **kwargs):
            if False:
                yield None

    monkeypatch.setattr("google.adk.runners.Runner", EmptyRunner)
    outcome = bridge.reason_finding(check_env_hygiene(root))
    assert outcome.failure is not None
    assert outcome.failure.code is FailureCode.EMPTY_OUTPUT


@pytest.mark.parametrize(
    ("exception", "expected"),
    [
        (errors.ClientError(401, {"message": DUMMY_SECRET}, None), FailureCode.AUTH_FAILED),
        (errors.ClientError(403, {"message": DUMMY_SECRET}, None), FailureCode.AUTH_FAILED),
        (errors.ClientError(400, {"message": DUMMY_SECRET}, None), FailureCode.ADK_ERROR),
        (errors.ServerError(503, {"message": DUMMY_SECRET}, None), FailureCode.UNAVAILABLE),
        (
            errors.ServerError(
                504,
                {"status": "DEADLINE_EXCEEDED", "message": DUMMY_SECRET},
                None,
            ),
            FailureCode.TIMEOUT,
        ),
        (httpx.ReadTimeout(DUMMY_SECRET), FailureCode.TIMEOUT),
        (httpx.ConnectError(DUMMY_SECRET), FailureCode.NETWORK_ERROR),
        (RuntimeError(DUMMY_SECRET), FailureCode.ADK_ERROR),
    ],
)
def test_installed_exception_types_map_without_exposing_text(
    exception: Exception,
    expected: FailureCode,
) -> None:
    assert bridge._classify_exception(exception) is expected
    assert DUMMY_SECRET not in make_failure(expected, exception).detail


def test_adk_configuration_and_final_response_extraction_are_offline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert MODEL_NAME == "gemini-3.7-flash"
    assert bridge.REQUEST_TIMEOUT_MS == 60_000
    valid_json = json.dumps(
        {
            "severity": "MEDIUM",
            "priority": "P2",
            "rationale": "Validated advisory.",
            "recommendation_class": "MANUAL_REVIEW",
        }
    )
    captured: dict[str, object] = {}

    class FakeRunner:
        def __init__(self, **kwargs) -> None:
            captured["runner"] = kwargs

        async def run_async(self, **kwargs):
            captured["run"] = kwargs
            yield SimpleNamespace(
                is_final_response=lambda: True,
                content=SimpleNamespace(
                    parts=[
                        SimpleNamespace(text="private thought", thought=True),
                        SimpleNamespace(text=valid_json, thought=False),
                    ]
                ),
            )

    monkeypatch.setattr("google.adk.runners.Runner", FakeRunner)
    payload = '{"status":"FAIL"}'
    assert bridge._run_adk(payload) == valid_json

    runner_config = captured["runner"]
    assert isinstance(runner_config, dict)
    assert runner_config["auto_create_session"] is True
    agent = runner_config["agent"]
    assert agent.model == MODEL_NAME
    assert agent.output_schema == reasoning_wire_schema()
    serialized_schema = json.dumps(agent.output_schema, sort_keys=True)
    assert "additionalProperties" not in serialized_schema
    assert "additional_properties" not in serialized_schema
    legacy_schema = _transformers.t_schema(
        None,
        json.loads(serialized_schema),
    )
    serialized_legacy_schema = json.dumps(
        legacy_schema.model_dump(exclude_none=True),
        sort_keys=True,
    )
    assert "additionalProperties" not in serialized_legacy_schema
    assert "additional_properties" not in serialized_legacy_schema
    assert agent.tools == []
    assert agent.sub_agents == []
    assert agent.output_key is None
    assert agent.generate_content_config.thinking_config.thinking_level.value == "LOW"
    http_options = agent.generate_content_config.http_options
    assert http_options.timeout == bridge.REQUEST_TIMEOUT_MS
    assert http_options.retry_options.attempts == 2
    assert http_options.retry_options.http_status_codes == [503]
    run_config = captured["run"]
    assert isinstance(run_config, dict)
    assert payload in run_config["new_message"].parts[0].text


def test_fix_without_reason_does_not_import_or_call_gemini(
    repo_factory: Callable[..., Path],
) -> None:
    root = repo_factory()
    script = (
        "import json, sys; "
        "from agentpulse.cli import main; "
        "before=set(sys.modules); "
        "main(['fix','--target',sys.argv[1]]); "
        "after=set(sys.modules); "
        "print('SDK_DELTA=' + json.dumps(sorted(name for name in after-before "
        "if name == 'google.adk' or name.startswith('google.adk.') "
        "or name == 'google.genai' or name.startswith('google.genai.'))))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script, str(root)],
        cwd=Path.cwd(),
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    marker = next(
        line for line in completed.stdout.splitlines() if line.startswith("SDK_DELTA=")
    )
    assert json.loads(marker.removeprefix("SDK_DELTA=")) == []
    assert completed.returncode == 0
    assert (root / ".gitignore").read_bytes() == b".envrc\nenv/\n.env/\n"


def test_fix_without_reason_never_calls_injected_reasoner(
    repo_factory: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = repo_factory()
    monkeypatch.setattr("sys.stdin", io.StringIO("yes\n"))

    def forbidden_reasoner(finding):
        raise AssertionError("reasoner must stay unresolved without --reason")

    assert main(
        ["fix", "--target", str(root)],
        reasoner=forbidden_reasoner,
    ) == 1
    assert (root / ".gitignore").read_bytes() == b".envrc\nenv/\n.env/\n"


def test_opposite_advice_cannot_change_remediation_authority(
    repo_factory: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = repo_factory()
    original = (root / ".gitignore").read_bytes()
    results = []
    approvals = []
    real_run_workflow = cli_module.run_workflow

    def recording_workflow(target, approval_provider):
        def recording_approval(finding, plan):
            approval = approval_provider(finding, plan)
            approvals.append(approval)
            return approval

        result = real_run_workflow(target, recording_approval)
        results.append(result)
        return result

    monkeypatch.setattr(cli_module, "run_workflow", recording_workflow)
    monkeypatch.setattr("sys.stdin", TtyInput())
    monkeypatch.setattr("builtins.input", lambda prompt: "yes")
    opposite = [
        _result(Severity.CRITICAL, Priority.P0, RecommendationClass.IGNORE_FILE),
        _result(Severity.LOW, Priority.P3, RecommendationClass.NO_ACTION),
    ]
    for outcome in opposite:
        assert main(
            ["fix", "--target", str(root), "--reason"],
            reasoner=lambda finding, value=outcome: value,
        ) == 0
        (root / ".gitignore").write_bytes(original)

    first, second = results
    assert first.plan is not None and second.plan is not None
    assert first.plan.target_root == second.plan.target_root == root
    assert first.plan.proposed_bytes == second.plan.proposed_bytes
    assert first.plan.unified_diff == second.plan.unified_diff
    assert first.plan.fingerprint == second.plan.fingerprint
    assert approvals[0] == Approval(Decision.APPROVED, first.plan.fingerprint)
    assert approvals[1] == Approval(Decision.APPROVED, second.plan.fingerprint)
    assert first.verified is second.verified is True


def test_model_recommendation_cannot_create_nonremediable_plan(
    repo_factory: Callable[..., Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = repo_factory(gitignore=b"/.env\n")
    subprocess.run(
        ["git", "add", "-f", "--", ".env"],
        cwd=root,
        check=True,
        capture_output=True,
        timeout=5,
    )

    def forbidden_reasoner(finding):
        raise AssertionError("reasoning cannot create a remediation plan")

    assert main(
        ["fix", "--target", str(root), "--reason"],
        reasoner=forbidden_reasoner,
    ) == 1
    captured = capsys.readouterr()
    assert "No remediation plan." in captured.out
    assert check_env_hygiene(root).reason_code is ReasonCode.ENV_TRACKED_IN_GIT
