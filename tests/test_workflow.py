"""End-to-end workflow, idempotence, and CLI secret-safety tests."""

from __future__ import annotations

import io
from collections.abc import Callable
from pathlib import Path

import pytest

from agentpulse.cli import main
from agentpulse.env_hygiene import check_env_hygiene
from agentpulse.findings import ReasonCode, Status
from agentpulse.remediation import Approval, Decision
from agentpulse.workflow import run_workflow
from tests.conftest import DUMMY_SECRET


def test_rejection_path_stays_fail_without_mutation(
    repo_factory: Callable[..., Path],
) -> None:
    root = repo_factory()
    before_bytes = (root / ".gitignore").read_bytes()

    def deny(finding, plan):
        return Approval(Decision.DENIED, plan.fingerprint)

    result = run_workflow(root, deny)
    assert result.before.status is Status.FAIL
    assert result.plan is not None
    assert result.outcome is not None and result.outcome.applied is False
    assert result.after.status is Status.FAIL
    assert result.transition == "FAIL->FAIL"
    assert result.verified is False
    assert (root / ".gitignore").read_bytes() == before_bytes


def test_approval_path_is_freshly_verified_fail_to_pass(
    repo_factory: Callable[..., Path],
) -> None:
    root = repo_factory()

    def approve(finding, plan):
        assert finding.reason_code is ReasonCode.ENV_NOT_IGNORED
        assert "+/.env" in plan.unified_diff
        return Approval(Decision.APPROVED, plan.fingerprint)

    result = run_workflow(root, approve)
    assert result.outcome is not None and result.outcome.integrity_verified
    assert result.after.reason_code is ReasonCode.ENV_IGNORED
    assert result.transition == "FAIL->PASS"
    assert result.verified is True


def test_already_safe_target_requests_no_approval(
    repo_factory: Callable[..., Path],
) -> None:
    root = repo_factory(gitignore=b"/.env\n")

    def unexpected_approval(finding, plan):
        raise AssertionError("approval must not be requested for PASS")

    result = run_workflow(root, unexpected_approval)
    assert result.before.status is Status.PASS
    assert result.plan is None
    assert result.outcome is None
    assert (root / ".gitignore").read_bytes() == b"/.env\n"


def test_second_run_is_idempotent(repo_factory: Callable[..., Path]) -> None:
    root = repo_factory()

    def approve(finding, plan):
        return Approval(Decision.APPROVED, plan.fingerprint)

    first = run_workflow(root, approve)
    first_bytes = (root / ".gitignore").read_bytes()
    second = run_workflow(root, approve)
    assert first.verified is True
    assert second.before.status is Status.PASS
    assert second.plan is None
    assert (root / ".gitignore").read_bytes() == first_bytes
    assert first_bytes.count(b"/.env") == 1


def test_secret_sentinel_absent_from_plan_workflow_and_cli_output(
    repo_factory: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = repo_factory()

    def deny(finding, plan):
        assert DUMMY_SECRET not in repr(plan)
        assert DUMMY_SECRET not in plan.unified_diff
        return Approval(Decision.DENIED, plan.fingerprint)

    result = run_workflow(root, deny)
    assert DUMMY_SECRET not in repr(result)
    monkeypatch.setattr("builtins.input", lambda prompt: "no")
    assert main(["fix", "--target", str(root)]) == 1
    captured = capsys.readouterr()
    assert DUMMY_SECRET not in captured.out
    assert DUMMY_SECRET not in captured.err


def test_non_interactive_yes_is_denied_without_mutation(
    repo_factory: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = repo_factory()
    before = (root / ".gitignore").read_bytes()
    non_interactive_stdin = io.StringIO("yes\n")
    assert non_interactive_stdin.isatty() is False
    monkeypatch.setattr("sys.stdin", non_interactive_stdin)

    assert main(["fix", "--target", str(root)]) == 1

    captured = capsys.readouterr()
    assert "Applied: false" in captured.out
    assert "Outcome: DENIED" in captured.out
    assert "Before/after: FAIL->FAIL" in captured.out
    assert "Verified: false" in captured.out
    assert non_interactive_stdin.tell() == 0
    assert (root / ".gitignore").read_bytes() == before
    assert check_env_hygiene(root).status is Status.FAIL
