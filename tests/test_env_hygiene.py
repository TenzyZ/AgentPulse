"""Detector behavior and secret-safety tests."""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

from agentpulse.env_hygiene import check_env_hygiene
from agentpulse.findings import ReasonCode, Status
from tests.conftest import DUMMY_SECRET


def test_env_absent_passes(repo_factory: Callable[..., Path]) -> None:
    finding = check_env_hygiene(repo_factory(env_exists=False))
    assert (finding.status, finding.reason_code) == (
        Status.PASS,
        ReasonCode.ENV_ABSENT,
    )


def test_unsafe_env_fails_and_is_remediable(repo_factory: Callable[..., Path]) -> None:
    finding = check_env_hygiene(repo_factory())
    assert (finding.status, finding.reason_code, finding.remediable) == (
        Status.FAIL,
        ReasonCode.ENV_NOT_IGNORED,
        True,
    )


def test_ignored_env_passes(repo_factory: Callable[..., Path]) -> None:
    finding = check_env_hygiene(repo_factory(gitignore=b"/.env\n"))
    assert (finding.status, finding.reason_code) == (
        Status.PASS,
        ReasonCode.ENV_IGNORED,
    )


def test_tracked_env_fails_without_remediation(repo_factory: Callable[..., Path]) -> None:
    root = repo_factory(gitignore=b"/.env\n")
    subprocess.run(
        ["git", "add", "-f", "--", ".env"],
        cwd=root,
        check=True,
        capture_output=True,
        timeout=5,
    )
    finding = check_env_hygiene(root)
    assert (finding.status, finding.reason_code, finding.remediable) == (
        Status.FAIL,
        ReasonCode.ENV_TRACKED_IN_GIT,
        False,
    )


def test_non_git_target_is_error(repo_factory: Callable[..., Path]) -> None:
    finding = check_env_hygiene(repo_factory(initialize_git=False))
    assert (finding.status, finding.reason_code) == (
        Status.ERROR,
        ReasonCode.TARGET_NOT_A_GIT_REPO,
    )


def test_non_regular_env_is_error(repo_factory: Callable[..., Path]) -> None:
    root = repo_factory(env_exists=False)
    (root / ".env").mkdir()
    finding = check_env_hygiene(root)
    assert (finding.status, finding.reason_code) == (
        Status.ERROR,
        ReasonCode.ENV_NOT_REGULAR_FILE,
    )


def test_git_unavailable_fails_closed(
    repo_factory: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    root = repo_factory()
    before = (root / ".gitignore").read_bytes()

    def unavailable(*args: object, **kwargs: object) -> None:
        raise OSError("simulated unavailable executable")

    monkeypatch.setattr("agentpulse.git_ignore.subprocess.run", unavailable)
    finding = check_env_hygiene(root)
    assert (finding.status, finding.reason_code) == (
        Status.ERROR,
        ReasonCode.GIT_UNAVAILABLE,
    )
    assert (root / ".gitignore").read_bytes() == before


def test_git_timeout_fails_closed(
    repo_factory: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    root = repo_factory()

    def timeout(*args: object, **kwargs: object) -> None:
        raise subprocess.TimeoutExpired(cmd="git", timeout=5)

    monkeypatch.setattr("agentpulse.git_ignore.subprocess.run", timeout)
    finding = check_env_hygiene(root)
    assert (finding.status, finding.reason_code) == (
        Status.ERROR,
        ReasonCode.GIT_ERROR,
    )


def test_unexpected_tracked_oracle_code_fails_closed(
    repo_factory: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    root = repo_factory()

    def unexpected(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if "rev-parse" in command:
            return subprocess.CompletedProcess(command, 0, "true\n", "")
        return subprocess.CompletedProcess(command, 7, "", "")

    monkeypatch.setattr("agentpulse.git_ignore.subprocess.run", unexpected)
    finding = check_env_hygiene(root)
    assert (finding.status, finding.reason_code) == (
        Status.ERROR,
        ReasonCode.GIT_ERROR,
    )


def test_sanitized_finding_omits_secret_and_absolute_path(
    repo_factory: Callable[..., Path],
) -> None:
    root = repo_factory()
    finding = check_env_hygiene(root)
    serialized = finding.to_json()
    assert DUMMY_SECRET not in serialized
    assert str(root) not in serialized
    assert ".env" in serialized


def test_detector_does_not_use_python_file_reads_for_env(
    repo_factory: Callable[..., Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    root = repo_factory()
    original_open = Path.open
    original_read_bytes = Path.read_bytes
    original_read_text = Path.read_text

    def guarded_open(path: Path, *args: object, **kwargs: object):
        if path.name == ".env":
            raise AssertionError("detector attempted to open .env")
        return original_open(path, *args, **kwargs)

    def guarded_read_bytes(path: Path) -> bytes:
        if path.name == ".env":
            raise AssertionError("detector attempted to read .env")
        return original_read_bytes(path)

    def guarded_read_text(path: Path, *args: object, **kwargs: object) -> str:
        if path.name == ".env":
            raise AssertionError("detector attempted to read .env")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)
    monkeypatch.setattr(Path, "read_text", guarded_read_text)
    assert check_env_hygiene(root).status is Status.FAIL
