"""Dry-run and approval-boundary tests for the only Phase 1 mutation."""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest

from agentpulse.env_hygiene import check_env_hygiene
from agentpulse.remediation import (
    Approval,
    Decision,
    apply_plan,
    build_plan,
)


def _plan(root: Path):
    plan = build_plan(check_env_hygiene(root))
    assert plan is not None
    return plan


def test_existing_gitignore_dry_run_is_byte_identical(
    repo_factory: Callable[..., Path],
) -> None:
    root = repo_factory(gitignore=b"env/\r\n")
    before = (root / ".gitignore").read_bytes()
    plan = _plan(root)
    assert plan.unified_diff
    assert "+/.env" in plan.unified_diff
    assert plan.before_sha256 is not None
    assert plan.after_sha256 != plan.before_sha256
    assert plan.proposed_bytes == b"env/\r\n/.env\r\n"
    assert (root / ".gitignore").read_bytes() == before


def test_missing_gitignore_dry_run_creates_nothing(
    repo_factory: Callable[..., Path],
) -> None:
    root = repo_factory(gitignore=None)
    plan = _plan(root)
    assert plan.existed_before is False
    assert plan.before_sha256 is None
    assert plan.proposed_bytes == b"/.env\n"
    assert "+/.env" in plan.unified_diff
    assert not (root / ".gitignore").exists()


def test_denied_plan_causes_zero_mutation(repo_factory: Callable[..., Path]) -> None:
    root = repo_factory()
    before = (root / ".gitignore").read_bytes()
    plan = _plan(root)
    outcome = apply_plan(plan, Approval(Decision.DENIED, plan.fingerprint))
    assert outcome.applied is False
    assert outcome.reason == "DENIED"
    assert (root / ".gitignore").read_bytes() == before


def test_wrong_fingerprint_is_refused(repo_factory: Callable[..., Path]) -> None:
    root = repo_factory()
    before = (root / ".gitignore").read_bytes()
    outcome = apply_plan(_plan(root), Approval(Decision.APPROVED, "wrong"))
    assert outcome.applied is False
    assert (root / ".gitignore").read_bytes() == before


def test_missing_approval_is_refused(repo_factory: Callable[..., Path]) -> None:
    root = repo_factory()
    before = (root / ".gitignore").read_bytes()
    outcome = apply_plan(_plan(root), None)
    assert outcome.applied is False
    assert outcome.reason == "APPROVAL_REQUIRED"
    assert (root / ".gitignore").read_bytes() == before


def test_drift_after_preview_is_refused(repo_factory: Callable[..., Path]) -> None:
    root = repo_factory()
    plan = _plan(root)
    external = b"external-change\n"
    (root / ".gitignore").write_bytes(external)
    outcome = apply_plan(plan, Approval(Decision.APPROVED, plan.fingerprint))
    assert outcome.applied is False
    assert outcome.reason == "TARGET_DRIFT"
    assert (root / ".gitignore").read_bytes() == external


@pytest.mark.parametrize(
    ("invalid_plan", "expected_reason"),
    [
        (lambda plan: replace(plan, action="delete_file"), "INVALID_ACTION"),
        (
            lambda plan: replace(plan, file="../escape/.gitignore"),
            "INVALID_DESTINATION",
        ),
        (lambda plan: replace(plan, file="another-file"), "INVALID_DESTINATION"),
    ],
)
def test_invalid_action_or_path_is_refused(
    repo_factory: Callable[..., Path],
    invalid_plan: Callable,
    expected_reason: str,
) -> None:
    root = repo_factory()
    before = (root / ".gitignore").read_bytes()
    plan = invalid_plan(_plan(root))
    outcome = apply_plan(plan, Approval(Decision.APPROVED, plan.fingerprint))
    assert outcome.applied is False
    assert outcome.reason == expected_reason
    assert (root / ".gitignore").read_bytes() == before
    assert not (root.parent / "escape" / ".gitignore").exists()


def test_approved_plan_writes_exact_previewed_bytes(
    repo_factory: Callable[..., Path],
) -> None:
    root = repo_factory(gitignore=b"env/")
    plan = _plan(root)
    outcome = apply_plan(plan, Approval(Decision.APPROVED, plan.fingerprint))
    assert outcome.applied is True
    assert outcome.integrity_verified is True
    assert (root / ".gitignore").read_bytes() == plan.proposed_bytes
    assert (root / ".gitignore").read_bytes() == b"env/\n/.env\n"


def test_tracked_env_has_no_plan(repo_factory: Callable[..., Path]) -> None:
    root = repo_factory(gitignore=b"/.env\n")
    subprocess.run(
        ["git", "add", "-f", "--", ".env"],
        cwd=root,
        check=True,
        capture_output=True,
        timeout=5,
    )
    assert build_plan(check_env_hygiene(root)) is None


def test_existing_gitignore_symlink_is_refused(
    repo_factory: Callable[..., Path],
) -> None:
    root = repo_factory()
    destination = root / ".gitignore"
    plan = _plan(root)
    destination.unlink()
    link_target = root / "external-ignore"
    link_target.write_bytes(b"external\n")
    try:
        destination.symlink_to(link_target)
    except OSError as error:
        pytest.skip(f"Windows symlink creation unavailable: {error}")

    outcome = apply_plan(plan, Approval(Decision.APPROVED, plan.fingerprint))
    assert outcome.applied is False
    assert outcome.reason == "SYMLINK_REFUSED"
    assert link_target.read_bytes() == b"external\n"
