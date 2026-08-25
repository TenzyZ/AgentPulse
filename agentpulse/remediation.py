"""Fingerprint-bound, exact-byte remediation for target-root ``.gitignore``."""

from __future__ import annotations

import base64
import difflib
import hashlib
import json
import os
import stat
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from agentpulse.findings import Finding, ReasonCode, Status


class Action(str, Enum):
    APPEND_ROOT_ENV_IGNORE = "append_root_env_ignore"


class Decision(str, Enum):
    APPROVED = "APPROVED"
    DENIED = "DENIED"


@dataclass(frozen=True)
class Approval:
    decision: Decision
    plan_fingerprint: str


@dataclass(frozen=True)
class RemediationPlan:
    check_id: str
    target_root: Path
    action: Action
    file: str
    line: str
    existed_before: bool
    before_sha256: str | None
    after_sha256: str
    unified_diff: str
    proposed_bytes: bytes
    fingerprint: str


@dataclass(frozen=True)
class RemediationOutcome:
    applied: bool
    reason: str
    integrity_verified: bool = False


class PlanError(ValueError):
    """Raised when an exact, safely reproducible plan cannot be built."""


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _new_line_bytes(before: bytes) -> bytes:
    without_crlf = before.replace(b"\r\n", b"")
    has_crlf = b"\r\n" in before
    if b"\r" in without_crlf or (has_crlf and b"\n" in without_crlf):
        raise PlanError("mixed or unsupported .gitignore newline bytes")
    return b"\r\n" if has_crlf else b"\n"


def _proposed_content(before: bytes) -> bytes:
    try:
        before.decode("utf-8")
    except UnicodeDecodeError as error:
        raise PlanError(".gitignore is not valid UTF-8") from error
    if b"\x00" in before:
        raise PlanError(".gitignore contains unsupported NUL bytes")

    newline = _new_line_bytes(before)
    prefix = b"" if not before or before.endswith(newline) else newline
    return before + prefix + b"/.env" + newline


def _unified_diff(before: bytes, after: bytes, existed_before: bool) -> str:
    before_lines = before.decode("utf-8").splitlines()
    after_lines = after.decode("utf-8").splitlines()
    from_file = ".gitignore" if existed_before else "/dev/null"
    lines = difflib.unified_diff(
        before_lines,
        after_lines,
        fromfile=from_file,
        tofile=".gitignore",
        lineterm="",
    )
    return "\n".join(lines) + "\n"


def _authorization_payload(plan: RemediationPlan) -> dict[str, object]:
    action = plan.action.value if isinstance(plan.action, Action) else str(plan.action)
    return {
        "check_id": plan.check_id,
        "target_root": str(plan.target_root),
        "action": action,
        "file": plan.file,
        "line": plan.line,
        "existed_before": plan.existed_before,
        "before_sha256": plan.before_sha256,
        "after_sha256": plan.after_sha256,
        "unified_diff": plan.unified_diff,
        "proposed_bytes_base64": base64.b64encode(plan.proposed_bytes).decode("ascii"),
    }


def _fingerprint(plan: RemediationPlan) -> str:
    canonical = json.dumps(
        _authorization_payload(plan),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return _sha256(canonical)


def build_plan(finding: Finding) -> RemediationPlan | None:
    """Build an exact dry-run plan only for an unignored, untracked root .env."""
    if not (
        finding.status is Status.FAIL
        and finding.reason_code is ReasonCode.ENV_NOT_IGNORED
        and finding.remediable
    ):
        return None

    root = finding.target_root.resolve()
    destination = root / ".gitignore"
    if destination.is_symlink():
        raise PlanError("existing .gitignore symlink is not eligible")
    existed_before = destination.exists()
    if existed_before and not destination.is_file():
        raise PlanError("existing .gitignore is not a regular file")
    before = destination.read_bytes() if existed_before else b""
    after = _proposed_content(before)
    provisional = RemediationPlan(
        check_id=finding.check_id,
        target_root=root,
        action=Action.APPEND_ROOT_ENV_IGNORE,
        file=".gitignore",
        line="/.env",
        existed_before=existed_before,
        before_sha256=_sha256(before) if existed_before else None,
        after_sha256=_sha256(after),
        unified_diff=_unified_diff(before, after, existed_before),
        proposed_bytes=after,
        fingerprint="",
    )
    return RemediationPlan(
        **{
            **provisional.__dict__,
            "fingerprint": _fingerprint(provisional),
        }
    )


def _refusal(reason: str) -> RemediationOutcome:
    return RemediationOutcome(applied=False, reason=reason)


def apply_plan(
    plan: RemediationPlan,
    approval: Approval | None,
) -> RemediationOutcome:
    """Apply approved bytes only after every authorization gate succeeds."""
    if approval is None:
        return _refusal("APPROVAL_REQUIRED")
    if approval.decision is not Decision.APPROVED:
        return _refusal("DENIED")
    if approval.plan_fingerprint != plan.fingerprint:
        return _refusal("APPROVAL_FINGERPRINT_MISMATCH")
    if plan.action is not Action.APPEND_ROOT_ENV_IGNORE:
        return _refusal("INVALID_ACTION")
    if plan.file != ".gitignore" or plan.line != "/.env":
        return _refusal("INVALID_DESTINATION")
    if _fingerprint(plan) != plan.fingerprint:
        return _refusal("PLAN_FINGERPRINT_INVALID")

    root = plan.target_root.resolve()
    if root != plan.target_root:
        return _refusal("TARGET_ROOT_CHANGED")
    destination = root / plan.file
    if destination.parent.resolve() != root or destination != root / ".gitignore":
        return _refusal("DESTINATION_ESCAPE")
    if destination.is_symlink():
        return _refusal("SYMLINK_REFUSED")

    exists_now = destination.exists()
    if exists_now != plan.existed_before:
        return _refusal("TARGET_DRIFT")
    if exists_now and not destination.is_file():
        return _refusal("DESTINATION_NOT_REGULAR")

    try:
        current = destination.read_bytes() if exists_now else b""
    except OSError:
        return _refusal("DESTINATION_READ_ERROR")
    current_hash = _sha256(current) if exists_now else None
    if current_hash != plan.before_sha256:
        return _refusal("TARGET_DRIFT")
    if _sha256(plan.proposed_bytes) != plan.after_sha256:
        return _refusal("PROPOSED_BYTES_INVALID")

    try:
        if exists_now:
            path_stat = destination.lstat()
            if not stat.S_ISREG(path_stat.st_mode):
                return _refusal("DESTINATION_NOT_REGULAR")
            with destination.open("r+b") as stream:
                open_stat = os.fstat(stream.fileno())
                if (path_stat.st_dev, path_stat.st_ino) != (
                    open_stat.st_dev,
                    open_stat.st_ino,
                ):
                    return _refusal("TARGET_DRIFT")
                if _sha256(stream.read()) != plan.before_sha256:
                    return _refusal("TARGET_DRIFT")
                stream.seek(0)
                stream.write(plan.proposed_bytes)
                stream.truncate()
                stream.flush()
                os.fsync(stream.fileno())
        else:
            with destination.open("xb") as stream:
                stream.write(plan.proposed_bytes)
                stream.flush()
                os.fsync(stream.fileno())
    except OSError:
        return _refusal("WRITE_ERROR")

    try:
        integrity_verified = _sha256(destination.read_bytes()) == plan.after_sha256
    except OSError:
        integrity_verified = False
    if not integrity_verified:
        return RemediationOutcome(
            applied=True,
            reason="POST_WRITE_INTEGRITY_FAILED",
            integrity_verified=False,
        )
    return RemediationOutcome(
        applied=True,
        reason="APPLIED",
        integrity_verified=True,
    )
