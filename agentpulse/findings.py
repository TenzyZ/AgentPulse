"""Typed, secret-free findings for the Phase 1 detector."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class Status(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    ERROR = "ERROR"


class ReasonCode(str, Enum):
    ENV_ABSENT = "ENV_ABSENT"
    ENV_IGNORED = "ENV_IGNORED"
    ENV_NOT_IGNORED = "ENV_NOT_IGNORED"
    ENV_TRACKED_IN_GIT = "ENV_TRACKED_IN_GIT"
    ENV_NOT_REGULAR_FILE = "ENV_NOT_REGULAR_FILE"
    TARGET_NOT_A_GIT_REPO = "TARGET_NOT_A_GIT_REPO"
    GIT_UNAVAILABLE = "GIT_UNAVAILABLE"
    GIT_ERROR = "GIT_ERROR"


@dataclass(frozen=True)
class Evidence:
    relative_path: str
    exists: bool
    is_file: bool
    git_tracked: bool | None
    git_ignored: bool | None
    gitignore_present: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "exists": self.exists,
            "is_file": self.is_file,
            "git_tracked": self.git_tracked,
            "git_ignored": self.git_ignored,
            "gitignore_present": self.gitignore_present,
        }


@dataclass(frozen=True)
class Finding:
    status: Status
    reason_code: ReasonCode
    remediable: bool
    evidence: Evidence
    message: str
    target_root: Path
    check_id: str = "secret_hygiene.env_ignored"
    schema_version: int = 1

    def to_sanitized_dict(self) -> dict[str, Any]:
        """Return the cloud-safe seam without machine-specific target paths."""
        return {
            "check_id": self.check_id,
            "schema_version": self.schema_version,
            "status": self.status.value,
            "reason_code": self.reason_code.value,
            "remediable": self.remediable,
            "evidence": self.evidence.to_dict(),
            "message": self.message,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_sanitized_dict(), sort_keys=True)
