"""Deterministic Git subprocess oracles for root ``.env`` hygiene."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from agentpulse.findings import ReasonCode

GIT_TIMEOUT_SECONDS = 5.0


@dataclass(frozen=True)
class GitBooleanResult:
    value: bool | None
    error_reason: ReasonCode | None = None

    @property
    def is_error(self) -> bool:
        return self.error_reason is not None


def _run_git(target_root: Path, arguments: list[str]) -> subprocess.CompletedProcess[str] | ReasonCode:
    try:
        return subprocess.run(
            ["git", *arguments],
            cwd=target_root,
            capture_output=True,
            text=True,
            check=False,
            timeout=GIT_TIMEOUT_SECONDS,
        )
    except OSError:
        return ReasonCode.GIT_UNAVAILABLE
    except subprocess.TimeoutExpired:
        return ReasonCode.GIT_ERROR


def is_git_work_tree(target_root: Path) -> GitBooleanResult:
    result = _run_git(target_root, ["rev-parse", "--is-inside-work-tree"])
    if isinstance(result, ReasonCode):
        return GitBooleanResult(None, result)
    if result.returncode != 0:
        return GitBooleanResult(None, ReasonCode.TARGET_NOT_A_GIT_REPO)
    if result.stdout.strip() != "true":
        return GitBooleanResult(None, ReasonCode.GIT_ERROR)
    return GitBooleanResult(True)


def is_env_tracked(target_root: Path) -> GitBooleanResult:
    result = _run_git(
        target_root,
        ["ls-files", "--error-unmatch", "--", ".env"],
    )
    if isinstance(result, ReasonCode):
        return GitBooleanResult(None, result)
    if result.returncode == 0:
        return GitBooleanResult(True)
    if result.returncode == 1:
        return GitBooleanResult(False)
    return GitBooleanResult(None, ReasonCode.GIT_ERROR)


def is_env_ignored(target_root: Path) -> GitBooleanResult:
    result = _run_git(target_root, ["check-ignore", "-q", "--", ".env"])
    if isinstance(result, ReasonCode):
        return GitBooleanResult(None, result)
    if result.returncode == 0:
        return GitBooleanResult(True)
    if result.returncode == 1:
        return GitBooleanResult(False)
    return GitBooleanResult(None, ReasonCode.GIT_ERROR)
