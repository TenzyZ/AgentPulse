"""Root ``.env`` secret-hygiene detector."""

from __future__ import annotations

from pathlib import Path

from agentpulse.findings import Evidence, Finding, ReasonCode, Status
from agentpulse.git_ignore import is_env_ignored, is_env_tracked, is_git_work_tree


_MESSAGES = {
    ReasonCode.ENV_ABSENT: "Root .env is absent.",
    ReasonCode.ENV_IGNORED: "Root .env is safely ignored by Git.",
    ReasonCode.ENV_NOT_IGNORED: "Root .env exists but is not ignored by Git.",
    ReasonCode.ENV_TRACKED_IN_GIT: "Root .env is already tracked by Git.",
    ReasonCode.ENV_NOT_REGULAR_FILE: "Root .env exists but is not a regular file.",
    ReasonCode.TARGET_NOT_A_GIT_REPO: "Target is not a usable Git work tree.",
    ReasonCode.GIT_UNAVAILABLE: "Git is unavailable.",
    ReasonCode.GIT_ERROR: "Git could not determine root .env hygiene.",
}


def _finding(
    target_root: Path,
    status: Status,
    reason: ReasonCode,
    *,
    exists: bool,
    is_file: bool,
    tracked: bool | None = None,
    ignored: bool | None = None,
    remediable: bool = False,
) -> Finding:
    return Finding(
        status=status,
        reason_code=reason,
        remediable=remediable,
        evidence=Evidence(
            relative_path=".env",
            exists=exists,
            is_file=is_file,
            git_tracked=tracked,
            git_ignored=ignored,
            gitignore_present=(target_root / ".gitignore").is_file(),
        ),
        message=_MESSAGES[reason],
        target_root=target_root,
    )


def check_env_hygiene(target_root: str | Path) -> Finding:
    """Classify root ``.env`` state without opening or reading that file."""
    root = Path(target_root).resolve()
    repository = is_git_work_tree(root)
    if repository.is_error:
        return _finding(
            root,
            Status.ERROR,
            repository.error_reason or ReasonCode.GIT_ERROR,
            exists=False,
            is_file=False,
        )

    env_path = root / ".env"
    if not env_path.exists():
        return _finding(
            root,
            Status.PASS,
            ReasonCode.ENV_ABSENT,
            exists=False,
            is_file=False,
        )
    if not env_path.is_file():
        return _finding(
            root,
            Status.ERROR,
            ReasonCode.ENV_NOT_REGULAR_FILE,
            exists=True,
            is_file=False,
        )

    tracked = is_env_tracked(root)
    if tracked.is_error:
        return _finding(
            root,
            Status.ERROR,
            tracked.error_reason or ReasonCode.GIT_ERROR,
            exists=True,
            is_file=True,
        )
    if tracked.value:
        return _finding(
            root,
            Status.FAIL,
            ReasonCode.ENV_TRACKED_IN_GIT,
            exists=True,
            is_file=True,
            tracked=True,
        )

    ignored = is_env_ignored(root)
    if ignored.is_error:
        return _finding(
            root,
            Status.ERROR,
            ignored.error_reason or ReasonCode.GIT_ERROR,
            exists=True,
            is_file=True,
            tracked=False,
        )
    if ignored.value:
        return _finding(
            root,
            Status.PASS,
            ReasonCode.ENV_IGNORED,
            exists=True,
            is_file=True,
            tracked=False,
            ignored=True,
        )
    return _finding(
        root,
        Status.FAIL,
        ReasonCode.ENV_NOT_IGNORED,
        exists=True,
        is_file=True,
        tracked=False,
        ignored=False,
        remediable=True,
    )
