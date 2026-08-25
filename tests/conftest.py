"""Isolated disposable Git fixtures for Phase 1 tests."""

from __future__ import annotations

import os
import subprocess
import tempfile
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

DUMMY_SECRET = "SENTINEL_NOT_A_REAL_SECRET_9f3a"


@pytest.fixture(autouse=True)
def isolated_git_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)
    monkeypatch.setenv("GIT_ATTR_NOSYSTEM", "1")


@pytest.fixture
def repo_factory() -> Iterator[Callable[..., Path]]:
    workspace = Path.cwd().resolve()
    temporary_directory = tempfile.TemporaryDirectory(prefix="agentpulse-tests-")
    temporary_root = Path(temporary_directory.name)

    def create(
        *,
        env_exists: bool = True,
        gitignore: bytes | None = b".envrc\nenv/\n.env/\n",
        initialize_git: bool = True,
    ) -> Path:
        root = (
            temporary_root / f"repo-{len(list(temporary_root.iterdir()))}"
        ).resolve()
        assert not root.is_relative_to(workspace)
        root.mkdir()
        if initialize_git:
            subprocess.run(
                ["git", "init", "-q"],
                cwd=root,
                check=True,
                capture_output=True,
                timeout=5,
            )
        if env_exists:
            (root / ".env").write_text(
                f"API_KEY={DUMMY_SECRET}\n",
                encoding="utf-8",
                newline="\n",
            )
        if gitignore is not None:
            (root / ".gitignore").write_bytes(gitignore)
        return root

    yield create
    temporary_directory.cleanup()
