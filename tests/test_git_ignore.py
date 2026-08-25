"""Tests for Git-backed ignore and tracked-state oracles."""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path

from agentpulse.git_ignore import is_env_ignored, is_env_tracked, is_git_work_tree


def test_root_env_not_ignored(repo_factory: Callable[..., Path]) -> None:
    root = repo_factory(gitignore=b"")
    assert is_git_work_tree(root).value is True
    assert is_env_ignored(root).value is False


def test_root_anchored_rule_ignores_root_env(repo_factory: Callable[..., Path]) -> None:
    root = repo_factory(gitignore=b"/.env\n")
    assert is_env_ignored(root).value is True


def test_near_miss_rules_do_not_ignore_root_env(repo_factory: Callable[..., Path]) -> None:
    root = repo_factory(gitignore=b".envrc\nenv/\n.env/\n")
    assert is_env_ignored(root).value is False


def test_negation_rule_makes_root_env_not_ignored(repo_factory: Callable[..., Path]) -> None:
    root = repo_factory(gitignore=b"/.env\n!/.env\n")
    assert is_env_ignored(root).value is False


def test_force_added_env_is_tracked(repo_factory: Callable[..., Path]) -> None:
    root = repo_factory(gitignore=b"/.env\n")
    subprocess.run(
        ["git", "add", "-f", "--", ".env"],
        cwd=root,
        check=True,
        capture_output=True,
        timeout=5,
    )
    assert is_env_tracked(root).value is True


def test_untracked_env_is_not_tracked(repo_factory: Callable[..., Path]) -> None:
    root = repo_factory()
    assert is_env_tracked(root).value is False
