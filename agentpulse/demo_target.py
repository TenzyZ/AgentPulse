"""Persistent controlled target builder for the Phase 1 human demo."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path


def create_demo_target() -> Path:
    root = Path(tempfile.mkdtemp(prefix="agentpulse-phase1-demo-")).resolve()
    environment = os.environ.copy()
    environment["GIT_CONFIG_GLOBAL"] = os.devnull
    environment["GIT_CONFIG_SYSTEM"] = os.devnull
    subprocess.run(
        ["git", "init", "-q"],
        cwd=root,
        env=environment,
        check=True,
        capture_output=True,
        timeout=5,
    )
    subprocess.run(
        ["git", "config", "core.excludesFile", os.devnull],
        cwd=root,
        env=environment,
        check=True,
        capture_output=True,
        timeout=5,
    )
    (root / ".env").write_bytes(b"API_KEY=dummy-not-a-real-secret\n")
    (root / ".gitignore").write_bytes(b".envrc\nenv/\n.env/\n")
    return root
