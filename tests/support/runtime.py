"""Small test arrangements built from fixed rosters and repositories."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def git_environment() -> dict[str, str]:
    """Keep fixture repositories independent of caller Git settings and wall time."""
    return {
        **{key: value for key, value in os.environ.items() if not key.startswith("GIT_")},
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_AUTHOR_NAME": "Fixture Author",
        "GIT_AUTHOR_EMAIL": "fixture@example.invalid",
        "GIT_COMMITTER_NAME": "Fixture Author",
        "GIT_COMMITTER_EMAIL": "fixture@example.invalid",
        "GIT_AUTHOR_DATE": "2000-01-01T00:00:00+00:00",
        "GIT_COMMITTER_DATE": "2000-01-01T00:00:00+00:00",
    }


def git(root, *args):
    return (
        subprocess.run(
            ["git", "-c", "commit.gpgsign=false", "-C", str(root), *args],
            check=True,
            capture_output=True,
            env=git_environment(),
        )
        .stdout.decode()
        .strip()
    )


def create_repository(root: Path) -> Path:
    source = root / "source"
    source.mkdir()
    for arguments in (
        ("init", "-b", "main"),
        ("config", "user.name", "Fixture Author"),
        ("config", "user.email", "fixture@example.invalid"),
        ("-c", "commit.gpgsign=false", "commit", "--allow-empty", "-m", "Initial commit"),
    ):
        git(source, *arguments)
    return source
