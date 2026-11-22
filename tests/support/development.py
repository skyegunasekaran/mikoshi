"""Run contributor checks from any supported shell, without requiring Make."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable
COMMANDS = {
    "lint": [[PYTHON, "-m", "ruff", "check", "src/mikoshi", "tests"]],
    "format-check": [[PYTHON, "-m", "ruff", "format", "--check", "src/mikoshi", "tests"]],
    "docs-check": [[PYTHON, "tests/support/check_repository.py"]],
    "test": [[PYTHON, "-m", "pytest", "-q"]],
    "format": [
        [PYTHON, "-m", "ruff", "check", "--select", "I", "--fix", "src/mikoshi", "tests"],
        [PYTHON, "-m", "ruff", "format", "src/mikoshi", "tests"],
    ],
    "build": [
        [PYTHON, "-m", "build"],
        [PYTHON, "tests/support/check_distribution.py", "dist"],
    ],
}
COMMANDS["check"] = sum(
    (COMMANDS[name] for name in ("lint", "format-check", "docs-check", "test")), []
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=sorted(COMMANDS))
    args = parser.parse_args()
    try:
        for command in COMMANDS[args.command]:
            subprocess.run(command, cwd=ROOT, check=True)
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode) from None


if __name__ == "__main__":
    main()
