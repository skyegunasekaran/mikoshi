"""Check distribution boundaries and smoke-test an installed wheel without network access."""

from __future__ import annotations

import argparse
import configparser
import os
import re
import subprocess
import tarfile
import tempfile
import tomllib
import venv
import zipfile
from email.parser import Parser
from pathlib import Path, PurePosixPath

from mikoshi import RUNTIME_ASSET_NAMES, RUNTIME_ASSET_SUFFIXES


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def check_distribution(directory: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    stem = f"{project['name'].replace('-', '_')}-{project['version']}"
    wheels = list(directory.glob(f"{stem}-*.whl"))
    require(len(wheels) == 1, f"Expected one wheel for {stem}, found {len(wheels)}")
    wheel = wheels[0].resolve()
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        require("mikoshi/prompts/worker_environment.md" in names, "Wheel lacks worker guidance")
        require("mikoshi/__main__.py" in names, "Wheel lacks the module entry point")
        for name in names:
            top = PurePosixPath(name).parts[0]
            require(top in {"mikoshi", f"{stem}.dist-info"}, f"Unexpected wheel content: {name}")
            require("__pycache__" not in name and not name.endswith(".pyc"), name)
        runtime_files = {
            path.relative_to(root / "src").as_posix()
            for path in (root / "src/mikoshi").rglob("*")
            if path.is_file()
            and (path.suffix in RUNTIME_ASSET_SUFFIXES or path.name in RUNTIME_ASSET_NAMES)
        }
        packaged_runtime = {name for name in names if name.startswith("mikoshi/")}
        require(
            packaged_runtime == runtime_files,
            f"Wheel runtime differs from source: {sorted(packaged_runtime ^ runtime_files)}",
        )
        for name in runtime_files:
            require(
                archive.read(name) == (root / "src" / name).read_bytes(),
                f"Wheel contains stale runtime content: {name}",
            )
        entry_points = configparser.ConfigParser()
        entry_points.read_string(archive.read(f"{stem}.dist-info/entry_points.txt").decode())
        require(
            dict(entry_points["console_scripts"]) == project["scripts"],
            "Wheel entry points differ from project metadata",
        )
        metadata = Parser().parsestr(archive.read(f"{stem}.dist-info/METADATA").decode())
        require(
            metadata["License-Expression"] == project["license"],
            "Wheel license metadata differs from project metadata",
        )
        for notice in project["license-files"]:
            require(
                archive.read(f"{stem}.dist-info/licenses/{notice}") == (root / notice).read_bytes(),
                f"Wheel notice differs from source: {notice}",
            )

    source = directory / f"{stem}.tar.gz"
    with tarfile.open(source) as archive:
        names = {str(PurePosixPath(name).relative_to(stem)) for name in archive.getnames()}
        required = {
            "README.md",
            "EXPERIMENTS.md",
            "ARCHITECTURE.md",
            "SCHEMA.md",
            "LICENSE",
            "THIRD_PARTY.md",
            "AGENTS.md",
            "Makefile",
            "tests/support/check_repository.py",
            "tests/support/development.py",
            "tests/support/check_distribution.py",
            "tests/README.md",
            "pyproject.toml",
            "MANIFEST.in",
            ".dockerignore",
        }
        require(required <= names, f"Source distribution missing: {sorted(required - names)}")
        forbidden = {
            "docs",
            "examples",
            "scripts",
            "tmp",
            "training",
            "output",
            "build",
            "dist",
            ".git",
            ".venv",
            "__pycache__",
        }
        for name in names:
            require(not forbidden.intersection(PurePosixPath(name).parts), name)
            parts = PurePosixPath(name).parts
            if parts and parts[0] == "src":
                require(
                    len(parts) == 1
                    or parts[1] in {"mikoshi", f"{project['name'].replace('-', '_')}.egg-info"},
                    f"Unexpected source package: {name}",
                )
            require(not any(part == ".env" or part.startswith(".env.") for part in parts), name)
        for notice in project["license-files"]:
            member = archive.extractfile(f"{stem}/{notice}")
            require(member is not None and member.read() == (root / notice).read_bytes(), notice)

    with tempfile.TemporaryDirectory(prefix="mikoshi-package-") as temporary:
        environment = Path(temporary) / "venv"
        # Match `python -m venv`: copying a relocatable POSIX interpreter can
        # break its relative shared-library lookup (for example uv's macOS Python).
        venv.EnvBuilder(with_pip=True, symlinks=os.name != "nt").create(environment)
        scripts = environment / ("Scripts" if os.name == "nt" else "bin")
        python = scripts / ("python.exe" if os.name == "nt" else "python")
        subprocess.run(
            [str(python), "-I", "-m", "pip", "install", "--no-index", "--no-deps", str(wheel)],
            check=True,
            capture_output=True,
            text=True,
        )
        for command in (
            [str(python), "-I", "-m", "mikoshi"],
            [str(scripts / ("mikoshi.exe" if os.name == "nt" else "mikoshi"))],
        ):
            result = subprocess.run(
                [*command, "--help"], cwd=temporary, check=True, capture_output=True, text=True
            )
            options = set(re.findall(r"^  (--[\w-]+)", result.stdout, re.MULTILINE))
            required_options = {
                "--repository",
                "--agents",
                "--trials",
                "--trial-concurrency",
                "--wall-time-seconds",
                "--max-output-tokens",
                "--context-window-tokens",
                "--compaction-input-tokens",
                "--provider-concurrency",
                "--team-protocol",
                "--temperature",
                "--top-p",
            }
            require(required_options <= options, "Installed CLI lacks experimental settings")
            for subcommand in (
                "init",
                "plan",
                "run",
                "runs",
                "schema",
                "validate",
                "check",
                "status",
                "watch",
                "stop",
                "score",
            ):
                subprocess.run(
                    [*command, subcommand, "--help"],
                    cwd=temporary,
                    check=True,
                    capture_output=True,
                    text=True,
                )
        subprocess.run(
            [
                str(python),
                "-I",
                "-c",
                "from mikoshi.prompts import WORKER_ENVIRONMENT; "
                "from mikoshi.persistence.store import WorldStore; "
                "from importlib.resources import files; "
                "from mikoshi.tools.commands import CommandRunner; "
                "from mikoshi.isolation.docker import DockerSettings; "
                "assert files('mikoshi.isolation').joinpath('Dockerfile').is_file(); "
                "assert WORKER_ENVIRONMENT.strip(); "
                "store = WorldStore(); store.close()",
            ],
            cwd=temporary,
            check=True,
        )
    print(f"Verified {wheel.name}, {source.name}, installed entry points, prompt, and SQLite store")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?", default=Path("dist"))
    check_distribution(parser.parse_args().directory)
