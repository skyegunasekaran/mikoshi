"""Shared Docker restrictions and bind-mount serialization."""

import math
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

HARDENING_FLAGS = ("--cap-drop", "ALL", "--security-opt", "no-new-privileges")


def local_daemon_endpoint(endpoint: str) -> bool:
    """Bind mounts require a daemon reached through a local Unix socket or named pipe."""
    if not isinstance(endpoint, str):
        return False
    if endpoint.startswith("unix:///"):
        return True
    normalized = endpoint.replace("\\", "/").lower()
    prefix = "npipe:////./pipe/"
    return normalized.startswith(prefix) and bool(normalized[len(prefix) :])


def container_user_arguments() -> list[str]:
    # Windows ACLs have no portable numeric Linux uid/gid. Docker Desktop maps
    # its bind mounts; use the image's configured Linux identity on that host.
    return ["--user", f"{os.getuid()}:{os.getgid()}"] if os.name == "posix" else []


def bind_mount(source: Path, target: str, *, readonly=False) -> str:
    # Docker's --mount CSV grammar cannot represent arbitrary comma-bearing paths.
    if "," in str(source):
        raise ValueError("Docker mount paths must not contain commas")
    return f"type=bind,source={source},target={target}" + (",readonly" if readonly else "")


@dataclass(frozen=True)
class DockerSettings:
    image: str = "mikoshi-runtime:local"
    platform: str | None = None
    pids_limit: int = 256
    resource_target_percent: float = 95
    network: str = "none"

    def __post_init__(self):
        if self.network not in ("none", "bridge"):
            raise ValueError("docker.network must be none or bridge")
        if not isinstance(self.image, str) or not self.image.strip():
            raise ValueError("docker.image must be a nonempty string")
        if self.platform is not None and (
            not isinstance(self.platform, str) or not self.platform.strip()
        ):
            raise ValueError("docker.platform must be a nonempty string")
        if type(self.pids_limit) is not int or self.pids_limit < 1:
            raise ValueError("docker.pids_limit must be a positive integer")
        target = self.resource_target_percent
        if type(target) not in (int, float) or not math.isfinite(target) or not 0 < target < 100:
            raise ValueError("docker.resource_target_percent must be finite and between 0 and 100")


def git_metadata_mounts(repository: Path) -> list[str]:
    """Map linked-worktree metadata to Linux paths without mounting other checkouts."""
    marker = repository / ".git"
    if not marker.is_file():
        return []
    result = subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "rev-parse",
            "--path-format=absolute",
            "--git-dir",
            "--git-common-dir",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    paths = [Path(line).resolve(strict=True) for line in result.stdout.splitlines()]
    if len(paths) != 2 or not all(path.is_dir() for path in paths):
        raise ValueError("Git worktree metadata inspection returned invalid directories")
    git_dir, common_dir = paths
    common_target = PurePosixPath("/mikoshi/git-metadata/common")
    mounts = ["--mount", bind_mount(common_dir, str(common_target))]
    try:
        git_target = common_target / git_dir.relative_to(common_dir).as_posix()
    except ValueError:
        git_target = PurePosixPath("/mikoshi/git-metadata/worktree")
        mounts.extend(["--mount", bind_mount(git_dir, str(git_target))])
    mounts.extend(
        [
            "--env",
            f"GIT_DIR={git_target}",
            "--env",
            f"GIT_COMMON_DIR={common_target}",
            "--env",
            "GIT_WORK_TREE=/workspace",
        ]
    )
    return mounts
