"""Command-only Docker execution and shared host edits, with a mocked daemon."""

import json
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest
from support.runtime import create_repository, git

from mikoshi.tools.commands import CommandRunner, _OutputCapture

IMAGE_ID = "sha256:" + "a" * 64


@pytest.fixture
def docker(monkeypatch):
    monkeypatch.setattr("mikoshi.tools.commands.shutil.which", lambda _: "/mock/docker")
    monkeypatch.delenv("DOCKER_HOST", raising=False)
    monkeypatch.delenv("DOCKER_CONTEXT", raising=False)

    def respond(*args, **kwargs):
        output = ""
        if args[:2] == ("context", "inspect"):
            output = json.dumps([{"Endpoints": {"docker": {"Host": "unix:///mock/docker.sock"}}}])
        elif args[:2] == ("image", "inspect"):
            output = json.dumps([{"Id": IMAGE_ID, "Os": "linux", "Architecture": "amd64"}])
        return subprocess.CompletedProcess(args, 0, output, "")

    calls = Mock(side_effect=respond)
    monkeypatch.setattr(CommandRunner, "_docker", calls)
    return calls


def runner_for(repository, docker, **options):
    from types import SimpleNamespace

    workspace = SimpleNamespace(
        canonical_repository=repository,
        command_working_directory=lambda *_: repository,
        file_state=lambda: {"whole_tree_observed": False},
        quarantine_repository=Mock(),
    )
    runner = CommandRunner(workspace, log_root=repository.parent / "logs", **options)
    runner._queue.sample = lambda _: {"cpu_percent": 0, "memory_percent": 0}
    return runner


def prepare(runner, argv=("python", "-c", "pass"), *, actor="peer", **options):
    return runner.prepare_execution(
        actor, {"id": 1, "requested_by": actor, "status": "approved", "argv": argv}, **options
    )


def test_host_file_pressure_blocks_probe_and_preserves_other_checks(tmp_path, docker, monkeypatch):
    from mikoshi.isolation.admission import PressureUnavailable

    runner = runner_for(tmp_path, docker)
    prepared = prepare(runner)
    probe = Mock(return_value={"exit_code": 0, "output": "mock pressure"})
    monkeypatch.setattr(runner, "_execute_container", probe)
    monkeypatch.setattr(
        "mikoshi.tools.commands.host_file_table_pressure",
        lambda: {"host_file_table_percent": 95},
    )
    assert runner._pressure_sample(prepared) == {"host_file_table_percent": 95}
    probe.assert_not_called()

    def unavailable():
        raise PressureUnavailable("sensor failed")

    monkeypatch.setattr("mikoshi.tools.commands.host_file_table_pressure", unavailable)
    with pytest.raises(PressureUnavailable, match="sensor failed"):
        runner._pressure_sample(prepared)
    probe.assert_not_called()
    assert json.loads((runner.log_root / "admission-error.json").read_text())["scope"] == (
        "controller_host_file_table"
    )

    monkeypatch.setattr(
        "mikoshi.tools.commands.host_file_table_pressure",
        lambda: {"host_file_table_percent": 50},
    )
    monkeypatch.setattr(
        "mikoshi.tools.commands.parse_pressure",
        lambda _: {"cpu_percent": 30, "memory_percent": 40},
    )
    assert runner._pressure_sample(prepared) == {
        "cpu_percent": 30,
        "memory_percent": 40,
        "host_file_table_percent": 50,
    }
    probe.assert_called_once()
    monkeypatch.setattr("mikoshi.tools.commands.host_file_table_pressure", lambda: None)
    assert runner._pressure_sample(prepared) == {"cpu_percent": 30, "memory_percent": 40}


def test_bridge_network_is_explicit_and_probes_stay_offline(tmp_path, docker):
    from mikoshi.isolation.docker import DockerSettings

    runner = runner_for(tmp_path, docker, options=DockerSettings(network="bridge"))
    prepared = prepare(runner)
    regular = runner._create_arguments("fixture", prepared)
    probe = runner._create_arguments("fixture-probe", prepared, probe=True)
    assert regular[regular.index("--network") + 1] == "bridge"
    assert probe[probe.index("--network") + 1] == "none"
    assert runner.execution_context()["network"] == "bridge"
    assert "internet access" in runner.execution_context()["note"]
    assert not any("API_KEY" in argument or "docker.sock" in argument for argument in regular)
    with pytest.raises(ValueError, match="network"):
        DockerSettings(network="host")


def test_bind_mount_edits_are_visible_to_host_and_file_tools(tmp_path, docker, monkeypatch):
    from support.organization import problem

    from mikoshi.persistence.store import WorldStore
    from mikoshi.runtime.organization import bootstrap
    from mikoshi.tools.workspace import LiveWorkspace, resolve_repository

    repository = tmp_path / "linked checkout"
    source = create_repository(tmp_path)
    git(source, "worktree", "add", "-b", "fixture", str(repository))
    (repository / "existing.txt").write_text("Uncommitted user work")
    store = WorldStore()
    bootstrap(store, problem(max_agents=1))
    workspace = LiveWorkspace(tmp_path / "output", store, repository)
    workspace.initialize(resolve_repository(repository))
    runner = CommandRunner(workspace)
    runner._queue.sample = lambda _: {"cpu_percent": 0, "memory_percent": 0}
    arguments = ("python", "-c", "print('hello')", "", "a b", "$(touch unexpected)", "界")
    prepared = prepare(runner, arguments, actor="participant")
    captured = []

    def attach(self, client):
        captured.append(client.argv)
        # Emulate the daemon writing through the mount the create call supplied.
        create = next(call.args for call in docker.call_args_list if call.args[0] == "create")
        mount = create[create.index("--mount") + 1]
        assert mount == f"type=bind,source={repository},target=/workspace"
        (repository / "result.txt").write_text("Docker command result")
        return {
            "_command_id": client.command_id,
            "status": "succeeded",
            "exit_code": 0,
            "output": "hello\n",
            "isolation": {"cleanup": {"confirmed": True, "scope": "process_group"}},
        }

    monkeypatch.setattr(_OutputCapture, "execute_prepared", attach)
    try:
        result = runner.finalize_execution(runner.execute_prepared(prepared), command_id=1)
        assert result["status"] == "succeeded"
        assert (
            workspace.read_text("participant", "result.txt")["content"] == "Docker command result"
        )
        assert (repository / "existing.txt").read_text() == "Uncommitted user work"
        create = next(call.args for call in docker.call_args_list if call.args[0] == "create")
        name = create[create.index("--name") + 1]
        assert create[create.index("--entrypoint") + 1] == arguments[0]
        assert create[create.index(IMAGE_ID) + 1 :] == arguments[1:]
        assert create[create.index("--workdir") + 1] == "/workspace"
        assert create[create.index("--network") + 1] == "none"
        metadata = git(
            repository, "rev-parse", "--path-format=absolute", "--git-dir", "--git-common-dir"
        ).splitlines()
        common = Path(metadata[-1]).resolve()
        assert f"type=bind,source={common},target=/mikoshi/git-metadata/common" in create
        assert "GIT_DIR=/mikoshi/git-metadata/common/worktrees/linked-checkout" in create
        assert "GIT_COMMON_DIR=/mikoshi/git-metadata/common" in create
        assert f"type=bind,source={source},target={source}" not in create
        assert "GIT_WORK_TREE=/workspace" in create
        assert not (source / "result.txt").exists()
        assert "--cpus" not in create and "--memory" not in create
        assert "--cap-drop" in create and "no-new-privileges" in create
        assert create[create.index("--pull") + 1] == "never"
        assert not any("API_KEY" in arg or "docker.sock" in arg for arg in create)
        assert f"type=bind,source={runner.log_root},target=/logs,readonly" in create
        assert runner.log_location(1)[1] == "/logs/command-1.log"
        assert captured == [("/mock/docker", "start", "--attach", name)]
        assert docker.call_args.args == ("rm", "--force", "--volumes", name)
        assert result["isolation"]["cleanup"] == {"confirmed": True, "scope": "container"}
        with pytest.raises(RuntimeError, match="only run once"):
            runner.execute_prepared(prepared)
    finally:
        store.close()


def test_unconfirmed_cleanup_halts_admission_and_remote_daemons_are_rejected(
    tmp_path, docker, monkeypatch
):
    runner = runner_for(tmp_path, docker)
    normal = docker.side_effect

    def invoke(*args, **options):
        if args[0] == "rm":
            raise RuntimeError("mock daemon failure")
        return normal(*args, **options)

    def attach(*args):
        return {"_command_id": 1, "status": "succeeded", "exit_code": 0}

    docker.side_effect = invoke
    monkeypatch.setattr(_OutputCapture, "execute_prepared", attach)
    result = runner.execute(
        "peer", {"id": 1, "requested_by": "peer", "status": "approved", "argv": ["true"]}
    )
    assert result["status"] == "failed"
    assert docker.call_args.args[:3] == ("rm", "--force", "--volumes")
    assert result["termination_reason"] == "command_cleanup_failed"
    assert result["isolation"]["cleanup"]["confirmed"] is False
    runner.workspace.quarantine_repository.assert_called_once()
    creates = sum(call.args[0] == "create" for call in docker.call_args_list)
    with pytest.raises(RuntimeError, match="admission stopped"):
        runner.execute_prepared(prepare(runner))
    assert sum(call.args[0] == "create" for call in docker.call_args_list) == creates
    monkeypatch.setenv("DOCKER_HOST", "tcp://remote.invalid:2375")
    with pytest.raises(ValueError, match="local Unix-socket"):
        runner_for(tmp_path, docker)
    monkeypatch.setenv("DOCKER_HOST", "npipe:////remote/pipe/docker_engine")
    with pytest.raises(ValueError, match="local Unix-socket"):
        runner_for(tmp_path, docker)


def test_named_pipe_daemon_and_windows_image_user_are_supported(tmp_path, docker, monkeypatch):
    from types import SimpleNamespace

    from mikoshi.isolation import docker as docker_settings

    monkeypatch.setenv("DOCKER_HOST", "npipe:////./pipe/docker_engine")
    monkeypatch.setattr(docker_settings, "os", SimpleNamespace(name="nt"))
    runner = runner_for(tmp_path, docker)
    arguments = runner._create_arguments("windows-fixture", prepare(runner))
    assert "--user" not in arguments
    assert "--cap-drop" in arguments
    assert "--security-opt" in arguments
    assert "HOME=/tmp" in arguments
