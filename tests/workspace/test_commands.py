"""The private Docker-client supervisor strips credentials and reaps children."""

import os
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from mikoshi.isolation.processes import (
    ProcessPipe,
    run_captured,
    spawn_process,
    terminate_process_tree,
)
from mikoshi.tools.commands import _OutputCapture


def test_attach_supervision_retains_output_and_confirms_timeout_cleanup(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-openai-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "synthetic-anthropic-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "synthetic-google-key")
    monkeypatch.setenv("GEMINI_API_KEY", "synthetic-gemini-key")
    workspace = SimpleNamespace(
        canonical_repository=tmp_path,
        command_working_directory=lambda *_: tmp_path,
        file_state=lambda: {"whole_tree_observed": False},
        quarantine_repository=lambda *_: None,
    )
    capture = _OutputCapture(workspace, log_root=tmp_path / "logs")
    script = """
import os, pathlib, subprocess, sys, time
keys = ('OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'GOOGLE_API_KEY', 'GEMINI_API_KEY')
print('credentials-hidden', not any(os.environ.get(key) for key in keys), flush=True)
subprocess.Popen([sys.executable, '-c', "import pathlib,time; time.sleep(1); pathlib.Path('child-survived').touch()"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
pathlib.Path('started').touch()
time.sleep(30)
"""
    result = capture.execute(
        "peer",
        {
            "id": 1,
            "requested_by": "peer",
            "status": "approved",
            "argv": [sys.executable, "-c", script],
        },
        timeout_seconds=0.5,
    )
    assert (tmp_path / "started").exists()
    assert result["termination_reason"] == "timed_out", result
    assert "credentials-hidden True" in result["output"]
    cleanup = result["isolation"]["cleanup"]
    assert cleanup["confirmed"] is True
    if os.name == "nt":
        assert cleanup == {"confirmed": True, "scope": "job_object", "active_processes": 0}
    else:
        assert cleanup == {"confirmed": True, "scope": "process_group", "live_pids": []}
    assert "credentials-hidden True" in (tmp_path / "logs" / "command-1.log").read_text()
    time.sleep(1)
    assert not (tmp_path / "child-survived").exists()


def test_pipe_capture_is_bounded_and_can_close_while_writer_remains_open():
    reader, writer = os.pipe()
    pipe = ProcessPipe(os.fdopen(reader, "rb", buffering=0))
    try:
        assert pipe.read() is None
        os.write(writer, b"abcde")
        assert pipe.read(max_bytes=2) == b"ab"
        assert pipe.read(max_bytes=3) == b"cde"
        assert pipe.read() is None
        started = time.monotonic()
        pipe.close()
        assert time.monotonic() - started < 0.5
        assert pipe.read() == b""
    finally:
        pipe.close()
        os.close(writer)


def test_natural_parent_exit_still_terminates_inherited_pipe_descendants(tmp_path):
    code = (
        "import pathlib,subprocess,sys; "
        "subprocess.Popen([sys.executable, '-c', "
        "\"import pathlib,time; time.sleep(1); pathlib.Path('orphan').touch()\"]); "
        "print('parent finished', flush=True)"
    )
    process = spawn_process(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    pipe = ProcessPipe(process.stdout)
    try:
        process.wait(timeout=5)
        cleanup = terminate_process_tree(process)
        assert cleanup["confirmed"] is True
        assert terminate_process_tree(process) == cleanup
        chunks = []
        deadline = time.monotonic() + 2
        while not pipe.eof and time.monotonic() < deadline:
            value = pipe.read()
            if value is not None:
                chunks.append(value)
            else:
                time.sleep(0.01)
        assert pipe.eof
        assert b"parent finished" in b"".join(chunks)
        time.sleep(1)
        assert not (tmp_path / "orphan").exists()
    finally:
        terminate_process_tree(process)
        pipe.close()


def test_windows_spawn_assigns_job_before_resuming_and_fails_closed(monkeypatch):
    from mikoshi.isolation import processes, windows_processes

    events = []
    process = Mock(pid=123, stdin=None, stdout=None, stderr=None)
    process._handle = 456

    class Job:
        def __init__(self):
            events.append("create job")

        def assign_and_resume(self, child):
            assert child is process
            events.append("assign and resume")
            raise OSError("job assignment denied")

        def close(self):
            events.append("close job")

    def launch(argv, **options):
        assert options["creationflags"] & 0x00000004  # CREATE_SUSPENDED
        assert options["creationflags"] & 0x00000200  # CREATE_NEW_PROCESS_GROUP
        events.append("create suspended process")
        return process

    monkeypatch.setattr(processes, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(windows_processes, "WindowsJob", Job)
    monkeypatch.setattr(processes.subprocess, "Popen", launch)
    with pytest.raises(OSError, match="job assignment denied"):
        spawn_process(["docker.exe", "start", "--attach", "fixture"])
    assert events == ["create job", "create suspended process", "assign and resume", "close job"]
    process.kill.assert_called_once()
    process.wait.assert_called_once_with(timeout=2)


def test_control_utility_capture_retains_separate_streams_and_bounds_output():
    result = run_captured(
        [sys.executable, "-c", "import sys; print('stdout'); print('stderr', file=sys.stderr)"],
        timeout=5,
    )
    assert result.returncode == 0
    assert result.stdout.strip() == b"stdout"
    assert result.stderr.strip() == b"stderr"
    with pytest.raises(subprocess.SubprocessError, match="capture limit"):
        run_captured([sys.executable, "-c", "print('x' * 65536)"], timeout=5, max_output_bytes=1024)


def test_control_utility_timeout_terminates_descendants_before_returning(tmp_path):
    code = (
        "import subprocess,sys,time; "
        "subprocess.Popen([sys.executable, '-c', "
        "\"import pathlib,time; time.sleep(1); pathlib.Path('survived').touch()\"]); "
        "print('started', flush=True); time.sleep(30)"
    )
    with pytest.raises(subprocess.TimeoutExpired) as error:
        run_captured([sys.executable, "-c", code], cwd=tmp_path, timeout=0.3)
    assert b"started" in error.value.output
    time.sleep(1)
    assert not (tmp_path / "survived").exists()


def test_windows_job_assigns_before_resume_and_verifies_all_members_exit(monkeypatch):
    import ctypes

    from mikoshi.isolation import windows_processes

    events = []
    counts = iter([2, 0])
    process = SimpleNamespace(_handle=23, pid=42, wait=lambda **_: events.append("reap leader"))

    class API:
        def __init__(self):
            self.kernel = self

        def CreateJobObjectW(self, security, name):
            assert security is None and name is None
            events.append("create")
            return 17

        def SetInformationJobObject(self, handle, kind, pointer, size):
            limits = ctypes.cast(
                pointer, ctypes.POINTER(windows_processes._ExtendedLimits)
            ).contents
            assert limits.BasicLimitInformation.LimitFlags == 0x2000
            assert kind == 9 and size == ctypes.sizeof(windows_processes._ExtendedLimits)
            return True

        def AssignProcessToJobObject(self, handle, process_handle):
            assert handle == 17 and process_handle == 23
            events.append("assign")
            return True

        def resume_primary_thread(self, pid):
            assert pid == 42
            events.append("resume")

        def TerminateJobObject(self, handle, code):
            events.append("terminate job")
            return True

        def QueryInformationJobObject(self, handle, kind, pointer, size, returned):
            assert kind == 1
            count = next(counts)
            ctypes.cast(
                pointer, ctypes.POINTER(windows_processes._Accounting)
            ).contents.ActiveProcesses = count
            events.append(f"active {count}")
            return True

        def close(self, handle):
            assert handle == 17
            events.append("close job")

    monkeypatch.setattr(windows_processes, "WindowsAPI", API)
    job = windows_processes.WindowsJob()
    job.assign_and_resume(process)
    assert job.terminate(process, timeout=1) == {
        "confirmed": True,
        "scope": "job_object",
        "active_processes": 0,
    }
    job.close()
    job.close()
    assert events == [
        "create",
        "assign",
        "resume",
        "terminate job",
        "reap leader",
        "active 2",
        "active 0",
        "close job",
    ]
