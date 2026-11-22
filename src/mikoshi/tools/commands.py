"""Docker commands against the live checkout, with bounded output and cleanup."""

from __future__ import annotations

import codecs
import json
import math
import os
import shutil
import subprocess
import threading
import time
import uuid
from concurrent.futures import CancelledError
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from mikoshi.contracts import COMMAND_ARGUMENT_CHARS
from mikoshi.isolation.admission import (
    PRESSURE_SCRIPT,
    CommandQueue,
    PressureUnavailable,
    host_file_table_pressure,
    host_file_table_source,
    parse_pressure,
)
from mikoshi.isolation.docker import (
    HARDENING_FLAGS,
    DockerSettings,
    bind_mount,
    container_user_arguments,
    git_metadata_mounts,
    local_daemon_endpoint,
)
from mikoshi.isolation.processes import (
    ProcessCleanupError,
    ProcessPipe,
    run_captured,
    spawn_process,
    terminate_process_tree,
)
from mikoshi.persistence.exports import write_json
from mikoshi.tools.text_tools import MAX_BYTES, MAX_LINES, format_size, truncate_tail
from mikoshi.tools.workspace import WorkspaceChangedError

DEFAULT_MAX_OUTPUT_BYTES = 64 * 1024 * 1024
HARD_MAX_OUTPUT_BYTES = 1024 * 1024 * 1024
DEFAULT_OUTPUT_DRAIN_SECONDS = 2.0
# Like Claude Code: a harness default the agent can raise per command, up to a maximum.
DEFAULT_COMMAND_TIMEOUT_SECONDS = 600
MAX_COMMAND_TIMEOUT_SECONDS = 3600
# Where commands see the shared checkout; file tools accept this prefix too.
CONTAINER_WORKSPACE = "/workspace"
PROVIDER_CREDENTIAL_ENV = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
)


class OutputLog:
    """Merged stdout/stderr with a tail preview and a flushed, bounded live log.

    Chunks are recorded in arrival order and immediately flushed to `path`,
    up to `max_bytes`. The preview continues tracking the tail after that limit.
    """

    def __init__(self, path: Path | None, visible_path: str | None, max_bytes: int):
        self.path, self.visible_path, self.max_bytes = path, visible_path, max_bytes
        self.tail = bytearray()
        self.tail_at_line_start = True
        self.total_bytes = 0
        self.stored_bytes = 0
        self.newlines = 0
        self.ends_with_newline = True
        self.open_line_bytes = 0
        self.closed_line_bytes = 0
        self.file = None
        self.write_error = None
        self._store(b"")

    @property
    def total_lines(self) -> int:
        return self.newlines + (0 if self.ends_with_newline else 1)

    @property
    def truncated(self) -> bool:
        return self.total_bytes > MAX_BYTES or self.total_lines > MAX_LINES

    def append(self, chunk: bytes) -> None:
        if not chunk:
            return
        self.total_bytes += len(chunk)
        self.newlines += chunk.count(b"\n")
        self.ends_with_newline = chunk.endswith(b"\n")
        if b"\n" in chunk:
            last = chunk.rfind(b"\n")
            previous = chunk.rfind(b"\n", 0, last)
            self.closed_line_bytes = (
                last - previous - 1 if previous >= 0 else self.open_line_bytes + last
            )
            self.open_line_bytes = len(chunk) - last - 1
        else:
            self.open_line_bytes += len(chunk)
        self.tail.extend(chunk)
        excess = len(self.tail) - 2 * MAX_BYTES
        if excess > 0:
            self.tail_at_line_start = self.tail[excess - 1] == 0x0A
            del self.tail[:excess]
        self._store(chunk)

    def _store(self, chunk: bytes) -> None:
        # A log failure never interrupts the command; the notice reports it instead.
        if self.path is None or self.write_error is not None:
            return
        try:
            if self.file is None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.file = open(self.path, "xb")
            room = self.max_bytes - self.stored_bytes
            if room > 0:
                self.file.write(chunk[:room])
                self.file.flush()
                self.stored_bytes += min(room, len(chunk))
        except OSError as error:
            self.write_error = f"{type(error).__name__}: {error}"
            self.close()

    def close(self) -> None:
        if self.file is not None and not self.file.closed:
            try:
                self.file.close()
            except OSError as error:
                self.write_error = self.write_error or f"{type(error).__name__}: {error}"

    def result(self) -> Mapping[str, Any]:
        self.close()
        text = self.tail.decode("utf-8", errors="replace")
        if not self.tail_at_line_start:
            # Drop the clipped first line unless it is the only (final, oversized) line.
            rest = text.partition("\n")[2]
            if rest:
                text = rest
        if not self.truncated:
            return {
                "output": text,
                "output_truncated": False,
                "log_path": self.visible_path
                if self.file is not None and self.write_error is None
                else None,
            }
        content, kept, truncated_by, partial = truncate_tail(text, MAX_LINES, MAX_BYTES)
        total = self.total_lines
        saved = self.file is not None and self.write_error is None
        where = (
            "Read retained output with read_output"
            if saved
            else f"Full output was not saved ({self.write_error})"
            if self.write_error
            else "Full output was not saved"
        )
        if saved and self.stored_bytes < self.total_bytes:
            where += f" (first {format_size(self.stored_bytes)} of {format_size(self.total_bytes)})"
        if partial:
            notice = (
                f"[Showing last {format_size(len(content.encode()))} of line {total} "
                f"(line is {format_size(self.open_line_bytes or self.closed_line_bytes)}). {where}]"
            )
        elif truncated_by == "bytes":
            notice = (
                f"[Showing lines {total - kept + 1}-{total} of {total} "
                f"({format_size(MAX_BYTES)} limit). {where}]"
            )
        else:
            notice = f"[Showing lines {total - kept + 1}-{total} of {total}. {where}]"
        return {
            "output": f"{content}\n\n{notice}",
            "output_truncated": True,
            "log_path": self.visible_path if saved else None,
        }


def append_note(result: dict, note: str) -> None:
    """Add a runtime status line (timeout, cancellation, launch error) after the output."""
    output = (result.get("output") or "").rstrip("\n")
    result["output"] = f"{output}\n\n{note}" if output else note


class CommandCleanupError(RuntimeError):
    """Process cleanup or its durable failure record could not be confirmed."""


@dataclass(frozen=True)
class CommandLimits:
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES  # Bytes kept in each full log.
    max_argv_items: int = 32
    max_arg_chars: int = COMMAND_ARGUMENT_CHARS
    default_timeout_seconds: int = DEFAULT_COMMAND_TIMEOUT_SECONDS
    max_timeout_seconds: int = MAX_COMMAND_TIMEOUT_SECONDS

    def __post_init__(self):
        if self.default_timeout_seconds > self.max_timeout_seconds:
            raise ValueError("default command timeout cannot exceed the maximum")
        for name in (
            "max_output_bytes",
            "max_argv_items",
            "max_arg_chars",
            "default_timeout_seconds",
            "max_timeout_seconds",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 0 < self.max_output_bytes <= HARD_MAX_OUTPUT_BYTES:
            raise ValueError(f"max_output_bytes must be between 1 and {HARD_MAX_OUTPUT_BYTES}")


@dataclass(frozen=True)
class PreparedExecution:
    argv: tuple[str, ...]
    working_directory: Path
    timeout_seconds: float | None  # The command timeout, capped by remaining run time.
    max_output_bytes: int
    command_id: int | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event, compare=False)
    queued_at: float = field(default_factory=time.monotonic)
    reported_timeout_seconds: float | None = None  # The timeout as the agent sees it.
    _execution_lock: Any = field(default_factory=threading.Lock, compare=False, repr=False)


class _OutputCapture:
    def __init__(
        self,
        workspace,
        limits=None,
        *,
        output_drain_seconds: float = DEFAULT_OUTPUT_DRAIN_SECONDS,
        log_root: Path | None = None,
        output_callback: Callable[[int | None, str], None] | None = None,
    ):
        if (
            isinstance(output_drain_seconds, bool)
            or not isinstance(output_drain_seconds, (int, float))
            or not math.isfinite(output_drain_seconds)
            or output_drain_seconds <= 0
        ):
            raise ValueError("output_drain_seconds must be positive and finite")
        self.workspace = workspace
        self.limits = limits or CommandLimits()
        self.output_drain_seconds = output_drain_seconds
        self.output_callback = output_callback
        root = getattr(workspace, "root", None)
        self.log_root = (
            Path(log_root)
            if log_root is not None
            else Path(root) / "command-logs"
            if root
            else None
        )

    def log_location(self, command_id) -> tuple[Path | None, str | None]:
        """Host path for a command's full log and the path commands can read it at."""
        if self.log_root is None or command_id is None:
            return None, None
        path = self.log_root / f"command-{command_id}.log"
        return path, str(path)

    def normalize_request(self, argv: Sequence[Any]) -> Mapping[str, Any]:
        if isinstance(argv, (str, bytes)) or not isinstance(argv, Sequence):
            raise TypeError("command argv must be an array of exact argument strings")
        if not 1 <= len(argv) <= self.limits.max_argv_items:
            raise ValueError("invalid number of command arguments")
        if (
            any(
                not isinstance(arg, str) or "\x00" in arg or len(arg) > self.limits.max_arg_chars
                for arg in argv
            )
            or not argv[0]
        ):
            raise ValueError("invalid command argument")
        return {"argv": tuple(argv)}

    def prepare_execution(
        self, agent_id, approved_request, *, timeout_seconds=None, cancel_event=None
    ):
        if (
            approved_request.get("status") != "approved"
            or approved_request.get("requested_by") != agent_id
        ):
            raise PermissionError("command identity does not match the active agent")
        argv = self.normalize_request(approved_request["argv"])["argv"]
        if timeout_seconds is not None and (
            not math.isfinite(timeout_seconds) or timeout_seconds <= 0
        ):
            raise ValueError("execution timeout must be positive and finite")
        return PreparedExecution(
            argv,
            self.workspace.command_working_directory(agent_id),
            timeout_seconds,
            self.limits.max_output_bytes,
            approved_request.get("id"),
            cancel_event if cancel_event is not None else threading.Event(),
        )

    def execute(self, agent_id, approved_request, **options):
        prepared = self.prepare_execution(agent_id, approved_request, **options)
        return self.finalize_execution(
            self.execute_prepared(prepared), command_id=prepared.command_id
        )

    def execute_prepared(self, prepared):
        if not prepared._execution_lock.acquire(blocking=False):
            raise RuntimeError("prepared command execution may only run once")
        start = time.monotonic()
        result = {
            "_command_id": prepared.command_id,
            "status": "failed",
            "exit_code": None,
            "output": "",
            "output_truncated": False,
            "log_path": None,
            "termination_reason": "cancelled",
            "isolation": {"backend": "docker_client", "required": True},
        }
        process = None
        output_pipe = None
        log = OutputLog(*self.log_location(prepared.command_id), prepared.max_output_bytes)
        notes = []
        errors = []

        streamed_bytes = 0
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        callback = self.output_callback

        def stream(chunk):
            nonlocal streamed_bytes, callback
            if callback is None or streamed_bytes >= prepared.max_output_bytes:
                return
            room = prepared.max_output_bytes - streamed_bytes
            bounded = chunk[:room]
            streamed_bytes += len(bounded)
            text = decoder.decode(bounded, final=not chunk or len(chunk) >= room)
            if text:
                try:
                    callback(prepared.command_id, text)
                except Exception as error:
                    notes.append(f"Live output unavailable: {type(error).__name__}")
                    callback = None

        def read_output(timeout):
            deadline = time.monotonic() + timeout
            while output_pipe is not None and not output_pipe.eof:
                chunk = output_pipe.read()
                if chunk is not None:
                    if chunk:
                        log.append(chunk)
                    stream(chunk)
                    return
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return
                prepared.cancel_event.wait(min(0.01, remaining))
            if timeout:
                prepared.cancel_event.wait(timeout)

        def check_active():
            if prepared.cancel_event.is_set():
                raise CancelledError()
            if (
                prepared.timeout_seconds is not None
                and time.monotonic() - start >= prepared.timeout_seconds
            ):
                raise TimeoutError()

        try:
            check_active()
            environment = dict(os.environ)
            for key in PROVIDER_CREDENTIAL_ENV:
                environment.pop(key, None)
            process = spawn_process(
                prepared.argv,
                cwd=prepared.working_directory,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # One stream, interleaved as the command wrote it.
                shell=False,
                close_fds=True,
            )
            result["isolation"]["pid"] = process.pid
            output_pipe = ProcessPipe(process.stdout)
            while process.poll() is None:
                read_output(0.05)
                check_active()
            check_active()
            result.update(
                exit_code=process.returncode,
                termination_reason="completed",
                status="succeeded" if process.returncode == 0 else "failed",
            )
        except CancelledError:
            result.update(termination_reason="cancelled")
            notes.append("Command cancelled.")
        except TimeoutError:
            result.update(status="timed_out", termination_reason="timed_out")
            limit = prepared.reported_timeout_seconds or prepared.timeout_seconds
            notes.append(f"Command timed out after {limit:g} seconds.")
        except ProcessCleanupError as error:
            result.update(
                status="failed",
                termination_reason="command_cleanup_failed",
                _quarantine_reason=f"Docker client launch cleanup unconfirmed: {error}",
            )
            notes.append(str(error))
        except Exception as error:
            result.update(termination_reason="execution_error")
            notes.append(f"{type(error).__name__}: {error}")
        finally:
            if process is not None:
                try:
                    cleanup = terminate_process_tree(process)
                    result["isolation"]["cleanup"] = cleanup
                    if not cleanup["confirmed"]:
                        raise CommandCleanupError(str(cleanup))
                except Exception as error:
                    result.update(
                        status="failed",
                        termination_reason="command_cleanup_failed",
                        _quarantine_reason=f"Docker client process cleanup unconfirmed: {error}",
                    )
                    result["isolation"]["cleanup"] = {
                        "confirmed": False,
                        "scope": "job_object" if os.name == "nt" else "process_group",
                        "error": str(error),
                    }
            try:
                drain_deadline = time.monotonic() + self.output_drain_seconds
                while (
                    output_pipe is not None
                    and not output_pipe.eof
                    and time.monotonic() < drain_deadline
                ):
                    read_output(0.05)
                if output_pipe is not None and not output_pipe.eof:
                    errors.append("output pipes stayed open after process-tree cleanup")
            except (OSError, ValueError) as error:
                errors.append(str(error))
            finally:
                if output_pipe is not None:
                    output_pipe.close()
                elif process is not None:
                    process.stdout.close()
            if errors:
                result["status"] = "failed"
                notes.append("Command output capture failed: " + "; ".join(errors))
                if result["termination_reason"] == "completed":
                    result["termination_reason"] = "output_capture_failed"
        result.update(log.result())
        for note in notes:
            append_note(result, note)
        result["duration_ms"] = round((time.monotonic() - start) * 1000)
        return result

    def finalize_execution(self, result, *, command_id=None):
        value = dict(result)
        actual_id = value.pop("_command_id", None)
        if command_id is not None and actual_id != command_id:
            raise ValueError("command result belongs to a different command")
        quarantine = value.pop("_quarantine_reason", None)
        fatal = value.pop("_fatal_error", None)
        if quarantine:
            try:
                self.workspace.quarantine_repository(quarantine)
            except Exception as error:
                raise CommandCleanupError("durable quarantine could not be written") from error
            value["workspace_state"] = None
            if fatal:
                raise CommandCleanupError(fatal)
            return value
        try:
            value["workspace_state"] = self.workspace.file_state()
            if value["workspace_state"].get("whole_tree_observed") is False:
                value.setdefault("isolation", {})["workspace_observation"] = (
                    "unobserved; repository was not inventoried"
                )
        except WorkspaceChangedError as error:
            value.update(workspace_state=None, workspace_error=str(error))
        except Exception as error:
            value.update(
                status="failed",
                termination_reason="workspace_validation_failed",
                workspace_state=None,
                workspace_error=f"{type(error).__name__}: {error}",
            )
        return value


class CommandRunner(_OutputCapture):
    """Run each command in a fresh container with the host checkout mounted."""

    def __init__(self, workspace, limits=None, *, options=None, **capture_options):
        super().__init__(workspace, limits, **capture_options)
        self.options = options or DockerSettings()
        self._queue = CommandQueue(
            lambda prepared: self._execute_container(prepared),
            self._pressure_sample,
            self.options.resource_target_percent,
        )
        self.executable = shutil.which("docker")
        if self.executable is None:
            raise ValueError("Docker is required; install Docker and start its daemon")
        bind_mount(workspace.canonical_repository, CONTAINER_WORKSPACE)
        if self.log_root is not None:
            bind_mount(self.log_root, "/logs", readonly=True)
            self.log_root.mkdir(parents=True, exist_ok=True)
        self._inspect_docker()

    def _docker(self, *args, timeout=30):
        environment = dict(os.environ)
        for key in PROVIDER_CREDENTIAL_ENV:
            environment.pop(key, None)
        result = run_captured(
            [self.executable, *args],
            env=environment,
            timeout=timeout,
        )
        result.stdout = result.stdout.decode("utf-8", errors="replace")
        result.stderr = result.stderr.decode("utf-8", errors="replace")
        if result.returncode:
            detail = (result.stderr or "")[:2000]
            for key in PROVIDER_CREDENTIAL_ENV:
                secret = os.environ.get(key)
                if secret:
                    detail = detail.replace(secret, "[redacted]")
            raise RuntimeError(f"Docker {args[0]} failed: {detail}")
        return result

    def _inspect_docker(self):
        # Bind mounts refer to daemon-local paths; a remote daemon would edit a
        # different checkout or fail to mount the requested directory.
        if os.environ.get("DOCKER_CONTEXT") or not os.environ.get("DOCKER_HOST"):
            context = json.loads(self._docker("context", "inspect").stdout)
            endpoint = context[0]["Endpoints"]["docker"]["Host"]
        else:
            endpoint = os.environ["DOCKER_HOST"]
        if not local_daemon_endpoint(endpoint):
            raise ValueError(
                "Docker commands require a local Unix-socket or Windows named-pipe daemon for bind mounts"
            )
        metadata = json.loads(self._docker("image", "inspect", self.options.image).stdout)
        if not isinstance(metadata, list) or len(metadata) != 1:
            raise ValueError("Docker image inspection returned invalid metadata")
        image = metadata[0]
        self.image_id = image["Id"]
        if not (
            isinstance(self.image_id, str)
            and self.image_id.startswith("sha256:")
            and len(self.image_id) == 71
            and all(c in "0123456789abcdef" for c in self.image_id[7:])
        ):
            raise ValueError("Docker image inspection did not return an immutable image ID")
        if image.get("Os") != "linux":
            raise ValueError("Docker command execution requires a Linux image")
        self.image_platform = "/".join(
            str(image[key]) for key in ("Os", "Architecture", "Variant") if image.get(key)
        )
        if self.options.platform and not (
            self.image_platform == self.options.platform
            or self.image_platform.startswith(self.options.platform + "/")
        ):
            raise ValueError("Local Docker image platform differs from deployment.docker.platform")

    def log_location(self, command_id):
        path, _ = super().log_location(command_id)
        return path, f"/logs/{path.name}" if path else None

    def execution_context(self):
        return {
            "working_directory": CONTAINER_WORKSPACE,
            "logs_directory": "/logs" if self.log_root else None,
            "backend": "docker",
            "sandboxed": True,
            "image": self.options.image,
            "image_id": self.image_id,
            "platform": self.image_platform,
            "network": self.options.network,
            "pids_per_command": self.options.pids_limit,
            "resource_target_percent": self.options.resource_target_percent,
            "host_file_table_sampling": {
                "scope": "controller_host",
                "source": host_file_table_source(),
                "supported": host_file_table_source() is not None,
                "unavailable": "queue and retry on supported platforms; otherwise omit this check",
            },
            "admission": (
                "FIFO; waits while Docker-host CPU or memory, or supported controller-host "
                "file-table usage, reaches the target"
            ),
            "note": (
                "Each command runs in a fresh Docker container. /workspace is the live host "
                "checkout and its edits persist. File tools use paths relative to that same "
                "checkout. Logs are mounted read-only at /logs. Container files outside "
                "/workspace, shell state, and background services do not persist between "
                "commands. No host credentials, environment, or Docker socket are passed in. "
                + (
                    "Commands have internet access through Docker's bridge network."
                    if self.options.network == "bridge"
                    else "Dependencies must be prepared in the image; commands have no network access."
                )
            ),
        }

    def _create_arguments(self, name, prepared, *, probe=False):
        options = self.options
        return [
            "create",
            "--name",
            name,
            "--pull",
            "never",
            "--network",
            "none" if probe else options.network,
            *HARDENING_FLAGS,
            "--init",
            "--no-healthcheck",
            "--restart",
            "no",
            "--log-driver",
            "none",
            *(
                ["--read-only", "--cpus", "0.1", "--memory", "64m", "--memory-swap", "64m"]
                if probe
                else []
            ),
            "--pids-limit",
            str(16 if probe else options.pids_limit),
            *container_user_arguments(),
            "--env",
            "HOME=/tmp",
            *(
                ["--mount", bind_mount(prepared.working_directory, CONTAINER_WORKSPACE)]
                if not probe
                else []
            ),
            *(
                ["--mount", bind_mount(self.log_root, "/logs", readonly=True)]
                if self.log_root and not probe
                else []
            ),
            *(git_metadata_mounts(prepared.working_directory) if not probe else []),
            "--workdir",
            CONTAINER_WORKSPACE,
            "--entrypoint",
            prepared.argv[0],
            *(["--platform", options.platform] if options.platform else []),
            self.image_id,
            *prepared.argv[1:],
        ]

    def execute_prepared(self, prepared):
        return self.submit(prepared).result()

    def submit(self, prepared):
        return self._queue.submit(prepared)

    def _pressure_sample(self, prepared):
        try:
            host_pressure = host_file_table_pressure() or {}
        except PressureUnavailable as error:
            if self.log_root:
                try:
                    write_json(
                        self.log_root / "admission-error.json",
                        {
                            "recorded_at": time.time(),
                            "scope": "controller_host_file_table",
                            "error": str(error),
                        },
                    )
                except OSError:
                    pass
            raise
        # Check before creating the Docker probe: that probe itself needs host
        # processes, pipes and files, and cleanup must retain resource headroom.
        if host_pressure.get("host_file_table_percent", 0) >= self.options.resource_target_percent:
            return host_pressure
        remaining = (
            prepared.queued_at + prepared.timeout_seconds - time.monotonic()
            if prepared.timeout_seconds is not None
            else 10
        )
        probe = replace(
            prepared,
            argv=("sh", "-c", PRESSURE_SCRIPT),
            command_id=None,
            timeout_seconds=min(10, max(0, remaining)),
            queued_at=time.monotonic(),
            max_output_bytes=4096,
            _execution_lock=threading.Lock(),
        )
        result = self._execute_container(probe, probe=True)
        if result.get("_quarantine_reason"):
            self.workspace.quarantine_repository(result["_quarantine_reason"])
            raise CommandCleanupError(result["_quarantine_reason"])
        if prepared.cancel_event.is_set() or (
            prepared.timeout_seconds is not None
            and time.monotonic() >= prepared.queued_at + prepared.timeout_seconds
        ):
            return {"cpu_percent": 100, "memory_percent": 100}
        try:
            if result.get("exit_code") != 0:
                raise ValueError("Docker resource observation failed: " + result.get("output", ""))
            return {**parse_pressure(result["output"]), **host_pressure}
        except ValueError as error:
            # Probe cleanup was confirmed above. Missing telemetry blocks admission,
            # not the agents or already-running commands. Preserve the last failure.
            if self.log_root:
                try:
                    write_json(
                        self.log_root / "admission-error.json",
                        {"recorded_at": time.time(), "error": str(error), "probe": result},
                    )
                except OSError:
                    pass
            raise PressureUnavailable(str(error)) from error

    def _execute_container(self, prepared, *, probe=False):
        deadline = (
            prepared.queued_at + prepared.timeout_seconds
            if prepared.timeout_seconds is not None
            else None
        )
        name = ("mikoshi-probe-" if probe else "mikoshi-command-") + uuid.uuid4().hex
        result = {
            "_command_id": prepared.command_id,
            "status": "failed",
            "exit_code": None,
            "output": "",
            "output_truncated": False,
            "log_path": None,
            "termination_reason": "execution_error",
        }
        attempted_create = created = False
        try:
            if prepared.cancel_event.is_set():
                raise CancelledError()
            remaining = deadline - time.monotonic() if deadline is not None else None
            if remaining is not None and remaining <= 0:
                raise TimeoutError()
            arguments = self._create_arguments(name, prepared, probe=probe)
            remaining = deadline - time.monotonic() if deadline is not None else None
            if remaining is not None and remaining <= 0:
                raise TimeoutError()
            attempted_create = True
            self._docker(*arguments, timeout=min(30, remaining) if remaining is not None else 30)
            created = True
            remaining = deadline - time.monotonic() if deadline is not None else None
            if remaining is not None and remaining <= 0:
                raise TimeoutError()
            attached = replace(
                prepared,
                argv=(self.executable, "start", "--attach", name),
                timeout_seconds=remaining,
                reported_timeout_seconds=prepared.reported_timeout_seconds
                or prepared.timeout_seconds,
                _execution_lock=threading.Lock(),
            )
            # Only the Docker attach client runs on the host. Its process-tree
            # cleanup is separate from destruction of the command container.
            capture = _OutputCapture(self.workspace, self.limits) if probe else self
            result = _OutputCapture.execute_prepared(capture, attached)
        except CancelledError:
            result.update(termination_reason="cancelled")
            append_note(result, "Command cancelled.")
        except (TimeoutError, subprocess.TimeoutExpired):
            result.update(status="timed_out", termination_reason="timed_out")
            append_note(result, "Command deadline expired during Docker admission or setup.")
        except Exception as error:
            append_note(result, f"{type(error).__name__}: {error}")
        finally:
            isolation = {
                "backend": "docker",
                "required": True,
                "container": name,
                "image_id": self.image_id,
                "network": self.options.network,
                "client": result.pop("isolation", None),
            }
            if attempted_create:
                try:
                    self._docker("rm", "--force", "--volumes", name)
                    if not created:
                        raise RuntimeError("Docker create outcome is uncertain")
                    isolation["cleanup"] = {"confirmed": True, "scope": "container"}
                except Exception as error:
                    isolation["cleanup"] = {
                        "confirmed": False,
                        "scope": "container",
                        "error": str(error),
                    }
                    result.update(
                        status="failed",
                        termination_reason="command_cleanup_failed",
                        _quarantine_reason=f"Docker container cleanup unconfirmed ({name}): {error}",
                    )
            else:
                isolation["cleanup"] = {
                    "confirmed": True,
                    "scope": "container",
                    "not_created": True,
                }
            result["isolation"] = isolation
            result["duration_ms"] = round((time.monotonic() - prepared.queued_at) * 1000)
        return result


COMPLETED_STATUSES = frozenset({"succeeded", "failed", "timed_out"})
