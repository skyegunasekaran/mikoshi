"""The host supervisor drains pipes and reaps trials when evaluation stops."""

import asyncio
import json
import os
import sys
import time

import pytest

from mikoshi.runtime import process


def test_supervisor_retains_both_streams_and_bounds_observer_events(tmp_path, monkeypatch):
    script = tmp_path / "worker.py"
    script.write_text(
        "import os\n"
        "for _ in range(64):\n"
        "    os.write(1, b'a' * 16384)\n"
        "    os.write(2, b'b' * 16384)\n"
        'os.write(2, b\'\\n{"progress": {"kind": "finished"}}\\n\')\n'
    )
    monkeypatch.setattr(process, "_command", lambda _: [sys.executable, str(script)])
    events = []

    def observer(event):
        events.append(event)
        if event.get("stream") == "stdout":
            raise RuntimeError("observer unavailable")

    async def run():
        return await asyncio.wait_for(
            process.run_process(tmp_path / "config.json", tmp_path / "run", observer), 10
        )

    assert asyncio.run(run()) == 0
    assert (tmp_path / "supervisor.stdout.log").read_bytes() == b"a" * 1048576
    assert (tmp_path / "supervisor.stderr.log").read_bytes().startswith(b"b" * 1048576)
    assert any(event.get("progress", {}).get("kind") == "finished" for event in events)
    assert all(len(event.get("text", "")) <= 8192 for event in events)
    assert sum(event.get("truncated", False) for event in events) == 2
    assert not (tmp_path / "run").exists()
    report = json.loads((tmp_path / "supervisor.json").read_text())
    assert report["returncode"] == 0 and not report["exports_present"]
    assert report["controller_cleanup"]["confirmed"]


@pytest.mark.parametrize("cancel_task", [True, False])
def test_supervisor_waits_for_startup_then_stops_and_reaps(tmp_path, monkeypatch, cancel_task):
    script = tmp_path / "worker.py"
    script.write_text(
        "import json, pathlib, sys, time\n"
        "root = pathlib.Path(sys.argv[1])\n"
        "time.sleep(0.2)\n"
        "assert not root.exists(), 'supervisor created output before startup'\n"
        "root.mkdir()\n"
        "(root / 'run.json').write_text(json.dumps({'status': 'running'}))\n"
        "while not (root / 'STOP').exists(): time.sleep(0.01)\n"
        "time.sleep(0.1)\n"
        "(root / 'run.json').write_text(json.dumps({'status': 'stopped'}))\n"
        "(root / 'records.jsonl').write_text('')\n"
    )
    run_dir = tmp_path / "run"
    monkeypatch.setattr(process, "_command", lambda _: [sys.executable, str(script), str(run_dir)])

    async def run():
        stop = False
        trial = asyncio.create_task(
            process.run_process(
                tmp_path / "config.json", run_dir, lambda _: None, stop_requested=lambda: stop
            )
        )
        await asyncio.sleep(0.05)
        if cancel_task:
            trial.cancel()
            await asyncio.sleep(0.05)
            trial.cancel()
            with pytest.raises(asyncio.CancelledError):
                await trial
        else:
            stop = True
            assert await trial == 0

    asyncio.run(run())
    report = json.loads((tmp_path / "supervisor.json").read_text())
    assert report["stop_requested"] and report["escalation"] is None
    assert report["cancellation_requested"] is cancel_task
    assert report["returncode"] == 0 and report["exports_present"]
    assert report["native_status"]["status"] == "stopped"


def test_supervisor_escalates_unresponsive_controller_without_claiming_cleanup(
    tmp_path, monkeypatch
):
    script = tmp_path / "worker.py"
    script.write_text(
        "import pathlib, signal, sys, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "root = pathlib.Path(sys.argv[1])\n"
        "root.mkdir()\n"
        "(root / 'run.json').write_text('{\"status\": \"running\"}')\n"
        "print('{\"ready\": true}', flush=True)\n"
        "while True: time.sleep(0.01)\n"
    )
    run_dir = tmp_path / "run"
    monkeypatch.setattr(process, "_command", lambda _: [sys.executable, str(script), str(run_dir)])
    monkeypatch.setattr(process, "_ESCALATION_SECONDS", 0.05)
    events = []

    async def run():
        return await asyncio.wait_for(
            process.run_process(
                tmp_path / "config.json",
                run_dir,
                events.append,
                stop_timeout_seconds=0.05,
                stop_requested=lambda: bool(events),
            ),
            5,
        )

    returncode = asyncio.run(run())
    assert returncode != 0
    report = json.loads((tmp_path / "supervisor.json").read_text())
    assert report["stop_requested"]
    assert report["escalation"] == ("terminate" if os.name == "nt" else "kill")
    assert not report["exports_present"] and report["native_status"]["status"] == "running"


def test_supervisor_reaps_descendants_that_inherit_output_pipes(tmp_path, monkeypatch):
    script = tmp_path / "worker.py"
    finished = tmp_path / "descendant-finished"
    script.write_text(
        "import subprocess, sys\n"
        "subprocess.Popen([sys.executable, '-c', "
        "'import pathlib, sys, time; time.sleep(30); pathlib.Path(sys.argv[1]).touch()', "
        "sys.argv[1]])\n"
        "print('controller finished', flush=True)\n"
    )
    monkeypatch.setattr(process, "_command", lambda _: [sys.executable, str(script), str(finished)])
    monkeypatch.setattr(process, "_PIPE_DRAIN_SECONDS", 0.05)

    async def run():
        return await asyncio.wait_for(
            process.run_process(tmp_path / "config.json", tmp_path / "run", lambda _: None), 3
        )

    started = time.monotonic()
    assert asyncio.run(run()) == 0
    assert time.monotonic() - started < 3
    report = json.loads((tmp_path / "supervisor.json").read_text())
    assert report["returncode"] == 0 and report["output_capture_complete"]
    assert report["controller_cleanup"]["confirmed"]
    assert not finished.exists()
