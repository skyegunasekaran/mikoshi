"""Live pressure gates FIFO admission without a fixed number of command slots."""

import threading
from concurrent.futures import wait
from types import SimpleNamespace

import pytest

from mikoshi.isolation.admission import CommandQueue


def prepared(ident, *, cancel=False, timeout=10):
    import time

    event = threading.Event()
    if cancel:
        event.set()
    return SimpleNamespace(
        ident=ident,
        cancel_event=event,
        timeout_seconds=timeout,
        queued_at=time.monotonic(),
        _execution_lock=threading.Lock(),
    )


def test_pressure_gates_fifo_and_queued_cancellation_without_stopping_active_work(monkeypatch):
    monkeypatch.setattr("mikoshi.isolation.admission.ADMISSION_INTERVAL_SECONDS", 0.01)
    pressure = {"cpu_percent": 95, "memory_percent": 0}
    sampled = threading.Event()
    running = threading.Event()
    release = threading.Event()
    order = []

    def sample(_):
        sampled.set()
        return dict(pressure)

    def execute(command):
        if command.cancel_event.is_set():
            return {"termination_reason": "cancelled"}
        order.append(command.ident)
        if command.ident == 1:
            pressure.update(memory_percent=95)
            running.set()
            assert release.wait(3)
        return {"termination_reason": "completed"}

    queue = CommandQueue(execute, sample, 95)
    first = queue.submit(prepared(1))
    second_command = prepared(2)
    second = queue.submit(second_command)
    third = queue.submit(prepared(3))
    try:
        assert sampled.wait(2) and not first.running() and not order
        pressure.update(cpu_percent=0)
        assert running.wait(2)
        assert not second.running() and not third.running()
        second_command.cancel_event.set()
        assert second.result(timeout=2)["termination_reason"] == "cancelled"
        assert not first.done() and not third.running()
        pressure.update(memory_percent=0)
        assert third.result(timeout=2)["termination_reason"] == "completed"
        assert order == [1, 3]  # Admission can expand while the first command is still running.
    finally:
        release.set()
        wait([first, second, third], timeout=3)

    def unavailable(_):
        raise RuntimeError("resource observation unavailable")

    queue.sample = unavailable
    with pytest.raises(RuntimeError, match="observation unavailable"):
        queue.submit(prepared(4)).result(timeout=2)
    assert order == [1, 3]
