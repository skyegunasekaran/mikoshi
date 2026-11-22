"""Addressed inboxes retain targeted operator messages without public comments."""

import json
from dataclasses import replace

import pytest
from support.organization import ScriptedProvider, problem, world

from mikoshi.persistence.store import WorldStore
from mikoshi.problem import ScheduledNotificationSpec
from mikoshi.runtime.budgets import ExecutionBudget
from mikoshi.runtime.notifications import Notifications


def _scheduled_world(tmp_path):
    provider = ScriptedProvider()
    provider.budget = ExecutionBudget(tmp_path / "budget.json", seconds=120)
    provider.budget.pause()
    evidence = "Operator evidence\n" + "Retained scheduled content. " * 100
    spec = replace(
        problem(max_agents=2),
        scheduled_notifications=(
            ScheduledNotificationSpec("evidence", 1, "participant", evidence),
        ),
    )
    store, runtime = world(tmp_path, provider, spec, enable_commands=False)
    return store, runtime, provider, evidence


def test_scheduled_inbox_delivery_is_targeted_durable_and_wakes_only_when_requested(tmp_path):
    store, runtime, provider, evidence = _scheduled_world(tmp_path)
    try:
        reader, writer = runtime._claim_wave(2, 1)
        store.finish_activation(reader.activation_id, reader.trigger.id, "yielded", 0)
        provider.budget.elapsed_before = 60
        runtime._scheduled.release_due()
        runtime._scheduled.release_due()
        assert store.queued_trigger_count() == 1
        inbox, cursor, sources = runtime._notifications.digest("participant")
        (item,) = inbox["updates"]
        assert item["kind"] == "scheduled_message"
        assert item["message"] == evidence and "truncated" not in item
        assert item["reference"].startswith("event:")
        assert sources[0]["source_ref"] == item["reference"]
        assert runtime._notifications.digest(writer.trigger.agent_id)[0]["updates"] == []
        assert store._connection.execute("SELECT COUNT(*) FROM page_comments").fetchone()[0] == 0
        assert evidence not in runtime._pages.render(
            runtime._pages._page("participant", "objective")
        )
        assert runtime._notifications.digest("participant")[0] == inbox

        archive = tmp_path / "archive.sqlite"
        store.backup(archive)
        restored = WorldStore(archive)
        try:
            assert Notifications(restored).digest("participant")[0] == inbox
            assert restored.list_events() == store.list_events()
        finally:
            restored.close()
        runtime._notifications.acknowledge("participant", cursor)
        assert runtime._notifications.digest("participant")[0]["updates"] == []

        (resumed,) = runtime._claim_wave(1, 2)
        store.finish_activation(resumed.activation_id, resumed.trigger.id, "yielded", 0)
        provider.budget.elapsed_before = 119
        runtime._scheduled.release_due()
        assert store.queued_trigger_count() == 0
        assert len(runtime._notifications.digest("participant")[0]["updates"]) == 2
        assert not store.event_observer_errors
    finally:
        store.close()


def test_scheduled_delivery_and_release_rollback_together(tmp_path, monkeypatch):
    store, runtime, provider, evidence = _scheduled_world(tmp_path)
    try:
        reader, _ = runtime._claim_wave(2, 1)
        store.finish_activation(reader.activation_id, reader.trigger.id, "yielded", 0)
        provider.budget.elapsed_before = 60
        tick, events = store.current_tick, store.list_events()
        deliver = type(runtime._notifications).deliver

        def interrupted(self, **arguments):
            deliver(self, **arguments)
            raise RuntimeError("Interrupted delivery")

        with monkeypatch.context() as patch:
            patch.setattr(type(runtime._notifications), "deliver", interrupted)
            with pytest.raises(RuntimeError, match="Interrupted delivery"):
                runtime._scheduled.release_due()
        assert store.current_tick == tick and store.list_events() == events
        assert store.queued_trigger_count() == 0
        assert runtime._notifications.digest("participant")[0]["updates"] == []
        assert (
            store._connection.execute(
                "SELECT value FROM metadata WHERE key='scheduled_notification_releases'"
            ).fetchone()
            is None
        )

        runtime._scheduled.release_due()
        runtime._scheduled.release_due()
        assert store.queued_trigger_count() == 1
        (message,) = runtime._notifications.digest("participant")[0]["updates"]
        assert message["message"] == evidence
        releases = json.loads(
            store._connection.execute(
                "SELECT value FROM metadata WHERE key='scheduled_notification_releases'"
            ).fetchone()[0]
        )
        assert message["reference"] == f"event:{releases['evidence']['event_id']}"
        assert "comment_id" not in message
    finally:
        store.close()
