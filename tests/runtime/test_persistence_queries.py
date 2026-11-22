"""Durable conversation, guidance, and integrity records survive publication and reopen."""

import json
import sqlite3
from contextlib import closing

from mikoshi.decisions import DecisionRequest
from mikoshi.persistence.store import WorldStore
from mikoshi.runtime.conversations import ConversationHistory
from mikoshi.runtime.repository_guidance import GuidanceLedger


def test_conversation_repository_commits_before_publication_and_deduplicates(tmp_path):
    database = tmp_path / "world.sqlite"
    with closing(WorldStore(database)) as store:
        history = ConversationHistory(store)
        observed = []

        def observer(exchange):
            with closing(sqlite3.connect(database)) as reader:
                observed.append(
                    reader.execute(
                        "SELECT exchange_key FROM agent_memory_exchanges WHERE sequence=?",
                        (exchange["sequence"],),
                    ).fetchone()[0]
                )

        store.observe_exchanges(observer)
        request = DecisionRequest("peer", "Work", '{"tick":1}', {}, conversation_id="peer")
        history.append_exchange("peer", "exchange-1", request, {"action": "inspect"}, {})
        history.append_exchange("peer", "exchange-1", request, {"action": "different"}, {})
        assert observed == ["exchange-1"]
        assert history.last_state("peer") == {"tick": 1}
        assert json.loads(history.last_exchange("peer")[1]["content"]) == {"action": "inspect"}
        assert history.last_exchange("someone-else") == []
        assert not store.event_observer_errors


def test_guidance_checkpoints_and_execution_quarantine_survive_reopening(tmp_path):
    database = tmp_path / "world.sqlite"
    document = {"path": "AGENTS.md", "content": "Preserve evidence", "scope": "."}
    with closing(WorldStore(database)) as store:
        guidance = GuidanceLedger(store)
        assert guidance.pending("peer", [document]) == [document]
        guidance.acknowledge("peer", [document])
        assert store.execution_quarantined is False
        store.quarantine_execution()
        assert store.latest_causal_wave_id() == 1
    with closing(WorldStore(database)) as store:
        guidance = GuidanceLedger(store)
        assert guidance.pending("peer", [document]) == []
        assert guidance.pending("someone-else", [document]) == [document]
        assert store.execution_quarantined is True
        guidance.acknowledge("peer", [], checkpoints=[{"id": "compacted"}])
        assert guidance.pending("peer", [document]) == [document]
